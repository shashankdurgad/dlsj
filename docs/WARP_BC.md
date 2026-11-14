# WARP-BC on the pick → inspect → place dataset

Paper: Yu et al., *WARP-RM: A Warp-Augmented Relative Progress Reward Model for Data Curation*,
arXiv 2606.28320 (v3). Code: `dlsj/warp/`.

## What the paper does
1. **WARP-RM** learns task progress with no labels. It uses frozen DINOv3 features and a bidirectional
   transformer. It is trained on the fastest successful demos, replayed at smoothly varying speeds (an AR(1)
   process on log-velocity) with random reversals. The target is the normalised elapsed time,
   y = (i_j − i_0) / C_norm, predicted as 30 two-hot bins.
2. Sliding windows give a per-frame **progress velocity** v̂: 1 means reference pace, 0 means stalled and
   below 0 means going backwards.
3. **WARP-BC** weights each action chunk by v̂ at its *last* frame: w = v̂_end · 1[v̂_end > τ], with τ = 1.
   Chunks with w = 0 are dropped. On T-shirt folding this kept ~34% of chunks and raised success from 2/20
   to 19/20 on the noisier tier.

## Why we can't apply it as-is
Our task has segments that are slow **on purpose** and look like "no progress" on camera:
- the hold in front of the `inspect` camera
- grasp alignment
- easing the brick onto the plate

A global τ = 1 would drop them, and the policy would never learn to present the brick. We also have
~300 episodes rather than the paper's 2–6k, so ACT can't afford to lose 65% of its data blindly.

## Our adaptation (`dlsj/warp/weights.py`)
| Guard | What it does |
|---|---|
| Stage-relative pace | v̂ is divided by the reference demos' median v̂ **for that stage** (pick / place, split at the recorder's `inspection_frame`). |
| Protected frames | The inspection hold (−1.0 s … +1.5 s around `inspection_frame`) and ±0.5 s around gripper close/open always get weight ≥ 1. |
| Coverage floor | Each stage is cut into 10 slices of normalised stage time. Any slice that keeps < 25% of its chunks gets its best ones (highest v̂) added back. |
| Failed episodes | Weight 0 (`success: false` in `meta/dlsj/labels.jsonl`). |

Each frame records *why* it was kept (`pace` / `protected` / `coverage` / `dropped` / `failed_episode`).
`coverage.csv` shows the kept fraction per stage × slice, which checks that the whole task is still covered.

## Scale changes from the paper
| | Paper (real) | Ours (default) | Why |
|---|---|---|---|
| Window N, stride S | 32, 1.5 s (46 s window) | 16, 0.5 s (7.5 s window) | Our episodes are ~10–25 s. The paper's sim also used S = 0.5 s. |
| Reference set | ~30% shortest | 40% shortest successful | Small dataset |
| RM size | 12 layers, d = 768, 15k steps | 6 layers, d = 512, 8k steps | Small dataset |
| Cameras | one view | `top` + `wrist` features concatenated | `top` sees task progress; `wrist` sees the grasp |
| Policy | π0, 1 s chunks | ACT, chunk_size 100 (3.3 s) | v̂ is read at the chunk's last frame |

## Run (GB10, inside `~/desmond`)
```bash
python -m dlsj.warp.features  --root DATA --out W/feats --cameras top,wrist
python -m dlsj.warp.train_rm  --root DATA --feats W/feats --out W/rm
python -m dlsj.warp.score     --root DATA --feats W/feats --rm W/rm/warp_rm.pt --out W/weights --chunk 100
# baseline and WARP-BC with identical settings:
python -m dlsj.warp.train_policy --dataset.repo_id=local/NAME --dataset.root=DATA --policy.type=act ... \
    --output_dir=runs/act_vanilla
python -m dlsj.warp.train_policy ... --sample_weighting.type=warp \
    --sample_weighting.progress_path=W/weights/warp_weights.parquet --output_dir=runs/act_warp
```

## Sanity checks before training a policy
- `train_rm` log: held-out velocity ≈ +1 played forwards and ≈ −1 played backwards.
- `score` summary: `corr_length_vs_mean_v` < 0, since longer (more hesitant) episodes should score slower.
  `min_phase_kept` ≥ `min_keep`.
- Check v̂ traces on a few episodes against the video: hesitations and regrasps should dip.
