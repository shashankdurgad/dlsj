# MolmoAct2-SO100_101 fine-tune in Physical AI Studio: two LoRA methods

Fine-tunes [`allenai/MolmoAct2-SO100_101`](https://huggingface.co/allenai/MolmoAct2-SO100_101) (Ai2, about 5.4B
parameters, already trained on SO-100/101 data with absolute joint control) on our 3-stage teleop demos. It
trains through **Physical AI Studio's own MolmoAct2 policy** at the pinned commit, so either method's checkpoint
exports to OpenVINO for the XPS.

## The two methods (`--method`)

Both use bf16-mixed, rank-64 / alpha-16 / dropout-0.05 LoRA, and 5e-5 for every learning-rate group. **Both train
the whole action expert and the whole connector in full**, and both put LoRA on the LLM and the ViT. They now differ
only in two tiny details:

| | **A `ai2`** (default): Ai2's LoRA recipe | **B `studio`**: Studio's LoRA leaves |
|---|---|---|
| LLM blocks (att_proj, attn_out, ff_proj, ff_out) | LoRA | LoRA |
| ViT blocks (wq/wk/wv/wo, w1/w2) | LoRA | LoRA |
| ViT patch embedding | frozen | LoRA |
| Connector (image_pooling_2d + image_projector) | **full** | **full** |
| Action expert | **full** | **full** |
| LLM final norm `ln_f` | full | frozen |
| Token embedding `wte`, `lm_head`, norms | frozen | frozen |

Both method classes derive from `policy_text.MolmoAct2Text128` (text padded to 128 tokens instead of Studio's
256, trained in so it matches the XPS export). **Exception:** the two Method A runs started on 2026-11-14
(`m2a-b300-20261114` on Modal, `m2a-ai2-20261114` on GCP) predate this and use 256 tokens; export and run those at
256.

Implemented in `methods.py` (`MolmoAct2Ai2Split`, `MolmoAct2StudioLoraFullExpert`). Each passes its adapter regex
(`train.AI2_LORA_TARGETS` / `train.STUDIO_VLM_LORA_TARGETS`) as `lora_target_modules`, then re-enables its full-tune
parts after Studio's LoRA injection (the action expert through Studio's `unfreeze_action_expert()`). It fails at
startup if any group ends up without the intended LoRA or full training, and logs the trainable count per group.
Checkpoints from either method load in the stock `physicalai.policies.MolmoAct2` for export: the adapters are
merged into the weights at export time.

Trainable parameters, computed from the checkpoint's own tensor shapes (safetensors headers at `152569fe`, 5.44B in
total). LoRA percentages are adapter size relative to the component (rank 64: r·(in+out) per adapted linear):

| Component | Params | A trains | B trains |
|---|---|---|---|
| LLM attention + MLP | 3,633M | 114.4M LoRA (3.1%) | 114.4M LoRA (3.1%) |
| ViT attention + MLP | 381M | 32.2M LoRA (8.5%) | 32.2M LoRA (8.5%) |
| ViT patch embedding | 0.7M | 0 | 0.1M LoRA (16%) |
| Connector | 57M | 57M full (100%) | 57M full (100%) |
| Action expert (36 layers, width 768) | 578M | 578M full (100%) | 578M full (100%) |
| LLM `ln_f` | 2.6k | full (100%) | 0 |
| Token embeddings, norms, ViT pos. embedding | 397M | 0 | 0 |
| LM head | 396M | 0 | 0 |
| **Total** | **5,442M** | **780.8M (14.35%)** | **780.9M (14.35%)** |

Training the action expert and connector in full adds about 10 GB of fp32 AdamW state per GPU, which is fine on
96 GB.

## Shared recipe

| Setting | Value | Source |
|---|---|---|
| Start checkpoint | `allenai/MolmoAct2-SO100_101` @ `152569fe` | Ai2 "SO100/SO101 fine-tuning" |
| Optimizer | AdamW (0.9, 0.95), eps 1e-6, wd 0, per-group grad clip 1.0 | Studio MolmoAct2 defaults = Ai2 |
| Schedule | warmup 200, cosine to 1e-6 over the run | Studio/Ai2 |
| Global batch | 32 (8 per GPU × 4 RTX PRO 6000) | LeRobot MolmoAct2 docs: "< 200 demos: 16–32" |
| Prompt metadata | setup `single so100/so101 robotic arm in molmoact2`, control `absolute joint pose` | checkpoint norm tag `so100_so101_molmoact2` |
| Action chunk | 30 predicted, 30 executed (1 s) | same norm tag |
| Joint frame | `adapt_to_so101: true`, no `calibration` (our joints are already LeRobot degrees) | Studio docs |
| Augmentation | 95% crop, ±5° rotation, colour jitter, blur p=0.2 | Ai2 |
| Cameras | wrist + top (top cropped to 560×480) | team decision |
| Stage-aware chunks | a chunk stops at the next grasp/inspection boundary, and the remaining slots repeat the last target | team 3-stage convention (`phase_data.py`) |
| Split | the team's explicit split from `CURATION.json`; Studio is not allowed to re-split | 431 README loading contract |

## Data: two snapshots on Spark 2 (`~/desmond/datasets/`)

`prepare_data.py` needs only numpy, pyarrow and ffmpeg. It runs on Spark 2 with
`~/desmond/training/prep-venv`. It keeps every source episode under its original index, removes the `inspect`
and depth cameras, and crops `top`. It writes MANIFEST.json with the episodes to use (selection minus the team's
`exclude_from_training`) and the team split restricted to them, plus train-only normalization,
SHA256SUMS.txt and READY.json.

| Snapshot | Source | Episodes (train / val) | Train frames | Calibration |
|---|---|---|---|---|
| `molmoact2-cal2-431-3stage-20261114` | `duplo-cal2-431-3stage-20261114` (431 stored, 3 excluded) | 428 (386 / 42) | 88,379 | **cal2** (follower.json `8a3e5e04`) = current arm |
| `molmoact2-517top250-3stage-20261114` | `duplo517-3stage-20261114` + `top250-shortlist/TOP-250-SHORTLIST.json` | 250 (223 / 27) | ≈40.8k | **old, pre-recalibration** |

**The 250 set was recorded before the 2026-11-14 recalibration.** A model trained on it expects the old joint frame
(pan off by 20.35°, lift by 7.96°). On today's arm, run it with `--joint-offset recal_2026-11-14`, or it misses
by about 9 cm. Never merge the two sets without converting one of them.

```bash
# on Spark 2, in ~/desmond/datasets (P=~/desmond/training/prep-venv/bin/python)
$P molmoact2_prepare_data.py duplo-cal2-431-3stage-20261114/duplo_teleop_cal2_rgb_3stage molmoact2-cal2-431-3stage-20261114 \
   --calibration duplo-cal2-100-20261114/calibration-observed-at-transfer/robots/so_follower/follower.json
$P molmoact2_prepare_data.py duplo517-3stage-20261114/duplo_teleop_3stage molmoact2-517top250-3stage-20261114 \
   --episodes duplo517-3stage-20261114/top250-shortlist/TOP-250-SHORTLIST.json --calibration-note "pre-recal ..."
# upload (from a machine with gcloud): gcloud storage rsync --recursive <snapshot> gs://onflow-30k-pi05-artifacts/datasets/<snapshot>
```

## Run on GCP (one 4× RTX PRO 6000 Spot VM per method)

```bash
DS=gs://onflow-30k-pi05-artifacts/datasets/molmoact2-cal2-431-3stage-20261114
bash launch_vm.sh m2a-cal2-431-20261114 --method ai2    --dataset $DS --steps 12000 --max-uptime-s 43200 [--create]
bash launch_vm.sh m2b-cal2-431-20261114 --method studio --dataset $DS --steps 12000 --max-uptime-s 43200 [--create]
```

Without `--create` it only prints what it would do. With it, it refuses to start until the dataset's READY.json is in GCS.

**Stage `gate`** (about 1 h): Studio setup, model pin check, dataset checksums, smoke run, 200-step timing
(`timing.json` with measured s/step and the projected hours), and an OpenVINO export of step 200 to
`gate/openvino-step200`. Load that export on the XPS GPU before paying for the full run. **Stage `full`:**
```bash
gcloud compute instances add-metadata <name> --zone=europe-west4-b --metadata=stage=full,max-uptime-s=43200
gcloud compute instances start <name> --zone=europe-west4-b
```
Checkpoints go to `runs/<name>/full/checkpoints/`: `snapshots/` every 2,000 steps (test these on the arm), `best/`
by validation loss, and `resume/last.ckpt`. After a Spot preemption, start the VM again and it resumes.

## Time estimate (gate `timing.json` gives the measured number)

Baseline, measured on the same VM type and Studio commit: π0.5 expert-only (VLM frozen, 3 cameras, batch 32,
bf16, gradient checkpointing) ran at **0.90–0.97 s/step** (`runs/pi05-517-3epoch-20261114`, `runs/pi05-four-20261114`).
MolmoAct2's forward pass is similar (about 4.5 vs 4.0 TFLOP per sample: 2 cameras × one 378² tile = about 196
tokens each, about 450 LLM tokens). But both methods back-propagate through the LoRA'd LLM and ViT, plus the
gradient-checkpoint recompute, which is about 3× π0.5-expert-only compute per sample. The π0.5 runs were largely
bound by overhead (about 7% of peak compute). **Estimate: about 2.1 s/step (1.5–3.0)**, the same for A and B.

| Run (per method, one 4× RTX PRO 6000 VM) | Steps | Epochs | Training | + gate, val, checkpoints | Total | `--max-uptime-s` |
|---|---|---|---|---|---|---|
| 431 cal2 | 12,000 | 4.3 | 7.0 h (5.0–10.0) | about 1.5 h | **≈8.5 h (6.5–11.5)** | 43200 |
| 431 cal2, shorter | 8,000 | 2.9 | 4.7 h (3.3–6.7) | about 1.3 h | ≈6 h (4.6–8) | 32400 |
| 250 old-cal | 8,000 | 6.3 | 4.7 h (3.3–6.7) | about 1.3 h | ≈6 h (4.6–8) | 32400 |

Add-ons are the gate (about 1 h: setup, 22 GB model download, smoke run, 200-step timing, OpenVINO export),
validation every 500 steps (about 8 min in total), and the about 28 GB resume checkpoint every 500 steps (15–25 min in total).
Disk needs about 210 GB of the 500 GB. Two VMs (A and B) take the same wall-clock time and twice the GPU-hours. If the
uptime cap hits first, training stops cleanly and resumes from `last.ckpt` when the VM restarts.

## Verified so far

- Unit tests: `python -m unittest discover -s training/molmoact2 -p 'test_*.py'` (methods, target regex against
  real module names, team split, exclusions, stage-aware chunks, explicit-split shim, augmentation).
- Method A's regex was checked against the module names in the pinned Studio source (`components/text.py`,
  `vision.py`, `backbone.py`).
- Both snapshots were checked independently on Spark 2 (`check_molmoact2_snapshot.py`, no LeRobot):
  - all checksums;
  - split = the team's CURATION split restricted to the selection, with exclusions honoured and no leakage (431:
    identical to the team's 386/42, train frames 88,379 = the team's figure);
  - only wrist + top in info, stats and episode metadata;
  - data parquet byte-identical to the source;
  - wrist frames identical and the top crop within 1.03% mean abs difference at every grasp/inspection boundary
    of 5 episodes.
- LeRobot 0.7 loads them with Studio's access pattern: episode subset, 30-step action chunks, shapes, and the
  stage instruction checked on sampled frames. Studio itself pins LeRobot 0.6.0; the gate's smoke run is the
  first load in that version.
- **Not yet run:** the Studio MolmoAct2 environment, any training step (so Method A's unfreeze check has not run
  yet), the OpenVINO export, and anything on a GPU or the XPS.
