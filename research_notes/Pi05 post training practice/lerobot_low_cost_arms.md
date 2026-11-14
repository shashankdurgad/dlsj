# Fine-tuning π0.5 with Hugging Face LeRobot on low-cost arms (SO-100/SO-101, Koch): practice as of Nov 2026

Research date: 2026-11-14. Sources are the LeRobot docs and source on `main`, LeRobot GitHub issues and PRs, Hugging Face Hub model cards, and arXiv papers from 2026. Anything from a single user's model card or GitHub comment is marked **(anecdotal)**.

## Q1. LeRobot pi05 training: commands, flags, hyperparameters, normalization, chunking, image size, tokenizer

### Takeaway
LeRobot's documented recipe is `lerobot-train --policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base --policy.dtype=bfloat16 --policy.gradient_checkpointing=true`, choosing either full fine-tuning or `freeze_vision_encoder`/`train_expert_only`. Most other values come from the config: AdamW with lr 2.5e-5, betas (0.9, 0.95), weight decay 0.01, grad clip 1.0, and cosine decay over 30k steps with 1k warmup to 2.5e-6. When `--steps` is lower, the warmup and decay auto-scale down. Other defaults are chunk_size and n_action_steps of 50, 224x224 resize-with-pad, tokenizer max length 200, and QUANTILES normalization for state and action. QUANTILES needs q01/q99 stats, which older datasets lack. Fix that with `lerobot-edit-dataset --operation.type recompute_stats`, or switch to MEAN_STD.

### Cited Findings
- **Install and auth:** `pip install -e ".[pi]"` (or `pip install 'lerobot[pi]'`). The tokenizer is the gated `google/paligemma-3b-pt-224`, so accept its license on the Hub and run `hf auth login` before training — [LeRobot pi05 docs](https://huggingface.co/docs/lerobot/pi05)
- **Documented LIBERO full fine-tune, "sized for a single 80 GB GPU":** `--policy.type=pi05 --policy.pretrained_path=lerobot/pi05_libero_base --policy.normalization_mapping='{"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}' --policy.n_action_steps=10 --policy.empty_cameras=1 --policy.freeze_vision_encoder=false --policy.train_expert_only=false --policy.gradient_checkpointing=true --policy.dtype=bfloat16 --batch_size=64 --num_workers=8 --steps=30000 --save_freq=5000`. The expert-only variant changes only `freeze_vision_encoder=true --train_expert_only=true`, which the docs describe as "less memory, at some cost in success rate" — [LeRobot pi05 docs](https://huggingface.co/docs/lerobot/pi05)
- **Other documented flags:** `--policy.compile_model=true` for faster training, `--policy.optimizer_fused=true` for fused AdamW (ignored on resume), and `--batch_size=64` "adapt based on GPU memory". Base options are `lerobot/pi05_base` and `lerobot/pi05_libero_base` — [LeRobot pi05 docs](https://huggingface.co/docs/lerobot/pi05)
- **`pretrained_path` vs `path`:** `--policy.pretrained_path` loads weights only. Stored settings such as `n_action_steps` and `empty_cameras` reset to defaults (50 and 0), so pass them explicitly. `--policy.path` loads weights plus the checkpoint `config.json`, and `--policy.type` must then be omitted. Use `--rename_map` when dataset camera or feature keys differ — [LeRobot pi05 docs](https://huggingface.co/docs/lerobot/pi05)
- **Config defaults** in `configuration_pi05.py` on main (fetched 2026-11-14) — [configuration_pi05.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/pi05/configuration_pi05.py):
  - Architecture and precision: `paligemma_variant="gemma_2b"`, `action_expert_variant="gemma_300m"`, `dtype=float32` by default (you must pass bfloat16).
  - Action chunking and padding: `chunk_size=50`, `n_action_steps=50`, `max_state_dim=max_action_dim=32`.
  - Flow matching: `num_inference_steps=10`.
  - Inputs: `image_resolution=(224,224)`, `empty_cameras=0`, `tokenizer_max_length=200`.
  - Normalization: VISUAL=IDENTITY, STATE=QUANTILES, ACTION=QUANTILES.
  - Training switches: `gradient_checkpointing=False`, `compile_model=False`, `compile_mode="max-autotune"`, `freeze_vision_encoder=False`, `train_expert_only=False`.
  - Optimizer: `optimizer_lr=2.5e-5`, `betas=(0.9,0.95)`, `eps=1e-8`, `weight_decay=0.01`, `grad_clip_norm=1.0`.
  - Scheduler: `scheduler_warmup_steps=1000`, `scheduler_decay_steps=30000`, `scheduler_decay_lr=2.5e-6`.
  - Optional features: `use_relative_actions=False` (with `relative_exclude_joints=["gripper"]`), `rtc_training_max_delay=0`, plus MEM memory options that are off by default.
- **Scheduler auto-scaling:** a code comment says the scheduler "auto-scale[s] if --steps < scheduler_decay_steps. For example, --steps=3000 will scale warmup to 100 and decay to 3000" — [configuration_pi05.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/pi05/configuration_pi05.py)
- **Quantile stats:** pi05 needs q01/q99 in `meta/stats.json`. Older datasets fail with `ValueError: QUANTILES normalization mode requires q01 and q99 stats`. The fix is `lerobot-edit-dataset --repo_id X --new_repo_id X --operation.type recompute_stats --operation.overwrite true`. The output lands in `$HF_LEROBOT_HOME/X`, so train with `--dataset.root=$HF_LEROBOT_HOME/X` or push it to the Hub. The alternative is MEAN_STD via `normalization_mapping` — [LeRobot pi05 docs](https://huggingface.co/docs/lerobot/pi05)
- **Merged or resumed datasets:** recording, resuming and merging aggregate quantiles conservatively (min for q≤50, max for q>50). Whole-dataset estimates come from `python src/lerobot/scripts/augment_dataset_quantile_stats.py --repo-id=X --overwrite --skip-images`. These are histogram estimates, so they change the normalized targets and the loss scale — [LeRobot pi05 docs](https://huggingface.co/docs/lerobot/pi05)
- An open PR fixes `recompute_stats`/`reencode_videos` for Hub datasets (opened 2026-11-04) — [PR #5240](https://github.com/huggingface/lerobot/pull/5240)
- **Relative actions:** recompute stats with `--operation.relative_action true --operation.chunk_size 50 --operation.relative_exclude_joints "['gripper']"`, then train with `--policy.use_relative_actions=true --policy.relative_exclude_joints='["gripper"]'` — [LeRobot pi05 docs](https://huggingface.co/docs/lerobot/pi05)
- **Training-time RTC:** an example uses `--policy.rtc_training_max_delay=10 --batch_size=8 --steps=30000` from `lerobot/pi05_base`, deployed with `lerobot-rollout --inference.type=rtc --inference.rtc.mode=trained --inference.rtc.execution_horizon=10 --robot.type=so100_follower --fps=50` — [LeRobot pi05 docs](https://huggingface.co/docs/lerobot/pi05)
- **Image preprocessing:** images go through `resize_with_pad_torch` to `image_resolution`, which preserves aspect ratio with padding. A 640x480 camera is letterboxed, not stretched — [modeling_pi05.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/pi05/modeling_pi05.py)
- **LeRobot LIBERO parity:** after 6k extra steps, LeRobot's pi05 averaged 97.5% against OpenPI's 96.85% — [LeRobot pi05 docs](https://huggingface.co/docs/lerobot/pi05)
- **SO-101 expert-only recipe that worked** (Oct 2026 paper):
  - Setup: LeRobot dev 0.6.2, `lerobot/pi05_base`, 6,000 steps × batch 32, AdamW wd 0.01, clip 1.0.
  - Learning rate: peak 2.5e-5 after 1,000 warmup steps, cosine to 2.5e-6 at 6,000.
  - Precision and memory: fp32 with gradient checkpointing.
  - Inputs: 2 cameras in 2 of 3 slots, quantile normalization, chunk 50 with all 50 executed.
  - The author notes openpi and the library examples run 20k–30k+ steps.
  - [arXiv 2610.37170](https://arxiv.org/html/2610.37170)
- **SO-101 Bench recipe** (full fine-tune, LeRobot `eb7a54c`):
  - Schedule: OpenPI `pi05_libero` scaled to budget, warmup to 5e-5 then constant, AdamW 0.9/0.95.
  - Normalization and chunking: quantile normalization, 30-step chunks with 15 executed.
  - Scale: global batch 256 on 4×A100-80GB for 36 h, with EMA 0.999 and bf16.
  - Checkpoints: best validated checkpoint was step 12000 out of 18000.
  - [5hadytru/so101_bench_pi05 model card](https://huggingface.co/5hadytru/so101_bench_pi05)
- **(anecdotal) Large SO-100/101 full fine-tune** (dongyoonkim/so101-pi05-base, via the VLASH stack on `PI05Policy`):
  - Optimizer: effective batch 256, AdamW (0.9, 0.95), wd 1e-10, peak lr 5e-5, cosine to 2.5e-6, 1000 warmup, 40k steps.
  - Normalization: MEAN_STD.
  - RTC: delay simulation with max_delay 8.
  - Compute: 8×A100 for about 40 h.
  - [dongyoonkim/so101-pi05-base](https://huggingface.co/dongyoonkim/so101-pi05-base)
- **(anecdotal) Second large run** (hqfang/pi05-so100_101, 1,209 SO-100/101 datasets, 36,877 episodes):
  - Optimizer: batch 256 × 150k steps, 1k warmup, cosine, peak lr 2.5e-5 to 2.5e-6.
  - Precision and compute: FP32 master weights with BF16 autocast on 4×GB200.
  - Normalization: pooled q01/q99 state and action normalization.
  - Needs a lerobot patch for camera padding.
  - No robot evaluation reported.
  - [hqfang/pi05-so100_101](https://huggingface.co/hqfang/pi05-so100_101)

### Inferences
- For a single SO-101 task, a reasonable LeRobot starting point is `--policy.pretrained_path=lerobot/pi05_base --policy.dtype=bfloat16 --policy.gradient_checkpointing=true --policy.train_expert_only=true --policy.freeze_vision_encoder=true --batch_size=32 --steps=6000–30000`. Pass `--policy.empty_cameras` explicitly when you have fewer than 3 cameras, and set `n_action_steps` explicitly as well. Recompute quantile stats first.
- Because `--steps` auto-scales the scheduler, a short run such as 3k steps still gets a full warmup-and-decay cycle.

### Gaps
- I found no official LeRobot page that gives an SO-101-specific pi05 command or hyperparameters. The docs give LIBERO examples and a generic `${HF_USER}/my_dataset` RTC example.
- I did not verify that the config defaults (lr etc.) match every release tag. Values are from `main` on 2026-11-14.

## Q2. Memory and compute: VRAM for full vs expert-only, training time

### Takeaway
A full fine-tune in LeRobot needs an 80 GB-class GPU. Batch 32 with compile ran out of memory on an A100-80GB, and batch 16 worked. Expert-only training reduces memory, but LeRobot has had an expert-only slowdown, about 6.6–7.8 s/step before an open fix. Inference in bf16 needs roughly 7–8 GB for weights. One chunk takes about 150 ms eager on an H200, or about 25 ms with `compile_model`, which costs a 3–7 minute warmup.

### Cited Findings
- **OpenPI's stated requirements,** quoted in a LeRobot issue: inference >8 GB, LoRA >22.5 GB, full fine-tune >70 GB (A100/H100) — [Issue #2216](https://github.com/huggingface/lerobot/issues/2216)
- **(anecdotal) Full fine-tune OOM:** the documented full fine-tune with `compile_model=true`, bf16, gradient checkpointing and batch 32 ran out of memory on an A100-80GB. Batch 16 worked (LeRobot ~0.4.4). Another user reported about 40 GB on L40-48GB with LeRobot v0.4.0, then OOM on 80 GB after upgrading. A maintainer closed the issue saying memory depends on batch size and on which parameters are trained — [Issue #2216](https://github.com/huggingface/lerobot/issues/2216)
- **Expert-only training speed:** expert-only training was slow because the frozen PaliGemma prefix was still checkpointed and the vision tower was kept in fp32. The PR measured about 6.6–7.8 s/step before the fix and about 1.67 s/step after, at batch 16 with bf16 and gradient checkpointing. The PR was still open as of Nov 2026 — [PR #3706](https://github.com/huggingface/lerobot/pull/3706)
- **SO-101 expert-only wall-clock:** about 6.3 s/step on one A100-80GB in fp32 with gradient checkpointing at batch 32, roughly 10.5 h for 6k steps. Inference ran in bf16 on a 16 GB RTX 5070 Ti — [arXiv 2610.37170](https://arxiv.org/html/2610.37170)
- **Parameter counts:** `lerobot/pi05_base` has 4,143,404,816 weights. Expert-only trains 693M (16.7%), of which about 263M sit in an unused vocab output layer, so about 430M are effectively updated — [arXiv 2610.37170](https://arxiv.org/html/2610.37170)
- **Inference latency on one H200** (Oct 2026, batch 1, 3 cameras):

  | Setting | ms per chunk | Warmup |
  |---|---|---|
  | fp32, as shipped | about 237 | none |
  | bf16 | about 151 | none |
  | bf16 + `compile_model` (max-autotune) | about 25 | 389 s cold, 46 s with a warm cache |

  86% of a bf16 chunk is the 10 denoise steps — [Issue #5148](https://github.com/huggingface/lerobot/issues/5148). Open PRs add a shared prefix KV cache and an opt-in CUDA graph — [PR #5152](https://github.com/huggingface/lerobot/pull/5152), [PR #5153](https://github.com/huggingface/lerobot/pull/5153)
- **(anecdotal) Deployment numbers:**
  - On an RTX 3090 Ti 24 GB, bf16 weights take about 7 GiB with "ample headroom" when compiled. Startup is about 25 s without compile and 2–3 min with it — [jinnymo/so101-pi05-base docs/06-inference.md](https://github.com/jinnymo/so101-pi05-base/blob/main/docs/06-inference.md)
  - OpenPi (JAX) fine-tune at 5k steps took about 3–4 h on an A100 (about $6–8 on RunPod). Remote inference ran at 1.7 Hz, rising to 3.7 Hz after JPEG-compressing images — [abdul004/pi05_so101_checkpoint](https://huggingface.co/abdul004/pi05_so101_checkpoint)
- **Large runs:** SO-101 Bench used 4×A100-80GB for 36 h. Its median server latency was 337 ms on an RTX A5000 — [arXiv 2610.36420](https://arxiv.org/html/2610.36420v1)
- **Export path:** an open PR adds ONNX/OpenVINO export for ACT and PI05 — [PR #3493](https://github.com/huggingface/lerobot/pull/3493)

### Inferences
- A 24 GB consumer GPU is enough for inference. Plan on an 80 GB GPU (A100/H100) for training, using expert-only with batch about 16–32, or full fine-tuning at batch ≤16.
- LoRA through the `peft` extra exists in LeRobot 0.6.x per arXiv 2610.37170. An early issue thread said LoRA was unsupported, but that was in 2025.

### Gaps
- I found no measured peak-VRAM figure for LeRobot pi05 expert-only training at a given batch size.
- No Koch-specific pi05 compute reports turned up.

## Q3. What SO-100/SO-101 users report: episodes, steps, success, comparisons

### Takeaway
On SO-101, π0.5 is consistently the strongest or near-strongest of the open VLAs, and it shines most in the low-data regime. One study measured 40% success at 12 demos and 70% at 25 demos with expert-only, against 3% and 28% for ACT. It plateaus around 70–75%, and ACT overtook it at 100–200 demos on a simple pick-and-place. Multi-task benchmarks put π0.5 first: 56% vs ACT 34% and SmolVLA 33% on 4 tasks with 100 demos each, and 47.6% pooled vs π0 35%, GR00T N1.7 29%, ACT 19% and SmolVLA 15% on 12 tasks with 50 demos each. A mobile bimanual toilet-cleaning study instead found π0 best.

### Cited Findings
- **Scaling study** (single SO-101 pick-and-place with distractors; 40 trials per cell; arXiv 2610.37170, 27 Oct 2026) — [arXiv 2610.37170](https://arxiv.org/html/2610.37170)

  | Recipe | 12 demos | 25 | 50 | 100 | 200 |
  |---|---|---|---|---|---|
  | π0.5 expert-only | 40% | 70% | 68% | 75% | 73% |
  | ACT | 3% | 28% | 45% | 93% | 88% |
  | SmolVLA | 5% | 10% | 53% | 35% | 38% |
  | π0.5 LoRA (r32) | 3% | 7% | 3% | 0% | 0% |

  - Fixed steps were a confound: all π0.5 cells used 6k steps, so π0.5 saw each frame about 31 times at 12 demos but only about 1.9 times at 200. The plateau may reflect under-training.
  - π0.5 never failed to start an episode. SmolVLA failed to start in 94 of 200 episodes.
  - π0.5's remaining failures were mostly lost grips.
- **Failure and recovery benchmark** (arXiv 2606.08881, June 2026; 4 SO-101 tasks, 100 teleop demos per task, 20 evaluation episodes per cell) — [arXiv 2606.08881](https://arxiv.org/html/2606.08881v2)
  - Average success: π0.5 56.25%, Wall-X 51.25%, ACT 33.75%, SmolVLA 32.5%.
  - π0.5 per task: Pen Transfer 95, Color Sorting 10, Object Packing 55, Pen Placement 65.
  - Recovery rates: π0.5 30.77%, Wall-X 20.51%, ACT 6.45%, SmolVLA 3.23%.
  - The paper gives no training hyperparameters.
- **ArmnetBench v0.1** (arXiv 2607.24481, July 2026; 12 tasks, 50 demos per task, per-task fine-tunes) — [arXiv 2607.24481](https://arxiv.org/html/2607.24481v1)
  - Strict success: π0.5 47.6%, π0 35.1%, GR00T N1.7 29.4%, Diffusion Policy 26.7%, ACT 19.2%, MolmoAct 2 18.9%, SmolVLA 15.0%.
  - π0.5 by embodiment: 45.4% single-arm, 52.1% bimanual.
  - cable_clip scored 0% for every policy.
- **SO-101 Bench** (arXiv 2610.36420, Oct 2026; language-conditioned spatial tasks, π0.5 full fine-tune on 3,629 simulated episodes) — [arXiv 2610.36420](https://arxiv.org/html/2610.36420v1)
  - Score: π0.5 41.0. By split: 58% seen, 40% unseen object, 28% unseen class.
  - Sim2real: 49% in simulation vs 18% on the real robot. GR00T N1.7 scored 56% in simulation on the same set.
- **Mobile bimanual toilet cleaning** (2×SO-101 on a TurtleBot 4): π0 performed best at 60% strict success, ahead of π0.5, SmolVLA and GR00T. This is per the search abstract only; I did not read the full paper — [arXiv 2611.15381](https://arxiv.org/abs/2611.15381)
- **Community model cards (anecdotal):**
  - dongyoonkim: about 90% success on "green cube into white cup" with only 14 demos of that task inside a 17k-episode SO-101 corpus. This is a qualitative single-task observation by the author — [dongyoonkim/so101-pi05-base](https://huggingface.co/dongyoonkim/so101-pi05-base)
  - abdul004: 72 demos and 5k steps (OpenPi). The author says pi0.5 handled edge positions where ACT failed — [abdul004/pi05_so101_checkpoint](https://huggingface.co/abdul004/pi05_so101_checkpoint)
  - pdhinaka: 50 episodes, 18,736 frames, 30 fps, wrist and top cameras at 640x480; the name suggests 20k steps. No success rate reported — [pdhinaka/pi05_so101_blue_cube_gray_box_cleaned_20k](https://huggingface.co/pdhinaka/pi05_so101_blue_cube_gray_box_cleaned_20k)
  - A repo suggests 30–50 good demos are usually enough for one task — [RL-Token-Pi05-open](https://github.com/Zhengsw03/RL-Token-Pi05-open) (from a search snippet, not verified)

### Inferences
- **Practical sizing for a single SO-101 task:** about 25–50 clean demos already give about 70% with expert-only π0.5. More data mainly helps if you also scale the number of steps.
- **When ACT may win:** with about 100 or more demos on a narrow task, ACT can match or beat π0.5 and costs far less to run.

### Gaps
- I found no LeRobot Worldwide Hackathon write-ups, Discord summaries or HF forum threads with π0.5 SO-101 numbers in this pass.
- No Koch-arm π0.5 results turned up.
- Most evaluations use 20–40 trials per cell, so differences of about 15 points are within noise, as arXiv 2610.37170 itself notes.

## Q4. Known issues in the LeRobot pi05 port

### Takeaway
The main known issues as of Nov 2026 are:
- Jitter or oscillation compared with OpenPI, linked to absolute vs delta actions and to padded action dims excluded from the loss, which drift at inference.
- Camera-slot shifting when a middle camera is missing.
- Settings lost between `pretrained_path` and `path`, and processor configs not saved.
- Quantile-stat failures on older datasets.
- `compile_model` producing NaN.
- About 0.17 s pauses between chunks without RTC.
- SO-101 units: `use_degrees=True` for joints and 0–100 for the gripper must match between dataset and deployment.

### Cited Findings
- **Jitter vs OpenPI:**
  - Users report "strong inference jitter" for LeRobot-trained pi05 against smooth OpenPI behavior on the same data. Contributor explanations: OpenPI uses delta actions while LeRobot uses absolute ones; "scaling and timing mismatches"; and LeRobot excluding padded action dims from the loss, so noise in those dims leaks into the real dims. The issue is open — [Issue #2559](https://github.com/huggingface/lerobot/issues/2559)
  - The padded-dim problem in detail: dims padded to `max_action_dim=32` are sliced out of the loss but still integrated by Euler at inference. They grow to about 1.13× the noise scale and feed back through `action_in_proj` into the real dims. Fix PRs are open — [Issue #5335](https://github.com/huggingface/lerobot/issues/5335), [PR #5351](https://github.com/huggingface/lerobot/pull/5351), [PR #2891](https://github.com/huggingface/lerobot/pull/2891)
- **Episode-end padding in the loss:** padded steps at episode end are trained toward the frozen final pose. The issue was filed for pi0; a pi05 fix PR is open — [Issue #5105](https://github.com/huggingface/lerobot/issues/5105), [PR #5108](https://github.com/huggingface/lerobot/pull/5108)
- **(anecdotal) End-pose stall:** a model card notes these policies can stall at the end-of-episode pose (arm up, gripper closed). The same card uses absolute gripper actions with relative deltas on the arm joints — [mattewg/pi05-so101-grab-adapter-30k](https://huggingface.co/mattewg/pi05-so101-grab-adapter-30k) (via search snippet; the card was not directly readable)
- **Camera slots:** pi0, pi05 and SmolVLA put the remaining cameras in the wrong slots when a middle camera is missing; for example, the wrist image lands in slot 0. Fix PRs are open (Oct 2026) — [Issue #5118](https://github.com/huggingface/lerobot/issues/5118), [PR #5129](https://github.com/huggingface/lerobot/pull/5129)
- **(anecdotal) Slot assignment advice:** put the external view in `base_0_rgb` and the wrist camera in `left_wrist_0_rgb`, and mask the unused third slot with `empty_cameras=1` — [dongyoonkim/so101-pi05-base](https://huggingface.co/dongyoonkim/so101-pi05-base)
- **Processor configs lost on reload:** pi05 processor settings are not saved, so a checkpoint trained with `use_proprioceptive_memory=True` reloads with state back in the prompt. Fix PR open — [PR #5097](https://github.com/huggingface/lerobot/pull/5097)
- **`--rename_map` ignored with relative actions:** `--rename_map` was silently ignored for PI05 when `use_relative_actions=true`. Closed in April 2026 — [Issue #3425](https://github.com/huggingface/lerobot/issues/3425)
- **Relative-action wording:** a docs PR clarifies that `use_relative_actions=false` does not mean the checkpoint predicts absolute actions if it was trained on controller deltas — [PR #5125](https://github.com/huggingface/lerobot/pull/5125)
- **`compile_model` NaN:** `compile_model=true` can silently train on NaN loss and crash at max-autotune, due to a miscompiled `make_att_2d_masks` (open) — [Issue #4178](https://github.com/huggingface/lerobot/issues/4178)
- **Exploding gradients:** a separate pi05 run hit loss/grad NaN, and one user saw `clip_grad_norm` return values far above the limit — [Issue #2465](https://github.com/huggingface/lerobot/issues/2465)
- **bf16 trainable weights:** `--policy.dtype=bfloat16` stores the whole pi05 model in bf16, so a bf16 recipe may hit the "bf16 update-erasure" effect reported for SmolVLA, whereas the default fp32 does not. This is under discussion — [Issue #5099](https://github.com/huggingface/lerobot/issues/5099)
- **Pause between chunks:** π0.5 paused about 0.17 s per 50-step chunk against about 0.01 s for ACT, without RTC — [arXiv 2610.37170](https://arxiv.org/html/2610.37170). The docs offer guided RTC at inference, or trained RTC via `rtc_training_max_delay` — [LeRobot pi05 docs](https://huggingface.co/docs/lerobot/pi05)
- **SO-101 units:** the SO follower config defaults to `use_degrees=True`, "for backward compatibility with previous policies/dataset". Body joints use `MotorNormMode.DEGREES`, or `RANGE_M100_100` when that is false; the gripper uses `RANGE_0_100` — [so_follower config](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/config_so_follower.py), [so_follower.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/so_follower.py)
- **(anecdotal) Corpus screening:** when building a pooled SO-101 corpus, datasets were rejected if actions were in the leader-arm calibration frame, in delta or raw units, or not in degrees. The reason given: "normalization statistics absorb scale but not a shifted zero point" — [dongyoonkim/so101-pi05-base](https://huggingface.co/dongyoonkim/so101-pi05-base)
- **Images:** a historical image-padding-value bug and openpi misalignments were addressed in [PR #2891](https://github.com/huggingface/lerobot/pull/2891), which also lists issues #2218, #2512 and #2650. The current code letterboxes with `resize_with_pad_torch` — [modeling_pi05.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/pi05/modeling_pi05.py)
- **(anecdotal) Inference gotchas** (VLASH/LeRobot 0.4.1): the gated tokenizer fails offline unless cached (`HF_HUB_OFFLINE=1` needs a full cache), UVC camera backend and pixel-format failures, and calibration JSON format mismatches — [jinnymo/so101-pi05-base troubleshooting](https://github.com/jinnymo/so101-pi05-base/blob/main/docs/05-troubleshooting.md). A related offline tokenizer issue with transformers 5.x — [Issue #4131](https://github.com/huggingface/lerobot/issues/4131)
- **(anecdotal) Checkpoint choice:** lowest training loss did not mean the best closed-loop behavior — [dongyoonkim/so101-pi05-base](https://huggingface.co/dongyoonkim/so101-pi05-base) (via search snippet). In SO-101 Bench the best validated checkpoint was step 12k, not the final 18k — [5hadytru/so101_bench_pi05](https://huggingface.co/5hadytru/so101_bench_pi05)
- **LIBERO reproduction:** several users report 0% success with `pi05_libero_base` in official evaluation, which shows how sensitive the port is to config and processor mismatch — [Issue #3638](https://github.com/huggingface/lerobot/issues/3638), [Issue #4206](https://github.com/huggingface/lerobot/issues/4206)

### Inferences
- **Train-time checklist:** recompute quantile stats, keep the same `use_degrees` and calibration between recording and deployment, set `empty_cameras` and `n_action_steps` explicitly, keep camera keys and order fixed, and avoid `compile_model` for training until #4178 is fixed.
- **Deploy-time checklist:** evaluate several checkpoints on the robot, and consider RTC or a smaller `n_action_steps` (for example 10–25) to reduce pauses between chunks.

### Gaps
- I found no dedicated report of a "gripper not closing" bug in the LeRobot pi05 port. The closest evidence is grip-loss failures (arXiv 2610.37170) and the end-pose stall note.
- No controlled study compares `n_action_steps` or temporal ensembling for pi05 on SO-101.

## Q5. Multi-task language conditioning with small datasets

### Takeaway
Evidence is thin and mixed. π0.5 fine-tuned on small single-environment data often uses the prompt as a task selector, or ignores it entirely. Language-dependent spatial generalization on SO-101 is weak: 58% on seen layouts, falling to 28% on unseen classes and to 18% on the real robot. Broad pooled SO-101 fine-tunes seem to preserve some instruction following.

### Cited Findings
- **(anecdotal) Prompt ignored:** an OpenPI user fine-tuned π0.5 on about 250 videos (about 90 min). The model ignored the prompt and did the same task regardless of wording — [openpi Issue #768](https://github.com/Physical-Intelligence/openpi/issues/768)
- **Instructions act as task selectors:** SO-101 Bench cites a LIBERO probe in which swapping the instruction text for a task-ID embedding nearly matches the best published results, "suggesting instructions are largely used to select memorised tasks". VLAs that manipulate reliably still failed on out-of-distribution spatial relations — [arXiv 2610.36420](https://arxiv.org/html/2610.36420v1)
- **(anecdotal) Broad fine-tune keeps grounding:** the 17k-episode pooled SO-101 fine-tune with 430 tasks handled a multi-cube colour instruction, which the author attributes to pretraining that the broad fine-tune preserved — [dongyoonkim/so101-pi05-base](https://huggingface.co/dongyoonkim/so101-pi05-base) (via search summary)
- **Prompt length:** the prompt plus the discretized state is tokenized with `tokenizer_max_length=200` — [configuration_pi05.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/pi05/configuration_pi05.py). The state is discretized into 256 bins and written into the prompt — [LeRobot pi05 docs](https://huggingface.co/docs/lerobot/pi05)
- **Data-side failure mode:** generic or identical `meta/tasks` strings across episodes leave nothing to bind prompts to. ArmnetBench rewrites source task strings with its own instructions — [ArmnetBench dataset](https://huggingface.co/datasets/armnet/armnetbench_v01_lerobot_so101) (via search summary)

### Inferences
- **Tips:**
  - Give each behaviour a distinct, consistent instruction string in `meta/tasks`.
  - Include visually ambiguous scenes where only the instruction disambiguates the task, so the model cannot infer it from the scene alone.
  - Balance episodes per instruction.
  - Prefer training the VLM (full fine-tune, or at least an unfrozen backbone) or starting from a broad SO-101 domain-adapted base when language binding matters. With `train_expert_only`, the frozen VLM still encodes the prompt, but the expert must learn to use it from few examples.
  - Evaluate with held-out instruction and scene pairings.
- These tips are reasoned from the cited findings and were not tested in a controlled study.

### Gaps
- I found no controlled SO-100/101 study of multi-instruction π0.5 fine-tuning on small datasets (for example 3–5 instructions × 20–50 demos each) with per-instruction success rates.
- No LeRobot-documented guidance on multi-task language practice for pi05 turned up.
