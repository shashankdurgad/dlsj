# Training on Intel XPU (Arc iGPU of the Core Ultra XPS)

An **alternative** path for training the team's models on Intel hardware: the Arc B390 iGPU of the XPS
(Core Ultra X9 388H, ~57 GiB of shared system memory visible to `torch.xpu`).

> **The team's deployed π0.5 checkpoint did not come from these scripts.** It was trained on cloud NVIDIA GPUs
> (4× RTX PRO 6000, action-expert-only, Hugging Face `w3joe/pi05-junction-so101-20261114`) with the recipe in
> [`training/pi05/`](../pi05/README.md). No π0.5 or SmolVLA training run has been completed on XPU (see the
> status table below).

| File | Trains | How |
|---|---|---|
| `train_pi05_xpu.py` | π0.5 fine-tune from `lerobot/pi05_base` | Studio `physicalai fit`, one XPU |
| `train_smolvla_xpu.py` | SmolVLA fine-tune from `lerobot/smolvla_base` (fallback VLA) | Studio `physicalai fit`, one XPU |
| `train_anomalib_xpu.sh` | Stage 1 Anomalib PaDiM and stage 2 circle/cross CNN | our `dlsj.inspect.train` / `train_classifier` with `--accelerator xpu` |
| `common.py` | shared: dataset validation, XPU trainer config, memory guard, launcher | |
| `../../tests/test_xpu_training.py` | hardware-free tests of the config generation | `pytest -q tests/test_xpu_training.py` |

## Status (14 Nov 2026, on the XPS)

| What | Status | Result |
|---|---|---|
| `train_pi05_xpu.py --plan-only` on `~/desmond/dlsj/data/duplo_teleop_rgb` | **validated: plan-only** | 417 episodes / 85,647 frames at the time; 76,805 train frames; batch 2 × accumulate 16 = 32; 2,401 optimizer steps/epoch |
| Generated π0.5 / π0.5 LoRA+smoke / SmolVLA configs parsed by `physicalai fit --print_config` (Studio 8f13241) | **validated: config parses** | all three exit 0; plugin, `xpu`/`xpu_single`, accumulation resolved as written |
| `train_pi05_xpu.py --smoke` (fast_dev_run), attempt 1 | **ran, failed in the data loader** | `pi05_base` loaded onto the XPU (Lightning summary: 695 M trainable, 3.4 B non-trainable, 4.1 B total). The first batch failed in DataLoader worker 0: torchcodec cannot load in the Studio venv (`OSError: libavutil.so.60: cannot open shared object file`, no system FFmpeg). Exit 1 after 84 s. Lowest MemAvailable was **17.9 GiB, down from 51.9 GiB** at start, during model load, before any training step. No step time was measured |
| `--video-backend pyav` (the fix, now the default) | **validated: decode only** | `LeRobotDataset(..., video_backend="pyav")[1234]` in the Studio venv decodes all 3 cameras in 0.04 s |
| `train_pi05_xpu.py --smoke`, attempt 2 (with pyav) | **not run** | after attempt 1, MemAvailable stayed at 47–49 GiB for 10+ min (recording session active), below the 50 GiB rule for starting a heavy job. Before that, a π0.5 OpenVINO export had been running |
| `train_pi05_xpu.py` real run | **not run** | |
| `train_smolvla_xpu.py --plan-only` | **validated: plan-only** | batch 8 × accumulate 8 = 64; 1,200 optimizer steps/epoch |
| `train_smolvla_xpu.py --smoke` / real run | **not run** | |
| `train_anomalib_xpu.sh padim` on `~/inference/inspect_data/synth3_train` (synthetic) | **validated: full run** | Anomalib PaDiM fit on XPU 3.2–3.4 s, whole script ~17 s wall incl. export + CPU calibration; peak RSS 3.8 GB |
| `train_anomalib_xpu.sh classifier` on `mark_crops_synth3` (synthetic) | **validated: short run** (2–3 epochs) | ~5–7 s/epoch for 720 crops, ~11 s wall; test acc 99.6 % on `mark_crops_synth3_test` (OpenVINO CPU) |
| bf16 autocast behaviour on XPU (toy model, Lightning 2.6.6) | **measured** | see gotcha 1 |

Synthetic-data numbers say the pipeline runs on XPU; they say nothing about accuracy on real bricks.

## π0.5 (`train_pi05_xpu.py`)

Reuses Tan's cloud recipe: it imports `training/pi05/train.py` for dataset validation and the step estimate,
starts from `training/pi05/full.json`, checks the Studio checkout against `training/pi05/versions.env`
(commit `8f1324161ece2b24f2d38765b6e943ed1961258c`), and rewrites the trainer for one XPU:

- `accelerator: xpu`, `strategy: xpu_single`, `devices: 1`;
- bf16 autocast via an explicit `MixedPrecision(precision="bf16-mixed", device="xpu")` plugin
  (`--precision 32` for fp32); see gotcha 1 for why not `precision: bf16-mixed`;
- per-device `--batch` (default 2) with gradient accumulation up to `--global-batch` (default 32, as in the
  cloud run);
- gradient checkpointing on, `compile_model: false`;
- default **action-expert-only with a frozen vision encoder** (`train_expert_only: true`,
  `freeze_vision_encoder: true`), as in the cloud run. `--lora` trains LoRA adapters instead; Studio rejects
  LoRA combined with the freeze flags, so `--lora` turns them off. `--full-finetune` exists but the estimate says
  it does not fit;
- `--no-labels`: for datasets without `meta/dlsj/labels.jsonl` (Desmond's raw dataset). Episodes and frame
  counts come from LeRobot `meta/episodes/*.parquet`. **Success is not checked, so every episode is trained
  on.** Tan's success-only check runs when you omit the flag;
- before starting it refuses to run (unless `--force`) if `studio_export.py`, `physicalai fit`, `lerobot-train`
  or `dlsj.inspect.train` is running, or if `MemAvailable` < estimate + `--min-free-gib`. During the run it
  `nice`s the job (19) and **kills it if MemAvailable drops below `--min-free-gib` (default 8 GiB)**, then
  prints the lowest MemAvailable seen and the wall time.

Run with the **Studio library venv's python/CLI directly** (it has `pyarrow` and torch 2.11+xpu), from the repo root:

```bash
PY=~/inference/physical-ai-studio/library/.venv/bin/python
DS=~/desmond/dlsj/data/duplo_teleop_rgb      # or a finalized copy, see below

# 1. Validate + estimate + print the config. No writes, no XPU.
nice -n 19 $PY training/xpu/train_pi05_xpu.py $DS --no-labels --plan-only

# 2. Smoke test: one train batch + one val batch (fast_dev_run), no checkpoints. Has not passed yet (see Status).
nice -n 19 $PY training/xpu/train_pi05_xpu.py $DS --no-labels --smoke --output ~/inference/runs/pi05_xpu_smoke

# 3. Real run, bounded by steps and wall time; resumable.
nice -n 19 $PY training/xpu/train_pi05_xpu.py $DS --no-labels \
    --max-steps 2000 --time-limit-min 600 --output ~/inference/runs/pi05_xpu_$(date +%m%d_%H%M)
nice -n 19 $PY training/xpu/train_pi05_xpu.py $DS --no-labels --output <same dir> --resume <same dir>/checkpoints/resume/last.ckpt
```

Before a real run, **copy the dataset to a frozen snapshot**: Desmond's dataset was still growing during this
session (212 episodes in the brief, 417 when the plan ran). `--no-labels` refuses a dataset whose
`meta/episodes` and `info.json` disagree, but cannot tell an in-progress recording that happens to be
consistent from a finished one.

Needs `HF_TOKEN` (or `~/.cache/huggingface/token`) with access to `google/paligemma-3b-pt-224`. The first run
downloads `lerobot/pi05_base` (`model.safetensors` is 14.5 GB; it was not cached on the XPS on 14 Nov).

### Memory and time

The script prints an **arithmetic estimate** (not measured). It uses parameter counts from the deployed checkpoint's
card: 4.146 B total, ~695.5 M trainable when expert-only. It counts bf16 weights, fp32 grads and AdamW moments
for the trainable parameters, a guessed 1 GiB per sample of activations with checkpointing, and 3 GiB of
overhead:

| Mode | Estimate (batch 2) |
|---|---|
| expert-only (default) | ~20.5 GiB (arithmetic). **Measured:** MemAvailable dropped by ~34 GiB during model load alone (smoke attempt 1), so plan on ≥40 GiB free |
| LoRA (trainable count assumed 150 M) | ~17 GiB, plus more activation memory (gradients flow through the VLM) |
| full fine-tune | ~62 GiB: does not fit |

On the iGPU, device memory **is** system RAM, so the data loader workers, the recorder, Studio's Docker UI and
any export all come out of the same 61 GiB. Expect a transient extra peak while weights load on the CPU before
moving to the XPU. **Time per step on the B390 has not been measured.** Run the smoke test, then time the first
~50 optimizer steps (`--max-steps 50`) and extrapolate before committing to a long run. For scale, the cloud
run did ~2,300 updates at global batch 32 in 50 minutes on 4× RTX PRO 6000.

## SmolVLA (`train_smolvla_xpu.py`)

Same flags and safety checks. The config follows Studio's `configs/physicalai/smolvla/pusht/default.yaml`
with the PLAYBOOK 4.2 changes: `xpu`/`xpu_single`, `num_cameras: 3` (not 0), `val_gym: null` with a 10 %
episode validation split, pretrained `lerobot/smolvla_base`, expert-only + frozen vision (`--train-vlm` to
train the VLM too), batch 8 × accumulate 8 = 64, ColorJitter as in the Studio config. Estimate ~7 GiB at batch 8
(not measured).

```bash
nice -n 19 $PY training/xpu/train_smolvla_xpu.py $DS --no-labels --plan-only
nice -n 19 $PY training/xpu/train_smolvla_xpu.py $DS --no-labels --smoke --output ~/inference/runs/smolvla_xpu_smoke
```

## Anomalib + mark classifier (`train_anomalib_xpu.sh`)

Wraps our existing trainers in the Anomalib env (`~/inference/.venv-anomalib`, torch 2.15+xpu). Both export a
static-shape OpenVINO IR and calibrate or evaluate it on `CALIB_DEVICE` (default `NPU`; use `CPU` when someone
else is using the NPU). Run from the repo root on the XPS:

```bash
# stage 1: PaDiM on plain bricks (good/), threshold calibrated with bad/ + minor/
CALIB_DEVICE=CPU training/xpu/train_anomalib_xpu.sh padim ~/inference/inspect_data/synth3_train ~/inference/exports/inspect/padim_xpu
# equivalent to:
nice -n 19 ~/inference/.venv-anomalib/bin/python -m dlsj.inspect.train --data <frames> --out <out> --model padim --accelerator xpu --calib-device CPU

# stage 2: circle vs cross CNN (EPOCHS default 15)
CALIB_DEVICE=CPU training/xpu/train_anomalib_xpu.sh classifier ~/inference/inspect_data/mark_crops_synth3 ~/inference/exports/inspect/mark_cnn_xpu ~/inference/inspect_data/mark_crops_synth3_test
# equivalent to:
nice -n 19 ~/inference/.venv-anomalib/bin/python -m dlsj.inspect.train_classifier --data <crops> --out <out> --arch mobilenet_v3_small --epochs 15 --accelerator xpu --device CPU --test <test crops>
```

PaDiM has no gradient training (one pass to fit a Gaussian per patch), so the XPU saves only a few seconds
over CPU. The trainers default to CPU so that the GPU stays free for π0.5.

During export, the PaDiM log shows a caught traceback that does not affect the result. The IR is written and
calibrated afterwards:
`RuntimeError: /project/onnx/version_converter/BaseConverter.h:63: adapter_lookup: Assertion `false` failed: No Adapter To Version $17 for Pad`
(onnxscript falls back and "The model was not modified").

## Gotchas

1. **`precision: bf16-mixed` silently runs fp32 on XPU** with Lightning 2.6.6 (Studio 8f13241). Lightning
   builds `MixedPrecision(device="cuda")` for every accelerator except cpu/mps. On the XPS, a toy model under
   `accelerator=xpu, strategy=xpu_single` gave:
   - `precision="bf16-mixed"`: output `float32`, `torch.is_autocast_enabled("xpu") == False`, warning
     *"CUDA is not available or torch_xla is imported. Disabling autocast."*
   - plugin `MixedPrecision("bf16-mixed", "xpu")`: output `bfloat16`, autocast enabled, no warning. **The
     scripts use this.**
   - `precision="16-mixed"`: fp32 again, plus *"torch.cuda.amp.GradScaler is enabled, but CUDA is not
     available. Disabling."* fp16 needs a GradScaler, and Lightning's is CUDA-only, so the scripts only offer
     bf16 (no scaler needed) or 32.

   π0.5's `dtype: bfloat16` keeps most weights in bf16 regardless, so the flag mostly affects the fp32-kept
   layers and activation dtype. PLAYBOOK 4.2's `--trainer.precision bf16-mixed` hits this issue. The plugin
   config parses with `physicalai fit --print_config`, but has not been run through a full π0.5 step.
2. **Do not use `uv run`** in `physical-ai-studio/library`: it re-syncs the env and drops the `xpu` extra,
   which leaves you with CPU torch. Call `.venv/bin/python` / `.venv/bin/physicalai` directly.
3. **Shared memory.** The iGPU allocates from system RAM. Stop the recorder (and any export) before a π0.5
   run. Keep `--num-workers` low (each worker decodes three 640×480 videos). Run everything under `nice -n 19`.
   The launcher's kill threshold is the last line of defence, not a plan.
4. **Studio version.** Train with the same Studio commit the XPS exports with (`versions.env`). The launcher
   refuses a mismatch.
5. **Labels.** `--no-labels` trains on every episode, including failed or aborted demos if the recorder kept
   them. Prefer a prepared success-only copy with `labels.jsonl` (then drop `--no-labels`).
6. **torchcodec does not load in the Studio venv on the XPS** (no system FFmpeg 4–8 shared libs), and LeRobot
   picks it by default because the Python package imports. The scripts pass `video_backend: pyav` (`--video-backend`).
   Installing FFmpeg system-wide (`apt install ffmpeg` gives libavutil) would also fix it. That is a sysadmin call on a shared machine.
7. Resume uses Studio's fit namespace: `--fit.ckpt_path` (the launcher adds it for `--resume`).
