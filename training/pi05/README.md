# Pi0.5 supervised imitation learning on GCP

This folder prepares **Physical AI Studio's training library**, with a separate
Python 3.12/CUDA environment. It does not change the recording environment or
start/provision a cloud VM. The initial recipe fully fine-tunes `lerobot/pi05_base`
using its flow-matching objective. LoRA and SnapFlow are disabled for the baseline.

## Hardware and sizing

Assumption: **RTX PRO 6000 Blackwell Server Edition, 96 GB per GPU**.
Confirm with `nvidia-smi`; RTX 6000 Ada is a different 48 GB card.

| GCP VM | GPUs | Total VRAM | vCPUs | System RAM |
|---|---:|---:|---:|---:|
| `g4-standard-192` | 4 | 384 GB | 192 | 720 GB |
| `g4-standard-384` | 8 | 768 GB | 384 | 1,440 GB |

In DDP, each GPU keeps its own model and optimizer. The total is not one pooled
memory allocation. Start with four GPUs; compare eight at the **same effective
batch size** before choosing the faster/cost-effective setup. Capacity and quota
must be checked separately. No cloud resources are provisioned by these scripts.

Conservative memory arithmetic for an approximately 4.1B-parameter model:

- 16 bytes/parameter for FP32 weights, gradients and two Adam moments is about
  **65.6 GB (61.1 GiB) per GPU** before activations, workspaces and communication.
- A 96 GB GPU nominally leaves about **30.4 GB** beyond that estimate. Some Pi0.5
  tensors use BF16, but actual optimizer/buffer behavior must be measured.
- BF16 mixed precision and gradient checkpointing are enabled; compilation is
  off initially. Per-GPU batch 8 is a **starting candidate**, not a demonstrated fit.
- Effective batch = GPUs × per-GPU batch × gradient accumulation. Defaults:
  4 × 8 × 1 = **32**; eight GPUs with batch 4 preserve 32; eight with batch 8 give 64.

Example dataset, **not a statement about the current recordings**: 150 successful
10-second episodes at 30 fps = 45,000 frames. A 10% episode validation split,
assuming equal episode lengths, leaves 40,500 training frames. With batch 32,
about 1,265 optimizer updates/epoch × 8 epochs = **10,120 updates**. With batch 64,
about 632 updates/epoch × 8 = **5,056 updates**. The differing step counts matter
when comparing runs. The launcher calculates the estimate using actual label
frame counts, distributed sampler padding and DataLoader tail dropping.

Measure steady-state **optimizer-step** time, including accumulation, after warmup.
For 10,120 updates, conditional training-only times are:

| Measured seconds/update | Wall time | GPU-hours on 4 GPUs |
|---:|---:|---:|
| 0.5 | 1.41 h | 5.62 |
| 1.0 | 2.81 h | 11.24 |
| 2.0 | 5.62 h | 22.49 |

These are arithmetic examples, not throughput predictions. Add model/data
downloads, validation, checkpoint writes and export. Cost = measured VM-hours ×
the selected region's whole-VM price, plus storage and networking. Eight GPUs do
not guarantee a 2× speedup. Put datasets on local fast storage and checkpoints
on durable storage; full optimizer checkpoints can occupy tens of GB each.

## Set up the training VM

Copy/clone this checkout to the VM, install NVIDIA drivers and
[uv](https://docs.astral.sh/uv/), then run from the repository root:

```bash
# Ubuntu 24.04 system libraries needed by OpenCV and video decoding.
sudo apt-get update
sudo apt-get install -y --no-install-recommends libgl1 libglib2.0-0t64 ffmpeg libturbojpeg python3.12-dev build-essential
bash training/pi05/setup.sh
training/pi05/.studio/library/.venv/bin/hf auth login
```

Accept the `google/paligemma-3b-pt-224` access conditions on Hugging Face first.
Use a token with read access and keep it out of Git. Setup checks out Studio at
the exact revision in `versions.env` and uses its **upstream lockfile**, including
the pinned Runtime source. That lock currently selects PyTorch 2.11/cu128 and
LeRobot 0.6.0. Do not install these into the recorder's LeRobot 0.7 environment.
`STUDIO_DIR` can relocate the checkout, but must be used consistently for setup
and training. The matching Studio/Runtime revision should also be used for export.

## Restore the cached base weights from GCS

The private `onflow-30k-pi05-artifacts` bucket in `europe-north1` contains the
base model at the exact Hugging Face revision below, with a checksum manifest.
The 14.47 GB weights were restored and their full SHA-256 verified against the
Hugging Face source. No authentication tokens are stored in the bucket.
Authenticate `gcloud` on the VM with an identity permitted to read this bucket:

```bash
mkdir -p "$HOME/models/pi05_base"
gcloud storage rsync --recursive \
  gs://onflow-30k-pi05-artifacts/models/lerobot/pi05_base/22c30e5b34f9e7ab738b1b34fc2fc5ecfe2d28f0/ \
  "$HOME/models/pi05_base/"
python3 training/pi05/train.py /data/so101_sft \
  --pretrained-model "$HOME/models/pi05_base" --gpus 4 --batch 8 \
  --output /checkpoints/pi05-run1
```

This avoids downloading the base weights from Hugging Face on each new VM.
The PaliGemma tokenizer is a separate small Hugging Face dependency; keep the
HF authentication steps above. Model initialization and moving weights into GPU
memory still happen at startup. Datasets and trained checkpoints are separate.

## Prepare the data

Finish recording and copy a **frozen local dataset snapshot** containing `meta/`,
`data/` and `videos/` to the VM. Keep the recorder's `meta/dlsj/labels.jsonl` sidecar.
The launcher checks metadata and requires:

- LeRobot v3, 30 fps; `wrist`, `top`, `inspect` image keys.
- Six measured joints and commanded actions in the recorder's original order.
- At least two finalized, successfully labelled episodes, with valid task strings.
- Exactly one label per episode and consistent recorded frame counts.

It rejects mixed success/failure recordings. Curate a **separate success-only
dataset** first; do not remove rows from the label sidecar to hide failed episodes.
Merge the three skills into one valid LeRobot dataset using the recorder-compatible
LeRobot tools, preserving/rebuilding correctly indexed labels. Keep the originals.
Suggested instruction strings from the playbook:

1. `pick up the brick and show it to the inspection camera`
2. `place the brick on the blue plate`
3. `place the brick on the pink plate`

Record each skill from the state where that skill will actually begin. Balance
the two place instructions and keep names, action units and camera order consistent
at deployment. The held-out validation **episodes** use seed 42. Loss validation is
not proof of physical task success. The pinned upstream DataModule computes missing
quantile normalization stats before the episode split; this is suitable for an
initial development check, but not a strict train-only-statistics evaluation.
Metadata inspection does not decode videos or check every recorded action; the
first GPU smoke run must exercise the actual loader and model.

## Plan, smoke-test, train, resume

All commands below run on the training VM, from the repository root:

```bash
# Metadata validation and budget only: no installation, writes or training.
python3 training/pi05/train.py /data/so101_sft --plan-only

# One training + validation batch using all four GPUs.
python3 training/pi05/train.py /data/so101_sft \
  --smoke --output /checkpoints/pi05-smoke

# Baseline: full SFT, 4 GPUs, per-GPU batch 8, 8 epochs.
python3 training/pi05/train.py /data/so101_sft \
  --gpus 4 --batch 8 --epochs 8 --output /checkpoints/pi05-run1

# Eight GPUs, preserving the baseline's global batch of 32.
python3 training/pi05/train.py /data/so101_sft \
  --gpus 8 --batch 4 --output /checkpoints/pi05-8gpu

# Resume the same run from its periodic recovery checkpoint.
python3 training/pi05/train.py /data/so101_sft \
  --output /checkpoints/pi05-run1 \
  --resume /checkpoints/pi05-run1/checkpoints/resume/last.ckpt
```

The wrapper invokes `physicalai fit`; it does not replace Studio's trainer.
Studio chooses its DDP strategy for multiple GPUs. Google's G4 P2P setting is
`PHB` for four GPUs and `SYS` for eight; the launcher sets this unless already
overridden. Each run writes its generated configuration and metadata digest next
to the logs. The digest covers metadata/labels, **not the full dataset contents**.
Keep the input snapshot immutable, preserve its cloud object generation/version,
and pin/cache the Hugging Face model revision for a reproducible production run.
Resume with the same dataset, batch, GPU count and epoch budget.

Recovery checkpoints are written every 250 optimizer steps under
`checkpoints/resume/`; the best validation checkpoint is kept separately under
`checkpoints/best/`. Use a persistent disk for the output directory and preserve
it when the Spot VM is deleted. An interruption loses progress since the last
completed checkpoint; there is no recovery checkpoint before step 250. These
full checkpoints include optimizer state and require substantial disk space.

After a successful smoke run, inspect per-GPU peak memory and time 100–200
steady-state updates. Adjust batch size only from those measurements. Set the
same dataset/camera mapping when using Studio's UI remote trainer; a UI-created
job will not automatically inherit this folder's settings.

## Export and deployment

Use the pinned environment's `physicalai export --help` and select
`physicalai.policies.Pi05`, the trained `.ckpt`, and backend `openvino`. Validate
PyTorch-vs-OpenVINO action agreement before controlling the robot. Then evaluate
held-out physical episodes and closed-loop success on the XPS. INT8 compression
and SnapFlow are follow-up experiments, each requiring accuracy and latency checks.

## Validation status

This folder can be checked without GPU packages:

```bash
python3 -m unittest discover -s training/pi05 -p 'test_*.py'
bash -n training/pi05/setup.sh
```

On 2026-11-13, a GCP `g4-standard-48` Spot VM passed 30 full-model optimizer
updates and one validation batch on synthetic three-camera data. With one RTX
PRO 6000, batch 8 and BF16 mixed precision, 20 post-warmup intervals averaged
1.063 seconds/update (median 1.025); peak allocated memory was 39.60 GiB.
See [the recorded configuration, results and limitations](validation/2026-11-13/REPORT.md).
Four-GPU Spot allocations were unavailable/preempted, so multi-GPU operation is
still unverified. No real-data quality, export, or robot success is claimed.

Sources: [GCP G4 specifications and P2P settings](https://docs.cloud.google.com/compute/docs/accelerator-optimized-machines),
[pinned Studio SO-101 recipe](https://github.com/open-edge-platform/physical-ai-studio/blob/8f1324161ece2b24f2d38765b6e943ed1961258c/library/configs/physicalai/pi05/so101/snapflow.yaml),
[Studio training library](https://github.com/open-edge-platform/physical-ai-studio/tree/8f1324161ece2b24f2d38765b6e943ed1961258c/library).
