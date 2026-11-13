# GCP pi0.5 smoke test, 2026-11-13

Project: `onflow-30k`. VM: `pi05-smoke-20261113`, `europe-north1-a`,
`g4-standard-48`, Spot, Ubuntu 24.04 PyTorch DLVM image
`pytorch-2-9-cu129-ubuntu-2404-nvidia-580-v20261111`.
The training packages were installed in a separate pinned Studio environment.

## Observed result

- NVIDIA RTX PRO 6000 Blackwell Server Edition; 95.0 GiB reported device memory.
- Driver 580.178.04; PyTorch 2.11.0+cu128; Studio
  `8f1324161ece2b24f2d38765b6e943ed1961258c`.
- Base model: `lerobot/pi05_base`, revision
  `22c30e5b34f9e7ab738b1b34fc2fc5ecfe2d28f0`.
- Full 4.1B-parameter model, BF16 mixed precision, per-GPU batch 8,
  gradient checkpointing enabled, compilation/LoRA/SnapFlow disabled.
- Synthetic LeRobot dataset: two 64-frame episodes, six joints, 30 fps,
  wrist/top/inspect videos at 640 x 480; one episode held out.
- Completed 30 optimizer updates and one sanity-validation batch. Training loss
  was checked for finiteness on every update. No persistent training checkpoint.
- 20 update intervals after warmup: median **1.025 s**, mean **1.063 s**,
  maximum **1.303 s**. Intervals include inter-batch/epoch loader work; validation,
  checkpoint writes, initial downloads, model initialization and warmup are excluded.
- Peak allocated GPU memory **39.600 GiB**; reserved **40.207 GiB**.

The initial runs found missing `libGL.so.1` and Python development headers.
Installing the documented Ubuntu system packages resolved those failures.
The base checkpoint lacks four `target_time_mlp_*` tensors added by Studio for
SnapFlow. Those heads are gated on SnapFlow, which was disabled in this run.

## Evidence and reproduction

`pi05-benchmark-result.json`, `pi05-benchmark-config.json` and
`pi05-benchmark.log` record the completed run. The adjacent synthetic-data script,
callback and shell script are the exact harness used on this disposable VM.
Their `/tmp` and `/home/w3joe` paths describe this run, not required user paths.

To reproduce after setup, copy the three harness files into `/tmp`, run
`pi05-make-synthetic-dataset.py` with Studio's Python, and generate a config with:

```bash
python3 training/pi05/train.py /tmp/pi05-synthetic-dataset \
  --gpus 1 --batch 8 --smoke --output /tmp/pi05-smoke-run
# The preceding command independently exercises one training/validation batch.
bash /tmp/pi05-run-benchmark.sh
```

The final cached-model smoke run also exited successfully with one training
batch (`train/loss: 1.066`) and one validation batch (`val/loss: 0.385`). Its
log and generated config are saved here. An earlier run exposed a checkpoint
write despite fast-dev mode; the launcher now explicitly removes checkpoint
callbacks and disables checkpointing/logging for smoke runs. The corrected run
left only 12 KiB of configuration/metadata, with no checkpoint files.

The updated setup script passed the CUDA, Triton driver compilation and CLI
import checks on the VM. Seven local launcher/preflight tests passed. The
temporary VM was deleted after collecting these artifacts; the model cache
remains in GCS for future training machines.

## Cache integrity

The private GCS bucket has uniform bucket-level access and enforced public-access
prevention. `cache-manifest.json` records the weights' object generation, CRC32C,
source revision and SHA-256. Downloading to a fresh directory from GCS produced
the same full SHA-256 as the Hugging Face source. Tokens were not uploaded.

## Estimate for 1,000 demonstrations

Assuming 10 seconds/demo at 30 fps, a 10% episode holdout and 8 epochs:

| Configuration | Effective batch | Updates | Compute time if mean step remains 1.063 s |
|---|---:|---:|---:|
| 4 GPUs x batch 8 | 32 | 67,496 | 19.9 h |
| 8 GPUs x batch 8 | 64 | 33,744 | 10.0 h |

These are conditional extrapolations from **one GPU**, not measured multi-GPU
training times. DDP communication, real video loading, validation, large optimizer
checkpoints and Spot recovery add time. Eight GPUs also change the global batch
and number of updates, which can affect learning. Double demonstration length
roughly doubles the update budget. Eight epochs is a starting budget, not evidence
of convergence; select the final policy using held-out physical task performance.

Four-GPU Spot attempts failed in `us-central1-b`, `us-central1-c`,
`europe-north1-b`, and `europe-north1-c`; another was preempted before boot in
`europe-west1-b`. This test does not establish four/eight-GPU capacity or NCCL
correctness. Synthetic-data loss does not establish robot behavior or data quality.
