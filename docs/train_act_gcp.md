# Train the ACT policy on GCP (RTX PRO 6000)

Trains LeRobot's standard ACT (ResNet-18 + transformer, 100-step action chunks) on the 417-episode Duplo
teleop dataset: pick the brick, show it to the inspection camera, place it on a plate. It is the quick baseline
while VLA data collection continues. ACT ignores the text instruction, so it learns the whole sequence from
the three cameras and the joint positions.

## The data

`akoniti/dlsj-duplo-417` (private Hugging Face dataset, 1.6 GB, LeRobot v3):

- 417 successful teleop episodes, 85,647 frames at 30 fps; cameras `wrist`, `top` and `inspect`, plus
  6 joint positions as state and action.
- The overhead camera (`top`) is **cropped to x 0-560 (560x480)** to remove the inspection station and
  bystanders. The same crop must be applied when the policy runs (export step below).
- Plate labels of 63 early episodes were corrected; see the dataset README. Labels don't affect ACT, but they
  matter for VLA training on the same data.

It was built from the raw recording with `scripts/prepare_train_set.py`. The raw recording on the XPS is never
edited.

## 1. VM

Use the same kind of VM as the π0.5 run: project `onflow-30k`, zone `europe-west4-b`, 4× RTX PRO 6000,
spot provisioning. The 4-GPU G4 machine type is `g4-standard-192`. Check what the zone offers with
`gcloud compute machine-types list --zones=europe-west4-b --filter="name~^g4"`.

```bash
gcloud compute instances create act-417 --project=onflow-30k --zone=europe-west4-b \
  --machine-type=g4-standard-192 --provisioning-model=SPOT --instance-termination-action=STOP \
  --image-family=ubuntu-2404-lts-amd64 --image-project=ubuntu-os-cloud --boot-disk-size=200GB \
  --maintenance-policy=TERMINATE
```

The VM needs an NVIDIA driver recent enough for Blackwell (570 or later, CUDA 12.8). A Deep Learning VM image
already has one; on plain Ubuntu install it as described in the GCP GPU-driver docs. Check it with `nvidia-smi`.

## 2. Environment

```bash
sudo apt-get update && sudo apt-get install -y git ffmpeg libgl1
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
git clone https://github.com/shashankdurgad/dlsj.git && cd dlsj
uv venv --python 3.12 .venv-train && source .venv-train/bin/activate
uv pip install "lerobot[dataset,training]==0.7.0" wandb
python -c "import torch; print(torch.__version__, torch.cuda.device_count(), torch.cuda.get_device_name(0))"
```

The PyPI torch wheels are built against CUDA 12.8 and support Blackwell. The last line should list 4 GPUs.

## 3. Train

```bash
export HF_TOKEN=hf_...          # token with read access to akoniti/dlsj-duplo-417
export WANDB_API_KEY=...        # optional; metrics go to W&B project dlsj-act
python scripts/train_act.py --run act_417 --steps 20000
```

- The script downloads the dataset to `data/dlsj-duplo-417`, then runs `lerobot_train` with DDP over all
  visible GPUs.
- The batch size is 16 per GPU, so 64 in total on 4 GPUs. On L40S this ran at about 4 steps/s, so 20k steps take
  roughly an hour on 4 GPUs. The RTX PRO 6000 should be at least as fast.
- Checkpoints are saved every 2k steps to `outputs/act_417/checkpoints/<step>/pretrained_model`, and the latest
  one is also at `checkpoints/last`.
- If the spot VM is preempted, restart it and run `python scripts/train_act.py --run act_417 --resume`.
- For a quick check first, run `python scripts/train_act.py --run smoke --steps 200 --save-freq 200`. It should
  print `step:200 ... loss:` lines within a few minutes and end with `exit 0`.

Loss starts at about 13 and drops below 3 within a few hundred steps. Use the W&B curve, or the console lines,
to see when it flattens.

## 3b. WARP ACT (WARP-BC weighted training)

Same VM, environment, dataset and command as above, plus one flag. Each training sample is weighted by a
per-frame score from a progress model trained on the fastest demos. Hesitant, fumbled stretches get weight 0;
the inspection hold, the successful grasp and the release are always kept. See `docs/WARP_BC.md`.

```bash
python scripts/train_act.py --run act_417_warp --steps 20000 \
    --warp-weights warp/duplo417/warp_weights.parquet
```

- The weights file is committed in the repo, so no extra download is needed. It was computed for exactly
  `akoniti/dlsj-duplo-417` (417 episodes, 85,647 frames). At start-up the script checks every episode's start
  index and length against the dataset, and stops if they differ. If the dataset is rebuilt with more episodes,
  the weights must be recomputed first (on the GB10: `dlsj.warp.features` → `train_rm` → `score`).
- Look for `WARP weights ...: {'episodes': 417, 'frames': 85647}` near the top of the log. Training lines then
  include `sample_weight_kept_in_batch` (≈0.6) and `sample_weight_mean_weight`.
- Everything else is identical to plain ACT, so the result can be compared with `act_417` directly. Use the same
  steps and batch size. Zero-weight samples still occupy batch slots, so each step effectively trains on ~60% of
  its batch.
- The checkpoint is a normal LeRobot ACT checkpoint. Export and deploy it exactly as in step 4, with the same crop.
- Resume after preemption: `python scripts/train_act.py --run act_417_warp --resume --warp-weights
  warp/duplo417/warp_weights.parquet`. Pass the flag again so the WARP trainer is used.

## 4. Bring the checkpoint back and export for the XPS

```bash
# from the VM to a bucket (or scp it directly)
gsutil -m cp -r outputs/act_417/checkpoints/last/pretrained_model gs://onflow-30k-pi05-artifacts/act/act_417/

# on a machine with the repo and LeRobot: export to OpenVINO, with the training crop
python -m dlsj.deploy.act_export --policy_path pretrained_model --out exports/act_417 --precision fp16 \
  --crop observation.images.top=0,0,560,480@640x480
python -m dlsj.deploy.act_export --benchmark exports/act_417 --device GPU   # on the XPS
```

The exported runtime, `dlsj.deploy.act_export.OpenVINOACT`, takes full 640x480 camera frames and applies the
crop itself. If a frame arrives at any other size, it raises an error rather than feeding the policy a wrong
view.

## Rebuilding the dataset with more episodes

On the XPS, stop the recorder cleanly so every take is sealed (`pkill -TERM -f "dlsj[.]teleop_record"`, as a
command of its own). Then export the colour streams and rebuild:

```bash
python -m lerobot.scripts.lerobot_edit_dataset --repo_id local/duplo_teleop --root data/duplo_teleop \
  --new_repo_id local/duplo_teleop_rgb --new_root data/duplo_teleop_rgb --operation.type remove_feature \
  --operation.feature_names '["observation.images.top_depth", "observation.images.inspect_depth"]'
python scripts/prepare_train_set.py --src data/duplo_teleop_rgb --dst data/duplo_act_N \
  --crop observation.images.top=0,0,560,480 \
  --corrections data/duplo_teleop/meta/dlsj/plate_corrections.json
hf upload <namespace>/<dataset> data/duplo_act_N . --repo-type dataset
```

Episodes from 312 onwards are recorded with correct plate labels, so the existing corrections file still
covers everything.
