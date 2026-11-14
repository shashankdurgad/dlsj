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
uv pip install "lerobot[dataset,training,multi_task_dit]==0.7.0" wandb   # multi_task_dit: for --policy dit
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

## 3c. Diffusion transformer (flow matching) instead of ACT

`--policy dit` trains LeRobot's Multi-Task DiT: a diffusion transformer on CLIP image features **and the
per-frame instruction text** ("pick up the brick…", "place the brick in the <colour> bowl"), trained with the
flow-matching objective. Unlike ACT it samples one trajectory rather than averaging several, which addresses
ACT's hovering/jittering over the bricks. It also reads which plate to use from the instruction.

```bash
python scripts/train_act.py --policy dit --run dit_517 --dataset <namespace>/dlsj-duplo-517
```

- Defaults for `dit`: flow matching, frames resized to 240x240 and then cropped to 224 (random crop in training,
  centre crop at inference), 10 ODE steps at inference, 32-step action horizon with 24 executed per call,
  batch 32 per GPU, 30k steps. Override any of them as extra arguments, e.g. `--policy.num_integration_steps=20`.
- 214M trainable parameters: the CLIP encoders are fine-tuned at 0.1x the learning rate. Smoke-tested on a GB10
  (~10 GB at batch 8, loss falling).
- At deployment the policy needs the current instruction as `task`, alongside the cameras and state.

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
