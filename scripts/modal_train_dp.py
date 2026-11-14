"""Train LeRobot's Diffusion Policy (Chi et al. 2023) on the pick-only set on a multi-GPU Modal node.

Task: pick a brick and bring it to the inspection camera (episodes end at the in-view marker + 0.3 s;
built by scripts/build_pick_set.py). The policy takes no text: the data contains only this one stage.

1. Upload the dataset:
       MODAL_PROFILE=desmond-zee modal volume put dlsj-teleop-data data/duplo_pickset_300 /duplo_pickset_300
2. Train, detached (smoke test: --steps 200 --run dp_smoke):
       MODAL_PROFILE=desmond-zee modal run --detach scripts/modal_train_dp.py --run dp_pick_v1 --gpus 8
3. Fetch a checkpoint (every --save-freq steps; also checkpoints/last):
       MODAL_PROFILE=desmond-zee modal volume get dlsj-act-runs /dp_pick_v1/checkpoints/last/pretrained_model ./outputs/dp_pick_v1

Hyperparameters follow the Diffusion Policy paper's real-world CNN setup (Chi et al. 2023, Sec. 5 / App. A):
observation horizon 2, prediction horizon 16, 8 actions executed per call, ResNet-18 (ImageNet weights, a
separate encoder per camera, spatial-softmax keypoints), 360x480 images (75% of the 480x640 frames; the cropped
560x480 top view is padded to 640 wide first, so aspect ratios are kept) with a random 324x432 crop (~90%), DDPM training with 100 steps and DDIM with 16 steps at inference, AdamW lr 1e-4 / wd 1e-6, cosine schedule
with 500 warmup steps, and the reference EMA (power 0.75, LeRobot `--ema.type=warmup`). Colour/brightness image
augmentation (UMI, Chi et al. 2024) is on for robustness to lighting. Metrics go to W&B (Modal secret `wandb`).
"""

import re
import subprocess
import threading
import time

import modal

app = modal.App("dlsj-dp")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .uv_pip_install("lerobot[dataset,training,diffusion]==0.7.0", "wandb")
)

data = modal.Volume.from_name("dlsj-teleop-data", create_if_missing=True)
runs = modal.Volume.from_name("dlsj-act-runs", create_if_missing=True)

_KV = re.compile(r"(?:^|\s)([a-z_/]+):(-?[0-9.]+(?:e[-+]?\d+)?)([KMB]?)(?=\s|$)")
_SCALE = {"": 1, "K": 1e3, "M": 1e6, "B": 1e9}
LOG_FREQ = 50

# Chi et al. 2023 real-world CNN Diffusion Policy, as LeRobot 0.7 arguments.
DP_ARGS = [
    "--policy.type=diffusion",
    "--policy.n_obs_steps=2", "--policy.horizon=16", "--policy.n_action_steps=8",
    "--policy.drop_n_last_frames=7",  # horizon - n_action_steps - n_obs_steps + 1
    # images are stored at 360x480: every camera at 75% of its 480x640 frame, the cropped 560x480 top view padded
    # to 640 first (build_pick_set.py --pad-to 640,480 --resize 360,480), so aspect ratios are kept;
    # random ~90% crop as in the paper
    "--policy.crop_shape=[324,432]", "--policy.crop_is_random=true",
    "--policy.vision_backbone=resnet18", "--policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1",
    "--policy.use_separate_rgb_encoder_per_camera=true", "--policy.spatial_softmax_num_keypoints=32",
    "--policy.noise_scheduler_type=DDIM", "--policy.num_train_timesteps=100", "--policy.num_inference_steps=16",
    "--policy.optimizer_lr=1e-4", "--policy.optimizer_weight_decay=1e-6",
    "--policy.scheduler_name=cosine", "--policy.scheduler_warmup_steps=500",
    "--ema.type=warmup",
    "--accelerator.mixed_precision=bf16",  # ~2x faster than fp32 on H100; no loss scaling needed
    "--dataset.image_transforms.enable=true",
    "--dataset.image_transforms.backend=gpu",  # colour augmentation on the GPU, not the dataloader CPUs
]


def parse_log_line(line: str) -> dict[str, float] | None:
    if "step:" not in line or "loss:" not in line:
        return None
    out = {k: float(v) * _SCALE[s] for k, v, s in _KV.findall(line)}
    return out if "step" in out and "loss" in out else None


@app.function(image=image, gpu="H100:8", cpu=64, memory=262144, timeout=12 * 3600,
              volumes={"/data": data, "/runs": runs}, secrets=[modal.Secret.from_name("wandb")])
def train(dataset: str = "duplo_pickset_300", run: str = "dp_pick_v1", steps: int = 20_000, batch_size: int = 32,
          gpus: int = 8, workers: int = 4, save_freq: int = 2_000, wandb_project: str = "dlsj-dp",
          crop: str = "324,432", wandb_entity: str = "james-j-carver-university-of-cambridge") -> str:
    import os

    import wandb

    cmd = [
        "torchrun", "--standalone", f"--nproc_per_node={gpus}", "-m", "lerobot.scripts.lerobot_train",
        f"--dataset.repo_id=local/{dataset}", f"--dataset.root=/data/{dataset}", "--dataset.video_backend=pyav",
        *[a for a in DP_ARGS if not a.startswith("--policy.crop_shape=")], f"--policy.crop_shape=[{crop}]",
        "--policy.device=cuda", "--policy.push_to_hub=false",
        f"--output_dir=/runs/{run}", f"--job_name={run}", f"--batch_size={batch_size}", f"--steps={steps}",
        f"--save_freq={save_freq}", f"--log_freq={LOG_FREQ}", f"--num_workers={workers}",
    ]
    print(" ".join(cmd), flush=True)
    config = {"dataset": dataset, "steps": steps, "batch_size_per_gpu": batch_size, "gpus": gpus,
              "global_batch": batch_size * gpus, "policy": "diffusion", "gpu": "H100", "args": DP_ARGS, "crop": crop}
    try:
        wb = wandb.init(project=wandb_project, name=run, config=config, entity=wandb_entity or None)
        print(f"W&B run: {wb.url}", flush=True)
    except Exception as e:  # noqa: BLE001  tracking must never stop training
        print(f"W&B disabled: {e}", flush=True)
        wb = None

    stop = threading.Event()

    def commit_periodically():  # make checkpoints visible outside the container during training
        while not stop.wait(300):
            runs.commit()

    threading.Thread(target=commit_periodically, daemon=True).start()
    t0 = time.time()
    # One maths thread per process: 8 trainers + 8 x `workers` dataloader processes otherwise each spawn a thread
    # per core, oversubscribe the 64 vCPUs and starve the GPUs (seen: ~1.4 s/step, GPUs at ~30% power).
    env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    logged = 0
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
            m = parse_log_line(line)
            if wb is not None and m:
                logged += 1
                try:
                    wb.log({k: v for k, v in m.items() if k != "step"}, step=logged * LOG_FREQ)
                except Exception as e:  # noqa: BLE001
                    print(f"W&B log failed: {e}", flush=True)
        code = proc.wait()
    finally:
        stop.set()
        runs.commit()
        if wb is not None:
            wb.finish()
    if code != 0:
        raise RuntimeError(f"lerobot-train exited with {code}")
    return f"{run}: {steps} steps on {gpus}x H100 in {(time.time() - t0) / 60:.1f} min -> dlsj-act-runs:/{run}"


@app.function(image=image, cpu=2, timeout=600)
def check() -> str:
    """Cheap image warm-up / install check: modal run scripts/modal_train_dp.py::check"""
    import diffusers
    import lerobot
    import torch
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

    DiffusionConfig()
    return f"lerobot {lerobot.__version__}, diffusers {diffusers.__version__}, torch {torch.__version__}"


@app.local_entrypoint()
def main(dataset: str = "duplo_pickset_300", run: str = "dp_pick_v1", steps: int = 20_000, batch_size: int = 32,
         gpus: int = 8, workers: int = 4, save_freq: int = 2_000, crop: str = "324,432", wandb_entity: str = "james-j-carver-university-of-cambridge"):
    """--crop H,W: random-crop size, ~90% of the stored images (360x480 -> 324,432; 288x384 -> 260,346)."""
    fn = train.with_options(gpu=f"H100:{gpus}", cpu=min(64, 8 * gpus), memory=32768 * gpus)  # Modal: <= 64 cores
    print(fn.remote(dataset=dataset, run=run, steps=steps, batch_size=batch_size, gpus=gpus, workers=workers,
                    save_freq=save_freq, crop=crop, wandb_entity=wandb_entity))
