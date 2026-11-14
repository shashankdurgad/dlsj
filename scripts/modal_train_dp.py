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
augmentation (UMI, Chi et al. 2024) is on for robustness to lighting. Metrics go to W&B (Modal secret `wandb-dlsj`, entity james-j-carver-university-of-cambridge).
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
    .add_local_python_source("dlsj")  # deploy code reused by the checkpoint sweep (run from the repo root)
)

data = modal.Volume.from_name("dlsj-teleop-data", create_if_missing=True)
runs = modal.Volume.from_name("dlsj-act-runs", create_if_missing=True)

# Stronger augmentation: every sample gets all six transforms, with a wider shift/rotation than LeRobot's default
# (+-5 deg, 5%); combine with a smaller random crop (--crop) to fight overfitting to the ~300 training scenes.
STRONG_AUG = [
    "--dataset.image_transforms.max_num_transforms=6",
    '--dataset.image_transforms.tfs={'
    '"brightness": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"brightness": [0.75, 1.25]}}, '
    '"contrast": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"contrast": [0.75, 1.25]}}, '
    '"saturation": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"saturation": [0.5, 1.5]}}, '
    '"hue": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"hue": [-0.05, 0.05]}}, '
    '"sharpness": {"weight": 1.0, "type": "SharpnessJitter", "kwargs": {"sharpness": [0.5, 1.5]}}, '
    '"affine": {"weight": 1.0, "type": "RandomAffine", "kwargs": {"degrees": [-8.0, 8.0], "translate": [0.08, 0.08]}}}',
]

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
              volumes={"/data": data, "/runs": runs}, secrets=[modal.Secret.from_name("wandb-dlsj")])
def train(dataset: str = "duplo_pickset_300", run: str = "dp_pick_v1", steps: int = 20_000, batch_size: int = 32,
          gpus: int = 8, workers: int = 7, save_freq: int = 2_000, wandb_project: str = "dlsj-dp",
          crop: str = "324,432", wandb_entity: str = "james-j-carver-university-of-cambridge",
          strong_aug: bool = False) -> str:
    import os
    import shutil

    import wandb

    # Train from the container's local disk: random video seeks over the Modal Volume (a network filesystem) cost
    # ~235 ms per sample vs ~6 ms locally, which starved the GPUs.
    t_copy = time.time()
    local_root = f"/tmp/{dataset}"
    shutil.copytree(f"/data/{dataset}", local_root, dirs_exist_ok=True)
    print(f"copied dataset to local disk in {time.time() - t_copy:.0f}s", flush=True)

    cmd = [
        "torchrun", "--standalone", f"--nproc_per_node={gpus}", "-m", "lerobot.scripts.lerobot_train",
        f"--dataset.repo_id=local/{dataset}", f"--dataset.root={local_root}", "--dataset.video_backend=torchcodec",  # 3.6x faster than pyav here, 1 ffmpeg thread per decoder
        *[a for a in DP_ARGS if not a.startswith("--policy.crop_shape=")], f"--policy.crop_shape=[{crop}]",
        *(STRONG_AUG if strong_aug else []), "--policy.device=cuda", "--policy.push_to_hub=false",
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


@app.function(image=image, gpu="L4", cpu=8, memory=32768, timeout=3600, volumes={"/data": data, "/runs": runs})
def sweep(run: str = "dp_pick_288", heldout: str = "duplo_pickset_heldout_288", steps: str = "", probes: int = 4,
          seed: int = 0, ablate: bool = False) -> str:
    """Score each checkpoint's EMA weights on held-out episodes: error of the predicted 0.5 s chunk (15 actions) vs the
    operator's actions, against a hold-still baseline, at random moments and in the 0.5 s before the grasp.

        modal run scripts/modal_train_dp.py::sweep --run dp_pick_288 --heldout duplo_pickset_heldout_288
    """
    import glob
    import os

    import numpy as np
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets import LeRobotDataset
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    from dlsj.deploy.async_chunks import AsyncChunker

    ckpts = sorted(glob.glob(f"/runs/{run}/checkpoints/[0-9]*/pretrained_model_ema"))
    if steps:
        ckpts = [c for c in ckpts if int(c.split("/")[-2]) in {int(x) for x in steps.split(",")}]
    root = f"/tmp/{heldout}"
    import shutil

    shutil.copytree(f"/data/{heldout}", root, dirs_exist_ok=True)
    ds = LeRobotDataset(f"local/{heldout}", root=root, video_backend="torchcodec")
    fe, A_all = np.array(ds.hf_dataset["episode_index"]), np.stack(ds.hf_dataset["action"])
    rng = np.random.default_rng(seed)
    probe_rows = []  # (row, kind)
    for e in np.unique(fe):
        r = np.flatnonzero(fe == e)
        g = A_all[r, 5]
        d = g[12:] - g[:-12]
        closes = np.flatnonzero(d < -15)
        for t in rng.integers(2, len(r) - 16, probes):
            probe_rows.append((int(r[t]), "random"))
        if len(closes):
            gt = int(closes[-1])
            for t in (gt - 15, gt - 8):
                if 1 <= t and t + 15 <= len(r):
                    probe_rows.append((int(r[t]), "pre-grasp"))

    def obs(row):
        it = ds[row]
        o = {"observation.state": it["observation.state"].numpy().astype(np.float32)}
        for k in ds.meta.video_keys:
            o[k] = (it[k].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        return o

    cache = {row: (obs(row - 1), obs(row)) for row, _ in probe_rows}
    out = []
    names = ["pan", "lift", "elbow", "wflex", "wroll"]
    for c in ckpts:
        cfg = PreTrainedConfig.from_pretrained(c)
        cfg.device, cfg.n_action_steps, cfg.num_inference_steps = "cuda", 15, 8
        pol = get_policy_class(cfg.type).from_pretrained(c, config=cfg).to("cuda").eval()
        pre, post = make_pre_post_processors(pol.config, pretrained_path=c,
                                             preprocessor_overrides={"device_processor": {"device": "cuda"}})
        ch = AsyncChunker(pol, pre, post, torch.device("cuda"))
        variants = [None] + (list(ds.meta.video_keys) if ablate else [])
        for blank in variants:
          res = {"random": [], "pre-grasp": []}
          for row, kind in probe_rows:
            o1, o2 = cache[row]
            if blank is not None:  # this camera replaced by flat grey
                o1, o2 = dict(o1), dict(o2)
                o1[blank] = np.full_like(o1[blank], 127)
                o2[blank] = np.full_like(o2[blank], 127)
            pred = ch.predict([(0, o1), (1, o2)])
            true = A_all[row: row + len(pred)]
            err = np.abs(pred[:, :5] - true[:, :5]).mean(0)
            hold = np.abs(true[:, :5] - o2["observation.state"][:5]).mean(0)
            res[kind].append((err, hold))
          step = int(c.split("/")[-2])
          tag = "all cameras" if blank is None else f"no {blank.split('.')[-1]}"
          for kind, v in res.items():
            if not v:
                continue
            err = np.median(np.stack([x[0] for x in v]), 0)
            hold = np.median(np.stack([x[1] for x in v]), 0)
            line = (f"step {step:6d} {tag:13s} {kind:9s} n={len(v):3d} model "
                    + " ".join(f"{n} {e:5.1f}" for n, e in zip(names, err))
                    + f" | mean {err.mean():5.2f} vs hold-still {hold.mean():5.2f}")
            print("SWEEP", line, flush=True)
            out.append(line)
        del pol
        torch.cuda.empty_cache()
    return "\n".join(out)


@app.function(image=image, cpu=8, timeout=600, volumes={"/data": data})
def check_decode(dataset: str = "duplo_pickset_300_288") -> str:
    """Time LeRobot sample loading with each video backend inside the training image."""
    import time as _t

    import numpy as np
    from lerobot.datasets import LeRobotDataset

    fps = 30
    dt = {k: [-1 / fps, 0.0] for k in ("observation.state", "observation.images.wrist", "observation.images.top",
                                         "observation.images.inspect")}
    dt["action"] = [i / fps for i in range(-1, 15)]
    out = []
    for backend in ("pyav", "torchcodec"):
        try:
            ds = LeRobotDataset(f"local/{dataset}", root=f"/data/{dataset}", delta_timestamps=dt, video_backend=backend)
            idx = np.random.default_rng(0).integers(0, ds.num_frames, 40)
            ds[int(idx[0])]
            t = _t.perf_counter()
            for i in idx[1:]:
                ds[int(i)]
            out.append(f"{backend}: {(_t.perf_counter() - t) / (len(idx) - 1) * 1000:.1f} ms/sample")
        except Exception as e:  # noqa: BLE001
            out.append(f"{backend}: FAILED {str(e).splitlines()[0][:160]}")
    print("DECODE BENCH:", " | ".join(out), flush=True)
    return " | ".join(out)


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
         gpus: int = 8, workers: int = 7, save_freq: int = 2_000, crop: str = "324,432", wandb_entity: str = "james-j-carver-university-of-cambridge",
         strong_aug: bool = False):
    """--crop H,W: random-crop size, ~90% of the stored images (360x480 -> 324,432; 288x384 -> 260,346)."""
    fn = train.with_options(gpu=f"H100:{gpus}", cpu=min(64, 8 * gpus), memory=32768 * gpus)  # Modal: <= 64 cores
    print(fn.remote(dataset=dataset, run=run, steps=steps, batch_size=batch_size, gpus=gpus, workers=workers,
                    save_freq=save_freq, crop=crop, wandb_entity=wandb_entity, strong_aug=strong_aug))
