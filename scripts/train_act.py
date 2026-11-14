"""Train LeRobot's ACT on the Duplo teleop dataset on a multi-GPU Linux machine (e.g. a GCP G4 VM with
RTX PRO 6000s), with metrics sent to Weights & Biases.

One policy covers the whole take: pick the brick, present it to the inspection camera, place it on a plate.
ACT ignores the text instruction. See docs/train_act_gcp.md for the full VM setup.

    export HF_TOKEN=...                    # read access to the private dataset
    export WANDB_API_KEY=...               # optional; without it, metrics only go to the console
    python scripts/train_act.py --run act_417 --steps 20000
    python scripts/train_act.py --policy dit --run dit_517 --dataset <namespace>/dlsj-duplo-517

It downloads the dataset (default: akoniti/dlsj-duplo-417) to --data-dir, then runs
`torchrun -m lerobot.scripts.lerobot_train` on all visible GPUs (DDP). `--batch-size` is per GPU.
Checkpoints land in --output-dir/<run>/checkpoints/<step>/pretrained_model.
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# LeRobot log line: "step:1K smpl:32K ep:150 epch:0.68 loss:0.412 grdn:12.3 lr:1.0e-05 updt_s:0.21 data_s:0.01"
_KV = re.compile(r"(?:^|\s)([a-z_/]+):(-?[0-9.]+(?:e[-+]?\d+)?)([KMB]?)(?=\s|$)")
_SCALE = {"": 1, "K": 1e3, "M": 1e6, "B": 1e9}


def parse_log_line(line: str) -> dict[str, float] | None:
    """Metrics from a LeRobot training log line, or None if the line isn't one."""
    if "step:" not in line or "loss:" not in line:
        return None
    out = {k: float(v) * _SCALE[s] for k, v, s in _KV.findall(line)}
    return out if "step" in out and "loss" in out else None


# Per-policy lerobot-train arguments and defaults. `dit` is LeRobot's Multi-Task DiT: a diffusion transformer on
# CLIP image + text features, trained here with the flow-matching objective, so it conditions on the per-frame
# instruction (pick / place on the <colour> plate) and samples one trajectory instead of averaging modes.
POLICIES = {
    "act": {"args": ["--policy.type=act"], "batch_size": 16, "steps": 20_000},
    "dit": {"args": [
        "--policy.type=multi_task_dit",
        "--policy.objective=flow_matching",
        "--policy.image_resize_shape=[240,240]",  # whole view, then the default 224x224 (random) crop
        "--policy.num_integration_steps=10",  # inference ODE steps; the default 100 is too slow for the XPS
    ], "batch_size": 32, "steps": 30_000},
}


def gpu_count() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
        return sum(1 for line in out.splitlines() if line.startswith("GPU "))
    except (OSError, subprocess.CalledProcessError):
        return 0


def fetch_dataset(repo: str, dest: Path, revision: str | None) -> Path:
    if (dest / "meta" / "info.json").exists():
        print(f"dataset already at {dest}", flush=True)
        return dest
    from huggingface_hub import snapshot_download

    print(f"downloading {repo} -> {dest}", flush=True)
    snapshot_download(repo_id=repo, repo_type="dataset", local_dir=dest, revision=revision)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", choices=sorted(POLICIES), default="act",
                    help="act = LeRobot ACT; dit = Multi-Task DiT (flow matching, text-conditioned)")
    ap.add_argument("--run", default="act_417", help="run name (output folder and W&B run)")
    ap.add_argument("--dataset", default="akoniti/dlsj-duplo-417", help="HF dataset repo")
    ap.add_argument("--revision", default=None, help="dataset revision (branch, tag or commit)")
    ap.add_argument("--data-dir", type=Path, default=None, help="default: data/<dataset name>")
    ap.add_argument("--output-dir", type=Path, default=Path("outputs"))
    ap.add_argument("--gpus", type=int, default=0, help="0 = all visible GPUs")
    ap.add_argument("--steps", type=int, default=None, help="default: 20000 for act, 30000 for dit")
    ap.add_argument("--batch-size", type=int, default=None, help="per GPU; default 16 for act, 32 for dit")
    ap.add_argument("--save-freq", type=int, default=2_000)
    ap.add_argument("--log-freq", type=int, default=50)
    ap.add_argument("--workers", type=int, default=6, help="dataloader workers per GPU")
    ap.add_argument("--wandb-project", default=None, help="default: dlsj-<policy>")
    ap.add_argument("--resume", action="store_true", help="resume --run from its last checkpoint")
    ap.add_argument("extra", nargs="*", help="extra lerobot-train arguments, e.g. --policy.chunk_size=50")
    args = ap.parse_args()
    pol = POLICIES[args.policy]
    args.steps = args.steps or pol["steps"]
    args.batch_size = args.batch_size or pol["batch_size"]
    args.data_dir = args.data_dir or Path("data") / args.dataset.split("/")[-1]
    args.wandb_project = args.wandb_project or f"dlsj-{args.policy}"

    gpus = args.gpus or gpu_count()
    if gpus < 1:
        sys.exit("no NVIDIA GPU visible (nvidia-smi -L)")
    root = fetch_dataset(args.dataset, args.data_dir, args.revision)
    out = args.output_dir / args.run
    if out.exists() and not args.resume:
        sys.exit(f"{out} exists; pick another --run or pass --resume")

    if args.resume:
        cmd = ["torchrun", "--standalone", f"--nproc_per_node={gpus}", "-m", "lerobot.scripts.lerobot_train",
               f"--config_path={out}/checkpoints/last/pretrained_model/train_config.json", "--resume=true"]
    else:
        cmd = [
            "torchrun", "--standalone", f"--nproc_per_node={gpus}", "-m", "lerobot.scripts.lerobot_train",
            f"--dataset.repo_id=local/{root.name}", f"--dataset.root={root}", "--dataset.video_backend=pyav",
            *pol["args"], "--policy.device=cuda", "--policy.push_to_hub=false",
            f"--output_dir={out}", f"--job_name={args.run}", f"--batch_size={args.batch_size}",
            f"--steps={args.steps}", f"--save_freq={args.save_freq}", f"--log_freq={args.log_freq}",
            f"--num_workers={args.workers}", *args.extra,
        ]
    print(" ".join(cmd), flush=True)

    wb = None
    if os.environ.get("WANDB_API_KEY"):
        try:
            import wandb

            wb = wandb.init(project=args.wandb_project, name=args.run, config={
                "dataset": args.dataset, "revision": args.revision, "steps": args.steps, "gpus": gpus,
                "batch_size_per_gpu": args.batch_size, "global_batch": args.batch_size * gpus, "policy": args.policy})
        except Exception as e:  # noqa: BLE001  tracking must never stop training
            print(f"W&B disabled: {e}", flush=True)

    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    logged = 0
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
            m = parse_log_line(line)
            if wb is not None and m:
                logged += 1  # LeRobot rounds the displayed step (1500 -> "2K"); there is one line per log_freq steps
                try:
                    wb.log({k: v for k, v in m.items() if k != "step"}, step=logged * args.log_freq)
                except Exception as e:  # noqa: BLE001
                    print(f"W&B log failed: {e}", flush=True)
        code = proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        code = proc.wait()
    finally:
        if wb is not None:
            wb.finish()
    print(f"{args.run}: exit {code} after {(time.time() - t0) / 60:.1f} min; checkpoints in {out}/checkpoints",
          flush=True)
    sys.exit(code)


if __name__ == "__main__":
    main()
