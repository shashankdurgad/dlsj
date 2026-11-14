"""One-hour SmolVLA fine-tune (Studio defaults) on the 3-stage Duplo dataset, wrist + top.

Studio SmolVLA defaults match LeRobot's smolvla_base fine-tune recipe: train the
action expert only, frozen vision encoder, AdamW lr 1e-4 (0.9, 0.95), weight
decay 1e-10, clip 10, cosine to 2.5e-6. Global batch 64 (LeRobot default).
Training wall time is capped by a Timer; pass --max-steps from a measured
s/step so the cosine schedule ends inside the hour.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

from train import HERE, inspect_dataset

CAMERAS = ("wrist", "top")
GLOBAL_BATCH = 64


def build_config(*, dataset, pretrained_model, output, episodes, gpus, max_steps, minutes=60,
                 warmup=None, val_every=500, val_batches=16, resume_every=1000, smoke=False):
    if GLOBAL_BATCH % gpus:
        raise ValueError(f"global batch {GLOBAL_BATCH} is not divisible by {gpus} GPUs")
    warmup = min(1000, max(1, max_steps // 10)) if warmup is None else warmup
    cfg = {
        "model": {"class_path": "physicalai.policies.SmolVLA", "init_args": {
            "pretrained_name_or_path": str(pretrained_model), "dtype": "bfloat16",
            "chunk_size": 50, "n_action_steps": 50, "num_cameras": len(CAMERAS),
            "freeze_vision_encoder": True, "train_expert_only": True, "train_state_proj": True,
            "optimizer_lr": 1e-4, "optimizer_betas": [0.9, 0.95], "optimizer_eps": 1e-8,
            "optimizer_weight_decay": 1e-10, "optimizer_grad_clip_norm": 10.0,
            "scheduler_warmup_steps": warmup, "scheduler_decay_steps": max_steps,
            "scheduler_decay_lr": 2.5e-6, "compile_model": False}},
        "data": {"class_path": "phase_data.PhaseDataModule", "init_args": {
            "repo_id": "local/duplo517-3stage-2cam", "root": str(Path(dataset).resolve()),
            "data_format": "physicalai", "episodes": list(episodes),
            "train_batch_size": GLOBAL_BATCH // gpus, "val_batch_size": 16, "num_workers": 8,
            "val_split": 0.1, "val_split_seed": 42, "video_backend": "pyav"}},
        "trainer": {
            "accelerator": "gpu", "devices": gpus, "num_nodes": 1,
            "strategy": "ddp_find_unused_parameters_true", "precision": "bf16-mixed",
            "max_steps": max_steps, "max_epochs": -1, "accumulate_grad_batches": 1,
            "log_every_n_steps": 10, "check_val_every_n_epoch": None, "val_check_interval": val_every,
            "limit_val_batches": val_batches, "num_sanity_val_steps": 1,
            "enable_checkpointing": False, "enable_progress_bar": False,
            "default_root_dir": str(Path(output).resolve()),
            "callbacks": [
                {"class_path": "official_callbacks.OfficialCheckpoints",
                 "init_args": {"resume_every": resume_every, "snapshot_steps": [max_steps]}},
                {"class_path": "lightning.pytorch.callbacks.LearningRateMonitor",
                 "init_args": {"logging_interval": "step"}},
                {"class_path": "lightning.pytorch.callbacks.Timer",
                 "init_args": {"duration": {"minutes": minutes}, "interval": "step"}}]}}
    if smoke:
        cfg["trainer"].update(fast_dev_run=1, callbacks=[], logger=False,
                              check_val_every_n_epoch=1, val_check_interval=1.0)
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", type=Path)
    p.add_argument("--pretrained-model", default="lerobot/smolvla_base")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpus", type=int, default=4, choices=(1, 2, 4, 8))
    p.add_argument("--max-steps", type=int, required=True)
    p.add_argument("--minutes", type=int, default=60)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    _, _, digest = inspect_dataset(args.dataset, cameras=CAMERAS)
    plan = json.loads((args.dataset / "CURATION.json").read_text())
    studio = Path(os.environ["STUDIO_DIR"])
    actual = subprocess.check_output(["git", "-C", str(studio), "rev-parse", "HEAD"], text=True).strip()
    expected = (HERE / "versions.env").read_text().split("STUDIO_COMMIT=", 1)[1].strip()
    if actual != expected:
        raise SystemExit(f"Studio revision mismatch: expected {expected}, got {actual}")
    if args.output.exists():
        raise SystemExit("Output directory already exists")
    args.output.mkdir(parents=True)
    cfg = build_config(dataset=args.dataset, pretrained_model=args.pretrained_model, output=args.output,
                       episodes=plan["included_episodes"], gpus=args.gpus, max_steps=args.max_steps,
                       minutes=args.minutes, val_every=args.val_every, smoke=args.smoke)
    path = args.output / "training-config.json"
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    plan.update(dataset_metadata_sha256=digest, studio_commit=actual, cameras=list(CAMERAS),
                policy="SmolVLA", gpus=args.gpus, global_batch=GLOBAL_BATCH, max_steps=args.max_steps,
                wall_minutes=args.minutes, base_model=args.pretrained_model)
    (args.output / "data-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    env = os.environ.copy()
    env.update(PYTHONPATH=str(HERE) + os.pathsep + env.get("PYTHONPATH", ""), OMP_NUM_THREADS="8")
    env.setdefault("NCCL_P2P_LEVEL", "SYS" if args.gpus == 8 else "PHB")
    subprocess.run([str(studio / "library/.venv/bin/physicalai"), "fit", "--config", str(path.resolve())],
                   env=env, check=True)


if __name__ == "__main__":
    main()
