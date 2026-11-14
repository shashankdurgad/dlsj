"""Full π0.5 fine-tune (every weight trainable) with openpi's official recipe.

Values follow openpi ``pi05_droid_finetune`` + its default AdamW/cosine schedule
(global batch 32, 20k steps, peak LR 2.5e-5, 1k warmup, cosine to 2.5e-6 over
30k steps, weight decay 1e-10, clip 1.0). Cameras: wrist + top only.
Trains with the pinned Physical AI Studio so the XPS export path still works.
EMA (openpi ema_decay=0.99) is NOT applied: the pinned Studio Pi05 has no EMA.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from train import HERE, inspect_dataset

CAMERAS = ("wrist", "top")
GLOBAL_BATCH = 32
RECIPE = {
    "train_expert_only": False,
    "freeze_vision_encoder": False,
    "optimizer_lr": 2.5e-5,
    "optimizer_betas": [0.9, 0.95],
    "optimizer_eps": 1e-8,
    "optimizer_weight_decay": 1e-10,
    "optimizer_grad_clip_norm": 1.0,
    "scheduler_warmup_steps": 1000,
    "scheduler_decay_steps": 30000,
    "scheduler_decay_lr": 2.5e-6,
    "normalization_mode": "QUANTILES",
    "chunk_size": 50,
    "n_action_steps": 50,
    "gradient_checkpointing": True,
    "compile_model": False,
    "lora_enabled": False,
    "snapflow_enabled": False,
}


def build_config(*, dataset, pretrained_model, output, episodes, gpus, max_steps=20000,
                 val_every=1000, val_batches=16, resume_every=2000, deadline_seconds=None,
                 smoke=False):
    if GLOBAL_BATCH % gpus:
        raise ValueError(f"global batch {GLOBAL_BATCH} is not divisible by {gpus} GPUs")
    per_gpu = GLOBAL_BATCH // gpus
    cfg = json.loads((HERE / "full.json").read_text())
    cfg["model"]["init_args"].update(RECIPE, pretrained_name_or_path=str(pretrained_model))
    cfg["data"]["class_path"] = "phase_data.PhaseDataModule"
    cfg["data"]["init_args"].update(root=str(Path(dataset).resolve()), episodes=list(episodes),
                                    train_batch_size=per_gpu, val_batch_size=8, num_workers=6,
                                    video_backend="pyav")
    callbacks = [{"class_path": "official_callbacks.OfficialCheckpoints",
                  "init_args": {"resume_every": resume_every}},
                 {"class_path": "lightning.pytorch.callbacks.LearningRateMonitor",
                  "init_args": {"logging_interval": "step"}}]
    if deadline_seconds:
        seconds = max(60, int(deadline_seconds))
        callbacks.append({"class_path": "lightning.pytorch.callbacks.Timer", "init_args": {
            "duration": {"hours": seconds // 3600, "minutes": seconds % 3600 // 60, "seconds": seconds % 60},
            "interval": "step"}})
    cfg["trainer"].update(devices=gpus, strategy="ddp", max_steps=max_steps, max_epochs=-1,
                          accumulate_grad_batches=1, precision="bf16-mixed",
                          check_val_every_n_epoch=None, val_check_interval=val_every,
                          limit_val_batches=val_batches, num_sanity_val_steps=1,
                          enable_checkpointing=False, enable_progress_bar=False,
                          default_root_dir=str(Path(output).resolve()), callbacks=callbacks)
    if smoke:
        cfg["trainer"].update(fast_dev_run=1, callbacks=[], logger=False,
                              check_val_every_n_epoch=1, val_check_interval=1.0)
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", type=Path)
    p.add_argument("--pretrained-model", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpus", type=int, default=8, choices=(1, 2, 4, 8))
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--resume-every", type=int, default=2000)
    p.add_argument("--resume", type=Path, help="Lightning resume checkpoint (with optimizer)")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    _, _, digest = inspect_dataset(args.dataset, cameras=CAMERAS)
    plan = json.loads((args.dataset / "CURATION.json").read_text())
    studio = Path(os.environ["STUDIO_DIR"])
    actual = subprocess.check_output(["git", "-C", str(studio), "rev-parse", "HEAD"], text=True).strip()
    expected = (HERE / "versions.env").read_text().split("STUDIO_COMMIT=", 1)[1].strip()
    if actual != expected:
        raise SystemExit(f"Studio revision mismatch: expected {expected}, got {actual}")
    if args.output.exists() and not args.resume:
        raise SystemExit("Output directory already exists; choose a new --output or pass --resume")
    args.output.mkdir(parents=True, exist_ok=True)
    deadline = None
    if os.environ.get("RUN_DEADLINE"):
        deadline = float(os.environ["RUN_DEADLINE"]) - time.time() - 900  # 15 min to save + upload
    cfg = build_config(dataset=args.dataset, pretrained_model=args.pretrained_model, output=args.output,
                       episodes=plan["included_episodes"], gpus=args.gpus, max_steps=args.max_steps,
                       val_every=args.val_every, resume_every=args.resume_every,
                       deadline_seconds=deadline, smoke=args.smoke)
    path = args.output / ("training-config.json" if not args.resume else f"training-config-resume-{int(time.time())}.json")
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    plan.update(dataset_metadata_sha256=digest, studio_commit=actual, cameras=list(CAMERAS),
                gpus=args.gpus, per_gpu_batch=GLOBAL_BATCH // args.gpus, global_batch=GLOBAL_BATCH,
                max_steps=args.max_steps, base_model=args.pretrained_model, ema="not available in pinned Studio Pi05",
                resume_from=str(args.resume) if args.resume else None)
    (args.output / "data-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    env = os.environ.copy()
    env.update(PYTHONPATH=str(HERE) + os.pathsep + env.get("PYTHONPATH", ""), OMP_NUM_THREADS="8")
    env.setdefault("NCCL_P2P_LEVEL", "SYS" if args.gpus == 8 else "PHB")
    command = [str(studio / "library/.venv/bin/physicalai"), "fit", "--config", str(path.resolve())]
    if args.resume:
        command += ["--fit.ckpt_path", str(args.resume.resolve())]
    subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()
