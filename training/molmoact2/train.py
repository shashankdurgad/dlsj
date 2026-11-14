"""Fine-tune MolmoAct2-SO100_101 in Physical AI Studio with Ai2's recommended small-dataset recipe.

Recipe sources (see README.md): allenai/molmoact2 experiments/README.md ("LoRA Fine-Tuning"; "use LoRA for
smaller datasets or similar embodiments"; SO100/SO101 fine-tuning starts from allenai/MolmoAct2-SO100_101), its
`so100_so101` mixture (tag so100_so101_molmoact2, horizon 30, absolute joint pose), and LeRobot's MolmoAct2 docs
("< 200 demonstrations: global batch 16-32, LoRA"; image augmentation). Trained through Studio's own MolmoAct2
policy (pinned commit) so the checkpoint exports to OpenVINO for the XPS.

    python train.py <prepared dataset> --output runs/plan --plan-only
    python train.py <prepared dataset> --output runs/smoke --smoke            # 1 train + 1 val batch
    python train.py <prepared dataset> --output runs/timing --steps 200
    python train.py <prepared dataset> --output runs/full                     # 8,000 steps, global batch 32
    python train.py <prepared dataset> --output runs/full --resume runs/full/checkpoints/resume/last.ckpt
"""

import argparse
import json
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
CAMERAS = ("observation.images.wrist", "observation.images.top")

# Studio MolmoAct2 init_args. NOTE: with no lora_target_modules, Studio freezes every base weight and puts
# rank-64/alpha-16 adapters on the LLM, ViT, connector AND action expert (mixins/peft + model.get_default_peft_targets
# at the pinned commit). Ai2's LoRA recipe differs: adapters on LLM + ViT only, connector and action expert trained
# in full. Rank/alpha/dropout (64/16/0.05) and the 5e-5 LRs do match Ai2.
RECIPE = {
    "pretrained_name_or_path": "allenai/MolmoAct2-SO100_101",
    "setup_type": "single so100/so101 robotic arm in molmoact2",  # exact strings of the checkpoint's norm tag
    "control_mode": "absolute joint pose",
    "chunk_size": 30, "n_action_steps": 30, "n_obs_steps": 1,
    "adapt_to_so101": True,          # train in the checkpoint's legacy SO-101 joint frame (Studio docs)
    # No `calibration`: our joints are already LeRobot degrees (gripper 0-100), the checkpoint's units, so the
    # scale must be 1. Studio's calibration scale assumes the +-100 range mode and would be wrong here; it would
    # also store a VM-only file path in the checkpoint hyperparameters.
    "preserve_pretrained_normalization_in_training": False,  # use our train-only stats
    "lora_enabled": True, "lora_rank": 64,
    "optimizer_lr": 5e-5, "optimizer_vit_lr": 5e-5, "optimizer_connector_lr": 5e-5,
    "optimizer_action_expert_lr": 5e-5,  # Ai2 LoRA recipe: 5e-5 for every group
    "optimizer_betas": [0.9, 0.95], "optimizer_eps": 1e-6, "optimizer_weight_decay": 0.0,
    "optimizer_grad_clip_norm": 1.0,
    "scheduler_warmup_steps": 200, "scheduler_decay_steps": None, "scheduler_decay_lr": 1e-6,
    "gradient_checkpointing": True, "use_random_input_noise": True, "compile_model": False,
}
GLOBAL_BATCH, STEPS = 32, 8000
AUGMENT = {"class_path": "augment.Ai2Augment"}


def versions():
    return dict(line.split("=", 1) for line in (HERE / "versions.env").read_text().splitlines()
                if line.strip() and not line.startswith("#"))


def check_dataset(root):
    """The snapshot must be what prepare_data.py writes."""
    info = json.loads((root / "meta/info.json").read_text())
    cams = sorted(k for k in info["features"] if k.startswith("observation.images."))
    if cams != sorted(CAMERAS):
        raise ValueError(f"expected cameras {sorted(CAMERAS)} (run prepare_data.py), found {cams}")
    if info["features"]["observation.images.top"]["shape"][:2] != [480, 560]:
        raise ValueError("top must be cropped to 560x480")
    for name in ("TRAIN-NORMALIZATION.json", "meta/dlsj/labels.jsonl", "calibration/follower.json",
                 "MANIFEST.json"):
        if not (root / name).exists():
            raise ValueError(f"missing {name}")
    manifest = json.loads((root / "MANIFEST.json").read_text())
    return info, manifest


def build_config(*, dataset, output, gpus, steps=STEPS, global_batch=GLOBAL_BATCH, smoke=False,
                 snapshot_every=2000, resume_every=500, val_every=500, deadline_seconds=None):
    if global_batch % gpus:
        raise ValueError(f"global batch {global_batch} is not divisible by {gpus} GPUs")
    dataset, output = Path(dataset), Path(output)
    model = dict(RECIPE)
    callbacks = [
        {"class_path": "lightning.pytorch.callbacks.ModelCheckpoint", "init_args": {  # resume (full state)
            "dirpath": str(output / "checkpoints/resume"), "every_n_train_steps": resume_every,
            "save_top_k": 1, "save_last": True, "filename": "step{step:06d}", "auto_insert_metric_name": False}},
        {"class_path": "lightning.pytorch.callbacks.ModelCheckpoint", "init_args": {  # kept for robot tests
            "dirpath": str(output / "checkpoints/snapshots"), "every_n_train_steps": snapshot_every,
            "save_top_k": -1, "save_weights_only": True, "filename": "step{step:06d}",
            "auto_insert_metric_name": False}},
        {"class_path": "lightning.pytorch.callbacks.ModelCheckpoint", "init_args": {
            "dirpath": str(output / "checkpoints/best"), "monitor": "val/loss", "mode": "min", "save_top_k": 1,
            "save_weights_only": True, "filename": "best-step{step:06d}", "auto_insert_metric_name": False}},
        {"class_path": "lightning.pytorch.callbacks.LearningRateMonitor", "init_args": {"logging_interval": "step"}},
        {"class_path": "progress.Progress"},
    ]
    if deadline_seconds:
        s = max(60, int(deadline_seconds))
        callbacks.append({"class_path": "lightning.pytorch.callbacks.Timer", "init_args": {
            "duration": {"hours": s // 3600, "minutes": s % 3600 // 60, "seconds": s % 60}, "interval": "step"}})
    config = {
        "model": {"class_path": "physicalai.policies.MolmoAct2", "init_args": model},
        "data": {"class_path": "phase_data.PhaseDataModule", "init_args": {
            "repo_id": "local/molmoact2", "root": str(dataset), "data_format": "physicalai",
            "train_batch_size": global_batch // gpus, "val_batch_size": 8, "num_workers": 6,
            "val_split": 0.1, "val_split_seed": 42, "video_backend": "pyav", "image_transforms": AUGMENT}},
        "trainer": {
            "accelerator": "gpu", "devices": gpus, "num_nodes": 1,
            "strategy": "ddp_find_unused_parameters_true" if gpus > 1 else "auto",
            "precision": "bf16-mixed", "max_steps": steps, "max_epochs": -1, "accumulate_grad_batches": 1,
            "log_every_n_steps": 10, "val_check_interval": val_every, "check_val_every_n_epoch": None,
            "limit_val_batches": 32, "num_sanity_val_steps": 1, "default_root_dir": str(output),
            "callbacks": callbacks},
    }
    if smoke:
        config["trainer"].update(fast_dev_run=1, callbacks=[], enable_checkpointing=False, logger=False,
                                 val_check_interval=1.0, check_val_every_n_epoch=1)
    return config


def plan(info, manifest, gpus, steps, global_batch):
    train_frames = info["total_frames"] * len(manifest["studio_split"]["train"]) // info["total_episodes"]
    return {"gpus": gpus, "per_gpu_batch": global_batch // gpus, "global_batch": global_batch, "steps": steps,
            "approx_epochs": round(steps * global_batch / train_frames, 1),
            "train_episodes": len(manifest["studio_split"]["train"]),
            "validation_episodes": manifest["studio_split"]["validation"], "versions": versions()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpus", type=int, choices=(1, 2, 4, 8), default=4)
    p.add_argument("--steps", type=int, default=STEPS)
    p.add_argument("--global-batch", type=int, default=GLOBAL_BATCH)
    p.add_argument("--snapshot-every", type=int, default=2000)
    p.add_argument("--deadline-seconds", type=int, help="stop training cleanly after this many seconds")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", type=Path, help="Lightning .ckpt to resume (same dataset/GPUs/batch)")
    p.add_argument("--plan-only", action="store_true")
    args = p.parse_args()
    dataset, output = args.dataset.resolve(), args.output.resolve()
    try:
        info, manifest = check_dataset(dataset)
        summary = plan(info, manifest, args.gpus, args.steps, args.global_batch)
        config = build_config(dataset=dataset, output=output, gpus=args.gpus, steps=args.steps,
                              global_batch=args.global_batch, smoke=args.smoke,
                              snapshot_every=args.snapshot_every, deadline_seconds=args.deadline_seconds)
    except (ValueError, OSError, KeyError) as exc:
        p.error(str(exc))
    print(json.dumps(summary, indent=2), flush=True)
    if args.plan_only:
        return
    studio = Path(os.environ.get("STUDIO_DIR", HERE / ".studio")).resolve()
    actual = subprocess.check_output(["git", "-C", str(studio), "rev-parse", "HEAD"], text=True).strip()
    if actual != versions()["STUDIO_COMMIT"]:
        p.error(f"Studio revision mismatch: expected {versions()['STUDIO_COMMIT']}, got {actual}")
    if output.exists() and not (args.resume or args.smoke):
        p.error("output exists; choose a new --output or pass --resume")
    output.mkdir(parents=True, exist_ok=True)
    (output / "training-config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "plan.json").write_text(json.dumps(summary, indent=2) + "\n")
    cmd = [str(studio / "library/.venv/bin/physicalai"), "fit", "--config", str(output / "training-config.json")]
    if args.resume:
        cmd += ["--fit.ckpt_path", str(args.resume.resolve())]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(HERE), os.environ.get("PYTHONPATH")])))
    env.setdefault("NCCL_P2P_LEVEL", "SYS" if args.gpus == 8 else "PHB")  # Google's G4 topology settings
    raise SystemExit(subprocess.run(cmd, cwd=HERE, env=env).returncode)


if __name__ == "__main__":
    main()
