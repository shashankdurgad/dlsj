"""Fine-tune SmolVLA (the team's fallback VLA) with Physical AI Studio on ONE Intel XPU.

Starts from Studio's `configs/physicalai/smolvla/pusht/default.yaml` with the PLAYBOOK 4.2 changes for SO-101:
accelerator xpu / strategy xpu_single, 3 cameras instead of `num_cameras: 0`, no PushT `val_gym` (episode
validation split instead), the pretrained `lerobot/smolvla_base`, bf16 autocast on XPU (see common.xpu_trainer)
and gradient accumulation to a target global batch. Dataset validation is shared with train_pi05_xpu.py.

    PY=~/inference/physical-ai-studio/library/.venv/bin/python
    nice -n 19 $PY training/xpu/train_smolvla_xpu.py <dataset> --no-labels --plan-only
    nice -n 19 $PY training/xpu/train_smolvla_xpu.py <finalized dataset copy> --no-labels --smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

SMOLVLA_TOTAL = 0.45e9      # SmolVLA paper: ~450 M parameters
SMOLVLA_TRAINABLE = 0.10e9  # action expert + projections (train_expert_only); approximate
SMOLVLA_ACT_GIB_PER_SAMPLE = 0.3  # guess

BASE_TRAINER = {
    "log_every_n_steps": 10,
    "check_val_every_n_epoch": 1,
    "callbacks": [
        {"class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
         "init_args": {"monitor": "val/loss", "mode": "min", "save_top_k": 1, "save_last": False,
                       "filename": "epoch{epoch:03d}", "auto_insert_metric_name": False}},
        {"class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
         "init_args": {"save_top_k": 1, "save_last": True, "every_n_train_steps": 250,
                       "save_on_train_epoch_end": False, "filename": "step{step:08d}",
                       "auto_insert_metric_name": False}},
    ],
}

COLOR_JITTER = {"class_path": "torchvision.transforms.v2.Compose", "init_args": {"transforms": [
    {"class_path": "torchvision.transforms.v2.ColorJitter",
     "init_args": {"contrast": [0.8, 1.2], "brightness": [0.8, 1.2]}}]}}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", type=Path)
    p.add_argument("--no-labels", action="store_true", help="No meta/dlsj/labels.jsonl: use LeRobot meta, all episodes")
    p.add_argument("--repo-id", default="local/so101_sft")
    p.add_argument("--pretrained-model", default="lerobot/smolvla_base")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--global-batch", type=int, default=64)
    p.add_argument("--accumulate", type=int)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--time-limit-min", type=float)
    p.add_argument("--precision", choices=("bf16-mixed", "32"), default="bf16-mixed")
    p.add_argument("--train-vlm", action="store_true", help="Also train the VLM (default: expert only, frozen vision)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--video-backend", default="pyav",
                   help="LeRobot video decoder. pyav: the Studio venv's torchcodec cannot load on the XPS (no system "
                        "FFmpeg libs: 'libavutil.so.60: cannot open shared object file')")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--output", type=Path, default=common.HERE / "runs/smolvla_xpu")
    p.add_argument("--studio-dir", type=Path, default=common.DEFAULT_STUDIO)
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", type=Path)
    p.add_argument("--nice", type=int, default=19)
    p.add_argument("--min-free-gib", type=float, default=8.0)
    p.add_argument("--force", action="store_true")
    return p


def build_config(args, plan: dict, accumulate: int, output: Path, root: Path) -> dict:
    steps = common.optimizer_steps(plan, args.max_steps)
    model = {"class_path": "physicalai.policies.SmolVLA", "init_args": {
        "pretrained_name_or_path": args.pretrained_model, "num_cameras": len(common.CAMERAS),
        "image_key_reorder_map": {}, "compile_model": False,
        "freeze_vision_encoder": not args.train_vlm, "train_expert_only": not args.train_vlm,
        "optimizer_lr": args.lr, "scheduler_warmup_steps": max(1, round(steps * 0.05))}}
    data = {"class_path": "physicalai.data.lerobot.LeRobotDataModule", "init_args": {
        "repo_id": args.repo_id, "root": str(root), "data_format": "physicalai",
        "train_batch_size": args.batch, "val_batch_size": args.batch, "num_workers": args.num_workers,
        "video_backend": args.video_backend,
        "val_split": 0.1, "val_split_seed": 42, "val_gym": None, "image_transforms": COLOR_JITTER}}
    trainer = common.xpu_trainer(BASE_TRAINER, precision=args.precision, accumulate=accumulate, output=output,
                                 epochs=args.epochs, max_steps=args.max_steps, time_limit_min=args.time_limit_min,
                                 smoke=args.smoke)
    return {"model": model, "data": data, "trainer": trainer}


def main(argv=None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    if min(args.batch, args.global_batch, args.epochs) < 1:
        p.error("batch, global-batch and epochs must be positive")
    root = args.dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    try:
        accumulate, effective = common.global_batch_plan(args.batch, args.global_batch, args.accumulate)
        plan = common.dataset_plan(root, require_labels=not args.no_labels, batch=args.batch,
                                   accumulate=accumulate, epochs=args.epochs)
    except (ValueError, OSError, KeyError) as exc:
        p.error(str(exc))
    plan.update(device="xpu:0 (single)", precision=args.precision, pretrained_model=args.pretrained_model,
                accumulate_grad_batches=accumulate, effective_batch=effective,
                planned_optimizer_steps=common.optimizer_steps(plan, args.max_steps))
    trainable = SMOLVLA_TOTAL if args.train_vlm else SMOLVLA_TRAINABLE
    mem = common.memory_estimate_gib(total_params=SMOLVLA_TOTAL, trainable_params=trainable,
                                     weight_bytes=4 if args.precision == "32" else 2, batch=args.batch,
                                     act_gib_per_sample=SMOLVLA_ACT_GIB_PER_SAMPLE)
    mem["note"] = "arithmetic estimate, not measured"
    config = build_config(args, plan, accumulate, output, root)
    print(json.dumps({"plan": plan, "memory_estimate": mem}, indent=2), flush=True)
    if args.plan_only:
        print(json.dumps({"training_config": config}, indent=2))
        return 0
    try:
        cli = common.check_studio(args.studio_dir.expanduser())
    except Exception as exc:  # noqa: BLE001
        p.error(str(exc))
    if output.exists() and not args.resume and not args.smoke:
        p.error("Output directory already exists; choose a new --output or explicitly --resume")
    problems = common.preflight(mem["total_gib"], force=args.force, min_free_gib=args.min_free_gib)
    if problems:
        p.error("not starting (use --force to override):\n- " + "\n- ".join(problems))
    config_path = common.write_run(output, config, plan)
    command = [str(cli), "fit", "--config", str(config_path)]
    if args.resume:
        command += ["--fit.ckpt_path", str(args.resume.expanduser().resolve())]
    return common.launch(command, cwd=args.studio_dir.expanduser() / "library", nice=args.nice,
                         min_free_gib=args.min_free_gib)


if __name__ == "__main__":
    sys.exit(main())
