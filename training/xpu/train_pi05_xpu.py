"""Fine-tune pi0.5 with Physical AI Studio (`physicalai fit`) on ONE Intel XPU (the Arc iGPU of a Core Ultra laptop).

This is an alternative to the cloud recipe in training/pi05/ (which is what produced the team's deployed
checkpoint, on NVIDIA GPUs). It reuses that recipe's dataset validation and step estimate, starts from the same
`full.json`, and rewrites it for a single XPU:

- accelerator xpu, strategy xpu_single, devices 1;
- bf16 autocast on XPU via an explicit MixedPrecision(device="xpu") plugin (`--precision 32` for fp32);
- small per-device batch + gradient accumulation up to `--global-batch` (default 32, as in the cloud run);
- gradient checkpointing on; action-expert-only with a frozen vision encoder by default (like the cloud run);
  `--lora` instead trains LoRA adapters (Studio forbids combining LoRA with the freeze flags).

Run with the Studio library venv's python (it has pyarrow and the pinned Studio):

    PY=~/inference/physical-ai-studio/library/.venv/bin/python
    nice -n 19 $PY training/xpu/train_pi05_xpu.py ~/desmond/dlsj/data/duplo_teleop_rgb --no-labels --plan-only
    nice -n 19 $PY training/xpu/train_pi05_xpu.py <finalized dataset copy> --no-labels --smoke
    nice -n 19 $PY training/xpu/train_pi05_xpu.py <finalized dataset copy> --no-labels --max-steps 2000 --time-limit-min 600

See training/xpu/README.md for what has and has not been run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

# Counts from the team's checkpoint card (w3joe/pi05-junction-so101-20261114): 4.146 B total parameters,
# ~695.5 M trainable with the VLM frozen (action expert + projections).
PI05_TOTAL = 4.146e9
PI05_EXPERT_TRAINABLE = 0.6955e9
PI05_LORA_TRAINABLE = 0.15e9  # assumption for rank 32 on attention+MLP of both experts; not measured
PI05_ACT_GIB_PER_SAMPLE = {"expert": 1.0, "lora": 2.5, "full": 2.5}  # guesses with gradient checkpointing


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", type=Path, help="Finalized local LeRobot v3 dataset (wrist/top/inspect, 6 joints, 30 fps)")
    p.add_argument("--no-labels", action="store_true",
                   help="Dataset has no meta/dlsj/labels.jsonl: take episodes/frames from LeRobot meta and train on "
                        "ALL episodes (success is not checked)")
    p.add_argument("--repo-id", default="local/so101_sft")
    p.add_argument("--pretrained-model", default="lerobot/pi05_base", help="HF repo ID or local directory")
    p.add_argument("--batch", type=int, default=2, help="Samples per optimizer micro-step on the XPU")
    p.add_argument("--global-batch", type=int, default=32, help="Target effective batch (reached by accumulation)")
    p.add_argument("--accumulate", type=int, help="Override the accumulation computed from --global-batch")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--max-steps", type=int, help="Stop after this many optimizer steps")
    p.add_argument("--time-limit-min", type=float, help="Lightning max_time, in minutes")
    p.add_argument("--precision", choices=("bf16-mixed", "32"), default="bf16-mixed")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--lora", action="store_true", help="LoRA adapters instead of action-expert-only")
    mode.add_argument("--full-finetune", action="store_true",
                      help="Train everything (does NOT fit in ~57 GiB of shared memory per the estimate)")
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lr", type=float, help="Override optimizer_lr (default from full.json: 2.5e-5)")
    p.add_argument("--video-backend", default="pyav",
                   help="LeRobot video decoder. pyav: the Studio venv's torchcodec cannot load on the XPS (no system "
                        "FFmpeg libs: 'libavutil.so.60: cannot open shared object file')")
    p.add_argument("--num-workers", type=int, default=2, help="Data loader workers (video decoding is CPU-heavy)")
    p.add_argument("--output", type=Path, default=common.HERE / "runs/pi05_xpu")
    p.add_argument("--studio-dir", type=Path, default=common.DEFAULT_STUDIO)
    p.add_argument("--plan-only", action="store_true", help="Validate + estimate + print the config; no writes, no XPU")
    p.add_argument("--smoke", action="store_true", help="fast_dev_run: one train + one val batch, no checkpoints")
    p.add_argument("--resume", type=Path, help="Resume a Lightning .ckpt (optimizer state included)")
    p.add_argument("--nice", type=int, default=19)
    p.add_argument("--min-free-gib", type=float, default=8.0,
                   help="Kill the run if MemAvailable drops below this (the iGPU uses system RAM)")
    p.add_argument("--force", action="store_true", help="Start even if other jobs run or memory looks short")
    return p


def mode_of(args) -> str:
    return "lora" if args.lora else "full" if args.full_finetune else "expert"


def build_config(args, plan: dict, accumulate: int, output: Path, root: Path) -> dict:
    config = json.loads((common.PI05_DIR / "full.json").read_text())
    model = config["model"]["init_args"]
    model.update(pretrained_name_or_path=args.pretrained_model, gradient_checkpointing=True, compile_model=False,
                 train_expert_only=False, freeze_vision_encoder=False, lora_enabled=False)
    mode = mode_of(args)
    if mode == "expert":
        model.update(train_expert_only=True, freeze_vision_encoder=True)
    elif mode == "lora":
        model.update(lora_enabled=True, lora_rank=args.lora_rank, lora_alpha=args.lora_rank)
    if args.lr:
        model["optimizer_lr"] = args.lr
    steps = common.optimizer_steps(plan, args.max_steps)
    model["scheduler_warmup_steps"] = max(1, round(steps * 0.05))
    config["data"]["init_args"].update(root=str(root), repo_id=args.repo_id, train_batch_size=args.batch,
                                       val_batch_size=args.batch, num_workers=args.num_workers,
                                       video_backend=args.video_backend)
    config["trainer"] = common.xpu_trainer(
        config["trainer"], precision=args.precision, accumulate=accumulate, output=output, epochs=args.epochs,
        max_steps=args.max_steps, time_limit_min=args.time_limit_min, smoke=args.smoke)
    return config


def estimate(args) -> dict:
    mode = mode_of(args)
    trainable = {"expert": PI05_EXPERT_TRAINABLE, "lora": PI05_LORA_TRAINABLE, "full": PI05_TOTAL}[mode]
    weight_bytes = 4 if args.precision == "32" else 2  # Pi05 dtype bfloat16 keeps most weights in bf16
    est = common.memory_estimate_gib(total_params=PI05_TOTAL, trainable_params=trainable, weight_bytes=weight_bytes,
                                     batch=args.batch, act_gib_per_sample=PI05_ACT_GIB_PER_SAMPLE[mode])
    est.update(mode=mode, trainable_params=f"{trainable / 1e6:.0f}M", note="arithmetic estimate, not measured")
    return est


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
    mem = estimate(args)
    config = build_config(args, plan, accumulate, output, root)
    print(json.dumps({"plan": plan, "memory_estimate": mem}, indent=2), flush=True)
    if mem["total_gib"] > 50:
        print(f"WARNING: estimated {mem['total_gib']} GiB exceeds what the Arc B390 can use (~57 GiB shared, "
              "minus the OS and data loader)", file=sys.stderr)
    if args.plan_only:
        print(json.dumps({"training_config": config}, indent=2))
        return 0
    try:
        cli = common.check_studio(args.studio_dir.expanduser())
    except Exception as exc:  # noqa: BLE001 - git missing, wrong revision, no venv
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
