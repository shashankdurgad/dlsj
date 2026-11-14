"""Launch a time-bounded Pi0.5 adaptation on a frozen RGB dataset.

Default: action expert only (VLM frozen). --full: fine-tune every weight, as in openpi.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

from train import HERE, inspect_dataset, estimate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--pretrained-model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true",
        help="Unfreeze the vision-language backbone (full fine-tune)")
    parser.add_argument("--max-steps", type=int,
        help="Step cap and LR decay length (default 3000 expert-only, 2200 full)")
    args = parser.parse_args()
    info, labels, digest = inspect_dataset(args.dataset)
    studio = Path(os.environ.get("STUDIO_DIR", HERE / ".studio")).resolve()
    expected = (HERE / "versions.env").read_text().split("STUDIO_COMMIT=", 1)[1].strip()
    actual = subprocess.check_output(["git", "-C", str(studio), "rev-parse", "HEAD"], text=True).strip()
    if actual != expected:
        raise RuntimeError(f"Studio revision mismatch: {actual}")
    if args.output.exists():
        raise RuntimeError("Choose a new output directory")
    args.output.mkdir(parents=True)
    config = json.loads((HERE / "full.json").read_text())
    # Full runs are ~1.2-1.6 s/step on 4 GPUs (vs 0.82 expert-only), so ~2,200 steps fit in
    # 50 min; decaying the LR over that many steps lets it actually reach its floor.
    max_steps = args.max_steps or (2200 if args.full else 3000)
    # Full checkpoints carry optimizer state for all ~4B params (tens of GB): save rarely.
    ckpt_every = 500 if args.full else 100
    config["model"]["init_args"].update(
        pretrained_name_or_path=args.pretrained_model,
        train_expert_only=not args.full, freeze_vision_encoder=not args.full,
        scheduler_warmup_steps=100, scheduler_decay_steps=max_steps)
    config["data"]["init_args"].update(root=str(args.dataset.resolve()),
        train_batch_size=8, val_batch_size=8, num_workers=4)
    config["trainer"].update(devices=4, max_epochs=100, max_steps=max_steps,
        strategy="ddp_find_unused_parameters_true",
        num_sanity_val_steps=1, limit_val_batches=16,
        check_val_every_n_epoch=None, val_check_interval=250,
        default_root_dir=str(args.output.resolve()), enable_progress_bar=False)
    for cb, name in zip(config["trainer"]["callbacks"], ("best", "resume")):
        cb["init_args"]["dirpath"] = str(args.output.resolve() / "checkpoints" / name)
    config["trainer"]["callbacks"][1]["init_args"]["every_n_train_steps"] = ckpt_every
    if args.full:
        # The best-val checkpoint may be rewritten every validation; keep it weights-only.
        config["trainer"]["callbacks"][0]["init_args"]["save_weights_only"] = True
    config["trainer"]["callbacks"].extend([
        {"class_path": "lightning.pytorch.callbacks.Timer", "init_args": {
            "duration": "00:00:50:00", "interval": "step"}},
        {"class_path": "hour_callback.HourProgress"}])
    if args.smoke:
        config["trainer"].update(fast_dev_run=1, callbacks=[],
            enable_checkpointing=False, logger=False,
            check_val_every_n_epoch=1, val_check_interval=1.0)
    config_path = args.output / "training-config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    plan = estimate(labels, 4, 8, 1, 1)
    plan.update(dataset_metadata_sha256=digest, total_frames=info["total_frames"],
        mode="pi05-full-finetune" if args.full else "pi05-action-expert-only",
        training_time_limit_minutes=50, max_optimizer_steps=max_steps,
        checkpoint_every_n_steps=ckpt_every, studio_commit=actual)
    (args.output / "data-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    env = os.environ.copy()
    env.update(NCCL_P2P_LEVEL="PHB", OMP_NUM_THREADS="8",
        PYTHONPATH=str(HERE) + os.pathsep + env.get("PYTHONPATH", ""))
    subprocess.run([str(studio / "library/.venv/bin/physicalai"), "fit",
                    "--config", str(config_path.resolve())], env=env, check=True)


if __name__ == "__main__":
    main()
