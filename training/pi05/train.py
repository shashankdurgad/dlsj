"""Validate a frozen local teleop dataset, estimate the run, then invoke Studio."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess

HERE = Path(__file__).resolve().parent
JOINTS = [f"{name}.pos" for name in (
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"
)]


def inspect_dataset(root):
    """Reject incomplete/mixed-success recordings without changing the dataset."""
    info_bytes = (root / "meta/info.json").read_bytes()
    labels_bytes = (root / "meta/dlsj/labels.jsonl").read_bytes()
    info = json.loads(info_bytes)
    if not str(info.get("codebase_version", "")).startswith("v3."):
        raise ValueError("Expected a LeRobot v3 dataset from the teleop recorder")
    if info.get("fps") != 30:
        raise ValueError("Expected recordings at 30 fps")
    features = info["features"]
    cameras = {key for key in features if key.startswith("observation.images.")}
    if cameras != {f"observation.images.{name}" for name in ("wrist", "top", "inspect")}:
        raise ValueError(f"Expected wrist, top, inspect camera features; found {sorted(cameras)}")
    for name in ("observation.state", "action"):
        feature = features[name]
        if feature["shape"] != [6] or feature.get("names") != JOINTS:
            raise ValueError(f"{name}: expected six joints in recorder order {JOINTS}")
    labels = [json.loads(line) for line in labels_bytes.splitlines() if line.strip()]
    ids = [label["episode_index"] for label in labels]
    count = info["total_episodes"]
    if count < 2 or sorted(ids) != list(range(count)):
        raise ValueError("Need at least two finalized episodes, each with exactly one label")
    if any(label.get("success") is not True for label in labels):
        raise ValueError("Dataset includes failure/unlabelled episodes. Prepare a success-only copy first")
    if any(not isinstance(label.get("task"), str) or not label["task"].strip() for label in labels):
        raise ValueError("Every demonstration needs a nonempty task instruction")
    if any(not isinstance(label.get("n_frames"), int) or label["n_frames"] <= 0 for label in labels):
        raise ValueError("Invalid episode frame counts")
    if sum(label["n_frames"] for label in labels) != info["total_frames"]:
        raise ValueError("Label frame counts disagree with info.json; finalize and copy the dataset first")
    return info, sorted(labels, key=lambda x: x["episode_index"]), hashlib.sha256(info_bytes + labels_bytes).hexdigest()


def estimate(labels, gpus, batch, accumulation, epochs):
    val_ids = set(random.Random(42).sample(range(len(labels)), max(1, int(len(labels) * 0.1))))
    frames = sum(label["n_frames"] for label in labels if label["episode_index"] not in val_ids)
    batches = math.ceil(frames / gpus) // batch
    if batches == 0:
        raise ValueError("Too few training frames for this GPU count and per-GPU batch size")
    steps = math.ceil(batches / accumulation)
    return {"gpus": gpus, "per_gpu_batch": batch,
            "effective_batch": gpus * batch * accumulation,
            "train_frames": frames, "validation_episode_ids": sorted(val_ids),
            "optimizer_steps_per_epoch": steps, "optimizer_steps": steps * epochs}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", type=Path, help="Finalized success-only local dataset snapshot")
    p.add_argument("--repo-id", default="local/so101_sft")
    p.add_argument("--pretrained-model", default="lerobot/pi05_base",
                   help="Hugging Face repo ID or local model directory restored from GCS")
    p.add_argument("--gpus", type=int, choices=(1, 2, 4, 8), default=4)
    p.add_argument("--batch", type=int, default=8, help="Samples per GPU")
    p.add_argument("--accumulate", type=int, default=1)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--output", type=Path, default=HERE / "runs/pi05")
    p.add_argument("--plan-only", action="store_true", help="Validate metadata and estimate; no GPU job or writes")
    p.add_argument("--smoke", action="store_true", help="One training and validation batch; no persistent checkpoints")
    p.add_argument("--resume", type=Path, help="Resume a Lightning .ckpt, including optimizer state")
    args = p.parse_args()
    if min(args.batch, args.accumulate, args.epochs) < 1:
        p.error("batch, accumulate and epochs must be positive")
    root = args.dataset.resolve()
    try:
        info, labels, digest = inspect_dataset(root)
        plan = estimate(labels, args.gpus, args.batch, args.accumulate, args.epochs)
    except (ValueError, OSError, KeyError) as exc:
        p.error(str(exc))
    plan.update({"dataset_metadata_sha256": digest, "total_frames": info["total_frames"],
                 "pretrained_model": args.pretrained_model,
                 "instructions": sorted({label["task"] for label in labels})})
    print(json.dumps(plan, indent=2), flush=True)
    if args.plan_only:
        return
    studio = Path(os.environ.get("STUDIO_DIR", HERE / ".studio")).resolve()
    expected = (HERE / "versions.env").read_text().split("STUDIO_COMMIT=", 1)[1].strip()
    actual = subprocess.check_output(["git", "-C", str(studio), "rev-parse", "HEAD"], text=True).strip()
    if actual != expected:
        p.error(f"Studio revision mismatch: expected {expected}, got {actual}")
    output = args.output.resolve()
    if output.exists() and not args.resume and not args.smoke:
        p.error("Output directory already exists; choose a new --output or explicitly --resume")
    config = json.loads((HERE / "full.json").read_text())
    config["model"]["init_args"]["pretrained_name_or_path"] = args.pretrained_model
    config["data"]["init_args"].update(root=str(root), repo_id=args.repo_id, train_batch_size=args.batch)
    config["trainer"].update(devices=args.gpus, max_epochs=args.epochs,
                              accumulate_grad_batches=args.accumulate, default_root_dir=str(output))
    for callback, directory in zip(config["trainer"]["callbacks"], ("best", "resume")):
        callback["init_args"]["dirpath"] = str(output / "checkpoints" / directory)
    # Keep LR warmup near 5% of this dataset-dependent budget.
    config["model"]["init_args"]["scheduler_warmup_steps"] = max(1, round(plan["optimizer_steps"] * 0.05))
    if args.smoke:
        config["trainer"].update(fast_dev_run=1, enable_checkpointing=False,
                                  callbacks=[], logger=False)
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "training-config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    (output / "data-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    command = [str(studio / "library/.venv/bin/physicalai"), "fit", "--config", str(config_path)]
    if args.resume:
        command.extend(["--fit.ckpt_path", str(args.resume.resolve())])
    env = os.environ.copy()
    # Google's G4 P2P settings differ across 4- and 8-GPU VM topology.
    env.setdefault("NCCL_P2P_LEVEL", "SYS" if args.gpus == 8 else "PHB")
    subprocess.run(command, cwd=HERE, env=env, check=True)


if __name__ == "__main__":
    main()
