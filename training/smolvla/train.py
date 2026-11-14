"""Fine-tune SmolVLA with the official LeRobot recipe on the 2-camera (wrist + top) Duplo data.

Official recipe (https://huggingface.co/docs/lerobot/smolvla): `lerobot-train --policy.path=lerobot/smolvla_base
--batch_size=64 --steps=20000` with a rename map onto the base model's camera names, on one GPU. Everything else
is the checkpoint's own config, which this launcher checks rather than overrides (RECIPE below).

    python train.py <prepared>/train --output runs/smolvla --smoke     # 4 steps, then exits
    python train.py <prepared>/train --output runs/timing --steps 200 --save-freq 200
    python train.py <prepared>/train --output runs/smolvla            # the full 20k-step run
    python train.py <prepared>/train --output runs/smolvla --resume   # after a Spot preemption

Checkpoints: <output>/checkpoints/<step>/pretrained_model (+ training_state), and checkpoints/last.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
JOINTS = [f"{n}.pos" for n in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")]
# Same mapping as LeRobot's SO-101 example: top -> camera1, wrist -> camera2 (camera3 is left out; SmolVLA
# skips image features missing from the batch).
RENAME_MAP = {"observation.images.top": "observation.images.camera1",
              "observation.images.wrist": "observation.images.camera2"}
BATCH, STEPS, SAVE_FREQ = 64, 20_000, 2_000
# smolvla_base config.json values the official recipe relies on. Checked, never overridden.
RECIPE = {
    "chunk_size": 50, "n_action_steps": 50, "num_steps": 10,
    "freeze_vision_encoder": True, "train_expert_only": True, "train_state_proj": True, "load_vlm_weights": True,
    "optimizer_lr": 1e-4, "optimizer_betas": [0.9, 0.95], "optimizer_eps": 1e-8,
    "optimizer_weight_decay": 1e-10, "optimizer_grad_clip_norm": 10,
    "scheduler_warmup_steps": 1000, "scheduler_decay_steps": 30000, "scheduler_decay_lr": 2.5e-6,
    "resize_imgs_with_padding": [512, 512],
    "normalization_mapping": {"VISUAL": "IDENTITY", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"},
    "vlm_model_name": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
}


def versions():
    return dict(line.split("=", 1) for line in (HERE / "versions.env").read_text().splitlines()
                if line.strip() and not line.startswith("#"))


def check_dataset(root):
    """The prepared split must be what prepare_data.py writes: wrist + top, six joints, 30 fps, q-free stats ok."""
    info = json.loads((root / "meta/info.json").read_text())
    feats = info["features"]
    cams = sorted(k for k in feats if k.startswith("observation.images."))
    if cams != sorted(RENAME_MAP):
        raise ValueError(f"expected cameras {sorted(RENAME_MAP)} (run prepare_data.py first), found {cams}")
    if info.get("fps") != 30:
        raise ValueError("expected 30 fps")
    for key in ("observation.state", "action"):
        if feats[key]["shape"] != [6] or feats[key].get("names") != JOINTS:
            raise ValueError(f"{key}: expected six joints in recorder order")
    stats = json.loads((root / "meta/stats.json").read_text())
    for key in ("observation.state", "action"):
        if not {"mean", "std"} <= set(stats.get(key, {})):
            raise ValueError(f"meta/stats.json lacks mean/std for {key}")
    return info


def check_base_config(config):
    """Fail loudly if the pinned smolvla_base no longer carries the official recipe values."""
    wrong = {k: (config.get(k), v) for k, v in RECIPE.items() if config.get(k) != v}
    if wrong:
        raise ValueError(f"smolvla_base config differs from the official recipe: {wrong}")


def build_command(*, dataset, output, steps=STEPS, batch=BATCH, save_freq=SAVE_FREQ, workers=8,
                  log_freq=50, video_backend=None, resume=False, python=sys.executable):
    module = [python, "-m", "lerobot.scripts.lerobot_train"]
    if resume:
        return module + [f"--config_path={output}/checkpoints/last/pretrained_model/train_config.json",
                         "--resume=true"]
    cmd = module + [
        f"--policy.path={versions()['SMOLVLA_REPO']}",
        f"--dataset.repo_id=local/{Path(dataset).name}", f"--dataset.root={dataset}",
        f"--rename_map={json.dumps(RENAME_MAP)}",
        f"--batch_size={batch}", f"--steps={steps}",
        f"--output_dir={output}", f"--job_name={Path(output).name}",
        "--policy.device=cuda", "--policy.push_to_hub=false",
        f"--save_freq={save_freq}", f"--log_freq={log_freq}", f"--num_workers={workers}", "--seed=1000",
    ]
    if video_backend:
        cmd.append(f"--dataset.video_backend={video_backend}")
    return cmd


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", type=Path, help="prepared train split (prepare_data.py <out>/train)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=STEPS)
    p.add_argument("--batch", type=int, default=BATCH)
    p.add_argument("--save-freq", type=int, default=SAVE_FREQ)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--video-backend", choices=("torchcodec", "pyav"), help="default: LeRobot's choice")
    p.add_argument("--device", default="cuda", help="cuda on the VM; cpu/mps only for a local smoke check")
    p.add_argument("--smoke", action="store_true", help="4 steps at batch 4, checkpoint at step 4")
    p.add_argument("--resume", action="store_true", help="resume --output from checkpoints/last")
    p.add_argument("--plan-only", action="store_true", help="print the command and exit")
    args = p.parse_args()
    dataset, output = args.dataset.resolve(), args.output.resolve()
    if not args.resume:
        info = check_dataset(dataset)
        if output.exists():
            p.error(f"{output} exists; choose a new --output or pass --resume")
    elif not (output / "checkpoints/last/pretrained_model/train_config.json").exists():
        p.error(f"nothing to resume in {output}")
    steps, batch, save = (4, 4, 4) if args.smoke else (args.steps, args.batch, args.save_freq)
    cmd = build_command(dataset=dataset, output=output, steps=steps, batch=batch, save_freq=save,
                        workers=2 if args.smoke else args.workers, log_freq=1 if args.smoke else 50,
                        video_backend=args.video_backend, resume=args.resume)
    if args.device != "cuda" and not args.resume:
        cmd = [c.replace("--policy.device=cuda", f"--policy.device={args.device}") for c in cmd]
    plan = {"command": cmd, "steps": steps, "batch": batch, "save_freq": save, "versions": versions()}
    if not args.resume:
        plan.update(frames=info["total_frames"], episodes=info["total_episodes"],
                    epochs=round(steps * batch / info["total_frames"], 2))
    print(json.dumps(plan, indent=2), flush=True)
    if args.plan_only:
        return
    if not args.resume:
        check_base_config(json.loads(pinned_main("SMOLVLA").read_text()))
        pinned_main("SMOLVLM")
    sys.exit(subprocess.run(cmd).returncode)


def pinned_main(name):
    """LeRobot/transformers load the repos' `main`; make sure `main` resolves to the pinned revision.

    On the VM the hub cache restored from GCS is used with HF_HUB_OFFLINE=1, so `main` is the pinned
    snapshot. Online, a newer upstream commit makes this fail instead of silently training on it.
    """
    from huggingface_hub import hf_hub_download

    v = versions()
    path = Path(hf_hub_download(v[f"{name}_REPO"], "config.json"))
    if path.parent.name != v[f"{name}_REVISION"]:
        raise RuntimeError(f"{v[f'{name}_REPO']} main is {path.parent.name}, pinned {v[f'{name}_REVISION']} "
                           "(restore the GCS hub cache, or update versions.env deliberately)")
    return path


if __name__ == "__main__":
    main()
