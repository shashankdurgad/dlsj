"""Shared helpers for single-device Intel XPU training through Physical AI Studio (`physicalai fit`).

Used by `train_pi05_xpu.py` and `train_smolvla_xpu.py`. Everything here except `launch()` is
hardware-free, so the config generation is unit-tested on any machine (tests/test_xpu_training.py).

Dataset validation and the step estimate are imported from Tan's cloud recipe (`training/pi05/train.py`)
so both paths agree on what a valid dataset is and how many optimizer steps an epoch has.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PI05_DIR = REPO / "training" / "pi05"
DEFAULT_STUDIO = Path(os.environ.get("STUDIO_DIR", Path.home() / "inference/physical-ai-studio"))
CAMERAS = ("wrist", "top", "inspect")
GIB = 1024**3


def _load_cloud_recipe():
    spec = importlib.util.spec_from_file_location("pi05_cloud_recipe", PI05_DIR / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


recipe = _load_cloud_recipe()


# --------------------------------------------------------------------------------------------- dataset

def check_info(info: dict) -> None:
    """The same metadata checks as recipe.inspect_dataset, minus the labels sidecar."""
    if not str(info.get("codebase_version", "")).startswith("v3."):
        raise ValueError("Expected a LeRobot v3 dataset from the teleop recorder")
    if info.get("fps") != 30:
        raise ValueError("Expected recordings at 30 fps")
    features = info["features"]
    cameras = {key for key in features if key.startswith("observation.images.")}
    if cameras != {f"observation.images.{name}" for name in CAMERAS}:
        raise ValueError(f"Expected wrist, top, inspect camera features; found {sorted(cameras)}")
    for name in ("observation.state", "action"):
        feature = features[name]
        if feature["shape"] != [6] or feature.get("names") != recipe.JOINTS:
            raise ValueError(f"{name}: expected six joints in recorder order {recipe.JOINTS}")


def read_episodes(root: Path) -> list[dict]:
    """Rows of meta/episodes/*/*.parquet: episode_index, length, tasks. Needs pyarrow (in the Studio venv)."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on the env
        raise ValueError("pyarrow is needed to read meta/episodes; run with the Studio venv's python") from exc
    files = sorted((root / "meta/episodes").glob("chunk-*/file-*.parquet"))
    if not files:
        raise ValueError(f"No meta/episodes/chunk-*/file-*.parquet under {root}")
    rows = []
    for f in files:
        rows.extend(pq.read_table(f, columns=["episode_index", "length", "tasks"]).to_pylist())
    return rows


def labels_from_meta(root: Path, info: dict, info_bytes: bytes) -> tuple[list[dict], str]:
    """Build Tan-style label rows from LeRobot meta when there is no meta/dlsj/labels.jsonl.

    Success is unknown (None): every finalized episode is trained on.
    """
    rows = read_episodes(root)
    ids = sorted(int(r["episode_index"]) for r in rows)
    count = info["total_episodes"]
    if count < 2 or ids != list(range(count)):
        raise ValueError(f"meta/episodes has {len(ids)} episodes, info.json says {count}. The dataset is probably "
                         "still being recorded: train on a finalized copy")
    labels = [{"episode_index": int(r["episode_index"]), "n_frames": int(r["length"]),
               "task": (r.get("tasks") or [""])[0], "success": None} for r in rows]
    if any(label["n_frames"] <= 0 for label in labels):
        raise ValueError("Invalid episode frame counts")
    if sum(label["n_frames"] for label in labels) != info["total_frames"]:
        raise ValueError("Episode lengths disagree with info.json total_frames; finalize and copy the dataset first")
    labels.sort(key=lambda x: x["episode_index"])
    summary = json.dumps([[x["episode_index"], x["n_frames"]] for x in labels]).encode()
    return labels, hashlib.sha256(info_bytes + summary).hexdigest()


def global_batch_plan(per_device: int, global_batch: int | None, accumulate: int | None) -> tuple[int, int]:
    """(accumulate_grad_batches, effective batch) on one device. `accumulate` wins over `global_batch`."""
    if per_device < 1:
        raise ValueError("per-device batch must be positive")
    if accumulate is None:
        accumulate = max(1, math.ceil((global_batch or per_device) / per_device))
    if accumulate < 1:
        raise ValueError("accumulate must be positive")
    return accumulate, per_device * accumulate


def dataset_plan(root: Path, *, require_labels: bool, batch: int, accumulate: int, epochs: int) -> dict:
    """Validate the dataset and estimate optimizer steps for ONE device (recipe.estimate with gpus=1)."""
    if require_labels:
        info, labels, digest = recipe.inspect_dataset(root)
        source = "meta/dlsj/labels.jsonl (success-only demonstrations)"
    else:
        info_bytes = (root / "meta/info.json").read_bytes()
        info = json.loads(info_bytes)
        check_info(info)
        labels, digest = labels_from_meta(root, info, info_bytes)
        source = "LeRobot meta/episodes (no labels.jsonl: success not checked, ALL episodes are used)"
    plan = recipe.estimate(labels, 1, batch, accumulate, epochs)
    plan.update({"episodes": len(labels), "total_frames": info["total_frames"], "label_source": source,
                 "dataset_metadata_sha256": digest,
                 "instructions": sorted({label["task"] for label in labels if label["task"]})})
    return plan


# --------------------------------------------------------------------------------------------- trainer

AMP_PLUGIN = "lightning.pytorch.plugins.precision.MixedPrecision"


def xpu_trainer(trainer: dict, *, precision: str, accumulate: int, output: Path, epochs: int,
                max_steps: int | None, time_limit_min: float | None, smoke: bool) -> dict:
    """Rewrite a Lightning trainer section for one Intel XPU.

    bf16-mixed: Lightning 2.6 maps `precision: bf16-mixed` to torch.autocast("cuda") for every accelerator except
    cpu/mps, so on XPU autocast is silently DISABLED and the step runs in fp32 (measured on the XPS, see README).
    We therefore pass an explicit MixedPrecision(precision="bf16-mixed", device="xpu") plugin and no `precision`
    flag (Lightning refuses both at once). fp16 ("16-mixed") is not offered: its GradScaler is CUDA-only there too.
    """
    t = copy.deepcopy(trainer)
    t.update(accelerator="xpu", strategy="xpu_single", devices=1, num_nodes=1,
             accumulate_grad_batches=accumulate, default_root_dir=str(output), max_epochs=epochs)
    t.pop("precision", None)
    if precision == "bf16-mixed":
        t["plugins"] = [{"class_path": AMP_PLUGIN, "init_args": {"precision": "bf16-mixed", "device": "xpu"}}]
    elif precision == "32":
        t["precision"] = "32-true"
    else:
        raise ValueError(f"precision must be bf16-mixed or 32, not {precision!r}")
    if max_steps:
        t["max_steps"] = max_steps
        t["max_epochs"] = None  # stop on steps (or the timer), not epochs
    if time_limit_min:
        minutes = int(round(time_limit_min))
        t["max_time"] = f"{minutes // 1440:02d}:{minutes % 1440 // 60:02d}:{minutes % 60:02d}:00"
    for callback, directory in zip(t.get("callbacks", []), ("best", "resume")):
        callback.setdefault("init_args", {})["dirpath"] = str(output / "checkpoints" / directory)
    if smoke:
        t.update(fast_dev_run=1, enable_checkpointing=False, callbacks=[], logger=False)
        t.pop("max_time", None)
    return t


def optimizer_steps(plan: dict, max_steps: int | None) -> int:
    return min(plan["optimizer_steps"], max_steps) if max_steps else plan["optimizer_steps"]


# --------------------------------------------------------------------------------------------- memory

def memory_estimate_gib(*, total_params: float, trainable_params: float, weight_bytes: float, batch: int,
                        act_gib_per_sample: float, overhead_gib: float = 3.0) -> dict:
    """Rough arithmetic, NOT a measurement: weights + grads + AdamW moments of the trainable params + activations.

    Grads and both Adam moments are counted in fp32 (4+8 bytes/param), which is conservative for bf16 params.
    On the Arc iGPU "device memory" IS system RAM, so this competes with the data loader and everything else.
    """
    weights = total_params * weight_bytes / GIB
    grads_optim = trainable_params * 12 / GIB
    activations = batch * act_gib_per_sample
    total = weights + grads_optim + activations + overhead_gib
    return {"weights_gib": round(weights, 1), "grads_and_adam_gib": round(grads_optim, 1),
            "activations_gib_guess": round(activations, 1), "overhead_gib": overhead_gib,
            "total_gib": round(total, 1)}


def mem_available_gib() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024 / GIB
    except OSError:
        return None
    return None


BUSY_PATTERNS = ("studio_export.py", "physicalai fit", "lerobot-train", "lerobot_train", "dlsj.inspect.train")


def busy_jobs() -> list[str]:
    """Other heavy jobs on this machine (an export, another training run)."""
    found = []
    for pattern in BUSY_PATTERNS:
        out = subprocess.run(["pgrep", "-af", pattern], capture_output=True, text=True).stdout
        found += [line for line in out.splitlines() if str(os.getpid()) != line.split()[0] and "pgrep" not in line]
    return found


def preflight(estimate_total_gib: float, *, force: bool, min_free_gib: float) -> list[str]:
    """Problems that should stop a real run on a shared machine. Empty list = go."""
    problems = []
    jobs = busy_jobs()
    if jobs:
        problems.append("other heavy jobs are running:\n  " + "\n  ".join(jobs))
    avail = mem_available_gib()
    if avail is not None and avail < estimate_total_gib + min_free_gib:
        problems.append(f"only {avail:.1f} GiB available; the estimate needs ~{estimate_total_gib:.0f} GiB "
                        f"+ {min_free_gib:.0f} GiB headroom")
    return [] if force else problems


# --------------------------------------------------------------------------------------------- studio

def expected_studio_commit() -> str:
    return (PI05_DIR / "versions.env").read_text().split("STUDIO_COMMIT=", 1)[1].split()[0].strip()


def studio_commit(studio: Path) -> str:
    return subprocess.check_output(["git", "-C", str(studio), "rev-parse", "HEAD"], text=True).strip()


def check_studio(studio: Path) -> Path:
    """Verify the Studio checkout matches training/pi05/versions.env; return its physicalai CLI."""
    expected, actual = expected_studio_commit(), studio_commit(studio)
    if actual != expected:
        raise ValueError(f"Studio revision mismatch: expected {expected}, got {actual}")
    cli = studio / "library/.venv/bin/physicalai"
    if not cli.exists():
        raise ValueError(f"{cli} not found. Use the Studio library venv synced with --extra xpu "
                         "(and call its binaries directly: `uv run` re-syncs and drops the xpu extra)")
    return cli


def write_run(output: Path, config: dict, plan: dict) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "training-config.json"
    path.write_text(json.dumps(config, indent=2) + "\n")
    (output / "data-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    return path


def launch(command: list[str], *, cwd: Path, nice: int, min_free_gib: float, log_every_s: float = 30.0) -> int:
    """Run `physicalai fit` niced, watch system memory, kill it if MemAvailable drops below `min_free_gib`.

    Prints the wall time and the lowest MemAvailable seen (the iGPU allocates from system RAM, so this is the
    number that matters on the XPS).
    """
    env = os.environ.copy()
    env.setdefault("PYTORCH_ENABLE_XPU_FALLBACK", "0")
    print("+", " ".join(command), flush=True)
    start, lowest, last_log = time.time(), float("inf"), 0.0
    proc = subprocess.Popen(command, cwd=cwd, env=env, preexec_fn=(lambda: os.nice(nice)) if nice else None,
                            start_new_session=True)
    try:
        while proc.poll() is None:
            avail = mem_available_gib()
            if avail is not None:
                lowest = min(lowest, avail)
                if avail < min_free_gib:
                    print(f"\n[xpu-train] MemAvailable {avail:.1f} GiB < {min_free_gib} GiB: killing the run",
                          file=sys.stderr, flush=True)
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                    break
                if time.time() - last_log > log_every_s:
                    last_log = time.time()
                    print(f"[xpu-train] {time.time() - start:7.0f}s  MemAvailable {avail:.1f} GiB", flush=True)
            time.sleep(2)
    except KeyboardInterrupt:
        os.killpg(proc.pid, signal.SIGINT)
        proc.wait()
    code = proc.wait()
    low = f"{lowest:.1f} GiB" if lowest != float("inf") else "n/a"
    print(f"[xpu-train] exit {code} after {time.time() - start:.0f}s; lowest MemAvailable {low}", flush=True)
    return code
