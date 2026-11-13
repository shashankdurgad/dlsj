"""Named follower poses saved next to the calibration, like Desmond's home pose.

`calibration/robots/so_follower/<follower_id>.<name>.json` holds {"shoulder_pan.pos": deg, ...} in the same
format as `<follower_id>.home.json` (see `dlsj.arms.save_home`). Used for the optional "near a saved pose"
skill termination (e.g. the `inspect` pose) and for homing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
JOINT_KEYS = [f"{m}.pos" for m in MOTORS]
ARM = slice(0, 5)  # every joint but the gripper
GRIPPER = 5


def _cal_dir() -> Path:
    from dlsj import arms  # arms imports nothing heavy at module level

    return arms.FOLLOWER_CAL_DIR


def pose_path(follower_id: str, name: str, cal_dir: Path | None = None) -> Path:
    return (cal_dir or _cal_dir()) / f"{follower_id}.{name}.json"


def to_vec(d: dict) -> np.ndarray:
    return np.array([float(d[k]) for k in JOINT_KEYS], dtype=np.float32)


def to_dict(v: np.ndarray) -> dict[str, float]:
    return {k: round(float(x), 2) for k, x in zip(JOINT_KEYS, v, strict=True)}


def load_pose(follower_id: str, name: str, cal_dir: Path | None = None) -> np.ndarray | None:
    p = pose_path(follower_id, name, cal_dir)
    return to_vec(json.loads(p.read_text())) if p.exists() else None


def save_pose(follower_id: str, name: str, pose: np.ndarray, cal_dir: Path | None = None) -> Path:
    p = pose_path(follower_id, name, cal_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(to_dict(pose), indent=2) + "\n")
    return p


def load_poses(follower_id: str, names: list[str], cal_dir: Path | None = None) -> dict[str, np.ndarray]:
    out = {}
    for n in names:
        v = load_pose(follower_id, n, cal_dir)
        if v is not None:
            out[n] = v
    return out
