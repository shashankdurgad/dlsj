"""Recorded command / measured joint sequences for the play identification (plain python, no Isaac, no video).

* teleop: duplo_teleop_cal2_rgb_456 (LeRobot v3, 30 fps): action = leader reading (deg, gripper 0-100), state = the
  follower's encoders. The leader is read through its own calibration, so command - measured carries a constant
  per-joint leader / follower offset on top of the servo error.
* traces: the XPS controller's per-tick traces (real_rollouts/home/desmond/pi05_traces/*.jsonl): `cmd` is exactly
  what was written to the follower's Goal_Position (after the controller's max_step, before LeRobot's
  max_relative_target clamp), `measured` its Present_Position read on the same tick, 30 Hz.
Both index the command of tick t against the reading of tick t (the reading is taken before the command is sent),
so the command of tick t first shows in the reading of tick t+1 (servo_sysid.md: delay 0 in this indexing).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
TELEOP = REPO / "sim/twin/data/canonical/duplo_teleop_cal2_rgb_456"
TRACES = REPO / "sim/twin/data/real_rollouts/home/desmond/pi05_traces"
JOINT_FIT = REPO / "sim/twin/align/joint_fit/result.json"
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def w_offsets() -> np.ndarray:
    """The twin profile's joint offsets (joint fit W, rad, 6 incl. gripper 0)."""
    return np.array(json.loads(JOINT_FIT.read_text())["joint_offsets_rad_for_sim"], float)


def teleop(root: Path = TELEOP) -> dict[int, dict]:
    sys.path.insert(0, str(REPO / "sim/residual_env"))
    from bases import load_episodes

    eps = load_episodes(str(root))
    out = {}
    for e, d in eps.items():
        out[e] = {"cmd": d["action"].astype(np.float64), "meas": d["state"].astype(np.float64),
                  "inspection_frame": d["inspection_frame"], "excluded": d["excluded"], "src": "teleop"}
    return out


def traces(root: Path = TRACES, min_len: int = 30) -> dict[str, dict]:
    """Contiguous 30 Hz tick runs of every trace: {"<file>:<k>": {cmd, meas, phase, t}} (split at tick gaps)."""
    out = {}
    for f in sorted(root.glob("*.jsonl")):
        rows = []
        for line in f.read_text().splitlines():
            if '"tick"' not in line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("kind") == "tick" and r.get("measured") and r.get("cmd"):
                rows.append((int(r["t"]), r["measured"], r["cmd"], r.get("phase", ""), r.get("stage", "")))
        if not rows:
            continue
        t = np.array([r[0] for r in rows])
        cut = np.where(np.diff(t) != 1)[0] + 1
        for k, (a, b) in enumerate(zip(np.r_[0, cut], np.r_[cut, len(rows)], strict=True)):
            if b - a < min_len:
                continue
            seg = rows[a:b]
            out[f"{f.stem}:{k}"] = {"t": t[a:b], "meas": np.array([r[1] for r in seg], float),
                                    "cmd": np.array([r[2] for r in seg], float),
                                    "phase": np.array([r[3] for r in seg]), "stage": np.array([r[4] for r in seg]),
                                    "src": "trace"}
    return out
