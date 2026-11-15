"""Shared pieces of the grasp gate: the held-out demo list (identical to GraspEnv's test pool) and small helpers.

The gate (sim/twin/grasp_gate) explains and tracks the replay gap: real demos that picked the brick on the robot
must also pick it when replayed in the twin. Every rung runs on the same held-out cal2 demos.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "sim/twin/data/canonical/duplo_teleop_cal2_rgb_456"
OUT = REPO / "sim/twin/out/grasp_gate"
PRE_MAX = int(math.ceil(1.2 * 30))  # GraspCfg.pre_close_s[1]


def close_frame(grip, insp, thr=12.0, stay=15.0, open_min=15.0):
    """validate.py --grasp_rule gripper: first frame the gripper reading falls through 12 and stays <= 15."""
    g = grip[:insp] if insp else grip
    if len(g) == 0 or g.max() < open_min:
        return None
    for f in range(1, len(g)):
        if g[f - 1] > thr >= g[f] and (g[f:] <= stay).all():
            return f
    return None


def load_demos(data: str | Path = DATA) -> dict:
    """{"eps": {ep: {action, state, inspection_frame, cf}}, "keep": [ep...], "test": [...], "train": [...]} with the
    keep list and the 2:1 split of sim/twin/rl/grasp_env.py (GraspEnv._load_pool, seed 0)."""
    sys.path.insert(0, str(REPO / "sim/residual_env"))
    from bases import load_episodes

    eps = load_episodes(str(data))
    keep = []
    for e, dd in sorted(eps.items()):
        cf = close_frame(dd["state"][:, 5], dd["inspection_frame"])
        dd["cf"] = cf
        if dd["excluded"] or cf is None or cf < PRE_MAX + 1 or not dd["inspection_frame"]:
            continue
        keep.append(e)
    perm = np.random.default_rng(0).permutation(len(keep))
    k = int(round(len(keep) * 2 / 3))
    return {"eps": eps, "keep": keep, "train": [keep[i] for i in np.sort(perm[:k])],
            "test": [keep[i] for i in np.sort(perm[k:])]}


def pick_test(demos: dict, n: int, seed: int = 0) -> list:
    """The gate's fixed demo subset: n held-out demos (seeded draw, sorted)."""
    t = demos["test"]
    if n >= len(t):
        return list(t)
    return sorted(np.random.default_rng(seed).choice(t, size=n, replace=False).tolist())


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score interval of k successes in n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return (round(c - h, 3), round(c + h, 3))


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1))
