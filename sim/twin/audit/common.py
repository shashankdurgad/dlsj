"""Shared paths / constants of the realism audit (standard library only, safe in every process)."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TWIN = REPO / "sim/twin"
OUT = TWIN / "out/audit"
SET = OUT / "set"
RUNS = OUT / "runs"
DATA = TWIN / "data/canonical/duplo_teleop_cal2_rgb_456"
FT_DIR = TWIN / "data/models/molmoact2_ft_8k"
CAMS = ("top", "wrist", "inspect")
POLICY_CAMS = ("top", "wrist")  # what MolmoAct2 is fed on the robot (wrist, top)
PHASES = ("start", "reach", "pregrasp", "grasp", "lift", "inspect", "place")
HELD = ("grasp", "lift", "inspect", "place")  # the target brick is between the jaws
# region bits of the sim-geometry masks (render_set.py)
BITS = {"table": 1, "plates": 2, "bricks": 4, "robot": 8, "props": 16}


def load_set() -> dict:
    return json.loads((SET / "set.json").read_text())


def jdump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=_default))


def _default(o):
    try:
        import numpy as np

        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
    except ImportError:
        pass
    return str(o)
