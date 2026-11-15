"""The twin's calibration profile: joe_cal2 (joint offsets, gripper mapping, wrist mount) with the measured
cameras, plus per-camera domain randomisation around them.

* top:     joe_cal2 pose (confirmed by the arm silhouettes and the board-on-table check), factory focal;
* inspect: ChArUco-measured pose and factory focal (cameras_measured.json); it sits on a swivel tripod, so it is
           randomised widely enough to cover the shifts seen (joe_cal2 and the silhouette fit are ~5 cm / 5 deg away);
* wrist:   joe_cal2 mount pose, measured focal (342 px vs the assumed 394).
The residual env's camera model is a centred pinhole with one focal (cv_to_usd_rows); the measured principal
points (top cx 304, wrist cx 339) are not modelled yet.

    from sim.twin.calib.profile import twin_profile, randomize
    prof = twin_profile()                         # nominal
    prof = randomize(twin_profile(), rng)         # one DR draw
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "sim/residual_env"))
from calib import cv_to_usd_rows, load_profile  # noqa: E402

USD = str(REPO / "sim/twin/assets/so101/usd/so101_new_calib/so101_new_calib.usda")
# DR half-ranges: (position m, rotation deg, focal fraction)
DR = {"top": (0.01, 1.0, 0.01), "inspect": (0.05, 5.0, 0.02), "wrist": (0.005, 3.0, 0.03)}


def _cams() -> dict:
    mea = json.loads((HERE / "cameras_measured.json").read_text())
    fit = json.loads((HERE / "cameras_fit.json").read_text())
    from sim.twin.robot import wrist_mount  # imported here to avoid an import cycle

    return {
        # top: pose fitted to arm silhouettes with the factory intrinsics (best held-out fit, cameras_fit.json)
        "top": (
            np.array(fit["top"]["world_to_cv"]),
            np.array(fit["top"]["position"]),
            (fit["top"]["fx"] + fit["top"]["fy"]) / 2,
        ),
        "inspect": (
            np.array(mea["inspect"]["world_to_cv"]),
            np.array(mea["inspect"]["position"]),
            (mea["inspect"]["fx"] + mea["inspect"]["fy"]) / 2,
        ),
        # wrist: mount pose fitted from rigid gripper evidence (fin silhouette, held bricks), sim/twin/robot/
        "wrist": wrist_mount.cams_entry(),
    }


def twin_profile(cams: dict | None = None):
    base = load_profile("joe_cal2")
    c = cams or _cams()
    return dataclasses.replace(
        base, name="twin_measured", robot_usd=USD, **{k: cv_to_usd_rows(*c[k]) for k in ("top", "inspect", "wrist")}
    )


def randomize(_prof_unused, rng: np.random.Generator):
    """A twin profile with every camera perturbed uniformly within DR.

    Position, rotation about a random axis and focal are each drawn uniformly within the half-ranges in DR.
    """
    c = {}
    for k, (R, p, f) in _cams().items():
        dp, dr, df = DR[k]
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        Rd = Rotation.from_rotvec(axis * np.radians(rng.uniform(-dr, dr))).as_matrix()
        c[k] = (R @ Rd.T, p + rng.uniform(-dp, dp, 3), f * (1 + rng.uniform(-df, df)))
    return twin_profile(c)


if __name__ == "__main__":
    p = twin_profile()
    print(p.name, "| offsets", np.round(p.joint_offsets, 3).tolist(), "| robot", Path(p.robot_usd).name)
    for k in ("top", "inspect", "wrist"):
        print(k, "rows[3] (position):", np.round(np.array(getattr(p, k).rows)[3], 3).tolist())
    q = randomize(None, np.random.default_rng(0))
    print("DR draw inspect position:", np.round(np.array(q.inspect.rows)[3], 3).tolist())
