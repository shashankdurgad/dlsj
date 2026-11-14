"""Calibration profiles for the residual env: joint offsets, cameras, gripper mapping, robot USD.

A profile pins everything that maps OUR units (LeRobot degrees, gripper 0-100) and the real cameras into the sim:

* "joe"       Joe's original camera-fit.json (old calibration): offsets, scene-final.usda cameras.
* "fit_old"   Joe's cameras + data-fitted body-joint offsets for old-calibration data (sim/realism/joint_offset_fit.md).
* "fit_cal2"  Joe's (old) cameras + data-fitted offsets for cal2 states.
* "joe_cal2"  Joe's cal2 calibration (pi05-isaac-so101/cal2-calibration/reference-cal2.json, his
              "replay-ep101-corrected-cameras"): multi-pose camera/joint fit on cal2 episodes 101/110 (held-out 120),
              top + wrist + inspect cameras, offsets incl. the reseated wrist_roll (-1.80 rad), his gripper encoder
              scale (0.0231 rad/unit from the follower's range). Pairs with cal2 datasets and the current robot, no
              recal offsets.

Gripper: q_rad = units * scale + bias. "joe"/"fit_*": scale = upper/100, bias = GRIP_OFFSET_UNITS * upper/100 (our
box finger pads stall on a 31.8 mm brick at the real hold 5.8). "joe_cal2": Joe's encoder scale with the bias
re-solved so OUR pads stall at the real hold value (Joe's own bias 0.105 rad assumes his CAD-jaw contact; with our
pads it would read ~1.7 instead of ~5.8), see GRIP_STALL_RAD_CAL2 below (measured pad stall).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
JOE = Path("/home/joe-spark-2/pi05-isaac-so101")
GRIPPER_UPPER = math.radians(100.000046)
GRIP_OFFSET_UNITS = 2.4  # pads stall on a 31.8 mm brick at 8.2 deg sim = real hold 5.8 units (old mapping)
GRIP_HOLD_REAL = 5.8  # units, A1/A2 (median 5.75-5.9)
GRIP_STALL_RAD = (GRIP_HOLD_REAL + GRIP_OFFSET_UNITS) / 100.0 * GRIPPER_UPPER  # nominal pad stall (old mapping)
# Measured in the env (validate.py, 20 cal2 replays, joe_cal2 cameras/offsets): the pads stall on the brick at
# 0.1076 rad (median hold read 4.26 units with bias = GRIP_STALL_RAD - 5.8*scale). Re-solved so the held brick reads
# the real 5.8: bias = 0.1076 - 5.8 * 0.0230767.
GRIP_STALL_RAD_CAL2 = 0.10756
CAL2_JSON = HERE / "assets/joe_cal2/reference-cal2.json"
PROFILES = ("joe", "fit_old", "fit_cal2", "joe_cal2")


@dataclass
class Camera:
    rows: list  # USD row-vector 4x4 (rows = camera x/y/z axes in the parent frame, row 4 = position)
    intr: dict  # focal_length / horizontal_aperture / vertical_aperture (mm, pinhole)

    @property
    def f_px(self) -> float:
        return self.intr["focal_length"] / self.intr["horizontal_aperture"] * 640.0


@dataclass
class Profile:
    name: str
    joint_offsets: list
    grip_scale: float
    grip_bias: float
    top: Camera | None = None  # None = scene.py defaults (Joe's scene-final.usda)
    wrist: Camera | None = None
    inspect: Camera | None = None
    robot_usd: str | None = None  # None = scene.ROBOT_USD
    gripper_effort: float | None = None  # N m; None = the arm's effort limit (2.9). Joe's 0.5 N m was tested for
    # joe_cal2 and changed nothing in the 20-episode replay or the hold value, so it is not set.


def cv_to_usd_rows(r_cv, position, focal_px) -> Camera:
    """Joe's convention (configure_reference.set_camera): r rows = OpenCV axes (x right, y down, z forward) in the
    parent frame -> USD camera rows x, -y, -z; focal 20 mm with apertures 20*W/f, 20*H/f."""
    r = np.asarray(r_cv, float)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = r[0], -r[1], -r[2]
    m[3, :3] = position
    f = float(focal_px)
    return Camera(m.tolist(), dict(focal_length=20.0, horizontal_aperture=20.0 * 640 / f,
                                   vertical_aperture=20.0 * 480 / f))


def _fit_offsets(key: str) -> list:
    for cand in (HERE / "joint_offsets_fit.json", HERE.parent / "realism" / "joint_offsets_fit.json"):
        if cand.exists():
            return list(json.loads(cand.read_text())[key]["offsets_rad_for_sim"])
    raise FileNotFoundError("joint_offsets_fit.json not found next to env.py or in sim/realism/")


def load_profile(name: str, cal2_json: str | Path | None = None) -> Profile:
    old_scale = GRIPPER_UPPER / 100.0
    old_bias = GRIP_OFFSET_UNITS * old_scale
    if name in ("", "joe"):
        return Profile("joe", [0.0, -0.2756362497772003, 0.056384559242785785, -0.45744995615777856, 0.0, 0.0],
                       old_scale, old_bias)
    if name in ("fit_old", "fit_cal2"):
        return Profile(name, _fit_offsets(name.removeprefix("fit_")), old_scale, old_bias)
    if name == "joe_cal2":
        p = Path(cal2_json) if cal2_json else CAL2_JSON
        ref = json.loads(p.read_text())
        usd = JOE / "assets/converted/so101_new_calib_11/so101_new_calib.usda"
        gm = ref["gripper_mapping"]
        scale = float(gm["scale_rad_per_unit"])
        return Profile(
            "joe_cal2", list(ref["joint_offsets_rad"]), scale, GRIP_STALL_RAD_CAL2 - GRIP_HOLD_REAL * scale,
            top=cv_to_usd_rows(ref["camera"]["world_to_cv"], ref["camera"]["position"], ref["camera"]["focal_px"]),
            wrist=cv_to_usd_rows(ref["wrist_camera"]["local_to_cv"], ref["wrist_camera"]["position_local"],
                                 ref["wrist_camera"]["focal_px"]),
            inspect=cv_to_usd_rows(ref["inspect_camera"]["world_to_cv"], ref["inspect_camera"]["position"],
                                   ref["inspect_camera"]["focal_px"]),
            robot_usd=str(usd))
    raise ValueError(f"unknown calibration profile {name!r}; choose from {PROFILES}")


def check_robot_usd(path: str) -> str:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"robot USD {path} not found. The joe_cal2 profile references Joe's converted SO-101 on the Spark "
            "(pi05-isaac-so101/assets/converted/so101_new_calib_11, content-identical to _2); it is not copied into "
            "the repo (30 MB). Run on the Spark or point Profile.robot_usd at a local copy.")
    return path
