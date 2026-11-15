"""Run provenance for config.json / checkpoints: the twin's physics, offsets, servo / play models and the md5s of the
calibration files a residual was trained against (so a deploy can check it runs the same twin)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

TWIN = Path(__file__).resolve().parents[1]
FILES = {"physics_params": TWIN / "physics/params.json", "play_params": TWIN / "dynamics/params.json",
         "joint_fit_result": TWIN / "align/joint_fit/result.json", "camera_color": TWIN / "visuals/camera_color.json",
         "camera_model": TWIN / "sensors/camera_model.json"}


def md5(path) -> str | None:
    p = Path(path)
    return hashlib.md5(p.read_bytes()).hexdigest() if p.exists() else None


def provenance(servo_params: str, play_cfg=None, extra_files: dict | None = None) -> dict:
    """phys.summary(), TWIN_PHYSICS_SET, DLSJ_JOINT_OFFSETS, the servo file, the play model's cfg + params and md5s."""
    from sim.twin.physics import apply as phys

    files = {**FILES, "servo_params": Path(servo_params), **(extra_files or {})}
    out = {"TWIN_PHYSICS_SET": os.environ.get("TWIN_PHYSICS_SET", ""),
           "DLSJ_JOINT_OFFSETS": os.environ.get("DLSJ_JOINT_OFFSETS", ""),
           "servo_params": str(servo_params), "physics": phys.summary(),
           "md5": {k: md5(v) for k, v in files.items()}}
    if play_cfg is not None:
        from dataclasses import asdict

        out["play_cfg"] = asdict(play_cfg)
        out["play_params"] = json.loads(Path(play_cfg.params).read_text())
    return out
