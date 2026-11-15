"""Run any twin script with the wrist camera from wrist_mount.json, without editing calib/profile.py.

    sim/twin/run.sh sim/twin/robot/with_mount.py [--wrist mount|joe] <repo script.py> [script args...]
    e.g. sim/twin/run.sh sim/twin/robot/with_mount.py sim/twin/env/replay_video.py sim --ep 76
         sim/twin/run.sh sim/twin/robot/with_mount.py sim/twin/visuals/render_compare.py --frame 75 --tag _mount

Patches, in this process only, the two places a profile's wrist camera comes from:
* calib/profile.py `_cams()["wrist"]` (twin_profile() / randomize(): TwinEnv, replay_video, DR);
* residual-env calib.load_profile (render_compare.py's `--profile joe_cal2`).
--wrist mount: wrist_mount.json (sim rotation, fitted centre, measured focal); --wrist joe: joe_cal2's pose with the
measured focal (= the current twin_profile(), as a baseline for render_compare). The paint.py mount follows the camera.

The permanent switch is one line in calib/profile.py `_cams()`:
    "wrist": __import__("sim.twin.robot.wrist_mount", fromlist=["cams_entry"]).cams_entry(),
"""
from __future__ import annotations

import dataclasses
import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "sim/residual_env"))


def patch(mode: str = "mount") -> tuple:
    """Install the wrist-camera override; returns the (local_to_cv, position, focal) used."""
    import calib

    from sim.twin.calib import profile
    from sim.twin.robot.wrist_mount import cams_entry

    orig_cams, orig_load = profile._cams, calib.load_profile
    entry = cams_entry() if mode == "mount" else orig_cams()["wrist"]

    def cams():
        c = orig_cams()
        c["wrist"] = entry
        return c

    def load_profile(name, *a, **kw):
        p = orig_load(name, *a, **kw)
        return dataclasses.replace(p, wrist=calib.cv_to_usd_rows(*entry)) if p.wrist is not None else p

    profile._cams = cams
    calib.load_profile = load_profile
    return entry


def main():
    """`with_mount.py [--wrist mount|joe] script.py args...`: patch, then run the script as __main__."""
    argv = sys.argv[1:]
    mode = "mount"
    if argv[:1] == ["--wrist"]:
        mode, argv = argv[1], argv[2:]
    R, t, f = patch(mode)
    print(f"[with_mount] wrist camera = {mode}: centre {[round(float(v), 4) for v in t]} focal {f:.1f}", flush=True)
    sys.argv = argv
    runpy.run_path(str(REPO / argv[0]), run_name="__main__")


if __name__ == "__main__":
    main()
