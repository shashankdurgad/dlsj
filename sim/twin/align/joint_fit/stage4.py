"""Refit with the wrist camera FIXED to the robot-model agent's mount (sim/twin/robot/wrist_mount.json: fitted to
evidence rigid in gripper_link, so independent of the joint offsets; its roll trades against wrist_roll).

  W:   offsets (5) + inspect pose + plane            | top + inspect + wrist table view
  W9:  as W + grasp-depth term (fingertip 9 mm below joe_cal2 at the gripper-close frames)
  WS9: gravity sag (3) + pan / roll offsets (lift / elbow / wflex at joe_cal2) + inspect + plane, + grasp term
  WE:  W + sag (3), cameras only: does a link sag proportional to the static gravity torque (not seen by the encoders)
       fit the silhouettes across poses with very different torques better than constant offsets?

    python sim/twin/align/joint_fit/stage4.py [--workers 16] [--runs W W9 WS9]  -> out/align/joint_fit/stage4.json
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit as F  # noqa: E402
import model as M  # noqa: E402
from stage2 import th_from  # noqa: E402

MOUNT = M.TWIN / "robot/wrist_mount.json"


def use_wrist_mount():
    """Replace the joe_cal2 wrist camera pose in model.BASE by wrist_mount.json (physical camera rotation, used with
    the measured intrinsics + distortion as everywhere in model.py). Call before the worker pool forks."""
    c = json.loads(MOUNT.read_text())["camera"]
    M.BASE["wrist"] = {**M.BASE["wrist"], "R": np.array(c["local_to_cv"]), "t": np.array(c["position_local"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--maxfev", type=int, default=900)
    ap.add_argument("--runs", nargs="+", default=["W", "W9", "WS9", "WE"])
    args = ap.parse_args()
    use_wrist_mount()
    F._init()
    pool = Pool(args.workers, initializer=F._init)
    ev = F.Evaluator(pool, args.workers)
    out = M.TWIN / "out/align/joint_fit"
    out_json = out / "stage4.json"
    thC = th_from(json.loads((out / "fit.json").read_text())["theta"]["C"])
    start = {**M.zero_theta(), "joints": thC["joints"], "inspect": thC["inspect"], "plane": thC["plane"]}
    sets = F.make_sets(ev, M.zero_theta())
    cams_only = [i for c in F.OBJ for i in sets[(c, False)]]
    with_grasp = cams_only + sets[("grasp", False)]
    res = json.loads(out_json.read_text()) if out_json.exists() else {}
    res["wrist_mount"] = str(MOUNT)
    th = {"joe_cal2 offsets + wrist_mount": M.zero_theta(), "C offsets + wrist_mount": start}
    th.update({k: th_from(v) for k, v in res.get("theta", {}).items() if k not in args.runs})
    runs = {"W": (["joints", "inspect", "plane"], lambda: start, cams_only),
            "W9": (["joints", "inspect", "plane"], lambda: th["W"] if "W" in th else start, with_grasp),
            "WS9": (["jpr", "sag", "inspect", "plane"],
                    lambda: {**start, "joints": np.zeros(5), "jpr": start["joints"][[0, 4]]}, with_grasp),
            "WE": (["joints", "sag", "inspect", "plane"], lambda: th["W"], cams_only)}
    for name in args.runs:
        free, st, idx = runs[name]
        print(name, free, flush=True)
        th[name], _ = F.fit(ev, free, st(), idx, maxfev=args.maxfev, label=name)
        F.report(th[name], name)
        res.setdefault("theta", {})[name] = {k: v.tolist() for k, v in th[name].items()}
        out_json.write_text(json.dumps(res, indent=1))
    F.scoreboard(ev, th, sets, res, "wrist camera fixed to wrist_mount.json")
    out_json.write_text(json.dumps(res, indent=1))
    pool.close()
    pool.join()
    print("STAGE4_OK", flush=True)


if __name__ == "__main__":
    main()
