"""Which offsets or sag reproduce a ~9 mm lower fingertip at the grasp frames, keeping the camera silhouettes?

The replay depth gap (gripper agent, real fin-ray fingers modelled) is 8-10 mm: the sim fingertip sits that much above
where the real one was at the gripper-close frames. Fits with a grasp-depth term (model.grasp_loss: fingertip at each
close frame 9 mm below where joe_cal2 puts it) added to the camera objective (top + inspect + table view):
  D9: offsets (5) + inspect pose + wrist mount + plane
  S9: gravity sag (3: q_lift/elbow/wflex += sag * static gravity torque) + pan / roll offsets (lift / elbow / wflex
      offsets at joe_cal2) + inspect + mount + plane
and without it:
  E:  C + sag (3), cameras only (do the cameras by themselves ask for sag?)
Compared with C (offsets, cameras only) by fit / held-out loss per camera and the fingertip change achieved.

    python sim/twin/align/joint_fit/stage3.py [--workers 16] [--runs D9 S9 E]   -> out/align/joint_fit/stage3.json
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--maxfev", type=int, default=1500)
    ap.add_argument("--runs", nargs="+", default=["D9", "S9", "E"])
    args = ap.parse_args()
    F._init()
    pool = Pool(args.workers, initializer=F._init)
    ev = F.Evaluator(pool, args.workers)
    out = M.TWIN / "out/align/joint_fit"
    out_json = out / "stage3.json"
    thC = th_from(json.loads((out / "fit.json").read_text())["theta"]["C"])
    sets = F.make_sets(ev, M.zero_theta())
    cams_only = [i for c in F.OBJ for i in sets[(c, False)]]
    with_grasp = cams_only + sets[("grasp", False)]
    res = json.loads(out_json.read_text()) if out_json.exists() else {}
    th = {"joe_cal2 (current twin)": M.zero_theta(), "C": thC}
    th.update({k: th_from(v) for k, v in res.get("theta", {}).items() if k not in th})
    runs = {
        "D9": (["joints", "inspect", "wrist", "plane"], thC, with_grasp),
        "S9": (["jpr", "sag", "inspect", "wrist", "plane"],
               {**thC, "joints": np.zeros(5), "jpr": thC["joints"][[0, 4]]}, with_grasp),
        "E": (["joints", "sag", "inspect", "wrist", "plane"], thC, cams_only),
    }
    for name in args.runs:
        free, start, idx = runs[name]
        print(name, free, flush=True)
        th[name], _ = F.fit(ev, free, start, idx, maxfev=args.maxfev, label=name)
        F.report(th[name], name)
        F.scoreboard(ev, th, sets, res, f"after {name}")
        out_json.write_text(json.dumps(res, indent=1))
    pool.close()
    pool.join()
    print("STAGE3_OK", flush=True)


if __name__ == "__main__":
    main()
