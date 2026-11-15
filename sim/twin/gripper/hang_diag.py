"""Fin-joint health over a visual-fit style pose sweep (TwinEnv teleports to recorded states, plates / bricks moved).

    sim/twin/run.sh sim/twin/gripper/hang_diag.py [--n 30] [--set key=json ...]
    sim/twin/run.sh sim/twin/gripper/hang_diag.py --set ... --sweep -- <sweep_looks.py args>   # the real sweep
Prints the fin hinge angles / speeds after every pose; a 240 s watchdog dumps the stacks and exits on a hang.
--sweep runs visuals/fit/sweep_looks.py itself with the physics overrides (run.sh passes no environment).
"""
import argparse
import faulthandler
import os
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--set", action="append", default=[])
ap.add_argument("--sweep", nargs=argparse.REMAINDER, default=None)
args = ap.parse_args()
if args.set:  # the physics layer reads these through TwinEnv's load_params
    os.environ["TWIN_PHYSICS_SET"] = ";".join(args.set)
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
if args.sweep is not None:
    import runpy

    script = str(REPO / "sim/twin/visuals/fit/sweep_looks.py")
    sys.argv = [script, *[a for a in args.sweep if a != "--"]]
    print("SWEEP", sys.argv, "set", args.set, flush=True)
    runpy.run_path(script, run_name="__main__")
    print("SWEEP_DONE", flush=True)
    os._exit(0)
faulthandler.dump_traceback_later(240, exit=True)
import numpy as np  # noqa: E402

from sim.twin.visuals.fit.harness import Harness  # noqa: E402


def arr(x):
    x = getattr(x, "torch", x)
    return x[0].detach().cpu().numpy()


h = Harness()
R = h.env.residual
names = R.robot.joint_names
fin = [i for i, nm in enumerate(names) if nm.startswith("fin_")]
print("DIAG joints", names, "set", args.set, flush=True)
for k, i in enumerate(np.linspace(0, len(h.samples) - 1, args.n).astype(int).tolist()):
    s = h.samples[i]
    faulthandler.dump_traceback_later(240, exit=True)
    t0 = time.time()
    h.pose(s)
    q, qd = arr(R.robot.data.joint_pos), arr(R.robot.data.joint_vel)
    fq = float(np.degrees(np.abs(q[fin])).max()) if fin else 0.0
    fv = float(np.abs(qd[fin]).max()) if fin else 0.0
    print(f"DIAG {k} sample {i} ep {s['ep']} {s['moment']} {time.time() - t0:.1f}s finite {np.isfinite(q).all()} "
          f"fin_deg_max {fq:.1f} fin_qd_max {fv:.2f} arm_deg {np.degrees(q[:6]).round(1).tolist()}", flush=True)
print("DIAG_DONE", flush=True)
faulthandler.cancel_dump_traceback_later()
os._exit(0)
