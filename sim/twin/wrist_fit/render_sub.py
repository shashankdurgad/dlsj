"""Render only the reach + pregrasp audit samples in the CURRENT twin (Isaac), for the fast wrist loop.

    NAME=dlsj-twin-wristfit sim/twin/run.sh sim/twin/wrist_fit/render_sub.py --run sim/twin/out/wrist_fit/runs/<tag> \
        [--phases reach,pregrasp] [--layouts <audit run>/layouts.json]

Runs the audit's render_set.py unchanged on a subset of the audit set (load_set patched before it is imported), so the
images and masks are exactly what the audit would render. Writes <run>/sim.npz (the subset, in set order).
"""
from __future__ import annotations

import runpy
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import sim.twin.audit.common as common  # noqa: E402

args = sys.argv[1:]


def opt(name, default):
    if name in args:
        k = args.index(name)
        v = args[k + 1]
        del args[k: k + 2]
        return v
    return default


phases = opt("--phases", "reach,pregrasp").split(",")
lay_src = opt("--layouts", str(REPO / "sim/twin/out/audit/runs/20261115-015346/layouts.json"))
run = Path(args[args.index("--run") + 1])
run.mkdir(parents=True, exist_ok=True)
if not (run / "layouts.json").exists():
    shutil.copy(lay_src, run / "layouts.json")
_orig = common.load_set


def load_subset() -> dict:
    m = _orig()
    return dict(m, samples=[s for s in m["samples"] if s["phase"] in phases])


common.load_set = load_subset
sys.argv = [str(REPO / "sim/twin/audit/render_set.py"), *args]
runpy.run_path(sys.argv[0], run_name="__main__")
