"""Held brick pose per gate run: median brick centre in gripper_link over the lifted ticks of the successful demos,
vs the real one fitted from 599 held wrist frames (perception agent): (-0.8, +2.8, -88.0) mm.

    sim/twin/run.sh sim/twin/grasp_gate/heldpose.py <tag> ...   (plain python)  -> out/grasp_gate/heldpose.json (merged)
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from sim.twin.grasp_gate.common import OUT  # noqa: E402

REAL = np.array([-0.8, 2.8, -88.0])
path = OUT / "heldpose.json"
res = json.loads(path.read_text()) if path.exists() else {}
for tag in sys.argv[1:]:
    tr = np.load(OUT / tag / "trace.npz")
    per = {d["episode"]: d for d in json.loads((OUT / tag / "result.json").read_text())["episodes"]}
    pts = []
    for k, ep in enumerate(tr["episodes"].tolist()):
        if not per[ep]["success"]:
            continue
        n = int(tr["end"][k] - tr["start"][k]) + 1
        h = tr["bp"][k, :n, 2] - tr["b0"][k, 2]
        near = np.linalg.norm(tr["bp"][k, :n] - tr["tip"][k, :n], axis=1) < 0.04
        sel = (h > 0.03) & near
        if sel.any():
            pts.append(np.median(tr["rel"][k, :n][sel], 0) * 1e3)
    if not pts:
        continue
    m = np.median(np.array(pts), 0)
    res[tag] = {"held_rel_mm": np.round(m, 1).tolist(), "dx_mm": round(float(m[0] - REAL[0]), 1),
                "dy_mm": round(float(m[1] - REAL[1]), 1), "dz_mm": round(float(m[2] - REAL[2]), 1), "n": len(pts)}
    print(tag, res[tag], flush=True)
path.write_text(json.dumps(res, indent=1))
