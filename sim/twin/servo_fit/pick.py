"""Per-joint reading / velocity RMSE (twin vs real) of every parameter set of twin_replay.py runs, split into trace
(deployed controller) and teleop windows; prints the best set per joint.

    sim/twin/dynamics/py.sh sim/twin/servo_fit/pick.py g_arm1 g_arm2
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "sim/twin/out/servo_fit"
J = ("pan", "lift", "elbow", "wflex", "wroll", "grip")


def main(tags):
    Wz = np.load(OUT / "windows.npz")
    rows = []
    for tag in tags:
        R = json.loads((OUT / tag / "result.json").read_text())
        Z = np.load(OUT / tag / "replay.npz")
        W, nw = R["W"], len(Wz["name"])
        n = R["args"].get("n_windows") or 0
        idx = np.linspace(0, nw - 1, n).round().astype(int) if n else np.arange(nw)
        T = Z["meas"].shape[1]
        real = Wz["meas"][idx, :T].astype(float)
        tele = Wz["src"][idx] == "teleop"
        for p, s in enumerate(R["sets"]):
            y = Z["meas"][p * W:(p + 1) * W].astype(float)
            out = {}
            for part, k in (("trace", ~tele), ("teleop", tele)):
                e = y[k] - real[k]
                ev = np.diff(y[k], axis=1) - np.diff(real[k], axis=1)
                out[part] = (np.sqrt((e ** 2).mean((0, 1))), np.sqrt((ev ** 2).mean((0, 1))))
            rows.append((f"{tag}:{p}", s, out))
    for name, s, out in rows:
        print(name, "K", s.get("K", [None])[:5], "d", s.get("delay"), "v", (s.get("vmax_s") or [None])[0],
              "stiff", (s.get("stiffness") or [None])[0])
        for part in ("trace", "teleop"):
            print(f"   {part:6s} rmse", out[part][0][:5].round(3).tolist(), "vel", out[part][1][:5].round(3).tolist())
    for part in ("trace", "teleop"):
        for j in range(5):
            b = sorted(rows, key=lambda r: r[2][part][0][j] + 2 * r[2][part][1][j])
            print(f"best {part} {J[j]}:", [(r[0], round(float(r[2][part][0][j]), 3), round(float(r[2][part][1][j]), 3))
                                          for r in b[:3]])


if __name__ == "__main__":
    main(sys.argv[1:])
