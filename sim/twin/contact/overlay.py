"""Sim brick vs real wrist frames: the gate's simulated brick (trace bp / bq) projected into the REAL wrist image
through the real recorded joints (perception/geometry.py wrist camera, the run's joint offsets), around the sim's
first fin-brick contact. Green = the sim brick body box, yellow dot = the sim fixed-fin tip (FK of the sim joints).

    sim/twin/run.sh sim/twin/contact/overlay.py --tag base_fx [--episodes 31 104 ...] [--offsets .../W9.json]
-> sim/twin/out/contact/<tag>/overlay_real_wrist.jpg (plain python, PyAV)
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from sim.twin.align import frames as F  # noqa: E402
from sim.twin.contact.forensics import _quat_rot  # noqa: E402
from sim.twin.perception.geometry import Geometry  # noqa: E402

OUT = REPO / "sim/twin/out/contact"
HALF = {0: (0.01585, 0.01585, 0.0096), 1: (0.03185, 0.01585, 0.0096), 2: (0.02385, 0.01585, 0.0096)}
EDGES = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--episodes", type=int, nargs="*")
    ap.add_argument("--offsets", default=str(REPO / "sim/twin/grasp_gate/offsets/W9.json"))
    ap.add_argument("--dt", default="-6,0,10,20", help="ticks relative to the sim's first contact (plus the close)")
    a = ap.parse_args()
    d = OUT / a.tag
    tr = np.load(d / "trace.npz")
    fx = {r["episode"]: r for r in json.loads((d / "forensics.json").read_text())}
    off = json.loads(Path(a.offsets).read_text())
    off = off.get("joint_offsets_rad") or off.get("offsets_rad") or off.get("joint_offsets") or off
    G = Geometry(offsets_rad=np.asarray(off, float)[:6] if not isinstance(off, dict) else None)
    fin_local = np.r_[tr["fin_tip_local"], 1.0]
    eps = a.episodes or [e for e, r in fx.items() if r["outcome"] != "success" and "first_shape" in r]
    rows = []
    for ep in eps:
        k = tr["episodes"].tolist().index(ep)
        r = fx[ep]
        ic = int(tr["cf"][k] - tr["start"][k])
        t0 = ic + int(r["first_tick_rel_close"])
        ticks = [t0 + int(v) for v in a.dt.split(",")] + [ic]
        ticks = [int(np.clip(t, 0, tr["q"].shape[1] - 1)) for t in ticks]
        frames = F.video_frames(ep, "wrist", [int(tr["start"][k]) + t for t in ticks])
        tiles = []
        for t in ticks:
            img = frames[int(tr["start"][k]) + t].copy()
            J = tr["s_real"][k, t][None]
            Rw, C = G.camera_pose(J)
            R = _quat_rot(tr["bq"][k, max(t - 1, 0)])
            h = np.array(HALF.get(int(tr["kind"][k]), HALF[0]))
            corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h
            X = tr["bp"][k, max(t - 1, 0)] + corners @ R.T
            uv, z = G.project(Rw, C, X[None])
            uv = uv[0]
            for i, j in EDGES:
                if z[0, i] > 0 and z[0, j] > 0:
                    cv2.line(img, tuple(int(v) for v in uv[i]), tuple(int(v) for v in uv[j]), (0, 255, 0), 2)
            M = G.gripper_pose(tr["q"][k, t][None])[0]
            ft = (M @ fin_local)[:3]
            fu, fz = G.project(Rw, C, ft[None, None])
            if fz[0, 0] > 0:
                cv2.circle(img, tuple(int(v) for v in fu[0, 0]), 6, (255, 255, 0), -1)
            t_img = cv2.resize(img, (320, 240))
            cv2.putText(t_img, f"{t - ic:+d}", (6, 232), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            tiles.append(t_img)
        row = np.concatenate(tiles, 1)
        cv2.putText(row, f"ep {ep} sim {r['outcome']} first {r['first_shape']} at close{r['first_tick_rel_close']}",
                    (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        rows.append(row)
    p = d / "overlay_real_wrist.jpg"
    cv2.imwrite(str(p), np.concatenate(rows, 0)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 80])
    print("wrote", p, flush=True)


if __name__ == "__main__":
    main()
