"""Fin silhouette alignment real vs sim wrist (CPU): red bottom-connected masks, IoU and per-side centroid offsets.

    python sim/twin/wrist_fit/fin_align.py --sim <run>/sim.npz [--phases reach,pregrasp]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from sim.twin.wrist_fit.edits import BITS, real_fins  # noqa: E402

SET = REPO / "sim/twin/out/audit/set"


def sides(m):
    out = {}
    W = m.shape[1]
    for k, sl in (("left", slice(0, W // 2)), ("right", slice(W // 2, W))):
        ys, xs = np.nonzero(m[:, sl])
        out[k] = (float(xs.mean() + sl.start), float(ys.mean()), int(len(xs))) if len(xs) > 200 else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True)
    ap.add_argument("--phases", default="reach,pregrasp")
    a = ap.parse_args()
    S = json.loads((SET / "set.json").read_text())["samples"]
    idx = [i for i, s in enumerate(S) if s["phase"] in a.phases.split(",")]
    real = np.load(SET / "real.npz")["wrist"]
    z = np.load(a.sim)
    full = z["wrist"].shape[0] == len(S)
    rows = []
    for n, i in enumerate(idx):
        j = i if full else n
        rm = real_fins(real[i])
        sm = real_fins(z["wrist"][j]) | ((z["mask_wrist"][j] & BITS["robot"]) > 0) & real_fins(z["wrist"][j])
        iou = float((rm & sm).sum() / max((rm | sm).sum(), 1))
        rs, ss = sides(rm), sides(sm)
        d = {k: (np.round(np.subtract(ss[k][:2], rs[k][:2]), 0).tolist() if rs[k] and ss[k] else None)
             for k in ("left", "right")}
        rows.append(dict(i=i, iou=round(iou, 3), d=d, real_px={k: rs[k] and rs[k][2] for k in rs},
                         sim_px={k: ss[k] and ss[k][2] for k in ss}))
        print(i, S[i]["phase"], "IoU", round(iou, 3), "sim-real centroid", d, "px", rows[-1]["real_px"],
              rows[-1]["sim_px"])
    ious = [r["iou"] for r in rows]
    dl = np.array([r["d"]["left"] for r in rows if r["d"]["left"]])
    dr = np.array([r["d"]["right"] for r in rows if r["d"]["right"]])
    print("FIN_ALIGN median IoU", np.median(ious), "left fin d (x,y) median", np.median(dl, 0) if len(dl) else None,
          "right fin d median", np.median(dr, 0) if len(dr) else None, "n right", len(dr))


if __name__ == "__main__":
    main()
