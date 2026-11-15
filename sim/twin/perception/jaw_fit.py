"""Fit the jaw (moving fin) angle seen by the REAL wrist camera vs the gripper reading.

    python sim/twin/perception/jaw_fit.py        (plain python; real_frames from real_detect.py)
    -> sim/twin/out/perception/jaw_fit.json, jaw_fit.jpg

The jaw fin is rigid in the camera frame except for its joint angle q, so its image silhouette is a function of q
only. Per real frame: red pixels (the TPU fins, HSV) in the image's right part (the fixed fin is on the left) vs the
projected CAD jaw-fin silhouette (wrist mount + measured intrinsics) for q on a grid; best IoU -> q_real. Then
q_real = a * reading + b (robust line). Compared with the twin map of the time, q = 0.02308 u + 0.105
(physics now uses bias -0.10).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from sim.twin.perception import geometry as G  # noqa: E402

OUT = REPO / "sim/twin/out/perception"


def red_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 0] <= 8) | (hsv[..., 0] >= 170)) & (hsv[..., 1] >= 120) & (hsv[..., 2] >= 70)
    return cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)).astype(bool)


def jaw_sil(q, cam, V, F):
    c, s = np.cos(q), np.sin(q)
    rot = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]]) @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    P = ((V @ rot.T + G.JAW_XYZ) - cam["t"]) @ cam["R"].T
    uv = np.stack([cam["fx"] * P[:, 0] / P[:, 2] + cam["cx"], cam["fy"] * P[:, 1] / P[:, 2] + cam["cy"]], 1)
    m = np.zeros((G.H, G.W), np.uint8)
    for t in F:
        if (P[t, 2] > 0.003).all():
            cv2.fillConvexPoly(m, np.round(uv[t] * 4).astype(np.int32), 1, shift=2)
    return m.astype(bool)


def main():
    from sim.twin.align import frames as F_

    cam = G.wrist_camera("physical")
    g = np.load(G.FINRAY)
    V, F = g["jaw_fin_v"].astype(float), g["jaw_fin_f"][::2]
    qs = np.linspace(0.05, 1.0, 39)
    sils = {round(q, 4): jaw_sil(q, cam, V, F) for q in qs}
    right = np.zeros((G.H, G.W), bool)
    right[:, 330:] = True
    rows = []
    for d in sorted((OUT / "real_frames").glob("ep*")):
        ep = int(d.name[2:])
        st = F_.states(ep)
        files = sorted(d.glob("f*.jpg"))[::3]
        for f in files:
            fr = int(f.stem[1:])
            u = float(st[fr, 5])
            mr = red_mask(cv2.imread(str(f))) & right
            if mr.sum() < 500:
                continue
            ious = [
                ((sils[round(q, 4)] & right & mr).sum() / max(((sils[round(q, 4)] & right) | mr).sum(), 1)) for q in qs
            ]
            k = int(np.argmax(ious))
            rows.append((ep, fr, u, float(qs[k]), float(ious[k])))
    a = np.array([r[2:] for r in rows])
    good = a[:, 2] > 0.4
    x, y = a[good, 0], a[good, 1]
    A = np.stack([x, np.ones_like(x)], 1)
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    for _ in range(3):  # trim outliers
        r = y - A @ coef
        keep = np.abs(r) < 2.5 * np.median(np.abs(r)) + 1e-6
        coef = np.linalg.lstsq(A[keep], y[keep], rcond=None)[0]
    bins = []
    for lo, hi in ((0, 8), (8, 14), (14, 20), (20, 27), (27, 40)):
        k = good & (a[:, 0] >= lo) & (a[:, 0] < hi)
        if k.sum() >= 5:
            bins.append(
                {
                    "reading": [lo, hi],
                    "n": int(k.sum()),
                    "q_real_median": round(float(np.median(a[k, 1])), 3),
                    "q_twin": round(float(G.GRIP_SCALE * np.median(a[k, 0]) + G.GRIP_BIAS), 3),
                    "iou_median": round(float(np.median(a[k, 2])), 3),
                }
            )
    res = {
        "n_frames": int(good.sum()),
        "fit_q_real": {"scale": round(float(coef[0]), 5), "bias": round(float(coef[1]), 4)},
        "twin": {"scale": G.GRIP_SCALE, "bias": G.GRIP_BIAS},
        "bins": bins,
    }
    (OUT / "jaw_fit.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
