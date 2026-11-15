"""Where does the HELD brick sit in the gripper, seen from the real wrist camera? (For the twin's held-brick look.)

    python sim/twin/perception/held_brick.py [--pose x,y,z ...]    (plain python; after real_detect.py / gt_real.py)
    -> sim/twin/out/perception/held_brick.json, held_brick.jpg

Real held frames: 6 .. 20 frames after the gripper closed (gripper reading < 12), the real detector's best box
(conf >= 0.3) nearest the brick. A 2x2 Duplo body (31.8 x 31.8 x 19.2 mm, studs ignored), axes aligned with
gripper_link, is placed at a centre p in gripper_link; its geometric VISIBLE box (wrist mount + measured intrinsics,
fins / bases occluding, image-clipped) is compared with the real boxes. p is fitted (median |edge error| over the
left / right / top edges; the bottom edge is usually the image border) on episodes whose wrist fit says 2x2.
Reported: the fitted p, and for candidate poses (e.g. the twin's held pose) the projected width / height / centre
relative to the real boxes, plus the size sensitivity to depth along the camera axis.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from sim.twin.perception import geometry as G  # noqa: E402
from sim.twin.perception.gt_real import iou  # noqa: E402

OUT = REPO / "sim/twin/out/perception"
SIZE = np.array([0.0318, 0.0318, 0.0192])


def boxes_at(geo, J, p_local):
    """Visible boxes (n, 4) of a 2x2 body centred at p_local (gripper_link), axes = gripper_link axes."""
    M = geo.gripper_pose(J)
    pos = (M[:, :3, :3] @ np.asarray(p_local, float)) + M[:, :3, 3]
    v = geo.visibility(J, pos, M[:, :3, :3], np.tile(SIZE, (len(J), 1)))
    b = v["box_vis"]
    return np.stack(
        [np.clip(b[:, 0], 0, 640), np.clip(b[:, 1], 0, 480), np.clip(b[:, 2], 0, 640), np.clip(b[:, 3], 0, 480)], 1
    ), v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", action="append", default=[], help="name=x,y,z (mm, gripper_link) to evaluate")
    a = ap.parse_args()
    from sim.twin.align import frames as F

    fits = json.loads((OUT / "real_fit.json").read_text())["episodes"]
    geo = G.Geometry()
    eps = [int(e) for e, f in fits.items() if f.get("src") == "wrist" and f["kind"] == "2x2"]
    J, R, E, FR = [], [], [], []
    for e in eps:
        rec = json.loads((OUT / f"real_dets/ep{e:03d}.json").read_text())
        st = F.states(e)
        cf = rec["cf"]
        for f, dets in zip(rec["frames"], rec["dets"], strict=True):
            if not 6 <= f - cf <= 20 or st[f, 5] >= 12:
                continue
            d = np.array(dets, float).reshape(-1, 5)
            d = d[d[:, 4] >= 0.3]
            if not len(d):
                continue
            J.append(st[f])
            R.append(d)
            E.append(e)
            FR.append(f)
    J = np.array(J)
    # first guess: the finray grasp point; pick each frame's box by IoU with it, refit, re-pick
    p0 = np.array([0.0063, 0.001, -0.0902])

    def pick(p):
        bp, _ = boxes_at(geo, J, p)
        return np.array([r[np.argmax(iou(r[:, :4], bp[i][None]))][:4] for i, r in enumerate(R)]), bp

    def cost(p, real):
        bp, _ = boxes_at(geo, J, p)
        e = np.abs(np.stack([bp[:, 0] - real[:, 0], bp[:, 2] - real[:, 2], bp[:, 1] - real[:, 1]], 1))
        return float(np.median(np.nan_to_num(e, nan=200.0)))

    p = p0
    for _ in range(3):
        real, _ = pick(p)
        r = minimize(
            cost,
            p,
            args=(real,),
            method="Nelder-Mead",
            options={
                "initial_simplex": [p, p + [0.006, 0, 0], p + [0, 0.006, 0], p + [0, 0, 0.006]],
                "xatol": 2e-4,
                "fatol": 0.05,
                "maxiter": 300,
            },
        )
        p = r.x
    real, _ = pick(p)
    keep = iou(real, boxes_at(geo, J, p)[0]) > 0.3
    rw, rh = real[:, 2] - real[:, 0], real[:, 3] - real[:, 1]
    rcx = (real[:, 0] + real[:, 2]) / 2

    def describe(q):
        bp, v = boxes_at(geo, J, q)
        w, h = bp[:, 2] - bp[:, 0], bp[:, 3] - bp[:, 1]
        k = keep & np.isfinite(w)
        return {
            "p_mm": (np.asarray(q) * 1e3).round(1).tolist(),
            "width_ratio_sim_over_real_median": round(float(np.median(w[k] / rw[k])), 3),
            "height_ratio_median": round(float(np.median(h[k] / rh[k])), 3),
            "centre_x_shift_px_median": round(float(np.median((bp[k, 0] + bp[k, 2]) / 2 - rcx[k])), 1),
            "top_edge_shift_px_median": round(float(np.median(bp[k, 1] - real[k, 1])), 1),
            "median_edge_err_px": round(cost(q, real), 1),
            "cam_depth_mm_median": round(float(np.median(v["z"][k])) * 1e3, 1),
            "vis_median": round(float(np.median(v["vis"][k])), 3),
        }

    res = {
        "n_frames": int(keep.sum()),
        "n_episodes": len(set(np.array(E)[keep].tolist())),
        "fitted": describe(p),
        "finray_grasp_point_15mm": describe(p0),
    }
    tip = np.array([-0.00821, 0.00097, -0.10522])
    res["fitted"]["height_above_fixed_fin_tip_mm"] = round(float(p[2] - tip[2]) * 1e3 - SIZE[2] * 500, 1)
    for s in a.pose:
        name, v = s.split("=")
        res[name] = describe(np.array([float(x) for x in v.split(",")]) / 1e3)
    # sensitivity: move the fitted pose 6 mm along the camera's optical axis (away from the camera)
    ax = geo.cam["R"][2]
    res["fitted_plus_6mm_away_from_camera"] = describe(p + 0.006 * ax)
    res["fitted_shift_11mm_minus_x"] = describe(p - np.array([0.011, 0, 0]))
    (OUT / "held_brick.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    # overlay: a few frames, real box (orange) vs fitted (green) vs finray point (magenta)
    import cv2

    tiles = []
    for i in np.where(keep)[0][:: max(1, keep.sum() // 8)][:8]:
        img = cv2.imread(str(OUT / f"real_frames/ep{E[i]:03d}/f{FR[i]:04d}.jpg"))
        if img is None:
            continue
        for q, col in ((p, (0, 255, 0)), (p0, (255, 0, 255))):
            b = boxes_at(geo, J[i : i + 1], q)[0][0]
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, 2)
        b = real[i]
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 165, 255), 2)
        tiles.append(cv2.resize(img, (320, 240)))
    if tiles:
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.concatenate([np.concatenate(tiles[k : k + 4], 1) for k in range(0, len(tiles), 4)], 0)
        cv2.imwrite(str(OUT / "held_brick.jpg"), sheet)


if __name__ == "__main__":
    main()
