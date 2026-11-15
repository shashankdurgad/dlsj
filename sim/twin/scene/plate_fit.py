"""Fit the plate profile (visuals/assets.plate_profile) to the real inspect-camera silhouettes (plain python).

    python sim/twin/scene/plate_fit.py [--every 3] [--max_eps 60]     # -> sim/twin/out/scene/plate_fit.json + jpgs

The inspect camera (measured pose, factory intrinsics) sees the plates nearly side-on from 10 cm above the table:
the silhouette is the convex hull of the rim ellipse and the wall down to the foot, so its height-to-width ratio
fixes the rim height and the wall shape for a known radius. Per plate (pose from layout.layout_from_episode on the
first top frame): the real mask is GrabCut-segmented around the predicted silhouette (its core seeds the plate, a
ring around it the table), holes (bricks, highlights) filled. The model silhouette gets a per-plate
2D shift and isotropic scale (absorbs the inspect camera's pose drift between sessions and the plate-pose error of
the top-camera layout; the scale keeps the shape ratios), and the profile parameters (rim_h, floor_frac, wall_pow)
are fitted to maximise the mean IoU on 2/3 of the episodes; the rest is held out. The radius comes from the top
camera (layout plate diameters, outer rim at 21 mm).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
OUT = REPO / "sim/twin/out/scene"
TABLE_Z = -0.002


def inspect_camera():
    """Real inspect camera: (R_world_to_cv, position, K) from calib/cameras_measured.json."""
    m = json.loads((REPO / "sim/twin/calib/cameras_measured.json").read_text())["inspect"]
    K = np.array([[m["fx"], 0, m["cx"]], [0, m["fy"], m["cy"]], [0, 0, 1.0]])
    return np.asarray(m["world_to_cv"], float), np.asarray(m["position"], float), K


def silhouette_poly(xy, radius, params, cam, seg: int = 72):
    """Image polygon (convex hull) of a plate at table position xy with profile params."""
    import cv2

    from sim.twin.visuals.assets import plate_profile

    prof = plate_profile(radius, ring_amp=0.0, n=60, **params)
    R, pos, K = cam
    a = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    # plate frame origin 3 mm above the table (floor top); profile z relative to it
    P = np.stack([xy[0] + prof[:, :1] * np.cos(a), xy[1] + prof[:, :1] * np.sin(a),
                  np.broadcast_to(TABLE_Z + 0.003 + prof[:, 1:2], (len(prof), seg))], -1).reshape(-1, 3)
    pc = (P - pos) @ R.T
    pc = pc[pc[:, 2] > 0.02]
    uv = (pc[:, :2] / pc[:, 2:3]) @ K[:2, :2].T + K[:2, 2]
    return cv2.convexHull(uv.astype(np.float32))[:, 0, :]


def legacy_profile_params():
    """The previous twin plate (before this fit): 23 mm rim, floor to 0.75 r, cosine-like wall (wall_pow ~1.3)."""
    return {"rim_h": 0.023, "floor_frac": 0.75, "wall_pow": 1.3}


def real_mask(img, poly, grow: int = 30):
    """Real plate region around the predicted silhouette (GrabCut seeded by it: the core is plate, a ring around it
    is probably table; the shaded outer wall and the rim highlights stay with the plate), holes filled, or None."""
    import cv2

    H, W = img.shape[:2]
    pred = np.zeros((H, W), np.uint8)
    cv2.fillPoly(pred, [poly.astype(np.int32)], 1)
    core = cv2.erode(pred, np.ones((21, 21), np.uint8)) > 0
    if core.sum() < 200:
        return None
    gc = np.full((H, W), cv2.GC_BGD, np.uint8)
    gc[cv2.dilate(pred, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)) > 0] = cv2.GC_PR_BGD
    gc[pred > 0] = cv2.GC_PR_FGD
    gc[core] = cv2.GC_FGD
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(np.ascontiguousarray(img[..., ::-1]), gc, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
    m = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lb, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    cs, _ = cv2.findContours((lb == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros((H, W), np.uint8)
    cv2.drawContours(out, cs, -1, 1, -1)
    x, y, w, h = cv2.boundingRect(max(cs, key=cv2.contourArea))
    if x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1:  # cut by the image border
        return None
    return out.astype(bool)


def iou_poly(poly, mask, du=0.0, dv=0.0, s=1.0):
    import cv2

    c = poly.mean(0)
    p = (poly - c) * s + c + [du, dv]
    m = np.zeros(mask.shape, np.uint8)
    cv2.fillPoly(m, [np.round(p).astype(np.int32)], 1)
    m = m > 0
    return float((m & mask).sum() / max((m | mask).sum(), 1))


def best_iou(poly, mask):
    """Max IoU over the per-plate nuisance (shift, scale)."""
    from scipy.optimize import minimize

    r = minimize(lambda x: -iou_poly(poly, mask, *x), np.array([0.0, 0.0, 1.0]), method="Nelder-Mead",
                 options={"xatol": 0.05, "fatol": 1e-4, "maxiter": 120, "initial_simplex": np.array(
                     [[0, 0, 1.0], [6, 0, 1.0], [0, 6, 1.0], [0, 0, 1.06]])})
    return -float(r.fun), r.x


def collect(eps, root, cam):
    from sim.twin.scene.layout import layout_from_episode
    from sim.twin.visuals.realdata import Dataset

    ds = Dataset(str(root))
    items = []
    for ep in eps:
        lay = layout_from_episode(ep, root)
        if not lay["plates"]:
            continue
        img = ds.frame(ep, "inspect", 0)
        for name, p in lay["plates"].items():
            xy, r = (p["x"], p["y"]), p["diam"] / 2
            poly = silhouette_poly(xy, r, legacy_profile_params(), cam)
            m = real_mask(img, poly)
            if m is None:
                continue
            items.append({"ep": ep, "plate": name, "xy": xy, "r": r, "mask": m, "img": img})
    return items


def mean_loss(items, params, cam):
    return float(np.mean([1 - best_iou(silhouette_poly(it["xy"], it["r"], params, cam), it["mask"])[0]
                          for it in items]))


def main():
    from scipy.optimize import minimize

    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--max_eps", type=int, default=60)
    ap.add_argument("--root", default=str(REPO / "sim/twin/data/canonical/duplo_teleop_cal2_rgb_456"))
    a = ap.parse_args()
    cam = inspect_camera()
    from sim.twin.visuals.realdata import Dataset

    n = len(Dataset(a.root))
    eps = list(range(0, n, a.every))[: a.max_eps]
    items = collect(eps, Path(a.root), cam)
    # drop plates whose real mask does not look like a plate even at the best legacy fit (occluded / merged)
    good = []
    for it in items:
        iou, _ = best_iou(silhouette_poly(it["xy"], it["r"], legacy_profile_params(), cam), it["mask"])
        it["iou_legacy"] = iou
        if iou > 0.6:
            good.append(it)
    eps_used = sorted({it["ep"] for it in good})
    test_eps = set(eps_used[2::3])
    train = [it for it in good if it["ep"] not in test_eps]
    test = [it for it in good if it["ep"] in test_eps]
    print(f"plates: {len(items)} candidates, {len(good)} usable, train {len(train)}, test {len(test)}", flush=True)
    keys = ("rim_h", "floor_frac", "wall_pow")
    lo, hi = np.array([0.014, 0.5, 1.0]), np.array([0.032, 0.88, 3.0])

    def f(x):
        x = np.clip(x, lo, hi)
        return mean_loss(train, dict(zip(keys, x, strict=True)), cam)

    x0 = np.array([0.023, 0.75, 1.6])
    r = minimize(f, x0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-5, "maxiter": 120,
                                                        "initial_simplex": np.array([x0, x0 + [0.004, 0, 0],
                                                                                     x0 + [0, 0.08, 0],
                                                                                     x0 + [0, 0, 0.6]])})
    fit = dict(zip(keys, np.clip(r.x, lo, hi).round(4).tolist(), strict=True))
    legacy = legacy_profile_params()
    res = {"fit": fit, "legacy": legacy, "n_train": len(train), "n_test": len(test),
           "train_iou": {"legacy": round(1 - mean_loss(train, legacy, cam), 4), "fit": round(1 - r.fun, 4)},
           "test_iou": {"legacy": round(1 - mean_loss(test, legacy, cam), 4),
                        "fit": round(1 - mean_loss(test, fit, cam), 4)},
           "test_episodes": sorted(test_eps), "radius_m_median_top_camera": round(float(np.median(
               [it["r"] for it in good])), 4)}
    print(json.dumps(res, indent=1), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plate_fit.json").write_text(json.dumps(res, indent=1))
    overlay(test[:8], legacy, fit, cam, OUT / "plate_fit_inspect.jpg")


def overlay(items, legacy, fit, cam, path):
    """Real inspect crops with the legacy (red) and fitted (green) silhouettes at their best nuisance."""
    import cv2

    tiles = []
    for it in items:
        img = np.ascontiguousarray(it["img"].copy())
        for params, col in ((legacy, (255, 60, 60)), (fit, (60, 255, 60))):
            poly = silhouette_poly(it["xy"], it["r"], params, cam)
            iou, (du, dv, s) = best_iou(poly, it["mask"])
            c = poly.mean(0)
            p = (poly - c) * s + c + [du, dv]
            cv2.polylines(img, [np.round(p).astype(np.int32)], True, col, 1)
        ys, xs = np.nonzero(it["mask"])
        x0, x1, y0, y1 = xs.min() - 15, xs.max() + 15, ys.min() - 15, ys.max() + 15
        crop = img[max(y0, 0):y1, max(x0, 0):x1]
        crop = cv2.resize(crop, (360, int(360 * crop.shape[0] / max(crop.shape[1], 1))))
        cv2.putText(crop, f"ep {it['ep']} {it['plate']}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tiles.append(crop)
    if not tiles:
        return
    h = max(t.shape[0] for t in tiles)
    tiles = [np.pad(t, ((0, h - t.shape[0]), (0, 0), (0, 0))) for t in tiles]
    rows = [np.concatenate(tiles[i:i + 4] + [np.zeros_like(tiles[0])] * (4 - len(tiles[i:i + 4])), 1)
            for i in range(0, len(tiles), 4)]
    cv2.imwrite(str(path), np.concatenate(rows, 0)[..., ::-1])


if __name__ == "__main__":
    main()
