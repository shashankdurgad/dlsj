"""Measure appearance parameters from the REAL recorded frames (plain Python, no Isaac).

* plates: segment orange / yellow / blue(="purple") plates in top frames by HSV, fit ellipses, back-project the outer
  contour to the plate-rim plane with the top camera model and fit a circle -> diameter; median plate colour;
* bricks: small saturated blobs on the table in top frames, clustered into colour classes -> median colours;
* jaws: red pixels in the bottom half of wrist frames (the fingers are always there) -> median colour;
  and the same for the top camera (red near the projected gripper is too small to segment reliably; skipped);
* robot body: dark pixels of the robot base region of the top median.

    docker run ... python sim/twin/visuals/measure_real.py [--every 8] [--rim-z 0.018]
    docker run ... python sim/twin/visuals/measure_real.py --ccm    # per-camera colour transforms (see ccm_main)

Writes sim/twin/out/visuals/real_appearance.json (+ debug overlays). Colours are sRGB 0-255 as the camera sees them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from sim.twin.visuals.table_texture import TABLE_TOP_Z, backproject, camera_from_profile  # noqa: E402

OUT = REPO / "sim/twin/out/visuals"

# OpenCV HSV (H 0-180)
PLATE_HSV = {
    "orange": ((4, 170, 120), (16, 255, 255)),
    "yellow": ((33, 90, 120), (48, 255, 255)),  # lime-ish yellow in the top camera
    "purple": ((90, 90, 120), (118, 255, 255)),  # lavender plate reads sky-blue in the top camera
}
BRICK_CLASSES = {  # name: (hsv lo, hsv hi)
    "light_blue": ((92, 120, 120), (112, 255, 255)),
    "yellow": ((24, 120, 130), (40, 255, 255)),
    "orange": ((7, 180, 150), (18, 255, 255)),
    "red": ((0, 170, 80), (5, 255, 255)),
    "red2": ((172, 170, 80), (180, 255, 255)),
    "brown": ((3, 120, 50), (18, 255, 140)),
    "cream": ((15, 45, 190), (35, 130, 255)),
}


def fit_circle(xy: np.ndarray):
    A = np.c_[2 * xy, np.ones(len(xy))]
    b = (xy ** 2).sum(1)
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    r = np.sqrt(c[2] + c[0] ** 2 + c[1] ** 2)
    return c[:2], r


def persistent_mask(median_rgb: np.ndarray | None = None) -> np.ndarray:
    """Pixels of persistent saturated / dark things in the top view (tripod clips, cables, robot base, background):
    from the per-pixel median of many episodes (table_texture.py). Brick detection ignores them."""
    if median_rgb is None:
        m = cv2.imread(str(OUT / "top_median.png"))
        if m is None:
            return np.zeros((480, 640), bool)
        median_rgb = m[..., ::-1]
    hsv = cv2.cvtColor(median_rgb, cv2.COLOR_RGB2HSV)
    bad = (hsv[..., 1] > 90) | (hsv[..., 2] < 60)
    bad[:45] = True  # beyond the table edge
    return cv2.dilate(bad.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0


def detect_layout(top: np.ndarray, r_cv, pos, K, rim_z: float = 0.018, brick_top_z: float = 0.022,
                  ignore: np.ndarray | None = None) -> dict:
    """Rough scene layout from one real top frame (env frame metres): plates {colour: (x, y, diameter)} and bricks
    [(colour class, x, y, yaw_rad, '2x2' | '2x4')]. Plates: largest blob per plate colour (outer contour back-projected
    to the rim plane, circle fit). Bricks: small saturated blobs back-projected to the brick-top plane (minAreaRect)."""
    hsv = cv2.cvtColor(top, cv2.COLOR_RGB2HSV)
    plates, plate_masks = {}, np.zeros(top.shape[:2], np.uint8)
    for name, (lo, hi) in PLATE_HSV.items():
        m = cv2.inRange(hsv, lo, hi)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(m)
        if n < 2:
            continue
        i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        if st[i, cv2.CC_STAT_AREA] < 4000:
            continue
        comp = (lab == i).astype(np.uint8)
        cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cs, key=cv2.contourArea)[:, 0, :].astype(float)
        inside = (c[:, 0] > 2) & (c[:, 0] < 637) & (c[:, 1] > 2) & (c[:, 1] < 477)  # drop image-border points
        P = backproject(c[inside], r_cv, pos, K, TABLE_TOP_Z + rim_z)
        if len(P) < 30:
            continue
        ctr, r = fit_circle(P[:, :2])
        plates[name] = (float(ctr[0]), float(ctr[1]), float(2 * r))
        cv2.drawContours(plate_masks, [c.astype(np.int32)[:, None]], -1, 1, -1)
    bricks = []
    for name, (lo, hi) in BRICK_CLASSES.items():
        m = cv2.inRange(hsv, lo, hi)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(m)
        for i in range(1, n):
            ar = st[i, cv2.CC_STAT_AREA]
            if not (150 < ar < 3000) or st[i, 1] < 50:
                continue
            ys, xs = np.nonzero(lab == i)
            if ignore is not None and ignore[ys, xs].mean() > 0.3:
                continue
            on_plate = plate_masks[ys, xs].mean() > 0.5
            if name in ("yellow", "orange") and on_plate and ar > 2000:
                continue
            z = TABLE_TOP_Z + brick_top_z + (0.006 if on_plate else 0.0)
            cs, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            c = max(cs, key=cv2.contourArea)[:, 0, :].astype(float)
            P = backproject(c, r_cv, pos, K, z)[:, :2].astype(np.float32)
            (cx, cy), (w, h), ang = cv2.minAreaRect(P)
            long_, short = max(w, h), min(w, h)
            if short < 0.018 or long_ > 0.085:
                continue
            kind = "2x4" if long_ > 0.05 and long_ / max(short, 1e-6) > 1.45 else "2x2"
            yaw = np.deg2rad(ang) + (np.pi / 2 if h > w else 0.0)
            bricks.append(dict(color=name.rstrip("2"), x=float(cx), y=float(cy), yaw=float(yaw), kind=kind,
                               on_plate=bool(on_plate), area=int(ar)))
    return dict(plates=plates, bricks=bricks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=6)
    ap.add_argument("--rim-z", type=float, default=0.018, help="plate rim height above the table (m)")
    ap.add_argument("--profile", default="joe_cal2")
    a = ap.parse_args()
    from sim.twin.visuals.realdata import Dataset

    ds = Dataset()
    r_cv, pos, K = camera_from_profile(a.profile)
    plates = {k: {"diam": [], "rgb": []} for k in PLATE_HSV}
    bricks = {k: [] for k in BRICK_CLASSES}
    jaw, dbg_done = [], 0
    ignore = persistent_mask()
    for ep in range(0, len(ds), a.every):
        top = ds.frame(ep, "top", 0)
        hsv = cv2.cvtColor(top, cv2.COLOR_RGB2HSV)
        dbg = top.copy()
        for name, (lo, hi) in PLATE_HSV.items():
            m = cv2.inRange(hsv, lo, hi)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            n, lab, st, _ = cv2.connectedComponentsWithStats(m)
            if n < 2:
                continue
            i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
            if st[i, cv2.CC_STAT_AREA] < 6000:
                continue
            x, y, w, h = st[i, :4]
            if x <= 1 or y <= 1 or x + w >= 639 or y + h >= 479:  # cut by the image border
                continue
            comp = (lab == i).astype(np.uint8)
            # fill holes (bricks on the plate) before taking the outer contour
            cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            c = max(cs, key=cv2.contourArea)[:, 0, :].astype(float)
            if len(c) < 50:
                continue
            filled = np.zeros_like(comp)
            cv2.drawContours(filled, [c.astype(np.int32)[:, None]], -1, 1, -1)
            hull = cv2.convexHull(c.astype(np.float32))[:, 0, :]
            solidity = cv2.contourArea(c.astype(np.float32)) / max(cv2.contourArea(hull), 1)
            if solidity < 0.93:  # occluded by the arm / merged with a brick
                continue
            P = backproject(c, r_cv, pos, K, TABLE_TOP_Z + a.rim_z)
            ctr, r = fit_circle(P[:, :2])
            resid = np.abs(np.linalg.norm(P[:, :2] - ctr, axis=1) - r)
            if np.median(resid) > 0.004:
                continue
            plates[name]["diam"].append(2 * r)
            inner = cv2.erode(comp, np.ones((15, 15), np.uint8)).astype(bool) & (hsv[..., 2] < 245)
            plates[name]["rgb"].append(np.median(top[inner], 0).tolist())
            cv2.drawContours(dbg, [c.astype(np.int32)[:, None]], -1, (255, 255, 255), 1)
        # bricks: small blobs outside the plates
        for name, (lo, hi) in BRICK_CLASSES.items():
            m = cv2.inRange(hsv, lo, hi)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            n, lab, st, _ = cv2.connectedComponentsWithStats(m)
            for i in range(1, n):
                ar = st[i, cv2.CC_STAT_AREA]
                w, h = st[i, 2], st[i, 3]
                if 120 < ar < 2500 and max(w, h) < 3.2 * min(w, h) and st[i, 1] > 60 and \
                        ignore[lab == i].mean() < 0.3:
                    px = top[lab == i]
                    bricks[name.rstrip("2")].append(np.median(px, 0).tolist())
        wr = ds.frame(ep, "wrist", 0)
        whsv = cv2.cvtColor(wr, cv2.COLOR_RGB2HSV)
        lower = np.zeros(whsv.shape[:2], bool)
        lower[300:, :] = True
        red = (((whsv[..., 0] < 6) | (whsv[..., 0] > 170)) & (whsv[..., 1] > 150) & (whsv[..., 2] > 80)) & lower
        if red.sum() > 3000:
            jaw.append(np.median(wr[red], 0).tolist())
        if dbg_done < 6:
            cv2.imwrite(str(OUT / f"measure_plates_{ep:03d}.jpg"), dbg[..., ::-1])
            dbg_done += 1

    med = cv2.imread(str(OUT / "top_median.png"))
    body = None
    if med is not None:
        reg = med[45:100, 330:420][..., ::-1].reshape(-1, 3)
        body = np.median(reg[reg.sum(1) < 150], 0).tolist()
        table = np.median(med[150:450, 50:550][..., ::-1].reshape(-1, 3), 0).tolist()
    res = {"rim_z_m": a.rim_z, "profile": a.profile, "plates": {}, "bricks": {}, "jaw_wrist_cam_rgb": None,
           "robot_body_top_rgb": body, "table_top_median_rgb": table if med is not None else None}
    for k, v in plates.items():
        d = np.array(v["diam"])
        if len(d):
            res["plates"][k] = dict(n=len(d), diameter_m_median=float(np.median(d)),
                                    diameter_m_p25_p75=np.percentile(d, [25, 75]).tolist(),
                                    rgb_median=np.median(np.array(v["rgb"]), 0).tolist())
    for k, v in bricks.items():
        if len(v) >= 3:
            res["bricks"][k] = dict(n=len(v), rgb_median=np.median(np.array(v), 0).tolist())
    if jaw:
        res["jaw_wrist_cam_rgb"] = np.median(np.array(jaw), 0).tolist()
    (OUT / "real_appearance.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))




# ------------------------------------------------------------------ per-camera colour transform (top -> wrist/inspect)
CAM_CLASS_HSV = {  # per camera: class -> (hsv lo, hsv hi); the same physical objects look different per camera
    "wrist": {"purple": ((100, 35, 140), (135, 200, 255)), "yellow": ((24, 55, 140), (45, 255, 255)),
              "orange": ((3, 130, 120), (18, 255, 255)), "table": ((0, 5, 110), (30, 60, 235))},
    "inspect": {"purple": ((100, 25, 100), (140, 200, 255)), "yellow": ((18, 150, 110), (34, 255, 255)),
                "orange": ((3, 150, 110), (16, 255, 255)), "table": ((5, 40, 90), (25, 150, 230))},
}


def class_colors(img: np.ndarray, classes: dict, min_area: int = 2500, rows=None) -> dict:
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    out = {}
    for name, (lo, hi) in classes.items():
        m = cv2.inRange(hsv, lo, hi)
        if rows is not None:
            m[:rows[0]] = 0
            m[rows[1]:] = 0
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        if name == "table":
            if m.sum() / 255 > 8000:
                out[name] = np.median(img[m > 0], 0)
            continue
        n, lab, st, _ = cv2.connectedComponentsWithStats(m)
        if n < 2:
            continue
        i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        if st[i, cv2.CC_STAT_AREA] >= min_area:
            core = cv2.erode((lab == i).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
            if core.sum() > 200:
                out[name] = np.median(img[core], 0)
    return out


def _lin(c):
    c = np.asarray(c, float) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def fit_ccm(top_rgb: dict, cam_rgb: dict) -> tuple[np.ndarray, dict]:
    """3x3 M (linear RGB) with cam ~= M @ top, least squares over the shared classes."""
    keys = [k for k in top_rgb if k in cam_rgb]
    A = np.array([_lin(top_rgb[k]) for k in keys])
    B = np.array([_lin(cam_rgb[k]) for k in keys])
    M, *_ = np.linalg.lstsq(A, B, rcond=None)
    M = M.T
    res = {k: (np.clip(M @ _lin(top_rgb[k]), 0, 1) ** (1 / 2.2) * 255).round(1).tolist() for k in keys}
    return M, res


def ccm_main(every: int = 6):
    from sim.twin.visuals.realdata import Dataset
    from sim.twin.visuals.twin_scene import PLATE_RGB

    ds = Dataset()
    med = cv2.imread(str(OUT / "top_median.png"))[..., ::-1]
    top = dict(PLATE_RGB)
    top["table"] = np.median(med[150:450, 50:550].reshape(-1, 3), 0)
    result = {}
    for cam, classes in CAM_CLASS_HSV.items():
        acc = {k: [] for k in classes}
        for ep in range(0, len(ds), every):
            n = int(ds.eps.loc[ep, "length"])
            for f in (0, n // 2):
                img = ds.frame(ep, cam, f)
                rows = (240, 480) if cam == "inspect" else (0, 330)  # inspect: table half; wrist: above the jaws
                for k, v in class_colors(img, classes, rows=rows).items():
                    acc[k].append(v)
        real = {k: np.median(np.array(v), 0).tolist() for k, v in acc.items() if len(v) >= 5}
        M, fitted = fit_ccm(top, real)
        result[cam] = dict(M_linear=M.tolist(), real_class_rgb=real, n={k: len(v) for k, v in acc.items()},
                           top_class_rgb={k: list(map(float, v)) for k, v in top.items()}, fitted_rgb=fitted)
        print(cam, json.dumps(result[cam], indent=1), flush=True)
    from sim.twin.visuals.twin_scene import CAMERA_COLOR_JSON

    CAMERA_COLOR_JSON.write_text(json.dumps(result, indent=1))
    print("wrote", CAMERA_COLOR_JSON)


if __name__ == "__main__":
    if "--ccm" in sys.argv:
        ccm_main()
    else:
        main()
