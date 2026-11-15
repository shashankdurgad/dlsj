"""Rebuild the table's wood texture at metric scale from the phone photo + the real top camera (plain Python).

    docker run ... python sim/twin/visuals/wood_texture.py [--mm 0.5] [--tile-px 2048] [--lf-mm 8] [--seed 0]

Why: the old texture was the top camera's median projected on the table plane (~1.2 mm / px, softer after video
compression) and mirror-filled outside the camera's footprint; the wrist and inspect cameras look at it from a few
centimetres / at grazing angles and saw smeared, mirrored grain.

What this does
1. top-camera median -> table plane (env frame, the PHYSICAL top camera: factory intrinsics incl. principal point).
2. registers sim/twin/data/photos/table_topdown.jpg (phone, ~0.35 mm / px) to the table plane: SIFT + RANSAC
   homography against (1), refined by ECC on band-passed images. The homography gives the photo its metric scale
   (checked with the plate diameter) and the table's orientation in the env frame (the table edge is NOT parallel
   to the env y axis; the fitted edge line is stored in the json).
3. clean-wood mask of the photo (no plates, bricks, cables, tripods, robot), illumination-flattened wood.
4. a seamless TILEABLE wood tile synthesised from the clean photo wood by image quilting (Efros & Freeman: overlap
   search + minimum-error boundary cuts, toroidal) in the env frame's orientation, so grain direction, grain width
   and knot size are the real ones.
5. the table texture (same layout as before: x -0.9 .. 1.2, y -1 .. 1, so everything that uses table_top.json keeps
   working): the registered photo itself where it shows clean wood (the real knots and scuffs at their real places, at
   photo resolution), the world-anchored tile elsewhere; colours mapped to what the TOP camera sees (Lab mean / std
   matched on the overlap at the camera's resolution); inside the top camera's footprint the low frequencies
   (sigma --lf-mm) are the top median's, so the top view keeps the real worn colours. No projected detail anywhere, so
   grazing views show isotropic photo-resolution grain. Persistent objects (tripod feet, cables) are NOT baked in;
   they are 3D props (sim/twin/scene/props.py).
6. a normal map from the fine grain (dark grooves = lower).
Writes sim/twin/visuals/textures/{table_top.jpg, table_top_base.jpg, table_top_n.png, table_top.json,
table_wood_tile.jpg, table_wood_tile_n.png} and debug images under sim/twin/out/visual_fit/wood/. Then re-apply the
fitted tone curve (contrast / mean / saturation under the light rig and camera model): wood_tone.py with the
parameters in table_top.json "tone" (fit/log.md).
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
sys.path.insert(0, str(REPO / "sim/residual_env"))
TEX = REPO / "sim/twin/visuals/textures"
DBG = REPO / "sim/twin/out/visual_fit/wood"
PHOTO = REPO / "sim/twin/data/photos/table_topdown.jpg"
MEDIAN = REPO / "sim/twin/out/visuals/top_median.png"
TABLE_TOP_Z = -0.002
REGION = (-0.9, 1.2, -1.0, 1.0)  # x0, x1, y0, y1 (env frame, m): the textured quad of twin_scene (as before)


def top_camera():
    """(R world->cv, position, K) of the physical top camera (reference-cal2 pose, factory intrinsics)."""
    ref = json.loads((REPO / "sim/residual_env/assets/joe_cal2/reference-cal2.json").read_text())["camera"]
    k = json.loads((REPO / "sim/twin/calib/realsense_intrinsics.json").read_text())["top"]
    K = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1.0]])
    return np.array(ref["world_to_cv"]), np.array(ref["position"]), K


def lin(c):
    """sRGB 0-255 -> linear float32."""
    c = np.asarray(c, np.float32) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def srgb(x):
    """Linear 0-1 -> sRGB uint8."""
    x = np.clip(x, 0, 1)
    return (np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055) * 255.0 + 0.5).astype(np.uint8)


def nblur(img, mask, sigma):
    """Gaussian blur that ignores pixels outside `mask` (normalised convolution)."""
    m = mask.astype(np.float32)
    a = cv2.GaussianBlur(img * m[..., None], (0, 0), sigma)
    b = cv2.GaussianBlur(m, (0, 0), sigma)
    return a / np.maximum(b, 1e-4)[..., None]


def median_plane(mm: float):
    """Top median on the REGION grid (row 0 = +y, col 0 = -x) + footprint mask."""
    med = cv2.imread(str(MEDIAN))[..., ::-1]
    r_cv, pos, K = top_camera()
    x0, x1, y0, y1 = REGION
    nx, ny = int(round((x1 - x0) * 1000 / mm)), int(round((y1 - y0) * 1000 / mm))
    xs = x0 + (np.arange(nx) + 0.5) * mm / 1000
    ys = y1 - (np.arange(ny) + 0.5) * mm / 1000
    gx, gy = np.meshgrid(xs, ys)
    pc = (np.stack([gx, gy, np.full_like(gx, TABLE_TOP_Z)], -1) - pos) @ r_cv.T
    u = (pc[..., 0] / pc[..., 2] * K[0, 0] + K[0, 2]).astype(np.float32)
    v = (pc[..., 1] / pc[..., 2] * K[1, 1] + K[1, 2]).astype(np.float32)
    h, w = med.shape[:2]
    valid = (pc[..., 2] > 0) & (u > 1) & (u < w - 2) & (v > 1) & (v < h - 2)
    tex = cv2.remap(np.ascontiguousarray(med), u, v, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return tex, valid


def register_photo(med_tex, valid, mm: float):
    """Homography photo px -> REGION px. SIFT at 1 mm / px, ECC refinement."""
    ph = cv2.imread(str(PHOTO))[..., ::-1]
    s_ph, s_tex = 0.4, mm / 1.0  # work at ~1 mm / px
    a = cv2.resize(ph, None, fx=s_ph, fy=s_ph, interpolation=cv2.INTER_AREA)
    b = cv2.resize(med_tex, None, fx=s_tex, fy=s_tex, interpolation=cv2.INTER_AREA)
    vb = cv2.resize(valid.astype(np.uint8), (b.shape[1], b.shape[0]), interpolation=cv2.INTER_NEAREST)
    clahe = cv2.createCLAHE(3.0, (8, 8))
    ga, gb = (clahe.apply(cv2.cvtColor(np.ascontiguousarray(x), cv2.COLOR_RGB2GRAY)) for x in (a, b))
    sift = cv2.SIFT_create(12000, contrastThreshold=0.012)
    k1, d1 = sift.detectAndCompute(ga, None)
    k2, d2 = sift.detectAndCompute(gb, cv2.erode(vb, np.ones((15, 15), np.uint8)))
    knn = cv2.BFMatcher().knnMatch(d1, d2, k=2)

    def bandpass(g):
        g = g.astype(np.float32)
        return cv2.GaussianBlur(g, (0, 0), 1.5) - cv2.GaussianBlur(g, (0, 0), 8)

    def corr(Hm):
        w = cv2.warpPerspective(ga, Hm, (gb.shape[1], gb.shape[0]))
        wm = cv2.warpPerspective(np.ones_like(ga), Hm, (gb.shape[1], gb.shape[0]))
        hsv = cv2.cvtColor(np.ascontiguousarray(b), cv2.COLOR_RGB2HSV)
        m = (cv2.erode(vb & wm, np.ones((25, 25), np.uint8)) > 0) & (hsv[..., 1] < 110) & (hsv[..., 2] > 70)
        if m.sum() < 20000:
            return -1.0, m
        x, y = bandpass(w)[m], bandpass(gb)[m]
        return float(np.corrcoef(x, y)[0, 1]), m

    # few of the matches are right (worn wood repeats): try several ratio / RANSAC settings, keep the homography
    # whose warped photo correlates best with the top median (band-passed)
    cv2.setRNGSeed(0)
    best = (-2.0, None, None, {})
    for ratio in (0.75, 0.8, 0.85):
        good = [m for m, n in knn if m.distance < ratio * n.distance]
        if len(good) < 12:
            continue
        p1 = np.float32([k1[g.queryIdx].pt for g in good])
        p2 = np.float32([k2[g.trainIdx].pt for g in good])
        for thr in (3.0, 5.0):
            for method in (cv2.RANSAC, cv2.USAC_MAGSAC):
                Hc, inl = cv2.findHomography(p1, p2, method, thr, maxIters=20000, confidence=0.999)
                if Hc is None or abs(np.linalg.det(Hc[:2, :2])) < 0.2 or abs(np.linalg.det(Hc[:2, :2])) > 3:
                    continue
                c, mk = corr(Hc)
                if c > best[0]:
                    best = (c, Hc, mk, dict(sift_matches=len(good), sift_inliers=int(inl.sum()), ratio=ratio))
    c0, H, m, info = best
    assert H is not None and c0 > 0.2, f"photo registration failed (best band-pass correlation {c0:.2f})"
    info["bandpass_corr_sift"] = round(c0, 3)
    try:  # ECC refinement in plane coordinates (photo pre-warped by the SIFT homography)
        w = cv2.warpPerspective(ga, H, (gb.shape[1], gb.shape[0]))
        _, W = cv2.findTransformECC(bandpass(gb), bandpass(w), np.eye(3, dtype=np.float32), cv2.MOTION_HOMOGRAPHY,
                                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 400, 1e-7),
                                    m.astype(np.uint8), 5)
        H2 = np.linalg.inv(W.astype(np.float64)) @ H  # W maps plane -> pre-warped photo coordinates
        H2 /= H2[2, 2]
        c1, _ = corr(H2)
        info["bandpass_corr_ecc"] = round(c1, 3)
        if c1 > c0:
            H = H2
    except cv2.error as e:
        info["ecc_error"] = str(e)[:160]
    # photo full-res px -> REGION px
    S1, S2 = np.diag([s_ph, s_ph, 1.0]), np.diag([1 / s_tex, 1 / s_tex, 1.0])
    Hf = S2 @ H @ S1
    return ph, Hf / Hf[2, 2], info


def wood_mask(P, pvalid, mm: float):
    """Clean wood in the registered photo: tan hue, moderate saturation; small dark knots kept, everything else
    (plates, bricks, cables, tripods, boxes) removed with a margin."""
    hsv = cv2.cvtColor(P, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    wood = (h >= 11) & (h <= 22) & (s >= 45) & (s <= 205) & (v >= 85) & pvalid

    def k(mm_):
        return np.ones((max(3, int(mm_ / mm) | 1),) * 2, np.uint8)

    w = cv2.morphologyEx(wood.astype(np.uint8), cv2.MORPH_CLOSE, k(9))  # knots / scratches are wood
    w = cv2.morphologyEx(w, cv2.MORPH_OPEN, k(25))  # isolated wood-coloured bits are not
    w = cv2.erode(w, k(14))  # margin (object shadows, plate rims)
    n, lab_, st, _ = cv2.connectedComponentsWithStats(w)  # the table: largest component (drops plate interiors)
    if n > 2:
        w = (lab_ == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    return w > 0


def table_edge(med, foot, mm: float):
    """Robot-side table edge as a line in env coordinates (point nearest the robot base, unit direction, inward
    normal), from the top median on the table plane: per row, the first wood texel after the dark / blue floor
    beyond the edge; robust line fit (the robot base and cables are outliers)."""
    x0, x1, y0, y1 = REGION
    hsv = cv2.cvtColor(med, cv2.COLOR_RGB2HSV)
    wood = ((hsv[..., 1] < 95) & (hsv[..., 2] > 85) & foot).astype(np.uint8)
    wood = cv2.morphologyEx(wood, cv2.MORPH_OPEN, np.ones((int(30 / mm) | 1,) * 2, np.uint8))
    rows, cols = [], []
    for r in range(0, wood.shape[0], 10):
        f = np.flatnonzero(foot[r])
        c = np.flatnonzero(wood[r])
        if len(c) > 100 and len(f) and c[0] > f[0] + 20 / mm:  # the footprint starts beyond the edge in this row
            rows.append(r)
            cols.append(c[0])
    rows, cols = np.array(rows, float), np.array(cols, float)
    keep = np.ones(len(rows), bool)
    for _ in range(6):
        a, b = np.polyfit(rows[keep], cols[keep], 1)
        res = cols - (a * rows + b)
        keep = np.abs(res - np.median(res[keep])) < 8 / mm
    pA = np.array([x0 + (b + 0.5) * mm / 1000, y1 - 0.5 * mm / 1000])
    pB = np.array([x0 + (a * 1000 + b + 0.5) * mm / 1000, y1 - 1000.5 * mm / 1000])
    d = (pB - pA) / np.linalg.norm(pB - pA)
    n = np.array([-d[1], d[0]])
    if n[0] < 0:
        n = -n
    p0 = pA + d * float((np.zeros(2) - pA) @ d)  # foot of the robot base on the edge line
    return p0, d, n, int(keep.sum())


def quilt(src, ok, T: int, rng, B: int = 176, ov: int = 48, ds: int = 4, tol: float = 0.08):
    """Toroidal image quilting: T x T tile from `src` (float32 HxWx3) using only blocks fully inside `ok`."""
    step = B - ov
    assert T % step == 0
    n = T // step
    ii = cv2.integral(ok.astype(np.uint8))
    full = (ii[B:, B:] - ii[:-B, B:] - ii[B:, :-B] + ii[:-B, :-B]) == B * B  # valid top-left corners
    assert full.sum() > 500, f"too little clean wood for {B} px blocks ({full.sum()} positions)"
    ssrc = cv2.resize(src, None, fx=1 / ds, fy=1 / ds, interpolation=cv2.INTER_AREA)
    b = B // ds
    fs = full[::ds, ::ds][:ssrc.shape[0] - b + 1, :ssrc.shape[1] - b + 1]
    out = np.zeros((T, T, 3), np.float32)
    filled = np.zeros((T, T), bool)

    def wrap(a0):
        return np.arange(a0, a0 + B) % T

    def seam(err):  # min-cost top-to-bottom path through a (B, ov) cost strip; returns the cut column per row
        c = err.copy()
        for r in range(1, c.shape[0]):
            p = c[r - 1]
            c[r] += np.minimum(p, np.minimum(np.r_[p[1:], np.inf], np.r_[np.inf, p[:-1]]))
        path = np.zeros(c.shape[0], int)
        path[-1] = int(np.argmin(c[-1]))
        for r in range(c.shape[0] - 2, -1, -1):
            j = path[r + 1]
            lo, hi = max(0, j - 1), min(c.shape[1], j + 2)
            path[r] = lo + int(np.argmin(c[r, lo:hi]))
        return path

    for i in range(n):
        for j in range(n):
            ry, rx = wrap(i * step), wrap(j * step)
            cur, fm = out[np.ix_(ry, rx)], filled[np.ix_(ry, rx)]
            if not fm.any():
                ys, xs = np.nonzero(full)
                k = rng.integers(len(ys))
                y, x = int(ys[k]), int(xs[k])
            else:
                t = cv2.resize(cur, (b, b), interpolation=cv2.INTER_AREA)
                m = cv2.resize(fm.astype(np.float32), (b, b), interpolation=cv2.INTER_AREA)
                m3 = np.repeat((m > 0.99).astype(np.float32)[..., None], 3, -1)
                e = cv2.matchTemplate(ssrc, t, cv2.TM_SQDIFF, mask=m3)
                hh, ww = min(e.shape[0], fs.shape[0]), min(e.shape[1], fs.shape[1])
                e = np.where(fs[:hh, :ww] & np.isfinite(e[:hh, :ww]), np.maximum(e[:hh, :ww], 0.0), np.inf)
                thr = e.min() * (1 + tol) + 1e-4
                ys, xs = np.nonzero(e <= thr)
                k = rng.integers(len(ys))
                y, x = int(ys[k]) * ds, int(xs[k]) * ds
                # snap to a valid full-resolution corner near the down-sampled pick
                if not full[y, x]:
                    yy, xx = np.nonzero(full[max(0, y - ds):y + ds + 1, max(0, x - ds):x + ds + 1])
                    y, x = max(0, y - ds) + int(yy[0]), max(0, x - ds) + int(xx[0])
            new = src[y:y + B, x:x + B]
            use = np.ones((B, B), bool)  # True = take the new block
            if fm.any():
                err = ((new - cur) ** 2).sum(-1)
                if fm[:, :ov].all():  # left overlap
                    p = seam(err[:, :ov])
                    use &= np.arange(B)[None, :] >= p[:, None]
                if fm[:ov, :].all():  # top overlap
                    p = seam(err[:ov, :].T)
                    use &= np.arange(B)[:, None] >= p[None, :]
                if fm[:, -ov:].all():  # right overlap (wrap-around)
                    p = seam(err[:, -ov:])
                    use &= np.arange(B)[None, :] < (B - ov + p[:, None])
                if fm[-ov:, :].all():  # bottom overlap (wrap-around)
                    p = seam(err[-ov:, :].T)
                    use &= np.arange(B)[:, None] < (B - ov + p[None, :])
                use |= ~fm
            a = cv2.GaussianBlur(use.astype(np.float32), (0, 0), 1.2)
            a = np.where(fm, a, 1.0)[..., None]
            out[np.ix_(ry, rx)] = a * new + (1 - a) * cur
            filled[np.ix_(ry, rx)] = True
    return out


def normal_map(rgb, mm: float, strength: float = 0.35, sigma_mm: float = 1.6):
    """Tangent-space normal map from the fine grain: height = high-passed luma (sRGB float 0-1; dark = lower)."""
    L = rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    h = L - cv2.GaussianBlur(L, (0, 0), sigma_mm / mm)
    h = cv2.GaussianBlur(h, (0, 0), 0.7) / max(float(L.mean()), 1e-3)
    gx, gy = cv2.Sobel(h, cv2.CV_32F, 1, 0, ksize=3) / 8, cv2.Sobel(h, cv2.CV_32F, 0, 1, ksize=3) / 8
    k = strength * 2.0 / mm  # relative luma -> slope
    n = np.stack([-k * gx, k * gy, np.ones_like(h)], -1)  # image rows run against +v
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return ((n * 0.5 + 0.5) * 255 + 0.5).astype(np.uint8)


def to_lab(x):
    """sRGB float 0-1 -> Lab."""
    return cv2.cvtColor(np.clip(x, 0, 1).astype(np.float32), cv2.COLOR_RGB2LAB)


def from_lab(x):
    """Lab -> sRGB float 0-1."""
    return np.clip(cv2.cvtColor(x.astype(np.float32), cv2.COLOR_LAB2RGB), 0, 1)


def main():
    """Build the table textures."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--mm", type=float, default=0.5, help="texel size (mm)")
    ap.add_argument("--tile-px", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lf-mm", type=float, default=8.0, help="sigma of the top-median low-pass kept in the footprint")
    ap.add_argument("--cam-psf-mm", type=float, default=1.5, help="blur at which photo / camera contrast is matched")
    a = ap.parse_args()
    mm = a.mm
    DBG.mkdir(parents=True, exist_ok=True)
    TEX.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    x0, x1, y0, y1 = REGION

    med, foot = median_plane(mm)
    ny, nx = med.shape[:2]
    photo, H, info = register_photo(med, foot, mm)
    print("registration", info, flush=True)
    P = cv2.warpPerspective(photo, H, (nx, ny), flags=cv2.INTER_LANCZOS4)
    pvalid = cv2.warpPerspective(np.full(photo.shape[:2], 255, np.uint8), H, (nx, ny)) > 250
    info["photo_mm_per_px"] = round(float(np.sqrt(abs(np.linalg.det(H[:2, :2]))) * mm), 3)
    cw = wood_mask(P, pvalid, mm)
    p0, d, n, npts = table_edge(med, foot, mm)
    info.update(edge_point=p0.round(4).tolist(), edge_dir=d.round(4).tolist(), edge_normal=n.round(4).tolist(),
                edge_rows=npts, edge_angle_deg=round(float(np.degrees(np.arctan2(d[0], d[1]))), 2))
    hsv = cv2.cvtColor(P, cv2.COLOR_RGB2HSV)  # scale check: the orange plate's diameter in the registered photo
    om = cv2.morphologyEx(cv2.inRange(hsv, (5, 170, 150), (16, 255, 255)), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    nc, _, st, _ = cv2.connectedComponentsWithStats(om)
    if nc > 1:
        i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        info["orange_plate_diam_m_in_photo"] = round(float(np.sqrt(st[i, cv2.CC_STAT_AREA] * 4 / np.pi) * mm / 1000), 4)
    print("edge / scale", {k: info[k] for k in info if k.startswith(("edge", "orange", "photo"))}, flush=True)

    # crop to the photo's bounding box for the photo-only steps
    ys, xs = np.nonzero(pvalid)
    r0, r1, c0, c1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    Pc, cwc = lin(P[r0:r1, c0:c1]), cw[r0:r1, c0:c1]
    base = nblur(Pc, cwc, 60 / mm)  # illumination (phone flash / sheen) flattened over the clean wood
    Pf = np.clip(Pc / np.maximum(base, 1e-3) * Pc[cwc].mean(0), 0, 1)
    Pf = (srgb(Pf).astype(np.float32) / 255.0)
    dbg = (P[r0:r1, c0:c1] * (0.3 + 0.7 * cwc[..., None])).astype(np.uint8)
    cv2.imwrite(str(DBG / "photo_wood_mask.jpg"), cv2.resize(dbg, None, fx=0.33, fy=0.33)[..., ::-1])

    # colour: photo wood -> what the top camera sees (Lab mean / std on the overlap, photo blurred to the camera PSF)
    medc = med[r0:r1, c0:c1].astype(np.float32) / 255.0
    mh = cv2.cvtColor(med[r0:r1, c0:c1], cv2.COLOR_RGB2HSV)
    med_wood = (mh[..., 1] < 95) & (mh[..., 2] > 75)
    ov = (foot[r0:r1, c0:c1] & cwc & med_wood).astype(np.uint8)
    ov = cv2.erode(ov, np.ones((int(10 / mm) | 1,) * 2, np.uint8)) > 0
    lp, lc = to_lab(cv2.GaussianBlur(Pf, (0, 0), a.cam_psf_mm / mm))[ov], to_lab(medc)[ov]
    mu_p, sd_p, mu_c, sd_c = lp.mean(0), lp.std(0), lc.mean(0), lc.std(0)
    gain = np.clip(sd_c / np.maximum(sd_p, 1e-4), 0.25, [2.0, 0.55, 0.55])  # chroma: grey scuffs must not go blue
    info.update(color_fit_px=int(ov.sum()), lab_photo=[mu_p.round(1).tolist(), sd_p.round(2).tolist()],
                lab_top_cam=[mu_c.round(1).tolist(), sd_c.round(2).tolist()], lab_gain=gain.round(3).tolist())

    def to_cam(x):
        return from_lab((to_lab(x) - mu_p) * gain + mu_c)

    Pm = to_cam(Pf)
    tile = to_cam(quilt(Pf, cwc, a.tile_px, rng))
    T = a.tile_px
    cols = (np.arange(nx) + int(round(x0 * 1000 / mm))) % T
    rows = (np.arange(ny) - int(round(y1 * 1000 / mm))) % T  # tile rows run towards -y; tile st = (x, y) / tile_m
    F = tile[np.ix_(rows, cols)]
    # the registered photo where it shows clean wood (feathered), the world-anchored tile elsewhere
    wp = cv2.erode(cwc.astype(np.uint8), np.ones((int(8 / mm) | 1,) * 2, np.uint8)).astype(np.float32)
    wp = cv2.GaussianBlur(wp, (0, 0), 3 / mm)[..., None]
    F[r0:r1, c0:c1] = wp * Pm + (1 - wp) * F[r0:r1, c0:c1]
    # inside the top camera's footprint: low frequencies from the top median (wood only; props are not baked in)
    mhf = cv2.cvtColor(med, cv2.COLOR_RGB2HSV)
    ok = foot & (mhf[..., 1] < 95) & (mhf[..., 2] > 75)
    ok = cv2.morphologyEx(ok.astype(np.uint8), cv2.MORPH_OPEN, np.ones((int(12 / mm) | 1,) * 2, np.uint8))
    fw = cv2.GaussianBlur(cv2.erode(ok, np.ones((int(24 / mm) | 1,) * 2, np.uint8)).astype(np.float32), (0, 0), 8 / mm)
    ys, xs = np.nonzero(fw > 1e-3)
    f0, f1, g0, g1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    s_lf = a.lf_mm / mm
    sub, okc = F[f0:f1, g0:g1], ok[f0:f1, g0:g1] > 0
    okc = cv2.erode(okc.astype(np.uint8), np.ones((int(5 / mm) | 1,) * 2, np.uint8)) > 0  # no cable / prop fringes
    med_lf = nblur(med[f0:f1, g0:g1].astype(np.float32) / 255.0, okc, s_lf)
    F[f0:f1, g0:g1] = sub + fw[f0:f1, g0:g1, None] * (med_lf - cv2.GaussianBlur(sub, (0, 0), s_lf))
    F = np.clip(F, 0, 1)
    info["footprint_frac"] = round(float((fw > 0.5).mean()), 3)

    u8 = (F * 255 + 0.5).astype(np.uint8)
    t8 = (tile * 255 + 0.5).astype(np.uint8)
    cv2.imwrite(str(TEX / "table_top.jpg"), u8[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(TEX / "table_top_base.jpg"), u8[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])  # wood_tone.py source
    cv2.imwrite(str(TEX / "table_wood_tile.jpg"), t8[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
    half = cv2.resize(F, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(TEX / "table_top_n.png"), normal_map(half, mm * 2)[..., ::-1])
    cv2.imwrite(str(TEX / "table_wood_tile_n.png"), normal_map(tile, mm)[..., ::-1])
    cv2.imwrite(str(DBG / "tile_2x2.jpg"), cv2.resize(np.tile(t8, (2, 2, 1)), None, fx=0.5, fy=0.5,
                                                    interpolation=cv2.INTER_AREA)[..., ::-1])
    small = cv2.resize(u8, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(DBG / "table_small.jpg"), small[..., ::-1])
    cx, cy = int((0.25 - x0) * 1000 / mm), int((y1 - 0.0) * 1000 / mm)  # close-up around env (0.25, 0), 30 x 20 cm
    cv2.imwrite(str(DBG / "table_closeup.jpg"), u8[cy - 200:cy + 200, cx - 300:cx + 300][..., ::-1])
    cv2.imwrite(str(DBG / "median_closeup.jpg"), med[cy - 200:cy + 200, cx - 300:cx + 300][..., ::-1])
    r_cv, pos, K = top_camera()
    vis = fw > 0.5
    meta = dict(x0=x0, y0=y0, width=x1 - x0, height=y1 - y0, z=TABLE_TOP_Z, nx=nx, ny=ny,
                note="texture row 0 = +y edge, col 0 = -x edge; uv = ((x-x0)/W, (y-y0)/H)",
                camera=dict(profile="reference-cal2 + factory intrinsics", R_world_to_cv=r_cv.tolist(),
                            position=pos.tolist(), K=K.tolist()),
                mean_rgb_visible=[float(v) for v in u8[vis].mean(0)], mm_per_px=mm,
                normal_map="table_top_n.png", tile=dict(file="table_wood_tile.jpg", normal="table_wood_tile_n.png",
                                                        size_m=T * mm / 1000, px=T, origin_env=[0.0, 0.0]),
                table_edge=dict(point=p0.tolist(), dir=d.tolist(), normal=n.tolist()), build=info)
    (TEX / "table_top.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(info), flush=True)
    print("WOOD_OK", nx, ny)


if __name__ == "__main__":
    main()
