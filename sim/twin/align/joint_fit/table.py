"""Wrist camera's view of the table as a world-frame constraint on the whole chain (joints + wrist mount).

The table is (close to) a plane, so a wrist frame and the top frame taken at the same instant are related by the
plane homography given both camera poses. Blue things on the table (blue plate, blue bricks, the blue marker discs)
are segmented in both images; wrist-blue pixels (upper 60 % of the view, away from the jaws / held brick) are cast
onto the plane z = h and projected into the top camera, and top-blue pixels are cast onto the plane and projected
into the wrist camera (with its measured distortion). Loss = symmetric mean chamfer distance (px at 320x240, capped).
Any error in the wrist camera's world pose (wrist_roll, wrist_flex, ... or the mount) shifts / rotates the match.
"""
from __future__ import annotations

import cv2
import model as M
import numpy as np

ROWS = 0.6  # wrist rows used (fraction from the top)


def blue_mask(rgb, cam):
    """Blue / lavender things. The lavender plate reads sky-blue (saturated) in the top camera but pale lavender in
    the wrist camera, hence the wider hue and lower saturation there. Top rows beyond the table edge are dropped."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    if cam == "wrist":
        m = (h >= 100) & (h <= 140) & (s > 60) & (v > 90)
    else:
        m = (h >= 95) & (h <= 125) & (s > 120) & (v > 60)
        m[:28] = False
    m = m.astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def build(items, D):
    """-> list of table items (one per wrist frame with enough blue in both views)."""
    key = {(it["cam"], it["ep"], it["fr"]): it for it in items}
    w = M.BASE["wrist"]
    K = np.array([[w["fx"], 0, w["cx"]], [0, w["fy"], w["cy"]], [0, 0, 1]])
    top = M.BASE["top"]
    out = []
    for it in items:
        if it["cam"] != "wrist" or ("top", it["ep"], it["fr"]) not in key:
            continue
        tp = key[("top", it["ep"], it["fr"])]
        bw = blue_mask(D["rgb"][it["i"]], "wrist")
        bw[int(ROWS * M.H):] = 0
        bt = blue_mask(D["rgb"][tp["i"]], "top")
        bt[tp["real"] > 0] = 0  # pixels on the arm (a held blue brick) are not on the table
        if bw.sum() < 40 or bt.sum() < 40:
            continue
        yx = np.argwhere(bw)
        yx = yx[np.random.default_rng(it["i"]).choice(len(yx), min(400, len(yx)), replace=False)]
        pts = (yx[:, ::-1].astype(np.float64) / M.S).reshape(-1, 1, 2)
        nrm = cv2.undistortPoints(pts, K, w["dist"]).reshape(-1, 2)
        ty = np.argwhere(bt)
        ty = ty[np.random.default_rng(it["i"] + 1).choice(len(ty), min(400, len(ty)), replace=False)]
        # top pixel rays in world
        uv = ty[:, ::-1].astype(np.float64) / M.S
        rc = np.stack([(uv[:, 0] - top["cx"]) / top["fx"], (uv[:, 1] - top["cy"]) / top["fy"], np.ones(len(uv))], 1)
        out.append(dict(i=it["i"], ep=it["ep"], fr=it["fr"], state=it["state"], held=it["held"], wrist_rays=nrm,
                        dt_top=cv2.distanceTransform((1 - bt).astype(np.uint8), cv2.DIST_L2, 3),
                        dt_wrist=cv2.distanceTransform((1 - bw).astype(np.uint8), cv2.DIST_L2, 3),
                        top_rays=rc @ top["R"], wrist_blue=bw, top_blue=bt))
    return out


def loss(robot, ti, theta, h=0.01, cams=None):
    cams = cams or M.cams_of(theta)
    T = robot.arm.fk(M.q_of(ti["state"], theta))
    wc = M.world_cam("wrist", cams, T)
    top = cams["top"]
    # wrist -> plane -> top
    d = np.c_[ti["wrist_rays"], np.ones(len(ti["wrist_rays"]))] @ wc["R"]
    s = (h - wc["t"][2]) / np.where(np.abs(d[:, 2]) > 1e-6, d[:, 2], 1e-6)
    ok = s > 0
    X = wc["t"] + s[:, None] * d
    uv, ok2 = M.project(X[ok], top)
    uv = uv[ok2] * M.S
    inside = (uv[:, 0] >= 0) & (uv[:, 0] <= M.W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= M.H - 1)
    a = np.r_[np.minimum(M.bilinear(ti["dt_top"], uv[inside]), M.CAP), np.full((~inside).sum() + (~ok).sum(), M.CAP)]
    # top -> plane -> wrist
    d = ti["top_rays"]
    s = (h - top["t"][2]) / d[:, 2]
    X = top["t"] + s[:, None] * d
    uv, ok = M.project(X, wc)
    uv = uv[ok] * M.S
    inside = (uv[:, 0] >= 0) & (uv[:, 0] <= M.W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= ROWS * M.H - 1)
    b = np.minimum(M.bilinear(ti["dt_wrist"], uv[inside]), M.CAP) if inside.sum() >= 20 else None
    return float(a.mean() + (b.mean() if b is not None else a.mean())), int(inside.sum())


def overlay(robot, ti, theta, rgb=None, h=0.01):
    """Mask in the wrist view of the top camera's blue cast through the plane z = h (inverse mapping per pixel)."""
    cams = M.cams_of(theta)
    T = robot.arm.fk(M.q_of(ti["state"], theta))
    wc = M.world_cam("wrist", cams, T)
    top = cams["top"]
    w = M.BASE["wrist"]
    K = np.array([[w["fx"], 0, w["cx"]], [0, w["fy"], w["cy"]], [0, 0, 1]])
    yy, xx = np.mgrid[0:M.H, 0:M.W]
    pts = (np.stack([xx.ravel(), yy.ravel()], 1).astype(np.float64) / M.S).reshape(-1, 1, 2)
    nrm = cv2.undistortPoints(pts, K, w["dist"]).reshape(-1, 2)
    d = np.c_[nrm, np.ones(len(nrm))] @ wc["R"]
    s = (h - wc["t"][2]) / np.where(np.abs(d[:, 2]) > 1e-6, d[:, 2], 1e-6)
    X = wc["t"] + np.maximum(s, 0)[:, None] * d
    uv, ok = M.project(X, top)
    uv = np.round(uv * M.S).astype(int)
    ok &= (s > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < M.W) & (uv[:, 1] >= 0) & (uv[:, 1] < M.H)
    m = np.zeros(M.H * M.W, np.uint8)
    m[ok] = ti["top_blue"][uv[ok, 1], uv[ok, 0]]
    return m.reshape(M.H, M.W)
