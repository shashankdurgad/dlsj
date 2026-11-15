"""Lighting metrics, real vs sim at matched states (numpy / cv2 only; no PyAV, importable inside Isaac).

    from sim.twin.lighting.metrics import light_metrics
    m = light_metrics(real, sim, masks, samples)     # (N, H, W, 3) uint8 sRGB each, masks (N, H, W) region bits

Region bits (visuals/fit/harness.py): 1 table, 2 plates, 4 bricks, 8 robot. All metrics are invariant to a global
gain and white balance per image (the real RealSenses ran auto exposure / auto white balance):
  brightness  table luma on a 4 x 4 grid, each image divided by its own table median; `grid_rms` = RMS of
              log(real / sim) over the cells (how well the light falloff across the table matches), `grad_*` = the
              fitted plane slope of log luma along image x / y (per 100 px).
  shadow      mid-episode frames (reach / grasp / inspect / place) divided by the same episode's start frame on table
              pixels that are table in both (occluders dilated), normalised by the median ratio: `area_*` = fraction
              darker than 0.85 (the arm's / held brick's cast shadow), `depth_*` = 1 - median ratio inside it, `iou` =
              overlap of the real and sim shadow masks, `dir_err_deg` = angle between the shadow-centroid offsets from
              the robot-mask centroid.
  highlight   per plate (connected component of bit 2): pixels brighter than the plate's median L* + 18 and less
              saturated: `area_*` = fraction of plate pixels, `iou`, `pos_err` = distance of the highlight centroids
              relative to the plate centre in plate radii.
  tint        table chromaticity (log R/G, log B/G) of the image's left vs right third minus the middle (a coloured
              light from one side shows as a left / right difference): `tint_err` = mean |real - sim|.
"""
from __future__ import annotations

import cv2
import numpy as np

MID = ("reach", "grasp", "inspect", "place")


def _luma(img):
    c = np.asarray(img, np.float32) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return lin @ np.array([0.2126, 0.7152, 0.0722], np.float32), lin


def _table(mask, grow: int = 7):
    occ = cv2.dilate(((mask & (2 | 4 | 8)) > 0).astype(np.uint8), np.ones((grow, grow), np.uint8)) > 0
    return ((mask & 1) > 0) & ~occ


def brightness(real, sim, masks) -> dict:
    """Light falloff across the table (see module doc)."""
    rms, gr, gs = [], [], []
    for r, s, m in zip(real, sim, masks, strict=True):
        t = _table(m)
        if t.sum() < 500:
            continue
        lr, ls = _luma(r)[0], _luma(s)[0]
        lr, ls = lr / np.median(lr[t]), ls / np.median(ls[t])
        H, W = t.shape
        d = []
        for i in range(4):
            for j in range(4):
                c = t[i * H // 4:(i + 1) * H // 4, j * W // 4:(j + 1) * W // 4]
                if c.sum() < 50:
                    continue
                a = lr[i * H // 4:(i + 1) * H // 4, j * W // 4:(j + 1) * W // 4][c]
                b = ls[i * H // 4:(i + 1) * H // 4, j * W // 4:(j + 1) * W // 4][c]
                d.append(np.log(np.median(a) / max(np.median(b), 1e-6)))
        rms.append(float(np.sqrt(np.mean(np.square(d)))))
        v, u = np.nonzero(t)
        A = np.c_[u / 100.0, v / 100.0, np.ones(len(u))]
        gr.append(np.linalg.lstsq(A, np.log(np.maximum(lr[t], 1e-4)), rcond=None)[0][:2])
        gs.append(np.linalg.lstsq(A, np.log(np.maximum(ls[t], 1e-4)), rcond=None)[0][:2])
    gr, gs = np.mean(gr, 0), np.mean(gs, 0)
    return dict(grid_rms=float(np.mean(rms)), grad_real=gr.round(4).tolist(), grad_sim=gs.round(4).tolist())


def _shadow(img_mid, img_start, ok):
    lm, ls = _luma(img_mid)[0], _luma(img_start)[0]
    q = lm / np.maximum(ls, 1e-4)
    q = q / np.median(q[ok])
    q = cv2.GaussianBlur(q.astype(np.float32), (0, 0), 1.5)
    return q, ok & (q < 0.85)


def shadow(real, sim, masks, samples, x_max: float = 1.0) -> dict:
    """Cast shadows of the arm / held brick (see module doc). Only image columns < x_max * W are used (top camera:
    0.78, the strip on the right holds the cables and the tripod, which move between frames)."""
    start = {s["ep"]: i for i, s in enumerate(samples) if s["moment"] == "start"}
    acc = {k: [] for k in ("area_real", "area_sim", "depth_real", "depth_sim", "iou", "dir_err_deg")}
    for i, s in enumerate(samples):
        j = start.get(s["ep"])
        if s["moment"] not in MID or j is None:
            continue
        ok = _table(masks[i], 21) & _table(masks[j], 21)  # wide margin: sim / real poses differ by ~10 px
        ok[:, int(x_max * ok.shape[1]):] = False
        if ok.sum() < 2000:
            continue
        qr, sr = _shadow(real[i], real[j], ok)
        qs, ss = _shadow(sim[i], sim[j], ok)
        acc["area_real"].append(sr.sum() / ok.sum())
        acc["area_sim"].append(ss.sum() / ok.sum())
        acc["depth_real"].append(1 - float(np.median(qr[sr])) if sr.sum() > 30 else 0.0)
        acc["depth_sim"].append(1 - float(np.median(qs[ss])) if ss.sum() > 30 else 0.0)
        if sr.sum() + ss.sum() > 60:
            acc["iou"].append((sr & ss).sum() / max((sr | ss).sum(), 1))
        rob = (masks[i] & 8) > 0
        if rob.sum() > 100 and sr.sum() > 60 and ss.sum() > 60:
            c0 = np.argwhere(rob).mean(0)
            a, b = np.argwhere(sr).mean(0) - c0, np.argwhere(ss).mean(0) - c0
            cos = a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9)
            acc["dir_err_deg"].append(float(np.degrees(np.arccos(np.clip(cos, -1, 1)))))
    return {k: (round(float(np.mean(v)), 4) if v else None) for k, v in acc.items()} | {"n": len(acc["area_real"])}


def _highlights(img, plate):
    lab = cv2.cvtColor(np.asarray(img, np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(np.asarray(img, np.uint8), cv2.COLOR_RGB2HSV)
    L0, s0 = np.median(lab[..., 0][plate]), np.median(hsv[..., 1][plate])
    return plate & (lab[..., 0] > L0 + 18) & (hsv[..., 1] < 0.75 * s0)


def highlight(real, sim, masks) -> dict:
    """Specular highlights on the glossy plates (see module doc)."""
    acc = {k: [] for k in ("area_real", "area_sim", "iou", "pos_err")}
    for r, s, m in zip(real, sim, masks, strict=True):
        n, lab = cv2.connectedComponents(cv2.erode(((m & 2) > 0).astype(np.uint8), np.ones((3, 3), np.uint8)))
        for k in range(1, n):
            p = lab == k
            if p.sum() < 400:
                continue
            hr, hs = _highlights(r, p), _highlights(s, p)
            acc["area_real"].append(hr.sum() / p.sum())
            acc["area_sim"].append(hs.sum() / p.sum())
            if hr.sum() + hs.sum() > 10:
                acc["iou"].append((hr & hs).sum() / max((hr | hs).sum(), 1))
            if hr.sum() > 5 and hs.sum() > 5:
                rad = np.sqrt(p.sum() / np.pi)
                acc["pos_err"].append(float(np.linalg.norm(np.argwhere(hr).mean(0) - np.argwhere(hs).mean(0)) / rad))
    return {k: (round(float(np.mean(v)), 4) if v else None) for k, v in acc.items()}


def tint(real, sim, masks) -> dict:
    """Left / right table chromaticity differences (see module doc)."""
    def lr(img, t):
        lin = _luma(img)[1]
        W = t.shape[1]
        out = []
        for a, b in ((0, W // 3), (W // 3, 2 * W // 3), (2 * W // 3, W)):
            c = t[:, a:b]
            if c.sum() < 100:
                return None
            m = lin[:, a:b][c].mean(0) + 1e-4
            out.append(np.log([m[0] / m[1], m[2] / m[1]]))
        return np.array([out[0] - out[1], out[2] - out[1]])

    er, rr, ss = [], [], []
    for r, s, m in zip(real, sim, masks, strict=True):
        t = _table(m)
        a, b = lr(r, t), lr(s, t)
        if a is None or b is None:
            continue
        er.append(np.abs(a - b).mean())
        rr.append(a)
        ss.append(b)
    return dict(tint_err=round(float(np.mean(er)), 4), real_lr=np.mean(rr, 0).round(4).tolist(),
                sim_lr=np.mean(ss, 0).round(4).tolist())


def light_metrics(real, sim, masks, samples, cam: str = "top") -> dict:
    """All lighting metrics of one camera (see module doc)."""
    x_max = 0.78 if cam == "top" else 1.0
    return dict(brightness=brightness(real, sim, masks), shadow=shadow(real, sim, masks, samples, x_max),
                highlight=highlight(real, sim, masks), tint=tint(real, sim, masks))
