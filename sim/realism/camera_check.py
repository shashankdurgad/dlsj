"""Stage A2 camera realism check + top-camera refinement.

1. For each rendered pose (sim top camera at a recorded joint state) vs the real top frame at that moment: red-finger
   masks (real mask restricted to a window around the sim fingers so red bricks don't count), centroid offset, IoU.
2. Refine the top camera (6-DoF pose, optional focal) by minimising, over all frames, the distance between the
   centroid of the FK finger points projected with the candidate camera and the real red-finger centroid (Nelder-Mead,
   numpy only). Report before/after pixel error (median, p90) on the fit set and a held-out set.

    python camera_check.py <a2 dir>   (needs camera-fit.json, extract_episodes.json, frames/, renders/)
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from table_plane import TopCamera  # noqa: E402


def red_mask(img: np.ndarray, sim: bool = False) -> np.ndarray:
    """Red/pink gripper fingers. Sim fingers are (0.65, 0.045, 0.12) -> hue ~350 deg; the sim's orange plate (~15 deg)
    must not count, so the sim mask uses only the pink side of red. Real: both sides of red, minus orange."""
    hsv = np.asarray(Image.fromarray(img).convert("HSV")).astype(int)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    if sim:
        return (h > 225) & (s > 120) & (v > 50)
    return ((h < 6) | (h > 225)) & (s > 110) & (v > 70)


def dilate(m: np.ndarray, r: int) -> np.ndarray:
    out = m.copy()
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return out
    y0, y1, x0, x1 = max(ys.min() - r, 0), min(ys.max() + r, m.shape[0] - 1), max(xs.min() - r, 0), min(xs.max() + r, m.shape[1] - 1)
    out[y0:y1 + 1, x0:x1 + 1] = True
    return out


def rot(rx, ry, rz):
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def nelder_mead(f, x0, step, iters=1500, tol=1e-6):
    n = len(x0)
    pts = [np.asarray(x0, float)] + [np.asarray(x0, float) + np.eye(n)[i] * step[i] for i in range(n)]
    vals = [f(p) for p in pts]
    for _ in range(iters):
        o = np.argsort(vals)
        pts, vals = [pts[i] for i in o], [vals[i] for i in o]
        if abs(vals[-1] - vals[0]) < tol:
            break
        c = np.mean(pts[:-1], 0)
        xr = c + (c - pts[-1]); fr = f(xr)
        if fr < vals[0]:
            xe = c + 2 * (c - pts[-1]); fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = c + 0.5 * (pts[-1] - c); fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                pts = [pts[0]] + [pts[0] + 0.5 * (p - pts[0]) for p in pts[1:]]
                vals = [vals[0]] + [f(p) for p in pts[1:]]
    return pts[int(np.argmin(vals))], min(vals)


def main(d: Path):
    fit = json.load(open(d / "camera-fit.json"))
    cam = TopCamera(fit)
    eps = json.load(open(d / "extract_episodes.json"))
    rows = []
    for e in eps:
        for k in ("mid2", "inspect"):
            tag = f"{e['set']}_{e['episode']:03d}_{k}"
            sim_p, real_p, fp = d / "renders" / f"{tag}_top_sim.png", d / "frames" / f"{tag}_top.jpg", d / "renders" / f"{tag}_fingers.npy"
            if not (sim_p.exists() and real_p.exists() and fp.exists()):
                continue
            sim = np.asarray(Image.open(sim_p).convert("RGB"))
            real = np.asarray(Image.open(real_p).convert("RGB"))
            ms = red_mask(sim, sim=True)
            if ms.sum() < 50:
                continue
            mr = red_mask(real) & dilate(ms, 70)
            if mr.sum() < 50:
                continue
            ys, xs = np.nonzero(ms); cs = np.array([xs.mean(), ys.mean()])
            ys, xs = np.nonzero(mr); cr = np.array([xs.mean(), ys.mean()])
            iou = (ms & mr).sum() / max((ms | mr).sum(), 1)
            P = np.load(fp).astype(float)
            rows.append({"tag": tag, "set": e["set"], "sim_c": cs, "real_c": cr, "iou": float(iou), "P": P,
                         "proj_c": cam.project(P).mean(0)})
    off = np.array([r["real_c"] - r["sim_c"] for r in rows])
    err0 = np.linalg.norm(off, axis=1)
    print(f"render-vs-real red-finger centroid: n={len(rows)}  offset mean (dx,dy)=({off[:,0].mean():+.1f},{off[:,1].mean():+.1f}) px, "
          f"|err| median {np.median(err0):.1f} p90 {np.percentile(err0,90):.1f} px, IoU median {np.median([r['iou'] for r in rows]):.2f}")
    # consistency check: projected FK finger centroid vs rendered centroid (should agree -> camera model == USD camera)
    pc = np.array([r["proj_c"] - r["sim_c"] for r in rows])
    print(f"projected-vs-rendered sim centroid: median |d| {np.median(np.linalg.norm(pc,axis=1)):.1f} px (model check)")

    # refinement: perturb camera rotation (rx,ry,rz rad, in the camera frame) + translation (m) + focal scale
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(rows)); nfit = int(0.7 * len(rows))
    tr, te = [rows[i] for i in idx[:nfit]], [rows[i] for i in idx[nfit:]]
    R0, C0, f0 = cam.R.copy(), cam.C.copy(), cam.f
    bias = np.array([r["proj_c"] - r["sim_c"] for r in rows]).mean(0)  # finger-point centroid vs mask centroid bias

    def errors(x, rs):
        cam.R = rot(*x[:3]) @ R0
        cam.C = C0 + x[3:6]
        cam.f = f0 * (1 + x[6])
        e = [np.linalg.norm(cam.project(r["P"]).mean(0) - bias - r["real_c"]) for r in rs]
        cam.R, cam.C, cam.f = R0, C0, f0
        return np.asarray(e)

    def loss(x):
        e = errors(x, tr)
        return float(np.mean(np.minimum(e, 40.0) ** 2)) + 50.0 * float(np.sum(np.asarray(x[3:6]) ** 2))  # robust, mild prior

    x0 = np.zeros(7)
    xb, _ = nelder_mead(loss, x0, [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    for name, rs in (("fit", tr), ("held-out", te)):
        a, b = errors(x0, rs), errors(xb, rs)
        print(f"{name:8s} n={len(rs)}: before median {np.median(a):.1f} p90 {np.percentile(a,90):.1f} px -> after median {np.median(b):.1f} p90 {np.percentile(b,90):.1f} px")
    print("refinement: rot (deg)", np.degrees(xb[:3]).round(2), "trans (mm)", (xb[3:6] * 1000).round(1), "focal x", round(1 + xb[6], 4))
    Rn, Cn, fn = rot(*xb[:3]) @ R0, C0 + xb[3:6], f0 * (1 + xb[6])
    res = {"n_frames": len(rows), "before": {"median_px": float(np.median(errors(x0, rows))), "p90_px": float(np.percentile(errors(x0, rows), 90))},
           "after": {"median_px": float(np.median(errors(xb, rows))), "p90_px": float(np.percentile(errors(xb, rows), 90))},
           "heldout_before_median_px": float(np.median(errors(x0, te))), "heldout_after_median_px": float(np.median(errors(xb, te))),
           "render_vs_real_offset_mean_px": off.mean(0).round(2).tolist(), "render_vs_real_median_px": float(np.median(err0)),
           "iou_median": float(np.median([r["iou"] for r in rows])),
           "refined_camera": {"world_to_cv": Rn.tolist(), "position": Cn.tolist(), "focal_px": fn, "principal_point": [320, 240],
                              "delta_rot_deg": np.degrees(xb[:3]).tolist(), "delta_trans_mm": (xb[3:6] * 1000).tolist(),
                              "focal_scale": 1 + xb[6]},
           "per_set_render_offset": {s: np.array([r["real_c"] - r["sim_c"] for r in rows if r["set"] == s]).mean(0).round(2).tolist()
                                     for s in ("old", "cal2")}}
    (d / "camera_check.json").write_text(json.dumps(res, indent=1))
    # overlay figure: real frame with sim red mask outline
    tiles = []
    for r in rows[:8]:
        real = np.asarray(Image.open(d / "frames" / f"{r['tag']}_top.jpg").convert("RGB")).copy()
        sim = np.asarray(Image.open(d / "renders" / f"{r['tag']}_top_sim.png").convert("RGB"))
        ms = red_mask(sim, sim=True)
        real[ms] = (0.5 * real[ms] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
        tiles.append(Image.fromarray(real).resize((320, 240)))
    s = Image.new("RGB", (320 * 4, 240 * 2))
    for i, t in enumerate(tiles):
        s.paste(t, ((i % 4) * 320, (i // 4) * 240))
    s.save(d / "overlay_sim_fingers_on_real.jpg")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
