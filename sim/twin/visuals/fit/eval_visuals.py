"""Real-vs-sim image metrics for the twin's look (numpy / cv2 / torch only; importable inside Isaac, no PyAV).

    python sim/twin/visuals/fit/eval_visuals.py --sim sim/twin/out/visual_fit/render_<tag>.npz [--tag <tag>]

The sim npz (written by fit_visuals.py --render / render_eval) holds, per camera, the twin's renders of every
evaluation sample (evalset.json, same order) and region bit-masks from the sim's own geometry
(1 table, 2 plates, 4 bricks, 8 robot). Metrics per camera, for the fit and the held-out episodes separately:
  color     per region: Lab mean of real / sim, dE (CIE76) of the per-sample means, std ratio, highlight strength
            (p99 - p50 of L) and highlight area (fraction of the region brighter than p50 + 15 L);
  sharp     variance of the Laplacian (whole image, and inside the table region), radial power spectrum in 3 bands
            (log10 sim / real);
  hist      L histogram distance (mean |CDF difference|, in L units) + mean L;
  ssim      SSIM (gaussian 1.5 px, luma) averaged over the table + plate regions;
  clf       real-vs-sim patch classifier: logistic regression on simple 32 px patch features (Lab mean / std, gradient,
            Laplacian, three FFT bands), trained on the fit episodes, accuracy on the held-out ones (50 % = cannot
            tell);
            `clf_tex` is the same without the colour means (texture / sharpness only).
`score(m)` folds one camera's metrics into the scalar the coordinate descent minimises.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[4]
OUT = REPO / "sim/twin/out/visual_fit"
CAMS = ("top", "wrist", "inspect")
REGIONS = {"table": 1, "plates": 2, "bricks": 4, "robot": 8}


def lab(img):
    """sRGB uint8 image -> CIE Lab float (L 0-100)."""
    return cv2.cvtColor(np.ascontiguousarray(img).astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def gray(img):
    """sRGB uint8 image -> float32 luma 0-255."""
    return cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2GRAY).astype(np.float32)


def lap_var(g, mask=None):
    """Variance of the Laplacian of a grey image (inside `mask` if given): the sharpness measure."""
    lp = cv2.Laplacian(g, cv2.CV_32F)
    return float(lp[mask].var()) if mask is not None and mask.sum() > 500 else float(lp.var())


_BANDS = ((0.02, 0.06), (0.06, 0.15), (0.15, 0.5))


def band_power(g):
    """Mean spectral power of a grey image in three radial frequency bands (cycles / px: _BANDS)."""
    f = np.fft.fftshift(np.abs(np.fft.fft2(g - g.mean())) ** 2)
    h, w = g.shape
    yy, xx = np.mgrid[:h, :w]
    r = np.hypot((yy - h // 2) / h, (xx - w // 2) / w)
    return [float(f[(r >= a) & (r < b)].mean()) for a, b in _BANDS]


def ssim_map(a, b):
    """Per-pixel SSIM of two float luma images (gaussian window, sigma 1.5 px)."""
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

    def blur(x):
        return cv2.GaussianBlur(x, (11, 11), 1.5)

    ma, mb = blur(a), blur(b)
    va, vb, cab = blur(a * a) - ma * ma, blur(b * b) - mb * mb, blur(a * b) - ma * mb
    return ((2 * ma * mb + c1) * (2 * cab + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2))


def region_mask(m, bit, erode=5):
    """Bool mask of the region bit(s) of a mask image, eroded by `erode` px (misalignment margin)."""
    r = ((m & bit) > 0).astype(np.uint8)
    return cv2.erode(r, np.ones((erode, erode), np.uint8)) > 0 if erode else r > 0


def patch_features(img, rng, n=24, size=32, color=True):
    """Simple features of `n` random `size` px patches: Lab mean (if `color`), Lab std, gradient, Laplacian,
    three FFT bands."""
    L = lab(img)
    g = L[..., 0] * 2.55
    h, w = g.shape
    out = []
    for _ in range(n):
        y, x = int(rng.integers(0, h - size)), int(rng.integers(0, w - size))
        p, pl = g[y:y + size, x:x + size], L[y:y + size, x:x + size].reshape(-1, 3)
        gx, gy = cv2.Sobel(p, cv2.CV_32F, 1, 0), cv2.Sobel(p, cv2.CV_32F, 0, 1)
        gm = np.hypot(gx, gy)
        f = np.abs(np.fft.fft2(p - p.mean())) ** 2
        fy, fx = np.meshgrid(np.fft.fftfreq(size), np.fft.fftfreq(size), indexing="ij")
        r = np.hypot(fx, fy)
        bands = [np.log10(f[(r >= a) & (r < b)].mean() + 1e-3) for a, b in ((0.03, 0.1), (0.1, 0.25), (0.25, 0.71))]
        tex = [*pl.std(0), np.log10(gm.mean() + 1e-2), np.log10(gm.std() + 1e-2),
               np.log10(cv2.Laplacian(p, cv2.CV_32F).var() + 1e-2), *bands]
        out.append(([*pl.mean(0)] if color else []) + tex)
    return np.asarray(out, np.float32)


def logreg_acc(Xtr, ytr, Xte, yte, steps=300):
    """Held-out accuracy of an L2-regularised logistic regression on standardised features (+ their squares)."""
    import torch

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    # quadratic expansion: lets the linear model see "too little / too much" of a feature
    A, B = torch.tensor(np.c_[Xtr, Xtr ** 2]), torch.tensor(np.c_[Xte, Xte ** 2])
    y = torch.tensor(ytr, dtype=torch.float32)
    w = torch.zeros(A.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(A @ w + b, y) + 1e-3 * (w ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return float((((B @ w + b) > 0).numpy() == (yte > 0.5)).mean())


def evaluate(real: dict, sim: dict, masks: dict, samples: list, cams=CAMS, clf: bool = True, idx=None) -> dict:
    """{cam: {split: metrics}}; real / sim: {cam: (N, H, W, 3) uint8}; masks: {cam: (N, H, W) uint8 bits}."""
    res = {}
    idx = list(range(len(samples))) if idx is None else list(idx)
    for cam in cams:
        if cam not in sim:
            continue
        res[cam] = {}
        feats = {}
        for split in ("fit", "held"):
            ids = [i for i in idx if samples[i]["split"] == split]
            if not ids:
                continue
            col = {r: dict(real=[], sim=[], sreal=[], ssim_=[], hl_r=[], hl_s=[], ha_r=[], ha_s=[]) for r in REGIONS}
            sh = dict(real=[], sim=[], treal=[], tsim=[], br=[], bs=[])
            hist_r, hist_s, ss = np.zeros(64), np.zeros(64), []
            fr, fs = [], []
            for i in ids:
                R, Sm, M = real[cam][i], sim[cam][i], masks[cam][i]
                lr, ls = lab(R), lab(Sm)
                for rn, bit in REGIONS.items():
                    m = region_mask(M, bit)
                    if m.sum() < 300:
                        continue
                    c = col[rn]
                    a, b = lr[m], ls[m]
                    c["real"].append(a.mean(0))
                    c["sim"].append(b.mean(0))
                    c["sreal"].append(a.std(0))
                    c["ssim_"].append(b.std(0))
                    for key, v in (("r", a[:, 0]), ("s", b[:, 0])):
                        p50, p99 = np.percentile(v, [50, 99])
                        c["hl_" + key].append(p99 - p50)
                        c["ha_" + key].append(float((v > p50 + 15).mean()))
                gr, gs = gray(R), gray(Sm)
                tm = region_mask(M, 1, 9)
                sh["real"].append(lap_var(gr))
                sh["sim"].append(lap_var(gs))
                if tm.sum() > 2000:
                    sh["treal"].append(lap_var(gr, tm))
                    sh["tsim"].append(lap_var(gs, tm))
                sh["br"].append(band_power(gr))
                sh["bs"].append(band_power(gs))
                hist_r += np.histogram(lr[..., 0], 64, (0, 100))[0]
                hist_s += np.histogram(ls[..., 0], 64, (0, 100))[0]
                sm = region_mask(M, 3, 7)
                if sm.sum() > 2000:
                    ss.append(float(ssim_map(gr, gs)[sm].mean()))
                if clf:
                    fr.append(patch_features(R, np.random.default_rng(1000 + i)))  # same patch positions
                    fs.append(patch_features(Sm, np.random.default_rng(1000 + i)))
            m = {"n": len(ids), "color": {}}
            for rn, c in col.items():
                if len(c["real"]) < 3:
                    continue
                a, b = np.array(c["real"]), np.array(c["sim"])
                m["color"][rn] = dict(
                    n=len(a), real_lab=a.mean(0).round(1).tolist(), sim_lab=b.mean(0).round(1).tolist(),
                    dE=round(float(np.linalg.norm(a - b, axis=1).mean()), 2),
                    dE_mean=round(float(np.linalg.norm(a.mean(0) - b.mean(0))), 2),
                    std_ratio_L=round(float(np.mean(c["ssim_"], 0)[0] / max(np.mean(c["sreal"], 0)[0], 1e-3)), 2),
                    highlight_real=round(float(np.mean(c["hl_r"])), 1),
                    highlight_sim=round(float(np.mean(c["hl_s"])), 1),
                    hl_area_real=round(float(np.mean(c["ha_r"])), 4), hl_area_sim=round(float(np.mean(c["ha_s"])), 4))
            br, bs = np.mean(sh["br"], 0), np.mean(sh["bs"], 0)
            m["sharp"] = dict(real=round(float(np.median(sh["real"])), 1), sim=round(float(np.median(sh["sim"])), 1),
                              table_real=round(float(np.median(sh["treal"])), 1) if sh["treal"] else None,
                              table_sim=round(float(np.median(sh["tsim"])), 1) if sh["tsim"] else None,
                              band_log10=[round(float(np.log10(s / r)), 2) for s, r in zip(bs, br, strict=True)])
            cr, cs = np.cumsum(hist_r) / hist_r.sum(), np.cumsum(hist_s) / hist_s.sum()
            ctr = (np.arange(64) + 0.5) * 100 / 64
            m["hist"] = dict(dist=round(float(np.abs(cr - cs).mean() * 100), 2),
                             L_real=round(float((hist_r * ctr).sum() / hist_r.sum()), 1),
                             L_sim=round(float((hist_s * ctr).sum() / hist_s.sum()), 1))
            m["ssim"] = round(float(np.mean(ss)), 3) if ss else None
            feats[split] = (np.concatenate(fr), np.concatenate(fs)) if fr else None
            res[cam][split] = m
        if clf and feats.get("fit") and feats.get("held"):
            (a, b), (c, d) = feats["fit"], feats["held"]
            Xtr, ytr = np.r_[a, b], np.r_[np.ones(len(a)), np.zeros(len(b))].astype(np.float32)
            Xte, yte = np.r_[c, d], np.r_[np.ones(len(c)), np.zeros(len(d))].astype(np.float32)
            res[cam]["held"]["clf"] = round(logreg_acc(Xtr, ytr, Xte, yte), 3)
            res[cam]["held"]["clf_tex"] = round(logreg_acc(Xtr[:, 3:], ytr, Xte[:, 3:], yte), 3)
    return res


def score(m: dict, w=None) -> float:
    """Scalar loss of one camera / split (lower = closer to real)."""
    w = dict(color=1.0, sharp=6.0, hist=1.0, ssim=20.0, hl=0.15) | (w or {})
    wt = {"table": 2.0, "plates": 1.5, "bricks": 1.0, "robot": 1.0}
    s, n = 0.0, 0.0
    for rn, c in m["color"].items():
        s += wt[rn] * (c["dE"] + w["hl"] * abs(c["highlight_real"] - c["highlight_sim"]))
        n += wt[rn]
    out = w["color"] * s / max(n, 1e-6)
    sh = m["sharp"]
    out += w["sharp"] * abs(np.log(max(sh["sim"], 1e-3) / max(sh["real"], 1e-3)))
    out += w["hist"] * m["hist"]["dist"]
    if m.get("ssim") is not None:
        out += w["ssim"] * (1 - m["ssim"])
    return float(out)


def summary_row(tag: str, res: dict, split: str = "held") -> dict:
    """One compact row per run (the numbers logged in fit/log.md)."""
    row = {"tag": tag}
    for cam, r in res.items():
        m = r.get(split)
        if not m:
            continue
        row[cam] = dict(score=round(score(m), 2), dE={k: v["dE"] for k, v in m["color"].items()},
                        sharp=[m["sharp"]["sim"], m["sharp"]["real"]], hist=m["hist"]["dist"], ssim=m["ssim"],
                        clf=m.get("clf"), clf_tex=m.get("clf_tex"),
                        L=[m["hist"]["L_sim"], m["hist"]["L_real"]])
    return row


def load_eval():
    """(samples, {cam: real frames}) of the evaluation set (make_evalset.py)."""
    meta = json.loads((OUT / "evalset.json").read_text())
    z = np.load(OUT / "evalset.npz")
    return meta["samples"], {c: z[c] for c in CAMS}


def apply_ccm(sim: dict, path=None) -> dict:
    """Apply the camera response (twin_scene.color_correct, camera_color.json or `path`) to raw renders."""
    sys.path.insert(0, str(REPO))
    from sim.twin.visuals.twin_scene import CAMERA_COLOR_JSON, color_correct

    return {c: np.stack([color_correct(f, c, Path(path) if path else CAMERA_COLOR_JSON) for f in v])
            for c, v in sim.items() if c in CAMS}


def main():
    """CLI: metrics of one render npz against the evaluation set."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True, help="render npz (raw renders + masks)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--ccm", default=None, help="camera_color.json to apply (default: the repo's); 'none' = raw")
    a = ap.parse_args()
    samples, real = load_eval()
    z = np.load(a.sim)
    sim = {c: z[c] for c in CAMS if c in z}
    masks = {c: z["mask_" + c] for c in sim}
    if a.ccm != "none":
        sim = apply_ccm(sim, a.ccm)
    res = evaluate(real, sim, masks, samples)
    tag = a.tag or Path(a.sim).stem.replace("render_", "")
    (OUT / f"metrics_{tag}.json").write_text(json.dumps(res, indent=1))
    for split in ("fit", "held"):
        print(split, json.dumps(summary_row(tag, res, split)))
    print("EVAL_OK", OUT / f"metrics_{tag}.json")


if __name__ == "__main__":
    main()
