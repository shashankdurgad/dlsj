"""Image-level realism metrics of one audit run: DINOv2 distances, real-vs-sim classifiers, matched-region statistics.

    python sim/twin/audit/perceptual.py --run sim/twin/out/audit/runs/<stamp>      # plain python, GPU

(b) DINOv2 ViT-B/14 (facebook/dinov2-base from the HF hub, cached in sim/twin/out/audit/hf), images at 448 x 588:
    * KID (unbiased polynomial-kernel MMD^2 on the CLS feature; mean +- sd over 200 subsets of 40) real-vs-sim per
      camera, with the real-vs-real baseline (audit frames vs the disjoint reference episodes);
    * paired cosine distance of the CLS feature and of the patch tokens: real-vs-sim, the floor (next real frame) and
      the ceiling (other episode, same phase); the per-patch distance map shows WHERE the images differ.
(d) real-vs-sim classifiers, 5 folds split by episode (never the same episode in train and test), with the same
    pipeline on real-vs-real (audit vs reference frames) as the control that should sit at chance:
    * a linear probe on DINOv2 patch tokens (semantic / appearance cues): patch and image accuracy, AUC, and the
      held-out per-patch P(sim) map as saliency;
    * a small CNN on raw 32 px crops, on RGB and on high-pass-only input (noise / texture / codec cues).
(c) matched regions from the sim's own geometry masks (table, plates, bricks, robot, props, background; eroded):
    Lab mean difference (dE, dL, da, db), contrast, sharpness (mean |Laplacian|, sim / real), fine-noise sigma, plus
    whole-image exposure, clipping, high-frequency energy, chroma sharpness and the real-vs-sim image shift from
    phase correlation of the gradient images (calibration / mount error in pixels).
Writes <run>/perceptual.json and perceptual.npz (maps for the report).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from sim.twin.audit.common import BITS, CAMS, OUT, PHASES, SET, jdump, load_set

GH, GW = 32, 42  # DINOv2 patch grid at 448 x 588
FOLDS = 5


# ------------------------------------------------------------------------------------------------ DINOv2
def dino_features(sets: dict, device: str) -> dict:
    """{name: (N, H, W, 3) uint8} -> {name: (cls (N, D), patches (N, GH * GW, D))} float32 (L2-normalised patches)."""
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained("facebook/dinov2-base", cache_dir=str(OUT / "hf")).to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    out = {}
    with torch.no_grad():
        for name, imgs in sets.items():
            cls, pat = [], []
            for i in range(0, len(imgs), 8):
                x = torch.from_numpy(imgs[i:i + 8]).to(device).permute(0, 3, 1, 2).float() / 255.0
                x = torch.nn.functional.interpolate(x, size=(GH * 14, GW * 14), mode="bicubic", antialias=True)
                h = model(pixel_values=(x - mean) / std).last_hidden_state.float()
                cls.append(h[:, 0].cpu().numpy())
                pat.append(h[:, 1:].cpu().numpy())
            out[name] = (np.concatenate(cls), np.concatenate(pat))
    del model
    return out


def kid(x: np.ndarray, y: np.ndarray, subsets: int = 200, size: int = 40, seed: int = 0) -> dict:
    """Kernel Inception Distance on arbitrary features: unbiased MMD^2 with k(a, b) = (a.b / d + 1)^3."""
    rng = np.random.default_rng(seed)
    d = x.shape[1]
    size = min(size, len(x), len(y))
    vals = []
    for _ in range(subsets):
        a = x[rng.choice(len(x), size, replace=False)].astype(np.float64)
        b = y[rng.choice(len(y), size, replace=False)].astype(np.float64)
        kxx, kyy, kxy = (a @ a.T / d + 1) ** 3, (b @ b.T / d + 1) ** 3, (a @ b.T / d + 1) ** 3
        m = size
        vals.append((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
                    - 2 * kxy.mean())
    return {"kid": float(np.mean(vals)), "sd": float(np.std(vals))}


def cosd(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    return 1.0 - (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)


def stat(v) -> dict:
    v = np.asarray(v, np.float64)
    v = v[np.isfinite(v)]
    if not len(v):
        return {"mean": None, "median": None, "n": 0}
    return {"mean": float(v.mean()), "median": float(np.median(v)), "n": int(len(v))}


# ------------------------------------------------------------------------------------------------ classifiers
def folds_by_episode(samples) -> np.ndarray:
    eps = sorted({s["ep"] for s in samples})
    return np.array([eps.index(s["ep"]) % FOLDS for s in samples])


def auc(score, label) -> float:
    score, label = np.asarray(score, np.float64), np.asarray(label)
    order = np.argsort(score)
    rank = np.empty(len(score))
    rank[order] = np.arange(1, len(score) + 1)
    pos = label == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return float((rank[pos].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def patch_probe(p0: np.ndarray, p1: np.ndarray, fold: np.ndarray, device: str) -> dict:
    """Linear probe class 0 (p0) vs class 1 (p1) on patch tokens (N, P, D); returns held-out metrics + P(1) maps."""
    import torch

    N, P, D = p0.shape
    maps0, maps1 = np.zeros((N, P), np.float32), np.zeros((N, P), np.float32)
    for f in range(FOLDS):
        tr, te = fold != f, fold == f
        xtr = torch.from_numpy(np.concatenate([p0[tr].reshape(-1, D), p1[tr].reshape(-1, D)])).to(device)
        ytr = torch.cat([torch.zeros(int(tr.sum()) * P), torch.ones(int(tr.sum()) * P)]).to(device)
        mu, sd = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True) + 1e-6
        w = torch.zeros(D, device=device, requires_grad=True)
        b = torch.zeros(1, device=device, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=0.01)
        xn = (xtr - mu) / sd
        for _ in range(150):
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(xn @ w + b, ytr) + 1e-2 * (w ** 2).sum()
            loss.backward()
            opt.step()
        with torch.no_grad():
            for src, dst in ((p0, maps0), (p1, maps1)):
                x = (torch.from_numpy(src[te].reshape(-1, D)).to(device) - mu) / sd
                dst[te] = (x @ w + b).reshape(-1, P).cpu().numpy()
    logit = np.concatenate([maps0, maps1])
    y = np.concatenate([np.zeros(N), np.ones(N)])
    img = logit.mean(1)
    return {"patch_acc": float(((logit > 0) == (y[:, None] == 1)).mean()),
            "image_acc": float(((img > 0) == (y == 1)).mean()), "image_auc": auc(img, y),
            "maps0": 1 / (1 + np.exp(-np.clip(maps0, -60, 60))), "maps1": 1 / (1 + np.exp(-np.clip(maps1, -60, 60)))}


def highpass(x):
    """x (B, 3, h, w) float in [0, 1] -> x - 5x5 box blur (+0.5): only fine texture / noise is left."""
    import torch

    k = torch.ones(3, 1, 5, 5, device=x.device) / 25.0
    blur = torch.nn.functional.conv2d(torch.nn.functional.pad(x, (2, 2, 2, 2), mode="replicate"), k, groups=3)
    return (x - blur) * 4.0 + 0.5


def crop_cnn(i0: np.ndarray, i1: np.ndarray, fold: np.ndarray, device: str, hp: bool, steps: int = 300,
             size: int = 32) -> dict:
    """Small CNN on random `size` px crops, class 0 (i0) vs 1 (i1); held-out crop / image accuracy + P(1) grids."""
    import torch
    from torch import nn

    N, H, W, _ = i0.shape
    gh, gw = H // size, W // size
    t0 = torch.from_numpy(i0).to(device).permute(0, 3, 1, 2)
    t1 = torch.from_numpy(i1).to(device).permute(0, 3, 1, 2)
    maps0, maps1 = np.zeros((N, gh, gw), np.float32), np.zeros((N, gh, gw), np.float32)
    g = torch.Generator(device="cpu").manual_seed(0)

    def prep(x):
        x = x.float() / 255.0
        return highpass(x) if hp else x

    for f in range(FOLDS):
        torch.manual_seed(f)
        net = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
                            nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(), nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(),
                            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 1)).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        tr = np.flatnonzero(fold != f)
        for _ in range(steps):
            B = 256
            idx = torch.from_numpy(tr[torch.randint(len(tr), (B,), generator=g).numpy()])
            ys = torch.randint(H - size + 1, (B,), generator=g)
            xs = torch.randint(W - size + 1, (B,), generator=g)
            lab = torch.randint(2, (B,), generator=g)
            crops = torch.stack([(t1 if lab[k] else t0)[idx[k], :, ys[k]:ys[k] + size, xs[k]:xs[k] + size]
                                 for k in range(B)])
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(net(prep(crops)).squeeze(1),
                                                                  lab.float().to(device))
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            for t, dst in ((t0, maps0), (t1, maps1)):
                for i in np.flatnonzero(fold == f):
                    x = t[i, :, :gh * size, :gw * size].unfold(1, size, size).unfold(2, size, size)
                    x = x.permute(1, 2, 0, 3, 4).reshape(-1, 3, size, size)
                    dst[i] = net(prep(x)).reshape(gh, gw).cpu().numpy()
    logit = np.concatenate([maps0, maps1]).reshape(2 * N, -1)
    y = np.concatenate([np.zeros(N), np.ones(N)])
    img = logit.mean(1)
    return {"crop_acc": float(((logit > 0) == (y[:, None] == 1)).mean()),
            "image_acc": float(((img > 0) == (y == 1)).mean()), "image_auc": auc(img, y),
            "maps0": 1 / (1 + np.exp(-np.clip(maps0, -60, 60))), "maps1": 1 / (1 + np.exp(-np.clip(maps1, -60, 60)))}


# ------------------------------------------------------------------------------------------------ region statistics
def lab(img):
    return cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)


def noise_sigma(g, m) -> float:
    """Robust sigma of the fine residual (grey - 3x3 median) inside mask m (grey levels)."""
    r = (g - cv2.medianBlur(g.astype(np.uint8), 3).astype(np.float32))[m]
    return float(1.4826 * np.median(np.abs(r - np.median(r)))) if r.size else float("nan")


def hf_fraction(g) -> float:
    """Share of the grey image's spectral power above 0.25 cycles / px (fine detail + noise)."""
    f = np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean()))) ** 2
    h, w = g.shape
    yy, xx = np.mgrid[:h, :w]
    r = np.hypot((yy - h / 2) / h, (xx - w / 2) / w)
    return float(f[r > 0.25].sum() / max(f.sum(), 1e-9))


def shift(real, sim):
    """(dx, dy, response): translation of the sim image relative to the real one (gradient images)."""
    def grad(i):
        g = cv2.GaussianBlur(gray(i), (0, 0), 1.5)
        return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))

    a, b = grad(real), grad(sim)
    (dx, dy), resp = cv2.phaseCorrelate(a, b, cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F))
    return float(dx), float(dy), float(resp)


def plate_geo(real, mask, cam: str):
    """Where the real plates are vs the twin's plate mask (bit 2): the real plates are the large saturated blobs near
    the sim plates. Returns (IoU, dx, dy) in px (sim centroid - real centroid) or None."""
    ms = (mask & BITS["plates"]) > 0
    if ms.sum() < 500:
        return None
    hsv = cv2.cvtColor(real, cv2.COLOR_RGB2HSV)
    mr = cv2.morphologyEx(((hsv[..., 1] > 110) & (hsv[..., 2] > 120)).astype(np.uint8), cv2.MORPH_OPEN,
                          np.ones((7, 7), np.uint8))
    near = cv2.dilate(ms.astype(np.uint8), np.ones((81, 81), np.uint8)) > 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(mr)
    keep = np.zeros(ms.shape, bool)
    for k in range(1, n):
        blob = lab == k
        if st[k, 4] > (2500 if cam == "top" else 15000) and st[k, 1] > 40 and near[blob].mean() > 0.8:
            keep |= blob
    if keep.sum() < 500:
        return None
    ys, xs = np.nonzero(ms)
    yr, xr = np.nonzero(keep)
    return float((ms & keep).sum() / (ms | keep).sum()), float(xs.mean() - xr.mean()), float(ys.mean() - yr.mean())


def region_stats(real, sim, mask) -> dict:
    """Per-region paired statistics of one real / sim image pair; mask = sim geometry bits."""
    k = np.ones((7, 7), np.uint8)
    regions = {n: cv2.erode(((mask & b) > 0).astype(np.uint8), k) > 0 for n, b in BITS.items()}
    regions["background"] = cv2.erode((mask == 0).astype(np.uint8), k) > 0
    lr, ls, gr, gs = lab(real), lab(sim), gray(real), gray(sim)
    lapr, laps = np.abs(cv2.Laplacian(gr, cv2.CV_32F)), np.abs(cv2.Laplacian(gs, cv2.CV_32F))
    out = {}
    for n, m in regions.items():
        if m.sum() < 300:
            continue
        a, b = lr[m].mean(0), ls[m].mean(0)
        out[n] = {"px": int(m.sum()), "dE": float(np.linalg.norm(b - a)), "dL": float(b[0] - a[0]),
                  "da": float(b[1] - a[1]), "db": float(b[2] - a[2]),
                  "chroma_ratio": float(np.hypot(b[1], b[2]) / max(np.hypot(a[1], a[2]), 1e-3)),
                  "contrast_ratio": float(ls[m][:, 0].std() / max(lr[m][:, 0].std(), 1e-3)),
                  "sharp_ratio": float(laps[m].mean() / max(lapr[m].mean(), 1e-3)),
                  "noise_real": noise_sigma(gr, m), "noise_sim": noise_sigma(gs, m),
                  # highlight strength: how far the brightest 0.5 % of the region sits above its median (L*)
                  "hl_real": float(np.percentile(lr[m][:, 0], 99.5) - np.median(lr[m][:, 0])),
                  "hl_sim": float(np.percentile(ls[m][:, 0], 99.5) - np.median(ls[m][:, 0]))}
    ycr, ycs = cv2.cvtColor(real, cv2.COLOR_RGB2YCrCb), cv2.cvtColor(sim, cv2.COLOR_RGB2YCrCb)
    cl = [float(np.abs(cv2.Laplacian(y[..., 1:].astype(np.float32), cv2.CV_32F)).mean()) for y in (ycr, ycs)]
    dx, dy, resp = shift(real, sim)
    mxr, mxs = real.max(-1), sim.max(-1)
    out["_global"] = {"lum_real": float(lr[..., 0].mean()), "lum_sim": float(ls[..., 0].mean()),
                      "clip_hi_real": float((mxr >= 250).mean()), "clip_hi_sim": float((mxs >= 250).mean()),
                      "clip_lo_real": float((mxr <= 8).mean()), "clip_lo_sim": float((mxs <= 8).mean()),
                      "hf_real": hf_fraction(gr), "hf_sim": hf_fraction(gs),
                      "sharp_real": float(lapr.mean()), "sharp_sim": float(laps.mean()),
                      "chroma_sharp_real": cl[0], "chroma_sharp_sim": cl[1],
                      "sat_real": float(cv2.cvtColor(real, cv2.COLOR_RGB2HSV)[..., 1].mean()),
                      "sat_sim": float(cv2.cvtColor(sim, cv2.COLOR_RGB2HSV)[..., 1].mean()),
                      "mae": float(np.abs(real.astype(np.float32) - sim.astype(np.float32)).mean()),
                      "shift_dx": dx, "shift_dy": dy, "shift_px": float(np.hypot(dx, dy)), "shift_resp": resp}
    return out


def aggregate(rows: list[dict]) -> dict:
    keys = sorted({k for r in rows for k in r})
    out = {}
    for k in keys:
        sub = [r[k] for r in rows if k in r]
        out[k] = {f: stat([s[f] for s in sub])["median"] for f in sub[0] if f != "px"}
        out[k]["n"] = len(sub)
    return out


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    run = Path(ap.parse_args().run)
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    S = load_set()["samples"]
    simz = np.load(run / "sim.npz")
    N = len(simz["top"])
    dark = [i for c in CAMS for i in range(N) if float(simz[c][i].mean()) <= 2.0]
    if dark:
        sys.exit(f"PERCEPTUAL_ABORT {len(dark)} black sim images: not auditing black frames")
    S = S[:N]
    phase = np.array([s["phase"] for s in S])
    fold = folds_by_episode(S)
    real, nbr, ref = (np.load(SET / f"{k}.npz") for k in ("real", "nbr", "ref"))
    res = {"n": N, "device": device, "cams": {}}
    maps = {}
    for cam in CAMS:
        I = {"real": real[cam][:N], "sim": simz[cam], "nbr": nbr[cam][:N], "ref": ref[cam][:N]}  # noqa: E741
        r = {}
        try:
            F = dino_features(I, device)
            cr = F["real"][0]
            r["kid"] = {"real_vs_sim": kid(cr, F["sim"][0]), "real_vs_real": kid(cr, F["ref"][0]),
                        "real_vs_next_frame": kid(cr, F["nbr"][0])}
            r["paired_cls"] = {k: stat(cosd(cr, F[k][0])) for k in ("sim", "nbr", "ref")}
            pd = {k: cosd(F["real"][1], F[k][1]) for k in ("sim", "nbr", "ref")}  # (N, P)
            r["paired_patch"] = {k: stat(v.mean(1)) for k, v in pd.items()}
            r["paired_cls_by_phase"] = {p: float(cosd(cr, F["sim"][0])[phase == p].mean()) for p in PHASES}
            r["paired_patch_by_phase"] = {p: float(pd["sim"].mean(1)[phase == p].mean()) for p in PHASES}
            maps[f"patchdist_{cam}"] = pd["sim"].reshape(N, GH, GW).astype(np.float32)
            maps[f"patchdist_nbr_{cam}"] = pd["nbr"].reshape(N, GH, GW).mean(0).astype(np.float32)
            pp = patch_probe(F["real"][1], F["sim"][1], fold, device)
            ctl = patch_probe(F["real"][1], F["ref"][1], fold, device)
            maps[f"probe_sim_{cam}"] = pp.pop("maps1").reshape(N, GH, GW)
            maps[f"probe_real_{cam}"] = pp.pop("maps0").reshape(N, GH, GW)
            ctl.pop("maps0"), ctl.pop("maps1")
            r["probe_dino"] = {"real_vs_sim": pp, "control_real_vs_real": ctl}
            del F
        except Exception as e:  # noqa: BLE001
            import traceback

            r["dino_error"] = f"{type(e).__name__}: {e}"
            print("[perceptual] DINO part failed for", cam, "\n", traceback.format_exc(limit=4), flush=True)
        for hp in (False, True):
            key = "cnn_highpass" if hp else "cnn_rgb"
            cc = crop_cnn(I["real"], I["sim"], fold, device, hp)
            ctl = crop_cnn(I["real"], I["ref"], fold, device, hp)
            maps[f"{key}_sim_{cam}"] = cc.pop("maps1")
            maps[f"{key}_real_{cam}"] = cc.pop("maps0")
            ctl.pop("maps0"), ctl.pop("maps1")
            r[key] = {"real_vs_sim": cc, "control_real_vs_real": ctl}
        rows = [region_stats(I["real"][i], I["sim"][i], simz["mask_" + cam][i]) for i in range(N)]
        r["regions"] = aggregate(rows)
        r["regions_by_phase"] = {p: aggregate([rows[i] for i in range(N) if phase[i] == p]) for p in PHASES}
        geo = [g for g in (plate_geo(I["real"][i], simz["mask_" + cam][i], cam) for i in range(N)) if g]
        if geo:
            g = np.array(geo)
            r["plate_geo"] = {"n": len(g), "iou": float(np.median(g[:, 0])), "dx": float(np.median(g[:, 1])),
                              "dy": float(np.median(g[:, 2])), "offset": float(np.median(np.hypot(g[:, 1], g[:, 2])))}
        r["per_sample"] = {k: [x["_global"][k] for x in rows] for k in ("mae", "shift_px", "shift_resp")}
        # the same global statistics real-vs-next-real-frame: the floor of mae / shift
        fl = [region_stats(I["real"][i], I["nbr"][i], simz["mask_" + cam][i])["_global"] for i in range(N)]
        r["floor_next_frame"] = {k: stat([x[k] for x in fl])["median"] for k in ("mae", "shift_px", "shift_resp")}
        res["cams"][cam] = r
        print(f"[perceptual] {cam} done ({time.time() - t0:.0f}s)", flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    jdump(run / "perceptual.json", res)
    np.savez_compressed(run / "perceptual.npz", **maps)
    for cam in CAMS:
        c = res["cams"][cam]
        print(f"PERCEPTUAL {cam}: KID {c.get('kid', {}).get('real_vs_sim')} | probe "
              f"{ {k: round(v, 3) for k, v in c.get('probe_dino', {}).get('real_vs_sim', {}).items()} } | cnn rgb "
              f"{c['cnn_rgb']['real_vs_sim']['crop_acc']:.3f} hp {c['cnn_highpass']['real_vs_sim']['crop_acc']:.3f}",
              flush=True)
    print("PERCEPTUAL_OK", flush=True)


if __name__ == "__main__":
    main()
