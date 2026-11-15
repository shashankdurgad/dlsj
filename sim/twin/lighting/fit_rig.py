"""Fit the twin's lights to the real top / inspect frames from the light-basis renders (plain Python, no Isaac).

    python sim/twin/lighting/fit_rig.py [--basis v1] [--cams top inspect] [--max-lights 6]

Model (linear RGB, per channel c, per sample s, per pixel p of the table / plates / bricks / robot regions):

    real[s, c, p] = g[s, c] * (B_zero[s, c, p] + sum_k w[k, c] * B_k[s, c, p])

B_k = basis_render.py's one-light renders at the sample's recorded state, w[k, c] >= 0 the light weights shared by all
samples (their colour = the light's colour), g[s, c] a free per-sample, per-channel gain: the real RealSenses ran auto
exposure and auto white balance, so only the lights' RELATIVE strengths, directions and colours are identifiable.
Alternating least squares: NNLS over w for fixed g, closed-form g for fixed w. Both images are blurred
(--blur px at the 160 x 120 fit resolution) because the sim / real camera poses differ by a few pixels.
Fitted on the 'fit' split,
reported on the held-out episodes (only g is re-fitted there, as the camera would).

Writes sim/twin/out/lighting/fit_<basis>.json (weights per basis light, the clustered rig proposal, metrics) and
fit_<basis>_dirs.png (fitted weight per direction).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import nnls

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
OUT = REPO / "sim/twin/out/lighting"
VF = REPO / "sim/twin/out/visual_fit"
REGION_BITS = 1 | 2 | 4 | 8
SIZE = {"top": (160, 120), "inspect": (160, 120)}  # fit resolution (both images are blurred anyway)


def to_lin(img):
    """sRGB uint8 -> linear float32."""
    c = np.asarray(img, np.float32) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def to_srgb(lin):
    """Linear float -> sRGB uint8."""
    c = np.clip(lin, 0.0, 1.0)
    return (np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055) * 255.0 + 0.5).astype(np.uint8)


def load(basis_tag: str, cam: str):
    """Basis renders (S, K, H, W, 3) linear, real frames (S, H, W, 3) linear, region masks (S, H, W) and metadata."""
    z = np.load(OUT / f"basis_{basis_tag}.npz")
    ids = z["ids"]
    meta = json.loads((VF / "evalset.json").read_text())["samples"]
    real_all = np.load(VF / "evalset.npz")[cam]
    masks_all = np.load(VF / "render_before.npz")[f"mask_{cam}"]
    w, h = SIZE[cam]
    real = np.stack([cv2.resize(real_all[i], (w, h), interpolation=cv2.INTER_AREA) for i in ids])
    mask = np.stack([cv2.resize(masks_all[i], (w, h), interpolation=cv2.INTER_NEAREST) for i in ids])
    B = z[cam]
    if B.shape[3] != w:
        B = np.stack([[cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA) for b in row] for row in B])
    return dict(B=to_lin(B), real=to_lin(real), real8=real, mask=mask, ids=ids, names=[str(n) for n in z["names"]],
                dirs=z["dirs"], intensity=z["intensity"], samples=[meta[i] for i in ids])


def valid_mask(d: dict, erode: int = 2) -> np.ndarray:
    """(S, H, W) pixels used by the fit: inside the regions, away from region borders, not saturated in the real
    frame."""
    out = []
    for m, r in zip(d["mask"], d["real8"], strict=True):
        ok = np.zeros(m.shape, bool)
        for bit in (1, 2, 4, 8):
            reg = ((m & bit) > 0).astype(np.uint8)
            ok |= cv2.erode(reg, np.ones((2 * erode + 1,) * 2, np.uint8)) > 0
        ok &= r.max(-1) < 250
        out.append(ok)
    return np.stack(out)


def blur(x, sigma: float):
    """Gaussian blur over the two image axes of (..., H, W, 3)."""
    if sigma <= 0:
        return x
    sh = x.shape
    flat = x.reshape(-1, *sh[-3:])
    return np.stack([cv2.GaussianBlur(f, (0, 0), sigma) for f in flat]).reshape(sh)


def fit(B, real, ok, sel, iters: int = 8, step: int = 2, fixed_w=None):
    """Alternating NNLS. B (S, K, H, W, 3) with B[:, 0] = the zero (emitters-only) basis, real (S, H, W, 3),
    ok (S, H, W),
    sel = sample indices used. Returns w (K, 3) (w[0] = 1) and g (S, 3)."""
    S, K = B.shape[:2]
    pix = [np.flatnonzero(ok[s][::step, ::step].ravel()) for s in range(S)]
    Bs = B[:, :, ::step, ::step].reshape(S, K, -1, 3)
    Rs = real[:, ::step, ::step].reshape(S, -1, 3)
    g = np.ones((S, 3))
    w = np.ones((K, 3)) * 0.05 if fixed_w is None else np.array(fixed_w, float)
    w[0] = 1.0
    for _ in range(iters if fixed_w is None else 1):
        if fixed_w is None:
            for c in range(3):
                A = np.concatenate([g[s, c] * Bs[s, 1:, pix[s], c] for s in sel])  # (P, K-1)
                y = np.concatenate([Rs[s, pix[s], c] - g[s, c] * Bs[s, 0, pix[s], c] for s in sel])
                w[1:, c] = nnls(A, y, maxiter=5000)[0]
        for s in range(S):
            for c in range(3):
                m = np.tensordot(w[:, c], Bs[s, :, pix[s], c].T, 1)
                g[s, c] = float(m @ Rs[s, pix[s], c] / max(m @ m, 1e-9))
    return w, g


def compose(B, w, g):
    """Model images (S, H, W, 3) linear from basis B, weights w (K, 3), gains g (S, 3)."""
    return np.einsum("skhwc,kc->shwc", B, w) * g[:, None, None, :]


def residual(model, real, ok, sel) -> float:
    """Relative RMS error of the linear images over the valid pixels of the samples `sel`."""
    e = [(model[s][ok[s]] - real[s][ok[s]]) for s in sel]
    r = [real[s][ok[s]] for s in sel]
    return float(np.sqrt(np.mean(np.concatenate(e) ** 2)) / np.sqrt(np.mean(np.concatenate(r) ** 2)))


def cluster(names, dirs, w, max_lights: int, min_share: float = 0.03) -> list[dict]:
    """Merge neighbouring basis directions into lights: greedy over the strongest, absorbing basis lights within 20 deg
    (distant) / the same elevation ring neighbours (side). Share = fraction of the total direct light (luminance)."""
    lum = w @ np.array([0.2126, 0.7152, 0.0722])
    idx = [k for k, n in enumerate(names) if n[0] in "DS"]
    total = float(sum(lum[k] for k in idx) + lum[names.index("dome")])
    left = set(idx)
    out = []
    while left and len(out) < max_lights:
        k0 = max(left, key=lambda k: lum[k])
        if lum[k0] / total < min_share:
            break
        grp = [k for k in left if names[k][0] == names[k0][0] and
               np.degrees(np.arccos(np.clip(dirs[k] @ dirs[k0], -1, 1))) < 20.0]
        left -= set(grp)
        wt = np.array([lum[k] for k in grp])
        d = (wt[:, None] * dirs[grp]).sum(0)
        d /= np.linalg.norm(d)
        rgb = w[grp].sum(0)
        out.append(dict(kind="side" if names[k0][0] == "S" else "overhead", members=[names[k] for k in grp],
                        dir=d.round(4).tolist(), elevation_deg=round(float(np.degrees(np.arcsin(d[2]))), 1),
                        azimuth_deg=round(float(np.degrees(np.arctan2(d[1], d[0]))), 1),
                        share=round(float(wt.sum() / total), 3), weight_rgb=rgb.round(4).tolist(),
                        rgb_norm=(rgb / max(rgb.max(), 1e-9)).round(3).tolist()))
    out.append(dict(kind="dome", share=round(float(lum[names.index("dome")] / total), 3),
                    weight_rgb=w[names.index("dome")].round(4).tolist()))
    return out


def dir_plot(names, dirs, w, path: Path):
    """Polar map of the fitted weight per basis light (centre = zenith, radius = 90 - elevation; +x env up)."""
    img = np.full((420, 420, 3), 255, np.uint8)
    lum = w @ np.array([0.2126, 0.7152, 0.0722])
    mx = max(lum[2:].max(), 1e-9)
    for r in (15, 30, 45, 60, 75):
        cv2.circle(img, (210, 210), int(r * 2.3), (220, 220, 220), 1)
    for k, n in enumerate(names):
        if n[0] not in "DS":
            continue
        el = np.degrees(np.arcsin(dirs[k][2]))
        az = np.arctan2(dirs[k][1], dirs[k][0])
        rr = (90 - el) * 2.3
        p = (int(210 + rr * np.sin(az)), int(210 - rr * np.cos(az)))
        rad = int(3 + 16 * lum[k] / mx)
        cv2.circle(img, p, rad, (0, 0, 0) if n[0] == "D" else (0, 0, 200), -1 if lum[k] > 0 else 1)
    cv2.putText(img, "+x (table)", (175, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(img, "+y", (395, 214), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.imwrite(str(path), img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis", default="v1")
    ap.add_argument("--cams", nargs="*", default=["top", "inspect"])
    ap.add_argument("--blur", type=float, default=1.2, help="px at the fit resolution")
    ap.add_argument("--max-lights", type=int, default=6)
    args = ap.parse_args()
    res = {}
    for cam in args.cams:
        d = load(args.basis, cam)
        ok = valid_mask(d)
        B = blur(d["B"], args.blur)
        R = blur(d["real"], args.blur)
        split = np.array([s["split"] for s in d["samples"]])
        fit_ids, held = np.flatnonzero(split == "fit"), np.flatnonzero(split == "held")
        w, g = fit(B, R, ok, fit_ids)
        _, g_all = fit(B, R, ok, np.arange(len(split)), fixed_w=w)  # held-out: gains only (auto exposure / WB)
        model = compose(B, w, g_all)
        # baseline: the same model with the dome alone (+ emitters), and every light at equal weight
        w_dome = np.zeros_like(w)
        w_dome[0] = 1.0
        w_dome[d["names"].index("dome")] = 1.0
        _, g_d = fit(B, R, ok, np.arange(len(split)), fixed_w=w_dome)
        res[cam] = dict(weights={n: w[k].round(5).tolist() for k, n in enumerate(d["names"])},
                        rig=cluster(d["names"], d["dirs"], w, args.max_lights),
                        rel_rmse=dict(fit=residual(model, R, ok, fit_ids), held=residual(model, R, ok, held),
                                      dome_only_held=residual(compose(B, w_dome, g_d), R, ok, held)),
                        gains=g_all.round(4).tolist())
        dir_plot(d["names"], d["dirs"], w, OUT / f"fit_{args.basis}_{cam}_dirs.png")
        print(cam, json.dumps(res[cam]["rel_rmse"]), flush=True)
        for L in res[cam]["rig"]:
            print("  ", json.dumps(L), flush=True)
        np.savez_compressed(OUT / f"fit_{args.basis}_{cam}_model.npz", model=to_srgb(model), real=d["real8"],
                            ids=d["ids"])
    (OUT / f"fit_{args.basis}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
