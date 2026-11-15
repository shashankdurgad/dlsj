"""Joint fit of the three cameras' colour responses with ONE shared albedo per plate / brick colour (plain python).

    python sim/twin/sensors/joint_color.py [--sim sim/twin/out/visual_fit/render_tex2.npz] [--iters 12] [--write]

Why: the real cameras record the same object differently (the lavender plate is sky-blue in the top D415, lavender in
the wrist / inspect cameras). One material albedo per object must map, through each camera's own response, to every
camera's recorded colour. Model per region (linear RGB means over the visual-fit evaluation set, fit split):

    real[c, k] = M_c . f(sim[c, k] * exp(d[class(k)]))      f = root-polynomial features (r, g, b, sqrt(rg), ...)

with sim[c, k] the twin's raw render of region k in camera c (lens + vignetting applied, the albedos of the render),
d the per-class, per-channel log albedo change (shared by all cameras; 0 for the neutral classes table / robot /
background, which anchor the gauge), M_c the 3 x 6 root-polynomial response of camera c (NO offset: black stays black,
ridge toward the identity). Alternating least squares: M_c given d (ridge), d given M (scipy least_squares, small
ridge). Classes: plate and brick components are labelled by the nearest colour of the render's palette
(visuals/twin_scene TwinLook plate_rgb / brick_rgb) in chromaticity.
--write: the joint responses go to visuals/camera_color.json (model "root_joint"); they are only consistent with the
twin once its plate / brick albedos are set to the joint ones, so write them together. Always writes
out/sensors/joint_albedo.json: per class the
albedo scale exp(d) (multiply the LINEAR albedo the render used) and the resulting sRGB albedo, with the per-class,
per-camera residual dE (CIE76 of region means, held-out samples) for identity / per-camera-only / joint fits.
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
from sim.twin.sensors import fit as F  # noqa: E402
from sim.twin.sensors import pipeline as pl  # noqa: E402

CAMS = ("top", "wrist", "inspect")
NEUTRAL = ("table", "robot", "background")


def lin_to_srgb(x):
    """Linear 0-1 -> sRGB 0-1."""
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def lab(lin_rgb):
    """Linear RGB (N, 3) -> CIE Lab (N, 3)."""
    s = lin_to_srgb(np.asarray(lin_rgb, np.float32)).astype(np.float32)[None]
    return cv2.cvtColor(s, cv2.COLOR_RGB2LAB)[0]


def palette() -> dict:
    """The render's plate / brick colours (sRGB 0-255) by class, from the current TwinLook."""
    from sim.twin.visuals.twin_scene import TwinLook

    look = TwinLook()
    return {"plate": dict(look.plate_rgb), "brick": dict(look.brick_rgb)}


def chroma(v):
    """Chromaticity of linear RGB vectors (rows sum to 1)."""
    v = np.clip(np.asarray(v, float), 1e-6, None)
    return v / v.sum(-1, keepdims=True)


def regions(real, sim, masks, cam: str, mdl_path: Path, none_path: Path, pal: dict):
    """Labelled region means: list of (class, sim linear mean, real linear mean, weight)."""
    pal_lin = {kind: {k: F.ms.srgb_lin(v) for k, v in d.items()} for kind, d in pal.items()}
    out = []
    for k in range(len(real)):
        s = F.lin(pl.apply(sim[k], cam, stateful=False, codec=False, model_path=mdl_path, color_path=none_path))
        r = F.lin(real[k])
        m = masks[k]
        regs = [("table", F.region(m, 1), 3.0), ("robot", F.region(m, 8), 1.5),
                ("background", cv2.erode((m == 0).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0, 1.0)]
        for bit, kind in ((2, "plate"), (4, "brick")):
            n, labl = cv2.connectedComponents(F.region(m, bit, 3).astype(np.uint8))
            for q in range(1, n):
                rm = labl == q
                if rm.sum() < 150:
                    continue
                c = chroma(s[rm].mean(0))
                names = list(pal_lin[kind])
                dist = [np.abs(c - chroma(pal_lin[kind][nm])).sum() for nm in names]
                regs.append((f"{kind}:{names[int(np.argmin(dist))]}", rm, 3.0))
        for cls, rm, w in regs:
            if rm.sum() < 150:
                continue
            a, b = s[rm].mean(0), r[rm].mean(0)
            if a.max() > 0.92 or b.max() > 0.92:
                continue
            out.append((cls, a, b, w))
    return out


def fit_joint(data: dict, iters: int, lam_d: float = 0.5):
    """Alternating least squares. data: {cam: [(cls, sim, real, w)]}. Returns (M per cam, d per class)."""
    from scipy.optimize import least_squares

    classes = sorted({c for v in data.values() for c, *_ in v if c not in NEUTRAL})
    d = {c: np.zeros(3) for c in classes}
    M = {}
    for _ in range(iters):
        for cam, rows in data.items():
            A = np.array([s * np.exp(d.get(c, np.zeros(3))) for c, s, _, _ in rows])
            B = np.array([r for _, _, r, _ in rows])
            W = np.array([w for *_, w in rows])
            M[cam], _ = F.solve_color(A, B, W, root=True)
        for cls in classes:
            obs = [(M[cam], s, r, w) for cam, rows in data.items() for c, s, r, w in rows if c == cls]

            def res(x, obs=obs):
                e = [np.sqrt(w) * (F.apply_color((s * np.exp(x))[None], Mc, np.zeros(3))[0] - r)
                     for Mc, s, r, w in obs]
                return np.r_[np.concatenate(e), np.sqrt(lam_d * len(obs) * 1e-3) * x]

            d[cls] = least_squares(res, d[cls], bounds=(-2.5, 2.5)).x
    return M, d


def residuals(data: dict, M: dict | None, d: dict | None) -> dict:
    """Median dE per class per camera of predicted vs real region means (identity when M is None)."""
    out: dict = {}
    for cam, rows in data.items():
        per: dict = {}
        for c, s, r, _ in rows:
            x = s * np.exp((d or {}).get(c, np.zeros(3)))
            p = x if M is None else F.apply_color(x[None], M[cam], np.zeros(3))[0]
            per.setdefault(c, []).append(float(np.linalg.norm(lab(p[None])[0] - lab(r[None])[0])))
        out[cam] = {c: round(float(np.median(v)), 2) for c, v in sorted(per.items())}
    return out


def main():
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sim", default=str(F.VF / "render_tex2.npz"))
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--write", action="store_true", help="write the joint responses to camera_color.json")
    a = ap.parse_args()
    meta = json.loads((F.VF / "evalset.json").read_text())["samples"]
    real_all, sim_all = np.load(F.VF / "evalset.npz"), np.load(a.sim)
    split = {s: [m["i"] for m in meta if m["split"] == s] for s in ("fit", "held")}
    model_all = json.loads(pl.MODEL_JSON.read_text())
    pal = palette()
    data: dict = {"fit": {}, "held": {}}
    for cam in CAMS:
        mdl = model_all.get(cam, {})
        lens = mdl.get("lens") or F.lens_model(cam)
        tmp, none = F.OUT / f"_j_{cam}.json", F.OUT / f"_jn_{cam}.json"
        tmp.write_text(json.dumps({cam: {"lens": lens, "vignette": mdl.get("vignette")}}))
        none.write_text("{}")
        mp = pl.lens_map(480, 640, lens)
        masks = np.stack([cv2.remap(m, mp[..., 0], mp[..., 1], cv2.INTER_NEAREST) for m in sim_all[f"mask_{cam}"]])
        for sp, ids in split.items():
            data[sp][cam] = regions(real_all[cam][ids], sim_all[cam][ids], masks[ids], cam, tmp, none, pal)
        tmp.unlink()
        none.unlink()
        print(f"[joint] {cam}: {len(data['fit'][cam])} fit / {len(data['held'][cam])} held regions", flush=True)
    M_sep, _ = fit_joint(data["fit"], iters=1, lam_d=1e9)  # per-camera only (albedos fixed)
    M, d = fit_joint(data["fit"], iters=a.iters)
    rep = {"residual_dE_held": {"identity": residuals(data["held"], None, None),
                                "per_camera": residuals(data["held"], M_sep, None),
                                "joint": residuals(data["held"], M, d)},
           "residual_dE_fit_joint": residuals(data["fit"], M, d), "albedo": {}}
    for cls, x in d.items():
        kind, name = cls.split(":")
        base = np.asarray(pal[kind].get(name, (128, 128, 128)), float)
        new = lin_to_srgb(F.ms.srgb_lin(base) * np.exp(x)) * 255
        rep["albedo"][cls] = {"render_srgb": base.round(1).tolist(), "linear_scale": np.exp(x).round(4).tolist(),
                              "joint_srgb": new.round(1).tolist()}
    for k, v in rep["residual_dE_held"].items():
        print(f"[joint] held dE {k}: {json.dumps(v)}", flush=True)
    print(f"[joint] albedo: {json.dumps(rep['albedo'])}", flush=True)
    (F.OUT / "joint_albedo.json").write_text(json.dumps(rep, indent=1))
    if a.write:
        col = json.loads(pl.COLOR_JSON.read_text()) if pl.COLOR_JSON.exists() else {}
        for cam in CAMS:
            keep = {k: v for k, v in col.get(cam, {}).items() if k in ("real_class_rgb", "n", "top_class_rgb")}
            col[cam] = dict(keep, M_root=M[cam].round(5).tolist(), M_linear=pl.root_to_linear(M[cam]),
                            offset_linear=[0.0, 0.0, 0.0], model="root_joint",
                            albedo_scales="out/sensors/joint_albedo.json")
        pl.write_json_atomic(pl.COLOR_JSON, col)


if __name__ == "__main__":
    main()
