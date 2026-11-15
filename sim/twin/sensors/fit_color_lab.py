"""Perceptual refit of a camera's colour response so saturated plates / bricks keep their chroma (plain python).

    python sim/twin/sensors/fit_color_lab.py [--cams wrist,inspect] [--terms 13] [--write]

The region-mean fit in fit.py (linear RGB, ridge toward the identity) is dominated by the table and the greys and
leaves saturated colours washed out in the wrist / inspect cameras (purple plate dE ~20). This fit minimises CIE Lab
error of the camera's whole deterministic chain, per region:

    lin sim (lens + vignette) -> M_root . root_features -> x exposure_gain -> sRGB -> tone (black, gamma, saturation)

against the real frames, with the twin's current look (visual-fit sweep_final_p*.npz raw renders of held-out
evaluation samples, *_1 = the final look). Regions: every plate and brick component (labelled by the materials'
palette), the table, the robot, the background. Real targets are robust: the median Lab of the real pixels inside
the dilated sim region whose chroma C* > 10 and hue lies within 60 deg of the sim region's current hue for saturated
classes (the episode layouts are approximate, so a plain region mean mixes in the table), the eroded-region median
for neutrals. Weights (--weights overrides): plates 4, bricks 3, table 3, robot 1, background 0.5; soft-L1 loss;
ridge toward the current response. Samples are split by episode (even fit / odd evaluation); the evaluation half
reports per-class dE (median) and the chroma statistics of the full pipeline (codec included), current vs refit.
--write puts the refit M_root into visuals/camera_color.json atomically: 13 terms as `M_root3` with 3x6 `M_root` and
3x3 `M_linear` stand-ins (older readers keep working).
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

WEIGHT = {"plate": 4.0, "brick": 3.0, "table": 3.0, "robot": 1.0, "background": 0.5}


def lab(srgb01):
    """sRGB 0-1 (N, 3) -> Lab (N, 3)."""
    return cv2.cvtColor(np.asarray(srgb01, np.float32)[None], cv2.COLOR_RGB2LAB)[0]


def chain(lin_sim, M, gain: float, tone: dict):
    """Region linear RGB -> displayed Lab through colour, exposure and tone (the pipeline's deterministic stages)."""
    x = np.clip(pl.root_features_np(lin_sim, M.shape[1]) @ M.T * gain, 0, 1)
    y = np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)
    b = float(tone.get("black", 0.0))
    y = np.clip((y - b) / (1 - b), 0, None) ** float(tone.get("gamma", 1.0))
    Y = (y * np.array(pl.LUMA)).sum(1, keepdims=True)
    y = np.clip(Y + float(tone.get("saturation", 1.0)) * (y - Y), 0, 1)
    return lab(y)


def hue(L):
    """Lab hue angle (deg)."""
    return np.degrees(np.arctan2(L[..., 2], L[..., 1]))


def regions(real, raw, mask, cur_out, cam: str, mdl_path: Path, pal: dict):
    """Labelled regions of one sample (mask and cur_out in real-camera pixels, raw = the render): [(kind, class,
    sim linear median, real Lab target)]; pal = linear chromaticity per class."""
    s = F.lin(pl.apply(raw, cam, stateful=False, stages={"lens", "vignette"}, model_path=mdl_path))
    lr = cv2.cvtColor(real.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    lc = cv2.cvtColor(cur_out.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    out = []
    for kind, rm in (("table", F.region(mask, 1, 9)), ("robot", F.region(mask, 8, 7)),
                     ("background", cv2.erode((mask == 0).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0)):
        if rm.sum() > 1500:
            out.append((kind, kind, np.median(s[rm], 0), np.median(lr[rm], 0)))
    for bit, kind, er, mn in ((2, "plate", 7, 800), (4, "brick", 3, 150)):
        n, lbl = cv2.connectedComponents(F.region(mask, bit, er).astype(np.uint8))
        for q in range(1, n):
            cm = lbl == q
            if cm.sum() < mn:
                continue
            ms = np.median(s[cm], 0)
            ms = ms / max(float(ms.sum()), 1e-6)
            name = min(pal[kind], key=lambda c: float(np.abs(pal[kind][c] - ms).sum()))
            h0 = np.median(hue(lc[cm]))
            dm = cv2.dilate(cm.astype(np.uint8), np.ones((21, 21), np.uint8)) > 0
            dh = np.abs((hue(lr) - h0 + 180) % 360 - 180)
            gate = dm & (np.hypot(lr[..., 1], lr[..., 2]) > 10) & (dh < 60) & (lr[..., 0] > 15) & (lr[..., 0] < 97)
            if gate.sum() < max(60, cm.sum() // 5):
                continue
            out.append((kind, f"{kind}:{name}", np.median(s[cm], 0), np.median(lr[gate], 0)))
    return out


def fit_M(rows, M0, gain, tone, lam: float = 0.02):
    """Soft-L1 Lab fit of M (3 x terms) starting from / regularised toward M0."""
    from scipy.optimize import least_squares

    S = np.array([r[2] for r in rows])
    T = np.array([r[3] for r in rows])
    w = np.sqrt(np.array([WEIGHT[r[0]] for r in rows]))[:, None]
    sh = M0.shape

    def res(p):
        M = p.reshape(sh)
        return np.r_[((chain(S, M, gain, tone) - T) * w).ravel(), np.sqrt(lam * len(rows)) * (p - M0.ravel())]

    # diff_step: the Lab conversion is float32 (cv2), the default finite-difference step is below its resolution
    return least_squares(res, M0.ravel(), loss="soft_l1", f_scale=3.0, diff_step=1e-3, max_nfev=4000).x.reshape(sh)


def class_dE(rows, M, gain, tone) -> dict:
    """Median dE per class (and per kind) of the chain vs the real targets."""
    S = np.array([r[2] for r in rows])
    T = np.array([r[3] for r in rows])
    d = np.linalg.norm(chain(S, M, gain, tone) - T, axis=1)
    out: dict = {}
    for (kind, cls, *_), e in zip(rows, d, strict=True):
        out.setdefault(cls, []).append(e)
        if kind in ("plate", "brick"):
            out.setdefault(f"ALL_{kind}", []).append(e)
    return {k: [round(float(np.median(v)), 2), len(v)] for k, v in sorted(out.items())}


def main():
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cams", default="wrist,inspect")
    ap.add_argument("--terms", type=int, default=13, choices=(6, 13))
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--lam", type=float, default=0.02, help="ridge toward the current response")
    ap.add_argument("--weights", default=None, help='JSON overrides of WEIGHT, e.g. {"brick": 6, "plate": 2}')
    a = ap.parse_args()
    if a.weights:
        WEIGHT.update(json.loads(a.weights))
    meta = json.loads((F.VF / "evalset.json").read_text())["samples"]
    real_all = np.load(F.VF / "evalset.npz")
    zs = [np.load(p) for p in sorted(F.VF.glob("sweep_final_p*.npz"))]
    ids = np.concatenate([z["ids"] for z in zs])
    from sim.twin.visuals.twin_scene import TwinLook

    look = TwinLook()  # the fitted look (materials.json): the palette the sweep renders used

    def chroma_lin(rgb255):
        v = F.ms.srgb_lin(rgb255)
        return v / v.sum()

    pal = {"plate": {k: chroma_lin(v) for k, v in look.plate_rgb.items()},
           "brick": {k: chroma_lin(v) for k, v in look.brick_rgb.items()}}
    models = json.loads(pl.MODEL_JSON.read_text())
    col = json.loads(pl.COLOR_JSON.read_text())
    rep = {"samples": int(len(ids)), "cams": {}}
    for cam in a.cams.split(","):
        m = models[cam]
        raw = np.concatenate([z[f"raw_{cam}_1"] for z in zs])
        mp = pl.lens_map(480, 640, m["lens"])
        masks = np.stack([cv2.remap(x, mp[..., 0], mp[..., 1], cv2.INTER_NEAREST)
                          for x in np.concatenate([z[f"mask_{cam}"] for z in zs])])
        tmp = F.OUT / f"_cl_{cam}.json"
        tmp.write_text(json.dumps({cam: {"lens": m["lens"], "vignette": m.get("vignette")}}))
        gain, tone = float(m.get("exposure_gain") or 1.0), m.get("tone", {})
        cur = np.asarray(col[cam].get("M_root3") or col[cam].get("M_root") or col[cam]["M_linear"], float)
        if cur.shape[1] == 3:
            cur = np.c_[cur, np.zeros((3, 3))]
        M0 = np.c_[cur, np.zeros((3, a.terms - cur.shape[1]))] if cur.shape[1] < a.terms else cur[:, :a.terms]
        rows = {"fit": [], "eval": []}
        outs = {}
        for k, i in enumerate(ids):
            o = pl.apply(raw[k], cam, stateful=False, codec=False)
            sp = "fit" if meta[int(i)]["ep"] % 2 == 0 else "eval"
            rows[sp] += regions(real_all[cam][int(i)], raw[k], masks[k], o, cam, tmp, pal)
            outs[k] = sp
        M = fit_M(rows["fit"], M0, gain, tone, lam=a.lam)
        r = {"n_regions": {s: len(v) for s, v in rows.items()},
             "eval_dE_current": class_dE(rows["eval"], M0, gain, tone),
             "eval_dE_refit": class_dE(rows["eval"], M, gain, tone),
             "fit_dE_refit": class_dE(rows["fit"], M, gain, tone)}
        # full pipeline (codec included) chroma statistics on the evaluation half, current vs refit
        ev = [k for k, sp in outs.items() if sp == "eval"]
        newc = dict(col)
        full = {"M_root3": M.round(5).tolist(), "M_root": pl.root_project(M, 6)} if a.terms == 13 else \
            {"M_root": M.round(5).tolist()}
        newc[cam] = dict({k: v for k, v in col[cam].items() if k != "M_root3"}, **full, M_linear=pl.root_project(M, 3),
                         offset_linear=[0.0, 0.0, 0.0], model=f"root{a.terms}_lab")
        cpath = F.OUT / f"_cl_color_{cam}.json"
        cpath.write_text(json.dumps(newc))
        stats = {}
        for tag, cp in (("current", None), ("refit", cpath)):
            fr = [pl.apply(raw[k], cam, stateful=False, color_path=cp, rng=np.random.default_rng(0)) for k in ev]
            luma = round(float(np.mean([F.ms.gray(f).mean() for f in fr])), 1)
            stats[tag] = dict(F.contrast(fr, masks[ev]), luma=luma)
        stats["real"] = dict(F.contrast(real_all[cam][ids[ev]], masks[ev]),
                             luma=round(float(np.mean([F.ms.gray(f).mean() for f in real_all[cam][ids[ev]]])), 1))
        r["eval_pipeline_stats"] = stats
        rep["cams"][cam] = r
        print(f"[lab] {cam} {json.dumps(r)}", flush=True)
        if a.write:
            col = json.loads(pl.COLOR_JSON.read_text())  # re-read: merge into the file as it is now
            col[cam] = newc[cam]
            pl.write_json_atomic(pl.COLOR_JSON, col)
        for p in (tmp, cpath):
            p.unlink(missing_ok=True)
    (F.OUT / "fit_color_lab.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
