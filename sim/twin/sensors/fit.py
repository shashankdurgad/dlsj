"""Fit the camera pipeline (pipeline.py) so twin frames have the real frames' statistics (plain python + PyAV).

    python sim/twin/sensors/fit.py [--sim sim/twin/out/visual_fit/render_tex2.npz] [--cams top,wrist,inspect]
                                   [--stages vignette,color,exposure,tone,mtf] [--vignette 0,0] [--no-write]
    (as fitted: top / inspect with --vignette=0,0, wrist with the vignette stage; one camera per process is fine,
    the JSON files are merged and written atomically)

Data: the visual fit's evaluation set (sim/twin/out/visual_fit/evalset.{npz,json}: real frames at five moments of
30 cal2 episodes, 2:1 fit / held-out split by episode) and the twin's RAW renders of the same samples with region
masks (render_<tag>.npz from visuals/fit/render_eval.py). Real-only targets: sensors/measure.py
(out/sensors/real_stats.json: temporal noise vs brightness) and the dataset's own I-frame packet sizes.
Stages per camera (fit samples only; every metric is reported on the held-out samples):
* vignette  radial gain 1 + v1 r^2 + v2 r^4 from the real / sim linear-luma ratio on the table region (wrist; the
            RealSenses use --vignette=0,0: their ratio profile is non-monotonic, i.e. lighting, not lens);
* color     every camera: linear 3x3 or root-polynomial 3x6 (no offset: black stays black) on region means of
            neutral AND saturated regions (table, robot, background; every plate and brick) after the lens and
            vignetting, ridge toward the identity; the model with the lower held-out error is kept ->
            camera_color.json (so one material albedo maps to each camera's recorded colour);
* exposure  wrist / inspect: the camera's exposure level, median over samples of (real / sim mean sRGB luma)^2.2
            after the colour matrix (top: 1, the lights are calibrated to it);
* tone      grid search of the ISP tone (black level, contrast exponent, saturation) on luma quantiles (p5-p95) and
            plate / brick chroma, after colour and exposure;
* mtf       grid search of blur_sigma, sharpen_amount and noise_mid through the full pipeline incl. the H.264 I-frame
            round-trip: radial power spectrum (bands 2-7), temporal noise at mid luma and I-frame size vs real.
Writes sensors/camera_model.json, out/sensors/fit_report_<cams>.json and side-by-sides out/sensors/sbs_<cam>.png
(real | sim raw | sim through the pipeline, with 2x crops).
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from sim.twin.sensors import measure as ms  # noqa: E402
from sim.twin.sensors import pipeline as pl  # noqa: E402
from sim.twin.sensors.codec import roundtrip  # noqa: E402

VF = REPO / "sim/twin/out/visual_fit"
OUT = REPO / "sim/twin/out/sensors"
CAMS = ("top", "wrist", "inspect")

# Measured on the real data (measure.py, exp: AE / AWB / flicker analysis in fit_report["measured"]); not fitted
# against the sim because they are temporal properties the evaluation set (single frames) cannot show.
MEASURED = {
    # AE / AWB: regression of a static region's linear luma (top / inspect: pixels the arm never covers; wrist: the
    # jaws, fixed in its view) on the luma of the changing content, I frames, lags 0-7 (0-0.47 s), 16 held-out
    # episodes. top 0.004-0.009, inspect 0.026 -> -0.003: no measurable response to the arm entering the view
    # (strength 0). wrist: +0.18 at lag 0 (shared light) turning to -0.17 at 4-7 I frames: a real AE that
    # compensates ~0.4 of the metered change with a ~0.3 s time constant. AWB: no compensating response on any
    # camera (static chroma follows the content chroma with slope >= 0), so awb_strength 0.
    # Flicker: I->I static-region gain pulse 0.13 % (top) / 0.45 % (inspect), no row banding above the noise
    # (fit_report["measured"]), so the flicker stage is off.
    # exposure_frames: wrist motion-blur anisotropy vs optical-flow speed -> ~30 ms (0.9 frame) exposure.
    "top": {"exposure_frames": 0.0, "ae": {"strength": 0.0, "tau": 6.0, "awb_strength": 0.0, "awb_tau": 10.0}},
    "inspect": {"exposure_frames": 0.0, "ae": {"strength": 0.0, "tau": 6.0, "awb_strength": 0.0, "awb_tau": 10.0}},
    "wrist": {"exposure_frames": 0.9, "ae": {"strength": 0.4, "tau": 9.0, "clip": (0.6, 1.6), "awb_strength": 0.0,
                                             "awb_tau": 10.0}},
}
GRID = {  # the rebuilt renders (render_tex2) are sharper than real for top / wrist, softer for inspect
    "top": {"blur_sigma": (0.0, 0.3, 0.5, 0.7, 0.9), "sharpen_amount": (0.0, 0.5, 1.0), "sharpen_radius": (1.0,),
            "noise_mid": (1.5, 2.5, 3.5)},
    "inspect": {"blur_sigma": (0.0, 0.3), "sharpen_amount": (0.0, 0.6, 1.2, 2.0, 3.0), "sharpen_radius": (1.0,),
                "noise_mid": (1.5, 2.5, 3.5)},
    "wrist": {"blur_sigma": (0.4, 0.7, 1.0, 1.3, 1.6), "sharpen_amount": (0.0,), "sharpen_radius": (1.0,),
              "noise_mid": (3.5, 4.5, 5.5)},
}
DR_HALF = {  # DR half-ranges at scale 1 (small shifts): log exposure, log WB per channel, log noise, + blur px,
    # log sharpening, crf offsets. Real spread for reference: top luma p5-p95 -3.5 % / +3.9 %, inspect +-9 %.
    "top": {"exposure": 0.05, "wb": 0.02, "noise": 0.3, "blur": 0.2, "sharpen": 0.25, "crf": (-2, 3)},
    "inspect": {"exposure": 0.08, "wb": 0.03, "noise": 0.3, "blur": 0.2, "sharpen": 0.25, "crf": (-2, 3)},
    "wrist": {"exposure": 0.10, "wb": 0.04, "noise": 0.3, "blur": 0.35, "sharpen": 0.3, "crf": (-2, 3)},
}


def lens_model(cam: str) -> dict:
    """Sim focal (the twin profile's) + the real intrinsics (factory RealSense / measured wrist) + wrist distortion."""
    if cam == "wrist":
        k = json.loads(pl.MEASURED_JSON.read_text())["wrist"]
        dist = [round(v, 5) for v in k["dist"]]
    else:
        k = json.loads((REPO / "sim/twin/calib/realsense_intrinsics.json").read_text())[cam]
        dist = [0.0] * 5
    try:
        from sim.twin.calib.profile import _cams

        f_sim = float(_cams()[cam][2])
    except Exception as e:  # noqa: BLE001
        print(f"[fit] {cam}: twin profile focal unavailable ({type(e).__name__}: {e}); using the real one", flush=True)
        f_sim = (k["fx"] + k["fy"]) / 2
    return {"f": round(f_sim, 3), "K_real": [round(float(k[c]), 3) for c in ("fx", "fy", "cx", "cy")], "dist": dist}


def lin(img):
    """sRGB uint8 -> linear float32."""
    return ms.srgb_lin(img).astype(np.float32)


def region(m, bit, erode=5):
    """Eroded bool mask of a region bit."""
    r = ((m & bit) > 0).astype(np.uint8)
    return cv2.erode(r, np.ones((erode, erode), np.uint8)) > 0 if erode else r > 0


def real_iframe_sizes(eps, ids_eps, cam: str) -> float:
    """Median I-frame packet size (bytes) of the dataset's stream of `cam` over the given episodes."""
    import av

    sizes = []
    for ep in sorted(set(ids_eps)):
        r = eps.loc[ep]
        k = f"videos/observation.images.{cam}"
        p = ms.DATA / k / f"chunk-{int(r[k + '/chunk_index']):03d}/file-{int(r[k + '/file_index']):03d}.mp4"
        t0, t1 = float(r[k + "/from_timestamp"]), float(r[k + "/to_timestamp"])
        with av.open(str(p)) as c:
            s = c.streams.video[0]
            c.seek(int(max(0.0, t0 - 0.2) / s.time_base), stream=s, backward=True)
            for pk in c.demux(s):
                if pk.pts is None:
                    continue
                t = float(pk.pts * s.time_base)
                if t >= t1:
                    break
                if t >= t0 and pk.is_keyframe:
                    sizes.append(pk.size)
    return float(np.median(sizes))


def real_noise(stats: dict, cam: str) -> float:
    """Real temporal I-I luma noise (DN) pooled over the 60-180 luma bins (sample-weighted)."""
    v, w = [], []
    for e in stats[cam]["noise_ii"]:
        for b, x in e["bins"].items():
            lo = int(b.split("-")[0])
            if 60 <= lo < 180:
                v.append(x["sigma_ycc"][0])
                w.append(x["n"])
    return float(np.average(v, weights=w))


def sim_noise(frames_a, frames_b) -> float:
    """Temporal noise (DN) of sim pairs (same render, independent noise) pooled over the 60-180 luma bins."""
    v, w = [], []
    for a, b in zip(frames_a, frames_b, strict=True):
        r = ms.temporal_noise(np.stack([a, b]), [(0, 1)])
        for k, x in r["bins"].items():
            lo = int(k.split("-")[0])
            if 60 <= lo < 180:
                v.append(x["sigma_ycc"][0])
                w.append(x["n"])
    return float(np.average(v, weights=w)) if v else 0.0


def spectrum(frames) -> np.ndarray:
    """Median radial log-power spectrum (measure.BANDS) over frames."""
    return np.median([ms.radial_spectrum(ms.gray(f)) for f in frames], 0)


def through(frames, cam: str, mdl: dict, rng, codec: bool = True, color_path: Path | None = None):
    """Frames through the pipeline with an explicit model dict (written to a temp JSON) and colour file, I frames
    (stateless)."""
    tmp = OUT / f"_model_{cam}.json"
    tmp.write_text(json.dumps({cam: mdl}))
    outs, sizes = [], []
    for f in frames:
        o = pl.apply(f, cam, rng=rng, stateful=False, codec=False, model_path=tmp, color_path=color_path)
        if codec:
            d, sz = roundtrip([o])
            outs.append(d[0])
            sizes.append(sz[0])
        else:
            outs.append(o)
    tmp.unlink(missing_ok=True)
    return outs, (float(np.median(sizes)) if sizes else None)


def fit_vignette(real, sim, masks, lens_mdl: dict, cam: str) -> list:
    """Radial gain (v1, v2) from the median real / sim luma ratio per r^2 bin on the table (+ background) region."""
    H, W = real.shape[1:3]
    yy, xx = np.mgrid[0:H, 0:W]
    r2 = ((xx - (W - 1) / 2) ** 2 + (yy - (H - 1) / 2) ** 2) / ((W / 2) ** 2 + (H / 2) ** 2)
    edges = np.linspace(0, 1.0, 11)
    per = []
    for k in range(len(real)):
        s = pl.apply(sim[k], cam, stateful=False, codec=False, model_path=lens_mdl)
        ok = region(masks[k], 1, 9)
        if ok.sum() < 5000:
            continue
        ratio = (lin(real[k]) @ ms_luma()) / np.maximum(lin(s) @ ms_luma(), 1e-3)
        ref = np.median(ratio[ok & (r2 < 0.2)]) if (ok & (r2 < 0.2)).sum() > 500 else np.median(ratio[ok])
        row = []
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            sel = ok & (r2 >= lo) & (r2 < hi)
            row.append(np.median(ratio[sel]) / ref if sel.sum() > 300 else np.nan)
        per.append(row)
    prof = np.nanmedian(np.array(per), 0)
    c = (edges[:-1] + edges[1:]) / 2
    ok = np.isfinite(prof)
    A = np.c_[c[ok], c[ok] ** 2]
    v = np.linalg.lstsq(A, prof[ok] - 1.0, rcond=None)[0]
    corner = 1 + v[0] + v[1]
    if corner < 0.45:  # keep it physical (the sim's light fall-off differences end up here too)
        v = v * (0.55 / max(1e-3, 1 - corner))
    return [round(float(v[0]), 4), round(float(v[1]), 4)], np.round(prof, 3).tolist(), len(per)


def ms_luma():
    """Rec.709 luma weights."""
    return np.array(pl.LUMA, np.float32)


def region_pairs(real, sim, masks, mdl_path: Path, cam: str):
    """Per-region linear-RGB means of sim (lens + vignette applied, no colour) and real: table, robot, every plate and
    every brick component, the unmasked background. Returns (A sim, B real, weights, saturated flags)."""
    A, B, Wt, sat = [], [], [], []
    for k in range(len(real)):
        s = lin(pl.apply(sim[k], cam, stateful=False, codec=False, model_path=mdl_path,
                         color_path=OUT / f"_none_{cam}.json"))
        r = lin(real[k])
        m = masks[k]
        regs = [(region(m, 1), 3.0, False), (region(m, 8), 1.5, False)]
        for bit in (2, 4):  # plates, bricks: the saturated colours the policy keys on
            n, lab = cv2.connectedComponents(region(m, bit, 3).astype(np.uint8))
            regs += [((lab == q), 3.0, True) for q in range(1, n)]
        regs.append((cv2.erode((m == 0).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0, 1.0, False))
        for rm, w, st in regs:
            if rm.sum() < (150 if st else 400):
                continue
            a, b = s[rm].mean(0), r[rm].mean(0)
            if a.max() > 0.92 or b.max() > 0.92:
                continue
            A.append(a)
            B.append(b)
            Wt.append(w)
            sat.append(st)
    return np.array(A), np.array(B), np.array(Wt), np.array(sat)


def root_feats(A):
    """Root-polynomial colour features r, g, b, sqrt(rg), sqrt(gb), sqrt(rb) (exposure-invariant)."""
    A = np.clip(A, 0, None)
    return np.c_[A, np.sqrt(A[:, 0] * A[:, 1]), np.sqrt(A[:, 1] * A[:, 2]), np.sqrt(A[:, 0] * A[:, 2])]


def solve_color(A, B, Wt, root: bool):
    """Ridge fit of B ~ M feats(A) toward the identity, WITHOUT an additive offset: the camera's black level stays at
    0 (an offset fitted on region means lifted the blacks by ~0.1 linear and washed out contrast and saturation)."""
    X = root_feats(A) if root else A
    n = X.shape[1]
    w = np.sqrt(Wt)[:, None]
    lam = 0.02 * len(A) * float((A ** 2).mean())
    prior = np.zeros((n, 3))
    prior[:3] = np.eye(3)
    sol = np.linalg.solve((X * w).T @ (X * w) + lam * np.eye(n), (X * w).T @ (B * w) + lam * prior)
    return sol.T, np.zeros(3)


def apply_color(A, M, off):
    """Region means through a fitted colour model (3x3 or 3x6)."""
    return (root_feats(A) if M.shape[1] == 6 else A) @ M.T + off


def fit_color(real, sim, masks, mdl_path: Path, cam: str, real_h, sim_h, masks_h) -> dict:
    """Per-camera colour response on neutral + saturated region means: linear 3x3 and root-polynomial 3x6 (each +
    offset), the one with the lower held-out error on the saturated regions (plates, bricks) is kept."""
    A, B, Wt, sat = region_pairs(real, sim, masks, mdl_path, cam)
    Ah, Bh, _, sath = region_pairs(real_h, sim_h, masks_h, mdl_path, cam)

    def err(M, off, X, Y, sel):
        return round(float(np.abs((X if M is None else apply_color(X, M, off))[sel] - Y[sel]).mean()), 4)

    out, best = {}, None
    for name, root in (("linear", False), ("root", True)):
        M, off = solve_color(A, B, Wt, root)
        e = {"fit_all": err(M, off, A, B, np.ones(len(A), bool)), "held_neutral": err(M, off, Ah, Bh, ~sath),
             "held_saturated": err(M, off, Ah, Bh, sath)}
        out[name] = {"M": M.round(5).tolist(), "offset": off.round(5).tolist(), "err": e}
        if best is None or e["held_saturated"] + e["held_neutral"] < best[1]:
            best = (name, e["held_saturated"] + e["held_neutral"])
    none = {"fit_all": err(None, None, A, B, np.ones(len(A), bool)), "held_neutral": err(None, None, Ah, Bh, ~sath),
            "held_saturated": err(None, None, Ah, Bh, sath)}
    b = out[best[0]]
    col = {"M_linear": pl.root_to_linear(b["M"]) if best[0] == "root" else b["M"], "offset_linear": b["offset"],
           **({"M_root": b["M"]} if best[0] == "root" else {}),
           "model": best[0], "n_pairs": int(len(A)), "n_saturated": int(sat.sum()),
           "mean_abs_err_linear": {"identity": none, **{k: v["err"] for k, v in out.items()}}}
    return col


def region_dE(real, sim, masks) -> float:
    """Median CIE76 dE of per-region Lab means (table, plates, robot) between real and sim frames."""
    out = []
    for k in range(len(real)):
        a, b = cv2.cvtColor(real[k].astype(np.float32) / 255, cv2.COLOR_RGB2LAB), \
            cv2.cvtColor(sim[k].astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
        for bit in (1, 2, 8):
            rm = region(masks[k], bit)
            if rm.sum() > 400:
                out.append(float(np.linalg.norm(a[rm].mean(0) - b[rm].mean(0))))
    return round(float(np.median(out)), 2) if out else None


def contrast(frames, masks) -> dict:
    """Median luma std (table, plates) and Lab chroma C* (plates, bricks) per region: contrast / saturation."""
    out = {"table_luma_std": [], "plate_luma_std": [], "plate_chroma": [], "brick_chroma": [], "luma_p5": []}
    for f, m in zip(frames, masks, strict=True):
        g = ms.gray(f)
        lab = cv2.cvtColor(f.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
        c = np.hypot(lab[..., 1], lab[..., 2])
        out["luma_p5"].append(float(np.percentile(g, 5)))
        for key, bit, arr in (("table_luma_std", 1, g), ("plate_luma_std", 2, g), ("plate_chroma", 2, c),
                              ("brick_chroma", 4, c)):
            rm = region(m, bit, 3)
            if rm.sum() > 150:
                out[key].append(float((arr[rm].std() if key.endswith("std") else np.median(arr[rm]))))
    return {k: round(float(np.median(v)), 2) if v else None for k, v in out.items()}


def tone_stats(frames, masks) -> np.ndarray:
    """Luma quantiles p5, p25, p50, p75, p95 (pooled) and median plate / brick Lab chroma."""
    g = np.concatenate([ms.gray(f)[::4, ::4].ravel() for f in frames])
    c = contrast(frames, masks)
    return np.r_[np.percentile(g, [5, 25, 50, 75, 95]), c["plate_chroma"] or 0.0, c["brick_chroma"] or 0.0]


def fit_tone(real, sim, masks, cam: str, mdl: dict, col_path: Path, n: int = 24) -> dict:
    """ISP tone (black, gamma, saturation) by grid search on n fit samples: luma quantiles + plate / brick chroma."""
    sel = np.linspace(0, len(real) - 1, min(n, len(real))).astype(int)
    real, sim, masks = real[sel], sim[sel], masks[sel]
    tgt = tone_stats(real, masks)
    tmp = OUT / f"_t_{cam}.json"
    base = {k: mdl[k] for k in ("lens", "vignette", "exposure_gain") if mdl.get(k) is not None}
    best = None
    for b, g, sat in itertools.product((0.0, 0.04, 0.08, 0.12, 0.16), (0.8, 0.9, 1.0, 1.1), (1.0, 1.2, 1.4, 1.6)):
        tmp.write_text(json.dumps({cam: dict(base, tone={"black": b, "gamma": g, "saturation": sat})}))
        fr = [roundtrip([pl.apply(f, cam, stateful=False, codec=False, model_path=tmp, color_path=col_path)])[0][0]
              for f in sim]  # through the codec: yuv420 + quantisation lower the chroma the real frames are measured at
        st = tone_stats(fr, masks)
        e = float(np.mean(((st - tgt) / np.r_[[10.0] * 5, 3.0, 3.0]) ** 2))
        if best is None or e < best[0]:
            best = (e, b, g, sat)
    tmp.unlink(missing_ok=True)
    return {"black": best[1], "gamma": best[2], "saturation": best[3], "err": round(best[0], 4)}


def metrics(real, sim_frames, masks, sizes_sim, noise_sim, tgt) -> dict:
    """Held-out comparison of one set of sim frames to the real ones."""
    sr, ss = spectrum(real), spectrum(sim_frames)
    return {"spec_diff_bands": np.round(ss - sr, 3).tolist(), "spec_rms_2_7": round(float(np.sqrt(np.mean((ss - sr)[2:]
                                                                                                         ** 2))), 3),
            "lap_real": round(float(np.median([ms.lap_var(ms.gray(f)) for f in real])), 1),
            "lap_sim": round(float(np.median([ms.lap_var(ms.gray(f)) for f in sim_frames])), 1),
            "iframe_bytes_real": tgt["size"], "iframe_bytes_sim": sizes_sim,
            "noise_dn_real": round(tgt["noise"], 3), "noise_dn_sim": None if noise_sim is None else round(noise_sim, 3),
            "region_dE": region_dE(real, sim_frames, masks),
            "luma_mean_real": round(float(np.mean([ms.gray(f).mean() for f in real])), 1),
            "luma_mean_sim": round(float(np.mean([ms.gray(f).mean() for f in sim_frames])), 1),
            "contrast_real": contrast(real, masks), "contrast_sim": contrast(sim_frames, masks)}


def sheet(real, raw, proc, path: Path, crop=(240, 180, 160, 120)):
    """real | sim raw | sim pipeline rows, each followed by 2x crops of the same window."""
    x, y, w, h = crop
    rows = []
    for a, b, c in zip(real, raw, proc, strict=True):
        top = np.concatenate([a, b, c], 1)
        z = [cv2.resize(im[y:y + h, x:x + w], (w * 4, h * 4), interpolation=cv2.INTER_NEAREST) for im in (a, b, c)]
        rows += [top, np.concatenate(z, 1)[:, : top.shape[1]]]
    img = np.concatenate([cv2.copyMakeBorder(r, 0, 4, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                          for r in rows], 0)
    cv2.imwrite(str(path), img[..., ::-1])


def main():
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sim", default=str(VF / "render_tex2.npz"))
    ap.add_argument("--cams", default=",".join(CAMS))
    ap.add_argument("--stages", default="vignette,color,exposure,tone,mtf")
    ap.add_argument("--n-search", type=int, default=24, help="fit samples used in the grid search")
    ap.add_argument("--sheets", type=int, default=4)
    ap.add_argument("--vignette", default=None,
                    help="v1,v2 instead of the vignette stage (top / inspect: 0,0 - the D415 ISP corrects lens "
                         "shading and their real / sim table ratios are dominated by the lighting, not monotonic)")
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    T0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads((VF / "evalset.json").read_text())["samples"]
    real_all = np.load(VF / "evalset.npz")
    sim_all = np.load(a.sim)
    fit_ids = [s["i"] for s in meta if s["split"] == "fit"]
    held_ids = [s["i"] for s in meta if s["split"] == "held"]
    stats = json.loads((OUT / "real_stats.json").read_text())
    eps = ms.episodes_table()
    model_all = json.loads(pl.MODEL_JSON.read_text()) if pl.MODEL_JSON.exists() else {}
    color_all = json.loads(pl.COLOR_JSON.read_text()) if pl.COLOR_JSON.exists() else {}
    report = {"sim": str(Path(a.sim).relative_to(REPO)), "cams": {}}
    rng = np.random.default_rng(0)
    for cam in a.cams.split(","):
        t0 = time.time()
        real, sim, masks = real_all[cam], sim_all[cam], sim_all[f"mask_{cam}"]
        mdl = dict(model_all.get(cam, {}))
        mdl.update({k: v for k, v in MEASURED[cam].items() if k != "ae"})
        mdl["ae"] = dict(mdl.get("ae", {}), **MEASURED[cam]["ae"])
        mdl.setdefault("codec", {"mode": "x264", "crf": 30, "emulate_q": 72})
        mdl["dr"] = DR_HALF[cam]
        mdl.setdefault("noise", {"mid": 1.5, "floor": 0.2, "chroma": 0.6, "corr": 0.5})
        mdl["lens"] = lens_model(cam)
        # the sim's region masks are in render pixels: move them into the real camera's pixels like the images
        mp = pl.lens_map(masks.shape[1], masks.shape[2], mdl["lens"])
        masks = np.stack([cv2.remap(m, mp[..., 0], mp[..., 1], cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
                          for m in masks])
        rep: dict = {}
        tgt = {"size": real_iframe_sizes(eps, [meta[i]["ep"] for i in fit_ids], cam), "noise": real_noise(stats, cam)}
        tgt_held = {"size": real_iframe_sizes(eps, [meta[i]["ep"] for i in held_ids], cam),
                    "noise": tgt["noise"]}
        rep["targets_fit"] = tgt
        tmp = OUT / f"_m_{cam}.json"
        none = OUT / f"_none_{cam}.json"
        none.write_text("{}")
        if a.vignette:
            mdl["vignette"] = [float(v) for v in a.vignette.split(",")]
            rep["vignette"] = {"v": mdl["vignette"], "from": "--vignette", "ratio_profile_r2": None, "n": 0,
                               "corner_gain": round(1 + sum(mdl["vignette"]), 3)}
        elif "vignette" in a.stages:
            tmp.write_text(json.dumps({cam: {"lens": mdl.get("lens")}}))
            v, prof, n = fit_vignette(real[fit_ids], sim[fit_ids], masks[fit_ids], tmp, cam)
            mdl["vignette"] = v
            rep["vignette"] = {"v": v, "ratio_profile_r2": prof, "n": n, "corner_gain": round(1 + v[0] + v[1], 3)}
            print(f"[fit] {cam} vignette {v} (corner {1 + v[0] + v[1]:.3f}) profile {prof}", flush=True)
        if "color" in a.stages:
            tmp.write_text(json.dumps({cam: {"lens": mdl.get("lens"), "vignette": mdl.get("vignette")}}))
            col = fit_color(real[fit_ids], sim[fit_ids], masks[fit_ids], tmp, cam, real[held_ids], sim[held_ids],
                            masks[held_ids])
            prev = {k: v for k, v in color_all.get(cam, {}).items()
                    if k not in ("blur_sigma", "sharpness_real_vs_sigma", "M_linear", "M_root", "offset_linear",
                                 "model", "n_pairs", "n_saturated", "mean_abs_err_linear")}
            prev.update(col)
            color_all[cam] = prev
            rep["color"] = col
            print(f"[fit] {cam} colour |err| {col['mean_abs_err_linear']}", flush=True)
        col_path = OUT / f"_c_{cam}.json"
        col_path.write_text(json.dumps(color_all))
        if "exposure" in a.stages and cam != "top":
            tmp.write_text(json.dumps({cam: {"lens": mdl.get("lens"), "vignette": mdl.get("vignette")}}))
            r = []
            for k in fit_ids:
                ys = float(ms.gray(pl.apply(sim[k], cam, stateful=False, codec=False, model_path=tmp,
                                            color_path=col_path)).mean())
                r.append((float(ms.gray(real[k]).mean()) / max(1.0, ys)) ** 2.2)  # sRGB mean -> linear gain
            mdl["exposure_gain"] = round(float(np.median(r)), 4)
            rep["exposure_gain"] = {"value": mdl["exposure_gain"],
                                    "p25_p75": np.percentile(r, [25, 75]).round(3).tolist()}
            print(f"[fit] {cam} exposure gain {mdl['exposure_gain']} (IQR {rep['exposure_gain']['p25_p75']})",
                  flush=True)
        # AE reference: the twin's nominal metered luma / chroma (metered before the exposure, like pipeline.apply)
        tmp.write_text(json.dumps({cam: {"lens": mdl.get("lens"), "vignette": mdl.get("vignette")}}))
        met, chs = [], []
        for k in fit_ids:
            x = lin(pl.apply(sim[k], cam, stateful=False, codec=False, model_path=tmp, color_path=col_path))
            met.append(float((x @ ms_luma()).mean()))
            c = x.reshape(-1, 3).mean(0)
            chs.append(c / np.exp(np.log(c).mean()))
        mdl["ae"].update({"m_ref": round(float(np.median(met)), 5),
                          "chroma_ref": np.round(np.median(chs, 0), 4).tolist()})
        if "tone" in a.stages:
            mdl["tone"] = fit_tone(real[fit_ids], sim[fit_ids], masks[fit_ids], cam, mdl, col_path)
            rep["tone"] = mdl["tone"]
            print(f"[fit] {cam} tone {mdl['tone']}", flush=True)
        if "mtf" in a.stages:
            sr = spectrum(real[fit_ids])
            srch = fit_ids[:: max(1, len(fit_ids) // a.n_search)][: a.n_search]
            nz_ids = srch[:8]
            best, trace = None, []
            g = GRID[cam]
            for bs, sa, sr_, nm in itertools.product(g["blur_sigma"], g["sharpen_amount"], g["sharpen_radius"],
                                                     g["noise_mid"]):
                m2 = dict(mdl, blur_sigma=bs, sharpen_amount=sa, sharpen_radius=sr_,
                          noise=dict(mdl["noise"], mid=nm), exposure_frames=0.0)
                fr, size = through(sim[srch], cam, m2, rng, color_path=col_path)
                fa, _ = through(sim[nz_ids], cam, m2, rng, color_path=col_path)
                fb, _ = through(sim[nz_ids], cam, m2, rng, color_path=col_path)
                ns = sim_noise(fa, fb)
                ss = spectrum(fr)
                e_spec = float(np.mean((ss - sr)[2:] ** 2))
                e_noise = float(np.log((ns + 0.05) / (tgt["noise"] + 0.05)) ** 2)
                e_size = float(np.log(size / tgt["size"]) ** 2)
                e = e_spec + 0.5 * e_noise + 0.5 * e_size
                trace.append([bs, sa, sr_, nm, round(e_spec, 4), round(ns, 3), round(size), round(e, 4)])
                if best is None or e < best[0]:
                    best = (e, bs, sa, sr_, nm)
            _, bs, sa, sr_, nm = best
            mdl.update(blur_sigma=bs, sharpen_amount=sa, sharpen_radius=sr_)
            mdl["noise"] = dict(mdl["noise"], mid=nm)
            rep["mtf_search"] = {"best": {"blur_sigma": bs, "sharpen_amount": sa, "sharpen_radius": sr_,
                                          "noise_mid": nm},
                                 "trace[bs,sa,sr,nm,e_spec,noise,size,e]": trace}
            print(f"[fit] {cam} mtf/noise best blur {bs} sharpen {sa} r {sr_} noise {nm} (e {best[0]:.4f})", flush=True)
        # held-out evaluation: raw, legacy colour correction, full pipeline
        H = held_ids
        mfinal = OUT / f"_final_{cam}.json"
        mfinal.write_text(json.dumps({cam: dict(mdl, exposure_frames=0.0)}))
        raw = list(sim[H])
        legacy = [legacy_color_correct(f, cam) for f in raw]
        proc, size_p = through(sim[H], cam, dict(mdl, exposure_frames=0.0), rng, color_path=col_path)
        _, size_raw = through(sim[H], cam, {"codec": {"mode": False}}, rng, color_path=none)
        pa, _ = through(sim[H[:8]], cam, dict(mdl, exposure_frames=0.0), rng, color_path=col_path)
        pb, _ = through(sim[H[:8]], cam, dict(mdl, exposure_frames=0.0), rng, color_path=col_path)
        rep["held"] = {"raw": metrics(real[H], raw, masks[H], size_raw, 0.0, tgt_held),
                       "legacy_color_correct": metrics(real[H], legacy, masks[H], None, None, tgt_held),
                       "pipeline": metrics(real[H], proc, masks[H], size_p, sim_noise(pa, pb), tgt_held)}
        for k, v in rep["held"].items():
            print(f"[fit] {cam} held {k}: {v}", flush=True)
        if a.sheets:
            sel = H[:: max(1, len(H) // a.sheets)][: a.sheets]
            pr, _ = through(sim[sel], cam, dict(mdl, exposure_frames=0.0), rng, color_path=col_path)
            sheet(list(real[sel]), list(sim[sel]), pr, OUT / f"sbs_{cam}.png")
        mdl.pop("M_linear", None)
        mdl.pop("offset_linear", None)
        model_all[cam] = mdl
        report["cams"][cam] = rep
        rep["time_s"] = round(time.time() - t0, 1)
        for p in (tmp, col_path, mfinal, none):
            p.unlink(missing_ok=True)
    model_all["_note"] = ("camera pipeline parameters per camera (sensors/pipeline.py), fitted by sensors/fit.py on "
                          "the visual-fit evaluation set; colour matrices live in visuals/camera_color.json")
    if not a.no_write:  # merge into the files as they are now (one fit process per camera may run in parallel)
        done = list(report["cams"])
        for path, new in ((pl.MODEL_JSON, model_all), (pl.COLOR_JSON, color_all)):
            cur = json.loads(path.read_text()) if path.exists() else {}
            cur.update({c: new[c] for c in done if c in new})
            if "_note" in new:
                cur["_note"] = new["_note"]
            pl.write_json_atomic(path, cur)
    (OUT / f"fit_report_{'_'.join(report['cams'])}.json").write_text(json.dumps(report, indent=1))
    print("FIT_OK", round(time.time() - T0), "s", flush=True)


def legacy_color_correct(img, cam: str):
    """The pre-sensor twin_scene.color_correct: linear 3x3 (+ offset) and gaussian blur from camera_color.json."""
    fit = json.loads(pl.COLOR_JSON.read_text()).get(cam) if pl.COLOR_JSON.exists() else None
    if not fit or "M_linear" not in fit:
        return img
    c = lin(img)
    out = np.clip(c @ np.asarray(fit["M_linear"]).T + np.asarray(fit.get("offset_linear", (0, 0, 0))), 0, 1)
    s = float(fit.get("blur_sigma", 0.0))
    if s > 0.05:
        out = cv2.GaussianBlur(out.astype(np.float32), (0, 0), s, borderType=cv2.BORDER_REPLICATE)
    srgb = np.where(out <= 0.0031308, out * 12.92, 1.055 * out ** (1 / 2.4) - 0.055)
    return (srgb * 255 + 0.5).astype(np.uint8)


if __name__ == "__main__":
    main()
