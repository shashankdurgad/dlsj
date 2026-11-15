"""Characterise the three real cameras from the recorded cal2 dataset (plain python + PyAV; never import in Isaac).

    python sim/twin/sensors/measure.py [--episodes 12] [--out sim/twin/out/sensors/real_stats.json]

Everything is measured on the DECODED dataset frames (what the policy saw: sensor + ISP + H.264 crf 30, g 2, yuv420p),
on held-out episodes (episode index % 3 == 2, the visual fit's held-out third). Per camera:
* gop       frame types (I / P) and packet sizes;
* noise     temporal noise sigma (DN, 0-255) vs brightness: std of (f[t+2] - f[t]) / sqrt 2 in static 16 px blocks,
            separately for I-I and P-P frame pairs, per luma bin and per channel (Y, Cb, Cr);
* sharp     variance of the Laplacian and the azimuthally averaged power spectrum of the luma (8 radial bands);
* vignette  radial luma fall-off of the median frame (wrist: the median over many moving views, i.e. the lens);
* ae        auto exposure: d log(static-region luma) / d log(frame mean luma) over each episode (linear light);
            0 = fixed exposure, < 0 = the camera brightens the static scene when the dark arm enters the view;
* motion    wrist sharpness vs the optical-flow speed (px / frame) and the exposure time that explains it;
* block     JPEG-blockiness: gradient energy at x % 8 == 0 relative to x % 4 == 0 (MJPG capture leaves 8 px seams).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
DATA = REPO / "sim/twin/data/canonical/duplo_teleop_cal2_rgb_456"
OUT = REPO / "sim/twin/out/sensors"
CAMS = ("top", "wrist", "inspect")
BANDS = (0.02, 0.04, 0.08, 0.12, 0.18, 0.25, 0.32, 0.40, 0.5)  # cycles / px (radial band edges)
LUMA_BINS = (0, 30, 60, 90, 120, 150, 180, 210, 256)


def episodes_table(root: Path = DATA):
    """The dataset's meta/episodes rows (pandas DataFrame indexed by episode_index)."""
    import pandas as pd
    import pyarrow.parquet as pq

    fs = sorted(glob.glob(str(root / "meta/episodes/*/*.parquet")))
    return pd.concat([pq.read_table(f).to_pandas() for f in fs]).set_index("episode_index", drop=False)


def episode_frames(eps, ep: int, cam: str, root: Path = DATA, max_frames: int | None = None):
    """Decode one episode of one camera in order: (frames uint8 (N, H, W, 3) RGB, key-frame flags (N,))."""
    import av

    r = eps.loc[ep]
    k = f"videos/observation.images.{cam}"
    path = root / k / f"chunk-{int(r[k + '/chunk_index']):03d}/file-{int(r[k + '/file_index']):03d}.mp4"
    t0, t1 = float(r[k + "/from_timestamp"]), float(r[k + "/to_timestamp"])
    frames, keys = [], []
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        c.seek(int(max(0.0, t0 - 0.2) / s.time_base), stream=s, backward=True)
        for fr in c.decode(s):
            t = float(fr.pts * s.time_base)
            if t < t0 - 0.5 / 30:
                continue
            if t >= t1 - 0.5 / 30 or (max_frames and len(frames) >= max_frames):
                break
            frames.append(fr.to_ndarray(format="rgb24"))
            keys.append(bool(fr.key_frame))
    return np.stack(frames), np.asarray(keys)


def ycc(img: np.ndarray) -> np.ndarray:
    """RGB uint8 -> float32 Y, Cb, Cr (BT.601 full range, the decoder's inverse), (..., 3)."""
    x = img.astype(np.float32)
    y = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
    return np.stack([y, 0.564 * (x[..., 2] - y), 0.713 * (x[..., 0] - y)], -1)


def srgb_lin(v):
    """sRGB 0-255 -> linear 0-1."""
    c = np.asarray(v, np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def temporal_noise(frames: np.ndarray, pairs, block: int = 16) -> dict:
    """Temporal noise sigma per luma bin from frame pairs (i, j) of a static camera: per 16 px block the std of
    (f_j - f_i) / sqrt 2; blocks whose mean changes (motion, exposure) are dropped. Returns {bin: [Y, Cb, Cr]} and the
    pooled list of (luma, sigma_Y) block samples."""
    H, W = frames.shape[1:3]
    hb, wb = H // block, W // block
    samples = []
    for i, j in pairs:
        a, b = ycc(frames[i]), ycc(frames[j])
        d = (b - a)[: hb * block, : wb * block].reshape(hb, block, wb, block, 3)
        m = a[: hb * block, : wb * block, 0].reshape(hb, block, wb, block).mean((1, 3))
        g = np.abs(cv2.Laplacian(a[..., 0], cv2.CV_32F))[: hb * block, : wb * block]
        g = g.reshape(hb, block, wb, block).mean((1, 3))
        dm = d.mean((1, 3))  # (hb, wb, 3)
        ds = d.std((1, 3)) / math.sqrt(2)
        ok = (np.abs(dm[..., 0]) < 0.6) & (g < 6.0)
        samples.append(np.c_[m[ok], ds[ok]])
    s = np.concatenate(samples) if samples else np.zeros((0, 4))
    out = {}
    for lo, hi in zip(LUMA_BINS[:-1], LUMA_BINS[1:], strict=True):
        sel = (s[:, 0] >= lo) & (s[:, 0] < hi)
        if sel.sum() >= 30:
            out[f"{lo}-{hi}"] = {"n": int(sel.sum()), "sigma_ycc": np.median(s[sel, 1:], 0).round(3).tolist()}
    return {"bins": out, "all_median_y": round(float(np.median(s[:, 1])), 3) if len(s) else None}


def radial_spectrum(gray: np.ndarray) -> np.ndarray:
    """Azimuthally averaged power (log10) of a Hann-windowed luma image in the BANDS (cycles / px)."""
    H, W = gray.shape
    win = np.outer(np.hanning(H), np.hanning(W)).astype(np.float32)
    g = (gray - gray.mean()) * win
    P = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(H))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(W))[None, :]
    r = np.sqrt(fx ** 2 + fy ** 2)
    return np.array([np.log10(P[(r >= lo) & (r < hi)].mean() + 1e-9) for lo, hi in zip(BANDS[:-1], BANDS[1:],
                                                                                         strict=True)])


def lap_var(gray: np.ndarray) -> float:
    """Variance of the Laplacian (sharpness)."""
    return float(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F).var())


def gray(img: np.ndarray) -> np.ndarray:
    """RGB uint8 -> float32 luma."""
    return ycc(img)[..., 0]


def vignette(med: np.ndarray, cx: float | None = None, cy: float | None = None) -> dict:
    """Fit lin(luma) = a (1 + k1 r^2 + k2 r^4) to the median frame (r normalised to the half-diagonal)."""
    H, W = med.shape[:2]
    cx = (W - 1) / 2 if cx is None else cx
    cy = (H - 1) / 2 if cy is None else cy
    yy, xx = np.mgrid[0:H, 0:W]
    r2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / ((W / 2) ** 2 + (H / 2) ** 2)
    y = srgb_lin(gray(med)).ravel()
    A = np.c_[np.ones_like(y), r2.ravel(), r2.ravel() ** 2]
    c = np.linalg.lstsq(A, y, rcond=None)[0]
    rr = r2.ravel()
    prof = [round(float(np.median(y[(rr >= lo) & (rr < lo + 0.1)])), 4) for lo in np.arange(0, 1.0, 0.1)]
    return {"k1": round(float(c[1] / c[0]), 4), "k2": round(float(c[2] / c[0]), 4),
            "corner_rel": round(float(1 + c[1] / c[0] + c[2] / c[0]), 4), "lin_profile_r2_bins": prof}


def auto_exposure(frames: np.ndarray, static_mask: np.ndarray) -> dict:
    """Slope of log(static-region linear luma) vs log(frame mean linear luma) over one episode (robust fit)."""
    lin = srgb_lin(np.arange(256))
    g = np.stack([gray(f) for f in frames]).clip(0, 255).astype(np.uint8)
    L = lin[g]
    st = np.log(L[:, static_mask].mean(1) + 1e-6)
    fm = np.log(L.reshape(len(L), -1).mean(1) + 1e-6)
    if fm.std() < 1e-3:
        return {"slope": 0.0, "frame_range": 0.0, "static_range": float(np.ptp(st))}
    slope = float(np.polyfit(fm - fm.mean(), st - st.mean(), 1)[0])
    return {"slope": round(slope, 3), "frame_log_range": round(float(np.ptp(fm)), 3),
            "static_log_range": round(float(np.ptp(st)), 3), "static_log_std": round(float(st.std()), 4)}


def blockiness(frames: np.ndarray) -> float:
    """Mean |horizontal + vertical luma step| on 8 px grid lines / on the other 4 px grid lines (1 = no 8 px seams)."""
    r8, r4 = [], []
    for f in frames:
        g = gray(f)
        dx = np.abs(np.diff(g, axis=1)).mean(0)  # step between column x and x+1
        dy = np.abs(np.diff(g, axis=0)).mean(1)
        for d in (dx, dy):
            idx = np.arange(len(d))
            r8.append(d[(idx % 8) == 7].mean())
            r4.append(d[(idx % 8) == 3].mean())
    return round(float(np.mean(r8) / np.mean(r4)), 4)


def flow_speed(a: np.ndarray, b: np.ndarray) -> float:
    """Median optical-flow magnitude (px / frame at 640 px width) between two frames (Farneback at 1/2 res)."""
    ga = cv2.resize(gray(a), None, fx=0.5, fy=0.5).astype(np.uint8)
    gb = cv2.resize(gray(b), None, fx=0.5, fy=0.5).astype(np.uint8)
    f = cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return float(np.median(np.linalg.norm(f, axis=-1)) * 2.0)


def motion_blur_fit(frames: np.ndarray, keys: np.ndarray) -> dict:
    """Wrist exposure time from motion blur: sharpness of I-frames vs flow speed. Synthetic linear motion blur of
    length s * t_exp * fps applied to the slow (sharp) frames must reproduce the fast frames' sharpness; returns the
    t_exp (ms) that fits best plus the binned curve."""
    idx = [i for i in range(1, len(frames) - 1) if keys[i]]
    sp = np.array([0.5 * (flow_speed(frames[i - 1], frames[i]) + flow_speed(frames[i], frames[i + 1])) for i in idx])
    sh = np.array([lap_var(gray(frames[i])) for i in idx])
    return {"speed": sp.tolist(), "sharp": sh.tolist()}


def measure(n_eps: int = 12, out: Path = OUT / "real_stats.json") -> dict:
    """Run every measurement on n_eps held-out episodes and write the JSON (+ median frames as PNG)."""
    eps = episodes_table()
    held = [int(e) for e in eps.index if int(e) % 3 == 2]
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(held, size=min(n_eps, len(held)), replace=False).tolist())
    res: dict = {"episodes": pick, "dataset": str(DATA.relative_to(REPO))}
    out.parent.mkdir(parents=True, exist_ok=True)
    for cam in CAMS:
        print(f"[measure] {cam}", flush=True)
        r: dict = {"noise_ii": [], "noise_pp": [], "noise_ip": [], "lap_i": [], "lap_p": [], "spec_i": [],
                   "ae": [], "block": [], "motion": {"speed": [], "sharp": []}, "luma_mean": []}
        med_frames = []
        for ep in pick:
            fr, keys = episode_frames(eps, ep, cam)
            r.setdefault("key_pattern", "".join("I" if k else "P" for k in keys[:12]))
            ii = [(i, i + 2) for i in range(0, len(fr) - 2, 2) if keys[i] and keys[i + 2]]
            pp = [(i, i + 2) for i in range(1, len(fr) - 2, 2) if not keys[i] and not keys[i + 2]]
            ip = [(i, i + 1) for i in range(0, len(fr) - 1, 2) if keys[i] and not keys[i + 1]]
            if cam != "wrist":
                r["noise_ii"].append(temporal_noise(fr, ii[::3]))
                r["noise_pp"].append(temporal_noise(fr, pp[::3]))
                r["noise_ip"].append(temporal_noise(fr, ip[::3]))
                st = np.stack([gray(f) for f in fr[::3]]).std(0)
                r["ae"].append(auto_exposure(fr, st < np.percentile(st, 30)))
            else:
                m = motion_blur_fit(fr[::1], keys)
                r["motion"]["speed"] += m["speed"]
                r["motion"]["sharp"] += m["sharp"]
                # temporal noise on the wrist only at rest (start of the episode, arm still)
                r["noise_ii"].append(temporal_noise(fr[:20], [p for p in ii if p[1] < 20]))
            sel = np.arange(0, len(fr), 10)
            r["lap_i"] += [lap_var(gray(fr[i])) for i in sel if keys[i]]
            r["lap_p"] += [lap_var(gray(fr[i + 1])) for i in sel if keys[i] and i + 1 < len(fr)]
            r["spec_i"] += [radial_spectrum(gray(fr[i])).tolist() for i in sel if keys[i]]
            r["block"].append(blockiness(fr[sel[:6]]))
            r["luma_mean"] += [float(gray(fr[i]).mean()) for i in sel]
            med_frames += list(fr[sel])
        med = np.median(np.stack(med_frames), 0).astype(np.uint8)
        cv2.imwrite(str(out.parent / f"real_median_{cam}.png"), med[..., ::-1])
        r["vignette_median"] = vignette(med)
        res[cam] = r
    out.write_text(json.dumps(res))
    return res


def main():
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--out", default=str(OUT / "real_stats.json"))
    a = ap.parse_args()
    measure(a.episodes, Path(a.out))


if __name__ == "__main__":
    main()
