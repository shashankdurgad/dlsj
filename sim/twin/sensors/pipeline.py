"""Camera sensor + image pipeline: a clean twin render -> what the recorded dataset contains for that camera.

    from sim.twin.sensors.pipeline import apply
    out = apply(img, "wrist")                    # uint8 sRGB (H, W, 3) / (N, H, W, 3), numpy or torch -> same type
    out = apply(img, "top", rng=rng, scale=0.5)  # + DR around the fitted camera (exposure, WB, noise, blur, codec)

Stages, in the physical order (parameters per camera in camera_model.json, fitted by fit.py on held-out cal2
episodes; the colour matrix in visuals/camera_color.json):
1. lens: one remap from the sim's centred pinhole (focal f) to the real camera's intrinsics: the principal point
   (top 304.2 / 239.0, inspect 309.6 / 252.0 factory; wrist centred, because its sim rotation already reproduces
   the measured 339.3 / 236.0) and the measured wrist lens distortion (cameras_measured.json, k1 0.054,
   k2 -0.066); for the distortion-free RealSenses this is the exact image translation c_real - c_sim;
2. colour: per-camera colour response from camera_color.json: root-polynomial `M_root` (3 x 6) or linear 3x3
   `M_linear`, + `offset_linear`, fitted on neutral AND saturated regions (table, robot, plates, bricks), so one
   albedo maps to each camera's recorded colour (lavender plate: sky-blue in top, lavender in wrist / inspect);
3. exposure: the camera's own exposure level `exposure_gain` (each real camera's AE settles on its own exposure
   under the shared lights; fitted for wrist / inspect, 1 for top whose look the lights are calibrated to), then
   auto exposure / auto white balance (stateful per stream): exposure follows (m_ref / m)^ae_strength of the metered
   linear luma m with a first-order lag of ae_tau frames, white balance follows the frame's grey-world chromaticity
   with awb_strength / awb_tau; at a scene cut (reset) the state jumps to the steady state;
4. vignetting: gain 1 + v1 r^2 + v2 r^4 (r = 1 at the half-diagonal);
5. optics blur: gaussian blur_sigma px (linear light) and motion blur along the image motion x exposure time
   (wrist; the motion is the global shift from the previous raw frame of the stream, or `motion=(dx, dy)` px/frame);
6. sRGB encode, ISP tone: black level `black` (display fraction clamped to 0), contrast exponent `gamma` and
   saturation `saturation` (chroma gain about the luma) from `tone` - the real D415 frames have deep blacks and
   saturated plates that a colour matrix alone (no offset) leaves soft; then sensor noise in DN:
   luma sigma(L) = noise_mid * sqrt(floor + (1 - floor) L / 128), chroma = chroma_ratio x luma, spatially
   correlated by a gaussian of noise_corr px (demosaic / ISP denoise);
   optional mains flicker (per-frame gain pulse + rolling-shutter row banding);
7. ISP sharpening: unsharp mask (sharpen_amount, sharpen_radius px) - the RealSense ISP sharpens, so its real
   frames are much sharper than the twin's renders;
8. quantise to 8 bit and the dataset's H.264 round-trip (codec.py: libx264 crf 30, g 2, yuv420p, I/P phase per
   stream; scene cuts force an I frame like x264's scenecut).

`scale` widens the DR draws (camera_model.json "dr" half-ranges at scale 1, all small): exposure, white balance,
noise, blur, sharpening, crf. Per-episode DR parameters come from `draw_dr(cam, rng, scale)` (TwinRandomizer does
this on every DR reset and registers them with `set_dr`); `apply(..., scale>0)` without `dr` draws fresh per call.

Inside Isaac this module is imported by twin_scene.color_correct (the TwinEnv image hook); it never imports PyAV
(the codec runs in a worker subprocess).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MODEL_JSON = HERE / "camera_model.json"
COLOR_JSON = REPO / "sim/twin/visuals/camera_color.json"
MEASURED_JSON = REPO / "sim/twin/calib/cameras_measured.json"
LUMA = (0.2126, 0.7152, 0.0722)

STAGES = frozenset(("lens", "color", "ae", "vignette", "blur", "tone", "noise", "sharpen", "codec"))
_cache: dict = {}
_streams: dict = {}  # stream -> Stream (codec phase, AE / AWB state, previous frame)
_dr: dict = {"params": {}, "alive": {}, "processed": {}}
_codec = None


def _json(path: Path) -> dict:
    """Read a JSON file, cached by mtime (fit.py rewrites camera_model.json while sims run); a file that cannot be
    read or parsed right now keeps the last good version."""
    hit = _cache.get(path)
    try:
        m = path.stat().st_mtime
        if hit and hit[0] == m:
            return hit[1]
        d = json.loads(path.read_text())
    except (OSError, ValueError):
        return hit[1] if hit else {}
    _cache[path] = (m, d)
    return d


def write_json_atomic(path: Path, obj) -> None:
    """Write JSON to a temp file next to `path` and rename it over `path` (readers never see a partial file)."""
    import os

    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def root_project(M, n: int) -> list:
    """Least-squares stand-in with n root-polynomial terms (3: linear) for a response M (3 x 6 / 13) over the RGB
    cube; written next to the full response so older readers (3x3 `M_linear`, 3x6 `M_root`) keep working."""
    g = np.stack(np.meshgrid(*[np.linspace(0.02, 0.9, 9)] * 3, indexing="ij"), -1).reshape(-1, 3)
    y = root_features_np(g, np.asarray(M).shape[1]) @ np.asarray(M, float).T
    X = g if n == 3 else root_features_np(g, n)
    return np.linalg.lstsq(X, y, rcond=None)[0].T.round(5).tolist()


def root_to_linear(M_root) -> list:
    """Least-squares 3x3 stand-in for a root-polynomial response over the RGB cube (written next to M_root so
    readers that only know `M_linear` keep working)."""
    g = np.stack(np.meshgrid(*[np.linspace(0.02, 0.9, 9)] * 3, indexing="ij"), -1).reshape(-1, 3)
    f = root_features_np(g, np.asarray(M_root).shape[1])
    return np.linalg.lstsq(g, f @ np.asarray(M_root, float).T, rcond=None)[0].T.round(5).tolist()


def model(cam: str, model_path: Path | None = None, color_path: Path | None = None) -> dict:
    """The camera model of `cam`: camera_model.json entry + the colour matrix of camera_color.json."""
    m = dict(_json(model_path or MODEL_JSON).get(cam, {}))
    col = _json(color_path or COLOR_JSON).get(cam) or {}
    m["M_linear"] = col.get("M_linear")
    m["offset_linear"] = col.get("offset_linear")
    m["M_root"] = col.get("M_root3") or col.get("M_root")  # M_root3 (3 x 13) when present; M_root stays 3 x 6
    if "blur_sigma" not in m and "blur_sigma" in col:  # legacy camera_color.json softness when no sensor fit
        m["blur_sigma"] = col["blur_sigma"]
    return m


class Stream:
    """Per-stream state: AE exposure, AWB gains, the previous raw frame (motion), codec GOP phase is in codec.H264."""

    def __init__(self):
        self.exposure: float | None = None
        self.wb: np.ndarray | None = None
        self.prev_small: np.ndarray | None = None
        self.t = 0


def reset(stream: str | None = None):
    """Forget the state of one stream or all (next frame is a scene cut: AE / AWB steady state, I frame)."""
    if stream is None:
        _streams.clear()
    else:
        _streams.pop(stream, None)
    if _codec is not None:
        _codec.reset(stream)


def set_dr(params: dict | None):
    """Register per-camera DR parameters (draw_dr output, {cam: {...}}) for the TwinEnv hook; None clears them."""
    _dr["params"] = dict(params or {})
    _dr["alive"] = {c: True for c in _dr["params"]}


def dr_heartbeat(cam: str, params: dict | None = None):
    """Called by randomize.postprocess on every DR frame: keeps the DR of `cam` (params, else the registered one)
    active for the next frame (a frame without a heartbeat in between, i.e. an evaluation reset after DR, drops back
    to the nominal camera)."""
    if params is not None:
        _dr["params"][cam] = params
    _dr["alive"][cam] = True


def consume_processed(cam: str) -> bool:
    """True once if the last frame of `cam` already went through camera_response (the TwinEnv hook)."""
    return bool(_dr["processed"].pop(cam, False))


def registered_dr(cam: str) -> dict | None:
    """The DR parameters to use for this frame of `cam` (consumes the heartbeat)."""
    p = _dr["params"].get(cam)
    if p is None or not _dr["alive"].get(cam, False):
        return None
    _dr["alive"][cam] = False
    return p


def draw_dr(cam: str, rng: np.random.Generator, scale: float = 1.0, model_path: Path | None = None,
            half_range: dict | None = None) -> dict:
    """Per-episode DR draw around the fitted camera: half-ranges (camera_model.json "dr", or `half_range`) x scale.
    Keys: exposure (log), wb (log, per channel, mean-free), noise (log gain), blur (+px, one-sided), sharpen (log),
    crf (offset range)."""
    h = half_range or model(cam, model_path).get("dr", {})
    s = float(scale)

    def u(k, default):
        return float(rng.uniform(-1, 1)) * float(h.get(k, default)) * s

    wb = np.exp([u("wb", 0.02) for _ in range(3)])
    lo, hi = h.get("crf", (-2, 3))
    return {"exposure": float(math.exp(u("exposure", 0.05))), "wb": (wb / np.exp(np.log(wb).mean())).round(4).tolist(),
            "noise": float(math.exp(u("noise", 0.3))), "blur_add": max(0.0, u("blur", 0.25)),
            "sharpen": float(math.exp(u("sharpen", 0.25))),
            "crf": int(round(rng.uniform(lo * s, hi * s))) if s > 0 else 0,
            "seed": int(rng.integers(2 ** 31))}


# ------------------------------------------------------------------------------------------------ torch helpers
def _torch():
    import torch
    import torch.nn.functional as F

    return torch, F


def _srgb_to_lin(x):
    torch, _ = _torch()
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055).clamp(min=0) ** 2.4)


def _lin_to_srgb(x):
    torch, _ = _torch()
    x = x.clamp(0, 1)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def root_features(r, g, b, n: int, sqrt, cbrt, cat):
    """Root-polynomial colour features (exposure-invariant): n = 6 (degree 2: r, g, b, sqrt(rg), sqrt(gb), sqrt(rb))
    or 13 (degree 3: + cbrt(r^2 g), cbrt(r^2 b), cbrt(g^2 r), cbrt(g^2 b), cbrt(b^2 r), cbrt(b^2 g), cbrt(rgb)).
    Array-library agnostic (numpy columns or torch N 1 H W planes)."""
    f = [r, g, b, sqrt(r * g), sqrt(g * b), sqrt(r * b)]
    if n == 13:
        f += [cbrt(r * r * g), cbrt(r * r * b), cbrt(g * g * r), cbrt(g * g * b), cbrt(b * b * r), cbrt(b * b * g),
              cbrt(r * g * b)]
    return cat(f)


def root_features_np(A, n: int = 6):
    """root_features of (N, 3) linear RGB rows -> (N, n)."""
    A = np.clip(np.asarray(A, float), 0, None)
    return root_features(A[:, 0], A[:, 1], A[:, 2], n, np.sqrt, np.cbrt, lambda f: np.stack(f, -1))


def _color(x, m: dict, dev):
    """Camera colour response on N 3 H W linear RGB: root-polynomial `M_root` (3 x 6 or 3 x 13, root_features;
    exposure-invariant, bends saturated colours a 3x3 cannot, e.g. the lavender plate that the top D415 records
    sky-blue) or linear `M_linear` (3 x 3), each + `offset_linear`."""
    torch, _ = _torch()
    if m.get("M_root") is not None:
        M = torch.tensor(m["M_root"], device=dev, dtype=torch.float32)
        r, g, b = x[:, 0:1].clamp(min=0), x[:, 1:2].clamp(min=0), x[:, 2:3].clamp(min=0)
        feats = root_features(r, g, b, M.shape[1], torch.sqrt, lambda v: v.clamp(min=0) ** (1.0 / 3.0),
                              lambda f: torch.cat(f, 1))
        x = torch.einsum("ij,njhw->nihw", M, feats)
    elif m.get("M_linear") is not None:
        x = torch.einsum("ij,njhw->nihw", torch.tensor(m["M_linear"], device=dev, dtype=torch.float32), x)
    else:
        return x
    if m.get("offset_linear") is not None:
        x = x + torch.tensor(m["offset_linear"], device=dev, dtype=torch.float32).view(1, 3, 1, 1)
    return x


def _gauss(x, sigma: float):
    """Separable gaussian blur of N C H W (reflect padding)."""
    torch, F = _torch()
    if sigma < 0.05:
        return x
    r = max(1, int(math.ceil(3 * sigma)))
    t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    C = x.shape[1]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).repeat(C, 1, 1, 1), groups=C)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).repeat(C, 1, 1, 1), groups=C)


def _motion(x, dx: float, dy: float):
    """Linear motion blur of N C H W along (dx, dy) px (box along the segment)."""
    torch, F = _torch()
    L = math.hypot(dx, dy)
    if L < 0.75:
        return x
    r = int(math.ceil(L / 2)) + 1
    n = 2 * r + 1
    k = np.zeros((n, n), np.float32)
    for t in np.linspace(-0.5, 0.5, max(8, int(8 * L))):
        k[int(round(r + t * dy)), int(round(r + t * dx))] += 1.0
    k /= k.sum()
    C = x.shape[1]
    kk = torch.from_numpy(k).to(x.device, x.dtype).view(1, 1, n, n).repeat(C, 1, 1, 1)
    return F.conv2d(F.pad(x, (r, r, r, r), mode="reflect"), kk, groups=C)


def _lens_grid(cam: str, H: int, W: int, device, lens: dict):
    """grid_sample grid: for every output pixel of the REAL camera (intrinsics K_real = fx, fy, cx, cy, distortion
    dist) the pixel of the sim's centred pinhole render (focal f) that sees the same ray. Without distortion and with
    equal focals this is the exact principal-point translation c_real - c_sim."""
    key = ("grid", cam, H, W, str(device), json.dumps(lens, sort_keys=True))
    if key in _cache:
        return _cache[key]
    import torch

    und = lens_map(H, W, lens)
    g = np.stack([und[..., 0] / (W - 1) * 2 - 1, und[..., 1] / (H - 1) * 2 - 1], -1)
    grid = torch.from_numpy(g.astype(np.float32))[None].to(device)
    _cache[key] = grid
    return grid


def lens_map(H: int, W: int, lens: dict) -> np.ndarray:
    """(H, W, 2) float32 sim-render pixel (x, y) seen by every real-camera pixel (cv2.remap map of the lens stage)."""
    import cv2

    s = W / 640.0
    f = float(lens["f"]) * s
    K_sim = np.array([[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1.0]])
    fx, fy, cx, cy = (float(v) * s for v in lens.get("K_real", (lens["f"], lens["f"], 319.5, 239.5)))
    K_real = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    pts = np.stack([xx.ravel(), yy.ravel()], -1).reshape(-1, 1, 2)
    dist = np.asarray(lens.get("dist", (0, 0, 0, 0, 0)), np.float64)
    return cv2.undistortPoints(pts, K_real, dist, P=K_sim).reshape(H, W, 2).astype(np.float32)


def _radius2(H: int, W: int, device):
    key = ("r2", H, W, str(device))
    if key not in _cache:
        torch, _ = _torch()
        yy, xx = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32),
                                torch.arange(W, device=device, dtype=torch.float32), indexing="ij")
        _cache[key] = (((xx - (W - 1) / 2) ** 2 + (yy - (H - 1) / 2) ** 2) / ((W / 2) ** 2 + (H / 2) ** 2))[None, None]
    return _cache[key]


def _global_shift(prev_small: np.ndarray, cur_small: np.ndarray, factor: float) -> tuple[float, float]:
    import cv2

    (dx, dy), resp = cv2.phaseCorrelate(prev_small, cur_small)
    return (dx * factor, dy * factor) if resp > 0.05 else (0.0, 0.0)


def _get_codec():
    global _codec
    if _codec is None:
        from sim.twin.sensors.codec import H264

        _codec = H264()
    return _codec


# ------------------------------------------------------------------------------------------------ entry point
def apply(img, cam: str, rng: np.random.Generator | None = None, scale: float = 0.0, *, dr: dict | None = None,
          linear: bool = False, stream: str | None = None, motion: tuple | None = None, codec: bool | str = True,
          stateful: bool = True, model_path: Path | None = None, color_path: Path | None = None,
          stages: frozenset | set | None = None):
    """Render -> dataset frame for camera `cam` ('top' | 'wrist' | 'inspect').

    img: uint8 sRGB (or float linear 0-1 with linear=True), (H, W, 3|4) or (N, H, W, 3|4), numpy or torch (any
    device). rng: numpy Generator for noise / DR (None = fresh entropy). scale: DR width (0 = the fitted camera).
    dr: explicit DR parameters (draw_dr); with scale > 0 and dr None one is drawn per call. stream: state key (default
    cam; a batch uses stream + index). motion: (dx, dy) px / frame of the image motion (else estimated for cameras
    with an exposure time). codec: True (H.264 worker, emulation fallback) | "emulate" | False. stateful: False =
    every frame is a scene cut (AE / AWB steady state, I frame), e.g. for independent evaluation samples.
    stages: subset of STAGES to run (None = all).
    Returns uint8 RGB of the input's type / device, 3 channels."""
    torch, F = _torch()
    m = model(cam, model_path, color_path)
    on = STAGES if stages is None else frozenset(stages)
    if "codec" not in on:
        codec = False
    rng = rng if rng is not None else np.random.default_rng()
    if dr is None and scale > 0:
        dr = draw_dr(cam, rng, scale, model_path)
    dr = dr or {}
    is_np = not isinstance(img, torch.Tensor)
    x = torch.from_numpy(np.ascontiguousarray(img)) if is_np else img
    dev = x.device
    single = x.dim() == 3
    x = (x[None] if single else x)[..., :3]
    x = x.float() if linear else _srgb_to_lin(x.float() / 255.0)
    x = x.permute(0, 3, 1, 2).contiguous()  # N 3 H W linear
    N, _, H, W = x.shape
    sx = W / 640.0
    # 1. lens
    if m.get("lens") and "lens" in on:
        g = _lens_grid(cam, H, W, dev, m["lens"]).expand(N, -1, -1, -1)
        x = F.grid_sample(x, g, mode="bilinear", padding_mode="border", align_corners=True)
    # 2. colour
    if "color" in on:
        x = _color(x, m, dev)
    x = x.clamp(min=0)
    # 3. AE / AWB (+ DR exposure / WB)
    ae = m.get("ae", {})
    lum = torch.tensor(LUMA, device=dev).view(1, 3, 1, 1)
    meter = (x * lum).sum(1).mean((1, 2)).clamp(min=1e-4).cpu().numpy()  # (N,)
    chroma = x.mean((2, 3)).clamp(min=1e-4).cpu().numpy()  # (N, 3)
    gains = np.ones((N, 3), np.float32)
    outs_small = []
    for i in range(N):
        key = (stream or cam) + (f"#{i}" if N > 1 else "")
        st = _streams.setdefault(key, Stream()) if stateful else Stream()
        e_ss = (float(ae.get("m_ref", meter[i])) / meter[i]) ** float(ae.get("strength", 0.0))
        e_ss = float(np.clip(e_ss, *ae.get("clip", (0.7, 1.4))))
        c = chroma[i] / np.exp(np.log(chroma[i]).mean())
        ref = np.asarray(ae.get("chroma_ref", c), np.float64)
        wb_ss = (ref / c) ** float(ae.get("awb_strength", 0.0))
        small = None
        if stateful:
            import cv2

            small = cv2.resize((x[i] * lum[0]).sum(0).detach().cpu().numpy(), (W // 4, H // 4),
                               interpolation=cv2.INTER_AREA).astype(np.float32)
            cut = st.prev_small is not None and float(np.abs(small - st.prev_small).mean()) > \
                float(ae.get("cut_thr", 0.08))
            if st.exposure is None or cut:
                st.exposure, st.wb = e_ss, wb_ss
            else:
                a = 1.0 - math.exp(-1.0 / max(1e-3, float(ae.get("tau", 1.0))))
                b = 1.0 - math.exp(-1.0 / max(1e-3, float(ae.get("awb_tau", 1.0))))
                st.exposure = st.exposure + a * (e_ss - st.exposure)
                st.wb = st.wb + b * (wb_ss - st.wb)
            if cut and _codec is not None:
                _codec.reset(key)
            e, wb = st.exposure, st.wb
        else:
            e, wb = e_ss, wb_ss
        gains[i] = e * wb * float(m.get("exposure_gain") or 1.0) * float(dr.get("exposure", 1.0)) * \
            np.asarray(dr.get("wb", (1, 1, 1)))
        outs_small.append((key, st, small))
    if "ae" in on:
        x = x * torch.from_numpy(gains).to(dev).view(N, 3, 1, 1)
    # 4. vignetting
    v = m.get("vignette")
    if v and "vignette" in on:
        r2 = _radius2(H, W, dev)
        x = x * (1 + float(v[0]) * r2 + float(v[1]) * r2 ** 2)
    # 5. optics blur + motion blur
    if "blur" in on:
        x = _gauss(x, (float(m.get("blur_sigma", 0.0)) + float(dr.get("blur_add", 0.0))) * sx)
    t_exp = float(m.get("exposure_frames", 0.0))  # exposure time in frame intervals (t_exp * fps)
    if (t_exp > 0 or motion is not None) and "blur" in on:
        ys = []
        for i, (_key, st, small) in enumerate(outs_small):
            if motion is not None:
                d = motion if np.ndim(motion) == 1 else motion[i]
            elif small is not None and st.prev_small is not None and st.prev_small.shape == small.shape:
                d = _global_shift(st.prev_small, small, 4.0 / sx)
            else:
                d = (0.0, 0.0)
            ys.append(_motion(x[i:i + 1], d[0] * t_exp * sx, d[1] * t_exp * sx))
        x = torch.cat(ys)
    for _key, st, small in outs_small:
        if small is not None:
            st.prev_small = small
            st.t += 1
    # 6. sRGB + noise (+ flicker)
    y = _lin_to_srgb(x)
    tone = m.get("tone")
    if tone and "tone" in on:
        b = float(tone.get("black", 0.0))
        y = ((y - b) / (1.0 - b)).clamp(min=0) ** float(tone.get("gamma", 1.0))
        sat = float(tone.get("saturation", 1.0))
        if sat != 1.0:
            Yd = (y * lum).sum(1, keepdim=True)
            y = Yd + sat * (y - Yd)
    y = y * 255.0
    nz = m.get("noise", {})
    s_mid = float(nz.get("mid", 0.0)) * float(dr.get("noise", 1.0))
    if s_mid > 0 and "noise" in on:
        gen = torch.Generator(device=dev).manual_seed(int(rng.integers(2 ** 31)))
        Y = (y * lum).sum(1, keepdim=True)
        fl = float(nz.get("floor", 0.2))
        sig = s_mid * torch.sqrt(fl + (1 - fl) * Y.clamp(min=0) / 128.0)
        n = torch.randn((N, 3, H, W), generator=gen, device=dev)
        cs = float(nz.get("corr", 0.0)) * sx
        if cs > 0.05:
            n = _gauss(n, cs)
            n = n / n.std(dim=(2, 3), keepdim=True).clamp(min=1e-6)
        cr = float(nz.get("chroma", 0.6))
        ny, ncb, ncr = n[:, 0:1] * sig, n[:, 1:2] * sig * cr, n[:, 2:3] * sig * cr
        y = y + torch.cat([ny + 1.402 * ncr, ny - 0.344 * ncb - 0.714 * ncr, ny + 1.772 * ncb], 1)
    flk = m.get("flicker")
    if flk and "noise" in on:
        rows = torch.arange(H, device=dev, dtype=torch.float32).view(1, 1, H, 1)
        ph = torch.from_numpy(rng.uniform(0, 2 * math.pi, (N, 1, 1, 1)).astype(np.float32)).to(dev)
        pulse = torch.from_numpy(rng.normal(0, float(flk.get("frame_gain", 0.0)), (N, 1, 1, 1))
                                 .astype(np.float32)).to(dev)
        period = max(1.0, float(flk.get("row_period", 144))) * H / 480
        band = float(flk.get("row_amp", 0.0)) * torch.sin(2 * math.pi * rows / period + ph)
        y = y * (1 + pulse + band)
    # 7. ISP sharpening
    a = float(m.get("sharpen_amount", 0.0)) * float(dr.get("sharpen", 1.0))
    if a > 0 and "sharpen" in on:
        y = y + a * (y - _gauss(y, float(m.get("sharpen_radius", 1.0)) * sx))
    # 8. quantise + codec
    out = (y.clamp(0, 255) + 0.5).to(torch.uint8).permute(0, 2, 3, 1).contiguous()
    mode = m.get("codec", {}).get("mode", "x264") if codec is True else codec
    if mode:
        arr = out.cpu().numpy()
        crf = int(m.get("codec", {}).get("crf", 30)) + int(dr.get("crf", 0))
        for i, (key, _st, _small) in enumerate(outs_small):
            if mode == "emulate":
                from sim.twin.sensors.codec import emulate

                arr[i] = emulate(arr[i], quality=int(m.get("codec", {}).get("emulate_q", 72)))
            else:
                c = _get_codec()
                arr[i] = c(arr[i], key, crf=crf, keyframe=None if stateful else True)
        out = torch.from_numpy(arr).to(dev)
    out = out[0] if single else out
    return out.cpu().numpy() if is_np else out


def camera_response(img, cam: str, color_path: Path | None = None):
    """The TwinEnv image hook (twin_scene.color_correct): the full camera pipeline of `cam` with the DR registered
    by TwinRandomizer for this episode (if its postprocess still runs) or the nominal camera."""
    p = registered_dr(cam)
    _dr["processed"][cam] = True
    return apply(img, cam, dr=p, color_path=color_path)


def color_only(img, cam: str, color_path: Path | None = None):
    """Only the linear-RGB colour matrix of camera_color.json (the pre-sensor color_correct, for colour fits)."""
    return apply(img, cam, stages={"color"}, stateful=False, color_path=color_path)
