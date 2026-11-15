"""Compose the presentation videos / grids (plain python, the image venv; PyAV is fine here, never in Isaac).

    python sim/twin/presentation/compose.py replay --episode 18          # real row over sim row, 1920x1080 H.264
    python sim/twin/presentation/compose.py wipe --episode 18            # top camera: real | sim, sweeping divider
    python sim/twin/presentation/compose.py split --episode 18           # top camera: fixed 50/50 split
    python sim/twin/presentation/compose.py dr_grid --episode 18 --seeds 1 2 ... --cam top   # DR draws: grid video
    python sim/twin/presentation/compose.py dr_stills --cam top --cols 4 --tile-w 480   # dr_stills.py PNGs -> grid

Inputs from prep.py (work/epNNN/real_<cam>.npy) and twin_replay.py (work/epNNN/sim_<cam>[_drS].npy, telemetry.json).
Outputs in sim/twin/out/presentation/.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[3]
WORK = REPO / "sim/twin/out/presentation/work"
OUT = REPO / "sim/twin/out/presentation"
CAMS = ("top", "wrist", "inspect")
CAM_NAME = {"top": "top camera", "wrist": "wrist camera", "inspect": "inspection camera"}
FPS = 30
_FONTS = {}


def font(size: int, bold: bool = False):
    key = (size, bold)
    if key not in _FONTS:
        import cv2

        base = Path(cv2.__file__).parent / "qt/fonts"
        inter = Path(cv2.__file__).parents[1] / "viser/client/src/assets/Inter-VariableFont_slnt,wght.ttf"
        f = None
        if inter.exists():
            try:
                f = ImageFont.truetype(str(inter), size)
                f.set_variation_by_axes([700 if bold else 450, 0])  # axes: weight, slant
            except Exception:  # noqa: BLE001
                f = None
        if f is None:
            f = ImageFont.truetype(str(base / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")), size)
        _FONTS[key] = f
    return _FONTS[key]


def pill(draw: ImageDraw.ImageDraw, xy, text, size=20, fg=(255, 255, 255), bg=(0, 0, 0, 150), bold=True, pad=8):
    f = font(size, bold)
    x, y = xy
    _, _, r, b = draw.textbbox((x + pad, y + pad // 2), text, font=f)
    draw.rounded_rectangle((x, y, r + pad, b + pad // 2 + 2), radius=6, fill=bg)
    draw.text((x + pad, y + pad // 2), text, font=f, fill=fg)
    return r + pad


class Encoder:
    def __init__(self, path: Path, w: int, h: int, fps: int = FPS, crf: int = 18):
        import av

        self.c = av.open(str(path), "w", options={"movflags": "+faststart"})
        self.s = self.c.add_stream("libx264", rate=fps)
        self.s.width, self.s.height, self.s.pix_fmt = w, h, "yuv420p"
        self.s.options = {"crf": str(crf), "preset": "slow", "profile": "high", "tune": "film"}
        self.path = path

    def write(self, img):
        import av

        arr = np.asarray(img.convert("RGB") if isinstance(img, Image.Image) else img)
        for p in self.s.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(arr), format="rgb24")):
            self.c.mux(p)

    def close(self):
        for p in self.s.encode():
            self.c.mux(p)
        self.c.close()
        print("WROTE", self.path, flush=True)


def load(ep: int, kind: str, cam: str, tag: str = ""):
    return np.load(WORK / f"ep{ep:03d}" / f"{kind}_{cam}{tag}.npy", mmap_mode="r")


def telemetry(ep: int, tag: str = "") -> dict:
    return json.loads((WORK / f"ep{ep:03d}" / f"telemetry{tag}.json").read_text())


def overlay(img: Image.Image, fn):
    o = Image.new("RGBA", img.size, (0, 0, 0, 0))
    fn(ImageDraw.Draw(o))
    return Image.alpha_composite(img.convert("RGBA"), o)


def phase_text(t: int, tel: dict) -> str:
    gf, insp, rel = tel["grasp_frame"], tel["inspection_frame"], tel.get("real_release")
    if t < gf - 3:
        return "reach"
    if t < gf + 12:
        return "grasp"
    if t < insp - 5:
        return "lift + carry to inspection"
    if rel is None or t < rel:
        return "present to inspection camera"
    return "place in plate"


def replay(ep: int, title: str | None, desc: str | None = None, note: str | None = None):
    tel = telemetry(ep)
    real = {c: load(ep, "real", c) for c in CAMS}
    sim = {c: load(ep, "sim", c) for c in CAMS}
    T = min(len(real["top"]), len(sim["top"]))
    W, H, TH, BH = 1920, 1080, 60, 60  # tiles 640x480, header / footer bars
    bricks = np.array(tel["tel"]["brick"])
    z0 = float(bricks[: max(1, tel["grasp_frame"] - 5), 2].mean())
    enc = Encoder(OUT / f"replay_ep{ep:03d}_real_vs_sim.mp4", W, H)
    tgt = tel.get("target") or {}
    title = title or "Recorded teleop demo (real)  vs  Isaac Lab digital twin replaying the same joint commands (sim)"
    for t in range(T):
        canvas = Image.new("RGB", (W, H), (14, 15, 18))
        for i, c in enumerate(CAMS):
            canvas.paste(Image.fromarray(np.asarray(real[c][t])), (640 * i, TH))
            canvas.paste(Image.fromarray(np.asarray(sim[c][t])), (640 * i, TH + 480))

        def deco(d, t=t):
            d.text((24, 16), title, font=font(24, True), fill=(240, 240, 240))
            d.text((W - 24, 18), f"episode {ep}   t = {t / FPS:4.1f} s", font=font(22, False), fill=(200, 200, 200),
                   anchor="ra")
            for i, c in enumerate(CAMS):
                pill(d, (640 * i + 10, TH + 10), f"REAL  ·  {CAM_NAME[c]}", 18, bg=(20, 20, 20, 170))
                pill(d, (640 * i + 10, TH + 490), f"SIM  ·  {CAM_NAME[c]}", 18, fg=(255, 255, 255),
                     bg=(0, 90, 170, 190))
            d.line((0, TH + 480, W, TH + 480), fill=(14, 15, 18, 255), width=3)
            d.line((640, TH, 640, TH + 960), fill=(14, 15, 18, 255), width=3)
            d.line((1280, TH, 1280, TH + 960), fill=(14, 15, 18, 255), width=3)
            zc = (bricks[t, 2] - z0) * 100
            held = "in gripper" if np.linalg.norm(bricks[t] - np.array(tel["tel"]["tip"][t])) < 0.035 else "free"
            foot = (f"{phase_text(t, tel)}   |   sim target brick ({desc or tgt.get('kind', '?')}): "
                    f"{zc:+5.1f} cm above the table, {held}")
            d.text((24, H - BH + 16), foot, font=font(22, False), fill=(225, 225, 225))
            d.text((W - 24, H - BH + 18), note or "open-loop replay · PhysX fin-ray gripper · camera sensor model",
                   font=font(18, False), fill=(150, 150, 150), anchor="ra")
        enc.write(overlay(canvas, deco))
    enc.close()


def wire_look(w: np.ndarray, base: np.ndarray) -> np.ndarray:
    """RTX 'shaded' wireframe (wires in material colours on white) -> wires on a dark blueprint background over a
    faint copy of the rendered frame; near-black wires (robot body) are drawn light cyan so they stay visible."""
    w = np.asarray(w).astype(np.float32)
    bg = w.min(-1) >= 246
    lum = w.mean(-1)
    wires = np.where((lum < 90)[..., None], np.array([120, 205, 255], np.float32), w)
    back = np.array([8, 13, 24], np.float32) + 0.16 * np.asarray(base).astype(np.float32)
    return np.clip(np.where(bg[..., None], back, wires), 0, 255).astype(np.uint8)


def smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def wipe(ep: int, fixed: bool, wire_s: float = 0.0, trans_s: float = 0.8):
    """Top camera, real | sim (registered). wire_s > 0: the sim side starts as the RTX wireframe pass of the same
    camera and timeline (sim_wire.npy), then turns into the render (split: a sweep across the sim half; wipe: a
    crossfade), while the real side stays real."""
    real, sim = load(ep, "real", "top"), load(ep, "sim", "top")
    wire = load(ep, "sim", "wire") if wire_s > 0 else None
    T = min(len(real), len(sim))
    W, H = 1280, 960
    name = f"top_{'split' if fixed else 'wipe'}_ep{ep:03d}{'_wire' if wire is not None else ''}.mp4"
    enc = Encoder(OUT / name, W, H)
    period = 7.0
    t_end = wire_s + trans_s if wire is not None else 0.0
    for t in range(T):
        ts = t / FPS
        r = Image.fromarray(np.asarray(real[t])).resize((W, H), Image.LANCZOS)
        simg = np.asarray(Image.fromarray(np.asarray(sim[t])).resize((W, H), Image.LANCZOS))
        if fixed or ts < t_end:
            frac = 0.5
        else:
            frac = 0.5 + 0.38 * math.sin(2 * math.pi * (ts - t_end) / period)
        x = int(round(frac * W))
        phase, sweep_x = "render", None
        if wire is not None and ts < t_end:
            wf = np.asarray(wire[t])
            if wf.shape[:2] != (H, W):
                wf = np.asarray(Image.fromarray(wf).resize((W, H), Image.LANCZOS))
            wl = wire_look(wf, simg)
            k = smoothstep((ts - wire_s) / trans_s)
            if fixed:  # sweep: render left of the line, wireframe right of it (inside the sim half)
                xs = int(round(W * (0.5 + 0.5 * k)))
                mix = wl.copy()
                mix[:, :xs] = simg[:, :xs]
                sweep_x = xs if 0 < k < 1 else None
            else:
                mix = (k * simg.astype(np.float32) + (1 - k) * wl.astype(np.float32)).astype(np.uint8)
            simg = mix
            phase = "wire" if k < 0.5 else "render"
        s = Image.fromarray(simg)
        canvas = s.copy()
        canvas.paste(r.crop((0, 0, x, H)), (0, 0))

        def deco(d, x=x, t=t, phase=phase, sweep_x=sweep_x):
            d.line((x, 0, x, H), fill=(255, 255, 255, 230), width=3)
            if sweep_x is not None:
                d.line((sweep_x, 0, sweep_x, H), fill=(120, 205, 255, 230), width=3)
            pill(d, (16, 16), "REAL", 26, bg=(20, 20, 20, 170))
            lab = "SIM  ·  geometry (wireframe)" if phase == "wire" else "SIM  (Isaac Lab twin)"
            f = font(26, True)
            wtxt = d.textbbox((0, 0), lab, font=f)[2]
            pill(d, (W - wtxt - 32, 16), lab, 26, bg=(0, 90, 170, 200))
            pill(d, (16, H - 50), f"top camera  ·  episode {ep}  ·  same instant, registered by the fitted camera "
                 f"model  ·  t = {t / FPS:4.1f} s", 19, bold=False, bg=(0, 0, 0, 150))
        enc.write(overlay(canvas, deco))
    enc.close()


def dr_grid(ep: int, seeds: list[int], cam: str, cols: int, tile_w: int, title: str | None = None):
    vids = [load(ep, "sim", cam, f"_dr{s}") for s in seeds]
    T = min(len(v) for v in vids)
    rows = math.ceil(len(vids) / cols)
    th = tile_w * 3 // 4
    gap, TH = 4, 56
    W, H = cols * tile_w + (cols - 1) * gap, TH + rows * th + (rows - 1) * gap
    W, H = W + W % 2, H + H % 2
    enc = Encoder(OUT / f"dr_grid_ep{ep:03d}_{cam}.mp4", W, H)
    for t in range(T):
        canvas = Image.new("RGB", (W, H), (14, 15, 18))
        for k, v in enumerate(vids):
            r, c = divmod(k, cols)
            canvas.paste(Image.fromarray(np.asarray(v[t])).resize((tile_w, th), Image.LANCZOS),
                         (c * (tile_w + gap), TH + r * (th + gap)))

        def deco(d, t=t):
            d.text((16, 14), title or f"Domain randomisation: {len(vids)} draws of lighting, spotlights, materials, "
                   f"camera pose and sensor · {CAM_NAME[cam]}", font=font(22, True), fill=(240, 240, 240))
            d.text((W - 16, 16), f"t = {t / FPS:4.1f} s", font=font(20, False), fill=(190, 190, 190), anchor="ra")
        enc.write(overlay(canvas, deco))
    enc.close()


def dr_still_grid(cam: str, cols: int, tile_w: int, n: int):
    """out/presentation/dr/<cam>_nominal.png + <cam>_s*.png -> one grid PNG (nominal first, outlined)."""
    src = OUT / "dr"
    files = [src / f"{cam}_nominal.png"] + sorted(src.glob(f"{cam}_s*.png"), key=lambda p: int(p.stem.split("_s")[-1]))
    files = files[:n]
    th = tile_w * 3 // 4
    gap, TH = 6, 64
    rows = math.ceil(len(files) / cols)
    W, H = cols * tile_w + (cols - 1) * gap, TH + rows * th + (rows - 1) * gap
    canvas = Image.new("RGB", (W, H), (14, 15, 18))
    for k, f in enumerate(files):
        r, c = divmod(k, cols)
        canvas.paste(Image.open(f).convert("RGB").resize((tile_w, th), Image.LANCZOS),
                     (c * (tile_w + gap), TH + r * (th + gap)))

    def deco(d):
        d.text((16, 12), f"Domain randomisation as the RL saw it  ·  {CAM_NAME[cam]}  ·  same scene, "
               f"{len(files) - 1} random draws", font=font(26, True), fill=(240, 240, 240))
        d.text((16, 42), "lighting rig, spotlights, materials and colours, camera pose / focal, sensor noise / blur / "
               "white balance, visual clutter", font=font(17, False), fill=(170, 170, 170))
        pill(d, (8, TH + 8), "nominal twin", 16, bg=(0, 90, 170, 200))
        for k in range(1, len(files)):
            r, c = divmod(k, cols)
            pill(d, (c * (tile_w + gap) + 8, TH + r * (th + gap) + 8), f"draw {k}", 16, bg=(20, 20, 20, 160))
    img = overlay(canvas, deco).convert("RGB")
    path = OUT / f"dr_montage_{cam}.png"
    img.save(path)
    img.save(path.with_suffix(".jpg"), quality=92)
    print("WROTE", path, flush=True)


def grade(files: list[str], names: list[str]):
    """Beauty PNGs (out/presentation/beauty) -> stills/<name>.png/.jpg: exposure so the 99.7th percentile of luma
    lands at ~240 (the dark-room renders are under-exposed), a gentle S-curve and a soft vignette. Colour balance is
    left as rendered."""
    dst = OUT / "stills"
    dst.mkdir(parents=True, exist_ok=True)
    for f, n in zip(files, names, strict=True):
        x = np.asarray(Image.open(OUT / "beauty" / f).convert("RGB")).astype(np.float32) / 255.0
        lin = x ** 2.2
        y = (0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2])
        gain = min(2.5, (0.87 ** 2.2) / max(1e-4, float(np.percentile(y, 99.7))))
        out = np.clip(lin * gain, 0, 1) ** (1 / 2.2)
        out = out + 0.06 * (out - 0.5) * (1 - np.abs(2 * out - 1))  # mild S-curve around mid-grey
        hh, ww = out.shape[:2]
        yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
        r2 = ((xx - ww / 2) / (ww / 2)) ** 2 + ((yy - hh / 2) / (hh / 2)) ** 2
        out = out * (1 - 0.18 * np.clip(r2 - 0.35, 0, None))[..., None]
        img = Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8))
        img.save(dst / f"{n}.png")
        img.save(dst / f"{n}.jpg", quality=93)
        print("WROTE", dst / f"{n}.png", f"gain {gain:.2f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("replay", "wipe", "split", "dr_grid", "dr_stills", "grade"))
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--title", default=None)
    ap.add_argument("--desc", default=None, help="replay: target brick description in the footer")
    ap.add_argument("--note", default=None, help="replay: footer note (right)")
    ap.add_argument("--seeds", type=int, nargs="*", default=[])
    ap.add_argument("--cam", default="top")
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--tile-w", type=int, default=640)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--wire-s", type=float, default=0.0, help="wipe / split: seconds of wireframe intro (sim_wire.npy)")
    ap.add_argument("--files", nargs="*", default=[], help="grade: beauty PNG names (out/presentation/beauty)")
    ap.add_argument("--names", nargs="*", default=[], help="grade: output names")
    a = ap.parse_args()
    if a.stage == "replay":
        replay(a.episode, a.title, a.desc, a.note)
    elif a.stage in ("wipe", "split"):
        wipe(a.episode, a.stage == "split", a.wire_s)
    elif a.stage == "dr_grid":
        dr_grid(a.episode, a.seeds, a.cam, a.cols, a.tile_w, a.title)
    elif a.stage == "dr_stills":
        dr_still_grid(a.cam, a.cols, a.tile_w, a.n)
    else:
        grade(a.files, a.names)
