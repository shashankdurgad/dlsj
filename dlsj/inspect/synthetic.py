"""Synthetic `inspect`-camera dataset: coloured toy bricks held by a gripper, plain, marked with a cross
(defective) or marked with a circle (minor defect).

Only for validating the pipeline end to end before real images exist; retrain on real `capture.py` data.

    python -m dlsj.inspect.synthetic --out ~/inference/inspect_data/synth3_train --good 120 --bad 30 --minor 30 --seed 0
    python -m dlsj.inspect.synthetic --out ~/inference/inspect_data/synth3_test  --good 40 --bad 40 --minor 40 --seed 1

Spin-and-inspect sweeps (`generate_sweeps`): the brick rolled through N angles, the mark on one face only (visible
in part of the sweep), occasional glare frames:

    python -m dlsj.inspect.synthetic --sweeps dataset --out ~/inference/inspect_data/synthsweep_train --good 60 --views 12
    python -m dlsj.inspect.synthetic --sweeps sweeps  --out ~/inference/inspect_data/synthsweep_eval \
        --good 20 --minor 20 --bad 20 --views 12 --seed 7

Writes full 640x480 RGB frames as `good/<brick>_f<k>.png` (plain), `bad/...` (cross), `minor/...` (circle):
k views of one brick, with small jitter between views, like the 3 frames grabbed at inspection. `dataset.json`
holds the suggested ROI and, per brick, the mark type and its bounding box in frame pixels (first view).
Circles are hand-drawn rings: varied size, thickness, ellipticity, marker colour, wobble and sometimes a small
gap or an overshoot where the pen started.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np

from dlsj.inspect.core import write_rgb

log = logging.getLogger("inspect.synthetic")

FRAME_W, FRAME_H = 640, 480
ROI = (160, 80, 480, 400)  # 320x320 around the present spot
# Brick colours (RGB). Dark ones get a white cross, light ones a black cross.
COLOURS = {
    "red": (200, 30, 35),
    "blue": (25, 70, 190),
    "yellow": (240, 200, 30),
    "green": (40, 150, 60),
    "white": (235, 235, 230),
    "orange": (240, 120, 20),
    "black": (35, 35, 40),
}


def _brick_params(rng: np.random.Generator) -> dict:
    name = rng.choice(list(COLOURS))
    return {
        "colour": name,
        "rgb": np.array(COLOURS[name], np.float32) * rng.uniform(0.9, 1.08),
        "cx": FRAME_W / 2 + rng.uniform(-12, 12),
        "cy": FRAME_H / 2 + 20 + rng.uniform(-10, 10),
        "w": 190 * rng.uniform(0.9, 1.1),
        "h": 110 * rng.uniform(0.9, 1.1),
        "angle": rng.uniform(-6, 6),
        "studs": int(rng.choice([2, 3, 4])),
        "light": rng.uniform(0.9, 1.1),
    }


def _cross_params(rng: np.random.Generator, brick: dict) -> dict:
    dark = brick["colour"] in ("black", "blue", "green", "red")
    size = rng.uniform(0.3, 0.6) * brick["h"]
    return {
        "colour": (245, 245, 245) if dark else (20, 20, 25),
        "thick": int(rng.integers(4, 11)),
        "size": size,
        # Centre on the face, relative to the brick centre.
        "dx": rng.uniform(-0.5, 0.5) * (brick["w"] - size) * 0.8,
        "dy": rng.uniform(-0.3, 0.3) * (brick["h"] - size) * 0.8 + 0.1 * brick["h"],
        "wobble": rng.uniform(0.0, 3.0),
        "rot": rng.uniform(-20, 20),
    }


# Marker colours (RGB) for circles; one with enough contrast to the brick is picked.
MARKERS = {"black": (20, 20, 25), "white": (245, 245, 245), "darkblue": (20, 30, 120), "red": (200, 20, 30)}


def _contrast(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, np.float32) - np.asarray(b, np.float32)))


def _circle_params(rng: np.random.Generator, brick: dict) -> dict:
    size = rng.uniform(0.3, 0.6) * brick["h"]  # outer diameter, same range as the cross extent
    ok = [m for m, c in MARKERS.items() if _contrast(c, brick["rgb"]) > 150]
    marker = str(rng.choice(ok)) if ok else ("white" if brick["colour"] in ("black", "blue") else "black")
    gap = 0.0 if rng.random() < 0.75 else rng.uniform(5, 30)  # degrees left open (25 % of rings)
    return {
        "colour": MARKERS[marker],
        "marker": marker,
        "thick": int(rng.integers(4, 11)),
        "size": size,
        "aspect": rng.uniform(0.8, 1.2),  # ellipticity of a hand-drawn ring
        "dx": rng.uniform(-0.5, 0.5) * (brick["w"] - size) * 0.8,
        "dy": rng.uniform(-0.3, 0.3) * (brick["h"] - size) * 0.8 + 0.1 * brick["h"],
        "wobble": rng.uniform(0.0, 0.08),  # relative radius wobble (low-frequency)
        "phase": rng.uniform(0, 2 * np.pi, 3),
        "start": rng.uniform(0, 360),
        "sweep": 360.0 - gap + (rng.uniform(0, 25) if gap == 0 else 0.0),  # closed rings may overshoot
        "rot": rng.uniform(-90, 90),
    }


def _circle_points(c: dict) -> np.ndarray:
    """Brick-local polyline of a wobbly ring (before the brick transform)."""
    t = np.deg2rad(c["start"] + np.linspace(0, c["sweep"], 72))
    r = c["size"] / 2 - c["thick"] / 2
    wob = 1 + c["wobble"] * (np.sin(2 * t + c["phase"][0]) + 0.6 * np.sin(3 * t + c["phase"][1])
                             + 0.4 * np.sin(5 * t + c["phase"][2])) / 2
    pts = np.stack([r * wob * np.cos(t) * c["aspect"], r * wob * np.sin(t) / c["aspect"]], axis=1)
    a = np.deg2rad(c["rot"])
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return pts @ rot.T + [c["dx"], c["dy"]]


def _cross_points(cross: dict, rng: np.random.Generator) -> list[np.ndarray]:
    s = cross["size"] / 2
    a = np.deg2rad(cross["rot"])
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    lines = []
    for p0, p1 in (((-s, -s), (s, s)), ((-s, s), (s, -s))):
        t = np.linspace(0, 1, 12)[:, None]
        line = np.array(p0) * (1 - t) + np.array(p1) * t
        line = line + rng.normal(0, cross["wobble"], line.shape)  # hand-drawn wobble
        lines.append(line @ rot.T + [cross["dx"], cross["dy"]])
    return lines


def mark_box(brick: dict, mark: dict, kind: str) -> tuple[int, int, int, int]:
    """Bounding box (frame pixels) of the mark's centre line, grown by half the stroke width."""
    if kind == "circle":
        pts = _to_frame(_circle_points(mark), brick)
    else:
        s = mark["size"] / 2
        a = np.deg2rad(mark["rot"])
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        corners = np.array([[-s, -s], [s, s], [-s, s], [s, -s]]) @ rot.T + [mark["dx"], mark["dy"]]
        pts = _to_frame(corners, brick)
    h = mark["thick"] / 2 + 1
    return (int(pts[:, 0].min() - h), int(pts[:, 1].min() - h), int(np.ceil(pts[:, 0].max() + h)),
            int(np.ceil(pts[:, 1].max() + h)))


def _to_frame(pts: np.ndarray, b: dict) -> np.ndarray:
    """Brick-local points (origin at the brick centre) → frame pixels, rotated by the brick angle."""
    a = np.deg2rad(b["angle"])
    r = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return pts @ r.T + np.array([b["cx"], b["cy"]])


def render(brick: dict, cross: dict | None, rng: np.random.Generator, circle: dict | None = None) -> np.ndarray:
    """One RGB frame. `rng` only drives per-view noise (pose jitter is already in `brick`)."""
    yy, xx = np.mgrid[0:FRAME_H, 0:FRAME_W].astype(np.float32)
    bg = 200 + 18 * (xx / FRAME_W) - 10 * (yy / FRAME_H)  # plain, lit backdrop
    img = np.repeat(bg[..., None], 3, axis=2) * np.array([1.0, 1.0, 0.97], np.float32)

    w, h = brick["w"], brick["h"]
    rgb = brick["rgb"]
    # Soft shadow, then the body.
    body = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    cv2.fillPoly(img, [np.round(_to_frame(body + [6, 8], brick)).astype(np.int32)], (150, 150, 145))
    img = cv2.GaussianBlur(img, (0, 0), 4)
    cv2.fillPoly(img, [np.round(_to_frame(body, brick)).astype(np.int32)], tuple(float(c) for c in rgb))
    # Top face strip (lighter) and studs on top.
    top = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, -h / 2 + 0.18 * h], [-w / 2, -h / 2 + 0.18 * h]])
    cv2.fillPoly(img, [np.round(_to_frame(top, brick)).astype(np.int32)], tuple(float(min(255, c * 1.15 + 15)) for c in rgb))
    n = brick["studs"]
    for i in range(n):
        sx = -w / 2 + w * (i + 0.5) / n
        stud = np.array([[sx - 0.11 * w / n * 2, -h / 2], [sx + 0.11 * w / n * 2, -h / 2],
                         [sx + 0.11 * w / n * 2, -h / 2 - 16], [sx - 0.11 * w / n * 2, -h / 2 - 16]])
        cv2.fillPoly(img, [np.round(_to_frame(stud, brick)).astype(np.int32)], tuple(float(c * 0.92) for c in rgb))
        c = np.round(_to_frame(np.array([[sx, -h / 2 + 0.55 * h]]), brick)).astype(int)[0]
        cv2.circle(img, (int(c[0]), int(c[1])), int(0.16 * w / n), tuple(float(min(255, v * 1.08 + 6)) for v in rgb), 2, cv2.LINE_AA)
    # Edge outline.
    cv2.polylines(img, [np.round(_to_frame(body, brick)).astype(np.int32)], True, tuple(float(c * 0.7) for c in rgb), 2, cv2.LINE_AA)

    if cross is not None:
        for line in _cross_points(cross, rng):
            pts = np.round(_to_frame(line, brick)).astype(np.int32)
            cv2.polylines(img, [pts], False, cross["colour"], cross["thick"], cv2.LINE_AA)
    if circle is not None:
        ring = _circle_points(circle) + rng.normal(0, 0.4, (72, 2))  # per-view pen jitter (sub-pixel)
        pts = np.round(_to_frame(ring, brick)).astype(np.int32)
        cv2.polylines(img, [pts], False, circle["colour"], circle["thick"], cv2.LINE_AA)

    # Gripper fingers: two dark bars coming down from the top edge onto the brick's sides.
    for side in (-1, 1):
        fx = brick["cx"] + side * (w / 2 + 6)
        x0, x1 = int(fx - 14), int(fx + 14)
        y1 = int(brick["cy"] + 0.25 * h)
        cv2.rectangle(img, (x0, 0), (x1, y1), (55, 55, 60), -1)
        cv2.rectangle(img, (x0 + 3, 0), (x0 + 7, y1 - 4), (85, 85, 90), -1)

    img = img * brick["light"] * rng.uniform(0.97, 1.03)
    img = cv2.GaussianBlur(img, (0, 0), 0.8) + rng.normal(0, 3.0, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def jitter_view(brick: dict, rng: np.random.Generator) -> dict:
    """Small change between the views of one inspection (arm settling, 1-2 px)."""
    b = dict(brick)
    b["cx"] += rng.normal(0, 1.0)
    b["cy"] += rng.normal(0, 1.0)
    b["angle"] += rng.normal(0, 0.3)
    return b


KIND_MARK = {"good": "none", "bad": "cross", "minor": "circle"}  # folder -> mark type


def generate(out: Path, n_good: int, n_bad: int, views: int, seed: int, n_minor: int = 0) -> None:
    rng = np.random.default_rng(seed)
    meta = []
    for kind, n in (("good", n_good), ("bad", n_bad), ("minor", n_minor)):
        if n <= 0 and kind == "minor":
            continue
        d = out / kind
        d.mkdir(parents=True, exist_ok=True)
        prefix = "c" if kind == "minor" else "b"  # brick ids stay unique across folders
        for i in range(n):
            brick = _brick_params(rng)
            cross = _cross_params(rng, brick) if kind == "bad" else None
            circle = _circle_params(rng, brick) if kind == "minor" else None
            views_b = [jitter_view(brick, rng) for _ in range(views)]
            for k, vb in enumerate(views_b):
                write_rgb(d / f"s{seed}{prefix}{i:04d}_f{k}.png", render(vb, cross, rng, circle))
            mark = cross or circle
            meta.append({"brick": f"s{seed}{prefix}{i:04d}", "kind": kind, "mark": KIND_MARK[kind],
                         "colour": str(brick["colour"]),
                         "cross_thick": cross["thick"] if cross else None,
                         "mark_thick": mark["thick"] if mark else None,
                         "mark_box": list(mark_box(views_b[0], mark, KIND_MARK[kind])) if mark else None,
                         "circle_gap_deg": round(360 - circle["sweep"], 1) if circle and circle["sweep"] < 360
                         else (0.0 if circle else None)})
    (out / "dataset.json").write_text(json.dumps(
        {"source": "synthetic", "seed": seed, "frame_size": [FRAME_W, FRAME_H], "crop": list(ROI),
         "views_per_brick": views, "bricks": meta}, indent=1) + "\n")
    log.info("wrote %d good + %d bad (cross) + %d minor (circle) bricks x %d views to %s", n_good, n_bad, n_minor,
             views, out)


# ----------------------------------------------------------------------------- sweeps: the brick spun on wrist_roll
DEPTH = 0.62  # side-face width relative to the front face
FACES = {"front": 0.0, "right": 90.0, "back": 180.0, "left": -90.0}  # face normal angles at roll 0


def _face_geometry(brick: dict, theta_deg: float):
    """Visible vertical faces of the box rolled by `theta` about the vertical axis (orthographic, camera on +z).

    Yields (face name, x_left, x_right, facing) with x in pixels relative to the brick centre and `facing` =
    cos(angle between the face normal and the camera axis)."""
    w, d = brick["w"], brick["w"] * DEPTH
    t = np.deg2rad(theta_deg)
    corners = {"A": (-w / 2, d / 2), "B": (w / 2, d / 2), "C": (w / 2, -d / 2), "D": (-w / 2, -d / 2)}
    xp = {k: x * np.cos(t) + z * np.sin(t) for k, (x, z) in corners.items()}
    edges = {"front": ("A", "B"), "right": ("B", "C"), "back": ("C", "D"), "left": ("D", "A")}
    for name, (a, b) in edges.items():
        n = np.deg2rad(FACES[name] + theta_deg)
        facing = float(np.cos(n))
        if facing > 0.02:
            x0, x1 = sorted((xp[a], xp[b]))
            yield name, x0, x1, facing


def _face_texture(brick: dict, face: str, mark: dict | None, kind: str, mark_face: str, rng: np.random.Generator,
                  glare: bool) -> np.ndarray:
    """Unforeshortened face image (h x face width), brick colour with the mark if it is on this face."""
    fw = int(round(brick["w"] * (1.0 if face in ("front", "back") else DEPTH)))
    fh = int(round(brick["h"]))
    rgb = brick["rgb"]
    tex = np.empty((fh, fw, 3), np.float32)
    tex[:] = rgb
    tex[: int(0.18 * fh)] = np.minimum(255, rgb * 1.15 + 15)  # lighter top strip, like the frontal render
    cv2.rectangle(tex, (0, 0), (fw - 1, fh - 1), tuple(float(c * 0.7) for c in rgb), 2, cv2.LINE_AA)
    if mark is not None and face == mark_face:
        m = dict(mark)
        lim = max(0.0, (fw - m["size"]) / 2 * 0.8)
        m["dx"] = float(np.clip(m["dx"], -lim, lim))
        off = np.array([fw / 2, fh / 2])
        if kind == "circle":
            ring = _circle_points(m) + rng.normal(0, 0.4, (72, 2))
            cv2.polylines(tex, [np.round(ring + off).astype(np.int32)], False, m["colour"], m["thick"], cv2.LINE_AA)
        else:
            for line in _cross_points(m, rng):
                cv2.polylines(tex, [np.round(line + off).astype(np.int32)], False, m["colour"], m["thick"],
                              cv2.LINE_AA)
    if glare:  # specular highlight: bright soft ellipse
        g = np.zeros((fh, fw), np.float32)
        cv2.ellipse(g, (int(fw * rng.uniform(0.3, 0.7)), int(fh * rng.uniform(0.35, 0.65))),
                    (int(fw * rng.uniform(0.08, 0.16)), int(fh * rng.uniform(0.08, 0.16))), rng.uniform(0, 180), 0, 360,
                    1.0, -1, cv2.LINE_AA)
        g = cv2.GaussianBlur(g, (0, 0), 3)[..., None]
        tex = tex * (1 - g) + 252 * g
    return tex


def render_rotated(brick: dict, theta_deg: float, mark: dict | None, kind: str, mark_face: str,
                   rng: np.random.Generator, glare: bool = False) -> np.ndarray:
    """One `inspect` frame of the held brick rolled by `theta_deg` (0 = the front face to the camera)."""
    yy, xx = np.mgrid[0:FRAME_H, 0:FRAME_W].astype(np.float32)
    bg = 200 + 18 * (xx / FRAME_W) - 10 * (yy / FRAME_H)
    img = np.repeat(bg[..., None], 3, axis=2) * np.array([1.0, 1.0, 0.97], np.float32)
    cx, cy, h = brick["cx"], brick["cy"], brick["h"]
    faces = list(_face_geometry(brick, theta_deg))
    xl, xr = min(f[1] for f in faces), max(f[2] for f in faces)
    sh = np.array([[cx + xl + 6, cy - h / 2 + 8], [cx + xr + 6, cy - h / 2 + 8], [cx + xr + 6, cy + h / 2 + 8],
                   [cx + xl + 6, cy + h / 2 + 8]])
    cv2.fillPoly(img, [np.round(sh).astype(np.int32)], (150, 150, 145))
    img = cv2.GaussianBlur(img, (0, 0), 4)
    y0 = int(round(cy - h / 2))
    glare_face = max(faces, key=lambda f: f[3])[0] if glare else None
    for name, x0, x1, facing in faces:
        pw = int(round(x1 - x0))
        if pw < 2:
            continue
        tex = _face_texture(brick, name, mark, kind, mark_face, rng, glare and name == glare_face)
        tex = cv2.resize(tex, (pw, tex.shape[0]), interpolation=cv2.INTER_AREA) * (0.6 + 0.4 * facing)
        xa = int(round(cx + x0))
        ya, yb, xb = max(0, y0), min(FRAME_H, y0 + tex.shape[0]), min(FRAME_W, xa + pw)
        img[ya:yb, max(0, xa):xb] = tex[ya - y0:yb - y0, max(0, xa) - xa:xb - xa]
    # studs along the top edge
    n = brick["studs"]
    for i in range(n):
        sx = cx + xl + (xr - xl) * (i + 0.5) / n
        sw = 0.22 * (xr - xl) / n
        cv2.rectangle(img, (int(sx - sw), y0 - 16), (int(sx + sw), y0), tuple(float(c * 0.92) for c in brick["rgb"]), -1)
    for side, xe in ((-1, xl), (1, xr)):  # gripper fingers on the silhouette's sides
        fx = cx + xe + side * 6
        cv2.rectangle(img, (int(fx - 14), 0), (int(fx + 14), int(cy + 0.25 * h)), (55, 55, 60), -1)
        cv2.rectangle(img, (int(fx - 11), 0), (int(fx - 7), int(cy + 0.25 * h) - 4), (85, 85, 90), -1)
    img = img * brick["light"] * rng.uniform(0.97, 1.03)
    img = cv2.GaussianBlur(img, (0, 0), 0.8) + rng.normal(0, 3.0, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


SWEEP_LABEL = {"plain": "good", "circle": "minor", "cross": "bad"}  # sweep label -> dataset folder


def sweep_angles(frames: int, range_deg: float) -> list[float]:
    return [float(a) for a in np.linspace(-range_deg / 2, range_deg / 2, frames)]


def generate_sweeps(out: Path, n_plain: int, n_circle: int, n_cross: int, frames: int = 12, range_deg: float = 180.0,
                    seed: int = 0, layout: str = "sweeps", glare_prob: float = 0.4,
                    mark_faces: tuple[str, ...] = ("front", "right", "left", "right", "left")) -> list[dict]:
    """Bricks spun through `frames` roll angles. The mark sits on one face (default: 1/5 front, 4/5 a side face,
    so it is visible in only part of the sweep); with `glare_prob` one frame of the sweep gets a specular highlight.

    layout "sweeps": `out/<id>_<label>/frame_XX.png` + `sweep.json` (the format `dlsj.loop.sweep_test` saves);
    layout "dataset": `out/{good,minor,bad}/<id>_f<k>.png` + `dataset.json` (Anomalib training / calibration)."""
    rng = np.random.default_rng(seed)
    angles = sweep_angles(frames, range_deg)
    metas = []
    for label, n in (("plain", n_plain), ("circle", n_circle), ("cross", n_cross)):
        for i in range(n):
            brick = _brick_params(rng)
            brick["angle"] = 0.0
            kind = {"plain": "none", "circle": "circle", "cross": "cross"}[label]
            mark = _circle_params(rng, brick) if kind == "circle" else _cross_params(rng, brick) if kind == "cross" \
                else None
            face = str(rng.choice(list(mark_faces))) if mark is not None else ""
            start = float(rng.uniform(-12, 12))  # how squarely the pick skill presented the brick
            glare_at = int(rng.integers(frames)) if rng.random() < glare_prob else -1
            bid = f"s{seed}{label[:2]}{i:04d}"
            vis = []
            imgs = []
            for k, a in enumerate(angles):
                vb = jitter_view(brick, rng)
                theta = start + a
                visible = bool(mark is not None and np.cos(np.deg2rad(FACES[face] + theta)) > 0.35)
                vis.append(visible)
                imgs.append(render_rotated(vb, theta, mark, kind, face, rng, glare=k == glare_at))
            meta = {"brick": bid, "label": label, "colour": str(brick["colour"]), "mark_face": face or None,
                    "start_offset_deg": round(start, 2), "glare_frame": glare_at if glare_at >= 0 else None,
                    "mark_visible": vis, "n_visible": int(sum(vis))}
            if layout == "dataset":
                d = out / SWEEP_LABEL[label]
                d.mkdir(parents=True, exist_ok=True)
                for k, im in enumerate(imgs):
                    write_rgb(d / f"{bid}_f{k}.png", im)
            else:
                d = out / f"{bid}_{label}"
                d.mkdir(parents=True, exist_ok=True)
                fr = []
                for k, (im, a) in enumerate(zip(imgs, angles)):
                    name = f"frame_{k:02d}.png"
                    write_rgb(d / name, im)
                    fr.append({"i": k, "file": name, "angle": round(a, 2), "target": round(a, 2), "settled": True,
                               "mark_visible": vis[k]})
                (d / "sweep.json").write_text(json.dumps(
                    {"version": 1, "label": label, "source": "synthetic", "seed": seed, "camera": "inspect",
                     "crop": list(ROI), "frames": fr, "synthetic": meta}, indent=1) + "\n")
            metas.append(meta)
    if layout == "dataset":
        (out / "dataset.json").write_text(json.dumps(
            {"source": "synthetic-sweeps", "seed": seed, "frame_size": [FRAME_W, FRAME_H], "crop": list(ROI),
             "views_per_brick": frames, "angles": angles, "bricks": metas}, indent=1) + "\n")
    log.info("wrote %d plain + %d circle + %d cross sweeps x %d frames (%s) to %s", n_plain, n_circle, n_cross, frames,
             layout, out)
    return metas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--good", type=int, default=120, help="plain bricks")
    ap.add_argument("--bad", type=int, default=30, help="cross-marked bricks (defective)")
    ap.add_argument("--minor", type=int, default=0, help="circle-marked bricks (minor defect), in minor/")
    ap.add_argument("--views", type=int, default=3, help="frames per brick")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweeps", choices=["sweeps", "dataset"], default=None,
                    help="spin-and-inspect data instead: --good/--minor/--bad bricks rolled through --views angles "
                         "over --range degrees; 'sweeps' = sweep folders, 'dataset' = good/ minor/ bad/ frames")
    ap.add_argument("--range", type=float, default=180.0, help="sweep range (deg)")
    ap.add_argument("--glare-prob", type=float, default=0.4, help="sweeps: chance of one glare frame per brick")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.sweeps:
        generate_sweeps(args.out.expanduser(), args.good, args.minor, args.bad, args.views, args.range, args.seed,
                        args.sweeps, args.glare_prob)
        return
    generate(args.out.expanduser(), args.good, args.bad, args.views, args.seed, args.minor)


if __name__ == "__main__":
    main()
