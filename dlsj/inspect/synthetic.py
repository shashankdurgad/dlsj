"""Synthetic `inspect`-camera dataset: coloured toy bricks held by a gripper, plain or marked with a cross.

Only for validating the pipeline end to end before real images exist; retrain on real `capture.py` data.

    python -m dlsj.inspect.synthetic --out ~/inference/inspect_data/synth_train --good 120 --bad 30 --seed 0
    python -m dlsj.inspect.synthetic --out ~/inference/inspect_data/synth_test  --good 40  --bad 40 --seed 1

Writes full 640x480 RGB frames as `good/bNNNN_f<k>.png` / `bad/...` (k views of one brick, with small
jitter between views, like the 3 frames grabbed at inspection) and `dataset.json` with the suggested ROI.
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


def _to_frame(pts: np.ndarray, b: dict) -> np.ndarray:
    """Brick-local points (origin at the brick centre) → frame pixels, rotated by the brick angle."""
    a = np.deg2rad(b["angle"])
    r = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return pts @ r.T + np.array([b["cx"], b["cy"]])


def render(brick: dict, cross: dict | None, rng: np.random.Generator) -> np.ndarray:
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
        s = cross["size"] / 2
        a = np.deg2rad(cross["rot"])
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        for p0, p1 in (((-s, -s), (s, s)), ((-s, s), (s, -s))):
            t = np.linspace(0, 1, 12)[:, None]
            line = np.array(p0) * (1 - t) + np.array(p1) * t
            line = line + rng.normal(0, cross["wobble"], line.shape)  # hand-drawn wobble
            line = line @ rot.T + [cross["dx"], cross["dy"]]
            pts = np.round(_to_frame(line, brick)).astype(np.int32)
            cv2.polylines(img, [pts], False, cross["colour"], cross["thick"], cv2.LINE_AA)

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


def generate(out: Path, n_good: int, n_bad: int, views: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    meta = []
    for kind, n in (("good", n_good), ("bad", n_bad)):
        d = out / kind
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            brick = _brick_params(rng)
            cross = _cross_params(rng, brick) if kind == "bad" else None
            for k in range(views):
                write_rgb(d / f"s{seed}b{i:04d}_f{k}.png", render(jitter_view(brick, rng), cross, rng))
            meta.append({"brick": f"s{seed}b{i:04d}", "kind": kind, "colour": str(brick["colour"]),
                         "cross_thick": cross["thick"] if cross else None})
    (out / "dataset.json").write_text(json.dumps(
        {"source": "synthetic", "seed": seed, "frame_size": [FRAME_W, FRAME_H], "crop": list(ROI),
         "views_per_brick": views, "bricks": meta}, indent=1) + "\n")
    log.info("wrote %d good + %d bad bricks x %d views to %s", n_good, n_bad, views, out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--good", type=int, default=120, help="plain bricks")
    ap.add_argument("--bad", type=int, default=30, help="cross-marked bricks")
    ap.add_argument("--views", type=int, default=3, help="frames per brick")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    generate(args.out.expanduser(), args.good, args.bad, args.views, args.seed)


if __name__ == "__main__":
    main()
