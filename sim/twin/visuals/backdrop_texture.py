"""Backdrops of the twin: what the cameras see behind / beyond the table (plain Python, PyAV: never import in Isaac).

    docker run ... python sim/twin/visuals/backdrop_texture.py [--episodes 100] [--keep 40]

1. inspect-camera billboard (textures/backdrop_inspect.jpg + .json): the laptop, the carton and the hall behind them,
   as a vertical billboard standing on the table ~0.75 m in front of the inspection camera. The old one was the median
   of frame 0 of every third episode: the inspection camera sits on a swivel tripod and moved between sessions, so
   that median was blurred, and its texture had only ~400 px across the camera's view. Now: the median over the
   `--keep` most mutually consistent of the last `--episodes` episodes (same camera pose; moving people average out),
   projected with the measured inspect camera (calib/cameras_measured.json) at 0.5 mm / texel over exactly the part
   of the billboard the camera sees (no edge-clamped smear; beyond it the hall cylinder shows).
2. hall cylinder (sim/twin/data/photos/derived/hall_cylinder.jpg, git-ignored like the photos): the upper part of
   sim/twin/data/photos/station_panorama.jpg (the hall above table level), people blurred beyond recognition
   (gaussian sigma ~1 % of the width), mirrored to close the circle. twin_scene puts it on a cylinder around the
   station, so a camera looking past the table edge sees the hall instead of the pale dome ("sky wedge").
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
TEX = REPO / "sim/twin/visuals/textures"
DERIVED = REPO / "sim/twin/data/photos/derived"
PANO = REPO / "sim/twin/data/photos/station_panorama.jpg"
OUT = REPO / "sim/twin/out/visual_fit"
TABLE_TOP_Z = -0.002


def inspect_camera():
    """(R world->cv, position, K) of the measured inspection camera (factory intrinsics incl. principal point)."""
    m = json.loads((REPO / "sim/twin/calib/cameras_measured.json").read_text())["inspect"]
    K = np.array([[m["fx"], 0, m["cx"]], [0, m["fy"], m["cy"]], [0, 0, 1.0]])
    return np.array(m["world_to_cv"]), np.array(m["position"]), K


def consistent_median(n_last: int, keep: int) -> np.ndarray:
    """Median of the inspect frames (frames 0 and 20) of the `keep` episodes, among the last `n_last`, whose
    background is closest to the group's median (i.e. recorded with the camera at the same pose)."""
    from sim.twin.visuals.realdata import Dataset

    ds = Dataset()
    eps = list(range(len(ds) - n_last, len(ds)))
    fr = [ds.frames(ep, "inspect", [0, 20]) for ep in eps]
    small = np.stack([cv2.resize(cv2.cvtColor(f[0], cv2.COLOR_RGB2GRAY), (80, 60), interpolation=cv2.INTER_AREA)[:32]
                      for f in fr]).astype(np.float32)
    ref = np.median(small, 0)
    for _ in range(3):  # iterate: the median of the closest frames defines the pose cluster
        dist = np.abs(small - ref).mean((1, 2))
        order = np.argsort(dist)[:keep]
        ref = np.median(small[order], 0)
    stack = np.stack([f for i in order for f in fr[i]])
    print(f"inspect median over {len(stack)} frames of {keep} episodes (distance {dist[order].mean():.1f} vs "
          f"{dist.mean():.1f} for all)", flush=True)
    return np.median(stack, 0).astype(np.uint8)


def billboard(img, edge_row: float = 250.0, mm: float = 0.5):
    """Billboard plane (origin = bottom-left, u = camera right (horizontal), v = up) where the image row `edge_row`
    meets the table, exactly as wide and high as the camera's view of it (beyond it the hall cylinder shows), and
    its texture by projecting `img` from the measured camera."""
    r_cv, pos, K = inspect_camera()
    h, w = img.shape[:2]

    def ray_table(u, v):
        d = np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0]) @ r_cv
        return pos + (TABLE_TOP_Z - pos[2]) / d[2] * d

    p_edge = ray_table(K[0, 2], edge_row)
    fwd = r_cv[2].copy()
    fwd[2] = 0
    fwd /= np.linalg.norm(fwd)
    up = np.array([0.0, 0.0, 1.0])
    u_ax = np.cross(fwd, up)
    dist = float((p_edge - pos) @ fwd)
    left, right = dist * K[0, 2] / K[0, 0], dist * (w - K[0, 2]) / K[0, 0]  # visible extent at the billboard
    origin = p_edge - u_ax * left
    width = left + right
    top_dir = np.array([0.0, -K[1, 2] / K[1, 1], 1.0]) @ r_cv  # exact: ray of the top-centre pixel
    height = float(pos[2] + top_dir[2] * dist / (top_dir @ fwd) - TABLE_TOP_Z - 0.003)
    nx, ny = int(width * 1000 / mm), int(height * 1000 / mm)
    gs, gt = np.meshgrid((np.arange(nx) + 0.5) / nx * width, height - (np.arange(ny) + 0.5) / ny * height)
    pts = origin + gs[..., None] * u_ax + gt[..., None] * up
    pc = (pts - pos) @ r_cv.T
    mx = (pc[..., 0] / pc[..., 2] * K[0, 0] + K[0, 2]).astype(np.float32)
    my = (pc[..., 1] / pc[..., 2] * K[1, 1] + K[1, 2]).astype(np.float32)
    tex = cv2.remap(img, mx.clip(0, w - 1), my.clip(0, h - 1), cv2.INTER_LANCZOS4)
    plane = dict(origin=origin.tolist(), u_axis=u_ax.tolist(), v_axis=up.tolist(), width=width, height=height,
                 edge_row=edge_row, distance_m=dist, mm_per_px=mm,
                 note="texture row 0 = top (v = height), col 0 = origin side; emissive (unlit) in twin_scene",
                 camera=dict(profile="cameras_measured.json", cam="inspect"))
    return tex, plane


def hall_cylinder(top_frac: float = 0.44, sigma_frac: float = 0.01) -> np.ndarray | None:
    """Upper `top_frac` of the panorama (the hall above table level), people blurred, mirrored to 360 degrees."""
    if not PANO.exists():
        return None
    p = cv2.imread(str(PANO))[..., ::-1]
    strip = p[4:int(p.shape[0] * top_frac)]
    strip = cv2.GaussianBlur(strip, (0, 0), sigma_frac * p.shape[1])
    return np.concatenate([strip, strip[:, ::-1]], 1)


def main():
    """Build both backdrops."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--keep", type=int, default=40)
    ap.add_argument("--edge-row", type=float, default=250.0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    med = consistent_median(a.episodes, a.keep)
    cv2.imwrite(str(OUT / "inspect_median_consistent.png"), med[..., ::-1])
    tex, plane = billboard(med, a.edge_row)
    cv2.imwrite(str(TEX / "backdrop_inspect.jpg"), tex[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
    (TEX / "backdrop_inspect.json").write_text(json.dumps(plane, indent=1))
    print("billboard", tex.shape, {k: plane[k] for k in ("width", "height", "distance_m")}, flush=True)
    hall = hall_cylinder()
    if hall is not None:
        DERIVED.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(DERIVED / "hall_cylinder.jpg"), hall[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
        print("hall cylinder", hall.shape, "mean rgb", hall.reshape(-1, 3).mean(0).round(1).tolist())
    print("BACKDROP_OK")


if __name__ == "__main__":
    main()
