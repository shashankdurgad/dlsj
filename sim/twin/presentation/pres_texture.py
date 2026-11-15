"""Presentation-only table texture (plain python; the twin's training texture is NOT changed):

    python sim/twin/presentation/pres_texture.py [--lf-mm 10] [--normal 0.15]

The training texture visuals/textures/table_top.jpg = the world-anchored quilted wood tile + the registered phone photo
+ (inside the top camera's footprint) the LOW frequencies of the top camera's median frame. That last step bakes in
what the top camera saw at the table plane: two pale sheen blobs (ceiling-light reflections in the lacquer), the
footprint's warmer rectangular border, brick-shaped smudges where bricks usually sat, and dark cable strokes near the
tripod. From the top camera they line up with reality; from the wrist / inspect / free cameras they read as a
stained patch. Here: low frequencies everywhere = the tile's (sigma --lf-mm), photo grain kept, dark strokes (cables)
replaced by the tile, and a weaker normal map (the grain normal map + lacquer clear coat give corrugated bands at the
inspect camera's grazing angle). Output: sim/twin/out/presentation/tex/table_pres.{jpg,json}, table_pres_n.png;
use with TwinLook(table_texture=<that jpg>) (beauty.py / twin_replay.py --pres-table).
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
OUT = REPO / "sim/twin/out/presentation/tex"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lf-mm", type=float, default=10.0)
    ap.add_argument("--normal", type=float, default=0.15, help="normal-map strength (wood_texture.py default 0.35)")
    ap.add_argument("--blur-mm", type=float, default=1.0, help="final gaussian blur of the albedo (mm)")
    ap.add_argument("--hf-gain", type=float, default=0.6, help="contrast of detail finer than --lf-mm (grain lines)")
    ap.add_argument("--name", default="table_pres2", help="output stem (a new name: RTX caches textures by path)")
    a = ap.parse_args()
    from sim.twin.visuals.wood_texture import normal_map

    meta = json.loads((TEX / "table_top.json").read_text())
    mm = float(meta["mm_per_px"])
    F = cv2.imread(str(TEX / "table_top.jpg"))[..., ::-1].astype(np.float32) / 255.0
    tile = cv2.imread(str(TEX / meta["tile"]["file"]))[..., ::-1].astype(np.float32) / 255.0
    ny, nx = F.shape[:2]
    Tp = int(meta["tile"]["px"])
    x0, y1 = float(meta["x0"]), float(meta["y0"]) + float(meta["height"])
    cols = (np.arange(nx) + int(round(x0 * 1000 / mm))) % Tp
    rows = (np.arange(ny) - int(round(y1 * 1000 / mm))) % Tp
    T = tile[np.ix_(rows, cols)]
    d = cv2.GaussianBlur(np.abs(F - T).mean(-1), (0, 0), 2 / mm)
    print("outside-patch match |F - T| median", float(np.median(d)), flush=True)
    mp = cv2.GaussianBlur((d > 0.02).astype(np.float32), (0, 0), 10 / mm)[..., None]
    s = a.lf_mm / mm
    w = np.array([0.2126, 0.7152, 0.0722], np.float32)
    # 1) strokes that are not wood (cables, their highlights, brick ghosts' edges): luma far from the local wood level
    lum, lloc = F @ w, cv2.GaussianBlur(cv2.medianBlur((F * 255).astype(np.uint8), 31), (0, 0), 8 / mm) @ w / 255.0
    odd = ((np.abs(lum - lloc) > 0.10) & (mp[..., 0] > 0.05)).astype(np.uint8)
    odd = cv2.morphologyEx(odd, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    odd = cv2.dilate(odd, np.ones((int(4 / mm) | 1,) * 2, np.uint8))
    md = cv2.GaussianBlur(odd.astype(np.float32), (0, 0), 1.5 / mm)[..., None]
    F1 = md * T + (1 - md) * F
    # 2) low frequencies (sheen blobs, footprint border, smudges) = the tile's; the photo's grain is kept
    F2 = F1 - cv2.GaussianBlur(F1, (0, 0), s) + cv2.GaussianBlur(T, (0, 0), s)
    F3 = np.clip(mp * F2 + (1 - mp) * T, 0, 1)
    print("non-wood stroke px replaced", int(odd.sum()), flush=True)
    lf = cv2.GaussianBlur(F3, (0, 0), s)
    F3 = lf + a.hf_gain * (F3 - lf)
    if a.blur_mm > 0:
        F3 = cv2.GaussianBlur(F3, (0, 0), a.blur_mm / mm)
    F3 = np.clip(F3, 0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    u8 = (F3 * 255 + 0.5).astype(np.uint8)
    cv2.imwrite(str(OUT / f"{a.name}.jpg"), u8[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 93])
    half = cv2.resize(F3, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(OUT / f"{a.name}_n.png"), normal_map(half, mm * 2, strength=a.normal)[..., ::-1])
    meta = dict(meta, normal_map=f"{a.name}_n.png", presentation=dict(
        source="table_top.jpg with tile low frequencies + dark strokes -> tile", lf_mm=a.lf_mm, normal=a.normal,
        blur_mm=a.blur_mm, hf_gain=a.hf_gain))
    (OUT / f"{a.name}.json").write_text(json.dumps(meta, indent=1))
    prev = np.concatenate([cv2.resize((F * 255).astype(np.uint8), (1050, 1000)),
                           cv2.resize(u8, (1050, 1000))], 1)
    cv2.imwrite(str(OUT / "compare_before_after.jpg"), prev[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("TEX_OK", OUT / f"{a.name}.jpg", flush=True)


if __name__ == "__main__":
    main()
