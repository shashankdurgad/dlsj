"""Lighting-DR grid: the twin at real recorded states under light draws at scale 0 / 0.5 / 1, next to the real frames.

    NAME=dlsj-twin-light sim/twin/run.sh sim/twin/lighting/dr_grid.py [--samples 2 37] [--draws 4] [--tag v1]

Only the light section of TwinRandomizer is enabled (cameras, materials, clutter, image effects off), so the grid
shows what the lighting DR alone does. Per evaluation sample one row per scale: real | nominal (scale 0) | `--draws`
random draws (seeded), for the top and inspect cameras. Writes sim/twin/out/lighting/dr_grid_<tag>_<cam>.jpg and
dr_grid_<tag>.json (the draws, incl. how many spots each had and the mean image brightness per draw).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
OUT = REPO / "sim/twin/out/lighting"
VF = REPO / "sim/twin/out/visual_fit"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, nargs="*", default=[2, 37, 113])
    ap.add_argument("--scales", type=float, nargs="*", default=[0.0, 0.5, 1.0])
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args()

    import cv2

    from sim.twin.env.randomize import TwinRandomizer
    from sim.twin.visuals.fit.harness import Harness
    from sim.twin.visuals.twin_scene import color_correct

    h = Harness(passes=3)
    dr = TwinRandomizer({k: {"enabled": False} for k in ("camera", "material", "clutter", "image")})
    real = np.load(VF / "evalset.npz")
    cams = ("top", "inspect")
    rows = {c: [] for c in cams}
    log = []

    def shot():
        h.render()
        img = h.render()
        return {c: color_correct(img[c], c) for c in cams}

    def tile(img, text):
        t = cv2.resize(np.ascontiguousarray(img), (320, 240), interpolation=cv2.INTER_AREA)
        cv2.putText(t, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return t

    for si in args.samples:
        s = h.samples[si]
        h.pose(s)
        for sc in args.scales:
            dr.restore(h.env)
            h.pose(s)
            nom = shot()
            row = {c: [tile(real[c][si], f"real ep{s['ep']} {s['moment']}"), tile(nom[c], f"nominal (s={sc})")]
                   for c in cams}
            dr.set_light_scale(sc)
            for k in range(args.draws):
                rng = np.random.default_rng(1000 * si + 10 * k + int(sc * 10))
                d = dr(h.env, rng)["light"]
                img = shot()
                n_sp = len(d["spots"][0])
                txt = f"s={sc} spots={n_sp}" + (" off" if d["off"] else "")
                for c in cams:
                    row[c].append(tile(img[c], txt))
                log.append(dict(sample=si, scale=sc, draw=k, spots=n_sp, off=d["off"], exposure=d["exposure"],
                                mean_top=round(float(img["top"].mean()), 1), light=d))
            for c in cams:
                rows[c].append(np.concatenate(row[c], 1))
        print(f"[dr_grid] sample {si} done", flush=True)
    dr.restore(h.env)
    OUT.mkdir(parents=True, exist_ok=True)
    for c in cams:
        cv2.imwrite(str(OUT / f"dr_grid_{args.tag}_{c}.jpg"), np.concatenate(rows[c], 0)[..., ::-1],
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
    (OUT / f"dr_grid_{args.tag}.json").write_text(json.dumps(log, indent=1))
    print("DR_GRID_OK", args.tag, flush=True)
    h.close()


if __name__ == "__main__":
    main()
