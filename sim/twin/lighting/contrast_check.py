"""Top-camera contrast / saturation of real vs sim (old / new rig) at held-out states (plain Python, no Isaac).

    python sim/twin/lighting/contrast_check.py [--tag v4] [--recorrect]

Reads validate_<tag>.npz (renders after the twin_scene.color_correct of that run) and prints metrics.contrast for the
real frames and both rigs (table / plate luma mean, luma std, Lab chroma). --recorrect also applies the CURRENT
color_correct('top') to the new rig's frames (to separate the camera model from the lights; use it on a run whose top
camera had no response model). Writes sim/twin/out/lighting/contrast_<tag>.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from sim.twin.lighting.metrics import contrast  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="v4")
ap.add_argument("--recorrect", action="store_true")
args = ap.parse_args()
z = np.load(REPO / f"sim/twin/out/lighting/validate_{args.tag}.npz")
ids = z["ids"]
real = np.load(REPO / "sim/twin/out/visual_fit/evalset.npz")["top"][ids]
masks = np.load(REPO / "sim/twin/out/visual_fit/render_before.npz")["mask_top"][ids]
rows = {"real": real, "sim old rig": z["old_top"], "sim new rig": z["new_top"]}
if args.recorrect:
    from sim.twin.visuals.twin_scene import color_correct

    rows["sim new rig + current color_correct"] = np.stack([color_correct(x, "top") for x in z["new_top"]])
res = {k: contrast(v, masks) for k, v in rows.items()}
print(json.dumps(res), flush=True)
(REPO / f"sim/twin/out/lighting/contrast_{args.tag}.json").write_text(json.dumps(res, indent=1))
