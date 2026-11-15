"""Top-camera contrast / saturation, real vs sim, split into lights and camera response (plain Python, no Isaac).

    python sim/twin/lighting/contrast_check.py [--tag v3]

Uses validate_<tag>.npz (sim renders at held-out states; top had no camera response then, i.e. raw renders) and
applies the CURRENT twin_scene.color_correct('top') on top, so the haze can be attributed: rig (raw old vs raw new) vs
camera model (raw new vs corrected new). Region masks from visual_fit/render_before.npz. Prints metrics.contrast.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from sim.twin.lighting.metrics import contrast  # noqa: E402
from sim.twin.visuals.twin_scene import color_correct  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="v3")
args = ap.parse_args()
z = np.load(REPO / f"sim/twin/out/lighting/validate_{args.tag}.npz")
ids = z["ids"]
real = np.load(REPO / "sim/twin/out/visual_fit/evalset.npz")["top"][ids]
masks = np.load(REPO / "sim/twin/out/visual_fit/render_before.npz")["mask_top"][ids]
rows = {"real": real, "sim old rig (raw)": z["old_top"], "sim new rig (raw)": z["new_top"],
        "sim new rig + current color_correct": np.stack([color_correct(x, "top") for x in z["new_top"]])}
res = {k: contrast(v, masks) for k, v in rows.items()}
print(json.dumps(res, indent=1))
(REPO / f"sim/twin/out/lighting/contrast_{args.tag}.json").write_text(json.dumps(res, indent=1))
