"""Render the whole evaluation set with the twin's current default look (+ region masks) and write the metrics.

    sim/twin/run.sh sim/twin/visuals/fit/render_eval.py --tag before [--sheet 6] [--passes 2]

Writes sim/twin/out/visual_fit/render_<tag>.npz (raw renders + masks), metrics_<tag>.json (camera response applied) and
sheet_<tag>_<cam>.jpg (real | sim pairs of held-out samples).
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--passes", type=int, default=2)
ap.add_argument("--sheet", type=int, default=6)
ap.add_argument("--masks-from", default=None, help="reuse the masks of render_<tag>.npz instead of re-rendering them")
args = ap.parse_args()

from sim.twin.visuals.fit.harness import OUT, Harness  # noqa: E402

h = Harness(passes=args.passes)
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from sim.twin.visuals.fit import eval_visuals as ev  # noqa: E402

t0 = time.time()
out = h.render_set(masks=args.masks_from is None)
if args.masks_from:
    z = np.load(OUT / f"render_{args.masks_from}.npz")
    out.update({k: z[k] for k in z.files if k.startswith("mask_")})
print(f"rendered {len(h.samples)} samples in {time.time() - t0:.0f} s", flush=True)
np.savez_compressed(OUT / f"render_{args.tag}.npz", **out)
samples, real = ev.load_eval()
sim = ev.apply_ccm({c: out[c] for c in ev.CAMS})
res = ev.evaluate(real, sim, {c: out["mask_" + c] for c in ev.CAMS}, samples)
(OUT / f"metrics_{args.tag}.json").write_text(json.dumps(res, indent=1))
for split in ("fit", "held"):
    print(split, json.dumps(ev.summary_row(args.tag, res, split)), flush=True)
held = [i for i, s in enumerate(samples) if s["split"] == "held"]
pick = held[:: max(1, len(held) // (args.sheet * 5))][: args.sheet * 5]
for c in ev.CAMS:
    rows = []
    for k in range(0, len(pick), 5):
        row = []
        for i in pick[k:k + 5]:
            pair = np.concatenate([real[c][i], sim[c][i]], 0)
            txt = f"ep{samples[i]['ep']} {samples[i]['moment']}"
            cv2.putText(pair, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            row.append(cv2.resize(pair, (320, 480), interpolation=cv2.INTER_AREA))
        rows.append(np.concatenate(row, 1))
    sheet = np.concatenate(rows, 0)[..., ::-1]
    cv2.imwrite(str(OUT / f"sheet_{args.tag}_{c}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
print("RENDER_EVAL_OK", args.tag, flush=True)
h.close()
