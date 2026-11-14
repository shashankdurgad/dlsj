"""Score cameras on the held-out recorded episodes with fit_cameras' silhouette metric (inspect camera):
joe_cal2 as-is, the silhouette fit (cameras_fit.json) and the ChArUco measurement (calib/cameras_measured.json)."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit_cameras as fc  # noqa: E402
from frames import states, video_frames  # noqa: E402
import cv2  # noqa: E402

TWIN = HERE.parent
D = np.load(TWIN / "out/align/points.npz", allow_pickle=True)
names = list(D["link_names"])
keep = np.isin(D["link"], [names.index(n) for n in fc.MOVING])
pts, link = D["pts"][:, keep].astype(np.float64), D["link"][keep]
samples = [tuple(map(int, s)) for s in D["samples"]]
n_top = json.loads((TWIN / "out/align/samples_split.json").read_text())["n_top"]
eps = sorted({e for e, _ in samples}); held = set(eps[2::3])
idx = [f for f in range(n_top, len(samples)) if samples[f][0] in held]
items, rgb = [], {}
for e in sorted({samples[f][0] for f in idx}):
    fs = [f for f in idx if samples[f][0] == e]
    n = len(states(e)); bgi = sorted(set(np.linspace(0, n - 1, 25).astype(int).tolist()))
    vf = video_frames(e, "inspect", sorted(set(bgi) | {samples[f][1] for f in fs}))
    bg = np.median(np.stack([vf[i] for i in bgi if i in vf]), 0).astype(np.uint8)
    for f in fs:
        if samples[f][1] in vf:
            rm = fc.real_mask(vf[samples[f][1]], bg)
            items.append((f, rm, cv2.distanceTransform((1 - rm).astype(np.uint8), cv2.DIST_L2, 3)))
            rgb[f] = vf[samples[f][1]]
P = fc.Problem("inspect", items, pts, link)
fit = json.loads((TWIN / "calib/cameras_fit.json").read_text())["inspect"]
mea = json.loads((TWIN / "calib/cameras_measured.json").read_text())["inspect"]
cams = {"joe_cal2": fc.load_joe("inspect"),
        "silhouette fit": {"R": np.array(fit["world_to_cv"]), "t": np.array(fit["position"]), **{k: fit[k] for k in ("fx", "fy", "cx", "cy")}},
        "ChArUco measured": {"R": np.array(mea["world_to_cv"]), "t": np.array(mea["position"]), **{k: mea[k] for k in ("fx", "fy", "cx", "cy")}}}
for k, c in cams.items():
    l, iou, n = P.loss(c, True)
    print(f"{k:18s} held-out inspect frames {n}: loss {l:6.2f}  IoU {iou:.3f}", flush=True)
tiles = []
for f, rm, _ in items[:: max(1, len(items) // 8)][:8]:
    img = rgb[f].copy()
    for c, col in ((cams["joe_cal2"], (255, 60, 60)), (cams["ChArUco measured"], (60, 255, 60))):
        uv, ok = fc.project(pts[f], c)
        sm = cv2.resize(fc.sim_mask(uv, ok, link, (P.h, P.w)), (640, 480), interpolation=cv2.INTER_NEAREST)
        cs, _ = cv2.findContours(sm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, cs, -1, col, 2)
    tiles.append(cv2.resize(img, (320, 240)))
while len(tiles) % 4:
    tiles.append(np.zeros_like(tiles[0]))
grid = np.concatenate([np.concatenate(tiles[i:i + 4], 1) for i in range(0, len(tiles), 4)], 0)
cv2.imwrite(str(TWIN / "out/align/overlay_inspect_measured.jpg"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
print("EVAL_OK")
