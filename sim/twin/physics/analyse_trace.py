"""Per-episode grasp diagnosis from a realism_gate --trace run: tip height at close, brick tilt while held, contact
angle (gripper q), brick depth between the fingers. Plain python (numpy)."""
import json
import sys
from pathlib import Path

import numpy as np

d = Path(sys.argv[1])
tr = np.load(d / "trace.npy")  # (T, n, 10): q, r, rel xyz, tau, target, tilt_deg, brick_z, tip_z
rt = np.load(d / "replay_traces.npz")
rep = json.loads((d / "validate.json").read_text())
tips = rep["replay_tip_height_at_grasp_m"]
zmax = rep["replay"]["brick_max_height_m"]
real = rt["real"]
print("ep  tip@close  lifted  tilt_held  q_held  relz_held  hold_sim  hold_real")
for k, e in enumerate(rt["episodes"].tolist()):
    g = real[k, :, 5]
    L = int(np.isfinite(g).sum())
    T = min(L, len(tr))
    held = tr[:T, k, 8] > 0.035  # brick centre > 3.5 cm above the env origin
    if held.any():
        w = np.nonzero(held)[0]
        x = tr[w, k]
        print(f"{e:4d} {tips[k]:.3f}  {zmax[k]:.3f}  {np.median(x[:, 7]):6.1f}  {np.median(x[:, 0]):.3f}  "
              f"{np.median(x[:, 4]):.4f}  {np.median(rt['sim'][k, w, 5]):.2f}  {np.median(g[w]):.2f}")
    else:
        print(f"{e:4d} {tips[k]:.3f}  {zmax[k]:.3f}   not lifted")
