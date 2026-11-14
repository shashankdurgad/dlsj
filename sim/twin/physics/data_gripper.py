"""Real gripper readings in the cal2 dataset: closed-on-nothing, hold, open (plain python, no Isaac)."""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[3]
root = REPO / "sim/twin/data/canonical/duplo_teleop_cal2_rgb_456"
fs = sorted(glob.glob(str(root / "data/*/*.parquet")))
t = pa.concat_tables([pq.read_table(f, columns=["observation.state", "action", "episode_index"]) for f in fs])
s = np.stack(t.column("observation.state").to_numpy(zero_copy_only=False))
a = np.stack(t.column("action").to_numpy(zero_copy_only=False))
ep = np.array(t.column("episode_index").to_pylist())
P = [0, 0.1, 1, 5, 25, 50, 75, 95, 99, 100]
print("frames", len(s), "episodes", len(np.unique(ep)))
print("grip state pct", np.percentile(s[:, 5], P).round(2).tolist())
print("grip action pct", np.percentile(a[:, 5], P).round(2).tolist())
mins = np.array([s[ep == e, 5].min() for e in np.unique(ep)])
amins = np.array([a[ep == e, 5].min() for e in np.unique(ep)])
print("per-ep min state pct", np.percentile(mins, [0, 5, 25, 50, 75, 95, 100]).round(2).tolist())
print("per-ep min action pct", np.percentile(amins, [0, 5, 25, 50, 75, 95, 100]).round(2).tolist())
# frames where the command is very low (operator squeezes) -> measured
low = a[:, 5] < 1.0
print("state where action<1: n", int(low.sum()), np.percentile(s[low, 5], [5, 50, 95]).round(2).tolist() if low.any() else None)
print("lowest-state episodes", np.argsort(mins)[:8].tolist(), np.sort(mins)[:8].round(2).tolist())
