"""How often recorded cal2 states (joe_cal2 offsets) fall outside the URDF / USD joint limits (Isaac clamps there)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import model as M
from fk import JOINTS, SO101
from frames import frames_table

t = frames_table()
S = np.stack(t["observation.state"].to_numpy()).astype(float)
lim = SO101().limits
for name, off in (("joe_cal2", M.JOE_OFFSETS),):
    q = np.deg2rad(S[:, :5]) + off
    print(name, "frames", len(S))
    for j in range(5):
        lo, hi = lim[j]
        b = (q[:, j] < lo) | (q[:, j] > hi)
        print(f"  {JOINTS[j]:14s} sim q range [{np.degrees(q[:, j].min()):7.1f}, {np.degrees(q[:, j].max()):7.1f}] deg "
              f"| limits [{np.degrees(lo):6.1f}, {np.degrees(hi):6.1f}] | beyond {100 * b.mean():5.2f}% of frames "
              f"(max excess {np.degrees(max(lo - q[:, j].min(), q[:, j].max() - hi, 0)):.1f} deg)")
