"""Check the numpy FK against Isaac: export_points.py's points.npz (Isaac link poses x USD mesh points).

(1) absolute: every Isaac point vs the nearest full-resolution URDF mesh vertex placed by our FK (Isaac samples USD
    mesh vertices, which are the URDF STL vertices) -> should be ~0;
(2) relative: points expressed in the link frame from the first sample, re-posed by our FK at every other sample.
Samples where a commanded joint is beyond the URDF limit are reported separately: Isaac stops those joints at the
(soft) limit, our FK does not; for them we also fit the joint angles Isaac actually reached.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fk import JOINTS, SO101  # noqa: E402

D = np.load(HERE.parents[1] / "out/align/points.npz", allow_pickle=True)
names, link, pts, Q = list(D["link_names"]), D["link"], D["pts"].astype(float), D["q"]
arm = SO101()
lo, hi = arm.limits[:, 0], arm.limits[:, 1]
print("profile", D["profile"], "| samples", len(Q))
print("URDF limits (deg):", {j: np.round(np.degrees(arm.limits[i]), 1).tolist() for i, j in enumerate(JOINTS)})
out = (Q < lo - 1e-4) | (Q > hi + 1e-4)
print("samples beyond a limit:", int(out.any(1).sum()), "/", len(Q), "| per joint",
      dict(zip(JOINTS, out.sum(0).tolist(), strict=True)))
T0 = arm.fk(Q[0])
loc = {n: (pts[0, link == i] - T0[n][:3, 3]) @ T0[n][:3, :3] for i, n in enumerate(names)}


def err(q, f):
    T = arm.fk(q)
    a, r = [], []
    for i, n in enumerate(names):
        P = pts[f, link == i]
        V = arm.meshes_full[n] @ T[n][:3, :3].T + T[n][:3, 3]
        a.append(cKDTree(V).query(P)[0].max())
        r.append(np.linalg.norm(loc[n] @ T[n][:3, :3].T + T[n][:3, 3] - P, axis=1).max())
    return np.array(a), np.array(r)


def resid(q, f):
    """Link points posed by FK at q minus Isaac's points at sample f (every 15th point)."""
    T = arm.fk(q)
    return np.concatenate([(loc[n] @ T[n][:3, :3].T + T[n][:3, 3] - pts[f, link == i])[::15].ravel()
                           for i, n in enumerate(names)])


A_in, A_out, A_fit, dq = [], [], [], []
for f in range(len(Q)):
    if out[f].any():
        A_out.append(err(Q[f], f)[0])
        r = least_squares(resid, np.clip(Q[f], lo, hi), args=(f,))
        A_fit.append(err(r.x, f)[0])
        dq.append(np.degrees(r.x - Q[f]))
    elif f % 3 == 0:
        A_in.append(np.maximum(*err(Q[f], f)))
A_in, A_out, A_fit = map(np.array, (A_in, A_out, A_fit))
print(f"\nwithin limits ({len(A_in)} samples), max error per link (mm):")
for i, n in enumerate(names):
    print(f"  {n:28s} {1e3 * A_in[:, i].max():7.3f}")
print(f"beyond limits ({len(A_out)} samples): commanded q -> max {1e3 * A_out.max():.2f} mm; "
      f"with the q Isaac reached (fitted) -> max {1e3 * A_fit.max():.3f} mm")
dq = np.array(dq)
print("  Isaac minus commanded at those samples (deg), min / max per joint:",
      {j: [round(dq[:, i].min(), 2), round(dq[:, i].max(), 2)] for i, j in enumerate(JOINTS)})
ok = A_in.max() < 1e-3 and A_fit.max() < 1e-3
print(f"FK_VERIFY {'OK' if ok else 'FAIL'}: within-limit worst {1e3 * A_in.max():.3f} mm")
