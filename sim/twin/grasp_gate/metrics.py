"""Outcome of each replayed demo from the gate's per-tick trace (numpy only, also usable offline on trace.npz).

Trace arrays (n demos, T ticks): q measured joints (ours), bp brick position (env frame), bq brick quaternion xyzw,
rel brick centre in the gripper_link frame, tip grasp point, fin fixed-finger tip, goal; per demo: start, cf, end
(dataset frames), b0 brick start position, s_real recorded states.

Outcome (first that applies): success (>= 3 cm above the start and within 4 cm of the grasp point for >= 15 ticks),
tipped (tilt > 35 deg while < 1 cm up), slipped (was >= 1 cm up at the grasp point, then lost), pushed (moved
> 1.5 cm on the table), missed.
"""
from __future__ import annotations

import numpy as np

from sim.twin.grasp_gate.common import wilson

FAIL = ("missed", "pushed", "slipped", "tipped")
J = ("pan", "lift", "elbow", "wflex", "wroll", "grip")


def _run(mask: np.ndarray) -> int:
    """Longest run of True."""
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def analyse(tr: dict, lift_m: float = 0.03, hold: int = 15) -> list:
    """Per-demo outcome and diagnostics (lengths in mm)."""
    out = []
    table = float(tr["table_z"])
    for k, ep in enumerate(tr["episodes"].tolist()):
        n = int(tr["end"][k] - tr["start"][k]) + 1
        ic = int(tr["cf"][k] - tr["start"][k])
        bp, rel, tip, fin = tr["bp"][k, :n], tr["rel"][k, :n], tr["tip"][k, :n], tr["fin"][k, :n]
        x, y, z, w = tr["bq"][k, :n].T
        zax = np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], 1)  # body z in world
        tilt = np.degrees(np.arccos(np.clip(zax @ zax[0], -1, 1)))  # tilt from the start pose (labelled poses)
        h = bp[:, 2] - tr["b0"][k, 2]
        near = np.linalg.norm(bp - tip, axis=1) < 0.04
        lifted = (h > lift_m) & near
        succ = _run(lifted) >= hold
        disp = np.linalg.norm(bp[:, :2] - tr["b0"][k, :2], axis=1)
        tipped = bool(((tilt > 35) & (h < 0.01)).any())
        ever = bool(((h > 0.01) & near).any())
        pushed = bool((disp[: ic + 20] > 0.015).any())
        oc = "success" if succ else "tipped" if tipped else "slipped" if ever else "pushed" if pushed else "missed"
        # centred: brick centre within 5 mm of the finger mid-plane (gripper y) and level, most of the lifted time
        cen = bool(succ and (((np.abs(rel[:, 1]) < 0.005) & (tilt < 15))[lifted].mean() >= 0.5))
        err = tr["q"][k, :n] - tr["s_real"][k, :n]
        a, b = max(ic - 10, 0), min(ic + 5, n - 1)
        d = {"episode": ep, "outcome": oc, "success": bool(succ), "centred": cen, "kind": int(tr["kind"][k]),
             "close_tick": ic,
             "rel_close_mm": np.round(rel[ic] * 1e3, 1).tolist(),  # brick centre in gripper_link at the close frame
             "rel_pre_mm": np.round(rel[max(ic - 15, 0)] * 1e3, 1).tolist(),
             "fin_tip_above_table_mm": round(float(fin[ic, 2] - table) * 1e3, 1),
             "fin_tip_min_above_table_mm": round(float(fin[a:b + 1, 2].min() - table) * 1e3, 1),
             "jaw_tip_min_above_table_mm": (round(float(tr["jfin"][k, a:b + 1, 2].min() - table) * 1e3, 1)
                                            if "jfin" in tr else None),
             "grip_lag_close3": round(float(tr["q"][k, min(ic + 3, n - 1), 5] - tr["s_real"][k, min(ic + 3, n - 1), 5]),
                                      1),  # sim - real gripper reading 3 ticks after the close frame
             "brick_below_tip_mm": round(float(tip[ic, 2] - bp[ic, 2]) * 1e3, 1),
             "disp_pre_close_mm": round(float(disp[max(ic - 3, 0)]) * 1e3, 1),
             "tilt_close_deg": round(float(tilt[ic]), 1), "max_height_mm": round(float(h.max()) * 1e3, 1),
             "q_err_close_deg": np.round(err[ic], 2).tolist(),
             "rmse_deg": np.round(np.sqrt((err ** 2).mean(0)), 2).tolist(),
             "grip_held": round(float(np.median(tr["q"][k, min(ic + 15, n - 1):n, 5])), 2),
             "grip_held_real": round(float(np.median(tr["s_real"][k, min(ic + 15, n - 1):n, 5])), 2)}
        out.append(d)
    return out


def summarise(per: list) -> dict:
    """Success with Wilson CI, centred rate, failure breakdown, medians of the diagnostics."""
    n = len(per)
    ns = sum(d["success"] for d in per)
    nc = sum(d["centred"] for d in per)

    def med(key, sel=None):
        v = [d[key] for d in per if sel is None or d["success"] == sel]
        return None if not v else np.round(np.median(np.array(v, float), 0), 1).tolist()

    res = {"n": n, "success": round(ns / max(n, 1), 4), "success_ci95": wilson(ns, n),
           "centred": round(nc / max(n, 1), 4), "centred_ci95": wilson(nc, n),
           "failures": {f: round(sum(d["outcome"] == f for d in per) / max(n, 1), 4) for f in FAIL},
           "fin_tip_above_table_mm": {"success": med("fin_tip_above_table_mm", True),
                                      "fail": med("fin_tip_above_table_mm", False)},
           "rel_close_mm": {"success": med("rel_close_mm", True), "fail": med("rel_close_mm", False)},
           "jaw_tip_min_above_table_mm": {"success": med("jaw_tip_min_above_table_mm", True),
                                          "fail": med("jaw_tip_min_above_table_mm", False)},
           "grip_lag_close3": {"success": med("grip_lag_close3", True), "fail": med("grip_lag_close3", False)},
           "disp_pre_close_mm": {"success": med("disp_pre_close_mm", True), "fail": med("disp_pre_close_mm", False)},
           "grip_held": {"sim_success": med("grip_held", True), "real": med("grip_held_real")},
           "rmse_deg": dict(zip(J, med("rmse_deg"), strict=True)),
           "q_err_close_deg": dict(zip(J, med("q_err_close_deg"), strict=True))}
    for kind, name in ((0, "2x2"), (1, "2x4"), (2, "curved")):
        sel = [d for d in per if d["kind"] == kind]
        if sel:
            res[f"success_{name}"] = [round(sum(d["success"] for d in sel) / len(sel), 3), len(sel)]
    return res
