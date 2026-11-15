"""Gripper-close forensics shared by the real-trace and the twin-rollout analyses (plain numpy, no Isaac).

A *close* is one closing cycle of the measured gripper: it starts when the reading drops below CLOSE_BELOW after
having been open (>= OPEN_ABOVE) and ends when it re-opens (>= OPEN_ABOVE) or the series ends. Each close is labelled

  empty : settled reading < EMPTY_BELOW (the real jaws read ~3.3-3.7 when shut on nothing, 4.5-6.1 on a Duplo)
  brick : settled reading >= EMPTY_BELOW (jaws blocked by something; the brick or, rarely, a neighbour)

and carries the fingertip geometry at the moment the jaws shut (forward kinematics of the measured joints with a
joint-offset set, sim/twin/align/joint_fit): the lowest CAD fin-tip height above the table, the TIP_LOCAL grasp
point (gripper_link (8, 0, -90) mm) position, and the gripper's tilt from vertical.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

TWIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TWIN / "align/joint_fit"))
sys.path.insert(0, str(TWIN.parents[1]))

OPEN_ABOVE = 25.0
CLOSE_BELOW = 15.0
EMPTY_BELOW = 4.2
TABLE_Z = -0.002  # table top in the robot-base frame (sim/residual_env/env.py)
TIP_LOCAL = np.array([0.008, 0.0, -0.090])
GRIP_SCALE, GRIP_BIAS = 0.0230767, -0.242  # joe_cal2 gripper units -> rad (sim/residual_env/calib.py)
FPS = 30.0


def offsets(name: str = "W9") -> list:
    R = json.loads((TWIN / "align/joint_fit/result.json").read_text())
    if name == "W":
        return R["joint_offsets_rad_for_sim"]
    return R["grasp_depth_candidates"][name]["joint_offsets_rad_for_sim"]


class Geometry:
    """Fin-tip / grasp-point kinematics of the measured joints (LeRobot deg, gripper 0-100)."""

    def __init__(self, offs: str = "W9"):
        from fk import SO101, sim_q

        from sim.twin.gripper import model

        self.arm, self.sim_q, self.model = SO101(), sim_q, model
        self.off = offsets(offs)
        g = model.geometry()
        self.fix_tip, self.jaw_tip = g["fixed_tip"].astype(float), g["jaw_tip"].astype(float)

    def at(self, state) -> dict:
        q = self.sim_q(state, self.off, GRIP_SCALE, GRIP_BIAS)
        G = self.arm.fk(q)["gripper_link"]
        pf = G[:3, :3] @ self.fix_tip + G[:3, 3]
        pj = G[:3, :3] @ self.model.jaw_to_grip(self.jaw_tip, q[5]) + G[:3, 3]
        tip = G[:3, :3] @ TIP_LOCAL + G[:3, 3]
        tilt = float(np.degrees(np.arccos(np.clip(G[2, 2], -1, 1))))  # fingers (gripper -z) vs straight down
        return {"low_mm": (min(pf[2], pj[2]) - TABLE_Z) * 1e3, "tip": tip, "tip_z_mm": (tip[2] - TABLE_Z) * 1e3,
                "mid": 0.5 * (pf + pj), "tilt_deg": tilt}


def segment_closes(g: np.ndarray) -> list[tuple[int, int]]:
    """(i_start, i_end) of each closing cycle of the measured gripper series g."""
    out, opened, i0 = [], bool(len(g) and g[0] >= OPEN_ABOVE), None
    for i, v in enumerate(g):
        if i0 is None and opened and v < CLOSE_BELOW:
            i0, opened = i, False
        elif v >= OPEN_ABOVE:
            if i0 is not None:
                out.append((i0, i))
                i0 = None
            opened = True
    if i0 is not None:
        out.append((i0, len(g)))
    return out


def describe_close(M: np.ndarray, C: np.ndarray | None, i0: int, i1: int, geo: Geometry | None) -> dict:
    """M, C: (T, 6) measured / commanded joints (deg, gripper 0-100); [i0, i1) one close from segment_closes."""
    g = M[:, 5]
    # the jaws' shut instant: first tick after i0 within 0.3 units of the segment's settled (median of the next 0.5 s)
    seg = g[i0:i1]
    settle_win = seg[min(len(seg) - 1, 6): min(len(seg), 21)]
    settled = float(np.median(settle_win if len(settle_win) else seg))
    shut = i0 + int(np.argmax(seg <= settled + 0.3)) if (seg <= settled + 0.3).any() else i0
    # command onset: last tick before i0 at which the commanded gripper was still >= 25 (policy decided to close)
    onset = None
    if C is not None:
        k = i0
        while k > 0 and C[k, 5] < OPEN_ABOVE:
            k -= 1
        onset = k + 1
    d = {"i0": i0, "i1": i1, "shut": shut, "onset": onset, "settled": round(settled, 2),
         "empty": settled < EMPTY_BELOW, "held_ticks": i1 - i0}
    if geo is not None:
        a = geo.at(M[shut])
        lo = min(geo.at(M[k])["low_mm"] for k in range(max(0, i0 - 3), min(len(M), shut + 4)))
        d.update(low_mm=round(a["low_mm"], 1), low_min_mm=round(lo, 1), tip_z_mm=round(a["tip_z_mm"], 1),
                 tip_xy_mm=[round(float(v) * 1e3, 1) for v in a["tip"][:2]], tilt_deg=round(a["tilt_deg"], 1))
        # lift after the close: tip z rise over the closed segment (brick lifted if held >= 15 ticks at +30 mm)
        z = np.array([geo.at(M[k])["tip_z_mm"] for k in range(shut, i1)])
        d["max_rise_mm"] = round(float(z.max() - a["tip_z_mm"]), 1) if len(z) else 0.0
        if not d["empty"] and len(z):
            up = z - a["tip_z_mm"] >= 30.0
            run = best = 0
            for u in up:
                run = run + 1 if u else 0
                best = max(best, run)
            d["lifted_30mm_0p5s"] = best >= 15
        else:
            d["lifted_30mm_0p5s"] = False
    return d


def t_reach(M: np.ndarray, geo: Geometry, z_mm: float = 30.0, fps: float = FPS) -> float | None:
    """Seconds until the grasp point (TIP_LOCAL) first comes within z_mm of the table (the approach's arrival)."""
    for k in range(0, len(M), 2):
        if geo.at(M[k])["tip_z_mm"] < z_mm:
            return round(k / fps, 2)
    return None
