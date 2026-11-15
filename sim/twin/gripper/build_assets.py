"""Build the twin's fin-ray gripper geometry from the open XLeRobot CAD (plain python: numpy, scipy, trimesh).

Source: Vector-Wangel/XLeRobot `hardware/SO101_soft_fin.stl` (Apache-2.0, see assets/LICENSE and assets/SOURCE.md),
the "Compliant Gripper (new)" linked from TheRobotStudio/SO-ARM100: one print plate with three bodies:
  * wrist-roll-follower base (rigid, replaces the stock follower incl. its fixed finger; clevis with 2 x M3),
  * moving-jaw base (rigid, servo-horn hub + clevis with 2 x M3),
  * the TPU fin (67 mm wedge, flat contact face, curved back, slanted ribs); printed twice, one per base.
Placement on the robot is derived, not eyeballed:
  1. each base is registered to the stock URDF mesh it replaces (same CAD axes -> axis-aligned rotation; translation
     from the shared servo body / horn hub; REG below, residuals in fit notes of the README);
  2. the fin goes where its two M3 holes meet the base's clevis holes (hole axis, hole pair and clevis mid-plane fix
     all 6 DOF; of the 4 hole-order / axis-sign solutions exactly one has the tip distal and the flat face inward).
Output `assets/finray_so101.npz` (metres): visual meshes and collider hulls of both fingers in their link frames
(gripper_link, moving_jaw_so101_v1_link), and `assets/finray_so101.json` (tips, contact faces, gap vs joint angle).

    python sim/twin/gripper/build_assets.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
SRC = ASSETS / "SO101_soft_fin.stl"
# URDF (so101_new_calib.urdf): visual origins of the stock meshes in their links, and the gripper joint
FOLLOWER_ORIGIN = np.array([0.0, -0.000218214, 0.000949706])  # rpy (-pi, 0, 0) in gripper_link
JAW_ORIGIN = np.array([0.0, 0.0, 0.0189])  # rpy 0 in moving_jaw_so101_v1_link
JOINT_XYZ = np.array([0.0202, 0.0188, -0.0234])  # gripper joint in gripper_link, rpy (pi/2, 0, 0), axis z
# registration print-plate coords (mm) -> stock mesh coords (mm): p_mesh = R p + t
REG = {"fixed": (np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0.0]]), np.array([-35.2, 79.01, 200.85])),
       "jaw": (np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1.0]]), np.array([127.33, 64.635, 24.0]))}
# clevis M3 holes of the bases (print-plate mm; hole pair at the clevis mid-plane, hole axis) and of the fin
BASE_HOLES = {"fixed": (np.array([[80.08, 157.901, 17.44], [80.08, 157.029, 7.478]]), np.array([1.0, 0, 0])),
              "jaw": (np.array([[86.55, 133.504, 24.78], [87.073, 123.518, 24.78]]), np.array([0, 0, -1.0]))}
FIN_HOLES = np.array([[114.02, 99.522, 11.824], [114.02, 89.522, 11.824]])
FIN_HOLE_AXIS = np.array([-0.101, 0.0, 0.995])
FIN_FACE_Y = 103.52  # the flat contact face of the fin (print-plate mm), normal +y
N_SEG = 4  # convex collider pieces along the fin


def rot_x(a: float) -> np.ndarray:
    """Rotation matrix about x."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_z(a: float) -> np.ndarray:
    """Rotation matrix about z."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def jaw_to_grip(p: np.ndarray, q: float) -> np.ndarray:
    """Points in the moving-jaw link frame -> gripper_link frame at gripper joint angle q (URDF kinematics)."""
    return p @ (rot_x(np.pi / 2) @ rot_z(q)).T + JOINT_XYZ


def _frame(p1, p2, axis):
    o = (p1 + p2) / 2
    u = (p1 - p2) / np.linalg.norm(p1 - p2)
    axis = axis - (axis @ u) * u
    axis = axis / np.linalg.norm(axis)
    return o, np.stack([u, axis, np.cross(u, axis)], 1)


def _to_link(which: str, p_mm: np.ndarray) -> np.ndarray:
    R, t = REG[which]
    m = (p_mm @ R.T + t) / 1000.0
    return m @ rot_x(np.pi).T + FOLLOWER_ORIGIN if which == "fixed" else m + JAW_ORIGIN


def _fin_to_plate(which: str):
    """Rigid transform taking the fin (print-plate coords) onto the base's clevis (print-plate coords)."""
    of, Ff = _frame(FIN_HOLES[0], FIN_HOLES[1], FIN_HOLE_AXIS)
    holes, axis = BASE_HOLES[which]
    ob, Fb = _frame(holes[0], holes[1], axis)
    R = Fb @ Ff.T
    return R, ob - R @ of


def _slab_points(mesh, x0: float, x1: float) -> np.ndarray:
    """Vertices of `mesh` with x0 <= x <= x1 plus the points where its edges cross the two cut planes."""
    v = mesh.vertices
    pts = [v[(v[:, 0] >= x0) & (v[:, 0] <= x1)]]
    a, b = v[mesh.edges_unique[:, 0]], v[mesh.edges_unique[:, 1]]
    for x in (x0, x1):
        cross = (a[:, 0] - x) * (b[:, 0] - x) < 0
        w = ((x - a[cross, 0]) / (b[cross, 0] - a[cross, 0]))[:, None]
        pts.append(a[cross] + w * (b[cross] - a[cross]))
    return np.concatenate(pts)


def split_parts() -> dict:
    """The three bodies of the print plate: {'fixed': base, 'jaw': base, 'fin': fin} (trimesh, mm)."""
    parts = sorted(trimesh.load(SRC, force="mesh").split(only_watertight=False), key=lambda p: -abs(p.volume))
    if len(parts) != 3:
        raise ValueError(f"expected 3 bodies in {SRC.name}, found {len(parts)}")
    return {"fixed": parts[0], "fin": parts[1], "jaw": parts[2]}


def build() -> dict:
    """Write assets/finray_so101.{npz,json}; returns the json summary."""
    parts = split_parts()
    fin = parts["fin"]
    arrays, summ = {}, {"source": SRC.name, "units": "m", "frames": {"fixed": "gripper_link",
                                                                    "jaw": "moving_jaw_so101_v1_link"}}
    x_lo, x_hi = fin.vertices[:, 0].min(), fin.vertices[:, 0].max()
    edges = np.linspace(x_lo, x_hi, N_SEG + 1)
    for which in ("fixed", "jaw"):
        R, t = _fin_to_plate(which)
        base_v = _to_link(which, parts[which].vertices)
        fin_v = _to_link(which, fin.vertices @ R.T + t)
        arrays[f"{which}_base_v"], arrays[f"{which}_base_f"] = base_v.astype(np.float32), parts[which].faces
        arrays[f"{which}_fin_v"], arrays[f"{which}_fin_f"] = fin_v.astype(np.float32), fin.faces
        for k in range(N_SEG):  # convex pieces: hull of the fin slab between two stations
            hull = trimesh.convex.convex_hull(_slab_points(fin, edges[k], edges[k + 1]))
            arrays[f"{which}_hull{k}_v"] = _to_link(which, hull.vertices @ R.T + t).astype(np.float32)
            arrays[f"{which}_hull{k}_f"] = hull.faces
        tip_mm = np.array([x_lo, FIN_FACE_Y, 5.5])
        face = np.array([[x_lo, FIN_FACE_Y, 0.0], [x_lo + 55.0, FIN_FACE_Y, 0.0], [x_lo + 55.0, FIN_FACE_Y, 12.0],
                         [x_lo + 10.0, FIN_FACE_Y, 10.0]])
        arrays[f"{which}_tip"] = _to_link(which, (tip_mm @ R.T + t)[None])[0]
        arrays[f"{which}_face"] = _to_link(which, face @ R.T + t)
        summ[which] = {"tip": arrays[f"{which}_tip"].round(5).tolist(),
                       "fin_volume_mm3": round(float(fin.volume), 1),
                       "base_bounds": [base_v.min(0).round(4).tolist(), base_v.max(0).round(4).tolist()]}
    # geometry in the gripper frame vs joint angle
    fx = arrays["fixed_face"]
    summ["fixed_tip_grip"] = arrays["fixed_tip"].round(5).tolist()
    kf = np.polyfit(fx[:, 2], fx[:, 0], 1)  # fixed contact face line x(z) in the gripper x-z plane
    summ["fixed_face_toe_in_deg"] = round(float(np.degrees(np.arctan(kf[0]))), 2)
    rows = []
    for q in np.round(np.arange(-0.2, 1.21, 0.02), 3):
        jf = jaw_to_grip(arrays["jaw_face"], q)
        tipj = jaw_to_grip(arrays["jaw_tip"][None], q)[0]
        k = np.polyfit(jf[:, 2], jf[:, 0], 1)
        z_fix_tip = float(arrays["fixed_tip"][2])

        def gap(dz, k=k, z_fix_tip=z_fix_tip):
            z = z_fix_tip + dz
            return round(float(np.polyval(k, z) - np.polyval(kf, z)) * 1e3, 2)

        rows.append({"q": float(q), "jaw_tip": tipj.round(5).tolist(),
                     "gap_tip_mm": gap(0.0), "gap_15mm_up_mm": gap(0.015), "gap_30mm_up_mm": gap(0.030)})
    summ["gap_vs_q"] = rows
    qs = np.array([r["q"] for r in rows])
    for key in ("gap_tip_mm", "gap_15mm_up_mm", "gap_30mm_up_mm"):
        g = np.array([r[key] for r in rows])
        summ[f"q_at_31.7mm_{key}"] = round(float(np.interp(31.7, g, qs)), 4)
        summ[f"q_at_0mm_{key}"] = round(float(np.interp(0.0, g, qs)), 4)
    np.savez_compressed(ASSETS / "finray_so101.npz", **arrays)
    (ASSETS / "finray_so101.json").write_text(json.dumps(summ, indent=1))
    return summ


if __name__ == "__main__":
    s = build()
    print(json.dumps({k: v for k, v in s.items() if k != "gap_vs_q"}, indent=1))
    for r in s["gap_vs_q"][::4]:
        print(r)
