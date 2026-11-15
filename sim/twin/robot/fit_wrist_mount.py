"""Fit the wrist camera's pose on the gripper (gripper_link frame) from the real evidence -> wrist_mount.json.

    python sim/twin/robot/real_wrist.py          # once: real wrist frames + states (PyAV) -> out/robot/real_wrist.npz
    python sim/twin/robot/fit_wrist_mount.py     # plain python (numpy / scipy / cv2), no Isaac, ~1 min

Measured intrinsics + distortion (calib/cameras_measured.json, fx 343 / cx 339). Evidence, all rigid in gripper_link
(independent of the arm's joint offsets):
1. fixed fin silhouette: the XLeRobot TPU fin-ray fixed finger (sim/twin/gripper/assets/finray_so101.npz, bolted to
   gripper_link) occupies the same pixels in every wrist frame; its persistent mask (red in >= 80 % of the
   real_wrist.npz frames with an unloaded fin) vs the model silhouette, symmetric boundary distance (chamfer, px).
   This fixes the roll that a far-field fit cannot separate from the wrist_roll joint offset (joe_cal2's pose is
   ~18 deg off in roll).
2. near field: a held 2x2 Duplo (31.8 mm, studs towards the wrist): its bottom-face edge and the 19.2 mm nearer
   top-face edge widths fix distance and tilt (pixels measured by hand, ep 18 frame 109 and ep 122 frame 100). Brick
   centred between the jaws (y = 0 +- 8 mm), against the fixed fin's contact face (x = face + 15.9 +- 5 mm), bottom
   at the fin tips (the inspect camera shows bricks held at the tips; fin tip z = -105 mm -> -98 +- 8 mm).
3. top camera: joe_cal2's wrist-housing landmark (reference-cal2.json housing_landmark_local, a point on the orange
   mount tracked in 12 top-camera frames, 6 px rms) = the centre of the frame's back face (+-8 mm, x +-10 mm).
4. phone photos: board centred over the jaws (x = 3 +- 6 mm), ~60 mm out on the +y side (y = 60 +- 10 mm).
joe_cal2's pose (fitted with f = 394 px and no principal-point offset) only seeds the optimiser.

The official camera URDF variant (assets/so101/urdf/so101_new_calib_camera.urdf, TheRobotStudio hex-nut bracket in
the same two M3 holes) has the camera ~4 cm higher, above the gripper servo; its V bracket is not the flat frame the
photos show. It is kept under "compare" only.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = HERE / "wrist_mount.json"
REAL = REPO / "sim/twin/out/robot/real_wrist.npz"
FIN = REPO / "sim/twin/gripper/assets/finray_so101.npz"
B, BH = 0.0318, 0.0192  # Duplo 2x2 width, body height (m)
# (episode, frame, bottom-edge centre u, bottom-edge v, bottom-edge width, top-edge v, top-edge width) in px
BRICKS = [(18, 109, 319, 262, 182, 330, 235), (122, 100, 332, 240, 175, 318, 205)]
BRICK_Y, BRICK_Z, BRICK_X_SD = (0.0, 0.008), (-0.098, 0.008), 0.005
PHOTO_XY, PHOTO_SD = np.array([0.003, 0.060]), np.array([0.006, 0.010])
LANDMARK_SD = np.array([0.010, 0.008, 0.008])
CHAMFER_SD = 2.0  # px
FIN_MIN_GRIP, FIN_FRACTION = 15.0, 0.8
LENS_BACK = 0.024  # optical centre -> PCB back (wrist_mount.LENS_Z)
MOUNT = dict(thickness=0.003, frame=0.046, window=0.026, neck_width=0.020, tab_length=0.014, board_hole=0.028,
             connector_offset=0.008, cable_to=[0.0, 0.026, 0.012])


def _rpy(xyz, rpy):
    m = np.eye(4)
    m[:3, :3] = Rot.from_euler("xyz", rpy).as_matrix()
    m[:3, 3] = xyz
    return m


def urdf_variant() -> tuple[np.ndarray, np.ndarray]:
    """Optical frame of so101_new_calib_camera.urdf (mount + camera fixed joints; lens centre 24 mm up the module)."""
    cam = _rpy([-0.0150, 0.0240, -0.0315], [-1.5708, -0.0008, -1.5708]) @ \
        _rpy([-0.04941, -0.03813, 0.0175], [0, -1.5708, -2.0076])
    return cam[:3, :3].T, cam[:3, :3] @ [0, 0, 0.024] + cam[:3, 3]


def _edge(m: np.ndarray) -> np.ndarray:
    e = m - cv2.erode(m, np.ones((3, 3), np.uint8))
    e[:3], e[-3:], e[:, :3], e[:, -3:] = 0, 0, 0, 0  # the image border cuts both silhouettes
    return e


def fin_mask(frames: np.ndarray, grip: np.ndarray) -> np.ndarray:
    """The fixed fin (rigid with the camera): pixels red (TPU) in >= FIN_FRACTION of the frames with an unloaded fin
    (gripper reading > FIN_MIN_GRIP: no brick bending or hiding it)."""
    sel = frames[grip > FIN_MIN_GRIP]
    n = np.zeros(frames.shape[1:3])
    for f in sel:
        h = cv2.cvtColor(f, cv2.COLOR_RGB2HSV)
        n += ((h[..., 0] < 6) | (h[..., 0] > 168)) & (h[..., 1] > 120) & (h[..., 2] > 90)
    m = (n >= FIN_FRACTION * len(sel)).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


class Camera:
    """Measured wrist intrinsics + distortion; projects gripper-frame points for a pose (R rows, centre t)."""

    def __init__(self, mea: dict):
        self.K = np.array([[mea["fx"], 0, mea["cx"]], [0, mea["fy"], mea["cy"]], [0, 0, 1.0]])
        self.dist = np.array(mea["dist"])

    def project(self, R, t, P):
        pc = (np.asarray(P) - t) @ R.T
        uv, _ = cv2.projectPoints(pc.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), self.K, self.dist)
        return uv.reshape(-1, 2), pc

    def silhouette(self, R, t, V, F) -> np.ndarray:
        uv, pc = self.project(R, t, V)
        ok = ((pc[:, 2] > 0.01) & (np.linalg.norm(pc[:, :2], axis=1) < 1.5 * np.maximum(pc[:, 2], 1e-6)))[F].all(1)
        m = np.zeros((480, 640), np.uint8)
        cv2.fillPoly(m, list(np.round(uv[F[ok]] * 8).astype(np.int32)), 1, shift=3)
        return m


def fit() -> dict:
    """Fit the physical camera (rotation, centre) + the two bricks' poses; returns the wrist_mount.json dict."""
    joe = json.loads((REPO / "sim/residual_env/assets/joe_cal2/reference-cal2.json").read_text())
    mea = json.loads((REPO / "sim/twin/calib/cameras_measured.json").read_text())["wrist"]
    Rj, tj, landmark = (np.array(joe["wrist_camera"]["local_to_cv"]), np.array(joe["wrist_camera"]["position_local"]),
                        np.array(joe["housing_landmark_local"]))
    cam = Camera(mea)
    fin = np.load(FIN)
    FV, FF = fin["fixed_fin_v"].astype(float), fin["fixed_fin_f"]
    kface = np.polyfit(fin["fixed_face"][:, 2], fin["fixed_face"][:, 0], 1)
    rw = np.load(REAL)
    real = fin_mask(rw["frames"], rw["states"][:, 5])
    real_dt = cv2.distanceTransform((1 - _edge(real)).astype(np.uint8), cv2.DIST_L2, 3)
    real_edge = _edge(real) > 0

    def chamfer(R, t, cap=40.0):
        pe = _edge(cam.silhouette(R, t, FV, FF))
        if pe.sum() < 20:
            return cap
        pdt = cv2.distanceTransform((1 - pe).astype(np.uint8), cv2.DIST_L2, 3)
        return 0.5 * float(np.minimum(real_dt[pe > 0], cap).mean() + np.minimum(pdt[real_edge], cap).mean())

    def pose(x):
        return Rot.from_rotvec(x[:3]).as_matrix() @ Rj, tj + x[3:6]

    def terms(x):
        R, t = pose(x)
        out = {"fin": (chamfer(R, t) / CHAMFER_SD) ** 2,
               "landmark": float(np.sum(((t - R[2] * (LENS_BACK + MOUNT["thickness"]) - landmark) / LANDMARK_SD) ** 2)),
               "photo": float(np.sum(((t[:2] - PHOTO_XY) / PHOTO_SD) ** 2)), "bricks": 0.0}
        for k, (_, _, uc, vb, wb, vt, wt) in enumerate(BRICKS):
            xc, yc, zb = x[6 + 3 * k: 9 + 3 * k]
            bot = np.array([[xc - B / 2, yc + B / 2, zb], [xc + B / 2, yc + B / 2, zb]])
            pb, _ = cam.project(R, t, bot)
            pt, _ = cam.project(R, t, bot + [0, 0, BH])
            r = np.r_[(pb.mean(0) - [uc, vb]) / 6, (pb[1, 0] - pb[0, 0] - wb) / 6, (pt[:, 1].mean() - vt) / 10,
                      (pt[1, 0] - pt[0, 0] - wt) / 12, (yc - BRICK_Y[0]) / BRICK_Y[1], (zb - BRICK_Z[0]) / BRICK_Z[1],
                      (xc - np.polyval(kface, zb + BH / 2) - B / 2) / BRICK_X_SD]
            out["bricks"] += float(np.sum(r ** 2))
        return out

    def total(x):
        return sum(terms(x).values())

    x0 = np.r_[np.zeros(6), [np.polyval(kface, BRICK_Z[0]) + B / 2, BRICK_Y[0], BRICK_Z[0]] * len(BRICKS)]
    rng = np.random.default_rng(0)
    best = None
    for seed in range(4):  # the silhouette term has local minima: a few restarts around joe_cal2
        xs = x0.copy()
        if seed:
            xs[:6] += rng.normal(0, [0.08, 0.08, 0.15, 0.006, 0.006, 0.006])
        s = minimize(total, xs, method="Powell", options=dict(xtol=1e-4, ftol=1e-6, maxfev=20000))
        if best is None or s.fun < best.fun:
            best = s
    R, t = pose(best.x)
    # centred-pinhole equivalent: rotate so the sim camera (f_sim, centre 320/240) renders the physical camera's pixels
    f_sim = (mea["fx"] + mea["fy"]) / 2
    Ks = np.array([[f_sim, 0, 320], [0, f_sim, 240], [0, 0, 1]])
    u, v = np.meshgrid(np.linspace(40, 600, 15), np.linspace(30, 450, 11))
    uv = np.stack([u.ravel(), v.ravel()], 1)
    X = (R.T @ (np.linalg.inv(cam.K) @ np.c_[uv, np.ones(len(uv))].T * 0.25)).T + t

    def sim_res(r):
        q = (X - t) @ (Rot.from_rotvec(r).as_matrix() @ R).T @ Ks.T
        return (q[:, :2] / q[:, 2:] - uv).ravel()

    s2 = minimize(lambda r: float(np.sum(sim_res(r) ** 2)), np.zeros(3), method="Powell")
    Rs = Rot.from_rotvec(s2.x).as_matrix() @ R
    rms_sim = float(np.sqrt(np.mean(sim_res(s2.x).reshape(-1, 2) ** 2) * 2))

    def cmp(Rx, tx):
        return {"position_local": np.round(tx, 5).tolist(), "optical_axis": np.round(Rx[2], 4).tolist(),
                "d_position_mm": round(float(np.linalg.norm(tx - t) * 1000), 1),
                "d_rotation_deg_vs_physical": round(float(np.degrees(Rot.from_matrix(Rx @ R.T).magnitude())), 2),
                "fin_chamfer_px": round(float(chamfer(Rx, tx)), 2)}

    Ru, tu = urdf_variant()
    tm = terms(best.x)
    bricks = best.x[6:].reshape(-1, 3)
    return {
        "frame": "gripper_link (SO-101 so101_new_calib.urdf); rows = OpenCV camera axes (x right, y down, z optical)",
        "camera": {"local_to_cv": np.round(R, 6).tolist(), "local_to_cv_sim": np.round(Rs, 6).tolist(),
                   "position_local": np.round(t, 6).tolist(), "focal_px": round(f_sim, 3),
                   "intrinsics_measured": {k: mea[k] for k in ("fx", "fy", "cx", "cy", "dist")},
                   "sim_model": f"centred pinhole f={f_sim:.1f}; local_to_cv_sim reproduces the physical camera's "
                                f"pixels within {rms_sim:.1f} px rms (principal point (339, 236) as a "
                                f"{np.degrees(np.linalg.norm(s2.x)):.1f} deg rotation; distortion not modelled)",
                   "optical_axis_tilt_from_gripper_axis_deg": round(float(np.degrees(np.arccos(-R[2, 2]))), 1)},
        "mount": MOUNT,
        "fit": {"fin_chamfer_px": round(float(chamfer(R, t)), 2), "terms": {k: round(v, 2) for k, v in tm.items()},
                "frame_back_minus_landmark_mm": np.round(
                    (t - R[2] * (LENS_BACK + MOUNT["thickness"]) - landmark) * 1000, 1).tolist(),
                "bricks_gripper_frame": [
                    {"episode": b[0], "frame": b[1], "centre_x": round(float(x[0]), 4),
                     "centre_y": round(float(x[1]), 4), "bottom_z": round(float(x[2]), 4)}
                    for b, x in zip(BRICKS, bricks, strict=True)]},
        "compare": {"joe_cal2": cmp(Rj, tj), "urdf_hexnut_variant": cmp(Ru, tu)},
        "provenance": "sim/twin/robot/fit_wrist_mount.py (evidence in its docstring)",
    }


def main():
    d = fit()
    OUT.write_text(json.dumps(d, indent=1) + "\n")
    c = d["camera"]
    print("optical centre (mm):", np.round(np.array(c["position_local"]) * 1000, 1).tolist(),
          "| axis", np.round(c["local_to_cv"][2], 3).tolist(), "| tilt", c["optical_axis_tilt_from_gripper_axis_deg"])
    print("fit:", json.dumps(d["fit"]))
    for k, v in d["compare"].items():
        print(f"{k}: d_position {v['d_position_mm']} mm, d_rotation {v['d_rotation_deg_vs_physical']} deg, "
              f"fin chamfer {v['fin_chamfer_px']} px")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
