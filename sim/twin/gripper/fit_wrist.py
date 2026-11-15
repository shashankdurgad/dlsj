"""Check / fit the fin-ray fingers against real wrist-camera frames (plain python, no Isaac).

Both red fingers are in every wrist frame. The fixed finger is rigid with the camera, so its silhouette checks the
fin placement and the wrist-camera pose; the moving finger's silhouette then gives the gripper joint angle of each
frame, and angle vs recorded reading gives the gripper mapping (q = units * scale + bias).

    python sim/twin/gripper/fit_wrist.py frames     # decode ~40 wrist frames spread over the reading (PyAV)
    python sim/twin/gripper/fit_wrist.py fit        # silhouette fit -> out/gripper/fit_wrist.{json,jpg}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares, minimize, minimize_scalar
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
HERE = Path(__file__).resolve().parent
OUT = REPO / "sim/twin/out/gripper"
REF = REPO / "sim/residual_env/assets/joe_cal2/reference-cal2.json"
MEASURED = REPO / "sim/twin/calib/cameras_measured.json"
WRIST_MOUNT = REPO / "sim/twin/robot/wrist_mount.json"
JOINT_XYZ = np.array([0.0202, 0.0188, -0.0234])
W, H = 640, 480


def jaw_to_grip(p: np.ndarray, q: float) -> np.ndarray:
    """Moving-jaw link frame -> gripper_link frame at joint angle q (URDF: rpy (pi/2, 0, 0), axis z)."""
    c, s = np.cos(q), np.sin(q)
    R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]]) @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    return p @ R.T + JOINT_XYZ


def red_mask(rgb: np.ndarray) -> np.ndarray:
    """Pixels of the red TPU fingers (saturated red)."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = ((h < 6) | (h > 168)) & (s > 120) & (v > 90)
    return cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


class WristCam:
    """Wrist camera in the gripper_link frame with the measured intrinsics. Pose: the fitted mount
    (sim/twin/robot/wrist_mount.json, physical rotation) when it exists, else joe_cal2's; `delta` (rotation vector,
    translation) is an optional correction on top."""

    def __init__(self, delta=None):
        k = json.loads(MEASURED.read_text())["wrist"]
        if WRIST_MOUNT.exists():
            from sim.twin.robot.wrist_mount import load as load_mount

            mount = load_mount(WRIST_MOUNT)
            self.R, self.pos, self.source = mount.R_cv.copy(), mount.t.copy(), "wrist_mount"
        else:
            ref = json.loads(REF.read_text())["wrist_camera"]
            self.R, self.pos, self.source = np.array(ref["local_to_cv"]), np.array(ref["position_local"]), "joe_cal2"
        self.K = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1.0]])
        self.dist = np.array(k["dist"])
        if delta is not None:
            self.R = Rotation.from_rotvec(delta[:3]).as_matrix() @ self.R
            self.pos = self.pos + np.asarray(delta[3:6])

    def project(self, p: np.ndarray):
        """(pixels (n, 2), depth (n,)) of gripper-frame points (distortion model applied)."""
        pc = (p - self.pos) @ self.R.T
        z = pc[:, 2]
        uv, _ = cv2.projectPoints(pc.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), self.K, self.dist)
        return uv.reshape(-1, 2), z

    def silhouette(self, verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """Filled silhouette mask of a mesh given in the gripper frame (triangles behind the camera dropped)."""
        uv, z = self.project(verts)
        r = np.linalg.norm((verts - self.pos) @ self.R.T[:, :2], axis=1) / np.maximum(z, 1e-6)
        ok = ((z > 0.01) & (r < 1.5))[faces].all(1)  # in front of the lens, inside the distortion model's range
        tri = np.round(uv[faces[ok]] * 8).astype(np.int32)
        tri = tri[(np.abs(tri) < 8 * 4000).all((1, 2))]
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, list(tri), 1, shift=3)
        return m


def load_model() -> dict:
    """assets/finray_so101.npz with float64 vertices."""
    d = np.load(HERE / "assets/finray_so101.npz")
    return {k: d[k].astype(np.float64) if k.endswith(("_v", "_tip", "_face")) else d[k] for k in d.files}


def edge(mask: np.ndarray) -> np.ndarray:
    """Boundary pixels of a mask that are not on the image border (the border cuts both silhouettes)."""
    e = mask - cv2.erode(mask, np.ones((3, 3), np.uint8))
    e[:3], e[-3:], e[:, :3], e[:, -3:] = 0, 0, 0, 0
    return e


def chamfer(pred: np.ndarray, real: np.ndarray, real_dt: np.ndarray, cap: float = 40.0) -> float:
    """Symmetric mean boundary distance (px, capped) between two masks; real_dt = distance to real's boundary."""
    pe = edge(pred)
    if pe.sum() < 20:
        return cap
    pred_dt = cv2.distanceTransform((1 - pe).astype(np.uint8), cv2.DIST_L2, 3)
    re = edge(real) > 0
    return 0.5 * (float(np.minimum(real_dt[pe > 0], cap).mean()) + float(np.minimum(pred_dt[re], cap).mean()))


def edge_dt(mask: np.ndarray) -> np.ndarray:
    """Distance (px) to the boundary of a mask."""
    return cv2.distanceTransform((1 - edge(mask)).astype(np.uint8), cv2.DIST_L2, 3)


def fit_q(cam: WristCam, model: dict, moving: np.ndarray) -> tuple[float, float]:
    """Joint angle whose moving-finger silhouette best matches the moving finger's red pixels (IoU)."""
    def cost(q):
        sil = cam.silhouette(jaw_to_grip(model["jaw_fin_v"], q), model["jaw_fin_f"])
        return 1.0 - (sil & moving).sum() / max((sil | moving).sum(), 1)

    grid = np.arange(-0.15, 0.9, 0.02)
    qb = grid[int(np.argmin([cost(q) for q in grid]))]
    r = minimize_scalar(cost, bounds=(qb - 0.02, qb + 0.02), method="bounded", options={"xatol": 5e-4})
    return float(r.x), float(r.fun)


def stage_frames(n_eps: int, per_ep: int):
    from sim.twin.align import frames as F

    rng = np.random.default_rng(0)
    eps = sorted(rng.choice(456, n_eps, replace=False).tolist())
    keep = {}
    targets = [3, 6, 10, 15, 22, 30, 40, 55]
    for ep in eps:
        s = F.states(ep)
        g = s[:, 5]
        idx = sorted({int(np.argmin(np.abs(g - t))) for t in rng.choice(targets, per_ep, replace=False)})
        fr = F.video_frames(ep, "wrist", idx)
        for i in idx:
            if i in fr:
                keep[f"{ep}_{i}"] = fr[i]
                keep[f"{ep}_{i}_grip"] = g[i]
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "fit_wrist_frames.npz", **keep)
    print("FRAMES_OK", len(keep) // 2)


def stage_fit(scale: float, bias0: float, fit_cam: bool):
    """Fit the wrist-camera pose on the fixed finger, then the joint angle per frame; write json + contact sheet."""
    model = load_model()
    d = np.load(OUT / "fit_wrist_frames.npz")
    keys = sorted(k for k in d.files if not k.endswith("_grip"))
    imgs = [d[k] for k in keys]
    grip = np.array([float(d[k + "_grip"]) for k in keys])
    reals = [red_mask(im) for im in imgs]
    ffv, fff = model["fixed_fin_v"], model["fixed_fin_f"]

    # 1. the fixed finger is rigid with the camera: red in most frames at the same pixels
    always = (np.mean(reals, 0) > 0.6).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(always)
    always = (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    always_dt = edge_dt(always)
    cam = WristCam()
    res = {"n_frames": len(keys), "wrist_pose": cam.source,
           "fixed_chamfer_px_nominal_cam": round(chamfer(cam.silhouette(ffv, fff), always, always_dt), 2)}
    if fit_cam:
        def cost(x):
            sil = WristCam(x * np.array([1, 1, 1, 0.1, 0.1, 0.1])).silhouette(ffv, fff)
            return chamfer(sil, always, always_dt) + 30.0 * float(np.sum(x[3:] ** 2))  # weak prior on position

        best = None
        for x0 in (np.zeros(6), np.array([0.15, 0, 0, 0, 0, 0]), np.array([-0.15, 0, 0, 0, 0, 0]),
                   np.array([0, 0.15, 0, 0, 0, 0]), np.array([0, -0.15, 0, 0, 0, 0])):
            r = minimize(cost, x0, method="Powell", options={"xtol": 1e-3, "ftol": 1e-3, "maxfev": 1500})
            if best is None or r.fun < best.fun:
                best = r
        delta = best.x * np.array([1, 1, 1, 0.1, 0.1, 0.1])
        cam = WristCam(delta)
        res["cam_delta_rotvec_rad"] = delta[:3].round(4).tolist()
        res["cam_delta_rot_deg"] = round(float(np.degrees(np.linalg.norm(delta[:3]))), 2)
        res["cam_delta_pos_mm"] = (delta[3:] * 1e3).round(2).tolist()
        res["cam_local_to_cv_fitted"] = cam.R.round(6).tolist()
        res["cam_position_local_fitted"] = cam.pos.round(5).tolist()
        res["fixed_chamfer_px_fitted_cam"] = round(chamfer(cam.silhouette(ffv, fff), always, always_dt), 2)
    fixed_sil = cam.silhouette(ffv, fff)
    not_fixed = 1 - cv2.dilate(fixed_sil | always, np.ones((9, 9), np.uint8))

    # 2. moving finger: joint angle per frame
    fits = [fit_q(cam, model, real & not_fixed) for real in reals]
    qs, costs = np.array([f[0] for f in fits]), np.array([f[1] for f in fits])
    good = (costs < 0.6) & (grip < 28.0)  # wider open: the moving finger leaves the image
    r2 = least_squares(lambda x: x[0] * grip[good] + x[1] - qs[good], [scale, bias0], loss="soft_l1", f_scale=0.02)
    r1 = least_squares(lambda b: scale * grip[good] + b[0] - qs[good], [bias0], loss="soft_l1", f_scale=0.02)
    given = scale * grip[good] + bias0 - qs[good]
    res.update(scale_given=scale, bias_given=bias0, n_good=int(good.sum()),
               bias_fit_scale_given=round(float(r1.x[0]), 4), scale_fit=round(float(r2.x[0]), 5),
               bias_fit_free_scale=round(float(r2.x[1]), 4),
               resid_median_rad_given_mapping=round(float(np.median(given)), 4),
               resid_rms_rad_given_mapping=round(float(np.sqrt(np.mean(given ** 2))), 4),
               resid_rms_rad_fit_bias=round(float(np.sqrt(np.mean(r1.fun ** 2))), 4),
               moving_iou_median=round(float(1 - np.median(costs)), 3),
               per_frame=[{"key": k, "grip": round(float(g), 2), "q_fit": round(float(q), 4),
                           "iou": round(1 - float(c), 3)} for k, g, q, c in zip(keys, grip, qs, costs, strict=True)])
    (OUT / "fit_wrist.json").write_text(json.dumps(res, indent=1))
    print("FIT", json.dumps({k: v for k, v in res.items() if k != "per_frame"}))

    # contact sheet: model outlines on the real frame: fixed finger white, moving finger at the mapping's q (green)
    # and at the fitted q (yellow)
    order = np.argsort(grip)
    tiles = []
    for i in order[np.linspace(0, len(order) - 1, 12).astype(int)]:
        im = imgs[i].copy()
        layers = [(fixed_sil, (255, 255, 255))]
        for q, col in ((scale * grip[i] + bias0, (0, 255, 0)), (qs[i], (255, 255, 0))):
            layers.append((cam.silhouette(jaw_to_grip(model["jaw_fin_v"], q), model["jaw_fin_f"]), col))
        for sil, col in layers:
            cs, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(im, cs, -1, col, 1)
        cv2.putText(im, f"{keys[i]} grip {grip[i]:.1f} q_fit {qs[i]:.3f} iou {1 - costs[i]:.2f}", (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        tiles.append(im)
    sheet = np.concatenate([np.concatenate(tiles[r * 4:r * 4 + 4], 1) for r in range(3)], 0)
    cv2.imwrite(str(OUT / "fit_wrist.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    cv2.imwrite(str(OUT / "fit_wrist_fixed_mask.png"), always * 255)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("frames", "fit"))
    ap.add_argument("--n_eps", type=int, default=14)
    ap.add_argument("--per_ep", type=int, default=3)
    ap.add_argument("--scale", type=float, default=0.02307670501098437)
    ap.add_argument("--bias", type=float, default=0.105)
    ap.add_argument("--fit_cam", type=int, default=1)
    a = ap.parse_args()
    if a.stage == "frames":
        stage_frames(a.n_eps, a.per_ep)
    else:
        stage_fit(a.scale, a.bias, bool(a.fit_cam))
