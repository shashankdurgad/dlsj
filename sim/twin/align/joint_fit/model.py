"""Silhouette model for the joint fit: cameras, rasterised SO-101 meshes (numpy FK), loss per frame.

Parameters theta (dict of arrays), all as deltas from the current twin profile (joe_cal2 joints / gripper /
wrist mount, measured inspect camera, confirmed top camera):
  joints  (5,) rad  added to joe_cal2 offsets for pan, lift, elbow, wrist_flex, wrist_roll
  grip    (2,)      gripper q = units * (scale0 + grip[1]) + bias0 + grip[0]   (moving-jaw visual angle)
  inspect (6,)      rotvec (applied in the camera frame: R = Rot(rv) R0) + translation (m) of the inspect camera
  wrist   (6,)      same for the wrist camera pose in the gripper_link frame
  top     (6,)      same for the top camera (kept 0 unless explicitly freed)
  plane   (1,)      height of the plane the table-view check casts onto, minus 1 cm (table.py)
  sag     (3,)      rad per N m: lift / elbow / wrist_flex angle += sag * static gravity torque (arm compliance)
  jpr     (2,)      extra pan / wrist_roll deltas (lets a fit free only those two joints' offsets)
Loss per frame (fit_cameras.py's, on triangle-rasterised silhouettes instead of per-link convex hulls):
  a = mean distance (px at 320x240, capped) of projected surface points to the real mask,
  b = mean distance of real-mask pixels within a band around the sim robot to the sim silhouette;
  IoU within the band. The base / shoulder silhouette is a don't-care region (the real mask is background-subtracted).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
TWIN = HERE.parents[1]
REPO = TWIN.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "sim/residual_env"))
from fk import SO101  # noqa: E402

H, W, S = 240, 320, 0.5  # working resolution (half of 640x480)
CAP = 25.0
JOE = json.loads((REPO / "sim/residual_env/assets/joe_cal2/reference-cal2.json").read_text())
MEAS = json.loads((TWIN / "calib/cameras_measured.json").read_text())
FAC = json.loads((TWIN / "calib/realsense_intrinsics.json").read_text())
JOE_OFFSETS = np.array(JOE["joint_offsets_rad"][:5])
GRIP_SCALE0 = float(JOE["gripper_mapping"]["scale_rad_per_unit"])
GRIP_BIAS0 = 0.10756 - 5.8 * GRIP_SCALE0  # the env's joe_cal2 bias (sim/residual_env/calib.py)
NAMES = ("joints", "grip", "inspect", "wrist", "top", "plane", "sag", "jpr")
SIZES = dict(joints=5, grip=2, inspect=6, wrist=6, top=6, plane=1, sag=3, jpr=2)


def zero_theta():
    return {k: np.zeros(n) for k, n in SIZES.items()}


def pack(theta, free):
    return np.concatenate([theta[k] for k in free])


def unpack(x, free, base):
    th = {k: v.copy() for k, v in base.items()}
    i = 0
    for k in free:
        th[k] = np.asarray(x[i:i + SIZES[k]], float)
        i += SIZES[k]
    return th


def base_cams():
    top = dict(R=np.array(JOE["camera"]["world_to_cv"]), t=np.array(JOE["camera"]["position"]),
               **{k: FAC["top"][k] for k in ("fx", "fy", "cx", "cy")})
    m = MEAS["inspect"]
    insp = dict(R=np.array(m["world_to_cv"]), t=np.array(m["position"]), **{k: m[k] for k in ("fx", "fy", "cx", "cy")})
    w = MEAS["wrist"]
    wrist = dict(R=np.array(JOE["wrist_camera"]["local_to_cv"]), t=np.array(JOE["wrist_camera"]["position_local"]),
                 dist=np.array(w["dist"]), **{k: w[k] for k in ("fx", "fy", "cx", "cy")})
    joe_insp = JOE["inspect_camera"]
    joe_inspect = dict(R=np.array(joe_insp["world_to_cv"]), t=np.array(joe_insp["position"]), fx=joe_insp["focal_px"],
                       fy=joe_insp["focal_px"], cx=320.0, cy=240.0)
    return dict(top=top, inspect=insp, wrist=wrist, joe_inspect=joe_inspect)


BASE = base_cams()


def perturb(cam, d):
    if not np.any(d):
        return cam
    return {**cam, "R": Rotation.from_rotvec(d[:3]).as_matrix() @ cam["R"], "t": cam["t"] + d[3:6]}


def cams_of(theta, base=BASE):
    return dict(top=perturb(base["top"], theta["top"]), inspect=perturb(base["inspect"], theta["inspect"]),
                wrist=perturb(base["wrist"], theta["wrist"]))


_ARM = None
CAM_MASS = (0.02, np.array([0.0, 0.06, -0.04]))  # wrist camera + mount on gripper_link (approx.)


def q_of(state, theta):
    global _ARM
    s = np.asarray(state, float)
    q = np.r_[np.deg2rad(s[:5]) + JOE_OFFSETS + theta["joints"],
              s[5] * (GRIP_SCALE0 + theta["grip"][1]) + GRIP_BIAS0 + theta["grip"][0]]
    if "jpr" in theta:
        q[0] += theta["jpr"][0]
        q[4] += theta["jpr"][1]
    sag = theta.get("sag")
    if sag is not None and np.any(sag):
        if _ARM is None:
            _ARM = SO101()
        tau = _ARM.gravity_torque(q, {"gripper_link": CAM_MASS})
        q[1:4] += sag * tau[1:4]
    return q


def world_cam(cam_name, cams, T):
    c = cams[cam_name]
    if cam_name != "wrist":
        return c
    G = T["gripper_link"]
    return {**c, "R": c["R"] @ G[:3, :3].T, "t": G[:3, 3] + G[:3, :3] @ c["t"]}


def project(P, cam):
    """P (N,3) world -> uv (N,2) at full resolution, ok mask (in front, distortion valid)."""
    Xc = (P - cam["t"]) @ cam["R"].T
    z = Xc[:, 2]
    ok = z > 0.01
    zz = np.where(ok, z, 1.0)
    x, y = Xc[:, 0] / zz, Xc[:, 1] / zz
    if "dist" in cam:
        k1, k2, p1, p2, k3 = cam["dist"]
        ok &= (np.abs(x) < 1.6) & (np.abs(y) < 1.6)
        r2 = x * x + y * y
        rad = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
        x, y = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x), y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([cam["fx"] * x + cam["cx"], cam["fy"] * y + cam["cy"]], 1), ok


class Robot:
    """Meshes per camera role: fg (scored), ignore (don't-care: static base / shoulder)."""

    def __init__(self, arm: SO101 | None = None):
        self.arm = arm or SO101()
        a = self.arm
        # The real gripper has TPU fin-ray fingers, not the URDF jaws: the fixed jaw (gripper_link geometry below
        # z = FINGER_Z) and the moving jaw are don't-care, like the static base / shoulder.
        body, fingers = [], []
        for _, V, F in a.parts["gripper_link"]:
            fz = V[F].mean(1)[:, 2]
            body.append(("gripper_link", V, F[fz >= FINGER_Z]))
            if (fz < FINGER_Z).any():
                fingers.append(("gripper_link", V, F[fz < FINGER_Z]))
        self.sets = {
            "arm": dict(fg=[(ln, V, F) for ln in ("upper_arm_link", "lower_arm_link", "wrist_link")
                            for (_, V, F) in a.parts[ln]] + body,
                        ig=[(ln, V, F) for ln in ("base_link", "shoulder_link", "moving_jaw_so101_v1_link")
                            for (_, V, F) in a.parts[ln]] + fingers),
            # wrist view: only the red jaws (fixed jaw = wrist_roll_follower part of gripper_link, moving jaw)
            "jaws": dict(fg=[("gripper_link", V, F) for (n, V, F) in a.parts_fine["gripper_link"] if "follower" in n]
                         + [("moving_jaw_so101_v1_link", V, F)
                            for (_, V, F) in a.parts_fine["moving_jaw_so101_v1_link"]],
                         ig=[]),
        }
        rng = np.random.default_rng(0)
        for s in self.sets.values():  # surface points for the a-term: ~400 vertices per part (of the kept faces)
            s["pts"] = []
            for (ln, V, F) in s["fg"]:
                u = np.unique(F)
                s["pts"].append((ln, V[rng.choice(u, min(400, len(u)), replace=False)]))

    def render(self, which, T, cam):
        s = self.sets[which]
        out = []
        for key in ("fg", "ig"):
            m = np.zeros((H, W), np.uint8)
            tris = []
            for ln, V, F in s[key]:
                P = V @ T[ln][:3, :3].T + T[ln][:3, 3]
                uv, ok = project(P, cam)
                good = ok[F].all(1)
                t = uv[F[good]] * S
                t = t[(np.abs(t) < 4000).all((1, 2))]
                tris.append(t)
            if tris:
                raster(np.concatenate(tris), m)
            out.append(m)
        P = np.concatenate([p @ T[ln][:3, :3].T + T[ln][:3, 3] for ln, p in s["pts"]])
        uv, ok = project(P, cam)
        uv = uv[ok] * S
        uv = uv[(uv[:, 0] >= 0) & (uv[:, 0] <= W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= H - 1)]
        return out[0], out[1], uv


RASTER_CHUNK = 32


def raster(t, m, chunk=RASTER_CHUNK):
    """Fill triangles t (N,3,2) (working-resolution pixels) into mask m: cv2.fillPoly in small chunks (one call
    with tens of thousands of polygons is ~100x slower)."""
    ti = np.round(t * 4).astype(np.int32)
    for k in range(0, len(ti), chunk):
        cv2.fillPoly(m, list(ti[k:k + chunk]), 1, lineType=cv2.LINE_8, shift=2)


def bilinear(img, uv):
    x, y = uv[:, 0], uv[:, 1]
    x0, y0 = np.clip(np.floor(x).astype(int), 0, W - 2), np.clip(np.floor(y).astype(int), 0, H - 2)
    fx, fy = x - x0, y - y0
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
            + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)


FINGER_Z = -0.045  # gripper_link z below which the URDF geometry is finger (motor / body end at ~-0.04)
BAND = {"top": 31, "inspect": 31, "wrist": 21}


def frame_loss(robot, cam_name, state, real, dt_real, theta, cams=None, want_masks=False, dc=None):
    """-> (loss, iou, valid[, sim mask]). dc: real-image don't-care mask (red fingers)."""
    cams = cams or cams_of(theta)
    T = robot.arm.fk(q_of(state, theta))
    cam = world_cam(cam_name, cams, T)
    sm, ig, uv = robot.render("jaws" if cam_name == "wrist" else "arm", T, cam)
    valid = len(uv) >= 50 and sm.sum() >= 30
    if not valid:
        return (np.nan, np.nan, False, sm) if want_masks else (np.nan, np.nan, False)
    a = np.minimum(bilinear(dt_real, uv), CAP).mean()
    k = BAND[cam_name]
    band = cv2.dilate(sm, np.ones((k, k), np.uint8)) > 0
    rb = (real > 0) & band & (ig == 0)
    if dc is not None:
        rb &= dc == 0
    if rb.sum() >= 30:
        dt_sim = cv2.distanceTransform((1 - sm).astype(np.uint8), cv2.DIST_L2, 3)
        b = np.minimum(dt_sim[rb], CAP).mean()
    else:
        b = CAP
    inter = ((sm > 0) & rb).sum()
    union = ((sm > 0) | rb).sum()
    r = (float(a + b), float(inter / max(union, 1)), True)
    return (*r, sm) if want_masks else r


TIP_LOCAL = np.array([0.008, 0.0, -0.090])  # env.py grasp point on gripper_link
GRASP_DROP = 0.009  # m: replay depth gap with the real fin-ray fingers modelled (8-10 mm, gripper agent)


def tip_z(robot, state, theta):
    """World z (m) of the env's grasp point TIP_LOCAL at a recorded state."""
    G = robot.arm.fk(q_of(state, theta))["gripper_link"]
    return float((G[:3, :3] @ TIP_LOCAL + G[:3, 3])[2])


def grasp_loss(robot, state, theta, z0):
    """Grasp-depth term at a gripper-close frame: the fingertip should sit GRASP_DROP below where joe_cal2 puts it
    (z0) -> (|dz + GRASP_DROP| in mm capped, dz = change vs joe_cal2 in mm)."""
    dz = tip_z(robot, state, theta) - z0
    return float(min(abs(dz + GRASP_DROP) * 1e3, CAP)), float(dz * 1e3)


def load_data():
    Z = np.load(TWIN / "out/align/joint_fit/data.npz", allow_pickle=True)
    D = {k: Z[k] for k in Z.files}  # NpzFile re-decompresses on every access
    eps = list(D["episodes"])
    held = set(eps[2::3])
    items = []
    for i in range(len(D["cam"])):
        m = D["mask"][i]
        items.append(dict(i=i, cam=str(D["cam"][i]), ep=int(D["ep"][i]), fr=int(D["fr"][i]), state=D["state"][i],
                          speed=float(D["speed"][i]), real=m, held=int(D["ep"][i]) in held, dc=D["dc"][i],
                          dt=cv2.distanceTransform((1 - m).astype(np.uint8), cv2.DIST_L2, 3)))
    return items, D
