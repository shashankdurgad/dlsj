"""Wrist-camera visibility of a brick: FK, the wrist camera, the gripper's occluders, brick face samples, ray tests.

Batched and backend-agnostic: every function takes numpy arrays or torch tensors (any device) and returns the same
kind (`xp` = numpy or torch is picked from the inputs). Used by the camera-free grasp RL env (torch, GPU), TwinEnv /
analysis (numpy) and the real-data fits (numpy).

Frames: `env` = robot base frame (the residual env / GraspEnv env frame, metres); `gripper_link` = the SO-101 link
the wrist camera and the fixed fin are bolted to. Joints in OUR units (LeRobot degrees of the cal2 frame, gripper
0-100), mapped like the twin: q = deg2rad(d) + joint_offsets, gripper q = 0.02308 u + GRIP_BIAS (physics, -0.10 rad).

Occluders (gripper_link / moving-jaw link, from sim/twin/gripper/assets/finray_so101.npz): the TPU fins as their 4
convex collider pieces each (the fin's ribbed outline is filled; see proxy accuracy in README), the two printed
bases as one convex hull each, plus optional boxes (other bricks). A brick sample point is visible when it is on a
camera-facing face, inside the image and the segment camera -> point crosses no occluder.

    G = Geometry()                                    # numpy float64; Geometry(torch_device="cuda:0") for torch
    pose = G.gripper_pose(joints_deg)                 # (..., 4, 4) gripper_link in env
    vis = G.visibility(joints_deg, brick_pos, brick_R, brick_size, others=None)
    vis["vis"], vis["occ_grip"], vis["oof"], vis["box_vis"], vis["box_amodal"], vis["dist"], ...
    rel = G.box_to_rel(box, joints_deg, plane_z)      # detector box -> brick centre - fingertip, gripper frame
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
URDF = REPO / "sim/twin/assets/so101/urdf/so101_new_calib.urdf"
FINRAY = REPO / "sim/twin/gripper/assets/finray_so101.npz"
WRIST_JSON = REPO / "sim/twin/robot/wrist_mount.json"
CHAIN = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
JAW_XYZ = np.array([0.0202, 0.0188, -0.0234])  # gripper joint origin in gripper_link, rpy (pi/2, 0, 0)


def _twin_gripper() -> tuple:
    """(fingertip / grasp point in gripper_link, gripper bias rad) of the twin, from sim/twin/physics/apply.py
    (the fin-ray grasp point and params.json's gripper bias); plain-number fallback when it cannot be imported."""
    try:
        from sim.twin.physics import apply as A

        g = A._p()["gripper"]
        b = g["finray"].get("bias_rad") if A.finray() else None
        b = g.get("bias_rad") if b is None else b
        return tuple(float(v) for v in A.tip_local()), float(b)
    except Exception:  # noqa: BLE001
        return (0.0063, 0.001, -0.0902), -0.10


_TIP, _BIAS = _twin_gripper()
# twin jaw map q = GRIP_SCALE u + GRIP_BIAS (profile scale, physics bias; -0.10 = servo map fitted on open frames)
GRIP_SCALE, GRIP_BIAS = 0.02307670501098437, _BIAS
# the jaw angle the REAL wrist camera sees vs the reading (jaw_fit.py, 496 real frames: the moving fin's silhouette):
# ~0.2 rad more closed than the twin's mapping at every opening (at the hold it includes the TPU fin's bend)
JAW_REAL = (0.02406, -0.1183)
TIP_LOCAL = _TIP  # the twin's one grasp point: the fin-ray point (0.0063, 0.001, -0.0902), physics apply.tip_local()
TABLE_Z = -0.002
W, H = 640, 480
PARTS = ("fixed_fin", "jaw_fin", "fixed_base", "jaw_base")


def _xp(*arrs):
    for a in arrs:
        if a is not None and type(a).__module__.startswith("torch"):
            import torch

            return torch
    return np


def _as(xp, a, like):
    if xp is np:
        return np.asarray(a, dtype=np.float64)
    return xp.as_tensor(np.asarray(a) if not hasattr(a, "device") else a, dtype=like.dtype, device=like.device)


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def default_offsets() -> list:
    """Twin joint offsets (rad): the twin profile's default (W9 from align/joint_fit/result.json, W if absent).
    DLSJ_JOINT_OFFSETS=joe_cal2 selects joe_cal2's explicitly; a missing result.json is an error, never a silent
    fallback."""
    import os

    if os.environ.get("DLSJ_JOINT_OFFSETS") == "joe_cal2":
        import sys

        sys.path.insert(0, str(REPO / "sim/residual_env"))
        from calib import load_profile

        return list(load_profile("joe_cal2").joint_offsets)
    p = REPO / "sim/twin/align/joint_fit/result.json"
    res = json.loads(p.read_text())  # FileNotFoundError if missing
    rad = (
        res.get("grasp_depth_candidates", {}).get("W9", {}).get("joint_offsets_rad_for_sim")
        or res["joint_offsets_rad_for_sim"]
    )
    return [float(v) for v in rad[:5]] + [0.0]


@lru_cache(maxsize=1)
def _urdf_chain():
    root = ET.parse(str(URDF)).getroot()
    joints = {j.get("name"): j for j in root.findall("joint")}
    T = []
    for name in CHAIN:
        o = joints[name].find("origin")
        M = np.eye(4)
        M[:3, :3] = _rpy(*[float(v) for v in o.get("rpy").split()])
        M[:3, 3] = [float(v) for v in o.get("xyz").split()]
        T.append(M)
    return np.stack(T)


def _planes(verts: np.ndarray, k: int = 40) -> np.ndarray:
    """Convex hull of verts -> k + 6 outward planes (k + 6, 4), n . p + d <= 0 inside: a k-DOP whose directions
    are the hull's area-weighted face normals clustered to k (plus the 6 axes), each plane at the support distance
    of the vertices. Always contains the hull; tight along the dominant faces (max excess ~0.5 mm on the fins)."""
    from scipy.spatial import ConvexHull

    hull = ConvexHull(verts)
    nrm = hull.equations[:, :3]
    tri = verts[hull.simplices]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    if len(nrm) <= k:
        dirs = nrm.copy()
    else:  # spherical k-means, area weighted, farthest-point init
        idx = [int(np.argmax(area))]
        for _ in range(k - 1):
            dmin = np.min(1 - nrm @ nrm[idx].T, axis=1)
            idx.append(int(np.argmax(dmin * (area > 0))))
        dirs = nrm[idx].copy()
        for _ in range(20):
            lab = np.argmax(nrm @ dirs.T, axis=1)
            for j in range(k):
                m = lab == j
                if m.any():
                    v = (nrm[m] * area[m, None]).sum(0)
                    dirs[j] = v / (np.linalg.norm(v) + 1e-12)
    dirs = np.concatenate([dirs, np.eye(3), -np.eye(3)])
    off = (verts @ dirs.T).max(0)  # n . x <= off
    return np.concatenate([dirs, -off[:, None]], 1)


@lru_cache(maxsize=1)
def occluder_planes() -> dict:
    """{part: [(k, 4) planes, ...]} in each part's link frame (fixed_* in gripper_link, jaw_* in the jaw link)."""
    g = np.load(FINRAY)
    out = {}
    for side in ("fixed", "jaw"):
        n = sum(1 for k in g.files if k.startswith(f"{side}_hull") and k.endswith("_v"))
        out[f"{side}_fin"] = [_planes(g[f"{side}_hull{k}_v"].astype(float)) for k in range(n)]
        out[f"{side}_base"] = [_planes(g[f"{side}_base_v"].astype(float))]
    return out


def _pad(pl: list) -> np.ndarray:
    """List of (k_i, 4) plane sets -> (m, kmax, 4), padded with an always-satisfied plane (0, 0, 0, -1)."""
    k = max(len(p) for p in pl)
    out = np.zeros((len(pl), k, 4))
    out[..., 3] = -1.0
    for i, p in enumerate(pl):
        out[i, : len(p)] = p
    return out


@lru_cache(maxsize=4)
def wrist_camera(model: str = "physical") -> dict:
    """The wrist camera in gripper_link: R (rows = OpenCV axes), t, fx, fy, cx, cy, dist.
    model "physical": measured intrinsics (real frames); "sim": the twin's centred pinhole (rendered frames)."""
    d = json.loads(WRIST_JSON.read_text())["camera"]
    m = d["intrinsics_measured"]
    if model == "sim":
        f = float(d["focal_px"])
        return dict(
            R=np.array(d["local_to_cv_sim"]), t=np.array(d["position_local"]), fx=f, fy=f, cx=W / 2, cy=H / 2, dist=None
        )
    return dict(
        R=np.array(d["local_to_cv"]),
        t=np.array(d["position_local"]),
        fx=m["fx"],
        fy=m["fy"],
        cx=m["cx"],
        cy=m["cy"],
        dist=np.array(m["dist"]),
    )


@lru_cache(maxsize=8)
def _face_samples(g: int):
    """Unit-box face samples: points in [-1, 1]^3 (6 g^2, 3), normals (6 g^2, 3), cell area fraction per point."""
    u = np.linspace(-1, 1, g)
    a, b = np.meshgrid(u, u, indexing="ij")
    a, b = a.ravel(), b.ravel()
    pts, nrm, ax = [], [], []
    for k in range(3):
        i, j = [x for x in range(3) if x != k]
        for s in (-1.0, 1.0):
            p = np.zeros((g * g, 3))
            p[:, k], p[:, i], p[:, j] = s, a, b
            n = np.zeros((g * g, 3))
            n[:, k] = s
            pts.append(p)
            nrm.append(n)
            ax.append(np.full(g * g, k))
    # trapezoid weights so edge / corner points count half / a quarter
    wl = np.ones(g)
    wl[[0, -1]] = 0.5
    w2 = (wl[:, None] * wl[None, :]).ravel()
    w2 = w2 / w2.sum()
    return np.concatenate(pts), np.concatenate(nrm), np.concatenate(ax), np.tile(w2, 6)


def _splat_mesh(V, F, cam, shape, scale):
    """Min camera depth of a triangle mesh (camera-frame vertices V (k, 3), faces F) on a (H/scale, W/scale) grid:
    dense barycentric samples per triangle (about 2 per grid pixel), then a 2x2 min filter to close cracks."""
    D = np.full(shape, np.inf)
    tri = V[F]
    ok = (tri[:, :, 2] > 0.003).all(1)
    tri = tri[ok]
    uv = (
        np.stack(
            [cam["fx"] * tri[..., 0] / tri[..., 2] + cam["cx"], cam["fy"] * tri[..., 1] / tri[..., 2] + cam["cy"]], -1
        )
        / scale
    )
    area = 0.5 * np.abs(np.cross(uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]))
    lev = np.clip(np.ceil(np.sqrt(area * 2.0)), 1, 64).astype(int)
    for L in np.unique(lev):
        t = tri[lev == L]
        i, j = np.meshgrid(np.arange(L + 1), np.arange(L + 1), indexing="ij")
        keep = (i + j) <= L
        a, b = i[keep] / L, j[keep] / L
        w = np.stack([1 - a - b, a, b], 1)  # (S, 3)
        P = np.einsum("sk,tkd->tsd", w, t).reshape(-1, 3)
        u = cam["fx"] * P[:, 0] / P[:, 2] + cam["cx"]
        v = cam["fy"] * P[:, 1] / P[:, 2] + cam["cy"]
        ui, vi = np.floor(u / scale).astype(int), np.floor(v / scale).astype(int)
        m = (ui >= 0) & (ui < shape[1]) & (vi >= 0) & (vi < shape[0])
        np.minimum.at(D, (vi[m], ui[m]), P[m, 2])
    E = D.copy()
    E[:-1] = np.minimum(E[:-1], D[1:])
    E2 = E.copy()
    E2[:, :-1] = np.minimum(E[:, :-1], E[:, 1:])
    return E2


def occluder_depth_maps(
    camera: str = "physical", grip_scale: float = GRIP_SCALE, grip_bias: float = GRIP_BIAS, scale: int = 2, parts=PARTS
) -> dict:
    """Wrist-camera depth maps of the gripper parts (they are rigid in the camera frame; the jaw parts per gripper
    reading 0..100 in 1-unit bins). {part: (1 | 101, H/scale, W/scale) float32, inf = free}. Cached in
    sim/twin/out/perception/occluder_depth_<camera>.npz (rebuilt when the camera / mapping changes)."""
    cam = wrist_camera(camera)
    key = np.r_[cam["R"].ravel(), cam["t"], cam["fx"], cam["fy"], cam["cx"], cam["cy"], grip_scale, grip_bias, scale]
    path = REPO / f"sim/twin/out/perception/occluder_depth_{camera}.npz"
    if path.exists():
        with np.load(path) as z:
            if np.allclose(z["key"], key) and all(p in z.files for p in parts):
                return {p: z[p] for p in parts}
    g = np.load(FINRAY)
    shape = (H // scale, W // scale)
    out = {}
    for part in parts:
        side, kind = part.split("_")
        V, F = g[f"{side}_{kind}_v"].astype(float), g[f"{side}_{kind}_f"]
        if side == "fixed":
            maps = [_splat_mesh((V - cam["t"]) @ cam["R"].T, F, cam, shape, scale)]
        else:
            maps = []
            for u in range(101):
                q = max(u * grip_scale + grip_bias, 0.0)
                c, s_ = math.cos(q), math.sin(q)
                rot = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]]) @ np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1.0]])
                Vg = V @ rot.T + JAW_XYZ
                maps.append(_splat_mesh((Vg - cam["t"]) @ cam["R"].T, F, cam, shape, scale))
        out[part] = np.stack(maps).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, key=key, **out)
    return out


class Geometry:
    """Batched geometry for one backend. numpy (default, float64) or torch (`torch_device`, float32)."""

    def __init__(
        self,
        offsets_rad=None,
        camera: str = "physical",
        torch_device: str | None = None,
        tip_local=TIP_LOCAL,
        grip_scale: float | None = None,
        grip_bias: float | None = None,
        samples: int = 7,
        parts=PARTS,
        occlusion: str = "depth",
    ):
        self.offsets = np.asarray(offsets_rad if offsets_rad is not None else default_offsets(), float)
        cam = wrist_camera(camera)
        self.cam = cam
        # jaw mapping: what the real camera sees (JAW_REAL) for the physical camera, the twin's own for camera="sim"
        dflt = JAW_REAL if camera == "physical" else (GRIP_SCALE, GRIP_BIAS)
        grip_scale = dflt[0] if grip_scale is None else grip_scale
        grip_bias = dflt[1] if grip_bias is None else grip_bias
        self.grip_scale, self.grip_bias = grip_scale, grip_bias
        pl = occluder_planes()
        self.parts = tuple(parts)
        fixed = [p for part in self.parts if part.startswith("fixed") for p in pl[part]]
        jaw = [p for part in self.parts if part.startswith("jaw") for p in pl[part]]
        self._part_of = {
            "fixed": [part for part in self.parts if part.startswith("fixed") for _ in pl[part]],
            "jaw": [part for part in self.parts if part.startswith("jaw") for _ in pl[part]],
        }
        P, N, AX, WF = _face_samples(samples)
        arrays = dict(
            T=_urdf_chain(),
            off=self.offsets[:5],
            Rc=cam["R"],
            tc=cam["t"],
            tip=np.r_[tip_local, 1.0],
            fixed=_pad(fixed) if fixed else np.zeros((0, 1, 4)),
            jaw=_pad(jaw) if jaw else np.zeros((0, 1, 4)),
            jxyz=JAW_XYZ,
            sP=P,
            sN=N,
            sW=WF,
            sAX=AX.astype(float),
        )
        if torch_device is None:
            self.xp = np
            self.a = arrays
        else:
            import torch

            self.xp = torch
            self.a = {k: torch.as_tensor(v, dtype=torch.float32, device=torch_device) for k, v in arrays.items()}
        self.K = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1.0]])
        # occlusion: "depth" = per-pixel depth maps of the CAD gripper meshes (exact silhouettes incl. the fin ribs,
        # one lookup per sample; default), "hull" = ray tests against the convex collider pieces (filled fins)
        self.occlusion = occlusion
        if occlusion == "depth":
            dm = occluder_depth_maps(camera, grip_scale, grip_bias, 2, self.parts)
            self.dmaps = {k: (v if self.xp is np else self.xp.as_tensor(v, device=torch_device)) for k, v in dm.items()}

    # ------------------------------------------------------------------ kinematics
    def gripper_pose(self, joints_deg):
        """(..., >=5) joints, our units -> (..., 4, 4) gripper_link pose in the env frame."""
        xp, a = self.xp, self.a
        q = (joints_deg[..., :5] * (math.pi / 180.0)) + a["off"]
        shp = q.shape[:-1]
        c, s = xp.cos(q), xp.sin(q)
        M = None
        for i in range(5):
            Rz = xp.zeros((*shp, 4, 4), **self._kw(q))
            Rz[..., 0, 0], Rz[..., 0, 1], Rz[..., 1, 0], Rz[..., 1, 1] = c[..., i], -s[..., i], s[..., i], c[..., i]
            Rz[..., 2, 2] = 1.0
            Rz[..., 3, 3] = 1.0
            step = a["T"][i] @ Rz
            M = step if M is None else M @ step
        return M

    def _kw(self, like):
        return {} if self.xp is np else {"dtype": like.dtype, "device": like.device}

    def jaw_rot(self, grip_units):
        """(...,) gripper reading -> (..., 3, 3) jaw link -> gripper_link rotation (Rx(pi/2) Rz(q))."""
        xp = self.xp
        q = grip_units * self.grip_scale + self.grip_bias
        q = q.clip(min=0.0) if xp is not np else np.maximum(q, 0.0)
        c, s = xp.cos(q), xp.sin(q)
        z, o = xp.zeros_like(q), xp.ones_like(q)
        # [[1,0,0],[0,0,-1],[0,1,0]] @ [[c,-s,0],[s,c,0],[0,0,1]]
        return xp.stack([xp.stack([c, -s, z], -1), xp.stack([z, z, -o], -1), xp.stack([s, c, z], -1)], -2)

    def camera_pose(self, joints_deg):
        """-> (Rw (..., 3, 3) world->cv rows, C (..., 3) camera centre in env)."""
        M = self.gripper_pose(joints_deg)
        Rg, pg = M[..., :3, :3], M[..., :3, 3]
        Rw = self.a["Rc"] @ _T(self.xp, Rg)
        C = pg + (Rg @ self.a["tc"][:, None])[..., 0]
        return Rw, C

    def tip(self, joints_deg):
        M = self.gripper_pose(joints_deg)
        return (M @ self.a["tip"])[..., :3], M[..., :3, :3]

    def project(self, Rw, C, X):
        """Env points X (..., P, 3) -> pixels (..., P, 2), depth z (..., P) (pinhole, no distortion)."""
        pc = ((X - C[..., None, :])[..., None, :] @ _T(self.xp, Rw)[..., None, :, :])[..., 0, :]
        z = pc[..., 2]
        zz = _clampmin(self.xp, z, 1e-6)
        u = self.cam["fx"] * pc[..., 0] / zz + self.cam["cx"]
        v = self.cam["fy"] * pc[..., 1] / zz + self.cam["cy"]
        return self.xp.stack([u, v], -1), z

    # ------------------------------------------------------------------ visibility
    def visibility(self, joints_deg, brick_pos, brick_R, brick_size, others=None, margin_px: float = 0.0):
        """Visibility of one brick per env in the wrist camera.

        joints_deg (n, 6) measured joints (gripper reading in [..., 5]); brick_pos (n, 3) brick body centre (env);
        brick_R (n, 3, 3) brick rotation (columns = brick axes in env); brick_size (n, 3) full body extents (m);
        others: optional (pos (n, m, 3), R (n, m, 3, 3), size (n, m, 3), valid (n, m) bool) other bricks as occluders.
        Returns a dict of (n,) / (n, 4) arrays:
          vis      visible fraction of the brick's camera-facing projected area (in frame and unoccluded)
          occ_grip fraction hidden by the gripper (fins / bases), in frame;  occ_other  hidden by other bricks
          oof      fraction outside the image;  area_px  projected camera-facing area (px^2, unclipped)
          box_amodal / box_vis  pixel boxes [x0, y0, x1, y1] of all / visible samples (box_vis NaN if none)
          uv (n, 2) brick-centre pixel, z (n,) its depth, dist (n,) camera distance, ang (n,) off-axis angle (deg),
          front (n,) brick centre in front of the camera, occ_part (n, n_parts) per-part hidden fraction.
        """
        xp, a = self.xp, self.a
        Rw, C = self.camera_pose(joints_deg)
        M = self.gripper_pose(joints_deg)
        Rg, pg = M[..., :3, :3], M[..., :3, 3]
        half = brick_size[..., None, :] * 0.5
        X = brick_pos[..., None, :] + ((a["sP"] * half)[..., None, :] @ _T(xp, brick_R)[..., None, :, :])[..., 0, :]
        Nw = (a["sN"][None, :, None, :] @ _T(xp, brick_R)[:, None, :, :])[..., 0, :]
        # cell area: face area * trapezoid weight; face area from the two in-face extents
        sz = brick_size
        face_area = xp.stack([sz[:, 1] * sz[:, 2], sz[:, 0] * sz[:, 2], sz[:, 0] * sz[:, 1]], -1)  # (n, 3)
        ax = a["sAX"].long() if xp is not np else a["sAX"].astype(int)
        dA = face_area[:, ax] * a["sW"]
        # camera coords
        rel = X - C[:, None, :]
        pc = (rel[..., None, :] @ _T(xp, Rw)[:, None, :, :])[..., 0, :]
        z = pc[..., 2]
        facing = (Nw * rel).sum(-1) < 0  # normal towards the camera
        zz = _clampmin(xp, z, 1e-4)
        # projected pixel area: fx fy |n . X| dA / z^3
        w = self.cam["fx"] * self.cam["fy"] * (-(Nw * rel).sum(-1)) * dA / zz**3
        w = xp.where(facing & (z > 1e-3), w, xp.zeros_like(w))
        u = self.cam["fx"] * pc[..., 0] / zz + self.cam["cx"]
        v = self.cam["fy"] * pc[..., 1] / zz + self.cam["cy"]
        inside = (u >= -margin_px) & (u <= W + margin_px) & (v >= -margin_px) & (v <= H + margin_px) & (z > 1e-3)
        # occlusion by gripper parts: test in gripper_link (fixed) and jaw-link (jaw) coordinates
        Cg = a["tc"]  # camera centre in gripper_link (constant)
        Xg = ((X - pg[:, None, :])[..., None, :] @ Rg[:, None, :, :])[..., 0, :]  # R^T (X - p) as row vectors
        n_parts = len(self.parts)
        hid_grip = xp.zeros_like(w, dtype=bool) if xp is np else xp.zeros_like(w, dtype=xp.bool)
        part_idx = {p: i for i, p in enumerate(self.parts)}
        per_part = [hid_grip.clone() if xp is not np else hid_grip.copy() for _ in range(n_parts)]
        if self.occlusion == "depth":
            ui = _clip_int(xp, u / 2.0, 0, W // 2 - 1)
            vi = _clip_int(xp, v / 2.0, 0, H // 2 - 1)
            gb = _clip_int(xp, joints_deg[:, 5] + 0.5, 0, 100)
            for part in self.parts:
                Dm = self.dmaps[part]
                gi = (
                    gb[:, None].expand(ui.shape)
                    if (xp is not np and Dm.shape[0] > 1)
                    else (np.broadcast_to(gb[:, None], ui.shape) if Dm.shape[0] > 1 else 0 * ui)
                )
                dz = Dm[gi, vi, ui]
                per_part[part_idx[part]] = inside & (dz < z - 1e-3)
        elif a["fixed"].shape[0]:
            h = _seg_hits_hulls(
                xp, Cg.expand(Xg.shape[0], 3) if xp is not np else np.broadcast_to(Cg, (Xg.shape[0], 3)), Xg, a["fixed"]
            )  # (n, P, m)
            for k, part in enumerate(self._part_of["fixed"]):
                per_part[part_idx[part]] = per_part[part_idx[part]] | h[..., k]
        if self.occlusion != "depth" and a["jaw"].shape[0]:
            Rj = self.jaw_rot(joints_deg[:, 5])  # jaw -> gripper_link
            Cj = ((Cg - a["jxyz"])[None, None, :] @ Rj)[:, 0, :]  # Rj^T (C - o)
            Xj = (Xg - a["jxyz"]) @ Rj
            h = _seg_hits_hulls(xp, Cj, Xj, a["jaw"])
            for k, part in enumerate(self._part_of["jaw"]):
                per_part[part_idx[part]] = per_part[part_idx[part]] | h[..., k]
        for hp in per_part:
            hid_grip = hid_grip | hp
        hid_other = xp.zeros_like(hid_grip)
        if others is not None:
            opos, oR, osz, ovalid = others
            for m in range(opos.shape[1]):
                hm = _seg_hits_box(xp, C, X, opos[:, m], oR[:, m], osz[:, m])
                hid_other = hid_other | (hm & ovalid[:, m, None])
        total = w.sum(-1)
        tot = _clampmin(xp, total, 1e-9)
        vis_m = inside & ~hid_grip & ~hid_other
        out = {
            "vis": (w * vis_m).sum(-1) / tot,
            "occ_grip": (w * (inside & hid_grip)).sum(-1) / tot,
            "occ_other": (w * (inside & ~hid_grip & hid_other)).sum(-1) / tot,
            "oof": (w * ~inside).sum(-1) / tot,
            "area_px": total,
            "occ_part": xp.stack([(w * (inside & hp)).sum(-1) / tot for hp in per_part], -1)
            if per_part
            else total[:, None] * 0,
        }
        fw = facing & (z > 1e-3)
        out["box_amodal"] = _bbox(xp, u, v, fw)
        out["box_vis"] = _bbox(xp, u, v, vis_m & fw)
        bc = ((brick_pos - C)[:, None, :] @ _T(xp, Rw))[:, 0, :]
        zc = _clampmin(xp, bc[:, 2], 1e-4)
        out["uv"] = xp.stack(
            [self.cam["fx"] * bc[:, 0] / zc + self.cam["cx"], self.cam["fy"] * bc[:, 1] / zc + self.cam["cy"]], -1
        )
        out["z"] = bc[:, 2]
        out["dist"] = _norm(xp, bc)
        out["ang"] = (
            xp.arccos(xp.clip(bc[:, 2] / _clampmin(xp, out["dist"], 1e-9), -1.0, 1.0)) * (180.0 / math.pi)
            if xp is np
            else xp.rad2deg(xp.arccos((bc[:, 2] / out["dist"].clamp(min=1e-9)).clamp(-1.0, 1.0)))
        )
        out["front"] = bc[:, 2] > 0.01
        return out

    # ------------------------------------------------------------------ detector box -> brick pose
    def box_to_rel(self, box, joints_deg, plane_z, centre_z=None, undistort: bool = False):
        """Detector box (n, 4) px -> brick-centre estimate relative to the fingertip in gripper_link (n, 3), and in
        env (n, 3). The box centre's ray is intersected with the horizontal plane z = plane_z (env; fitted on real
        data, see detector_model.json "plane_dz"), then the point is moved to z = centre_z (default plane_z).
        Same function on the robot and in the twin, so the residual sees the same estimator in both."""
        xp = self.xp
        Rw, C = self.camera_pose(joints_deg)
        cx = (box[:, 0] + box[:, 2]) * 0.5
        cy = (box[:, 1] + box[:, 3]) * 0.5
        if undistort and xp is np and self.cam.get("dist") is not None:
            import cv2

            pts = cv2.undistortPoints(
                np.stack([cx, cy], -1)[:, None, :].astype(np.float64), self.K, self.cam["dist"], P=self.K
            )[:, 0]
            cx, cy = pts[:, 0], pts[:, 1]
        d_cv = xp.stack(
            [(cx - self.cam["cx"]) / self.cam["fx"], (cy - self.cam["cy"]) / self.cam["fy"], xp.ones_like(cx)], -1
        )
        d = (d_cv[:, None, :] @ Rw)[:, 0, :]  # Rw^T d
        pz = plane_z if not hasattr(plane_z, "shape") else plane_z
        s = (pz - C[:, 2]) / _signed_clamp(xp, d[:, 2], 1e-6)
        P = C + s[:, None] * d
        if centre_z is not None:
            P = xp.stack([P[:, 0], P[:, 1], P[:, 2] * 0 + centre_z], -1)
        p_tip, Rg = self.tip(joints_deg)
        rel = ((P - p_tip)[:, None, :] @ Rg)[:, 0, :]
        return rel, P


# ---------------------------------------------------------------------------------------------------- helpers
def _T(xp, M):
    return M.swapaxes(-1, -2) if xp is np else M.transpose(-1, -2)


def _clip_int(xp, a, lo, hi):
    if xp is np:
        return np.clip(np.floor(a), lo, hi).astype(int)
    return a.floor().clamp(lo, hi).long()


def _clampmin(xp, a, m):
    return np.maximum(a, m) if xp is np else a.clamp(min=m)


def _signed_clamp(xp, a, m):
    s = xp.where(a >= 0, xp.ones_like(a), -xp.ones_like(a))
    return s * (np.maximum(np.abs(a), m) if xp is np else a.abs().clamp(min=m))


def _norm(xp, a):
    return np.linalg.norm(a, axis=-1) if xp is np else a.norm(dim=-1)


def _bbox(xp, u, v, m):
    big = 1e9
    x0 = xp.where(m, u, xp.full_like(u, big)).min(-1)
    y0 = xp.where(m, v, xp.full_like(v, big)).min(-1)
    x1 = xp.where(m, u, xp.full_like(u, -big)).max(-1)
    y1 = xp.where(m, v, xp.full_like(v, -big)).max(-1)
    if xp is not np:
        x0, y0, x1, y1 = x0.values, y0.values, x1.values, y1.values
    b = xp.stack([x0, y0, x1, y1], -1)
    anym = m.any(-1)
    return xp.where(anym[:, None], b, xp.full_like(b, float("nan")))


def _seg_hits_hulls(xp, C, X, planes):
    """Segments C (n, 3) -> X (n, P, 3) against convex hulls planes (m, k, 4) -> (n, P, m) bool (hit before X)."""
    D = X - C[:, None, :]  # (n, P, 3)
    nrm, d = planes[..., :3], planes[..., 3]  # (m, k, 3), (m, k)
    nc = (C @ nrm.reshape(-1, 3).T).reshape(C.shape[0], 1, *d.shape) + d  # (n, 1, m, k)  n.C + d
    nd = (D.reshape(-1, 3) @ nrm.reshape(-1, 3).T).reshape(*D.shape[:2], *d.shape)  # (n, P, m, k)
    eps = 1e-12
    t = -nc / xp.where(xp.abs(nd) < eps, xp.full_like(nd, eps), nd)
    big = 1e9
    enter = xp.where(nd < -eps, t, xp.full_like(t, -big))
    exit_ = xp.where(nd > eps, t, xp.full_like(t, big))
    parallel_out = (xp.abs(nd) <= eps) & (nc > 0)
    s0 = enter.max(-1)
    s1 = exit_.min(-1)
    if xp is not np:
        s0, s1 = s0.values, s1.values
    s0 = xp.maximum(s0, xp.zeros_like(s0))
    s1 = xp.minimum(s1, xp.full_like(s1, 1.0 - 1e-4))
    return (s0 < s1) & ~parallel_out.any(-1)


def _seg_hits_box(xp, C, X, pos, R, size):
    """Segments C (n, 3) -> X (n, P, 3) against oriented boxes (n,) -> (n, P) bool (slab test, hit before X)."""
    Cl = ((C - pos)[:, None, :] @ R)[:, 0, :]  # R^T (C - pos)
    Xl = (X - pos[:, None, :]) @ R
    D = Xl - Cl[:, None, :]
    h = size[:, None, :] * 0.5
    eps = 1e-12
    Dd = xp.where(xp.abs(D) < eps, xp.full_like(D, eps), D)
    t1 = (-h - Cl[:, None, :]) / Dd
    t2 = (h - Cl[:, None, :]) / Dd
    tmin = xp.minimum(t1, t2).max(-1)
    tmax = xp.maximum(t1, t2).min(-1)
    if xp is not np:
        tmin, tmax = tmin.values, tmax.values
    tmin = xp.maximum(tmin, xp.zeros_like(tmin))
    tmax = xp.minimum(tmax, xp.full_like(tmax, 1.0 - 1e-4))
    return tmin < tmax


def yaw_R(yaw, pose: str = "up", xp=np):
    """Brick rotation (n, 3, 3) for a yaw (rad) about z and a resting pose: up (studs up) | side (lying on a long side,
    studs horizontal) | end (standing on a 2-stud end). Body size stays (L, W, H) in the brick's own axes."""
    yaw = xp.asarray(yaw) if xp is np else yaw
    c, s = xp.cos(yaw), xp.sin(yaw)
    z, o = xp.zeros_like(c), xp.ones_like(c)
    Rz = xp.stack([xp.stack([c, -s, z], -1), xp.stack([s, c, z], -1), xp.stack([z, z, o], -1)], -2)
    if pose == "up":
        return Rz
    if pose == "side":  # brick y (width) -> vertical: rotate +90 deg about the brick x
        Rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]])
    else:  # end: brick x (length) -> vertical
        Rx = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0.0]])
    return Rz @ (Rx if xp is np else xp.as_tensor(Rx, dtype=Rz.dtype, device=Rz.device))


def quat_xyzw_to_R(q):
    """(n, 4) xyzw (Isaac Lab 3.0) -> (n, 3, 3)."""
    xp = _xp(q)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return xp.stack(
        [
            xp.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            xp.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            xp.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )
