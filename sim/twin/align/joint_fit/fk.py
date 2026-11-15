"""Fast numpy forward kinematics of the SO-101 from the URDF (the same file the twin's USD was converted from).

    from fk import SO101
    arm = SO101()                       # parses urdf/so101_new_calib.urdf, loads + decimates the STL meshes
    T = arm.fk(q6)                      # {link: 4x4 pose in the robot-base frame}, q6 = sim joint angles (rad)
    V = arm.points(q6, links)           # surface vertices of `links` in the base frame

q6 is in the sim's joint convention, i.e. q = deg2rad(state[:5]) + profile.joint_offsets[:5] and
gripper q = units * grip_scale + grip_bias (export_points.py / sim/residual_env/env.py).
Verified against Isaac's own link poses (verify_fk.py): max point error < 1e-5 m.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
URDF = REPO / "sim/twin/assets/so101/urdf/so101_new_calib.urdf"
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
LINKS = ("base_link", "shoulder_link", "upper_arm_link", "lower_arm_link", "wrist_link", "gripper_link",
         "moving_jaw_so101_v1_link")
MOVING = LINKS[2:]


def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _origin(el) -> np.ndarray:
    T = np.eye(4)
    o = el.find("origin") if el is not None else None
    if o is not None:
        T[:3, :3] = rpy_to_R(*map(float, o.get("rpy", "0 0 0").split()))
        T[:3, 3] = list(map(float, o.get("xyz", "0 0 0").split()))
    return T


def axis_rot(axis, q) -> np.ndarray:
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    T = np.eye(4)
    T[:3, :3] = np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * K @ K
    return T


@lru_cache(maxsize=64)
def _mesh(path: str, scale: float, grid: float):
    """STL vertices / faces (m); grid > 0: vertex-clustering decimation on a `grid`-metre lattice (silhouette error
    <= grid; keeps the outline, unlike edge-collapse which stalls at ~1/3 on these CAD meshes)."""
    import trimesh
    m = trimesh.load(path, force="mesh")
    v, f = np.asarray(m.vertices, float) * scale, np.asarray(m.faces, np.int64)
    if grid > 0:
        cell = np.floor(v / grid).astype(np.int64)
        _, inv = np.unique(cell, axis=0, return_inverse=True)
        inv = inv.ravel()
        nv = inv.max() + 1
        cnt = np.bincount(inv, minlength=nv)
        v2 = np.stack([np.bincount(inv, v[:, k], nv) for k in range(3)], 1) / cnt[:, None]
        f2 = inv[f]
        f2 = f2[(f2[:, 0] != f2[:, 1]) & (f2[:, 1] != f2[:, 2]) & (f2[:, 0] != f2[:, 2])]
        f2 = np.unique(np.sort(f2, 1), axis=0)
        v, f = v2, f2
    return v, f


class SO101:
    def __init__(self, urdf: Path = URDF, grid: float = 0.002, grid_fine: float = 0.0007):
        root = ET.parse(urdf).getroot()
        self.joints = {}
        for j in root.findall("joint"):
            self.joints[j.get("name")] = dict(
                parent=j.find("parent").get("link"), child=j.find("child").get("link"), T=_origin(j),
                axis=list(map(float, j.find("axis").get("xyz").split())) if j.find("axis") is not None else [0, 0, 1],
                type=j.get("type"),
                limit=(float(j.find("limit").get("lower")), float(j.find("limit").get("upper")))
                if j.find("limit") is not None else None)
        # parts[link] = [(stl name, V, F)] decimated on `grid`; parts_fine on `grid_fine` (close-up wrist view)
        self.meshes, self.meshes_full, self.parts, self.parts_fine = {}, {}, {}, {}
        self.inertial = {}  # link -> (mass kg, COM in link frame)
        for link in root.findall("link"):
            n = link.get("name")
            ine = link.find("inertial")
            if ine is not None and float(ine.find("mass").get("value")) > 1e-6:
                self.inertial[n] = (float(ine.find("mass").get("value")), _origin(ine)[:3, 3])
            Vs, Fs, Vf = [], [], []
            for vis in link.findall("visual"):
                me = vis.find("geometry/mesh")
                if me is None:
                    continue
                sc = float(me.get("scale", "1 1 1").split()[0])
                T = _origin(vis)
                p = str(urdf.parent / me.get("filename"))
                v, f = _mesh(p, sc, grid)
                Fs.append(f + sum(len(x) for x in Vs))
                Vs.append(v @ T[:3, :3].T + T[:3, 3])
                self.parts.setdefault(n, []).append((Path(p).stem, Vs[-1], f))
                vq, fq = _mesh(p, sc, grid_fine)
                self.parts_fine.setdefault(n, []).append((Path(p).stem, vq @ T[:3, :3].T + T[:3, 3], fq))
                vf, _ = _mesh(p, sc, 0.0)
                Vf.append(vf @ T[:3, :3].T + T[:3, 3])
            if Vs:
                self.meshes[n] = (np.concatenate(Vs), np.concatenate(Fs))
                self.meshes_full[n] = np.concatenate(Vf)
        self.limits = np.array([self.joints[j]["limit"] for j in JOINTS])  # (6, 2) rad
        self.chain = {}  # child link -> joint name
        for jn, j in self.joints.items():
            self.chain[j["child"]] = jn

    def fk(self, q6) -> dict:
        q = dict(zip(JOINTS, np.asarray(q6, float), strict=True))
        out = {"base_link": np.eye(4)}

        def pose(link):
            if link in out:
                return out[link]
            j = self.joints[self.chain[link]]
            T = pose(j["parent"]) @ j["T"]
            if j["type"] == "revolute":
                T = T @ axis_rot(j["axis"], q[self.chain[link]])
            out[link] = T
            return T
        for n in list(self.meshes) + ["gripper_frame_link"]:
            pose(n)
        return out

    def gravity_torque(self, q6, extra=None) -> np.ndarray:
        """Static gravity torque (N m) about each of the 5 arm joint axes (sign: along the joint axis) at q6, from the
        URDF link masses / COMs (+ `extra` {link: (mass, com_local)}, e.g. the wrist camera)."""
        T = self.fk(q6)
        # extra masses are ADDED to the link's own (e.g. the camera on gripper_link), not replacing it
        masses = [(ln, m, c) for ln, (m, c) in self.inertial.items()] + \
                 [(ln, m, c) for ln, (m, c) in (extra or {}).items()]
        w = np.array([0.0, 0.0, -9.81])
        tau = np.zeros(5)
        for k, jn in enumerate(JOINTS[:5]):
            j = self.joints[jn]
            Tj = T[j["child"]]  # joint frame = child link frame
            axis, pj = Tj[:3, :3] @ np.asarray(j["axis"]), Tj[:3, 3]
            sub = set(self._subtree(j["child"]))
            for ln, m, c in masses:
                if ln in sub:
                    cw = T[ln][:3, :3] @ c + T[ln][:3, 3]
                    tau[k] += axis @ np.cross(cw - pj, m * w)
        return tau

    def _subtree(self, link):
        out = [link]
        for j in self.joints.values():
            if j["parent"] == link:
                out += self._subtree(j["child"])
        return out

    def points(self, q6, links=LINKS, full=False):
        T = self.fk(q6)
        P, L = [], []
        for i, n in enumerate(links):
            v = self.meshes_full[n] if full else self.meshes[n][0]
            P.append(v @ T[n][:3, :3].T + T[n][:3, 3])
            L.append(np.full(len(v), i))
        return np.concatenate(P), np.concatenate(L)


def sim_q(state, offsets, grip_scale, grip_bias):
    """Recorded LeRobot state (deg, gripper 0-100) -> sim joint angles (rad), as export_points.py / env.py."""
    s = np.asarray(state, float)
    return np.r_[np.deg2rad(s[..., :5]) + np.asarray(offsets[:5]), s[..., 5] * grip_scale + grip_bias]
