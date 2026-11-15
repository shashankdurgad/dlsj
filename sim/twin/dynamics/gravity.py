"""Batched static gravity torque and fingertip position of the SO-101 (numpy or torch), straight from the URDF.

    from sim.twin.dynamics.gravity import Arm
    arm = Arm()                                   # parses sim/twin/assets/so101/urdf/so101_new_calib.urdf (no meshes)
    tau = arm.gravity_torque(q)                   # q (..., 6) sim rad -> (..., 5) N m about pan..wrist_roll axes
    tip = arm.point(q, "gripper_link", TIP_LOCAL) # (..., 3) m in the robot base frame

Same convention as align/joint_fit/fk.py (torque sign along the joint axis, i.e. the direction gravity drives the
joint angle) and the same wrist-camera mass (20 g on gripper_link), checked against it in ident.py. Works on numpy
arrays or torch tensors (any device), so the play model can run it per tick inside the GPU envs.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
URDF = REPO / "sim/twin/assets/so101/urdf/so101_new_calib.urdf"
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
CAM_MASS = ("gripper_link", 0.02, (0.0, 0.06, -0.04))  # = align/joint_fit/model.py CAM_MASS
TIP_LOCAL = (0.008, 0.0, -0.090)  # residual_env TIP_LOCAL: grasp point in gripper_link
G = (0.0, 0.0, -9.81)


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _origin(el):
    T = np.eye(4)
    if el is not None:
        T[:3, :3] = _rpy(*[float(v) for v in el.get("rpy", "0 0 0").split()])
        T[:3, 3] = [float(v) for v in el.get("xyz", "0 0 0").split()]
    return T


@lru_cache(maxsize=2)
def _parse(urdf: str):
    root = ET.parse(urdf).getroot()
    joints = {}
    for j in root.findall("joint"):
        ax = j.find("axis")
        joints[j.get("name")] = dict(parent=j.find("parent").get("link"), child=j.find("child").get("link"),
                                     T=_origin(j.find("origin")), type=j.get("type"),
                                     axis=np.array([float(v) for v in
                                                    (ax.get("xyz") if ax is not None else "0 0 0").split()]))
    inertial = {}
    for ln in root.findall("link"):
        ine = ln.find("inertial")
        if ine is not None and float(ine.find("mass").get("value")) > 1e-6:
            inertial[ln.get("name")] = (float(ine.find("mass").get("value")), _origin(ine.find("origin"))[:3, 3])
    return joints, inertial


class Arm:
    """Serial chain base -> ... -> gripper_link (+ moving jaw) with URDF masses, batched over leading dims."""

    def __init__(self, urdf: Path | str = URDF, camera_mass: bool = True):
        self.joints, inertial = _parse(str(urdf))
        self.masses = dict(inertial)
        if camera_mass:
            ln, m, c = CAM_MASS
            m0, c0 = self.masses[ln]  # lump the camera into gripper_link (combined COM)
            self.masses[ln] = (m0 + m, (m0 * c0 + m * np.asarray(c)) / (m0 + m))
        self.by_child = {j["child"]: (n, j) for n, j in self.joints.items()}
        self.order = []  # links in topological order, with their parent joint
        frontier = ["base_link"]
        while frontier:
            p = frontier.pop(0)
            for n, j in self.joints.items():
                if j["parent"] == p:
                    self.order.append((j["child"], n))
                    frontier.append(j["child"])

    @staticmethod
    def _rot(axis, q, xp):
        """Rodrigues for a fixed unit axis and batched angles q (...,) -> (..., 3, 3)."""
        x, y, z = (float(a) for a in axis)
        c, s = xp.cos(q), xp.sin(q)
        C = 1 - c
        rows = [[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                [z * x * C - y * s, z * y * C + x * s, c + z * z * C]]
        return xp.stack([xp.stack(r, -1) for r in rows], -2)

    def fk(self, q):
        """q (..., 6) sim rad -> {link: (R (...,3,3), p (...,3))} in the base frame."""
        xp = _xp(q)
        lead = q.shape[:-1]
        eye = _const(np.eye(3), q).expand(*lead, 3, 3) if xp is not np else np.broadcast_to(np.eye(3), (*lead, 3, 3))
        zero = _const(np.zeros(3), q).expand(*lead, 3) if xp is not np else np.zeros((*lead, 3))
        out = {"base_link": (eye, zero)}
        for child, jn in self.order:
            j = self.joints[jn]
            Rp, pp = out[j["parent"]]
            Rj, pj = _const(j["T"][:3, :3], q), _const(j["T"][:3, 3], q)
            R = Rp @ Rj
            p = pp + (Rp @ pj[..., None])[..., 0]
            if j["type"] == "revolute":
                R = R @ self._rot(j["axis"], q[..., JOINTS.index(jn)], xp)
            out[child] = (R, p)
        return out

    def point(self, q, link, local):
        R, p = self.fk(q)[link]
        return p + (R @ _const(np.asarray(local, float), q)[..., None])[..., 0]

    def gravity_torque(self, q):
        """(..., 5) N m: torque of gravity on everything distal to each arm joint, about its axis."""
        xp = _xp(q)
        T = self.fk(q)
        g = _const(np.asarray(G), q)
        com = {ln: T[ln][1] + (T[ln][0] @ _const(c, q)[..., None])[..., 0] for ln, (_, c) in self.masses.items()}
        out = []
        for jn in JOINTS[:5]:
            j = self.joints[jn]
            R, pj = T[j["child"]]
            axis = (R @ _const(j["axis"], q)[..., None])[..., 0]
            tot = None
            for ln in self._subtree(j["child"]):
                if ln in self.masses:
                    m = self.masses[ln][0]
                    t = (axis * xp_cross(com[ln] - pj, m * g, xp)).sum(-1)
                    tot = t if tot is None else tot + t
            out.append(tot)
        return xp.stack(out, -1) if xp is not np else np.stack(np.broadcast_arrays(*out), -1)

    @lru_cache(maxsize=16)  # noqa: B019  (one Arm per process)
    def _subtree(self, link):
        out = [link]
        for j in self.joints.values():
            if j["parent"] == link:
                out += self._subtree(j["child"])
        return tuple(out)


def _xp(a):
    if isinstance(a, np.ndarray):
        return np
    import torch

    return torch


def _const(a, like):
    if isinstance(like, np.ndarray):
        return np.asarray(a, float)
    import torch

    return torch.as_tensor(np.asarray(a, float), dtype=like.dtype, device=like.device)


def xp_cross(a, b, xp):
    if xp is np:
        return np.cross(a, b)
    b = b.expand_as(a)
    return xp.linalg.cross(a, b)


def sim_q(state_deg, offsets_rad, grip_scale: float = 0.0, grip_bias: float = 0.0):
    """LeRobot state (deg, gripper units) -> sim rad, as env.py / align/joint_fit/fk.py sim_q (numpy)."""
    s = np.asarray(state_deg, float)
    q = np.empty(s.shape, float)
    q[..., :5] = np.deg2rad(s[..., :5]) + np.asarray(offsets_rad[:5])
    q[..., 5] = s[..., 5] * grip_scale + grip_bias
    return q
