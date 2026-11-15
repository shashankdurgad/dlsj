"""Batched SO-101 forward kinematics and a small fingertip IK (torch, GPU), from the twin's URDF.

Units: joints in OUR units (LeRobot degrees of the calibration frame, gripper 0-100); the sim angle is
q = deg2rad(d) + joint_offsets (the calibration profile), exactly as ResidualEnv.to_sim. Positions are in the robot
base frame (= env frame: the robot's root sits at the env origin with identity rotation). The fingertip point is the
residual env's TIP_LOCAL in the gripper_link frame. `sim/twin/rl/check_kin.py` checks fk() against PhysX body poses.

    kin = Kin(offsets_rad, device)
    p, R = kin.fk(joints_deg)                       # (..., 3) tip position, (..., 3, 3) gripper_link rotation
    q2 = kin.ik_tip(joints_deg, p_target)           # pan / lift / elbow moved so the tip reaches p_target
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
URDF = REPO / "sim/twin/assets/so101/urdf/so101_new_calib.urdf"
CHAIN = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
TIP_LOCAL = (0.008, 0.0, -0.090)  # residual env's grasp point (gripper_link frame)


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]]


class Kin:
    def __init__(self, offsets_rad, device="cuda:0", urdf: str | Path = URDF, tip_local=TIP_LOCAL):
        self.device = device
        root = ET.parse(str(urdf)).getroot()
        joints = {j.get("name"): j for j in root.findall("joint")}
        T = []
        for name in CHAIN:
            j = joints[name]
            o = j.find("origin")
            xyz = [float(v) for v in o.get("xyz").split()]
            rpy = [float(v) for v in o.get("rpy").split()]
            M = torch.eye(4)
            M[:3, :3] = torch.tensor(_rpy(*rpy))
            M[:3, 3] = torch.tensor(xyz)
            T.append(M)
            a = [float(v) for v in j.find("axis").get("xyz").split()]
            assert a == [0.0, 0.0, 1.0], f"{name}: only z joint axes supported, got {a}"
        self.T = torch.stack(T).to(device)  # (5, 4, 4) parent -> joint frame at q = 0
        self.off = torch.as_tensor(offsets_rad[:5], dtype=torch.float32, device=device)
        self.tip = torch.tensor([*tip_local, 1.0], device=device)

    def fk(self, joints_deg: torch.Tensor):
        """joints_deg (..., >=5) ours -> tip position (..., 3), gripper_link rotation (..., 3, 3)."""
        q = torch.deg2rad(joints_deg[..., :5]) + self.off
        shp = q.shape[:-1]
        M = torch.eye(4, device=self.device).expand(*shp, 4, 4)
        c, s = torch.cos(q), torch.sin(q)
        for i in range(5):
            Rz = torch.zeros(*shp, 4, 4, device=self.device)
            Rz[..., 0, 0], Rz[..., 0, 1], Rz[..., 1, 0], Rz[..., 1, 1] = c[..., i], -s[..., i], s[..., i], c[..., i]
            Rz[..., 2, 2] = 1.0
            Rz[..., 3, 3] = 1.0
            M = M @ self.T[i] @ Rz
        p = (M @ self.tip)[..., :3]
        return p, M[..., :3, :3]

    def jac3(self, joints_deg: torch.Tensor, eps: float = 0.2):
        """d tip / d (pan, lift, elbow) in m per degree, (..., 3, 3) (finite differences)."""
        p0, _ = self.fk(joints_deg)
        cols = []
        for i in range(3):
            d = joints_deg.clone()
            d[..., i] += eps
            cols.append((self.fk(d)[0] - p0) / eps)
        return torch.stack(cols, -1), p0

    def ik_tip(self, joints_deg: torch.Tensor, p_target: torch.Tensor, iters: int = 2, damp: float = 4e-4,
               max_step_deg: float = 20.0) -> torch.Tensor:
        """Move pan / lift / elbow (wrist joints and gripper unchanged) so that fk(tip) -> p_target.
        Damped least squares, `iters` Gauss-Newton steps."""
        q = joints_deg.clone()
        eye = torch.eye(3, device=self.device)
        for _ in range(iters):
            J, p = self.jac3(q)
            e = (p_target - p).unsqueeze(-1)
            JJt = J @ J.transpose(-1, -2) + (damp ** 2) * eye
            dq = (J.transpose(-1, -2) @ torch.linalg.solve(JJt, e)).squeeze(-1)
            q[..., :3] = q[..., :3] + dq.clamp(-max_step_deg, max_step_deg)
        return q
