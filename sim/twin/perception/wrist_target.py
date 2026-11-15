"""Wrist-camera brick target for the pick residual: real detector boxes -> the target brick's pose relative to the
gripper. numpy only (+ the standard library), shared UNCHANGED by the robot controller and the twin.

Pipeline on the robot, per wrist frame during the pick (gated like the residual, by the pick prompt):
  (a) boxes = OVBrickDetector(brick-detector-v1, "NPU").detect(bgr)            dlsj/inspect/brick_detector.py, ~4 ms
  (b) target = tracker.update(boxes, joints)       reject the fins' and the plates' false positives, track the target
  (c) target.rel = brick centre - fingertip in gripper_link (m), from the box by ray -> table plane (box_to_rel)
The twin's DetectorModel (detector_model.py) produces the target box statistically and goes through the same (c);
(b) is applied to real boxes only (its false-target rate on the recorded data is in the README).

    from sim.twin.perception.wrist_target import WristCamera, TargetTracker
    cam = WristCamera()                                    # wrist mount + URDF + twin joint offsets (real FK)
    trk = TargetTracker(cam)
    trk.reset()                                            # at the start of every pick
    t = trk.update([(x0, y0, x1, y1, conf), ...], joints)  # joints: the 6 measured readings (LeRobot units)
    t.valid, t.rel (3,), t.pos (3,) env xy, t.conf, t.box, t.misses
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
URDF = REPO / "sim/twin/assets/so101/urdf/so101_new_calib.urdf"
WRIST_JSON = REPO / "sim/twin/robot/wrist_mount.json"
FINRAY = REPO / "sim/twin/gripper/assets/finray_so101.npz"
OFFSETS_JSON = REPO / "sim/twin/align/joint_fit/result.json"
CHAIN = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
JAW_XYZ = np.array([0.0202, 0.0188, -0.0234])
TABLE_Z = -0.002
BRICK_CENTRE_Z = TABLE_Z + 0.0096  # assumed: a brick standing studs-up on the table
W, H = 640, 480


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


@lru_cache(maxsize=2)
def _chain(urdf: str):
    joints = {j.get("name"): j for j in ET.parse(urdf).getroot().findall("joint")}
    T = []
    for name in CHAIN:
        o = joints[name].find("origin")
        M = np.eye(4)
        M[:3, :3] = _rpy(*[float(v) for v in o.get("rpy").split()])
        M[:3, 3] = [float(v) for v in o.get("xyz").split()]
        T.append(M)
    return np.stack(T)


class WristCamera:
    """FK + the physical wrist camera (measured intrinsics, fitted mount). All batched over a leading axis."""

    def __init__(
        self,
        offsets_rad=None,
        tip_local=(0.0063, 0.001, -0.0902),  # the fin-ray grasp point (physics apply.tip_local, rl/DEPLOY.md)
        grip_scale: float = 0.02406,
        grip_bias: float = -0.1183,
        urdf: str | Path = URDF,
        wrist_json: str | Path = WRIST_JSON,
        plane_dz: float = 0.004,
        cover_weight: float = 0.0,
    ):
        if offsets_rad is None:
            res = json.loads(Path(OFFSETS_JSON).read_text())  # the twin's default: W9, else fit W
            offsets_rad = (
                res.get("grasp_depth_candidates", {}).get("W9", {}).get("joint_offsets_rad_for_sim")
                or res["joint_offsets_rad_for_sim"]
            )[:5]
        self.off = np.asarray(offsets_rad[:5], float)
        self.T = _chain(str(urdf))
        c = json.loads(Path(wrist_json).read_text())["camera"]
        m = c["intrinsics_measured"]
        self.Rc, self.tc = np.array(c["local_to_cv"]), np.array(c["position_local"])
        self.fx, self.fy, self.cx, self.cy = m["fx"], m["fy"], m["cx"], m["cy"]
        self.tip = np.r_[tip_local, 1.0]
        self.grip_scale, self.grip_bias = grip_scale, grip_bias
        self.cover_weight = cover_weight
        self._fin_cache: dict = {}
        self.plane_dz = plane_dz  # ray-plane height above the brick centre (fitted on real data, detector_model.json)

    def gripper_pose(self, joints):
        j = np.atleast_2d(np.asarray(joints, float))
        q = np.deg2rad(j[:, :5]) + self.off
        M = np.tile(np.eye(4), (len(j), 1, 1))
        for i in range(5):
            Rz = np.tile(np.eye(4), (len(j), 1, 1))
            c, s = np.cos(q[:, i]), np.sin(q[:, i])
            Rz[:, 0, 0], Rz[:, 0, 1], Rz[:, 1, 0], Rz[:, 1, 1] = c, -s, s, c
            M = M @ self.T[i] @ Rz
        return M

    def camera(self, joints):
        """(Rw (n, 3, 3) world -> camera rows, C (n, 3) camera centre, M (n, 4, 4) gripper pose)."""
        M = self.gripper_pose(joints)
        return self.Rc @ M[:, :3, :3].transpose(0, 2, 1), M[:, :3, 3] + M[:, :3, :3] @ self.tc, M

    def project(self, joints, X):
        """Env points X (n, 3) -> pixels (n, 2), depth (n,)."""
        Rw, C, _ = self.camera(joints)
        pc = np.einsum("nij,nj->ni", Rw, np.atleast_2d(X) - C)
        z = np.maximum(pc[:, 2], 1e-6)
        return np.stack([self.fx * pc[:, 0] / z + self.cx, self.fy * pc[:, 1] / z + self.cy], 1), pc[:, 2]

    def box_to_rel(self, boxes, joints, centre_z: float = BRICK_CENTRE_Z):
        """Boxes (n, 4) px -> (rel (n, 3) brick centre - fingertip in gripper_link, pos (n, 3) env, depth (n,)).
        Ray through the box centre intersected with z = centre_z + plane_dz, then moved to z = centre_z.
        Same numbers as geometry.Geometry.box_to_rel (the twin's batched copy)."""
        b = np.atleast_2d(np.asarray(boxes, float))
        Rw, C, M = self.camera(np.broadcast_to(np.atleast_2d(joints), (len(b), np.atleast_2d(joints).shape[1])))
        u, v = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
        d = np.einsum(
            "nji,nj->ni", Rw, np.stack([(u - self.cx) / self.fx, (v - self.cy) / self.fy, np.ones_like(u)], 1)
        )
        dz = np.where(np.abs(d[:, 2]) < 1e-6, -1e-6, d[:, 2])
        s = (centre_z + self.plane_dz - C[:, 2]) / dz
        P = C + s[:, None] * d
        P[:, 2] = centre_z
        tip = (M @ self.tip)[:, :3]
        rel = np.einsum("nji,nj->ni", M[:, :3, :3], P - tip)
        depth = np.einsum("nj,nj->n", Rw[:, 2], P - C)
        return rel, P, np.where(s > 0, depth, -1.0)

    # ------------------------------------------------------------------ fin masks (false positives)
    def _fin_poly_local(self, which: str, grip_bin: int):
        key = (which, grip_bin)
        if key not in self._fin_cache:
            self._fin_cache[key] = self._fin_poly(which, grip_bin)
        return self._fin_cache[key]

    def _fin_poly(self, which: str, grip_bin: int):
        v = _fin_verts(which)
        if which == "jaw":
            q = max((grip_bin + 0.5) * self.grip_scale + self.grip_bias, 0.0)  # real jaw angle (jaw_fit.py)
            c, s = math.cos(q), math.sin(q)
            rot = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]]) @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
            v = v @ rot.T + JAW_XYZ
        pc = (v - self.tc) @ self.Rc.T  # gripper_link -> camera (the fins are rigid in the gripper frame)
        pc = pc[pc[:, 2] > 0.005]
        uv = np.stack([self.fx * pc[:, 0] / pc[:, 2] + self.cx, self.fy * pc[:, 1] / pc[:, 2] + self.cy], 1)
        return _hull(uv)

    def fin_coverage(self, box, grip: float) -> float:
        """max(cover_weight * fraction of the box inside either fin's convex outline, IoU of the box with a fin's image
        box), fixed fin and jaw fin at this reading. The fins' false positives box the whole (clipped) fin, so the IoU
        term catches them; the outline term (off by default) would also reject a held brick between the bent fins."""
        x0, y0, x1, y1 = (float(t) for t in box[:4])
        if x1 <= x0 or y1 <= y0:
            return 0.0
        gx, gy = np.meshgrid(np.linspace(x0, x1, 9), np.linspace(y0, y1, 9))
        pts = np.stack([gx.ravel(), gy.ravel()], 1)
        cov = np.zeros(len(pts), bool)
        best_iou = 0.0
        for which, gb in (("fixed", 0), ("jaw", int(np.clip(grip, 0, 100)))):
            hull = self._fin_poly_local(which, gb)
            cov |= _in_hull(pts, hull)
            if len(hull):  # the detector boxes the whole (image-clipped) fin
                fb = (
                    max(hull[:, 0].min(), 0),
                    max(hull[:, 1].min(), 0),
                    min(hull[:, 0].max(), W),
                    min(hull[:, 1].max(), H),
                )
                ix = max(0.0, min(x1, fb[2]) - max(x0, fb[0])) * max(0.0, min(y1, fb[3]) - max(y0, fb[1]))
                ua = (x1 - x0) * (y1 - y0) + (fb[2] - fb[0]) * (fb[3] - fb[1]) - ix
                best_iou = max(best_iou, ix / max(ua, 1e-9))
        return max(float(cov.mean()) * self.cover_weight, best_iou)


def _hull(uv):
    """Convex hull (counter-clockwise in image coordinates) of points uv (k, 2) (monotone chain)."""
    p = uv[np.lexsort((uv[:, 1], uv[:, 0]))]

    def half(seq):
        h = []
        for q in seq:
            while len(h) >= 2 and np.cross(h[-1] - h[-2], q - h[-2]) <= 0:
                h.pop()
            h.append(q)
        return h

    return np.array(half(p)[:-1] + half(p[::-1])[:-1])


def _in_hull(pts, hull):
    """Points (m, 2) inside the convex polygon hull (k, 2) from _hull."""
    if len(hull) < 3:
        return np.zeros(len(pts), bool)
    e = np.roll(hull, -1, 0) - hull
    rel = pts[:, None, :] - hull[None]
    return (e[None, :, 0] * rel[..., 1] - e[None, :, 1] * rel[..., 0] >= 0).all(1)


@lru_cache(maxsize=2)
def _fin_verts(which: str):
    return np.load(FINRAY)[f"{which}_fin_v"].astype(float)[::3]


@dataclass
class Target:
    valid: bool
    rel: np.ndarray  # (3,) brick centre - fingertip, gripper_link (m); zeros if not valid
    pos: np.ndarray  # (3,) env-frame estimate
    conf: float
    box: tuple | None
    misses: int  # frames since the last accepted box
    tracking: bool  # a target has been acquired in this pick


class TargetTracker:
    """Picks the target brick's box in each wrist frame.

    Rejects: boxes below `conf_min`; boxes mostly covered by a fin outline (`fin_max` of the box; the TPU fins score
    0.5-0.7 as "bricks" in almost every frame); boxes whose size does not fit a Duplo at the depth its ray-plane
    point implies (plates, the robot base, half-image boxes); points behind the camera or > `reach` from the
    fingertip. Then: while tracking, the box nearest the last estimate within a gate that grows with the misses;
    when not tracking (or lost for `lose_after` frames), the box nearest the fingertip within `acquire`."""

    def __init__(
        self,
        cam: WristCamera | None = None,
        conf_min: float = 0.5,
        fin_max: float = 0.4,
        reach: float = 0.15,
        acquire: float = 0.08,
        gate: float = 0.025,
        gate_growth: float = 0.005,
        gate_max: float = 0.06,
        lose_after: int = 15,
        size_lo: float = 0.4,
        size_hi: float = 2.2,
    ):
        self.cam = cam or WristCamera()
        self.conf_min, self.fin_max, self.reach, self.acquire = conf_min, fin_max, reach, acquire
        self.gate, self.gate_growth, self.gate_max, self.lose_after = gate, gate_growth, gate_max, lose_after
        self.size_lo, self.size_hi = size_lo, size_hi
        self.reset()

    def reset(self):
        self.est, self.misses, self.last = None, 0, None

    def candidates(self, boxes, joints):
        """Accepted boxes: list of (box (4,), conf, rel (3,), pos (3,))."""
        b = np.asarray([x for x in boxes if x[4] >= self.conf_min], float).reshape(-1, 5)
        if not len(b):
            return []
        j = np.asarray(joints, float)
        rel, pos, depth = self.cam.box_to_rel(b[:, :4], j[None].repeat(len(b), 0))
        tip = (self.cam.gripper_pose(j[None])[0] @ self.cam.tip)[:3]
        out = []
        for k in range(len(b)):
            if depth[k] <= 0.02 or np.linalg.norm(pos[k, :2] - tip[:2]) > self.reach:
                continue
            side = max(b[k, 2] - b[k, 0], b[k, 3] - b[k, 1])
            lo, hi = self.cam.fx * 0.032 / depth[k], self.cam.fx * 0.075 / depth[k]
            clipped = b[k, 0] <= 1 or b[k, 1] <= 1 or b[k, 2] >= W - 1 or b[k, 3] >= H - 1
            if side > self.size_hi * hi or (side < self.size_lo * lo and not clipped):
                continue
            if self.cam.fin_coverage(b[k], j[5]) > self.fin_max:
                continue
            out.append((b[k, :4], float(b[k, 4]), rel[k], pos[k]))
        return out

    def update(self, boxes, joints) -> Target:
        c = self.candidates(boxes, joints)
        j = np.asarray(joints, float)
        pick = None
        if self.est is not None and self.misses < self.lose_after and c:
            g = min(self.gate + self.gate_growth * self.misses, self.gate_max)
            d = [np.linalg.norm(x[3][:2] - self.est[:2]) for x in c]
            k = int(np.argmin(d))
            pick = c[k] if d[k] <= g else None
        elif c:
            tip = (self.cam.gripper_pose(j[None])[0] @ self.cam.tip)[:3]
            d = [np.linalg.norm(x[3][:2] - tip[:2]) - 0.01 * x[1] for x in c]
            k = int(np.argmin(d))
            pick = c[k] if d[k] <= self.acquire else None
        if pick is None:
            self.misses += 1
            if self.misses >= self.lose_after:
                self.est = None
            return Target(
                False,
                np.zeros(3),
                np.zeros(3) if self.est is None else self.est.copy(),
                0.0,
                None,
                self.misses,
                self.est is not None,
            )
        self.est, self.misses = pick[3].copy(), 0
        return Target(True, pick[2], pick[3], pick[1], tuple(pick[0].tolist()), 0, True)
