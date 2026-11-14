"""Numpy forward kinematics of Joe's SO-101 USD (from dump_kinematics.py's kin.json). Column-vector 4x4 transforms."""
import json
import numpy as np

CHAIN = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]  # base -> gripper_link
JOINTS = CHAIN + ["gripper"]


def quat_R(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def T(pos, rot):
    m = np.eye(4); m[:3, :3] = quat_R(*rot); m[:3, 3] = pos
    return m


def axis_rot(axis, q):
    c, s = np.cos(q), np.sin(q)
    m = np.eye(4)
    if axis == "X": m[1:3, 1:3] = [[c, -s], [s, c]]
    elif axis == "Y": m[0, 0], m[0, 2], m[2, 0], m[2, 2] = c, s, -s, c
    else: m[0:2, 0:2] = [[c, -s], [s, c]]
    return m


class SO101FK:
    def __init__(self, kin_path):
        k = json.load(open(kin_path))
        self.k = k
        self.base = np.array(k["base_world"]).T  # USD row-vector -> column
        self.j = k["joints"]
        self.T0 = {n: T(self.j[n]["pos0"], self.j[n]["rot0"]) for n in JOINTS}
        self.T1i = {n: np.linalg.inv(T(self.j[n]["pos1"], self.j[n]["rot1"])) for n in JOINTS}
        self.fixed_pts = np.c_[np.array(k["fixed_finger_pts_gripper"]), np.ones(len(k["fixed_finger_pts_gripper"]))]
        self.jaw_pts = np.c_[np.array(k["jaw_pts_jaw"]), np.ones(len(k["jaw_pts_jaw"]))]
        self.tip = np.r_[k["tip_local_gripper"], 1.0]

    def links(self, q):
        """q: radians for the 6 joints (shoulder_pan..gripper) -> world transforms of gripper_link and moving jaw."""
        M = self.base.copy()
        for i, n in enumerate(CHAIN):
            M = M @ self.T0[n] @ axis_rot(self.j[n]["axis"], q[i]) @ self.T1i[n]
        G = M
        J = G @ self.T0["gripper"] @ axis_rot(self.j["gripper"]["axis"], q[5]) @ self.T1i["gripper"]
        return G, J

    def points(self, q):
        G, J = self.links(q)
        fingers = np.r_[(self.fixed_pts @ G.T)[:, :3], (self.jaw_pts @ J.T)[:, :3]]
        return fingers, (G @ self.tip)[:3]
