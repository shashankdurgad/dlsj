"""The SO-101 wrist camera and its printed mount on the twin, in the gripper_link frame.

Pose (wrist_mount.json, written by fit_wrist_mount.py): the camera's optical centre and axes in gripper_link
(OpenCV axes: x right, y down, z = optical axis), fitted to evidence that is rigid in gripper_link (the fixed fin-ray
finger's silhouette in the real wrist frames, held bricks, the top camera's housing landmark, phone photos) with the
measured intrinsics (fx 343, cx 339). Two rotations are stored:
* local_to_cv:      the physical camera (measured principal point (339, 236));
* local_to_cv_sim:  the twin's camera model (centred pinhole, one focal), rotated so that it reproduces the physical
                    camera's pixels (a principal-point offset is, to < 5 px, a small camera rotation).

Mount visual (mount_parts): the real mount is a flat orange printed frame holding the 32x32 UVC board (4 M2 screws),
lens towards the fingertips, with a bent neck screwed by 2 M3 screws into the two holes on the +y face of the
wrist-roll follower (the hex-nut recesses, 8.2 mm apart at z = -23.4 mm). The camera module is the official
wrist_camera_so101_v1.stl (TheRobotStudio, board + M12 lens, fetched with the URDF). The frame is built around the
camera pose it is given and the neck is re-anchored to the screw holes, so the mount stays attached under camera DR.

    from sim.twin.robot.wrist_mount import cams_entry, load, wrist_camera
    R, p, f = cams_entry()          # drop-in for calib/profile.py _cams()["wrist"] (sim rotation, position, focal)
    cam = wrist_camera()            # residual-env Camera (USD rows) for Profile.wrist
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
JSON = HERE / "wrist_mount.json"
CAMERA_STL = REPO / "sim/twin/assets/so101/urdf/assets/wrist_camera_so101_v1.stl"
LENS_Z = 0.024  # optical centre on the camera module's +z axis (m; lens front at 0.0263, PCB at 0)
# M3 holes on the follower's +y face (gripper_link, m): the neck tab's screws
SCREW_HOLES = np.array([[-0.0050, 0.0228, -0.0234], [0.0032, 0.0228, -0.0234]])


@dataclass
class WristMount:
    R_cv: np.ndarray  # (3, 3) rows = physical camera axes in gripper_link
    R_sim: np.ndarray  # (3, 3) rows = centred-pinhole equivalent (the twin's camera model)
    t: np.ndarray  # (3,) optical centre in gripper_link (m)
    focal_px: float  # measured wrist focal (mean of fx, fy)
    dims: dict  # mount dimensions (m), see wrist_mount.json "mount"


def load(path: str | Path = JSON) -> WristMount:
    """Read wrist_mount.json."""
    d = json.loads(Path(path).read_text())
    c = d["camera"]
    return WristMount(np.array(c["local_to_cv"]), np.array(c["local_to_cv_sim"]), np.array(c["position_local"]),
                      float(c["focal_px"]), d["mount"])


def cams_entry(m: WristMount | None = None) -> tuple:
    """(local_to_cv, position, focal_px) in the format of calib/profile.py `_cams()["wrist"]` (sim rotation)."""
    m = m or load()
    return m.R_sim, m.t, m.focal_px


def wrist_camera(m: WristMount | None = None):
    """The wrist camera as a residual-env Camera (USD row matrix in gripper_link + pinhole intrinsics)."""
    sys.path.insert(0, str(REPO / "sim/residual_env"))
    from calib import cv_to_usd_rows

    return cv_to_usd_rows(*cams_entry(m))


def rows_to_cv(rows) -> tuple[np.ndarray, np.ndarray]:
    """USD camera rows (calib.Camera.rows) -> (OpenCV rows R_cv, position)."""
    r = np.asarray(rows, float)
    return np.stack([r[0, :3], -r[1, :3], -r[2, :3]]), r[3, :3].copy()


# ---------------------------------------------------------------------------------------------------- mesh helpers
def _box(center, axes, size) -> tuple:
    """Oriented box: axes = (3, 3) columns = box x / y / z in the parent frame. Flat-shaded (pts, nrm, counts, idx)."""
    h = np.asarray(size, float) / 2
    ax = np.asarray(axes, float)
    pts, nrm = [], []
    for k in range(3):
        for s in (-1.0, 1.0):
            n = np.zeros(3)
            n[k] = s
            u, v = [i for i in range(3) if i != k]
            quad = []
            for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = n * h
                p[u], p[v] = a * h[u], b * h[v]
                quad.append(p)
            if np.dot(np.cross(quad[1] - quad[0], quad[2] - quad[0]), n) < 0:
                quad = quad[::-1]
            pts += quad
            nrm += [n] * 4
    pts = np.asarray(pts) @ ax.T + center
    nrm = np.asarray(nrm) @ ax.T
    return pts, nrm, np.full(6, 4), np.arange(24)


def _merge(meshes) -> tuple:
    pts, nrm, cnt, idx, off = [], [], [], [], 0
    for p, n, c, i in meshes:
        pts.append(p)
        nrm.append(n)
        cnt.append(c)
        idx.append(np.asarray(i) + off)
        off += len(p)
    return np.concatenate(pts), np.concatenate(nrm), np.concatenate(cnt), np.concatenate(idx)


def _cylinder(p0, p1, r, seg=12) -> tuple:
    """Open tube from p0 to p1 (smooth side normals)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    a = p1 - p0
    a /= np.linalg.norm(a)
    u = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(a, u)
    ang = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    ring = np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * v
    pts = np.concatenate([p0 + r * ring, p1 + r * ring])
    nrm = np.concatenate([ring, ring])
    quads = [[k, (k + 1) % seg, seg + (k + 1) % seg, seg + k] for k in range(seg)]
    return pts, nrm, np.full(seg, 4), np.ravel(quads)


def _stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Binary STL -> (triangles (n, 3, 3), face normals (n, 3)), no dependencies."""
    b = path.read_bytes()
    n = int(np.frombuffer(b, np.uint32, 1, 80)[0])
    rec = np.frombuffer(b, np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), n, 84)
    tri = rec["v"].astype(np.float64)
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    return tri, nrm


def camera_module_mesh(R_cv, t) -> tuple:
    """The official 32x32 UVC module mesh (metres, lens on +z) placed with its optical centre at t (gripper_link)."""
    tri, n = _stl(CAMERA_STL)
    Rm = np.asarray(R_cv, float).T  # module axes (= OpenCV camera axes) as columns
    org = np.asarray(t, float) - Rm @ [0, 0, LENS_Z]
    pts = tri.reshape(-1, 3) @ Rm.T + org
    nrm = np.repeat(n @ Rm.T, 3, axis=0)
    return pts, nrm, np.full(len(tri), 3), np.arange(len(pts))


def mount_parts(R_cv, t, dims: dict | None = None) -> dict:
    """Visual meshes of the real wrist-camera mount in gripper_link: {"plate": orange printed frame + bent neck + screw
    tab, "screws": black screw heads, "board": camera module, "cable": USB lead}. Each value = (pts, nrm, counts, idx).
    R_cv / t: the physical camera (rows = OpenCV axes in gripper_link, optical centre)."""
    d = dims or load().dims
    t = np.asarray(t, float)
    ex, ey, ez = np.asarray(R_cv, float)  # image right, image down (towards the gripper body), optical axis
    th, big, win = d["thickness"], d["frame"], d["window"]
    rim = (big - win) / 2
    axes = np.stack([ex, ey, ez], 1)
    back = t - ez * LENS_Z  # PCB back plane centre; the frame lies behind it
    c_frame = back - ez * th / 2
    plate = [_box(c_frame + axes @ off, axes, size) for off, size in (
        ((0, -(win + rim) / 2, 0), (big, rim, th)), ((0, (win + rim) / 2, 0), (big, rim, th)),
        ((-(win + rim) / 2, 0, 0), (rim, win, th)), (((win + rim) / 2, 0, 0), (rim, win, th)))]
    # screw tab flat on the follower's +y face over the two M3 holes; neck from the frame's lower edge to its foot
    face_y = SCREW_HOLES[0, 1]
    tab_c = np.r_[SCREW_HOLES[:, 0].mean(), face_y + th / 2, SCREW_HOLES[0, 2]]
    plate.append(_box(tab_c, np.eye(3)[:, [0, 2, 1]], (d["neck_width"], d["tab_length"], th)))
    e0 = c_frame + ey * (big / 2 - rim / 2)
    e1 = tab_c - np.r_[0, 0, d["tab_length"] / 2 - th / 2]  # the tab rises from the neck (inspect camera)
    run = e1 - e0
    ln = np.linalg.norm(run)
    a = run / ln
    wdir = ex - a * (ex @ a)
    wdir /= np.linalg.norm(wdir)
    plate.append(_box((e0 + e1) / 2, np.stack([wdir, a, np.cross(wdir, a)], 1), (d["neck_width"], ln + th, th)))
    r_head, h_head = 0.0027, 0.0018
    screws = [_cylinder(h + [0, face_y + th - h[1], 0], h + [0, face_y + th + h_head - h[1], 0], r_head)
              for h in SCREW_HOLES]
    for sx in (-1, 1):
        for sy in (-1, 1):
            p = c_frame + d["board_hole"] / 2 * (sx * ex + sy * ey)
            screws.append(_cylinder(p - ez * th / 2, p - ez * (th / 2 + h_head), 0.002))
    conn = back - ez * (th + 0.002) + ey * d["connector_offset"]
    cable = [_cylinder(back + ey * d["connector_offset"], conn, 0.0022),
             _cylinder(conn, np.asarray(d["cable_to"], float), 0.0022)]
    return {"plate": _merge(plate), "screws": _merge(screws), "board": camera_module_mesh(R_cv, t),
            "cable": _merge(cable)}
