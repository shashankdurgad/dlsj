"""Real-episode scene layouts from the first top-camera frame (plain python; no Isaac).

    from sim.twin.scene.layout import layout_from_episode
    lay = layout_from_episode(18)                      # canonical cal2 dataset, cached in sim/twin/out/scene/layouts/
    env.reset(layout=lay)                              # TwinEnv: plates, target, distractors, start pose, bowl

    python sim/twin/scene/layout.py detect --episodes 0 18 76 --debug   # layouts (+ overlay jpgs)
    python sim/twin/scene/layout.py stats --every 3               # brick kind / colour statistics -> scene_stats.json

Layout dict (env frame, metres; the robot base is the origin):
    {"episode", "dataset", "bowl": target plate colour (episode label),
     "plates": {colour: {"x", "y", "diam", "score", "visible"}},  only the plates seen in the frame
     "target": {"color", "kind", "x", "y", "yaw", "source"},       kind '2x2' | '2x4' | 'curved_2x3'
     "distractors": [{"color", "kind", "x", "y", "yaw", "on_plate"}],
     "start_state": 6 floats (LeRobot units), "grasp_frame", "tip_xy_at_grasp"}

Plate detector (robust version of visuals/measure_real.detect_layout, which mistook blue objects / the inspection
tripod's blue feet for the purple plate):
* colour mask per plate (the lavender plate reads sky blue in the top camera), minus the known props (tripod feet
  discs, the strip beyond the table edge, the robot base);
* every blob's outer contour (image-border points dropped) is back-projected to the rim plane and a circle of the
  plate radius (92 +- 8 mm) is fitted by RANSAC; a candidate needs enough inlier contour (arc coverage) and a filled
  area consistent with a plate; the best candidate per colour wins, overlapping plates are resolved by score;
* the episode label (target plate colour, posttrain_v3_1 / cal2 labels) is used to disambiguate: the labelled plate
  must exist, so it is searched again with relaxed thresholds if missing.
Bricks: saturated blobs per colour class (palette of visuals/twin_scene.BRICK_RGB), back-projected to the brick-top
plane; minAreaRect sides -> kind by the side ratio after removing the ~4 mm the visible side faces add (2x2 < 1.28
< curved 2x3 < 1.75 < 2x4; the two modes of the measured long-side histogram over 152 episodes are 36-44 mm and 56-64
mm), light blue is always a 2x2 (station inventory). The target is the brick nearest the fingertip (FK,
sim/twin/rl/kin.py) at the frame the gripper closes.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
DATA = REPO / "sim/twin/data/canonical/duplo_teleop_cal2_rgb_456"
POST = REPO / "sim/twin/data/posttrain_v3_1"
CACHE = REPO / "sim/twin/out/scene/layouts"
STATS_JSON = Path(__file__).resolve().parent / "scene_stats.json"
TABLE_Z = -0.002
RIM_Z = 0.021  # outer rim seen from the top camera (plate rim 23 mm, lip rounded)
PLATE_R = 0.092
BRICK_TOP = 0.0192
SIDE_INFLATION = 0.004  # m: the top view adds the visible side faces to each rectangle side (short side 2x2 ~36 mm)
KIND_RATIO = (1.28, 1.75)  # (long - infl) / (short - infl): 2x2 ~1.1 | curved 2x3 ~1.5 | 2x4 ~2.0
# station inventory (photos, frames): the light-blue Duplo is a 2x2; red is a 2x4 (or 2x2), never curved
COLOR_KINDS = {"light_blue": ("2x2",), "red": ("2x4", "2x2")}

PLATE_HSV = {  # OpenCV HSV (H 0-180) as the top camera sees them
    "orange": ((3, 150, 110), (17, 255, 255)),
    "yellow": ((25, 70, 110), (48, 255, 255)),
    "purple": ((90, 70, 110), (118, 255, 255)),  # lavender plate: sky blue in the top camera
}
BRICK_HSV = {
    "light_blue": [((92, 120, 100), (112, 255, 255))],
    "yellow": [((22, 100, 120), (50, 255, 255))],  # reads lime in the top camera
    "orange": [((7, 170, 140), (19, 255, 255))],
    "red": [((0, 150, 60), (6, 255, 255)), ((170, 150, 60), (180, 255, 255))],
    "brown": [((2, 110, 40), (20, 255, 150))],
    "cream": [((12, 30, 185), (95, 85, 255))],  # cream reads pale green-grey in the top camera
}
# brick colours a plate's own pixels (rim glare included) also match: inside that plate they need extra contrast
SAME_AS_PLATE = {"purple": ("light_blue",), "orange": ("orange", "red"), "yellow": ("yellow",)}


# ------------------------------------------------------------------------------------------------ cameras
def top_camera():
    """Real top camera: joe_cal2 pose + factory intrinsics (fx, fy, cx 304, cy) -> (R_world_to_cv, pos, K)."""
    ref = json.loads((REPO / "sim/residual_env/assets/joe_cal2/reference-cal2.json").read_text())["camera"]
    k = json.loads((REPO / "sim/twin/calib/realsense_intrinsics.json").read_text())["top"]
    K = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1.0]])
    return np.array(ref["world_to_cv"], float), np.array(ref["position"], float), K


def project(P, cam):
    R, pos, K = cam
    pc = (np.atleast_2d(P) - pos) @ R.T
    return (pc[:, :2] / pc[:, 2:3]) @ K[:2, :2].T + K[:2, 2], pc[:, 2]


def backproject(uv, cam, z):
    R, pos, K = cam
    uv = np.atleast_2d(np.asarray(uv, float))
    d = np.c_[(uv[:, 0] - K[0, 2]) / K[0, 0], (uv[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(uv))] @ R
    t = (z - pos[2]) / d[:, 2]
    return pos + t[:, None] * d


# ------------------------------------------------------------------------------------------------ masks
def props_mask(cam, shape=(480, 640)) -> np.ndarray:
    """Pixels of known static props in the top view: tripod feet / legs, the strip beyond the table edge, the robot
    base (the plate and brick detectors ignore them)."""
    import cv2

    from sim.twin.scene.props import PROPS, table_edge_line, tripod_geometry

    m = np.zeros(shape, np.uint8)
    g = tripod_geometry()
    for f in g["feet"]:
        uv, z = project(np.r_[f, TABLE_Z], cam)
        if z[0] > 0:
            r = int(1.25 * PROPS["foot_r"] * cam[2][0, 0] / z[0]) + 6
            cv2.circle(m, tuple(int(v) for v in uv[0]), r, 1, -1)
        hub = np.r_[g["hub_xy"], 0.06]
        pts = project(np.linspace(hub, np.r_[f, 0.0], 12), cam)[0].astype(np.int32)
        cv2.polylines(m, [pts], False, 1, 14)
    hub3 = np.r_[g["hub_xy"], 0.06]
    for P3, rad in ((hub3, 0.05), (g["cam_pos"], 0.075)):
        uv, z = project(P3, cam)
        if z[0] > 0:
            cv2.circle(m, tuple(int(v) for v in uv[0]), int(rad * cam[2][0, 0] / z[0]), 1, -1)
    # beyond the table edge (+ 1.5 cm margin)
    ys = np.linspace(-0.8, 0.8, 50)
    ea, eb = table_edge_line()
    edge = np.c_[ea + eb * ys + 0.015, ys, np.full(50, TABLE_Z)]
    uv = project(edge, cam)[0]
    poly = np.r_[uv, [[700, -50], [-60, -50]]].astype(np.int32)
    cv2.fillPoly(m, [poly], 1)
    # robot base (r 7 cm around the origin, up to 12 cm high)
    for z in (0.0, 0.06, 0.12):
        uv, _ = project(np.c_[0.075 * np.cos(np.linspace(0, 6.3, 24)), 0.075 * np.sin(np.linspace(0, 6.3, 24)),
                              np.full(24, z)], cam)
        cv2.fillPoly(m, [uv.astype(np.int32)], 1)
    return m.astype(bool)


def arm_mask(state_deg, cam, shape=(480, 640), radius: float = 0.045) -> np.ndarray:
    """Pixels within `radius` of the arm (FK link origins + fingertip, sim/twin/rl/kin.py) at a joint state."""
    import cv2
    import torch

    kin = _kin()
    q = torch.deg2rad(torch.as_tensor(np.asarray(state_deg[:5]), dtype=torch.float32)) + kin.off
    M = torch.eye(4)
    pts = [np.zeros(3), np.array([0.0, 0.0, 0.06])]
    for i in range(5):
        c, s = torch.cos(q[i]), torch.sin(q[i])
        Rz = torch.eye(4)
        Rz[0, 0], Rz[0, 1], Rz[1, 0], Rz[1, 1] = c, -s, s, c
        M = M @ kin.T[i] @ Rz
        pts.append(M[:3, 3].numpy().copy())
    pts.append((M @ kin.tip)[:3].numpy())
    pts.append((M @ torch.tensor([0.0, 0.0, -0.11, 1.0]))[:3].numpy())
    m = np.zeros(shape, np.uint8)
    for k, (a, b) in enumerate(zip(pts[:-1], pts[1:], strict=True)):
        rad = radius if k < len(pts) - 3 else radius * 0.6  # the fingers are slimmer than the links
        for p in np.linspace(a, b, 12):
            uv, z = project(p, cam)
            if z[0] > 0.05:
                cv2.circle(m, tuple(int(v) for v in uv[0]), int(rad * cam[2][0, 0] / z[0]), 1, -1)
    return m.astype(bool)


_KIN = None


def _kin():
    global _KIN
    if _KIN is None:
        sys.path.insert(0, str(REPO / "sim/residual_env"))
        from calib import load_profile

        from sim.twin.rl.kin import Kin

        _KIN = Kin(load_profile("joe_cal2").joint_offsets, device="cpu")
    return _KIN


def classify_kind(long_: float, short: float, color: str | None = None) -> str:
    """Brick kind from its top-view rectangle (m, sides included) via the side ratio; colour inventory prior."""
    ratio = (long_ - SIDE_INFLATION) / max(short - SIDE_INFLATION, 1e-3)
    kind = "2x2" if ratio < KIND_RATIO[0] or long_ < 0.045 else ("curved_2x3" if ratio < KIND_RATIO[1] else "2x4")
    allowed = COLOR_KINDS.get(color)
    return kind if not allowed or kind in allowed else allowed[0]


def _reclassify(lay: dict) -> dict:
    """Re-derive brick kinds of a cached layout from its stored rectangle sides (classifier updates)."""
    for b in lay.get("distractors", []) + ([lay["target"]] if lay.get("target") else []):
        if "long" in b and "short" in b:
            b["kind"] = classify_kind(b["long"], b["short"], b.get("color"))
    return lay


def _contour_pts(comp):
    import cv2

    cs, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)[:, 0, :].astype(float)
    return c


def ransac_circle(P, r0=PLATE_R, r_tol=0.008, iters=300, tol=0.004, rng=None):
    """Circle (centre, r) through 2D points P with r in [r0 - r_tol, r0 + r_tol]: 3-point RANSAC + least squares
    refit on the inliers. Returns (centre, r, inlier mask)."""
    rng = rng or np.random.default_rng(0)
    best = (None, None, np.zeros(len(P), bool))
    if len(P) < 12:
        return best
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        A = np.array([[b[0] - a[0], b[1] - a[1]], [c[0] - a[0], c[1] - a[1]]]) * 2
        y = np.array([b @ b - a @ a, c @ c - a @ a])
        if abs(np.linalg.det(A)) < 1e-9:
            continue
        ctr = np.linalg.solve(A, y)
        r = np.linalg.norm(a - ctr)
        if abs(r - r0) > r_tol:
            continue
        inl = np.abs(np.linalg.norm(P - ctr, axis=1) - r) < tol
        if inl.sum() > best[2].sum():
            best = (ctr, r, inl)
    ctr, r, inl = best
    if ctr is None:
        return best
    Q = P[inl]
    A = np.c_[2 * Q, np.ones(len(Q))]
    sol, *_ = np.linalg.lstsq(A, (Q ** 2).sum(1), rcond=None)
    c2 = sol[:2]
    r2 = math.sqrt(max(sol[2] + c2 @ c2, 1e-12))
    if abs(r2 - r0) <= r_tol:
        ctr, r = c2, r2
    else:  # keep the radius prior, refit the centre only
        r = float(np.clip(r2, r0 - r_tol, r0 + r_tol))
    inl = np.abs(np.linalg.norm(P - ctr, axis=1) - r) < tol
    return ctr, r, inl


def detect_plates(top, cam, label: str | None = None, ignore=None, relaxed=False) -> dict:
    """{colour: dict(x, y, diam, score, coverage, area_ratio)} for the plates in one top frame."""
    import cv2

    hsv = cv2.cvtColor(top, cv2.COLOR_RGB2HSV)
    ignore = props_mask(cam) if ignore is None else ignore
    H, W = top.shape[:2]
    out = {}
    for name, (lo, hi) in PLATE_HSV.items():
        if relaxed and name != label:
            continue
        m = cv2.inRange(hsv, lo, hi) > 0
        m &= ~ignore
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(m)
        cands = []
        for i in range(1, n):
            area = st[i, cv2.CC_STAT_AREA]
            if area < (800 if relaxed else 2500):
                continue
            c = _contour_pts(lab == i)
            if c is None or len(c) < 40:
                continue
            inside = (c[:, 0] > 3) & (c[:, 0] < W - 4) & (c[:, 1] > 3) & (c[:, 1] < H - 4)
            P = backproject(c[inside], cam, TABLE_Z + RIM_Z)[:, :2]
            if len(P) < 30:
                continue
            ctr, r, inl = ransac_circle(P)
            if ctr is None:
                continue
            # arc coverage of the inliers (angle histogram) and filled-area consistency in the image
            ang = np.arctan2(*(P[inl] - ctr).T[::-1])
            cov = len(np.unique(((ang + np.pi) / (2 * np.pi) * 36).astype(int))) / 36.0
            circ = np.c_[ctr[0] + r * np.cos(np.linspace(0, 2 * np.pi, 72)),
                         ctr[1] + r * np.sin(np.linspace(0, 2 * np.pi, 72)), np.full(72, TABLE_Z + RIM_Z)]
            uv, _ = project(circ, cam)
            disk = np.zeros((H, W), np.uint8)
            cv2.fillPoly(disk, [uv.astype(np.int32)], 1)
            disk_in = disk.sum()
            if disk_in < 500:
                continue
            fill = float(((lab == i) & (disk > 0)).sum() / disk_in)  # how much of the projected disk is plate colour
            spill = float(((lab == i) & (disk == 0)).sum() / max(area, 1))  # blob outside the disk
            score = cov * fill * (1.0 - min(spill, 1.0)) * min(1.0, inl.sum() / 40.0)
            ok = cov >= (0.2 if relaxed else 0.33) and fill >= (0.35 if relaxed else 0.5) and spill < 0.35
            if ok:
                cands.append(dict(x=float(ctr[0]), y=float(ctr[1]), diam=float(2 * r), score=round(float(score), 3),
                                  coverage=round(cov, 2), fill=round(fill, 2), spill=round(spill, 2),
                                  inliers=int(inl.sum()), visible=True))
        if cands:
            out[name] = max(cands, key=lambda d: d["score"])
    # two plates cannot overlap: drop the weaker one of an overlapping pair (keep the labelled plate)
    names = sorted(out, key=lambda k: -(out[k]["score"] + (10 if k == label else 0)))
    keep = {}
    for k in names:
        if all(math.hypot(out[k]["x"] - v["x"], out[k]["y"] - v["y"]) > 0.15 for v in keep.values()):
            keep[k] = out[k]
    return keep


def detect_bricks(top, cam, plates: dict, ignore=None, min_rect: float = 0.68) -> list[dict]:
    """Bricks [{color, kind, x, y, yaw, on_plate, long, short, area}] in one top frame. Light-blue blobs within
    11 cm of the inspection tripod's hub are its blue feet (the tripod moves a few cm between sessions)."""
    import cv2

    from sim.twin.scene.props import tripod_geometry

    hub = tripod_geometry()["hub_xy"]

    hsv = cv2.cvtColor(top, cv2.COLOR_RGB2HSV)
    ignore = props_mask(cam) if ignore is None else ignore
    H, W = top.shape[:2]
    # plate disks (image) per colour
    disks = {}
    for name, p in plates.items():
        circ = np.c_[p["x"] + p["diam"] / 2 * np.cos(np.linspace(0, 2 * np.pi, 72)),
                     p["y"] + p["diam"] / 2 * np.sin(np.linspace(0, 2 * np.pi, 72)), np.full(72, TABLE_Z + RIM_Z)]
        d = np.zeros((H, W), np.uint8)
        cv2.fillPoly(d, [project(circ, cam)[0].astype(np.int32)], 1)
        disks[name] = d.astype(bool)
    any_disk = np.zeros((H, W), bool)
    for d in disks.values():
        any_disk |= d
    out = []
    for color, ranges in BRICK_HSV.items():
        m = np.zeros((H, W), bool)
        for lo, hi in ranges:
            m |= cv2.inRange(hsv, lo, hi) > 0
        m &= ~ignore
        # inside a plate of the same hue the plate itself matches: keep only clearly darker / more saturated pixels
        for pname, d in disks.items():
            if color in SAME_AS_PLATE.get(pname, ()):
                v_med = np.median(hsv[..., 2][d & m]) if (d & m).any() else 255
                s_med = np.median(hsv[..., 1][d & m]) if (d & m).any() else 255
                inner = d & m
                m[inner] = (hsv[..., 2][inner] < v_med - 35) | (hsv[..., 1][inner] > min(255, s_med + 45))
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(m)
        for i in range(1, n):
            ar = st[i, cv2.CC_STAT_AREA]
            if not 160 < ar < 4000:
                continue
            comp = lab == i
            ys, xs = np.nonzero(comp)
            on_plate = bool(any_disk[ys, xs].mean() > 0.5)
            z = TABLE_Z + BRICK_TOP + (0.003 if on_plate else 0.0)
            c = _contour_pts(comp)
            P = backproject(c, cam, z)[:, :2].astype(np.float32)
            (cx, cy), (w, h), ang = cv2.minAreaRect(P)
            long_, short = max(w, h), min(w, h)
            if short < 0.018 or long_ > 0.09 or short > 0.05:
                continue
            if cv2.contourArea(P) / max(w * h, 1e-12) < min_rect:  # bricks are rectangles from above
                continue
            if color == "light_blue" and math.hypot(cx - hub[0], cy - hub[1]) < 0.11:
                continue
            kind = classify_kind(long_, short, color)
            yaw = math.radians(ang) + (math.pi / 2 if h > w else 0.0)
            out.append(dict(color=color, kind=kind, x=float(cx), y=float(cy), yaw=float(yaw), on_plate=on_plate,
                            long=round(float(long_), 4), short=round(float(short), 4), area=int(ar)))
    # the same brick matched by two classes (orange / brown / red overlap): keep the larger blob
    out.sort(key=lambda b: -b["area"])
    keep = []
    for b in out:
        if all(math.hypot(b["x"] - k["x"], b["y"] - k["y"]) > 0.02 for k in keep):
            keep.append(b)
    return keep


# ------------------------------------------------------------------------------------------------ episodes
def _labels(root: Path) -> dict:
    p = root / "meta/dlsj/labels.jsonl"
    if not p.exists():
        return {}
    rows = [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
    return {r["episode_index"]: r for r in rows}


def grasp_frame(states: np.ndarray, insp: int | None, thr=12.0, stay=15.0, open_min=15.0):
    """Frame the gripper closes on the brick (validate_twin / residual validate.py rule)."""
    g = states[:insp, 5] if insp else states[:, 5]
    if len(g) == 0 or g.max() < open_min:
        return None
    for f in range(1, len(g)):
        if g[f - 1] > thr >= g[f] and (g[f:] <= stay).all():
            return f
    return None


def fingertip(states_deg: np.ndarray) -> np.ndarray:
    """FK fingertip (env frame) of the twin robot (sim/twin/rl/kin.py, joe_cal2 offsets) for (N, 6) states."""
    import torch

    p, _ = _kin().fk(torch.as_tensor(np.atleast_2d(states_deg), dtype=torch.float32))
    return p.numpy()


def _read_first_and_states(ep: int, root: Path):
    from sim.twin.visuals.realdata import Dataset

    ds = Dataset(str(root))
    return ds, ds.frame(ep, "top", 0), ds.states(ep)


def detect_episode(ep: int, root: Path = DATA, debug_dir: Path | None = None, top=None, states=None) -> dict:
    """Layout of one episode (decodes the first top frame with PyAV: call in plain python, not in Isaac)."""
    lab = _labels(root).get(ep, {})
    if top is None or states is None:
        _, top, states = _read_first_and_states(ep, root)
    cam = top_camera()
    ign = props_mask(cam, top.shape[:2]) | arm_mask(states[0], cam, top.shape[:2])
    bowl = lab.get("bowl")
    plates = detect_plates(top, cam, label=bowl, ignore=ign)
    if bowl and bowl not in plates:
        plates.update(detect_plates(top, cam, label=bowl, ignore=ign, relaxed=True))
    bricks = detect_bricks(top, cam, plates, ignore=ign)
    gf = grasp_frame(states, lab.get("inspection_frame"))
    target, tip = None, None
    if gf is not None:
        tip = fingertip(states[gf])[0]
        target = _nearest(bricks, tip, "detected brick nearest the grasp fingertip")
        if target is not None:
            bricks = [b for b in bricks if b is not target["_src"]]
        else:  # the start-pose arm may hide it from the masked pass: again without the arm mask, looser shape test
            near = _nearest(detect_bricks(top, cam, plates, ignore=props_mask(cam, top.shape[:2]), min_rect=0.5), tip,
                            "relaxed detection near the grasp fingertip")
            target = near or dict(color=_footprint_color(top, cam, tip), kind="2x2", x=float(tip[0]),
                                  y=float(tip[1]), yaw=0.0, on_plate=False,
                                  source="grasp fingertip; colour from the footprint pixels")
        target.pop("_src", None)
    lay = {"episode": int(ep), "dataset": root.name, "bowl": bowl, "plates": plates, "target": target,
           "distractors": bricks, "start_state": np.round(states[0], 3).tolist(), "grasp_frame": gf,
           "tip_xy_at_grasp": None if tip is None else np.round(tip[:2], 4).tolist(), "n_frames": int(len(states))}
    if debug_dir is not None:
        draw_layout(top, lay, cam, Path(debug_dir) / f"layout_{root.name}_ep{ep:03d}.jpg")
    return lay


def draw_layout(top, lay, cam, path):
    import cv2

    img = np.ascontiguousarray(top.copy())
    col = {"orange": (255, 140, 0), "yellow": (255, 255, 0), "purple": (200, 120, 255)}
    for name, p in lay["plates"].items():
        circ = np.c_[p["x"] + p["diam"] / 2 * np.cos(np.linspace(0, 2 * np.pi, 72)),
                     p["y"] + p["diam"] / 2 * np.sin(np.linspace(0, 2 * np.pi, 72)), np.full(72, TABLE_Z + RIM_Z)]
        cv2.polylines(img, [project(circ, cam)[0].astype(np.int32)], True, col.get(name, (255, 255, 255)), 2)
        u = project(np.array([[p["x"], p["y"], TABLE_Z + RIM_Z]]), cam)[0][0].astype(int)
        cv2.putText(img, f"{name} {p['score']:.2f}" + (" *" if name == lay.get("bowl") else ""), tuple(u),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    bs = [dict(b, _t=False) for b in lay["distractors"]] + ([dict(lay["target"], _t=True)] if lay["target"] else [])
    for b in bs:
        L = {"2x2": 0.0317, "2x4": 0.0637, "curved_2x3": 0.0477}[b["kind"]]
        c, s = math.cos(b["yaw"]), math.sin(b["yaw"])
        corners = np.array([[sx * L / 2, sy * 0.0317 / 2] for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
        P = np.c_[b["x"] + corners[:, 0] * c - corners[:, 1] * s, b["y"] + corners[:, 0] * s + corners[:, 1] * c,
                  np.full(4, TABLE_Z + BRICK_TOP)]
        cv2.polylines(img, [project(P, cam)[0].astype(np.int32)], True, (255, 0, 0) if b["_t"] else (255, 255, 255),
                      2 if b["_t"] else 1)
        u = project(np.array([[b["x"], b["y"], TABLE_Z + BRICK_TOP]]), cam)[0][0].astype(int)
        cv2.putText(img, f"{(b['color'] or '?')[:5]} {b['kind'][:6]}", (u[0] - 20, u[1] + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0) if b["_t"] else (255, 255, 255), 1)
    if lay.get("tip_xy_at_grasp"):
        u = project(np.array([[*lay["tip_xy_at_grasp"], TABLE_Z + 0.01]]), cam)[0][0].astype(int)
        cv2.drawMarker(img, tuple(u), (0, 255, 0), cv2.MARKER_CROSS, 14, 2)
    cv2.putText(img, f"ep {lay['episode']} bowl {lay['bowl']}", (8, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])


def _nearest(bricks, tip, source, max_d: float = 0.06):
    if not bricks:
        return None
    d = [math.hypot(b["x"] - tip[0], b["y"] - tip[1]) for b in bricks]
    k = int(np.argmin(d))
    if d[k] >= max_d:
        return None
    return dict(bricks[k], source=source, dist_to_tip=round(d[k], 4), _src=bricks[k])


def _footprint_color(top, cam, tip) -> str | None:
    """Palette class (twin_scene.BRICK_RGB) nearest to the median colour of a 2x2 footprint at the fingertip."""
    import cv2

    from sim.twin.visuals.twin_scene import BRICK_RGB

    sq = np.array([[sx, sy] for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]) * 0.012
    P = np.c_[tip[0] + sq[:, 0], tip[1] + sq[:, 1], np.full(4, TABLE_Z + BRICK_TOP)]
    m = np.zeros(top.shape[:2], np.uint8)
    cv2.fillPoly(m, [project(P, cam)[0].astype(np.int32)], 1)
    if m.sum() < 10:
        return None
    lab = cv2.cvtColor(top, cv2.COLOR_RGB2LAB)[m > 0].astype(float)
    med = np.median(lab, 0)
    names = list(BRICK_RGB)
    ref = cv2.cvtColor(np.uint8([[BRICK_RGB[k] for k in names]]), cv2.COLOR_RGB2LAB)[0].astype(float)
    return names[int(np.argmin(np.linalg.norm(ref - med, axis=1)))]


def _in_isaac() -> bool:
    return any(m in sys.modules for m in ("omni.kit.app", "isaacsim", "omni.usd"))


def layout_from_episode(ep: int, dataset: str | Path = DATA, cache: bool = True, refresh: bool = False) -> dict:
    """Layout of a real episode. Cached as JSON (sim/twin/out/scene/layouts/<dataset>/ep<NNN>.json); computed in a
    subprocess when called from an Isaac process (PyAV must not be imported there: it blacks out the cameras)."""
    root = Path(dataset)
    if not root.is_absolute() and not root.exists():
        root = REPO / "sim/twin/data" / str(dataset)
    path = CACHE / root.name / f"ep{int(ep):03d}.json"
    if cache and path.exists() and not refresh:
        return _reclassify(json.loads(path.read_text()))
    if _in_isaac():
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "detect", "--root", str(root),
                        "--episodes", str(int(ep))], check=True, cwd=str(REPO))
        return _reclassify(json.loads(path.read_text()))
    lay = detect_episode(int(ep), root)
    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(lay, indent=1))
    return lay


# ------------------------------------------------------------------------------------------------ statistics
def stats(layouts: list[dict]) -> dict:
    """Brick kind / colour / count frequencies and plate visibility over real first frames."""
    from collections import Counter

    kinds, colors, ck, tk, tc, n_d, on_plate, n_pl, vis = (Counter() for _ in range(9))
    for L in layouts:
        bs = L["distractors"]
        n_d[len(bs)] += 1
        n_pl[len(L["plates"])] += 1
        for k in L["plates"]:
            vis[k] += 1
        for b in bs:
            kinds[b["kind"]] += 1
            colors[b["color"]] += 1
            ck[f"{b['color']}|{b['kind']}"] += 1
            on_plate[bool(b["on_plate"])] += 1
        t = L.get("target")
        if t and t.get("color"):
            tk[t["kind"]] += 1
            tc[t["color"]] += 1
            ck[f"{t['color']}|{t['kind']}"] += 1
            kinds[t["kind"]] += 1
            colors[t["color"]] += 1

    def norm(c):
        s = sum(c.values()) or 1
        return {str(k): round(v / s, 4) for k, v in sorted(c.items(), key=lambda kv: -kv[1])}

    return {"episodes": len(layouts), "brick_kind": norm(kinds), "brick_color": norm(colors),
            "color_kind": norm(ck), "target_kind": norm(tk), "target_color": norm(tc),
            "distractor_count": {str(k): v for k, v in sorted(n_d.items())},
            "distractor_on_plate_frac": round(on_plate[True] / max(1, sum(on_plate.values())), 3),
            "plates_visible_count": {str(k): v for k, v in sorted(n_pl.items())},
            "plate_visible_frac": {k: round(v / max(1, len(layouts)), 3) for k, v in vis.items()},
            "plate_layouts": [{k: [round(p["x"], 3), round(p["y"], 3)] for k, p in L["plates"].items()}
                              for L in layouts if len(L["plates"]) >= 2],
            "counts": {"bricks": sum(kinds.values()), "targets_with_colour": sum(tk.values())}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("detect", "stats"))
    ap.add_argument("--root", default=str(DATA))
    ap.add_argument("--episodes", type=int, nargs="*")
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)
    from sim.twin.visuals.realdata import Dataset

    ds = Dataset(str(root))
    eps = a.episodes if a.episodes else list(range(0, len(ds), a.every))
    dbg = REPO / "sim/twin/out/scene/layout_debug" if a.debug else None
    lays = []
    for ep in eps:
        path = CACHE / root.name / f"ep{ep:03d}.json"
        if path.exists() and not a.refresh and not a.debug:
            lays.append(_reclassify(json.loads(path.read_text())))
            continue
        lay = detect_episode(ep, root, debug_dir=dbg, top=ds.frame(ep, "top", 0), states=ds.states(ep))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(lay, indent=1))
        lays.append(lay)
        t = lay["target"] or {}
        print(f"ep {ep}: bowl {lay['bowl']} plates {sorted(lay['plates'])} target {t.get('color')} {t.get('kind')} "
              f"d={t.get('dist_to_tip')} | {len(lay['distractors'])} distractors", flush=True)
    if a.cmd == "stats":
        s = stats(lays)
        s["dataset"] = root.name
        s["every"] = a.every
        STATS_JSON.write_text(json.dumps(s, indent=1))
        print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
