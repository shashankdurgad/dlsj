"""Static props of the real station that the cameras see (visual meshes; colliders only where the arm can reach).

    from sim.twin.scene.props import add_props
    info = add_props(num_envs, profile)          # after InteractiveScene + apply_twin_visuals, before sim.reset()

Positions are in the env frame (robot base at the origin, x away from the robot, +y to the right of the top view),
back-projected from the measured top camera (joe_cal2 profile, table plane z = -0.002) on the per-pixel median of the
cal2 top frames (sim/twin/out/visuals/top_median.png) and single first frames, and sized from the station photos
(sim/twin/data/photos/*, the user's phone photos). See PROPS below for every number and where it comes from.

* inspection camera: Intel RealSense D415 (99 x 23 x 20 mm, aluminium) at the measured inspect pose
  (calib/cameras_measured.json, ChArUco) on a silver mini tripod (ball head + 3 legs) whose blue rubber feet (blue
  round stickers) are the two blue discs at the right of the top view: (0.290, 0.186) and (0.384, 0.143); the third
  foot is outside the view (a symmetric tripod: hub on the bisector, 120 deg apart). Collider: camera + head (box).
* black cables: the robot's / camera's USB cables, polylines back-projected from the first top frame of episode 0,
  lying on the table (r 2.5 mm tubes). No collider (soft, on the table).
* white box: the box behind the inspection camera (top-right corner of the top view, station photos); 0.24 x 0.13 x
  0.115 m. Collider (inside the arm's reach envelope).
* overhead camera tripod (Manfrotto, black): only one rubber foot is visible (bottom-left corner of the top view,
  ~(0.446, -0.375)); its legs are not seen by any camera (a guessed leg crossed the inspect view, where the real frames
  show none), so only that foot is modelled. No collider.
* table edge / floor: the wooden table top ends along x = -0.019 + 0.26 y (the wood boundary along the top of the
  top view, back-projected: (-0.136, -0.45) .. (0.081, 0.386); the edge is 14.6 deg off the robot's y axis and the
  robot base overhangs it, clamped). The twin's textured table quad is clipped to that line, a 28 mm board edge is
  added, and beyond it the blue floor / cloth (median colour of that strip) 0.74 m below. The physics table cube is
  unchanged (its render geometry is hidden; it still extends under the robot).
* robot base clamp: the blue printed clamp bracket left of the base at the edge (top view (-0.098..-0.021, -0.03))
  and a black clamp bar / jaws gripping the board under it.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
TABLE_Z = -0.002

PROPS = {
    # x_edge(y) = a + b * y; fallback only: table_edge_line() reads visuals/textures/table_top.json["table_edge"] (the
    # textured quad's own edge, 16.0 deg), which agrees with the top-median edge (-0.019 + 0.26 y) within 1.5 cm
    "table_edge": (-0.019, 0.26),
    "table_thickness": 0.028,
    "floor_z": -0.74,
    "floor_rgb": (8, 52, 112),  # strip beyond the edge: median (0, 64, 130), darkened (shadow under the table)
    "edge_rgb": (118, 104, 82),
    # inspection camera tripod (top view blue blobs, back-projected on the table plane)
    "feet_xy": [(0.294, 0.193), (0.384, 0.151)],
    "foot_r": 0.022,
    "tripod_hub_z": 0.068,
    "tripod_leg_r": 0.0055,
    "d415_size": (0.099, 0.020, 0.023),  # long x depth x height in the camera frame (x right, depth, y)
    # cables: (x, y) polylines on the table (episode 0 first top frame)
    "cables": [
        [(-0.03, 0.10), (-0.015, 0.109), (0.023, 0.139), (0.076, 0.183), (0.138, 0.221), (0.196, 0.249),
         (0.243, 0.258), (0.287, 0.259), (0.33, 0.275), (0.36, 0.225)],
        [(0.05, 0.25), (0.03, 0.30), (0.01, 0.344), (0.053, 0.369), (0.08, 0.386), (0.13, 0.42)],
        [(0.38, 0.20), (0.40, 0.26), (0.36, 0.33), (0.25, 0.37), (0.16, 0.36)],  # D415 cable to the box side
    ],
    "cable_r": 0.0025,
    # white box (centre x, y; size x, y, z)
    "box_center": (-0.03, 0.36),
    "box_size": (0.24, 0.13, 0.115),
    # overhead tripod foot: seen in one session only (the tripod moves between sessions), so off by default;
    # set to (0.446, -0.375) to show it
    "ovh_foot": None,
    # robot base clamp (blue bracket centre x, y, size), black bar under the edge
    "clamp_bracket": ((-0.06, -0.03), (0.075, 0.03, 0.035)),
}


# ------------------------------------------------------------------------------------------------ meshes
def tube_mesh(a, b, r, seg: int = 16, caps: bool = True):
    """Cylinder from a to b (radius r), smooth side normals, flat caps."""
    from sim.twin.visuals.assets import MeshBuf

    a, b = np.asarray(a, float), np.asarray(b, float)
    ax = b - a
    L = np.linalg.norm(ax)
    w = ax / max(L, 1e-9)
    u = np.cross(w, [0, 0, 1.0] if abs(w[2]) < 0.9 else [1.0, 0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(w, u)
    t = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    ring = np.cos(t)[:, None] * u + np.sin(t)[:, None] * v
    buf = MeshBuf()
    pts = np.r_[a + r * ring, b + r * ring]
    buf.add(pts, np.r_[ring, ring], [[k, (k + 1) % seg, seg + (k + 1) % seg, seg + k] for k in range(seg)])
    if caps:
        for c, n, rev in ((a, -w, True), (b, w, False)):
            p = np.r_[[c], c + r * ring]
            f = [[0, 1 + k, 1 + (k + 1) % seg] for k in range(seg)]
            buf.add(p, np.tile(n, (seg + 1, 1)), [x[::-1] for x in f] if rev else f)
    return buf.arrays()


def sweep_mesh(path_xyz, r, seg: int = 10, samples: int = 8):
    """Tube along a polyline (Catmull-Rom smoothed), smooth normals; for cables."""
    from sim.twin.visuals.assets import MeshBuf

    P = np.asarray(path_xyz, float)
    pts = []
    for i in range(len(P) - 1):
        p0, p1, p2, p3 = P[max(i - 1, 0)], P[i], P[i + 1], P[min(i + 2, len(P) - 1)]
        for s in np.linspace(0, 1, samples, endpoint=False):
            s2, s3 = s * s, s * s * s
            pts.append(0.5 * ((2 * p1) + (-p0 + p2) * s + (2 * p0 - 5 * p1 + 4 * p2 - p3) * s2 +
                              (-p0 + 3 * p1 - 3 * p2 + p3) * s3))
    pts.append(P[-1])
    C = np.asarray(pts)
    T = np.gradient(C, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True).clip(1e-9)
    t = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    V, N = [], []
    for c, w in zip(C, T, strict=True):
        u = np.cross(w, [0, 0, 1.0])
        u /= max(np.linalg.norm(u), 1e-9)
        v = np.cross(w, u)
        ring = np.cos(t)[:, None] * u + np.sin(t)[:, None] * v
        V.append(c + r * ring)
        N.append(ring)
    V, N = np.concatenate(V), np.concatenate(N)
    m = len(C)
    faces = [[i * seg + k, i * seg + (k + 1) % seg, (i + 1) * seg + (k + 1) % seg, (i + 1) * seg + k]
             for i in range(m - 1) for k in range(seg)]
    buf = MeshBuf()
    buf.add(V, N, faces)
    return buf.arrays()


def _xf(pts, R, t):
    return np.asarray(pts) @ np.asarray(R).T + np.asarray(t)


# ------------------------------------------------------------------------------------------------ USD
def _mats(stage):
    from sim.twin.visuals.materials import preview_material

    L = "/World/Looks/TwinProps"
    return {
        "alu": preview_material(stage, f"{L}/Aluminium", (196, 198, 202), 0.35, metallic=0.3),
        "d415_front": preview_material(stage, f"{L}/D415Front", (14, 14, 16), 0.15, clearcoat=0.6),
        "black": preview_material(stage, f"{L}/BlackPlastic", (22, 22, 24), 0.55),
        "rubber": preview_material(stage, f"{L}/Rubber", (18, 18, 20), 0.8),  # overhead tripod feet
        "feet": preview_material(stage, f"{L}/BlueFeet", (28, 96, 214), 0.45),
        "cable": preview_material(stage, f"{L}/Cable", (14, 14, 15), 0.45, clearcoat=0.2),
        "white": preview_material(stage, f"{L}/WhiteBox", (232, 232, 228), 0.75),
        "floor": preview_material(stage, f"{L}/Floor", PROPS["floor_rgb"], 0.9),
        "edge": preview_material(stage, f"{L}/TableEdge", PROPS["edge_rgb"], 0.8),
        "base_blue": preview_material(stage, f"{L}/BaseBlue", (20, 70, 170), 0.5),
    }


def _mesh(stage, path, m, mat):
    from sim.twin.visuals.assets import define_mesh
    from sim.twin.visuals.materials import bind

    pts, nrm, c, i = m
    mesh = define_mesh(stage, path, pts, nrm, c, i)
    bind(mesh.GetPrim(), mat)
    return mesh


def _collider_box(stage, path, center, size, R=None):
    """Static (no rigid body) box collider, invisible to the cameras (purpose guide)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    M = np.eye(4)
    M[:3, :3] = (np.eye(3) if R is None else np.asarray(R)) * np.asarray(size)[None, :]
    M[:3, 3] = center
    cube.AddTransformOp().Set(Gf.Matrix4d(*M.T.reshape(-1).tolist()))
    cube.CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim()).CreateContactOffsetAttr().Set(0.004)
    return cube


def tripod_geometry(profile=None) -> dict:
    """Hub, feet and D415 pose of the inspection camera tripod (env frame)."""
    F = np.asarray(PROPS["feet_xy"], float)
    d = np.linalg.norm(F[1] - F[0])
    mid = F.mean(0)
    perp = np.array([-(F[1] - F[0])[1], (F[1] - F[0])[0]]) / d
    if perp[1] < 0:  # the hub is on the far (+y) side of the visible feet
        perp = -perp
    hub = mid + perp * d / (2 * math.sqrt(3))
    third = hub - ((F[0] - hub) + (F[1] - hub))
    third = hub + (third - hub) / np.linalg.norm(third - hub) * np.linalg.norm(F[0] - hub)
    cam = _inspect_pose(profile)
    return {"hub_xy": hub, "feet": np.r_[F, [third]], "cam_pos": cam[0], "cam_R": cam[1]}


def _inspect_pose(profile=None):
    """(position, R with columns = camera x right, y down, z forward in the env frame) of the inspect camera."""
    p = REPO / "sim/twin/calib/cameras_measured.json"
    if profile is not None and getattr(profile, "inspect", None) is not None:
        rows = np.asarray(profile.inspect.rows, float)
        R = np.stack([rows[0, :3], -rows[1, :3], -rows[2, :3]], 1)
        return rows[3, :3].copy(), R
    m = json.loads(p.read_text())["inspect"]
    return np.asarray(m["position"], float), np.asarray(m["world_to_cv"], float).T


def add_props(num_envs: int = 1, profile=None, colliders: bool = True, clip_table: bool = True) -> dict:
    """Author the props under /World/envs/env_<e>/TwinProps (static; call before sim.reset())."""
    import omni.usd

    from sim.twin.visuals.assets import box_mesh, cylinder_mesh, hide_geometry

    stage = omni.usd.get_context().get_stage()
    M = _mats(stage)
    geo = tripod_geometry(profile)
    info = {"tripod_hub_xy": np.round(geo["hub_xy"], 3).tolist(), "tripod_feet": np.round(geo["feet"], 3).tolist(),
            "colliders": []}
    th = PROPS["table_thickness"]
    ex0, exy = table_edge_line()
    ang = math.atan(exy)  # edge direction (dx/dy) -> the table frame is rotated by -ang about z
    Rt = np.array([[math.cos(ang), math.sin(ang), 0], [-math.sin(ang), math.cos(ang), 0], [0, 0, 1.0]])
    for e in range(num_envs):
        root = f"/World/envs/env_{e}/TwinProps"
        stage.DefinePrim(root, "Xform")
        # ---- table edge, floor
        if clip_table:
            info["table_quad_clipped"] = _clip_table_quad(stage, e, ex0, exy)
            tab = stage.GetPrimAtPath(f"/World/envs/env_{e}/Table")
            if tab:
                hide_geometry(tab)
            # slab in the table frame (x' perpendicular to the edge), its -x' face on the edge line through (ex0, 0)
            pts, nrm, c, i = box_mesh((1.6, 2.6, th), (0.8, 0.0, 0.0), bottom=False)
            org = np.array([ex0, 0.0, TABLE_Z - th / 2 - 0.0002])
            _mesh(stage, f"{root}/TableSlab", (_xf(pts, Rt, org), nrm @ Rt.T, c, i), M["edge"])
            _mesh(stage, f"{root}/Floor", box_mesh((4.0, 5.0, 0.01), (-1.0, 0.0, PROPS["floor_z"])), M["floor"])
            for k, y in enumerate((-0.55, 0.55)):  # trestle legs under the edge (dark, rarely visible)
                p = org + Rt @ np.array([0.08, y, 0.0])
                _mesh(stage, f"{root}/TableLeg{k}", box_mesh((0.05, 0.05, abs(PROPS["floor_z"]) - th),
                                                            (p[0], p[1], (PROPS["floor_z"] + TABLE_Z - th) / 2)),
                      M["black"])
        # ---- robot base clamp: blue printed bracket at the edge, black clamp bar and jaws through the board
        (cx, cy), cs = PROPS["clamp_bracket"]
        _mesh(stage, f"{root}/ClampBracket", box_mesh(cs, (cx, cy, TABLE_Z + cs[2] / 2)), M["base_blue"])
        xe = ex0 + exy * cy
        _mesh(stage, f"{root}/ClampBar", box_mesh((0.014, 0.03, 0.11), (xe - 0.012, cy, TABLE_Z - 0.035)),
              M["black"])
        _mesh(stage, f"{root}/ClampJawBottom", box_mesh((0.06, 0.035, 0.012), (xe + 0.02, cy, TABLE_Z - th - 0.007)),
              M["black"])
        # ---- inspection camera + mini tripod
        cam_p, cam_R = geo["cam_pos"], geo["cam_R"]
        L, D, H = PROPS["d415_size"]
        # camera frame: x right (long side), y down (height), z forward; body behind the optical centre
        Rb = np.stack([cam_R[:, 0], cam_R[:, 2], -cam_R[:, 1]], 1)  # body axes: long, forward, up
        c_body = cam_p - cam_R[:, 2] * (D / 2) + cam_R[:, 0] * 0.0
        pts, nrm, c, i = box_mesh((L, D, H))
        _mesh(stage, f"{root}/D415", (_xf(pts, Rb, c_body), nrm @ Rb.T, c, i), M["alu"])
        pts, nrm, c, i = box_mesh((L - 0.006, 0.0008, H - 0.005))
        _mesh(stage, f"{root}/D415Front", (_xf(pts, Rb, cam_p + cam_R[:, 2] * 0.0002), nrm @ Rb.T, c, i),
              M["d415_front"])
        hub = np.r_[geo["hub_xy"], TABLE_Z + PROPS["tripod_hub_z"]]
        under = c_body + Rb[:, 2] * (-H / 2)
        _mesh(stage, f"{root}/TripodHead", tube_mesh(under, hub + [0, 0, 0.012], 0.009), M["alu"])
        _mesh(stage, f"{root}/TripodBall", cylinder_mesh(0.011, 0.02, tuple(hub + [0, 0, 0.004])), M["alu"])
        for k, f in enumerate(geo["feet"]):
            foot = np.r_[f, TABLE_Z + 0.006]
            _mesh(stage, f"{root}/TripodLeg{k}", tube_mesh(hub, foot, PROPS["tripod_leg_r"]), M["alu"])
            _mesh(stage, f"{root}/TripodFoot{k}", cylinder_mesh(PROPS["foot_r"], 0.0012,
                                                                (f[0], f[1], TABLE_Z + 0.0007), seg=32), M["feet"])
        if colliders:
            _collider_box(stage, f"{root}/D415Collider", c_body, (L + 0.004, D + 0.004, H + 0.004), Rb)
            hc = (under + hub) / 2
            _collider_box(stage, f"{root}/TripodHeadCollider", hc, (0.03, 0.03, max(0.01, under[2] - hub[2] + 0.02)))
            info["colliders"] += ["D415", "TripodHead"]
        # ---- cables
        for k, poly in enumerate(PROPS["cables"]):
            P = np.c_[np.asarray(poly, float), np.full(len(poly), TABLE_Z + PROPS["cable_r"])]
            _mesh(stage, f"{root}/Cable{k}", sweep_mesh(P, PROPS["cable_r"]), M["cable"])
        # ---- white box
        bc, bsz = PROPS["box_center"], PROPS["box_size"]
        _mesh(stage, f"{root}/WhiteBox", box_mesh(bsz, (bc[0], bc[1], TABLE_Z + bsz[2] / 2)), M["white"])
        if colliders:
            _collider_box(stage, f"{root}/WhiteBoxCollider", (bc[0], bc[1], TABLE_Z + bsz[2] / 2), bsz)
            info["colliders"].append("WhiteBox")
        # ---- overhead camera tripod: the visible rubber foot
        if PROPS["ovh_foot"] is not None:
            foot = np.r_[PROPS["ovh_foot"], TABLE_Z]
            _mesh(stage, f"{root}/OverheadFoot", tube_mesh(foot, foot + [0, 0, 0.03], 0.02), M["rubber"])
    return info


def table_edge_line() -> tuple[float, float]:
    """(a, b) of the real table edge x = a + b * y (env frame), from the textured table's own layout
    (visuals/textures/table_top.json "table_edge": point + direction) so the clip, the texture and the board edge
    agree; PROPS["table_edge"] when that file has none."""
    p = REPO / "sim/twin/visuals/textures/table_top.json"
    try:
        te = json.loads(p.read_text())["table_edge"]
        (px, py), (dx, dy) = te["point"], te["dir"]
        b = dx / dy
        return float(px - b * py), float(b)
    except (OSError, KeyError, ValueError, ZeroDivisionError):
        return tuple(PROPS["table_edge"])


def _clip_table_quad(stage, e: int, x0_edge: float, dxdy: float) -> bool:
    """Cut twin_scene's textured TableTopVisual quad at the real table edge x = x0_edge + dxdy * y (the -x side of
    the quad becomes the slanted edge; st follows the moved vertices)."""
    from pxr import UsdGeom, Vt

    prim = stage.GetPrimAtPath(f"/World/envs/env_{e}/TableTopVisual")
    if not prim:
        return False
    mesh = UsdGeom.Mesh(prim)
    P = np.asarray(mesh.GetPointsAttr().Get(), float)
    pv = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
    st = np.asarray(pv.Get(), float) if pv else None
    x0, x1 = P[:, 0].min(), P[:, 0].max()
    left = np.isclose(P[:, 0], x0)
    xe = x0_edge + dxdy * P[left, 1]
    if np.any(xe <= x0) or np.any(xe >= x1):
        return False
    P[left, 0] = xe
    if st is not None:
        st[left, 0] = (xe - x0) / (x1 - x0)
        pv.Set(Vt.Vec2fArray.FromNumpy(st.astype(np.float32)))
    mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(P.astype(np.float32)))
    mesh.CreateExtentAttr([tuple(map(float, P.min(0))), tuple(map(float, P.max(0)))])
    return True
