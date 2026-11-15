"""Fin-ray TPU fingers of the real station (XLeRobot soft gripper) for the twin: colliders + visuals.

Geometry: assets/finray_so101.npz (build_assets.py, from the open CAD; link frames, metres). Kinematics are the stock
SO-101 gripper: the fixed finger is part of gripper_link, the other finger turns with the moving jaw (one servo).

Used by the physics layer (sim/twin/physics/apply.py, `gripper.collider = "finray"`) and by the visuals
(sim/twin/visuals/paint.py). Compliance model: each fin is a chain of convex pieces rigidly attached to its link;
the TPU / fin-ray give is PhysX compliant contact on the fin material (spring stiffness + damping instead of a hard
non-penetration constraint), see README.md for the calibration and what was tried.

Bending fins (`hinges` > 0): the fin's convex pieces are split at their boundaries into a chain of rigid segments,
root -> tip, each a light rigid body nested under the previous one and joined to it by a revolute spring-damper
joint (axis = the gripper joint axis, so the fin bows in the closing plane). The root piece stays on the link. The
joints are articulation DOFs named fin_<fixed|jaw>_<a|b|c>; the physics layer drives them with an implicit
actuator (target 0 = the CAD shape, stiffness = the fin's bending stiffness).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NPZ = HERE / "assets/finray_so101.npz"
FIXED_VISUAL = "wrist_roll_follower_so101_v1"
JAW_VISUAL = "moving_jaw_so101_v1"
JOINT_XYZ = np.array([0.0202, 0.0188, -0.0234])  # gripper joint origin in gripper_link (URDF), rpy (pi/2, 0, 0)
BRICK_W = 0.0317


@lru_cache(maxsize=1)
def geometry() -> dict:
    """The fin-ray arrays (see build_assets.py): <fixed|jaw>_{base,fin}_{v,f}, _hull<k>_{v,f}, _tip, _face."""
    with np.load(NPZ) as d:
        return {k: d[k] for k in d.files}


def n_segments() -> int:
    """Convex collider pieces per fin."""
    return sum(1 for k in geometry() if k.startswith("fixed_hull") and k.endswith("_v"))


def jaw_to_grip(p: np.ndarray, q: float) -> np.ndarray:
    """Points in the moving-jaw link frame -> gripper_link frame at joint angle q (URDF kinematics)."""
    c, s = np.cos(q), np.sin(q)
    rot = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]]) @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    return np.asarray(p, float) @ rot.T + JOINT_XYZ


def _fin_axis(which: str) -> np.ndarray:
    """Unit vector from a fin's root to its tip in its link frame (the direction `extend` moves the fin)."""
    geo = geometry()
    v = geo[f"{which}_tip"].astype(float) - geo[f"{which}_hull{n_segments() - 1}_v"].astype(float).mean(0)
    return v / np.linalg.norm(v)


SEG_NAMES = "abcdef"
HINGE_AXIS = {"fixed": "Y", "jaw": "Z"}  # the gripper joint axis in each link frame (URDF rpy (pi/2, 0, 0), axis z)


def segments(which: str, hinges: int, extend: float = 0.0, slices: bool = False, root_len: float = 0.012) -> list:
    """Fin chain root -> tip. Item 0 = the part fixed to the link; each later item is a segment: name, collider
    pieces, hinge point in the link frame, the parent's origin in the link frame.
    slices=False (1-3 hinges): the CAD convex pieces (hull 3 = root .. hull 0 = tip) cut at their boundaries.
    slices=True (1-5 hinges): the fin mesh cut along its axis into a clamped root of `root_len` and `hinges` equal
    slices (one convex hull each, 1.5 mm overlap); hinges at the cuts, on the mid cross-section."""
    if slices and hinges > 0:
        return _slice_chain(which, int(hinges), extend, root_len)
    geo = geometry()
    n = n_segments()
    order = list(range(n - 1, -1, -1))  # root -> tip
    hinges = int(max(0, min(hinges, n - 1)))
    cuts = {0: [], 1: [1], 2: [1, 2], 3: [1, 2, 3]}[hinges]  # chain positions where a new segment starts
    shift = fin_shift(which, extend)
    centre = [geo[f"{which}_hull{k}_v"].astype(float).mean(0) + shift for k in order]
    out = [{"name": None, "hulls": [], "origin": np.zeros(3), "parent_origin": None}]
    for i, k in enumerate(order):
        if i in cuts:
            hinge = 0.5 * (centre[i - 1] + centre[i])
            out.append({"name": f"fin_{which}_{SEG_NAMES[len(out) - 1]}", "hulls": [], "origin": hinge,
                        "parent_origin": out[-1]["origin"]})
        out[-1]["hulls"].append(k)
    for seg in out:
        seg["pieces"] = [(f"TwinFin{k}", geo[f"{which}_hull{k}_v"].astype(float) + shift, geo[f"{which}_hull{k}_f"])
                         for k in seg["hulls"]]
    return out


def _slice_chain(which: str, hinges: int, extend: float, root_len: float) -> list:
    geo = geometry()
    v = geo[f"{which}_fin_v"].astype(float) + fin_shift(which, extend)
    axis = _fin_axis(which)
    s = v @ axis
    s0, s1 = float(s.min()), float(s.max())
    cuts = [s0 + root_len + (s1 - s0 - root_len) * j / hinges for j in range(hinges)]
    bounds = [s0 - 1.0, *cuts, s1 + 1.0]
    ov = 0.0015
    out = []
    for j in range(hinges + 1):
        pts = v[(s >= bounds[j] - ov) & (s <= bounds[j + 1] + ov)]
        if j == 0:
            seg = {"name": None, "origin": np.zeros(3), "parent_origin": None}
        else:
            ring = v[np.abs(s - cuts[j - 1]) < 0.0015]
            hinge = 0.5 * (ring.min(0) + ring.max(0))
            seg = {"name": f"fin_{which}_{SEG_NAMES[j - 1]}", "origin": hinge, "parent_origin": out[-1]["origin"]}
        seg["hulls"] = []
        seg["pieces"] = [(f"TwinFinS{j}", pts, None)]
        out.append(seg)
    return out


def segment_paths(link_path: str, chain: list[dict]) -> list[str]:
    """USD paths of the chain items: the link itself, then nested Xforms TwinFinSeg_<a|b|c>."""
    paths = [str(link_path)]
    for seg in chain[1:]:
        paths.append(f"{paths[-1]}/TwinFinSeg_{seg['name'][-1]}")
    return paths


def _ensure_xform(stage, path: str, translate: np.ndarray):
    """Define (or reuse) a segment Xform at `path` with a translate op (shared by the collider and visual passes,
    whichever runs first)."""
    from pxr import Gf, UsdGeom

    x = UsdGeom.Xform.Define(stage, path)
    ops = [op for op in x.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
    op = ops[0] if ops else x.AddTranslateOp()
    op.Set(Gf.Vec3d(*map(float, translate)))
    return x


def add_hinges(stage, link_path: str, which: str, chain: list[dict], spec: dict) -> list[str]:
    """Rigid bodies + revolute spring joints for the chain's segments (after their colliders exist). spec: the
    params.json gripper.finray.bend block (stiffness / damping per joint in N m/rad and N m s/rad, segment mass,
    limit). USD angular drives are per degree; the actuator written by the physics layer overrides them in SI."""
    import math

    from pxr import Gf, PhysxSchema, UsdPhysics

    paths = segment_paths(link_path, chain)
    names = []
    for j, seg in enumerate(chain[1:], start=1):
        body = stage.GetPrimAtPath(paths[j])
        UsdPhysics.RigidBodyAPI.Apply(body)
        UsdPhysics.MassAPI.Apply(body).CreateMassAttr().Set(float(spec["segment_mass_kg"]))
        PhysxSchema.PhysxRigidBodyAPI.Apply(body).CreateMaxDepenetrationVelocityAttr().Set(1.0)
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"{paths[j]}/{seg['name']}")
        joint.CreateAxisAttr().Set(HINGE_AXIS[which])
        joint.CreateBody0Rel().SetTargets([paths[j - 1]])
        joint.CreateBody1Rel().SetTargets([paths[j]])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*map(float, seg["origin"] - seg["parent_origin"])))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
        lim = float(spec.get("limit_deg", 45.0))
        joint.CreateLowerLimitAttr().Set(-lim)
        joint.CreateUpperLimitAttr().Set(lim)
        k, c = joint_gains(spec, j - 1, which)
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        drive.CreateTypeAttr().Set("force")
        drive.CreateStiffnessAttr().Set(k * math.pi / 180.0)
        drive.CreateDampingAttr().Set(c * math.pi / 180.0)
        drive.CreateTargetPositionAttr().Set(math.degrees(rest_angle(spec, j - 1, which)))
        g = curl_gearing(spec, j - 1, len(chain) - 1)
        if g:  # fin-ray curl: this hinge turns against the root hinge (q_j + g q_root = 0)
            mim = PhysxSchema.PhysxMimicJointAPI.Apply(joint.GetPrim(), "rot" + HINGE_AXIS[which])
            mim.CreateReferenceJointRel().SetTargets([f"{paths[1]}/{chain[1]['name']}"])
            mim.CreateGearingAttr().Set(float(g))
            mim.CreateOffsetAttr().Set(0.0)
        names.append(seg["name"])
    return names


CURL_SIGN = {"fixed": -1.0, "jaw": -1.0}  # raw hinge angle sign that turns each fin toward the other (fin_shape.py)


def rest_angle(spec: dict, i: int, which: str) -> float:
    """Unloaded angle (rad, raw joint convention) of hinge i: spec rest_curl_deg (scalar or list root -> tip,
    + = curled toward the other fin), 0 = the CAD shape. A curled-in tip closes the jaw's V at the bottom, so the
    squeeze draws the brick up into the fins as a fin-ray does."""
    v = spec.get("rest_curl_deg", 0.0) or 0.0
    d = float(v[min(i, len(v) - 1)] if isinstance(v, list) else v)
    return CURL_SIGN[which] * np.radians(d)


def curl_gearing(spec: dict, i: int, hinges: int) -> float:
    """Mimic gearing of hinge i (0 = root, never coupled): spec curl_gearing (scalar for the distal half of the
    hinges, or a list per hinge root -> tip); 0 = an independent spring hinge."""
    g = spec.get("curl_gearing", 0.0) or 0.0
    if i == 0:
        return 0.0
    if isinstance(g, list):
        return float(g[min(i, len(g) - 1)])
    return float(g) if i >= (hinges + 1) // 2 else 0.0


def joint_gains(spec: dict, i: int, which: str = "fixed") -> tuple[float, float]:
    """(stiffness N m/rad, damping N m s/rad) of hinge i (0 = root) of the `which` fin from the bend spec (scalars or
    per-joint lists; stiffness_jaw_nm_per_rad, when set, is the moving fin's own stiffness)."""
    def pick(v):
        return float(v[min(i, len(v) - 1)] if isinstance(v, list) else v)

    k = spec.get("stiffness_jaw_nm_per_rad") if which == "jaw" else None
    return pick(spec["stiffness_nm_per_rad"] if k is None else k), pick(spec["damping_nms_per_rad"])


def fin_shift(which: str, extend: float) -> np.ndarray:
    """Translation of the `which` ('fixed' | 'jaw') fin in its link frame for a fin mounted `extend` metres further
    out along its own axis (0 = the CAD mount; an experiment switch for fingertip-position studies)."""
    return _fin_axis(which) * float(extend)


def fixed_face_x(z: float) -> float:
    """x of the fixed fin's contact face at height z (gripper_link frame; the face toes in ~5 deg)."""
    f = geometry()["fixed_face"].astype(float)
    k = np.polyfit(f[:, 2], f[:, 0], 1)
    return float(np.polyval(k, z))


def tip_local(depth: float = 0.015, brick_w: float = BRICK_W, extend: float = 0.0) -> tuple:
    """Grasp point in the gripper_link frame: centre of a `brick_w` wide brick held against the fixed fin's contact
    face, `depth` up from the fin tip (the residual env's TIP_LOCAL for these fingers)."""
    shift = fin_shift("fixed", extend)
    tip = geometry()["fixed_tip"].astype(float) + shift
    z = float(tip[2]) + depth
    return (round(fixed_face_x(z - shift[2]) + shift[0] + brick_w / 2, 4), round(float(tip[1]), 4), round(z, 4))


def gap(q: float, depth: float) -> float:
    """Distance between the two contact faces `depth` up from the fixed fin tip at joint angle q (rigid fins)."""
    z = float(geometry()["fixed_tip"][2]) + depth
    jf = jaw_to_grip(geometry()["jaw_face"], q)
    k = np.polyfit(jf[:, 2], jf[:, 0], 1)
    return float(np.polyval(k, z)) - fixed_face_x(z)


def _define_mesh(stage, path: str, verts: np.ndarray, faces: np.ndarray, flat_normals: bool):
    from pxr import UsdGeom, Vt

    mesh = UsdGeom.Mesh.Define(stage, path)
    if flat_normals:  # un-welded triangles with face normals (CAD edges stay sharp)
        tri = verts[faces]
        nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
        pts = tri.reshape(-1, 3)
        idx = np.arange(len(pts), dtype=np.int32)
        mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.repeat(nrm, 3, 0).astype(np.float32)))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    else:
        pts, idx = verts, faces.reshape(-1).astype(np.int32)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(pts, np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(idx) // 3, 3, np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.ascontiguousarray(idx)))
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.CreateExtentAttr([tuple(map(float, pts.min(0))), tuple(map(float, pts.max(0)))])
    return mesh


def add_colliders(stage, grip_prim, jaw_prim, fin_material, base_material, contact_offset: float,
                  rest_offset: float, bind, extend: float = 0.0, bend: dict | None = None) -> dict:
    """Convex-hull colliders of both fingers under the physics links: the rigid printed bases (one hull each) and
    the fins (`n_segments()` hulls each, named TwinFin<k>). `bind(prim, material)` binds a physics material. With
    `bend` (params.json gripper.finray.bend, hinges > 0) the fin hulls go on hinged segments (see `segments`)."""
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    geo = geometry()
    hinges = int((bend or {}).get("hinges", 0))
    made, joints = [], []
    for which, link in (("fixed", grip_prim), ("jaw", jaw_prim)):
        chain = segments(which, hinges, extend, bool((bend or {}).get("slices", False)),
                         float((bend or {}).get("root_len", 0.012)))
        paths = segment_paths(link.GetPath(), chain)
        for j, seg in enumerate(chain[1:], start=1):
            _ensure_xform(stage, paths[j], seg["origin"] - seg["parent_origin"])
        pieces = [(str(link.GetPath()), "TwinFinBase", geo[f"{which}_base_v"], None, base_material)]
        for j, seg in enumerate(chain):
            pieces += [(paths[j], name, (pts - seg["origin"]).astype(np.float32), faces, fin_material)
                       for name, pts, faces in seg["pieces"]]
        for parent, name, verts, faces, mat in pieces:
            if faces is None:  # the hull approximation only reads the points; any valid face list will do
                faces = np.array([[0, k, k + 1] for k in range(1, len(verts) - 1)], np.int32)
            mesh = _define_mesh(stage, f"{parent}/{name}", verts.astype(np.float32), faces, False)
            mesh.CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
            prim = mesh.GetPrim()
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set("convexHull")
            px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            px.CreateContactOffsetAttr().Set(float(contact_offset))
            px.CreateRestOffsetAttr().Set(float(rest_offset))
            bind(prim, mat)
            made.append(name)
        if hinges:
            joints += add_hinges(stage, link.GetPath(), which, chain, bend)
    return {"pieces_per_env": len(made), "extend": float(extend), "fin_joints": joints,
            "fixed_tip": np.round(geo["fixed_tip"] + fin_shift("fixed", extend), 4).tolist()}


def _split_fin_visual(which: str, chain: list[dict], extend: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """The fin's visual mesh cut into the chain's parts (triangles by their centroid along the fin axis, against the
    hinge points), each in its part's frame: [(verts, faces)] root -> tip."""
    geo = geometry()
    v = geo[f"{which}_fin_v"].astype(float) + fin_shift(which, extend)
    f = geo[f"{which}_fin_f"]
    axis = _fin_axis(which)
    s = v[f].mean(1) @ axis
    bounds = [-np.inf] + [float(seg["origin"] @ axis) for seg in chain[1:]] + [np.inf]
    parts = []
    for j, seg in enumerate(chain):
        tri = f[(s >= bounds[j]) & (s < bounds[j + 1])]
        used, inv = np.unique(tri.reshape(-1), return_inverse=True)
        parts.append(((v[used] - seg["origin"]).astype(np.float32), inv.reshape(-1, 3).astype(np.int32)))
    return parts


def replace_visuals(stage, robot_root, body_material, fin_material, bind, extend: float = 0.0,
                    hinges: int = 0, slices: bool = False, root_len: float = 0.012) -> int:
    """Swap the stock jaw visuals of one robot (the fixed-finger follower and the moving jaw) for the printed bases
    (body material) and the red fins. With `hinges` > 0 the fin is cut into the bending segments and each part sits
    on its segment Xform (shared with the colliders). `bind(prim, material)` binds a render material. Returns the
    meshes added."""
    from pxr import Usd

    geo = geometry()
    links = {p.GetName(): p for p in Usd.PrimRange(robot_root)
             if p.GetName() in ("gripper_link", "moving_jaw_so101_v1_link")}
    n = 0
    for which, link_name, stock in (("fixed", "gripper_link", FIXED_VISUAL),
                                    ("jaw", "moving_jaw_so101_v1_link", JAW_VISUAL)):
        link = links.get(link_name)
        if link is None:
            continue
        for child in link.GetChildren():
            if child.GetName() == stock:
                child.SetActive(False)
        mesh = _define_mesh(stage, f"{link.GetPath()}/Finray_base", geo[f"{which}_base_v"].astype(np.float32),
                            geo[f"{which}_base_f"], True)
        bind(mesh.GetPrim(), body_material)
        n += 1
        chain = segments(which, hinges, extend, slices, root_len)
        paths = segment_paths(link.GetPath(), chain)
        for j, (verts, faces) in enumerate(_split_fin_visual(which, chain, extend)):
            if j:
                _ensure_xform(stage, paths[j], chain[j]["origin"] - chain[j]["parent_origin"])
            mesh = _define_mesh(stage, f"{paths[j]}/Finray_fin", verts, faces, True)
            bind(mesh.GetPrim(), fin_material)
            n += 1
    return n
