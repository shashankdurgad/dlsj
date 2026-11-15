"""Contact / brick realism keys for the twin (opt-in, default off). Called by sim/twin/physics/apply.py only when
params has a `contact` section (params.json has none; set keys with --set contact.<key>=json or TWIN_PHYSICS_SET):

    patch_scene_cfg(scene_cfg, P)   before InteractiveScene (rigid-body props of the brick cfgs)
    post_spawn(stage, n, P) -> dict after spawning, before sim.reset() (colliders, offsets, mass, scene settings)

Keys (all optional; a missing key leaves the twin unchanged):
  contact.brick_collider     "studs" | "shell" | "vhacd": replace the box / hull brick colliders by the LDraw shape
                             (sim/twin/contact/bricks.py): studs = body + 16-gon stud prisms; shell = deck + walls +
                             underside tubes + studs (curved: studs); vhacd = PhysX convex decomposition of the LDraw
                             mesh. Uses bricks.contact_offset / rest_offset and the brick's bound physics material.
  contact.brick_inertia      "mesh": centre of mass + principal inertia of the hollow LDraw solid (scaled to the mass)
  contact.brick_mass         "mesh": mass = LDraw volume x ABS 1050 kg/m^3 (default: bricks.mass_kg, BrickLink)
  contact.vhacd              {"max_hulls": 24, "hull_vertex_limit": 32, "voxel_resolution": 200000, "error": 1.0}
  contact.brick_ccd          true: swept CCD on the bricks (+ scene CCD); "speculative": speculative CCD only
  contact.brick_torsion_m    [patch_radius, min_patch_radius] (m) torsional friction on the brick shapes
  contact.table_offsets      [contact_offset, rest_offset] (m) on the table shapes
  contact.fin_offsets        [contact_offset, rest_offset] (m) on the fin-ray fins + bases (TwinFin*)
  contact.brick_offsets      [contact_offset, rest_offset] (m) on every brick shape (overrides bricks.*_offset)
  contact.scene              PhysX scene attributes {"frictionType": "patch" | "oneDirectional" | "twoDirectional",
                             "frictionOffsetThreshold": m, "frictionCorrelationDistance": m, "bounceThreshold": m/s,
                             "enableStabilization": bool, "enableEnhancedDeterminism": bool}
  contact.brick_material     extra PhysxMaterialAPI attributes on the brick material {"improvePatchFriction": bool}
  contact.report             true: PhysxContactReportAPI on the bricks (forensics, sim/twin/contact/gate_run.py)
  contact.tip_cap            soft / slippery fin tips (the gate's pushes are the fixed fin's tip slice pressing on the
                             brick's top face and dragging it, contact/forensics.py): the distal fin piece is cut
                             `len_m` from its tip along the fin axis; the end cap gets its own material
                             {"len_m": 0.003, "static_friction": 0.3, "dynamic_friction": 0.25,
                              "friction_combine": "average", "compliant_contact_stiffness": N/m (0 = rigid),
                              "compliant_contact_damping": N s/m, "sides": "both" | "fixed", "whole_slice": false}
                             (whole_slice: the whole distal slice gets the material, no cut). Note the bricks combine
                             `max`, so cap-brick friction is max(cap, bricks.material) = 0.45 / 0.38 at the defaults.
Solver iterations, depenetration velocity, sleep / stabilisation thresholds and friction are already params keys
(bricks.solver_position_iterations, bricks.max_depenetration_velocity, bricks.sleep_threshold, robot.*, *.material).
"""
from __future__ import annotations

import math

import numpy as np

from sim.twin.contact import bricks as B

BRICK_PREFIXES = ("Brick", "Distractor", "pool_")


# ------------------------------------------------------------------------------------------------ for the envs
def brick_collider(P: dict | None = None) -> str:
    """The active brick collider: "box" (params.json default) | "studs" | "shell" | "vhacd"."""
    if P is None:
        from sim.twin.physics import apply

        P = apply._p()
    return (_c(P).get("brick_collider") or "box") if P else "box"


def rest_lift_m(pose: str, P: dict | None = None) -> float:
    """Extra resting height (m) of a brick in `pose` over the box collider's: an upside-down ("down") brick stands on
    its 4.4 mm studs when the collider has them; up / side poses are unchanged. Env spawners (GraspEnv reset_envs /
    _place_distractors, TwinEnv BrickPool.place, gate POSE) add this to the centre height so nothing spawns 4.4 mm
    inside the table (PhysX would launch it at max_depenetration_velocity)."""
    return B.STUD_H if pose == "down" and brick_collider(P) in ("studs", "shell", "vhacd") else 0.0


def _c(P: dict) -> dict:
    return P.get("contact") or {}


def patch_scene_cfg(scene_cfg, P: dict) -> None:
    """Rigid-body CCD on the brick cfgs (other keys are stage edits in post_spawn)."""
    c = _c(P)
    if c.get("brick_ccd") is None:
        return
    for name in list(vars(scene_cfg)):
        sp = getattr(getattr(scene_cfg, name), "spawn", None)
        rp = getattr(sp, "rigid_props", None)
        if rp is None or not (name == "brick" or name.startswith("distractor")):
            continue
        if c["brick_ccd"] == "speculative":
            rp.enable_speculative_ccd = True
        else:
            rp.enable_ccd = bool(c["brick_ccd"])


# ------------------------------------------------------------------------------------------------ stage helpers
def _brick_roots(stage, n: int) -> list:
    """(env, root prim) of every rigid brick body directly under the env prims."""
    from pxr import UsdPhysics

    out = []
    for e in range(n):
        env = stage.GetPrimAtPath(f"/World/envs/env_{e}")
        if not env:
            continue
        for p in env.GetChildren():
            if p.GetName().startswith(BRICK_PREFIXES) and p.HasAPI(UsdPhysics.RigidBodyAPI):
                out.append((e, p))
    return out


def _colliders(root) -> list:
    from pxr import Usd, UsdPhysics

    return [p for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.CollisionAPI)]


def _kind(root) -> str:
    """Brick kind from the root's local bound (the cuboid size, or the curved USD's body)."""
    from pxr import Usd, UsdGeom

    if "curved" in root.GetName().lower():
        return "curved_2x3"
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.guide])
    rng = cache.ComputeUntransformedBound(root).ComputeAlignedRange()
    return B.kind_from_size(np.asarray(rng.GetSize()))


def _offsets(prim, co, ro) -> None:
    from pxr import PhysxSchema

    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    if co is not None:
        px.CreateContactOffsetAttr().Set(float(co))
    if ro is not None:
        px.CreateRestOffsetAttr().Set(float(ro))


def _bound_physics_material(prims):
    from pxr import UsdShade

    for p in prims:
        mat, _rel = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial("physics")
        if mat:
            return mat
    return None


def _convex_mesh(stage, path: str, pts: np.ndarray):
    from pxr import UsdGeom, UsdPhysics, Vt

    F = B.hull_faces(pts)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(F)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([int(v) for v in F.reshape(-1)]))
    mesh.CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
    prim = mesh.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set("convexHull")
    return prim


def _vhacd_mesh(stage, path: str, kind: str, cfg: dict):
    from pxr import PhysxSchema, UsdGeom, UsdPhysics, Vt

    s = B.ldraw_solid(kind)
    if s is None:
        raise FileNotFoundError("contact.brick_collider=vhacd needs the LDraw library (scene/ldraw.py fetch)")
    V, F = s
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(V.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(F)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([int(v) for v in F.reshape(-1)]))
    mesh.CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
    prim = mesh.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set("convexDecomposition")
    cd = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
    cd.CreateMaxConvexHullsAttr().Set(int(cfg.get("max_hulls", 24)))
    cd.CreateHullVertexLimitAttr().Set(int(cfg.get("hull_vertex_limit", 32)))
    cd.CreateVoxelResolutionAttr().Set(int(cfg.get("voxel_resolution", 200000)))
    cd.CreateErrorPercentageAttr().Set(float(cfg.get("error", 1.0)))
    cd.CreateShrinkWrapAttr().Set(True)
    return prim


def _replace_collider(stage, root, kind: str, variant: str, P: dict, cfg: dict) -> int:
    """Disable the brick's own collider(s) and add the LDraw-shaped ones under <root>/ContactCollider."""
    from pxr import UsdGeom, UsdPhysics, UsdShade

    old = _colliders(root)
    mat = _bound_physics_material(old) or _bound_physics_material([root])
    for p in old:
        UsdPhysics.CollisionAPI(p).CreateCollisionEnabledAttr().Set(False)
    base = str(root.GetPath()) + "/ContactCollider"
    UsdGeom.Xform.Define(stage, base)
    if variant == "vhacd":
        prims = [_vhacd_mesh(stage, base + "/ldraw", kind, cfg.get("vhacd") or {})]
    else:
        prims = [_convex_mesh(stage, f"{base}/{name}", pts) for name, pts in B.parts(kind, variant)]
    co = (cfg.get("brick_offsets") or [P["bricks"]["contact_offset"], None])[0]
    ro = (cfg.get("brick_offsets") or [None, P["bricks"]["rest_offset"]])[1]
    for prim in prims:
        _offsets(prim, co, ro)
        if mat:
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")
    return len(prims)


def _mass(root, kind: str, P: dict, cfg: dict) -> dict:
    from pxr import Gf, UsdPhysics

    ma = UsdPhysics.MassAPI.Apply(root)
    mass = ma.GetMassAttr().Get()
    if cfg.get("brick_mass") == "mesh":
        mass = None
    elif not mass:
        mass = float(P["bricks"]["mass_kg"].get(kind, 0.00675))
    mp = B.mass_props(kind, mass)
    ma.CreateMassAttr().Set(float(mp["mass"]))
    if cfg.get("brick_inertia") == "mesh":
        ma.CreateCenterOfMassAttr().Set(Gf.Vec3f(*map(float, mp["com"])))
        ma.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*map(float, mp["inertia"])))
        ma.CreatePrincipalAxesAttr().Set(Gf.Quatf(*map(float, mp["axes_wxyz"])))
    return {"mass_g": round(mp["mass"] * 1e3, 2), "com_mm": np.round(mp["com"] * 1e3, 2).tolist(),
            "inertia_kgmm2": np.round(mp["inertia"] * 1e6, 4).tolist()}


def _scene_attrs(stage, attrs: dict) -> dict:
    from pxr import PhysxSchema, Usd, UsdPhysics

    done = {}
    for p in Usd.PrimRange(stage.GetPseudoRoot()):
        if not p.IsA(UsdPhysics.Scene):
            continue
        api = PhysxSchema.PhysxSceneAPI.Apply(p)
        for k, v in attrs.items():
            getattr(api, f"Create{k[0].upper()}{k[1:]}Attr")().Set(v)
        done[str(p.GetPath())] = attrs
    return done


def _clip(P: np.ndarray, s: np.ndarray, cut: float, keep_low: bool) -> np.ndarray:
    """Points of the convex hull of P cut by the plane s = cut (keep s <= cut, or s >= cut): the kept points plus
    every hull-edge crossing (all kept x dropped pairs; a superset of the cut polygon's vertices)."""
    try:
        from scipy.spatial import ConvexHull

        v = ConvexHull(P).vertices
        P, s = P[v], s[v]
    except Exception:  # noqa: BLE001
        pass
    keep = s <= cut if keep_low else s >= cut
    A, sA, Bp, sB = P[keep], s[keep], P[~keep], s[~keep]
    if not len(Bp):
        return P
    t = ((cut - sA)[:, None] / (sB[None, :] - sA[:, None]))[..., None]
    X = (A[:, None, :] + t * (Bp[None, :, :] - A[:, None, :])).reshape(-1, 3)
    X = np.unique(np.round(X, 7), axis=0)
    return np.r_[A, X]


def _xf(prim) -> np.ndarray:
    from pxr import Usd, UsdGeom

    return np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))


def _tip_caps(stage, n: int, spec: dict) -> dict:
    """Split the distal fin piece of each fin into body + tip cap with its own (slippery / soft) material."""
    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

    from sim.twin.gripper import model

    mat = UsdShade.Material.Define(stage, "/World/PhysicsMaterials/ContactFinTip")
    UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm = UsdPhysics.MaterialAPI(mat.GetPrim())
    pm.CreateStaticFrictionAttr().Set(float(spec.get("static_friction", 0.3)))
    pm.CreateDynamicFrictionAttr().Set(float(spec.get("dynamic_friction", 0.25)))
    pm.CreateRestitutionAttr().Set(0.0)
    px = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
    px.CreateFrictionCombineModeAttr().Set(spec.get("friction_combine", "average"))
    if spec.get("compliant_contact_stiffness"):
        px.CreateCompliantContactStiffnessAttr().Set(float(spec["compliant_contact_stiffness"]))
        px.CreateCompliantContactDampingAttr().Set(float(spec.get("compliant_contact_damping", 0.0)))
    L = float(spec.get("len_m", 0.003))
    sides = ("fixed",) if spec.get("sides") == "fixed" else ("fixed", "jaw")
    info = {"caps": 0}
    for e in range(n):
        root = stage.GetPrimAtPath(f"/World/envs/env_{e}/Robot")
        if not root:
            continue
        links = {p.GetName(): p for p in Usd.PrimRange(root) if p.GetName() in ("gripper_link",
                                                                               "moving_jaw_so101_v1_link")}
        for which in sides:
            link = links["gripper_link" if which == "fixed" else "moving_jaw_so101_v1_link"]
            jaw_path = str(links["moving_jaw_so101_v1_link"].GetPath())
            axis = model._fin_axis(which)
            best = None
            for p in Usd.PrimRange(link):
                nm = p.GetName()
                if not (p.IsA(UsdGeom.Mesh) and p.HasAPI(UsdPhysics.CollisionAPI) and nm.startswith("TwinFin")
                        and not nm.startswith(("TwinFinBase", "TwinFinTip"))):
                    continue
                if which == "fixed" and str(p.GetPath()).startswith(jaw_path):
                    continue
                pts = np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get(), float)
                T = _xf(p) @ np.linalg.inv(_xf(link))
                PL = (np.c_[pts, np.ones(len(pts))] @ T)[:, :3]
                smax = float((PL @ axis).max())
                if best is None or smax > best[0]:
                    best = (smax, p, PL, T)
            if best is None:
                continue
            smax, prim, PL, T = best
            if spec.get("whole_slice"):
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")
                info["caps"] += 1
                continue
            s = PL @ axis
            Ti = np.linalg.inv(T)

            def local(Q, Ti=Ti):
                return (np.c_[Q, np.ones(len(Q))] @ Ti)[:, :3].astype(np.float32)

            body, cap = _clip(PL, s, smax - L, True), _clip(PL, s, smax - L, False)
            mesh = UsdGeom.Mesh(prim)
            mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(local(body)))
            mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * (len(body) - 2)))
            mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray([v for k in range(1, len(body) - 1)
                                                             for v in (0, k, k + 1)]))
            mesh.CreateExtentAttr([tuple(map(float, local(body).min(0))), tuple(map(float, local(body).max(0)))])
            cpath = str(prim.GetPath().GetParentPath()) + "/TwinFinTipCap"
            cm = UsdGeom.Mesh.Define(stage, cpath)
            cl = local(cap)
            # the cap lives next to the distal piece: same parent, the distal piece's local transform folded in
            Mloc = np.asarray(UsdGeom.Xformable(prim).GetLocalTransformation())
            cl = (np.c_[cl, np.ones(len(cl))] @ Mloc)[:, :3].astype(np.float32)
            cm.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(cl))
            cm.CreateFaceVertexCountsAttr(Vt.IntArray([3] * (len(cl) - 2)))
            cm.CreateFaceVertexIndicesAttr(Vt.IntArray([v for k in range(1, len(cl) - 1) for v in (0, k, k + 1)]))
            cm.CreateExtentAttr([tuple(map(float, cl.min(0))), tuple(map(float, cl.max(0)))])
            cm.CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
            cp = cm.GetPrim()
            UsdPhysics.CollisionAPI.Apply(cp)
            UsdPhysics.MeshCollisionAPI.Apply(cp).CreateApproximationAttr().Set("convexHull")
            opx = PhysxSchema.PhysxCollisionAPI(prim)
            _offsets(cp, opx.GetContactOffsetAttr().Get() if opx else None,
                     opx.GetRestOffsetAttr().Get() if opx else None)
            UsdShade.MaterialBindingAPI.Apply(cp).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")
            info["caps"] += 1
            if e == 0:
                info[which] = {"piece": prim.GetName(), "cap_points": len(cl), "body_points": len(body),
                               "cap_len_mm": round(L * 1e3, 1)}
    return info


def post_spawn(stage, n: int, P: dict) -> dict:
    """Stage edits for the contact keys (before sim.reset()). Returns a summary for the logs."""
    from pxr import PhysxSchema, Usd, UsdPhysics

    cfg = _c(P)
    info: dict = {"keys": sorted(cfg)}
    roots = _brick_roots(stage, n)
    variant = cfg.get("brick_collider")
    kinds: dict = {}
    for e, root in roots:
        kind = _kind(root)
        if e == 0:
            kinds[root.GetName()] = kind
        if variant and variant != "box":
            k = _replace_collider(stage, root, kind, variant, P, cfg)
            if e == 0:
                info.setdefault("collider_pieces", {})[root.GetName()] = k
        elif cfg.get("brick_offsets"):
            for p in _colliders(root):
                _offsets(p, *cfg["brick_offsets"])
        if cfg.get("brick_inertia") == "mesh" or cfg.get("brick_mass") == "mesh":
            m = _mass(root, kind, P, cfg)
            if e == 0:
                info.setdefault("mass", {})[root.GetName()] = m
        if cfg.get("brick_torsion_m"):
            r, rmin = cfg["brick_torsion_m"]
            for p in _colliders(root):
                if p.GetAttribute("physics:collisionEnabled").Get() is False:
                    continue
                px = PhysxSchema.PhysxCollisionAPI.Apply(p)
                px.CreateTorsionalPatchRadiusAttr().Set(float(r))
                px.CreateMinTorsionalPatchRadiusAttr().Set(float(rmin))
        if cfg.get("report"):
            cr = PhysxSchema.PhysxContactReportAPI.Apply(root)
            cr.CreateThresholdAttr().Set(0.0)
    info["kinds"] = kinds
    if cfg.get("table_offsets") or cfg.get("fin_offsets"):
        nt = nf = 0
        for e in range(n):
            env = stage.GetPrimAtPath(f"/World/envs/env_{e}")
            if not env:
                continue
            for p in Usd.PrimRange(env):
                path = str(p.GetPath())
                if not p.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                if cfg.get("table_offsets") and "/Table" in path:
                    _offsets(p, *cfg["table_offsets"])
                    nt += 1
                elif cfg.get("fin_offsets") and p.GetName().startswith("TwinFin"):
                    _offsets(p, *cfg["fin_offsets"])
                    nf += 1
        info["offset_shapes"] = {"table": nt, "fins": nf}
    if cfg.get("brick_material"):
        from pxr import UsdShade

        mats = set()
        for _e, root in roots:
            m = _bound_physics_material(_colliders(root) or [root])
            if m and str(m.GetPath()) not in mats:
                mats.add(str(m.GetPath()))
                px = PhysxSchema.PhysxMaterialAPI.Apply(UsdShade.Material(m).GetPrim())
                for k, v in cfg["brick_material"].items():
                    getattr(px, f"Create{k[0].upper()}{k[1:]}Attr")().Set(v)
        info["brick_materials"] = sorted(mats)
    if cfg.get("tip_cap"):
        info["tip_cap"] = _tip_caps(stage, n, dict(cfg["tip_cap"]))
    scene = dict(cfg.get("scene") or {})
    if cfg.get("brick_ccd") not in (None, False, "speculative"):
        scene.setdefault("enableCCD", True)
    if scene:
        info["scene"] = _scene_attrs(stage, scene)
    return info


def describe_offsets(stage, n: int = 1) -> dict:
    """Effective contact / rest offsets (authored; None = PhysX default) and materials of env_0's shapes."""
    from pxr import PhysxSchema, Usd, UsdPhysics, UsdShade

    out = {}
    env = stage.GetPrimAtPath("/World/envs/env_0")
    for p in Usd.PrimRange(env, Usd.TraverseInstanceProxies()):
        if not p.HasAPI(UsdPhysics.CollisionAPI):
            continue
        en = p.GetAttribute("physics:collisionEnabled").Get()
        px = PhysxSchema.PhysxCollisionAPI(p)
        co = px.GetContactOffsetAttr().Get() if px else None
        ro = px.GetRestOffsetAttr().Get() if px else None
        mat, _ = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial("physics")
        key = str(p.GetPath()).replace("/World/envs/env_0/", "")
        out[key] = {"enabled": en, "contact_offset": None if co is None or math.isinf(co) else round(co, 5),
                    "rest_offset": None if ro is None or math.isinf(ro) else round(ro, 5),
                    "material": str(mat.GetPath()) if mat else None}
    return out
