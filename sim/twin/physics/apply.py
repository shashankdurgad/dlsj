"""Twin physics layer: grasp / contact realism for the SO-101 Duplo station (Isaac Lab 3.0).

Interface for the core env (sim/twin/env/twin_env.py):
    patch_scene_cfg(scene_cfg, profile) -> scene_cfg   before InteractiveScene: actuator effort, materials, masses,
                                                       contact offsets, plate collider swap; sets profile.grip_bias
    post_spawn(env)                                    after spawning, before sim.reset(): finger colliders, plate
                                                       colliders (stage edits)
    patch_profile(profile) -> profile                  gripper unit mapping that matches the finger colliders
    sample_physics_dr(rng) -> dict / apply_physics_dr(scene, draw)   physics domain randomisation
All numbers live in params.json next to this file (load_params(path, overrides) to change them).

What it changes and why (measurements: validate_physics.py, README in this folder):
0. Fingers (`gripper.collider = "finray"`, the default): the real station has the XLeRobot soft gripper, printed
   bases + TPU fin-ray fins (sim/twin/gripper). Convex fin pieces with PhysX compliant contact (TPU give, no solver
   buzz), TPU-on-ABS friction, and the residual env's fingertip point (TIP_LOCAL) moved to these fins. The modes
   below are the stock rigid jaw, kept for comparison.
1. Finger colliders. The twin robot USD (URDF converter v0.3.2) instances its collision meshes: the convex hull
   colliders of the fixed finger (`wrist_roll_follower_so101_v1_1/...`) and moving jaw (`moving_jaw_so101_v1_1/...`)
   are instance proxies, so the residual env's `app.finger_pads` cannot disable them (it applies CollisionAPI to the
   VISUAL instance Xforms instead, which PhysX then parses as an extra convex hull). Result: hulls fill the jaw gap
   and push the brick out. Here: the teammate's pads (if any) are removed, the collision instances are deactivated,
   and the jaw is rebuilt from the CAD meshes (`gripper.collider`):
     pads_cad              boxes fitted to the CAD fingertips (inner face exact, full finger thickness/width/length)
     convex_decomposition  the CAD meshes de-instanced, V-HACD convex decomposition (cooks on Isaac Sim 6.2)
     sdf                   the CAD meshes as SDF colliders
2. Gripper mapping. With CAD-accurate inner faces the jaw stalls on a 31.7 mm brick where Joe's cal2 calibration
   says it should (q = units * 0.02308 + 0.105 rad: "zero bias estimated against 32 mm CAD-jaw contact and recorded
   grasp state near 6"); the residual env's re-solved bias (-0.0263) only compensated its pads. `gripper.bias_rad`.
3. Gripper effort: LeRobot's SO-101 follower limits the gripper servo to ~50 % torque / current
   (Max_Torque_Limit 500, Protection_Current 250) -> ~1.0-1.5 N m of a ~2.9 N m STS3215 instead of the arm's 2.9.
4. Bricks: Duplo masses (BrickLink: 2x2 3437 = 6.75 g, 2x4 3011 ~12 g; the env had 9.3 / 18 g), ABS friction,
   low restitution, more solver iterations, contact / rest offsets.
5. Plates: the env's flat 6 mm cylinder collider is disabled and the measured plate profile (visuals/assets.py:
   floor, sloped wall, 23 mm rim, 2 mm shell) becomes a triangle-mesh collider on the kinematic plate body.
"""
from __future__ import annotations

import copy
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PARAMS_JSON = HERE / "params.json"
PARAMS: dict = {}
FIXED_FINGER = "wrist_roll_follower_so101_v1"
MOVING_JAW = "moving_jaw_so101_v1"
GRIP_LINK = "gripper_link"
JAW_LINK = "moving_jaw_so101_v1_link"


# ------------------------------------------------------------------------------------------------ params
def load_params(path: str | Path | None = None, overrides=()) -> dict:
    """Load params.json (+ 'dotted.key=json' overrides) into PARAMS and return it. The environment variable
    TWIN_PHYSICS_SET ('key=json;key=json') adds overrides for scripts without a --set option, e.g.
    TWIN_PHYSICS_SET=gripper.collider=pads_cad for the stock-jaw benches."""
    p = json.loads(Path(path or PARAMS_JSON).read_text())
    env_ov = [o for o in os.environ.get("TWIN_PHYSICS_SET", "").split(";") if o]
    for ov in [*env_ov, *(overrides or ())]:
        key, val = ov.split("=", 1)
        try:
            v = json.loads(val)
        except json.JSONDecodeError:
            v = val
        d = p
        ks = key.split(".")
        for k in ks[:-1]:
            d = d.setdefault(k, {})
        d[ks[-1]] = v
    PARAMS.clear()
    PARAMS.update(p)
    return PARAMS


def _p() -> dict:
    if not PARAMS:
        load_params()
    return PARAMS


# ------------------------------------------------------------------------------------------------ profile
def finray() -> bool:
    """True when the fin-ray fingers (sim/twin/gripper) are the active finger model."""
    return _p()["gripper"]["collider"] == "finray"


def jaw_material() -> dict:
    """Physics material spec of the active finger contact surfaces."""
    g = _p()["gripper"]
    return g["finray"]["material"] if finray() else g["material"]


def fin_bend() -> dict | None:
    """The bending-fin spec (params.json gripper.finray.bend) when the fin-ray fingers are on and hinged, else None."""
    if not finray():
        return None
    b = _p()["gripper"]["finray"].get("bend") or {}
    return b if int(b.get("hinges", 0)) > 0 else None


def grip_effort() -> float:
    """Gripper servo effort limit (N m) for the active fingers."""
    g = _p()["gripper"]
    e = g["finray"].get("effort_nm") if finray() else None
    return float(g["effort_nm"] if e is None else e)


def tip_local() -> tuple:
    """The fingertip / grasp point in the gripper_link frame for the active fingers (residual env TIP_LOCAL)."""
    if finray():
        from sim.twin.gripper import model

        f = _p()["gripper"]["finray"]
        if f.get("tip_local") is not None:  # explicit point (x, y, z) instead of the CAD-derived one
            return tuple(f["tip_local"])
        return model.tip_local(float(f["grasp_depth"]), extend=float(f.get("extend", 0.0)))
    return tuple(_p()["gripper"].get("tip_local", (0.008, 0.0, -0.090)))


def patch_profile(profile):
    """Set the gripper bias that matches the finger colliders (in place; also returned), and the residual env's
    fingertip point: its module constant TIP_LOCAL (read by ResidualEnv.__init__ into `tip_local`, used by
    `tip_w()`, the grasp check and the replays' brick placement)."""
    g = _p()["gripper"]
    b = g["finray"].get("bias_rad") if finray() else None
    b = g.get("bias_rad") if b is None else b
    if profile is not None and b is not None:
        profile.grip_bias = float(b)
    offs = (_p().get("robot") or {}).get("joint_offsets_rad")
    if profile is not None and offs:  # e.g. the joint fit's W9 (align/joint_fit/result.json grasp_depth_candidates)
        offs = [float(v) for v in offs][:6]
        profile.joint_offsets = offs + [0.0] * (6 - len(offs))
    renv = sys.modules.get("env")  # sim/residual_env/env.py, when the caller has it on sys.path
    if renv is None or not hasattr(renv, "TIP_LOCAL"):
        try:
            import env as renv
        except ImportError:
            renv = None
    if renv is not None and hasattr(renv, "TIP_LOCAL"):
        renv.TIP_LOCAL = tuple(float(v) for v in tip_local())
    return profile


# ------------------------------------------------------------------------------------------------ scene cfg
def _material(spec: dict):
    import isaaclab.sim as sim_utils

    kw = {}
    if spec.get("compliant_contact_stiffness"):  # spring contact (N/m), e.g. the table under the pressing fin tips
        kw = {"compliant_contact_stiffness": float(spec["compliant_contact_stiffness"]),
              "compliant_contact_damping": float(spec.get("compliant_contact_damping", 0.0))}
    return sim_utils.RigidBodyMaterialCfg(
        static_friction=spec["static_friction"], dynamic_friction=spec["dynamic_friction"],
        restitution=spec.get("restitution", 0.0), friction_combine_mode=spec.get("friction_combine", "average"),
        restitution_combine_mode=spec.get("restitution_combine", "average"), **kw)


def _brick_kind(size) -> str:
    """Mass class from the footprint's long side: 2x2 (31.7 mm), curved_2x3 (LDraw 2302b, 47.7 mm), 2x4 (63.6 mm)."""
    long_side = max(size[0], size[1])
    if long_side > 0.056:
        return "2x4"
    return "curved_2x3" if long_side >= 0.040 else "2x2"


def patch_scene_cfg(scene_cfg, profile=None):
    """Actuator effort, materials, brick masses / contacts, plate cylinders without collision (post_spawn adds the
    profile collider). Works on the residual env's / twin's InteractiveSceneCfg (attribute names robot, table,
    brick, distractor*, bowl_*)."""
    import isaaclab.sim as sim_utils

    P = _p()
    g = P["gripper"]
    act = scene_cfg.robot.actuators.get("gripper")
    bend = fin_bend()
    if bend is not None and act is not None:  # the fin hinges (sim/twin/gripper/model.py) as spring-damper joints
        from sim.twin.gripper import model as fin_model

        names = [(w, i, f"fin_{w}_{fin_model.SEG_NAMES[i]}") for w in ("fixed", "jaw")
                 for i in range(int(bend["hinges"]))]
        k = {n: fin_model.joint_gains(bend, i, w)[0] for w, i, n in names}
        c = {n: fin_model.joint_gains(bend, i, w)[1] for w, i, n in names}
        scene_cfg.robot.actuators["fins"] = type(act)(joint_names_expr=["fin_.*"], stiffness=k, damping=c,
                                                      effort_limit_sim=50.0, velocity_limit_sim=100.0)
        rest = {n: fin_model.rest_angle(bend, i, w) for w, i, n in names}
        if any(rest.values()):  # pre-curled hinges: their default position is the drive target the env holds
            jp = dict(scene_cfg.robot.init_state.joint_pos or {})
            jp.update(rest)
            scene_cfg.robot.init_state.joint_pos = jp
    if act is not None:
        act.effort_limit_sim = grip_effort()
        if g.get("stiffness") is not None:
            act.stiffness = float(g["stiffness"])
        if g.get("damping") is not None:
            act.damping = float(g["damping"])
    R = P.get("robot") or {}
    arm = scene_cfg.robot.actuators.get("arm")
    if arm is not None and R.get("arm_stiffness") is not None:  # softer arm servos (dict joint -> N m/rad or scalar)
        arm.stiffness = R["arm_stiffness"]
    ap = getattr(scene_cfg.robot.spawn, "articulation_props", None)
    if ap is not None:
        if R.get("solver_position_iterations"):
            ap.solver_position_iteration_count = int(R["solver_position_iterations"])
        if R.get("solver_velocity_iterations"):
            ap.solver_velocity_iteration_count = int(R["solver_velocity_iterations"])
    if R.get("max_depenetration_velocity") is not None:
        rp = scene_cfg.robot.spawn.rigid_props or sim_utils.RigidBodyPropertiesCfg()
        rp.max_depenetration_velocity = float(R["max_depenetration_velocity"])
        scene_cfg.robot.spawn.rigid_props = rp
    if getattr(scene_cfg, "table", None) is not None and P.get("table"):
        scene_cfg.table.spawn.physics_material = _material(P["table"])
    B = P["bricks"]
    for name in list(vars(scene_cfg)):
        obj = getattr(scene_cfg, name)
        sp = getattr(obj, "spawn", None)
        if sp is None:
            continue
        if name == "brick" or name.startswith("distractor"):
            kind = _brick_kind(sp.size)
            sp.mass_props = sim_utils.MassPropertiesCfg(mass=float(B["mass_kg"][kind]))
            sp.physics_material = _material(B["material"])
            sp.collision_props = sim_utils.CollisionPropertiesCfg(contact_offset=B["contact_offset"],
                                                                  rest_offset=B["rest_offset"])
            sp.rigid_props = sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=B["max_depenetration_velocity"],
                solver_position_iteration_count=B["solver_position_iterations"],
                solver_velocity_iteration_count=B["solver_velocity_iterations"],
                sleep_threshold=B.get("sleep_threshold"), stabilization_threshold=B.get("stabilization_threshold"))
        elif name.startswith("bowl_"):
            pl = P["plates"]
            sp.physics_material = _material(pl["material"])
            if pl["collider"] == "profile_mesh":
                sp.collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=False)
    patch_profile(profile)
    return scene_cfg


# ------------------------------------------------------------------------------------------------ post spawn
def _num_envs(env) -> int:
    if isinstance(env, int):
        return env
    for path in ("num_envs", "scene.num_envs", "cfg.num_envs"):
        o = env
        try:
            for a in path.split("."):
                o = getattr(o, a)
            return int(o)
        except AttributeError:
            continue
    raise ValueError("post_spawn: cannot find num_envs on %r" % (env,))


def post_spawn(env) -> dict:
    """Stage edits before sim.reset(): finger colliders, plate colliders. `env` = TwinEnv / InteractiveScene /
    anything with num_envs (or an int)."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    n = _num_envs(env)
    info = {"gripper": _finger_colliders(stage, n), "plates": _plate_colliders(stage, n)}
    margin = float((_p().get("robot") or {}).get("limit_margin_deg", 0.0) or 0.0)
    if margin:
        info["joint_limits"] = _widen_limits(stage, n, margin)
    return info


ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


def _widen_limits(stage, n: int, margin_deg: float) -> dict:
    """Open the arm joints' URDF position limits by `margin_deg` on each side (params robot.limit_margin_deg;
    robot.limit_margin_joints picks the joints, default the five arm joints). The cal2 demos go past the URDF's
    wrist_flex +95 deg (8.5 % of frames, up to 3.5 deg) and shoulder_lift -100 deg (1.2 %, up to 6.9 deg), and PhysX
    clamps there. Authored on the stage before sim.reset(), so the shared USD is unchanged."""
    from pxr import Usd, UsdPhysics

    names = set((_p().get("robot") or {}).get("limit_margin_joints") or ARM_JOINTS)
    out = {}
    for e in range(n):
        root = stage.GetPrimAtPath(f"/World/envs/env_{e}/Robot")
        if not root:
            continue
        for prim in Usd.PrimRange(root):
            if prim.GetName() in names and prim.IsA(UsdPhysics.RevoluteJoint):
                j = UsdPhysics.RevoluteJoint(prim)
                lo, hi = j.GetLowerLimitAttr().Get(), j.GetUpperLimitAttr().Get()
                if lo is None or hi is None:
                    continue
                j.GetLowerLimitAttr().Set(float(lo) - margin_deg)
                j.GetUpperLimitAttr().Set(float(hi) + margin_deg)
                if e == 0:
                    out[prim.GetName()] = [round(float(lo) - margin_deg, 2), round(float(hi) + margin_deg, 2)]
    return out


def _physics_material(stage, path: str, spec: dict):
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr().Set(float(spec["static_friction"]))
    pm.CreateDynamicFrictionAttr().Set(float(spec["dynamic_friction"]))
    pm.CreateRestitutionAttr().Set(float(spec.get("restitution", 0.0)))
    px = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
    px.CreateFrictionCombineModeAttr().Set(spec.get("friction_combine", "average"))
    px.CreateRestitutionCombineModeAttr().Set(spec.get("restitution_combine", "average"))
    if spec.get("compliant_contact_stiffness"):  # spring contact (N/m per contact point) instead of a hard one
        px.CreateCompliantContactStiffnessAttr().Set(float(spec["compliant_contact_stiffness"]))
        px.CreateCompliantContactDampingAttr().Set(float(spec.get("compliant_contact_damping", 0.0)))
    return mat


def _bind_phys(prim, mat):
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def _mesh_points_in(prim_mesh, frame_prim) -> np.ndarray:
    """Mesh points (instance proxies fine) in the frame of `frame_prim` (authored pose)."""
    from pxr import Usd, UsdGeom

    pts = np.asarray(UsdGeom.Mesh(prim_mesh).GetPointsAttr().Get(), float)
    T = UsdGeom.Xformable(prim_mesh).ComputeLocalToWorldTransform(Usd.TimeCode.Default()) * \
        UsdGeom.Xformable(frame_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()
    return (np.c_[pts, np.ones(len(pts))] @ np.asarray(T))[:, :3]


def _collision_instance(link, base: str):
    """(instance Xform, collision mesh proxy) of `base` under `link` (the instance whose mesh has CollisionAPI)."""
    from pxr import Usd, UsdGeom, UsdPhysics

    for child in link.GetChildren():
        if not child.GetName().startswith(base):
            continue
        for p in Usd.PrimRange(child, Usd.TraverseInstanceProxies()):
            if p.IsA(UsdGeom.Mesh) and p.HasAPI(UsdPhysics.CollisionAPI):
                return child, p
    return None, None


def _fit_pad(pts_g: np.ndarray, inner_sign: float, cfg: dict):
    """Box (centre, half) in the gripper_link frame around the CAD fingertip: z from the tip up `pad_length`,
    x from the inner face to the outer side (capped at `max_thickness`), y over the finger width."""
    zmin = pts_g[:, 2].min()
    sel = pts_g[pts_g[:, 2] < zmin + cfg["pad_length"]]
    x = sel[:, 0]
    if inner_sign > 0:  # fixed finger (negative x side): inner face = max x
        x_in = x.max()
        x_out = max(x.min(), x_in - cfg["max_thickness"])
    else:  # moving jaw: inner face = min x
        x_in = x.min()
        x_out = min(x.max(), x_in + cfg["max_thickness"])
    lo = np.array([min(x_in, x_out), sel[:, 1].min(), zmin])
    hi = np.array([max(x_in, x_out), sel[:, 1].max(), zmin + cfg["pad_length"]])
    return (lo + hi) / 2, (hi - lo) / 2, float(x_in)


def _finger_colliders(stage, n: int) -> dict:
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

    g = _p()["gripper"]
    mode = g["collider"]
    mat = _physics_material(stage, "/World/PhysicsMaterials/TwinJaw", g["material"])
    out = {"mode": mode, "envs": 0}
    for e in range(n):
        root = stage.GetPrimAtPath(f"/World/envs/env_{e}/Robot")
        if not root:
            continue
        prims = {p.GetName(): p for p in Usd.PrimRange(root)}
        grip, jaw = prims.get(GRIP_LINK), prims.get(JAW_LINK)
        if grip is None or jaw is None:
            continue
        # undo app.finger_pads: its boxes and the CollisionAPI it put on the visual instance Xforms
        for link in (grip, jaw):
            for c in link.GetChildren():
                if c.GetName() in ("FixedPad", "MovingPad"):
                    stage.RemovePrim(c.GetPath())
        for name in (FIXED_FINGER, MOVING_JAW):
            p = prims.get(name)
            if p is not None and p.HasAPI(UsdPhysics.CollisionAPI):
                p.RemoveAPI(UsdPhysics.CollisionAPI)
        if mode == "hull":  # leave the USD's convex hulls (for comparison only)
            out["envs"] += 1
            continue
        if mode == "finray":  # printed bases + TPU fins (sim/twin/gripper) instead of the stock jaw hulls
            from sim.twin.gripper import model

            for link, base in ((grip, FIXED_FINGER), (jaw, MOVING_JAW)):
                inst, _mesh = _collision_instance(link, base)
                if inst is not None:
                    inst.SetActive(False)
            fin_mat = _physics_material(stage, "/World/PhysicsMaterials/TwinFin", g["finray"]["material"])
            info = model.add_colliders(stage, grip, jaw, fin_mat, mat, g["finray"]["contact_offset"],
                                       g["finray"]["rest_offset"], _bind_phys,
                                       extend=float(g["finray"].get("extend", 0.0)), bend=fin_bend())
            if e == 0:
                out["finray"] = info
            out["envs"] += 1
            continue
        for link, base, inner_sign, pad_name in ((grip, FIXED_FINGER, +1.0, "TwinFixedPad"),
                                                 (jaw, MOVING_JAW, -1.0, "TwinMovingPad")):
            inst, mesh = _collision_instance(link, base)
            if inst is None:
                out.setdefault("missing", []).append(f"env_{e}/{base}")
                continue
            if mode == "pads_cad":
                pts_g = _mesh_points_in(mesh, grip)
                c_g, half, x_in = _fit_pad(pts_g, inner_sign, g["pad"])
                if e == 0:
                    out[pad_name] = {"centre_grip": np.round(c_g, 4).tolist(), "half": np.round(half, 4).tolist(),
                                     "inner_face_x": round(x_in, 4)}
                inst.SetActive(False)
                g_w = UsdGeom.Xformable(grip).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                l_w = UsdGeom.Xformable(link).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                to_link = g_w * l_w.GetInverse()
                c = to_link.Transform(Gf.Vec3d(*map(float, c_g)))
                rot = to_link.ExtractRotationQuat()
                cube = UsdGeom.Cube.Define(stage, str(link.GetPath()) + "/" + pad_name)
                cube.CreateSizeAttr(1.0)
                cube.AddTranslateOp().Set(c)
                cube.AddOrientOp().Set(Gf.Quatf(rot))
                cube.AddScaleOp().Set(Gf.Vec3f(*map(float, 2 * half)))
                cube.CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
                prim = cube.GetPrim()
            else:  # convex_decomposition | sdf on the de-instanced CAD mesh
                inst.SetInstanceable(False)
                prim = stage.GetPrimAtPath(inst.GetPath().AppendChild(base))
                if not prim:
                    prim = [p for p in Usd.PrimRange(inst) if p.IsA(UsdGeom.Mesh)][0]
                mc = UsdPhysics.MeshCollisionAPI.Apply(prim)
                if mode == "convex_decomposition":
                    mc.CreateApproximationAttr().Set("convexDecomposition")
                    cd = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
                    cd.CreateMaxConvexHullsAttr().Set(int(g["convex_decomposition"]["max_hulls"]))
                    cd.CreateHullVertexLimitAttr().Set(int(g["convex_decomposition"]["hull_vertex_limit"]))
                    cd.CreateVoxelResolutionAttr().Set(int(g["convex_decomposition"]["voxel_resolution"]))
                    cd.CreateShrinkWrapAttr().Set(True)
                    cd.CreateErrorPercentageAttr().Set(float(g["convex_decomposition"].get("error_percent", 2.0)))
                elif mode == "sdf":
                    mc.CreateApproximationAttr().Set("sdf")
                    sd = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
                    sd.CreateSdfResolutionAttr().Set(int(g["sdf"]["resolution"]))
                else:
                    raise ValueError(f"unknown gripper.collider {mode!r}")
            UsdPhysics.CollisionAPI.Apply(prim)
            px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            px.CreateContactOffsetAttr().Set(float(g["contact_offset"]))
            px.CreateRestOffsetAttr().Set(float(g["rest_offset"]))
            _bind_phys(prim, mat)
        out["envs"] += 1
    return out


def plate_collider_mesh(radius: float, seg: int = 64):
    """(points, face counts, indices) triangle mesh of the plate profile (visuals/assets.py) in the plate frame."""
    from sim.twin.visuals import assets

    P = _p()["plates"]
    prof = assets.plate_profile(radius, rim_h=P["rim_h"], wall=P["wall"], floor_frac=P["floor_frac"],
                                floor_h=P["floor_h"], ring_amp=0.0, n=P.get("profile_points", 48))
    # drop consecutive duplicate (r, z) and keep r = 0 only once at each end
    keep = [0] + [k for k in range(1, len(prof)) if np.linalg.norm(prof[k] - prof[k - 1]) > 1e-6]
    prof = prof[keep]
    a = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    pts, idx = [], []
    ring_of = []
    for r, z in prof:
        if r < 1e-6:
            ring_of.append([len(pts)] * seg)
            pts.append((0.0, 0.0, z))
        else:
            ring_of.append(list(range(len(pts), len(pts) + seg)))
            pts.extend((r * math.cos(t), r * math.sin(t), z) for t in a)
    for i in range(len(prof) - 1):
        A, Bn = ring_of[i], ring_of[i + 1]
        for k in range(seg):
            k2 = (k + 1) % seg
            for tri in ((A[k], Bn[k], Bn[k2]), (A[k], Bn[k2], A[k2])):
                if len(set(tri)) == 3:
                    idx.extend(tri)
    return np.asarray(pts, np.float32), np.full(len(idx) // 3, 3, np.int32), np.asarray(idx, np.int32)


def _plate_colliders(stage, n: int) -> dict:
    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics, Vt

    P = _p()["plates"]
    if P["collider"] != "profile_mesh":
        return {"mode": P["collider"]}
    mat = _physics_material(stage, "/World/PhysicsMaterials/TwinPlate", P["material"])
    cache = {}
    out = {"mode": "profile_mesh", "plates": 0}
    for e in range(n):
        env = stage.GetPrimAtPath(f"/World/envs/env_{e}")
        if not env:
            continue
        for bowl in env.GetChildren():
            if not bowl.GetName().startswith("Bowl_"):
                continue
            name = bowl.GetName()[5:]
            r = float(P.get("radius_by_colour", {}).get(name, P["radius"]))
            if r not in cache:
                cache[r] = plate_collider_mesh(r, P.get("segments", 64))
            pts, cnt, idx = cache[r]
            # the spawner's cylinder: no collision (patch_scene_cfg did it; repeat in case it was not called)
            for p in Usd.PrimRange(bowl):
                if p.HasAPI(UsdPhysics.CollisionAPI) and p.GetName() != "TwinPlateCollider":
                    UsdPhysics.CollisionAPI(p).CreateCollisionEnabledAttr().Set(False)
            mesh = UsdGeom.Mesh.Define(stage, str(bowl.GetPath()) + "/TwinPlateCollider")
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts))
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(cnt))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(idx))
            mesh.CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
            prim = mesh.GetPrim()
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set("none")
            px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            px.CreateContactOffsetAttr().Set(float(P["contact_offset"]))
            px.CreateRestOffsetAttr().Set(0.0)
            _bind_phys(prim, mat)
            out["plates"] += 1
    out["triangles"] = {str(k): int(len(v[1])) for k, v in cache.items()}
    return out


# ------------------------------------------------------------------------------------------------ brick variants
def brick_usd(kind: str, out_dir: str | Path | None = None) -> str:
    """A USD file (body-centred, convex-hull collider, visual Duplo mesh) for a brick variant; spawn it with
    sim_utils.UsdFileCfg(usd_path=..., mass_props=..., rigid_props=...). Needs the Kit app (pxr)."""
    from pxr import Usd, UsdGeom, UsdPhysics, Vt

    from sim.twin.visuals import assets

    out_dir = Path(out_dir or HERE.parent / "out/physics/assets")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"duplo_{kind}.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Brick")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    p, nrm, c, i = assets.brick_mesh(kind)
    vis = assets.define_mesh(stage, "/Brick/Visual", p, nrm, c, i)
    hull = assets.brick_hull_points(kind)
    col = UsdGeom.Mesh.Define(stage, "/Brick/Collider")
    col.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(hull.astype(np.float32)))
    # the hull approximation only needs the points; give PhysX a valid (fan) face list
    col.CreateFaceVertexCountsAttr(Vt.IntArray([3] * (len(hull) - 2)))
    col.CreateFaceVertexIndicesAttr(Vt.IntArray([v for k in range(1, len(hull) - 1) for v in (0, k, k + 1)]))
    col.CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
    UsdPhysics.CollisionAPI.Apply(col.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(col.GetPrim()).CreateApproximationAttr().Set("convexHull")
    _ = vis
    stage.GetRootLayer().Save()
    return str(path)


# ------------------------------------------------------------------------------------------------ physics DR
def sample_physics_dr(rng: np.random.Generator) -> dict:
    """One draw of the physics DR ranges in params.json["dr"] (uniform; mass as a scale)."""
    out = {}
    for k, (lo, hi) in _p().get("dr", {}).items():
        out[k] = float(rng.uniform(lo, hi))
    return out


def apply_physics_dr(scene, draw: dict, env_ids=None):
    """Apply a draw (sample_physics_dr) to all / given envs at run time: brick friction + mass scale, jaw friction
    (the robot shapes bound to the jaw material), gripper effort. Call after sim.reset(), at an env reset.
    Isaac Lab 3.0 PhysX views return warp arrays (wp.to_torch / wp.from_torch); materials live on the CPU."""
    import torch
    import warp as wp

    P = _p()
    rob = scene["robot"]
    dev = rob.device
    ids = torch.arange(scene.num_envs) if env_ids is None else torch.as_tensor(env_ids).cpu().long()
    ids_wp = wp.from_torch(ids.to(torch.int32))
    bm = P["bricks"]["material"]
    for nm in [k for k in vars(scene.cfg) if k == "brick" or k.startswith("distractor")]:
        obj = scene[nm]
        if "brick_friction" in draw:
            m = wp.to_torch(obj.root_view.get_material_properties()).clone()
            m[ids, :, 0] = draw["brick_friction"]
            m[ids, :, 1] = draw["brick_friction"] * bm["dynamic_friction"] / bm["static_friction"]
            obj.root_view.set_material_properties(wp.from_torch(m), ids_wp)
        if "brick_mass_scale" in draw:
            kind = _brick_kind(obj.cfg.spawn.size)
            masses = torch.full((len(ids), obj.num_bodies), P["bricks"]["mass_kg"][kind] * draw["brick_mass_scale"],
                                device=dev)
            obj.set_masses_index(masses=masses, env_ids=ids.to(dev))
    if "jaw_friction" in draw:
        view = rob.root_view
        m = wp.to_torch(view.get_material_properties()).clone()
        key = id(view)
        if key not in _JAW_SHAPES:  # shapes carrying the jaw material at first call (nominal values)
            jm = jaw_material()
            _JAW_SHAPES[key] = ((m[0, :, 0] - jm["static_friction"]).abs() < 1e-4) & \
                ((m[0, :, 1] - jm["dynamic_friction"]).abs() < 1e-4)
        sh = _JAW_SHAPES[key]
        if bool(sh.any()):
            jm = jaw_material()
            for i in ids.tolist():
                m[i, sh, 0] = draw["jaw_friction"]
                m[i, sh, 1] = draw["jaw_friction"] * jm["dynamic_friction"] / jm["static_friction"]
            view.set_material_properties(wp.from_torch(m), ids_wp)
    if "gripper_effort_nm" in draw:
        j = rob.joint_names.index("gripper")
        rob.write_joint_effort_limit_to_sim_index(limits=torch.full((len(ids), 1), float(draw["gripper_effort_nm"]),
                                                                    device=dev),
                                                  joint_ids=[j], env_ids=ids.to(dev))
    fins = [i for i, n in enumerate(rob.joint_names) if n.startswith("fin_")]
    if "fin_stiffness_scale" in draw and fins and fin_bend() is not None:  # bending-fin hinge stiffness x scale
        from sim.twin.gripper import model as fin_model

        b = fin_bend()
        k = [fin_model.joint_gains(b, fin_model.SEG_NAMES.index(rob.joint_names[i][-1]),
                                   "jaw" if "_jaw_" in rob.joint_names[i] else "fixed")[0] for i in fins]
        stiff = torch.tensor(k, device=dev).repeat(len(ids), 1) * float(draw["fin_stiffness_scale"])
        rob.write_joint_stiffness_to_sim_index(stiffness=stiff, joint_ids=fins, env_ids=ids.to(dev))
    return draw


_JAW_SHAPES: dict = {}


def nominal_physics() -> dict:
    """The nominal values of the DR keys (to undo a draw)."""
    P = _p()
    return {"brick_friction": P["bricks"]["material"]["static_friction"],
            "jaw_friction": jaw_material()["static_friction"], "brick_mass_scale": 1.0,
            "gripper_effort_nm": grip_effort(), "fin_stiffness_scale": 1.0}


def summary() -> dict:
    """The chosen numbers (for logs)."""
    return copy.deepcopy(_p())
