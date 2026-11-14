"""Launch Isaac Lab (with the GB10 rendering fix), build the scene and the ResidualEnv.

GB10 note: with Isaac Lab's default Fabric scene delegate the RTX render delegate reads camera transforms from Fabric
and every camera renders from the world origin (constant ~8 mm depth, black RGB). Disabling the delegate fixes it.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KIT_ARGS = "--/app/useFabricSceneDelegate=false --/rtx/hydra/readTransformsFromFabricInRenderDelegate=false"
DEFAULT_SCENE_PARAMS = str(HERE / "scene_params.json")
DEFAULT_SERVO_PARAMS = str(HERE / "servo_params.json")


def launch(cameras: bool):
    from isaaclab.app import AppLauncher

    return AppLauncher(headless=True, enable_cameras=cameras, kit_args=KIT_ARGS).app


def paint_robot(num_envs: int):
    """Joe's materials: black body, red fingers (wrist_roll_follower / moving_jaw meshes)."""
    import omni.usd
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    stage = omni.usd.get_context().get_stage()

    def mat(path, rgb, rough):
        m = UsdShade.Material.Define(stage, path)
        sh = UsdShade.Shader.Define(stage, path + "/S")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
        m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        return m

    black = mat("/World/Looks/RobotBlack", (0.006, 0.0066, 0.0072), 0.35)
    red = mat("/World/Looks/FingersRed", (0.38, 0.0035, 0.0134), 0.4)
    for e in range(num_envs):
        root = stage.GetPrimAtPath(f"/World/envs/env_{e}/Robot")
        if not root:
            continue
        for p in _descendants(root):
            if p.IsA(UsdGeom.Mesh):
                UsdShade.MaterialBindingAPI.Apply(p).Bind(
                    red if p.GetName().startswith(("wrist_roll_follower", "moving_jaw")) else black,
                    UsdShade.Tokens.strongerThanDescendants)


def finger_collisions(num_envs: int, approx: str = "convexDecomposition"):
    """The converted SO-101 USD gives every mesh a convexHull collider. For the fixed jaw (wrist_roll_follower on
    gripper_link) and the moving jaw the hull fills the gap between the fingers, so a brick can never sit between
    them (replayed demos close on nothing). Re-approximate those two meshes before physics parsing."""
    import omni.usd
    from pxr import PhysxSchema, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    n_set = 0
    for e in range(num_envs):
        root = stage.GetPrimAtPath(f"/World/envs/env_{e}/Robot")
        if not root:
            continue
        for p in _descendants(root):
            if p.GetName() in ("wrist_roll_follower_so101_v1", "moving_jaw_so101_v1"):
                UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr().Set(approx)
                if approx == "convexDecomposition":
                    cd = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(p)
                    cd.CreateMaxConvexHullsAttr().Set(32)
                    cd.CreateHullVertexLimitAttr().Set(64)
                    cd.CreateVoxelResolutionAttr().Set(300000)
                    cd.CreateShrinkWrapAttr().Set(True)
                n_set += 1
    return n_set


def finger_pads(num_envs: int, friction: float = 0.9):
    """Box pads at the measured inner faces of both fingers (gripper_link frame, jaw at q=0) and no collision on the
    original finger meshes. Measured from the USD meshes: fixed-finger inner face x = -0.0079 m, moving-jaw inner face
    x = +0.0094 m at q=0; fingertip region z in [-0.106, -0.07]; finger width y +-0.0065. Convex hulls of the finger
    meshes fill the finger gap (they push the brick out) and convex decomposition does not cook here."""
    import omni.usd
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    stage = omni.usd.get_context().get_stage()
    mat = UsdShade.Material.Define(stage, "/World/PhysicsMaterials/FingerPad")
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr().Set(friction)
    pm.CreateDynamicFrictionAttr().Set(friction * 0.85)
    pm.CreateRestitutionAttr().Set(0.0)
    n = 0
    for e in range(num_envs):
        root = stage.GetPrimAtPath(f"/World/envs/env_{e}/Robot")
        if not root:
            continue
        prims = {p.GetName(): p for p in Usd.PrimRange(root)}
        grip, jaw = prims.get("gripper_link"), prims.get("moving_jaw_so101_v1_link")
        for name in ("wrist_roll_follower_so101_v1", "moving_jaw_so101_v1"):
            if name in prims:
                UsdPhysics.CollisionAPI.Apply(prims[name]).CreateCollisionEnabledAttr().Set(False)
        # pad boxes (centre, half-size) in the gripper_link frame at q=0
        pads = {"FixedPad": (grip, Gf.Vec3d(-0.0117, 0.0, -0.088), Gf.Vec3d(0.0038, 0.0068, 0.018)),
                "MovingPad": (jaw, Gf.Vec3d(0.0132, 0.0, -0.088), Gf.Vec3d(0.0038, 0.0068, 0.018))}
        g_w = UsdGeom.Xformable(grip).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        for pname, (parent, c_grip, half) in pads.items():
            # express the pad in its parent link's frame (moving jaw: USD-authored pose = q = 0)
            p_w = UsdGeom.Xformable(parent).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            to_parent = g_w * p_w.GetInverse()
            c = to_parent.Transform(c_grip)
            rot = to_parent.ExtractRotationQuat()
            cube = UsdGeom.Cube.Define(stage, str(parent.GetPath()) + "/" + pname)
            cube.CreateSizeAttr(1.0)
            cube.AddTranslateOp().Set(c)
            cube.AddOrientOp().Set(Gf.Quatf(rot))
            cube.AddScaleOp().Set(Gf.Vec3f(*(2 * half)))
            cube.CreatePurposeAttr().Set(UsdGeom.Tokens.guide)  # invisible in renders
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            px = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
            px.CreateContactOffsetAttr().Set(0.002)
            px.CreateRestOffsetAttr().Set(0.0)
            UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")
            n += 1
    return n


def _descendants(prim):
    from pxr import Usd

    return list(Usd.PrimRange(prim))


def make(cfg, base=None, device: str = "cuda:0"):
    """(sim, scene, layout, env) after the app is launched."""
    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene

    sys.path.insert(0, str(HERE))
    from env import ResidualEnv
    from scene import build_scene_cfg, load_layout

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / cfg.physics_hz, device=device))
    from calib import load_profile
    from scene import Brick

    layout = load_layout(cfg.scene_params or DEFAULT_SCENE_PARAMS, getattr(cfg, "camera_source", "joe"))
    if getattr(cfg, "target_size", None):
        t = layout.target
        sx, sy, sz = cfg.target_size
        mass = t.mass * (sx * sy * sz) / (t.size[0] * t.size[1] * t.size[2])
        layout.target = Brick(t.name, tuple(cfg.target_size), t.pos, t.color, mass)
    profile = load_profile(getattr(cfg, "calib", "") or cfg.joint_offsets)
    print("calibration profile:", profile.name, "| offsets", [round(v, 4) for v in profile.joint_offsets],
          "| gripper q = units *", round(profile.grip_scale, 5), "+", round(profile.grip_bias, 4), flush=True)
    scene = InteractiveScene(build_scene_cfg(cfg.num_envs, layout, cfg.cameras, profile=profile))
    if cfg.cameras:
        paint_robot(cfg.num_envs)
    import os

    if os.environ.get("FINGER_PADS", "1") == "1":
        print("finger pads:", finger_pads(cfg.num_envs), flush=True)
    sim.reset()
    scene.reset()
    if not cfg.servo_params:
        cfg.servo_params = DEFAULT_SERVO_PARAMS
    env = ResidualEnv(cfg, base, sim, scene, layout, device)
    return sim, scene, layout, env
