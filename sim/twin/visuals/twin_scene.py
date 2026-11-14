"""Apply the twin's visual realism to the residual env's scene (sim/residual_env/scene.py) without changing physics.

    from sim.twin.visuals.twin_scene import TwinLook, twin_scene_cfg, apply_twin_visuals
    cfg = twin_scene_cfg(num_envs, layout, profile, look)          # residual env cfg + lights (+ inspect camera)
    scene = InteractiveScene(cfg)
    apply_twin_visuals(num_envs, layout, profile, look)            # before sim.reset()
    sim.reset(); sync = VisualSync(scene, wrist_cam_rel=wrist_cam_rel())   # usd_sync.py
    ... physics steps ...; sync(); sim.render(); scene.update(0.0)  # every rendered frame

The cameras render visual proxies, not the physics prims (GB10 render path, see usd_sync.py for why).

What it does (all visual-only; colliders, masses, frictions untouched):
1. robot: a physics-free visual copy (RobotVisual), de-instanced visual meshes painted black, red-orange jaws, orange wrist-camera mount (paint.py);
2. table: a textured quad (texture = real top-camera median back-projected on the table plane, table_texture.py)
   0.3 mm above the physics table; the table cube's own sides get the mean wood colour;
3. plates: the kinematic collider cylinders are hidden (purpose=guide) and a dish mesh is added as a visual proxy
   (TwinVis/Bowl_<name>, follows the body via VisualSync); colours from the real top frames;
4. bricks: collider boxes hidden, Duplo meshes with studs as proxies (TwinVis/<Brick|DistractorN>); colour per
   brick from a palette sampled from the real top frames (set_brick_color to change at run time);
5. inspect-camera backdrop: a 4 x 0.45 m billboard standing on the table ~0.75 m in front of the inspect camera
   (outside the top camera's view), textured by projecting the median real inspect frame (laptop, carton, people);
6. lighting: dome (diffuse hall light) + a soft distant key light, linear tonemapping; intensities scaled by the
   exposure calibration in sim/twin/visuals/lighting.json (render_compare.py --calibrate writes it).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEX = HERE / "textures/table_top.jpg"
BACKDROP = HERE / "textures/backdrop_inspect.jpg"
LIGHTING_JSON = HERE / "lighting.json"
CAMERA_COLOR_JSON = HERE / "camera_color.json"

# sRGB colours as the REAL top camera sees them (measure_real.py over 114 episodes' start frames, cal2 dataset).
# The lavender ("purple") plate reads sky-blue in the top camera (it is lavender in the wrist / inspect cameras); we
# match the top camera, which the policy relies on most.
PLATE_RGB = {"orange": (174, 91, 44), "yellow": (157, 190, 110), "purple": (9, 152, 240)}
PLATE_DIAM = {"orange": 0.182, "yellow": 0.184, "purple": 0.189}  # m, outer rim (measure_real.py)
BRICK_RGB = {  # median blob colours in real top frames; cream picked by hand (too close to the wood to segment)
    "light_blue": (5, 138, 202), "yellow": (155, 173, 86), "orange": (166, 85, 41), "red": (124, 24, 43),
    "brown": (110, 60, 37), "cream": (205, 200, 160),
}
# residual env brick prims -> default palette colour (the env's own colour names)
DEFAULT_BRICK_COLOR = {"Brick": "light_blue", "Distractor0": "light_blue", "Distractor1": "yellow",
                       "Distractor2": "cream", "Distractor3": "orange", "Distractor4": "red", "Distractor5": "brown",
                       "Distractor6": "yellow", "Distractor7": "orange"}


@dataclass
class TwinLook:
    table_texture: str = str(TEX)
    table_roughness: float = 0.7
    plate_rgb: dict = field(default_factory=lambda: dict(PLATE_RGB))
    plate_diam: dict = field(default_factory=lambda: dict(PLATE_DIAM))
    plate_roughness: float = 0.22
    plate_clearcoat: float = 0.4
    brick_rgb: dict = field(default_factory=lambda: dict(BRICK_RGB))
    brick_color: dict = field(default_factory=lambda: dict(DEFAULT_BRICK_COLOR))
    brick_roughness: float = 0.3
    brick_clearcoat: float = 0.3
    # lights (before the exposure scale): dome = diffuse hall light, key = soft overhead distant light
    dome_intensity: float = 1000.0
    dome_rgb: tuple = (1.0, 1.0, 1.0)
    key_intensity: float = 350.0
    key_angle_deg: float = 12.0
    key_rot_xyz_deg: tuple = (10.0, -20.0, 0.0)  # tilt of the -z light direction (USD rotateXYZ)
    exposure_scale: float | None = None  # None = lighting.json (calibrated) or 1.0
    linear_tonemap: bool = True
    inspect_camera: bool = True
    wrist_mount: bool = True
    backdrop_texture: str | None = str(BACKDROP)  # inspect-camera background billboard (None = none)
    backdrop_gain: float | None = None  # None = lighting.json or 2.0 (a vertical surface gets ~half the light)

    def bd_gain(self) -> float:
        if self.backdrop_gain is not None:
            return self.backdrop_gain
        if LIGHTING_JSON.exists():
            return float(json.loads(LIGHTING_JSON.read_text()).get("backdrop_gain", 2.0))
        return 2.0

    def scale(self) -> float:
        if self.exposure_scale is not None:
            return self.exposure_scale
        if LIGHTING_JSON.exists():
            return float(json.loads(LIGHTING_JSON.read_text()).get("exposure_scale", 1.0))
        return 1.0


def _rscene():
    import sys

    p = str(Path(__file__).resolve().parents[2] / "residual_env")
    if p not in sys.path:
        sys.path.insert(0, p)
    import scene as rscene

    return rscene


def twin_scene_cfg(num_envs: int, layout, profile, look: TwinLook | None = None, cameras: bool = True, **kw):
    """The residual env's InteractiveSceneCfg with the twin's lights (+ the inspect camera from the profile)."""
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    rscene = _rscene()
    look = look or TwinLook()
    cfg = rscene.build_scene_cfg(num_envs, layout, cameras, profile=profile, **kw)
    s = look.scale()
    cfg.dome.spawn = sim_utils.DomeLightCfg(intensity=look.dome_intensity * s, color=tuple(look.dome_rgb))
    cfg.sun.spawn = sim_utils.DistantLightCfg(intensity=look.key_intensity * s, angle=look.key_angle_deg)
    cfg.sun.init_state.rot = rscene._euler_xyz_to_quat_xyzw(*look.key_rot_xyz_deg)
    if cameras and look.inspect_camera and profile is not None and getattr(profile, "inspect", None) is not None:
        m, intr = np.asarray(profile.inspect.rows), profile.inspect.intr
        cfg.inspect_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/InspectCam", update_period=0.0, height=480, width=640, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=intr["focal_length"],
                                             horizontal_aperture=intr["horizontal_aperture"],
                                             vertical_aperture=intr["vertical_aperture"], clipping_range=(0.005, 20.0)),
            offset=CameraCfg.OffsetCfg(pos=tuple(float(v) for v in m[3, :3]), rot=rscene.mat_to_quat_xyzw(m),
                                       convention="opengl"))
    return cfg


def set_render_settings(look: TwinLook | None = None):
    """Linear tonemapping (output = sRGB(exposure * radiance)) so material colours map predictably to pixels."""
    import carb

    look = look or TwinLook()
    st = carb.settings.get_settings()
    before = {k: st.get(k) for k in ("/rtx/post/tonemap/op", "/rtx/post/histogram/enabled",
                                     "/rtx/post/tonemap/filmIso", "/rtx/post/tonemap/cameraShutter",
                                     "/rtx/post/tonemap/fNumber")}
    if look.linear_tonemap:
        st.set("/rtx/post/tonemap/op", 1)  # 0 clamp, 1 linear, ... 6 ACES
        st.set("/rtx/post/histogram/enabled", False)
    return before


def _table_quad(stage, env: int, look: TwinLook, mat, wood_mat):
    from pxr import Usd

    from sim.twin.visuals.assets import define_mesh, quad_mesh
    from sim.twin.visuals.materials import bind

    ext = json.loads(Path(look.table_texture).with_suffix(".json").read_text())
    root = stage.GetPrimAtPath(f"/World/envs/env_{env}/Table")
    # the table Xform sits at (0.25, 0, -0.027) (scene.py); author the quad in env coordinates under a sibling
    pts, nrm, c, i, st = quad_mesh(ext["x0"], ext["y0"], ext["width"], ext["height"], ext["z"] + 0.0003)
    mesh = define_mesh(stage, f"/World/envs/env_{env}/TableTopVisual", pts, nrm, c, i, st=st)
    bind(mesh.GetPrim(), mat)
    if root:
        for p in Usd.PrimRange(root):
            bind(p, wood_mat)
    return mesh


def _backdrop(stage, env: int, look: TwinLook):
    from sim.twin.visuals.assets import define_mesh
    from sim.twin.visuals.materials import bind, preview_material

    pl = json.loads(Path(look.backdrop_texture).with_suffix(".json").read_text())
    o, u, v = (np.asarray(pl[k], float) for k in ("origin", "u_axis", "v_axis"))
    W, H = pl["width"], pl["height"]
    pts = np.stack([o, o + u * W, o + u * W + v * H, o + v * H])
    n = np.cross(u, v)
    g = look.bd_gain()
    mat = preview_material(stage, "/World/Looks/Twin/Backdrop", (g, g, g), 1.0, linear=True,
                           texture=str(Path(look.backdrop_texture).resolve()), specular=0.0)
    mesh = define_mesh(stage, f"/World/envs/env_{env}/BackdropVisual", pts, np.tile(n, (4, 1)), [4], [0, 1, 2, 3],
                       st=np.array([[0, 0], [1, 0], [1, 1], [0, 1.0]]))
    bind(mesh.GetPrim(), mat)
    return mesh


def _robot_visual(stage, env: int, robot_usd: str, robot: str = "Robot", vis: str = "RobotVisual") -> dict:
    """Physics-free visual copy of the robot (USD Physics variant 'none', collision instances deactivated); the
    physics robot's own visual meshes are hidden (purpose=guide). Poses come from VisualSync (usd_sync.py)."""
    from pxr import Usd, UsdGeom

    from sim.twin.visuals.paint import _is_visual_instance

    prim = stage.DefinePrim(f"/World/envs/env_{env}/{vis}", "Xform")
    prim.GetReferences().AddReference(robot_usd)
    vs = prim.GetVariantSets()
    if vs.HasVariantSet("Physics"):
        vs.GetVariantSet("Physics").SetVariantSelection("none")
    n_off = 0
    for p in list(Usd.PrimRange(prim)):
        if p.IsInstance() and not _is_visual_instance(p):
            p.SetActive(False)  # collision meshes (would become static colliders without a rigid body)
            n_off += 1
    n_hidden = 0
    root = stage.GetPrimAtPath(f"/World/envs/env_{env}/{robot}")
    for p in Usd.PrimRange(root):
        if _is_visual_instance(p):
            UsdGeom.Imageable(p).CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
            n_hidden += 1
    return {"collision_off": n_off, "physics_visuals_hidden": n_hidden}


def _vis_xform(stage, env: int, name: str, src_prim):
    """/World/envs/env_i/TwinVis/<name>, initialised at the physics prim's current (spawn) pose."""
    from pxr import Usd, UsdGeom

    from sim.twin.visuals.usd_sync import set_local

    stage.DefinePrim(f"/World/envs/env_{env}/TwinVis", "Xform")
    x = UsdGeom.Xform.Define(stage, f"/World/envs/env_{env}/TwinVis/{name}")
    w = UsdGeom.Xformable(src_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    pw = UsdGeom.Xformable(x.GetPrim()).ComputeParentToWorldTransform(Usd.TimeCode.Default())
    x.AddTranslateOp()
    x.AddOrientOp()
    set_local(x.GetPrim(), w * pw.GetInverse())
    return x.GetPrim()


def apply_twin_visuals(num_envs: int, layout, profile, look: TwinLook | None = None) -> dict:
    """Build the visual twin on the spawned scene. Call after InteractiveScene(cfg), before sim.reset(); then create
    `usd_sync.VisualSync(scene, wrist_cam_rel=WRIST_CAM_REL)` after sim.reset() and call it before every render."""
    import omni.usd

    from sim.twin.visuals.assets import define_mesh, duplo_mesh, hide_geometry, plate_mesh
    from sim.twin.visuals.materials import bind, preview_material
    from sim.twin.visuals.paint import add_wrist_camera_mount, paint_robot

    look = look or TwinLook()
    stage = omni.usd.get_context().get_stage()
    info = {"render_settings_before": set_render_settings(look)}
    robot_usd = getattr(profile, "robot_usd", None) or _rscene().ROBOT_USD
    info["robot_visual"] = [_robot_visual(stage, e, robot_usd) for e in range(num_envs)]
    info["robot_paint"] = paint_robot(num_envs, robot="RobotVisual")
    if look.wrist_mount and profile is not None and getattr(profile, "wrist", None) is not None:
        info["wrist_mount"] = add_wrist_camera_mount(
            num_envs, profile.wrist.rows, gripper_rel="RobotVisual/" + _rscene().GRIPPER_LINK.split("/", 1)[1])
    L = "/World/Looks/Twin"
    ext = json.loads(Path(look.table_texture).with_suffix(".json").read_text())
    wood = tuple(ext.get("mean_rgb_visible", (130, 132, 124)))
    m_table = preview_material(stage, f"{L}/TableTop", (255, 255, 255), look.table_roughness,
                               texture=str(Path(look.table_texture).resolve()))
    m_wood = preview_material(stage, f"{L}/TableWood", wood, look.table_roughness)
    m_plate = {k: preview_material(stage, f"{L}/Plate_{k}", rgb, look.plate_roughness, clearcoat=look.plate_clearcoat)
               for k, rgb in look.plate_rgb.items()}
    m_brick = {k: preview_material(stage, f"{L}/Brick_{k}", rgb, look.brick_roughness, clearcoat=look.brick_clearcoat)
               for k, rgb in look.brick_rgb.items()}
    bricks = [("Brick", layout.target)] + [(f"Distractor{i}", d) for i, d in enumerate(layout.distractors)]
    n_plates = n_bricks = 0
    for e in range(num_envs):
        _table_quad(stage, e, look, m_table, m_wood)
        if look.backdrop_texture and Path(look.backdrop_texture).exists():
            _backdrop(stage, e, look)
        for name, (x, y, r, rgb) in layout.bowls.items():
            prim = stage.GetPrimAtPath(f"/World/envs/env_{e}/Bowl_{name}")
            if not prim:
                continue
            hide_geometry(prim)
            vis = _vis_xform(stage, e, f"Bowl_{name}", prim)
            diam = look.plate_diam.get(name, 2 * r)
            pts, nrm, c, i = plate_mesh(diam / 2)
            mesh = define_mesh(stage, f"{vis.GetPath()}/Dish", pts, nrm, c, i)
            bind(mesh.GetPrim(), m_plate.get(name) or preview_material(stage, f"{L}/Plate_{name}",
                                                                         np.multiply(rgb, 255), look.plate_roughness))
            n_plates += 1
        for pname, b in bricks:
            prim = stage.GetPrimAtPath(f"/World/envs/env_{e}/{pname}")
            if not prim:
                continue
            hide_geometry(prim)
            vis = _vis_xform(stage, e, pname, prim)
            nx = max(1, round(b.size[0] / 0.0159))
            ny = max(1, round(b.size[1] / 0.0159))
            pts, nrm, c, i = duplo_mesh(nx, ny, size=b.size)
            mesh = define_mesh(stage, f"{vis.GetPath()}/Duplo", pts, nrm, c, i)
            bind(mesh.GetPrim(), m_brick[look.brick_color.get(pname, "yellow")])
            n_bricks += 1
    info.update(plates=n_plates, bricks=n_bricks, exposure_scale=look.scale(), backdrop_gain=look.bd_gain())
    return info


def color_correct(rgb, cam: str, path: Path = CAMERA_COLOR_JSON):
    """Per-camera colour response: the twin's materials are matched to the TOP camera; the real wrist and inspect
    cameras render the same objects differently (the lavender plate reads sky-blue in the top camera and pale
    lavender in the wrist / inspect ones; different white balance). Apply the fitted linear-RGB 3x3
    (measure_real.py --ccm) to a sim image (HxWx3 uint8 numpy or torch) of camera `cam` ('wrist' | 'inspect');
    other cameras / missing fit -> unchanged."""
    if not path.exists():
        return rgb
    fit = json.loads(path.read_text()).get(cam)
    if not fit:
        return rgb
    M = np.asarray(fit["M_linear"], np.float32)
    try:
        import torch

        if isinstance(rgb, torch.Tensor):
            c = rgb[..., :3].float() / 255.0
            lin = torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
            out = (lin @ torch.as_tensor(M.T, device=rgb.device)).clamp(0, 1)
            srgb = torch.where(out <= 0.0031308, out * 12.92, 1.055 * out ** (1 / 2.4) - 0.055)
            return (srgb * 255.0 + 0.5).to(torch.uint8)
    except ImportError:
        pass
    c = np.asarray(rgb)[..., :3].astype(np.float32) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    out = np.clip(lin @ M.T, 0, 1)
    srgb = np.where(out <= 0.0031308, out * 12.92, 1.055 * out ** (1 / 2.4) - 0.055)
    return (srgb * 255.0 + 0.5).astype(np.uint8)


def wrist_cam_rel() -> str:
    """Env-relative path of the residual env's wrist camera prim (child of the physics gripper_link)."""
    return _rscene().GRIPPER_LINK + "/WristCam"


def set_brick_color(env: int, prim_name: str, color: str):
    import omni.usd
    from pxr import UsdShade

    from sim.twin.visuals.materials import bind

    stage = omni.usd.get_context().get_stage()
    mesh = stage.GetPrimAtPath(f"/World/envs/env_{env}/TwinVis/{prim_name}/Duplo")
    mat = stage.GetPrimAtPath(f"/World/Looks/Twin/Brick_{color}")
    if mesh and mat:
        bind(mesh, UsdShade.Material(mat))
        return True
    return False


def yaw_quat_xyzw(yaw: float):
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
