"""Apply the twin's visual realism to the residual env's scene (sim/residual_env/scene.py) without changing physics.

    from sim.twin.visuals.twin_scene import TwinLook, twin_scene_cfg, apply_twin_visuals
    cfg = twin_scene_cfg(num_envs, layout, profile, look)          # residual env cfg + lights (+ inspect camera)
    scene = InteractiveScene(cfg)
    apply_twin_visuals(num_envs, layout, profile, look)            # before sim.reset()
    sim.reset(); sync = VisualSync(scene, wrist_cam_rel=wrist_cam_rel())   # usd_sync.py
    ... physics steps ...; sync(); sim.render(); scene.update(0.0)  # every rendered frame

The cameras render visual proxies, not the physics prims (GB10 render path, see usd_sync.py for why).

What it does (all visual-only; colliders, masses, frictions untouched):
1. robot: a physics-free visual copy (RobotVisual), de-instanced visual meshes painted black, red-orange jaws,
   orange wrist-camera mount (paint.py);
2. table: a textured quad 0.3 mm above the physics table: lacquered worn wood at metric scale (wood_texture.py: the
   phone photo registered to the table plane + quilted wood around it, colours as the top camera sees them, normal
   map from the grain, clear-coat for the lacquer); the table cube's own sides get the mean wood colour;
3. plates: the kinematic collider cylinders are hidden (purpose=guide) and a dish mesh is added as a visual proxy
   (TwinVis/Bowl_<name>, follows the body via VisualSync); colours from the real top frames;
4. bricks: collider boxes hidden, Duplo meshes with studs as proxies (TwinVis/<Brick|DistractorN>); colour per
   brick from a palette sampled from the real top frames (set_brick_color to change at run time);
5. inspect-camera backdrop: an unlit billboard standing on the table ~0.77 m in front of the inspect camera
   (outside the top camera's view), textured by projecting the median real inspect frame (laptop, carton; moving
   people average out), and an unlit hall cylinder around the station (upper part of the station panorama, people
   blurred) so that cameras looking past the table see the hall, not the dome (backdrop_texture.py);
6. lighting: dome (diffuse hall light) + a very soft overhead key + the rig of sim/twin/lighting/rig.py (fluorescent
   strips ~10 m up, warm lamps ~6 m out along the inspect view, RTX flat ambient off; fitted in
   sim/twin/lighting/log.md), linear tonemapping;
7. render settings (antialiasing mode etc.) from render_settings.json.

The fitted look lives in JSON next to this file and is what `TwinLook()` returns by default:
lighting.json (lights, exposure), materials.json (gloss / colours, backdrop gains), render_settings.json (RTX
settings) and camera_color.json (per-camera response applied by `color_correct`). materials.json is written by
fit/fit_visuals.py from real-vs-sim metrics (fit/log.md). `refresh_look(look)` re-applies lights / materials / render
settings on a live stage.
"""
from __future__ import annotations

import json
import math
from dataclasses import MISSING, dataclass, field, fields, replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEX = HERE / "textures/table_top.jpg"
BACKDROP = HERE / "textures/backdrop_inspect.jpg"
LIGHTING_JSON = HERE / "lighting.json"
MATERIALS_JSON = HERE / "materials.json"
RENDER_JSON = HERE / "render_settings.json"
CAMERA_COLOR_JSON = HERE / "camera_color.json"
HALL = HERE.parent / "data/photos/derived/hall_cylinder.jpg"  # git-ignored (from the station panorama)
LOOKS = "/World/Looks/Twin"

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


def _fitted() -> dict:
    """The fitted look: lighting.json + materials.json (flat TwinLook field names) + render_settings.json."""
    out = {}
    for p in (LIGHTING_JSON, MATERIALS_JSON):
        if p.exists():
            out.update(json.loads(p.read_text()))
    if RENDER_JSON.exists():
        out["render_settings"] = json.loads(RENDER_JSON.read_text()).get("carb", {})
    return out


@dataclass
class TwinLook:
    """The twin's look. Fields left at their class default take the fitted value from the JSON files (if any)."""

    table_texture: str = str(TEX)
    table_tint: tuple = (1.0, 1.0, 1.0)  # linear multiplier of the wood texture (albedo under the lacquer)
    table_roughness: float = 0.7  # the wood under the lacquer
    table_clearcoat: float = 0.0  # thin glossy lacquer on the worn wood (sheen at grazing angles)
    table_clearcoat_roughness: float = 0.15
    table_specular: float = 0.5  # base-layer dielectric specular level (0.5 = ior 1.5; the texture bakes the sheen)
    table_normal: bool = True  # use the texture's normal map (table_top.json "normal_map") when it exists
    plate_rgb: dict = field(default_factory=lambda: dict(PLATE_RGB))
    plate_diam: dict = field(default_factory=lambda: dict(PLATE_DIAM))
    plate_roughness: float = 0.22
    plate_clearcoat: float = 0.4
    plate_clearcoat_roughness: float = 0.05
    plate_specular: float = 0.5
    brick_rgb: dict = field(default_factory=lambda: dict(BRICK_RGB))
    brick_color: dict = field(default_factory=lambda: dict(DEFAULT_BRICK_COLOR))
    brick_roughness: float = 0.3
    brick_clearcoat: float = 0.3
    brick_clearcoat_roughness: float = 0.05
    brick_specular: float = 0.5
    robot_body_rgb: tuple = (24, 27, 33)  # sRGB 0-255 as the top camera sees the printed body / the servos
    robot_body_roughness: float = 0.55
    robot_servo_rgb: tuple = (20, 20, 23)
    robot_servo_roughness: float = 0.4
    # lights (before the exposure scale): dome = diffuse hall light, key = soft overhead distant light,
    # ceiling = small distant lights that make the highlights on plates / bricks
    dome_intensity: float = 1000.0
    dome_rgb: tuple = (1.0, 1.0, 1.0)
    key_intensity: float = 350.0
    key_angle_deg: float = 12.0
    key_rot_xyz_deg: tuple = (10.0, -20.0, 0.0)  # tilt of the -z light direction (USD rotateXYZ)
    key_rgb: tuple = (1.0, 1.0, 1.0)
    ceiling_lights: list = field(default_factory=list)  # [{"dir": unit vector TO the light, "intensity", ...}]
    side_lights: list = field(default_factory=list)  # warm side lamps [{"pos", "radius", "intensity", "rgb"}]
    ambient_intensity: float | None = None  # RTX flat ambient term (None = leave the renderer default)
    exposure_scale: float | None = None  # None = lighting.json (calibrated) or 1.0
    linear_tonemap: bool = True
    render_settings: dict = field(default_factory=dict)  # carb settings (render_settings.json "carb")
    inspect_camera: bool = True
    wrist_mount: bool = True
    backdrop_texture: str | None = str(BACKDROP)  # inspect-camera background billboard (None = none)
    backdrop_gain: float | None = None  # None = lighting.json or 1.0 (the billboard is unlit: gain x texture)
    hall_texture: str | None = str(HALL)  # hall cylinder around the station (None / missing file = none)
    hall_gain: float = 1.0
    hall_radius: float = 3.0
    fitted: bool = True  # False: ignore the fitted JSON values (class defaults only)

    def __post_init__(self):
        if not self.fitted:
            return
        fit = _fitted()
        for f in fields(self):
            if f.name not in fit or f.name in ("exposure_scale", "backdrop_gain"):
                continue
            default = f.default_factory() if f.default_factory is not MISSING else f.default
            if getattr(self, f.name) == default:
                v = fit[f.name]
                setattr(self, f.name, tuple(v) if isinstance(default, tuple) else v)

    def bd_gain(self) -> float:
        if self.backdrop_gain is not None:
            return self.backdrop_gain
        if self.fitted and MATERIALS_JSON.exists():
            return float(json.loads(MATERIALS_JSON.read_text()).get("backdrop_gain", 1.0))
        return 1.0

    def scale(self) -> float:
        if self.exposure_scale is not None:
            return self.exposure_scale
        if self.fitted and LIGHTING_JSON.exists():
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
    cfg.sun.spawn = sim_utils.DistantLightCfg(intensity=look.key_intensity * s, angle=look.key_angle_deg,
                                              color=tuple(look.key_rgb))
    cfg.sun.init_state.rot = rscene._euler_xyz_to_quat_xyzw(*look.key_rot_xyz_deg)
    # ceiling tubes / side lamps / RTX ambient: sim.twin.lighting.rig.build (apply_twin_visuals)
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


def quat_z_to(direction) -> tuple:
    """xyzw quaternion that turns +z into `direction` (a distant light shines along its -z, so pass the unit vector
    pointing TO the light)."""
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    axis = np.cross([0.0, 0.0, 1.0], d)
    n = np.linalg.norm(axis)
    if n < 1e-9:
        return (0.0, 0.0, 0.0, 1.0) if d[2] > 0 else (1.0, 0.0, 0.0, 0.0)
    half = 0.5 * math.atan2(n, d[2])
    axis = axis / n * math.sin(half)
    return (float(axis[0]), float(axis[1]), float(axis[2]), math.cos(half))


def set_render_settings(look: TwinLook | None = None):
    """Linear tonemapping (output = sRGB(exposure * radiance)) so material colours map predictably to pixels, plus
    the fitted RTX settings of render_settings.json (antialiasing mode etc.). Returns the previous values."""
    import carb

    look = look or TwinLook()
    st = carb.settings.get_settings()
    keys = ["/rtx/post/tonemap/op", "/rtx/post/histogram/enabled", "/rtx/post/tonemap/filmIso",
            "/rtx/post/tonemap/cameraShutter", "/rtx/post/tonemap/fNumber", *look.render_settings]
    before = {k: st.get(k) for k in keys}
    if look.linear_tonemap:
        st.set("/rtx/post/tonemap/op", 1)  # 0 clamp, 1 linear, ... 6 ACES
        st.set("/rtx/post/histogram/enabled", False)
    for k, v in look.render_settings.items():
        st.set(k, v)
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
    mat = preview_material(stage, f"{LOOKS}/Backdrop", (g, g, g), 1.0, linear=True, emissive=True,
                           texture=str(Path(look.backdrop_texture).resolve()))
    mesh = define_mesh(stage, f"/World/envs/env_{env}/BackdropVisual", pts, np.tile(n, (4, 1)), [4], [0, 1, 2, 3],
                       st=np.array([[0, 0], [1, 0], [1, 1], [0, 1.0]]))
    bind(mesh.GetPrim(), mat)
    return mesh


def _hall(stage, env: int, look: TwinLook, center=(0.3, 0.0), z0: float = -0.75, z1: float = 2.6,
          strip=(0.0, 1.5), seg: int = 72):
    """Unlit cylinder around the station carrying the hall strip of the panorama (full turn = the texture once;
    the strip spans z = strip[0] .. strip[1], edge-clamped below / above)."""
    from sim.twin.visuals.assets import define_mesh
    from sim.twin.visuals.materials import bind, preview_material

    a = np.linspace(0.0, 2 * np.pi, seg + 1)
    ring = np.c_[center[0] + look.hall_radius * np.cos(a), center[1] + look.hall_radius * np.sin(a)]
    pts = np.r_[np.c_[ring, np.full(seg + 1, z0)], np.c_[ring, np.full(seg + 1, z1)]]
    nrm = np.tile(np.c_[-np.cos(a), -np.sin(a), np.zeros(seg + 1)], (2, 1))
    t0, t1 = (z0 - strip[0]) / (strip[1] - strip[0]), (z1 - strip[0]) / (strip[1] - strip[0])
    st = np.r_[np.c_[a / (2 * np.pi), np.full(seg + 1, t0)], np.c_[a / (2 * np.pi), np.full(seg + 1, t1)]]
    idx = np.array([[k, k + seg + 1, k + seg + 2, k + 1] for k in range(seg)]).ravel()
    g = look.hall_gain
    mat = preview_material(stage, f"{LOOKS}/Hall", (g, g, g), 1.0, linear=True, emissive=True,
                           texture=str(Path(look.hall_texture).resolve()))
    mesh = define_mesh(stage, f"/World/envs/env_{env}/HallVisual", pts, nrm, [4] * seg, idx, st=st,
                       double_sided=True)
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
    from sim.twin.visuals.paint import RobotLook, add_wrist_camera_mount, paint_robot

    look = look or TwinLook()
    stage = omni.usd.get_context().get_stage()
    info = {"render_settings_before": set_render_settings(look)}
    from sim.twin.lighting import rig

    info["lights"] = rig.build(stage, look, num_envs)
    robot_usd = getattr(profile, "robot_usd", None) or _rscene().ROBOT_USD
    info["robot_visual"] = [_robot_visual(stage, e, robot_usd) for e in range(num_envs)]
    rl = replace(RobotLook(), body=tuple(look.robot_body_rgb), body_roughness=look.robot_body_roughness,
                 servo=tuple(look.robot_servo_rgb), servo_roughness=look.robot_servo_roughness)
    info["robot_paint"] = paint_robot(num_envs, look=rl, robot="RobotVisual")
    if look.wrist_mount and profile is not None and getattr(profile, "wrist", None) is not None:
        info["wrist_mount"] = add_wrist_camera_mount(
            num_envs, profile.wrist.rows, gripper_rel="RobotVisual/" + _rscene().GRIPPER_LINK.split("/", 1)[1])
    L = LOOKS
    ext = json.loads(Path(look.table_texture).with_suffix(".json").read_text())
    wood = tuple(ext.get("mean_rgb_visible", (130, 132, 124)))
    nmap = Path(look.table_texture).parent / ext.get("normal_map", "_none_")
    m_table = preview_material(stage, f"{L}/TableTop", tuple(look.table_tint), look.table_roughness, linear=True,
                               clearcoat=look.table_clearcoat, clearcoat_roughness=look.table_clearcoat_roughness,
                               specular=look.table_specular,
                               texture=str(Path(look.table_texture).resolve()),
                               normal_texture=str(nmap.resolve()) if look.table_normal and nmap.exists() else None)
    m_wood = preview_material(stage, f"{L}/TableWood", wood, look.table_roughness)
    m_plate = {k: preview_material(stage, f"{L}/Plate_{k}", rgb, look.plate_roughness, clearcoat=look.plate_clearcoat,
                                   clearcoat_roughness=look.plate_clearcoat_roughness, specular=look.plate_specular)
               for k, rgb in look.plate_rgb.items()}
    m_brick = {k: preview_material(stage, f"{L}/Brick_{k}", rgb, look.brick_roughness, clearcoat=look.brick_clearcoat,
                                   clearcoat_roughness=look.brick_clearcoat_roughness, specular=look.brick_specular)
               for k, rgb in look.brick_rgb.items()}
    bricks = [("Brick", layout.target)] + [(f"Distractor{i}", d) for i, d in enumerate(layout.distractors)]
    n_plates = n_bricks = 0
    for e in range(num_envs):
        _table_quad(stage, e, look, m_table, m_wood)
        if look.backdrop_texture and Path(look.backdrop_texture).exists():
            _backdrop(stage, e, look)
        if look.hall_texture and Path(look.hall_texture).exists() and num_envs == 1:  # envs are 2.5 m apart
            _hall(stage, e, look)
            info["hall"] = True
        for name, (_x, _y, r, rgb) in layout.bowls.items():
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


def _ior(specular: float) -> float:
    """UsdPreviewSurface ior of a dielectric specular level (materials.preview_material: F0 = 0.08 * level)."""
    f0 = 0.08 * float(specular)
    return (1 + f0 ** 0.5) / max(1 - f0 ** 0.5, 1e-3)


def refresh_look(look: TwinLook) -> dict:
    """Re-apply `look`'s lights, material parameters and render settings on the live stage (after
    apply_twin_visuals; used by the fitting loop and to switch looks without rebuilding the scene). Textures, plate
    sizes and the geometry are not changed. Returns what was set."""
    import omni.usd
    from pxr import Gf

    from sim.twin.visuals.materials import srgb_to_linear

    stage = omni.usd.get_context().get_stage()
    done = {"render": set_render_settings(look) is not None, "attrs": 0}

    def put(path: str, name: str, value):
        prim = stage.GetPrimAtPath(path)
        attr = prim.GetAttribute(name) if prim else None
        if attr:
            attr.Set(value)
            done["attrs"] += 1

    def surface(mat: str, rgb255=None, **inputs):
        if rgb255 is not None:
            put(f"{LOOKS}/{mat}/Shader", "inputs:diffuseColor", Gf.Vec3f(*srgb_to_linear(rgb255)))
        for k, v in inputs.items():
            put(f"{LOOKS}/{mat}/Shader", f"inputs:{k}", float(v))

    s = look.scale()
    put("/World/Light", "inputs:intensity", look.dome_intensity * s)
    put("/World/Light", "inputs:color", Gf.Vec3f(*look.dome_rgb))
    put("/World/Sun", "inputs:intensity", look.key_intensity * s)
    put("/World/Sun", "inputs:angle", look.key_angle_deg)
    put("/World/Sun", "inputs:color", Gf.Vec3f(*look.key_rgb))
    from sim.twin.lighting import rig

    done["rig"] = rig.refresh(stage, look)
    surface("TableTop", roughness=look.table_roughness, clearcoat=look.table_clearcoat,
            clearcoatRoughness=look.table_clearcoat_roughness, ior=_ior(look.table_specular))
    surface("TableWood", roughness=look.table_roughness)
    put(f"{LOOKS}/TableTop/Tex", "inputs:scale", Gf.Vec4f(*look.table_tint, 1.0))
    for k, rgb in look.plate_rgb.items():
        surface(f"Plate_{k}", rgb, roughness=look.plate_roughness, clearcoat=look.plate_clearcoat,
                clearcoatRoughness=look.plate_clearcoat_roughness, ior=_ior(look.plate_specular))
    for k, rgb in look.brick_rgb.items():
        surface(f"Brick_{k}", rgb, roughness=look.brick_roughness, clearcoat=look.brick_clearcoat,
                clearcoatRoughness=look.brick_clearcoat_roughness, ior=_ior(look.brick_specular))
    surface("RobotBody", look.robot_body_rgb, roughness=look.robot_body_roughness)
    surface("RobotServo", look.robot_servo_rgb, roughness=look.robot_servo_roughness)
    for mat, g in (("Backdrop", look.bd_gain()), ("Hall", look.hall_gain)):
        put(f"{LOOKS}/{mat}/Tex", "inputs:scale", Gf.Vec4f(g, g, g, 1.0))
    return done


def color_correct(rgb, cam: str, path: Path = CAMERA_COLOR_JSON, sensor: bool = True):
    """Per-camera response: twin RAW render -> what the recorded dataset contains for `cam` ('top' | 'wrist' |
    'inspect'). Runs the camera pipeline of sim/twin/sensors/pipeline.py: wrist lens distortion, the linear-RGB colour
    matrix `M_linear` / `offset_linear` of camera_color.json (`path`; materials are matched to the top camera, so top
    is identity), auto exposure, vignetting, optics / motion blur, sensor noise, ISP sharpening and the dataset's
    H.264 (crf 30, g 2, yuv420p) round-trip, with the episode's image DR if TwinRandomizer registered one
    (randomize.py). sensor=False: only the colour matrix (the pre-sensor behaviour, for colour fits).
    HxWx3 uint8 numpy or torch in, the same type out. Cameras without a fit pass through the stages they have."""
    from sim.twin.sensors.pipeline import camera_response, color_only

    return camera_response(rgb, cam, path) if sensor else color_only(rgb, cam, path)


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
