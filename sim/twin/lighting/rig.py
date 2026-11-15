"""The twin's nominal light rig (USD), fitted to the real station videos (lighting/fit_rig.py, log.md).

    from sim.twin.lighting import rig
    info = rig.build(stage, look, num_envs)     # apply_twin_visuals: ceiling tubes + side lamps + RTX ambient
    rig.refresh(stage, look)                    # twin_scene.refresh_look: re-apply intensities / colours / poses

The dome (/World/Light) and the soft overhead key (/World/Sun) are spawned by twin_scene.twin_scene_cfg from the
TwinLook fields; this module adds what lighting.json describes beyond them:
* ceiling_lights: the fluorescent tubes ~10 m above the table whose reflections are the highlights on the glossy
  plates. Entry: {"dir": unit vector from the workspace centre TO the light, "intensity": irradiance it gives at the
  workspace centre on a surface facing it (distant-light units, before the exposure scale), "rgb", and either
  "type": "rect" with "dist" (m), "size" ([long, short] m), "yaw_deg" (long axis about the light's axis), or the
  legacy distant light ("angle_deg")}. Rect lights are global prims (/World/TwinCeil<i>) placed relative to env 0:
  10 m away, the direction changes by < 4 deg over neighbouring envs.
* side_lights: warm lamps beside the station, per env (/World/envs/env_<e>/TwinSide<i>): {"pos" (env frame, m),
  "radius" (m), "intensity" (irradiance at the workspace centre, as above), "rgb"}.
* ambient_intensity: RTX's flat, unshadowed ambient term (/rtx/sceneDb/ambientLightIntensity; Isaac Lab's default 1.0
  with colour 0.1 adds ~18 % of the nominal table light without any occlusion, which washes out shadows). The fitted
  rig sets it to 0 and lets the dome provide the (occluded) ambient light.
Intensities are stored as irradiance at the workspace centre so the visual fit (visuals/fit/fit_visuals.py, which
scales ceiling "intensity" values) keeps its meaning for any light type.
"""
from __future__ import annotations

import math

import numpy as np

CENTER = np.array([0.30, 0.0, 0.0])  # workspace centre, env frame (table top ~ z 0)
AMBIENT_KEY = "/rtx/sceneDb/ambientLightIntensity"


def rect_radiance(irradiance: float, dist: float, size) -> float:
    """USD RectLight intensity (radiance, normalize off) that gives `irradiance` at `dist` on a surface facing it."""
    return float(irradiance) * dist ** 2 / max(float(size[0]) * float(size[1]), 1e-9)


def sphere_radiance(irradiance: float, dist: float, radius: float) -> float:
    """USD SphereLight intensity (radiance, normalize off) that gives `irradiance` at `dist` (projected disk pi r^2)."""
    return float(irradiance) * dist ** 2 / (math.pi * max(float(radius), 1e-6) ** 2)


def quat_z_to(direction) -> tuple:
    """xyzw quaternion turning +z into `direction` (lights shine along their -z: pass the vector TO the light)."""
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    axis = np.cross([0.0, 0.0, 1.0], d)
    n = np.linalg.norm(axis)
    if n < 1e-9:
        return (0.0, 0.0, 0.0, 1.0) if d[2] > 0 else (1.0, 0.0, 0.0, 0.0)
    half = 0.5 * math.atan2(n, d[2])
    axis = axis / n * math.sin(half)
    return (float(axis[0]), float(axis[1]), float(axis[2]), math.cos(half))


def _pose(stage, path: str, pos, quat_xyzw):
    """Author translate + orient on a light prim (replacing earlier ops)."""
    from pxr import Gf, UsdGeom

    xf = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
    x, y, z, w = quat_xyzw
    xf.AddOrientOp().Set(Gf.Quatf(float(w), float(x), float(y), float(z)))


def _yawed(dirv, yaw_deg: float) -> tuple:
    """Orientation of a rect light facing CENTER along -z, its long (x) axis horizontal, turned by yaw about
    its axis."""
    from scipy.spatial.transform import Rotation

    d = np.asarray(dirv, float) / np.linalg.norm(dirv)
    zx = np.cross([0.0, 0.0, 1.0], d)
    x = zx / np.linalg.norm(zx) if np.linalg.norm(zx) > 1e-6 else np.array([1.0, 0.0, 0.0])
    y = np.cross(d, x)
    R = Rotation.from_matrix(np.stack([x, y, d], 1)) * Rotation.from_euler("z", yaw_deg, degrees=True)
    return tuple(R.as_quat())


def ceiling_pose(c: dict) -> tuple:
    """(position (3,), quat xyzw) of a ceiling light entry (env 0 frame; distant lights: position 0)."""
    if c.get("type", "distant") == "rect":
        d = np.asarray(c["dir"], float) / np.linalg.norm(c["dir"])
        return CENTER + float(c.get("dist", 10.0)) * d, _yawed(d, float(c.get("yaw_deg", 0.0)))
    return np.zeros(3), quat_z_to(c["dir"])


def _ceiling(stage, i: int, c: dict, s: float) -> str:
    from pxr import Gf, UsdLux

    path = f"/World/TwinCeil{i}"
    rgb = Gf.Vec3f(*map(float, c.get("rgb", (1.0, 1.0, 1.0))))
    if c.get("type", "distant") == "rect":
        size = c.get("size", (1.2, 0.3))
        old = stage.GetPrimAtPath(path)
        if old and old.GetTypeName() != "RectLight":
            stage.RemovePrim(path)
        lt = UsdLux.RectLight.Define(stage, path)
        lt.CreateWidthAttr(float(size[0]))
        lt.CreateHeightAttr(float(size[1]))
        lt.CreateIntensityAttr(rect_radiance(float(c["intensity"]) * s, float(c.get("dist", 10.0)), size))
    else:
        old = stage.GetPrimAtPath(path)
        if old and old.GetTypeName() != "DistantLight":
            stage.RemovePrim(path)
        lt = UsdLux.DistantLight.Define(stage, path)
        lt.CreateAngleAttr(float(c.get("angle_deg", 4.0)))
        lt.CreateIntensityAttr(float(c["intensity"]) * s)
    lt.CreateColorAttr(rgb)
    _pose(stage, path, *ceiling_pose(c))
    return path


def _side(stage, e: int, i: int, c: dict, s: float) -> str:
    from pxr import Gf, UsdLux

    path = f"/World/envs/env_{e}/TwinSide{i}"
    pos = np.asarray(c["pos"], float)
    lt = UsdLux.SphereLight.Define(stage, path)
    lt.CreateRadiusAttr(float(c.get("radius", 0.1)))
    lt.CreateIntensityAttr(sphere_radiance(float(c["intensity"]) * s, float(np.linalg.norm(pos - CENTER)),
                                           float(c.get("radius", 0.1))))
    lt.CreateColorAttr(Gf.Vec3f(*map(float, c.get("rgb", (1.0, 1.0, 1.0)))))
    _pose(stage, path, pos, (0.0, 0.0, 0.0, 1.0))
    return path


def apply_rtx(look) -> dict:
    """RTX lighting settings of the rig (flat ambient term). Returns the previous values."""
    import carb

    st = carb.settings.get_settings()
    before = {AMBIENT_KEY: st.get(AMBIENT_KEY)}
    amb = getattr(look, "ambient_intensity", None)
    if amb is not None:
        st.set(AMBIENT_KEY, float(amb))
    return before


def build(stage, look, num_envs: int) -> dict:
    """Create the rig's ceiling tubes (global) and side lamps (per env) on the stage and apply the RTX settings."""
    s = look.scale()
    ceil = [_ceiling(stage, i, c, s) for i, c in enumerate(look.ceiling_lights)]
    side = [_side(stage, e, i, c, s) for e in range(num_envs) for i, c in enumerate(getattr(look, "side_lights", []))]
    return {"ceiling": ceil, "side": side, "rtx_before": apply_rtx(look)}


def refresh(stage, look) -> dict:
    """Re-apply the rig of `look` on a live stage (prims are re-authored in place; env count from the stage)."""
    n = 0
    while stage.GetPrimAtPath(f"/World/envs/env_{n}"):
        n += 1
    return build(stage, look, max(n, 1))


def nominal_lights(look) -> list[dict]:
    """Every light of the rig as plain dicts (kind, dir TO the light from CENTER, irradiance at CENTER before the
    exposure scale, rgb): the reference the lighting DR jitters around and that its spot strengths are relative to."""
    from scipy.spatial.transform import Rotation

    k = Rotation.from_euler("xyz", look.key_rot_xyz_deg, degrees=True).apply([0.0, 0.0, 1.0])
    out = [dict(kind="dome", path="/World/Light", irradiance=float(look.dome_intensity), rgb=list(look.dome_rgb)),
           dict(kind="key", path="/World/Sun", dir=k.tolist(), irradiance=float(look.key_intensity),
                rgb=list(look.key_rgb))]
    for i, c in enumerate(look.ceiling_lights):
        out.append(dict(kind="ceiling", path=f"/World/TwinCeil{i}", index=i, dir=list(c["dir"]),
                        irradiance=float(c["intensity"]), rgb=list(c.get("rgb", (1.0, 1.0, 1.0))), cfg=dict(c)))
    for i, c in enumerate(getattr(look, "side_lights", [])):
        d = np.asarray(c["pos"], float) - CENTER
        out.append(dict(kind="side", index=i, dir=(d / np.linalg.norm(d)).tolist(), irradiance=float(c["intensity"]),
                        rgb=list(c.get("rgb", (1.0, 1.0, 1.0))), cfg=dict(c)))
    return out


def table_irradiance(look) -> float:
    """Approximate direct irradiance of the rig on the horizontal table at CENTER (before the exposure scale; the dome
    as a uniform sky above the hall cylinder's ~41 deg horizon: pi * I * (1 - sin^2(41 deg)))."""
    e = math.pi * float(look.dome_intensity) * (1.0 - math.sin(math.radians(41.0)) ** 2)
    for L in nominal_lights(look):
        if L["kind"] != "dome":
            e += L["irradiance"] * max(0.0, float(L["dir"][2]))
    return e
