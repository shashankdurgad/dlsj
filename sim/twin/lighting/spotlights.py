"""Lighting domain randomisation of the twin: nominal-rig jitter, ceiling-panel dimming and random spotlights.

    from sim.twin.lighting.spotlights import LIGHT_DR, draw_lights
    d = draw_lights(LIGHT_DR, rng, nominal, num_envs, scale=0.5)    # plain numpy, JSON-friendly
    # applied by sim/twin/env/randomize.py (TwinRandomizer._lights) on the live stage

Everything scales with ONE number `scale` in [0, 1] (cfg["light"]["scale"]; TwinRandomizer.set_scale(s)):
0 = the nominal fitted rig exactly, 1 = the full ranges below. Half-ranges are multiplied by scale; the number of
spotlights is drawn from 0 .. round(spots_max * scale); probabilities (ceiling panel off, coloured spot, panorama dome)
are multiplied by scale; the spot strength upper bound grows from its lower bound to `spot_strength[1]`.

What is drawn (one global draw for the shared rig, one draw per env for the spots):
1. global exposure: all lights x a log-uniform draw from the `exposure` range shrunk towards 1 by scale (the cameras
   ran auto exposure, so this mostly moves the light balance against the unlit hall / backdrop);
2. per nominal light: gain exp(U(-1, 1) * log(1 + gain_<kind>) * scale), colour temperature nominal * kelvin_gain(6500
   +- kelvin * scale) and a green-magenta tint +- tint * scale (fluorescent tubes vary in both), direction of the
   key / ceiling tubes tilted <= dir_deg * scale, side lamps moved <= side_pos_m * scale;
3. light-off event (probability p_off * scale): one ceiling tube (or the key, if there are no tubes) dimmed to
   U(off_gain) of its value;
4. spotlights: n ~ U{0 .. round(spots_max * scale)} per env, each a small sphere light with a cone (UsdLux
   ShapingAPI): position at height U(spot_height) above the table and U(spot_dist) horizontally from the workspace
   centre at a random azimuth, aimed at the workspace centre +- spot_aim_m, cone half-angle U(spot_cone_deg), cone
   softness U(spot_softness), source radius U(spot_radius) (shadow softness), strength = irradiance at the aim point
   relative to the nominal rig's table irradiance, U(spot_strength[0], lo + (hi - lo) * scale), colour temperature
   U(spot_kelvin), green-magenta tint +- spot_tint, and with probability p_spot_colour * scale a saturated hue
   (stage lights; the venue has purple ones) mixed in by U(spot_colour_sat).
Physics is never touched.
"""
from __future__ import annotations

import colorsys
import math

import numpy as np

LIGHT_DR = {
    "enabled": True,
    "scale": 1.0,  # 0 = nominal, 1 = full range (the RL shift ramp drives this: TwinRandomizer.set_light_scale)
    # nominal rig jitter (half-ranges at scale 1)
    "exposure": (0.8, 1.25),  # global multiplier at scale 1 (log-uniform; scale s: 1 + (lo - 1) s .. 1 + (hi - 1) s)
    "gain": {"dome": 0.35, "key": 0.4, "ceiling": 0.4, "side": 0.6},
    "kelvin": 900.0,  # +- K around each light's nominal colour
    "tint": 0.04,  # +- green-magenta (G channel vs R, B)
    "dir_deg": 10.0,  # key / ceiling tube direction tilt
    "side_pos_m": 0.4,  # side lamp position jitter (per axis)
    "p_off": 0.3,  # probability of one ceiling tube / the key being dimmed
    "off_gain": (0.0, 0.3),
    "p_dome_texture": 0.25,  # panorama dome (randomize.build_textures 'dome'), azimuth random
    # random spotlights
    "spots_max": 3,
    "spot_height": (0.35, 1.8),  # m above the table
    "spot_dist": (0.0, 1.4),  # m, horizontal distance from the workspace centre
    "spot_aim_m": 0.25,  # aim point = workspace centre + U(-a, a) in x / y
    "spot_cone_deg": (10.0, 60.0),  # half-angle
    "spot_softness": (0.0, 1.0),
    "spot_radius": (0.01, 0.15),  # source radius: small = hard shadows
    "spot_strength": (0.05, 1.5),  # irradiance at the aim point / nominal table irradiance
    "spot_kelvin": (2200.0, 9000.0),
    "spot_tint": 0.06,
    "p_spot_colour": 0.3,
    "spot_colour_sat": (0.3, 0.9),
}
WORK_CENTER = (0.30, 0.0)  # env frame x, y of the workspace centre
LUM = np.array([0.2126, 0.7152, 0.0722])


def _kelvin_srgb(k: float) -> np.ndarray:
    """Black-body colour (Tanner Helland's fit), sRGB 0-1."""
    t = k / 100.0
    r = 255.0 if t <= 66 else 329.698727446 * (t - 60) ** -0.1332047592
    g = 99.4708025861 * math.log(t) - 161.1195681661 if t <= 66 else 288.1221695283 * (t - 60) ** -0.0755148492
    b = 255.0 if t >= 66 else (0.0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307)
    return np.clip([r, g, b], 0, 255) / 255.0


def _lin(c):
    c = np.clip(np.asarray(c, float), 0, 1)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def kelvin_rgb(k: float, ref: float = 6500.0) -> np.ndarray:
    """Linear RGB multiplier of a black body at `k` kelvin relative to `ref`, unit luminance."""
    c = _lin(_kelvin_srgb(k)) / np.maximum(_lin(_kelvin_srgb(ref)), 1e-6)
    return c / float(c @ LUM)


def tint_rgb(t: float) -> np.ndarray:
    """Green (+) / magenta (-) tint multiplier, unit luminance."""
    c = np.array([1.0 - 0.5 * t, 1.0 + t, 1.0 - 0.5 * t])
    return c / float(c @ LUM)


def _sym(rng, half: float) -> float:
    return float(rng.uniform(-half, half)) if half > 0 else 0.0


def _gain(rng, half: float, scale: float) -> float:
    return float(math.exp(_sym(rng, math.log1p(half) * scale)))


def _tilt(d, deg: float, rng) -> list:
    """Unit vector `d` tilted by U(0, deg) degrees in a random direction (kept above the horizon)."""
    d = np.asarray(d, float) / np.linalg.norm(d)
    if deg <= 0:
        return d.tolist()
    a = np.cross(d, rng.normal(size=3))
    a /= np.linalg.norm(a)
    t = math.radians(rng.uniform(0, deg))
    out = d * math.cos(t) + np.cross(a, d) * math.sin(t)
    out[2] = max(out[2], 0.05)
    return (out / np.linalg.norm(out)).tolist()


def _r(x, n: int = 4):
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_r(v, n) for v in x]
    return round(float(x), n) if isinstance(x, (float, np.floating)) else x


def draw_spot(c: dict, rng, scale: float, center=WORK_CENTER) -> dict:
    """One random spotlight (env frame): pos, aim, cone, softness, radius, strength (relative), linear rgb."""
    az = rng.uniform(0, 2 * math.pi)
    rho = rng.uniform(*c["spot_dist"])
    pos = [center[0] + rho * math.cos(az), center[1] + rho * math.sin(az), rng.uniform(*c["spot_height"])]
    aim = [center[0] + _sym(rng, c["spot_aim_m"]), center[1] + _sym(rng, c["spot_aim_m"]), 0.0]
    lo, hi = c["spot_strength"]
    rgb = kelvin_rgb(rng.uniform(*c["spot_kelvin"])) * tint_rgb(_sym(rng, c["spot_tint"]))
    hue = None
    if rng.uniform() < c["p_spot_colour"] * scale:
        hue = float(rng.uniform())
        sat = rng.uniform(*c["spot_colour_sat"])
        col = _lin(colorsys.hsv_to_rgb(hue, 1.0, 1.0))
        col = col / float(col @ LUM)
        rgb = (1 - sat) * rgb + sat * col
    rgb = rgb / float(rgb @ LUM)
    return dict(pos=_r(pos), aim=_r(aim), cone_deg=_r(rng.uniform(*c["spot_cone_deg"]), 2),
                softness=_r(rng.uniform(*c["spot_softness"]), 3), radius=_r(rng.uniform(*c["spot_radius"])),
                strength=_r(rng.uniform(lo, lo + (hi - lo) * scale)), rgb=_r(rgb),
                hue=None if hue is None else _r(hue, 3))


def draw_lights(c: dict, rng, nominal: list[dict], num_envs: int, scale: float | None = None) -> dict:
    """Draw the lighting DR (see module doc). `nominal` = rig.nominal_lights(look). Returns a JSON-friendly dict:
    {scale, exposure, lights: [{kind, path / index, gain, rgb (linear multiplier of the nominal colour), dir?, pos?}],
    off: {kind, index, gain} | None, dome_texture: {azimuth_deg} | None, spots: [[spot, ...] per env]}."""
    s = float(np.clip(c.get("scale", 1.0) if scale is None else scale, 0.0, 1.0))
    lo, hi = (1.0 + (float(v) - 1.0) * s for v in c["exposure"])
    exp = float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
    out = {"scale": s, "exposure": _r(exp), "lights": [], "off": None, "dome_texture": None}
    for L in nominal:
        g = _gain(rng, c["gain"].get(L["kind"], 0.0), s)
        rgb = kelvin_rgb(6500.0 + _sym(rng, c["kelvin"] * s)) * tint_rgb(_sym(rng, c["tint"] * s))
        rec = {k: L[k] for k in ("kind", "path", "index") if k in L}
        rec.update(gain=_r(g), rgb=_r(rgb))
        if L["kind"] in ("key", "ceiling"):
            rec["dir"] = _r(_tilt(L["dir"], c["dir_deg"] * s, rng))
        if L["kind"] == "side":
            rec["pos"] = _r(np.asarray(L["cfg"]["pos"], float) + rng.uniform(-1, 1, 3) * c["side_pos_m"] * s *
                            np.array([1.0, 1.0, 0.5]))
        out["lights"].append(rec)
    if rng.uniform() < c["p_off"] * s:
        cand = [i for i, L in enumerate(nominal) if L["kind"] == "ceiling"] or \
               [i for i, L in enumerate(nominal) if L["kind"] == "key"]
        if cand:
            i = int(cand[rng.integers(len(cand))])
            g = rng.uniform(*c["off_gain"])
            out["lights"][i]["gain"] = _r(out["lights"][i]["gain"] * g)
            out["off"] = {"kind": nominal[i]["kind"], "index": nominal[i].get("index"), "gain": _r(g)}
    if rng.uniform() < c["p_dome_texture"] * s:
        out["dome_texture"] = {"azimuth_deg": _r(rng.uniform(0, 360), 1)}
    nmax = int(round(c["spots_max"] * s))
    out["spots"] = [[draw_spot(c, rng, s) for _ in range(int(rng.integers(0, nmax + 1)))] for _ in range(num_envs)]
    return out


def spot_radiance(strength: float, e_nominal: float, pos, aim, radius: float) -> float:
    """SphereLight intensity (normalize off) for `strength` x the nominal table irradiance at the aim point."""
    from sim.twin.lighting.rig import sphere_radiance

    d = float(np.linalg.norm(np.asarray(pos, float) - np.asarray(aim, float)))
    return sphere_radiance(strength * e_nominal, d, radius)


def apply_spots(stage, set_attr, set_pose, env: int, spots: list[dict], e_nominal: float, n_slots: int) -> list[str]:
    """Author the spots of one env on /World/envs/env_<e>/TwinDRLights/Spot<j> (created once with intensity 0, so the
    randomiser's restore returns them to dark). set_attr(path, name, value) / set_pose(path, pos, quat_xyzw) are the
    randomiser's recording setters. Unused slots get intensity 0."""
    from pxr import Gf, UsdLux

    from sim.twin.lighting.rig import quat_z_to

    paths = []
    for j in range(max(n_slots, len(spots))):
        path = f"/World/envs/env_{env}/TwinDRLights/Spot{j}"
        if not stage.GetPrimAtPath(path):
            lt = UsdLux.SphereLight.Define(stage, path)
            lt.CreateIntensityAttr(0.0)
            lt.CreateRadiusAttr(0.05)
            lt.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
            sh = UsdLux.ShapingAPI.Apply(lt.GetPrim())
            sh.CreateShapingConeAngleAttr(45.0)
            sh.CreateShapingConeSoftnessAttr(0.5)
        paths.append(path)
        if j >= len(spots):
            set_attr(path, "inputs:intensity", 0.0)
            continue
        sp = spots[j]
        pos, aim = np.asarray(sp["pos"], float), np.asarray(sp["aim"], float)
        set_pose(path, pos, quat_z_to(pos - aim))
        set_attr(path, "inputs:radius", float(sp["radius"]))
        set_attr(path, "inputs:shaping:cone:angle", float(sp["cone_deg"]))
        set_attr(path, "inputs:shaping:cone:softness", float(sp["softness"]))
        set_attr(path, "inputs:color", Gf.Vec3f(*map(float, sp["rgb"])))
        set_attr(path, "inputs:intensity", spot_radiance(sp["strength"], e_nominal, pos, aim, sp["radius"]))
    return paths
