"""Visual domain randomisation for the SO-101 twin (cameras, lights, materials, textures, clutter, image effects).

    from sim.twin.env.randomize import TwinRandomizer, postprocess
    dr = TwinRandomizer({"clutter": {"enabled": False}})     # partial overrides of DEFAULTS (deep-merged)
    draw = dr(env, rng)          # on reset, before rendering: env.sim / scene / stage / visuals / profile (+ env.sync)
    img = postprocess(img, "wrist", draw)       # per rendered frame, AFTER twin_scene.color_correct (image-space DR)
    dr.restore(env)              # back to the nominal look (e.g. an evaluation reset after DR episodes)

    python sim/twin/env/randomize.py --build-textures          # photo-derived texture variants (no Isaac; cv2)

Everything acts on what the GB10 renderer actually reads (plain USD; the Fabric scene delegate is off): the camera
prims' local xforms and USD focal lengths, the light prims, the twin's materials / texture files, the visual proxies
(`TwinVis/*`, `RobotVisual`), and VisualSync's wrist-camera offset. Physics is never touched.

What is randomised, and why these ranges (real reference: sim/twin/out/dr/real_stats.json, 96 frames = start + mid
of 48 cal2 episodes; `test_randomize.py --real-stats`):
1. cameras: top / inspect pose + focal per env around env.profile's nominal, with sim.twin.calib.profile.DR's
   half-ranges (top 1 cm / 1 deg / 1 %, inspect 5 cm / 5 deg / 2 % (swivel tripod), wrist 5 mm / 3 deg / 3 %),
   the same sampling as profile.randomize (uniform position, rotation about a random axis). The wrist camera's mount
   offset is applied through VisualSync (it re-places the wrist camera every render from `sync.cams[e][2]`), and the
   orange mount plate / board meshes move with it (else a camera pushed backwards would look through the plate).
   Focal: one draw per camera per call, shared by all envs (Isaac Lab: per-env intrinsics are not supported by the
   tiled renderer, see Camera.set_intrinsic_matrices). sensor.data.intrinsic_matrices is NOT updated (USD only).
2. lights: overall exposure 0.88-1.06 (real top table luma p5-p95 = -6 % / +4 % of the median; the nominal sim table
   renders ~4 % brighter; image-space gain adds a little), key/dome balance 0.7-1.35, colour temperature dome 5800-7200 K / key 5500-7500 K (normalised to 6500 K
   and constant luminance; real top table R/G spans +-4.5 %, B/G +-9 %), key direction tilted <= 20 deg from nominal
   with a free azimuth, key softness 8-25 deg; optional dome texture from the station panorama (p 0.25).
3. materials / textures: brick colours = the env's own palette class (twin_scene.BRICK_RGB, measured) with hue +-5 deg,
   saturation / value x0.85-1.15; distractors re-drawn from the palette (excluding the target's colour class;
   the target's class is never changed, so the instruction stays valid); plate colours hue +-4 deg, S x0.9-1.1,
   V x0.88-1.12, roughness 0.12-0.35; table texture variant (camera-projected original / phone-photo wood outside
   the camera-seen area / phone-photo wood everywhere) + brightness x0.92-1.06, per-channel tint +-3 %, texture shift
   +-2 cm, roughness 0.55-0.85; robot body value x0.7-1.5 (near black), roughness 0.35-0.75; jaw hue +-4 deg;
   inspect backdrop variant (median of the real inspect frames / 3 crops of the station panorama) and gain 0.8-1.2.
4. clutter (visual only, no colliders): a dark cable (p 0.5, 20-45 cm, curved) and a hand-sized dark blob (p 0.3)
   on the table inside the top camera's view, kept clear of plates, bricks and the robot base.
5. image (postprocess): per camera gain / contrast / white balance / gaussian blur / wrist motion blur / gaussian
   noise / JPEG re-encode; the wrist camera is blurred most because the real wrist frames are mostly soft (variance
   of the Laplacian median 15.5, p5 6.4, vs 122 top and 285 inspect). The twin's top / inspect renders are already
   much softer than the real frames (texture resolution), so they get little blur. The inspect gain is centred
   above 1 (0.97-1.2) because the nominal twin's inspect image is ~15 % darker than the real one (sim luma median
   88 vs real 103); revisit if the core look is recalibrated. Known nominal bias not covered: inspect R/G sim 1.19
   vs real 1.05 (twin_scene.color_correct('inspect')).
"""
from __future__ import annotations

import colorsys
import copy
import json
import math
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
VIS = REPO / "sim/twin/visuals"
PHOTOS = REPO / "sim/twin/data/photos"
TEX_OUT = REPO / "sim/twin/out/dr/textures"  # git-ignored (derived from git-ignored photos)
CAMS = ("top", "wrist", "inspect")
SENSOR = {"top": "top_cam", "wrist": "wrist_cam", "inspect": "inspect_cam"}
LOOKS = "/World/Looks/Twin"
DR_LOOKS = "/World/Looks/TwinDR"
PLATE_D = 0.184  # m, real plate diameter (scale reference in the top-down photo)

DEFAULTS = {
    "camera": {"enabled": True, "pose": True, "focal": True, "cams": ["top", "inspect", "wrist"],
               "half_range": None,  # {cam: (pos m, rot deg, focal frac)}; None = sim.twin.calib.profile.DR
               "scale": 1.0},  # multiplies every half-range
    "light": {"enabled": True, "exposure": (0.88, 1.06), "key_gain": (0.7, 1.35), "dome_kelvin": (5800.0, 7200.0),
              "key_kelvin": (5500.0, 7500.0), "key_tilt_deg": 20.0, "key_azimuth_deg": 180.0,
              "key_angle_deg": (8.0, 25.0), "p_dome_texture": 0.25},
    "material": {"enabled": True,
                 "brick_hue_deg": 5.0, "brick_sat": (0.85, 1.15), "brick_val": (0.85, 1.15),
                 "brick_roughness": (0.2, 0.45), "recolor_distractors": True, "recolor_target": False,
                 "plate_hue_deg": 4.0, "plate_sat": (0.9, 1.1), "plate_val": (0.88, 1.12),
                 "plate_roughness": (0.12, 0.35),
                 "table_variants": {"original": 0.4, "photo_outside": 0.4, "photo_full": 0.2},
                 "table_gain": (0.92, 1.06), "table_tint": 0.03, "table_shift_m": 0.02, "table_roughness": (0.55, 0.85),
                 "robot_body_val": (0.7, 1.5), "robot_roughness": (0.35, 0.75), "jaw_hue_deg": 4.0,
                 "jaw_val": (0.9, 1.1),
                 "backdrop_variants": {"original": 0.4, "pano0": 0.2, "pano1": 0.2, "pano2": 0.2},
                 "backdrop_gain": (0.8, 1.2)},
    "clutter": {"enabled": True, "p_cable": 0.5, "p_blob": 0.3, "cable_len": (0.2, 0.45), "cable_radius": 0.003,
                "cable_segments": 12, "blob_size": ((0.045, 0.03, 0.015), (0.065, 0.045, 0.025)),
                "clearance": {"bowl": 0.115, "brick": 0.045, "robot": 0.16}},
    "image": {"enabled": True,
              "top": {"gain": (0.96, 1.04), "contrast": (0.95, 1.05), "wb": 0.03, "p_blur": 0.1,
                      "blur_sigma": (0.3, 0.6), "p_motion": 0.0, "motion_len": (3, 7), "noise": (0.0, 3.0),
                      "p_jpeg": 0.5, "jpeg_q": (40, 90)},
              "wrist": {"gain": (0.92, 1.15), "contrast": (0.9, 1.1), "wb": 0.07, "p_blur": 0.5,
                        "blur_sigma": (0.5, 1.6), "p_motion": 0.3, "motion_len": (3, 13), "noise": (0.0, 4.0),
                        "p_jpeg": 0.5, "jpeg_q": (35, 85)},
              "inspect": {"gain": (0.97, 1.2), "contrast": (0.93, 1.07), "wb": 0.05, "p_blur": 0.15,
                          "blur_sigma": (0.3, 0.7), "p_motion": 0.0, "motion_len": (3, 7), "noise": (0.0, 3.0),
                          "p_jpeg": 0.5, "jpeg_q": (40, 90)}},
}


def _merge(a: dict, b: dict | None) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) and \
            not k.endswith("variants") else copy.deepcopy(v)
    return out


# ------------------------------------------------------------------------------------------------ colour helpers
def srgb_to_lin(c):
    c = np.clip(np.asarray(c, float), 0, 1)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(c):
    c = np.clip(np.asarray(c, float), 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def _kelvin_srgb(k: float) -> np.ndarray:
    """Black-body colour (Tanner Helland's fit), sRGB 0-1."""
    t = k / 100.0
    r = 255.0 if t <= 66 else 329.698727446 * (t - 60) ** -0.1332047592
    g = 99.4708025861 * math.log(t) - 161.1195681661 if t <= 66 else 288.1221695283 * (t - 60) ** -0.0755148492
    b = 255.0 if t >= 66 else (0.0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307)
    return np.clip([r, g, b], 0, 255) / 255.0


def kelvin_gain(k: float, ref: float = 6500.0) -> np.ndarray:
    """Linear RGB multiplier of a light at `k` kelvin relative to `ref`, normalised to unit luminance."""
    c = srgb_to_lin(_kelvin_srgb(k)) / np.maximum(srgb_to_lin(_kelvin_srgb(ref)), 1e-6)
    return c / float(c @ [0.2126, 0.7152, 0.0722])


def jitter_srgb(rgb255, rng, hue_deg: float, sat=(1, 1), val=(1, 1)) -> tuple:
    r, g, b = (np.clip(np.asarray(rgb255, float), 0, 255) / 255.0).tolist()
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h = (h + rng.uniform(-hue_deg, hue_deg) / 360.0) % 1.0
    s = float(np.clip(s * rng.uniform(*sat), 0, 1))
    v = float(np.clip(v * rng.uniform(*val), 0, 1))
    return tuple(float(x) * 255.0 for x in colorsys.hsv_to_rgb(h, s, v))


def _choice(rng, probs: dict, available) -> str:
    keys = [k for k, p in probs.items() if p > 0 and k in available]
    if not keys:
        return "original"
    p = np.array([probs[k] for k in keys], float)
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


def _r(x, n=4):
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_r(v, n) for v in x]
    return round(float(x), n) if isinstance(x, (float, np.floating)) else x


# ------------------------------------------------------------------------------------------------ randomiser
class TwinRandomizer:
    """Called by the core env on reset as `draw = dr(env, rng)`; returns what it drew (JSON-friendly)."""

    def __init__(self, cfg: dict | None = None):
        self.cfg = _merge(DEFAULTS, cfg)
        self._nom_attr: dict = {}  # (path, attr) -> nominal value (restore)
        self._nom_xf: dict = {}  # prim path -> nominal local Gf.Matrix4d
        self._nom_bind: dict = {}  # mesh path -> nominal material path
        self._brick_class: dict = {}  # (env, prim) -> palette class while bound to a DR material
        self._wrist_nominal: dict = {}  # env -> nominal local0 of the wrist camera (VisualSync)
        self._textures: dict | None = None
        self._focal_touched: dict = {}  # sensor prim path -> (sensor, nominal intr)
        self.last: dict = {}

    # ---------------------------------------------------------------- USD write helpers (record nominal first)
    def _set(self, stage, path: str, name: str, value, vtype=None) -> bool:
        prim = stage.GetPrimAtPath(path)
        if not prim:
            return False
        a = prim.GetAttribute(name)
        if not a:
            if vtype is None:
                return False
            a = prim.CreateAttribute(name, vtype)
        key = (path, name)
        if key not in self._nom_attr:
            self._nom_attr[key] = a.Get() if a.HasAuthoredValue() else None
        a.Set(value)
        return True

    def _get_nom(self, stage, path: str, name: str):
        key = (path, name)
        if key in self._nom_attr:
            return self._nom_attr[key]
        prim = stage.GetPrimAtPath(path)
        a = prim.GetAttribute(name) if prim else None
        return a.Get() if a else None

    def _set_xf(self, stage, path: str, mat) -> bool:
        from pxr import Usd, UsdGeom

        from sim.twin.visuals.usd_sync import set_local

        prim = stage.GetPrimAtPath(path)
        if not prim:
            return False
        if path not in self._nom_xf:
            self._nom_xf[path] = UsdGeom.Xformable(prim).GetLocalTransformation(Usd.TimeCode.Default())
        set_local(prim, mat)
        return True

    def _bind(self, stage, mesh_path: str, mat_path: str):
        from pxr import UsdShade

        from sim.twin.visuals.materials import bind

        mesh = stage.GetPrimAtPath(mesh_path)
        if not mesh:
            return
        if mesh_path not in self._nom_bind:
            self._nom_bind[mesh_path] = str(UsdShade.MaterialBindingAPI(mesh).GetDirectBinding().GetMaterialPath())
        bind(mesh, UsdShade.Material(stage.GetPrimAtPath(mat_path)))

    def _material(self, stage, path: str, rgb255, roughness: float, clearcoat: float = 0.0):
        """A per-env DR material (created once, then only its inputs change)."""
        from pxr import Gf

        from sim.twin.visuals.materials import preview_material

        if not stage.GetPrimAtPath(path):
            preview_material(stage, path, rgb255, roughness, clearcoat=clearcoat)
        sh = stage.GetPrimAtPath(path + "/Shader")
        sh.GetAttribute("inputs:diffuseColor").Set(Gf.Vec3f(*map(float, srgb_to_lin(np.asarray(rgb255) / 255.0))))
        sh.GetAttribute("inputs:roughness").Set(float(roughness))

    # ---------------------------------------------------------------- entry points
    def __call__(self, env, rng: np.random.Generator) -> dict:
        t0 = time.perf_counter()
        stage = getattr(env, "stage", None)
        if stage is None:
            import omni.usd

            stage = omni.usd.get_context().get_stage()
        envs = list(range(env.scene.num_envs))
        c = self.cfg
        draw: dict = {}
        if c["camera"]["enabled"]:
            draw["camera"] = self._cameras(env, stage, envs, rng)
        if c["light"]["enabled"]:
            draw["light"] = self._lights(env, stage, rng)
        if c["material"]["enabled"]:
            draw["material"] = self._materials(env, stage, envs, rng)
        if c["clutter"]["enabled"]:
            draw["clutter"] = self._clutter(env, stage, envs, rng)
        if c["image"]["enabled"]:
            draw["image"] = self._image(rng)
        draw["time_s"] = round(time.perf_counter() - t0, 4)
        self.last = draw
        try:
            env.dr_draw = draw  # so the env's render() can call postprocess(img, cam, env.dr_draw)
        except Exception:
            pass
        return draw

    def postprocess(self, img, cam: str, draw: dict | None = None, rng=None):
        return postprocess(img, cam, self.last if draw is None else draw, rng)

    def restore(self, env=None):
        """Undo every change (nominal attributes, xforms, bindings, wrist offset; clutter hidden)."""
        from pxr import Gf, Sdf, UsdGeom  # noqa: F401

        from sim.twin.visuals.usd_sync import set_local

        import omni.usd

        stage = getattr(env, "stage", None) or omni.usd.get_context().get_stage()
        for (path, name), v in self._nom_attr.items():
            prim = stage.GetPrimAtPath(path)
            a = prim.GetAttribute(name) if prim else None
            if a:
                a.Clear() if v is None else a.Set(v)
        for path, m in self._nom_xf.items():
            prim = stage.GetPrimAtPath(path)
            if prim:
                set_local(prim, m)
        for sensor, intr in list(self._focal_touched.values()):
            self._focal_api(sensor, intr, 1.0)
        for mesh_path, mat_path in self._nom_bind.items():
            if mat_path:
                self._bind(stage, mesh_path, mat_path)
        self._brick_class.clear()
        sync = _find_sync(env) if env is not None else None
        if sync is not None:
            for e, local0 in self._wrist_nominal.items():
                if e in sync.cams:
                    cam, bi, _, stale = sync.cams[e]
                    sync.cams[e] = (cam, bi, local0, stale)
        for p in stage.Traverse():
            if p.GetName() == "TwinDR" and "/World/envs/" in str(p.GetPath()):
                UsdGeom.Imageable(p).MakeInvisible()
        self.last = {}
        if env is not None:
            try:
                env.dr_draw = {}
            except Exception:
                pass

    # ---------------------------------------------------------------- 1. cameras
    def _cameras(self, env, stage, envs, rng) -> dict:
        from pxr import Gf, UsdGeom
        from scipy.spatial.transform import Rotation

        from sim.twin.visuals.usd_sync import env_path

        c = self.cfg["camera"]
        hr = c["half_range"]
        if hr is None:
            from sim.twin.calib.profile import DR as hr
        prof = env.profile
        out = {}
        for cam in c["cams"]:
            key = SENSOR[cam]
            nom = getattr(prof, cam, None)
            if key not in env.scene.sensors or nom is None:
                continue
            dp, dr, df = (float(v) * float(c["scale"]) for v in hr[cam])
            rows0 = np.asarray(nom.rows, float)
            fscale = 1.0 + rng.uniform(-df, df) if c["focal"] else 1.0
            rec = {"focal_scale": _r(fscale), "dpos": [], "drot_deg": []}
            sensor = env.scene.sensors[key]
            sync = _find_sync(env) if cam == "wrist" else None
            for e in envs:
                rows = rows0.copy()
                if c["pose"]:
                    axis = rng.normal(size=3)
                    axis /= np.linalg.norm(axis)
                    ang = rng.uniform(-dr, dr)
                    Q = Rotation.from_rotvec(axis * np.radians(ang)).as_matrix()
                    d = rng.uniform(-dp, dp, 3)
                    rows[:3, :3] = rows0[:3, :3] @ Q.T
                    rows[3, :3] = rows0[3, :3] + d
                    rec["dpos"].append(_r(d))
                    rec["drot_deg"].append(_r(ang, 3))
                M = Gf.Matrix4d(*[float(v) for v in rows.flatten()])
                path = env_path(sensor.cfg.prim_path, e)
                if cam == "wrist":
                    self._wrist(stage, sync, e, rows0, M, rec)
                else:
                    self._set_xf(stage, path, M)
                if c["focal"]:
                    self._set(stage, path, "focalLength", float(nom.intr["focal_length"] * fscale))
                    if not UsdGeom.Camera(stage.GetPrimAtPath(path)):
                        rec["warn"] = "no USD camera at " + path
            if c["focal"]:
                rec["focal_api"] = self._focal_api(sensor, nom.intr, fscale)
            out[cam] = rec
        return out

    def _focal_api(self, sensor, intr, fscale) -> bool:
        """The renderer takes runtime intrinsics from the Camera sensor (Kit: Fabric), not from the USD attribute
        (measured: a USD-only focal change does not change the image), so set them through Isaac Lab as well."""
        try:
            import torch

            h, w = sensor.image_shape
            f = intr["focal_length"] / intr["horizontal_aperture"] * w * fscale
            K = torch.tensor([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=torch.float32, device=getattr(sensor, "_device", None) or "cuda:0")
            sensor.set_intrinsic_matrices(K.expand(sensor.num_instances, 3, 3).contiguous(),
                                          focal_length=float(intr["focal_length"] * fscale))
            self._focal_touched[sensor.cfg.prim_path] = (sensor, intr)
            return True
        except Exception as ex:  # noqa: BLE001
            print(f"[TwinRandomizer] set_intrinsic_matrices failed: {ex}", flush=True)
            return False

    def _wrist(self, stage, sync, e, rows0, M, rec):
        """Wrist mount offset: VisualSync re-places the wrist camera every render as local0 * gripper pose, so the
        perturbed offset goes into sync.cams[e]; the mount plate / board (RobotVisual gripper_link) follow it."""
        from pxr import Gf

        from sim.twin.visuals.twin_scene import _rscene

        if sync is None or e not in getattr(sync, "cams", {}):
            rec["warn"] = "no VisualSync with a wrist camera found (env.sync): wrist pose not randomised"
            return
        cam, bi, local0, stale = sync.cams[e]
        self._wrist_nominal.setdefault(e, local0)
        sync.cams[e] = (cam, bi, M, stale)
        N = Gf.Matrix4d(*[float(v) for v in rows0.flatten()])
        delta = N.GetInverse() * M  # camera-rigid motion in the gripper frame (row-vector convention)
        g = f"/World/envs/env_{e}/RobotVisual/" + _rscene().GRIPPER_LINK.split("/", 1)[1]
        for name in ("CamMountPlate", "CamBoard"):
            self._set_xf(stage, f"{g}/{name}", delta)

    # ---------------------------------------------------------------- 2. lights
    def _lights(self, env, stage, rng) -> dict:
        from pxr import Gf, Sdf

        c = self.cfg["light"]
        dome = getattr(getattr(env.scene.cfg, "dome", None), "prim_path", "/World/Light")
        key = getattr(getattr(env.scene.cfg, "sun", None), "prim_path", "/World/Sun")
        exp = rng.uniform(*c["exposure"])
        kg = rng.uniform(*c["key_gain"])
        dk, kk = rng.uniform(*c["dome_kelvin"]), rng.uniform(*c["key_kelvin"])
        out = {"exposure": _r(exp), "key_gain": _r(kg), "dome_kelvin": round(dk), "key_kelvin": round(kk)}
        tex = self._tex()
        use_tex = "dome" in tex and rng.uniform() < c["p_dome_texture"]
        dome_gain = 1.0
        if use_tex:
            dome_gain = tex["dome"]["gain"]
            self._set(stage, dome, "inputs:texture:file", Sdf.AssetPath(tex["dome"]["file"]), Sdf.ValueTypeNames.Asset)
            self._set(stage, dome, "inputs:texture:format", "latlong", Sdf.ValueTypeNames.Token)
            az = rng.uniform(0, 360)
            nom = self._nom_xf_of(stage, dome)
            self._set_xf(stage, dome, nom * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), az)))
            out["dome_texture"] = {"azimuth_deg": round(az, 1)}
        elif (dome, "inputs:texture:file") in self._nom_attr:  # back to the untextured dome
            self._restore_attr(stage, dome, "inputs:texture:file")
            if dome in self._nom_xf:
                self._set_xf(stage, dome, self._nom_xf[dome])
        for path, gain, kel in ((dome, exp * dome_gain, dk), (key, exp * kg, kk)):
            i0 = self._get_nom(stage, path, "inputs:intensity")
            col0 = self._get_nom(stage, path, "inputs:color")
            if i0 is not None:
                self._set(stage, path, "inputs:intensity", float(i0 * gain))
            col0 = np.asarray(col0 if col0 is not None else (1.0, 1.0, 1.0), float)
            self._set(stage, path, "inputs:color", Gf.Vec3f(*map(float, col0 * kelvin_gain(kel))),
                      Sdf.ValueTypeNames.Color3f)
        # key direction: nominal, tilted about a random horizontal axis, then turned about world z
        tilt = rng.uniform(0, c["key_tilt_deg"])
        tax = rng.uniform(0, 2 * math.pi)
        az = rng.uniform(-c["key_azimuth_deg"], c["key_azimuth_deg"])
        R = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(math.cos(tax), math.sin(tax), 0), tilt)) * \
            Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), az))
        nom = self._nom_xf_of(stage, key)
        t = nom.ExtractTranslation()
        rot = Gf.Matrix4d(nom).SetTranslateOnly(Gf.Vec3d(0, 0, 0))
        self._set_xf(stage, key, rot * R * Gf.Matrix4d().SetTranslate(t))
        ang = rng.uniform(*c["key_angle_deg"])
        self._set(stage, key, "inputs:angle", float(ang))
        out.update(key_tilt_deg=_r(tilt, 2), key_azimuth_deg=_r(az, 2), key_angle_deg=_r(ang, 2))
        return out

    def _nom_xf_of(self, stage, path):
        from pxr import Usd, UsdGeom

        if path in self._nom_xf:
            return self._nom_xf[path]
        return UsdGeom.Xformable(stage.GetPrimAtPath(path)).GetLocalTransformation(Usd.TimeCode.Default())

    def _restore_attr(self, stage, path, name):
        v = self._nom_attr.get((path, name))
        a = stage.GetPrimAtPath(path).GetAttribute(name)
        if a:
            a.Clear() if v is None else a.Set(v)

    # ---------------------------------------------------------------- 3. materials / textures
    def _materials(self, env, stage, envs, rng) -> dict:
        from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

        from sim.twin.visuals.twin_scene import BRICK_RGB

        c = self.cfg["material"]
        out: dict = {"bricks": [], "plates": []}
        palette = dict(getattr(getattr(env, "look", None), "brick_rgb", None) or BRICK_RGB)
        for e in envs:
            root = stage.GetPrimAtPath(f"/World/envs/env_{e}/TwinVis")
            if not root:
                continue
            bricks, plates = {}, {}
            for ch in root.GetChildren():
                n = ch.GetName()
                if n.startswith("Bowl_") and stage.GetPrimAtPath(f"{ch.GetPath()}/Dish"):
                    plates[n[5:]] = f"{ch.GetPath()}/Dish"
                elif stage.GetPrimAtPath(f"{ch.GetPath()}/Duplo"):
                    bricks[n] = f"{ch.GetPath()}/Duplo"
            # brick palette classes as the env bound them (Brick_<class>), or as we last drew them
            cls = {}
            for n, mp in bricks.items():
                bound = str(UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(mp)).GetDirectBinding().GetMaterialPath())
                if bound.startswith(LOOKS + "/Brick_"):
                    cls[n] = bound.rsplit("/Brick_", 1)[1]
                    self._nom_bind[mp] = bound  # restore -> the env's latest palette choice
                else:
                    cls[n] = self._brick_class.get((e, n), "yellow")
            tgt = cls.get("Brick")
            eb = {}
            for n, mp in bricks.items():
                k = cls[n]
                if (n == "Brick" and c["recolor_target"]) or (n != "Brick" and c["recolor_distractors"]):
                    opts = [p for p in palette if p != tgt or n == "Brick"]
                    k = opts[int(rng.integers(len(opts)))]
                self._brick_class[(e, n)] = k
                rgb = jitter_srgb(palette.get(k, (200, 200, 200)), rng, c["brick_hue_deg"], c["brick_sat"],
                                  c["brick_val"])
                path = f"{DR_LOOKS}/env_{e}/Brick_{n}"
                self._material(stage, path, rgb, rng.uniform(*c["brick_roughness"]), clearcoat=0.3)
                self._bind(stage, mp, path)
                eb[n] = k
            out["bricks"].append(eb)
            ep = {}
            for name, mp in plates.items():
                src = self._nom_bind.get(mp) or str(
                    UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(mp)).GetDirectBinding().GetMaterialPath())
                col = stage.GetPrimAtPath(src + "/Shader").GetAttribute("inputs:diffuseColor").Get() \
                    if stage.GetPrimAtPath(src + "/Shader") else None
                rgb0 = lin_to_srgb(np.asarray(col if col is not None else (0.5, 0.5, 0.5), float)) * 255.0
                rgb = jitter_srgb(rgb0, rng, c["plate_hue_deg"], c["plate_sat"], c["plate_val"])
                path = f"{DR_LOOKS}/env_{e}/Plate_{name}"
                self._material(stage, path, rgb, rng.uniform(*c["plate_roughness"]), clearcoat=0.4)
                self._bind(stage, mp, path)
                ep[name] = _r(rgb, 1)
            out["plates"].append(ep)
        # table: texture variant (one material per variant, switched by rebinding: changing a material's texture
        # file and back leaves RTX on a lower mip of the original -> restore residue), brightness / tint, uv shift,
        # roughness
        tex = self._tex()
        var = _choice(rng, c["table_variants"], {"original"} | set(tex.get("table", {})))
        mat = self._variant(stage, "TableTop", var, tex.get("table", {}).get(var), envs, "TableTopVisual")
        g = rng.uniform(*c["table_gain"])
        tint = 1.0 + rng.uniform(-c["table_tint"], c["table_tint"], 3)
        tp = f"{mat}/Tex"
        s0 = self._get_nom(stage, f"{LOOKS}/TableTop/Tex", "inputs:scale")
        s0 = np.asarray(s0 if s0 is not None else (1, 1, 1, 1), float)
        self._set(stage, tp, "inputs:scale", Gf.Vec4f(*map(float, s0[:3] * g * tint), float(s0[3])))
        wp = f"{LOOKS}/TableWood/Shader"
        w0 = self._get_nom(stage, wp, "inputs:diffuseColor")
        if w0 is not None:
            self._set(stage, wp, "inputs:diffuseColor", Gf.Vec3f(*map(float, np.asarray(w0) * g * tint)))
        rough = rng.uniform(*c["table_roughness"])
        self._set(stage, f"{mat}/Shader", "inputs:roughness", float(rough))
        shift = rng.uniform(-c["table_shift_m"], c["table_shift_m"], 2)
        ext = json.loads((VIS / "textures/table_top.json").read_text())
        for e in envs:
            mp = f"/World/envs/env_{e}/TableTopVisual"
            prim = stage.GetPrimAtPath(mp)
            if not prim:
                continue
            st0 = self._get_nom(stage, mp, "primvars:st")
            if st0 is None:
                continue
            st = np.asarray(st0, np.float32) + np.array([shift[0] / ext["width"], shift[1] / ext["height"]], np.float32)
            self._set(stage, mp, "primvars:st", Vt.Vec2fArray.FromNumpy(st))
        out["table"] = {"variant": var, "gain": _r(g), "tint": _r(tint), "roughness": _r(rough),
                        "shift_m": _r(shift)}
        # robot body / jaws (shared materials)
        bv = rng.uniform(*c["robot_body_val"])
        br = rng.uniform(*c["robot_roughness"])
        for m in ("RobotBody", "RobotServo"):
            sp = f"{LOOKS}/{m}/Shader"
            c0 = self._get_nom(stage, sp, "inputs:diffuseColor")
            if c0 is not None:
                rgb = np.clip(lin_to_srgb(np.asarray(c0, float)) * bv, 0, 1)
                self._set(stage, sp, "inputs:diffuseColor", Gf.Vec3f(*map(float, srgb_to_lin(rgb))))
                self._set(stage, sp, "inputs:roughness", float(br))
        jp = f"{LOOKS}/RobotJaw/Shader"
        j0 = self._get_nom(stage, jp, "inputs:diffuseColor")
        if j0 is not None:
            rgb = jitter_srgb(lin_to_srgb(np.asarray(j0, float)) * 255, rng, c["jaw_hue_deg"], (1, 1), c["jaw_val"])
            self._set(stage, jp, "inputs:diffuseColor", Gf.Vec3f(*map(float, srgb_to_lin(np.asarray(rgb) / 255))))
        out["robot"] = {"body_val": _r(bv), "roughness": _r(br)}
        # inspect backdrop: texture variant + gain
        if stage.GetPrimAtPath(f"{LOOKS}/Backdrop/Tex"):
            bvar = _choice(rng, c["backdrop_variants"], {"original"} | set(tex.get("backdrop", {})))
            mat = self._variant(stage, "Backdrop", bvar, tex.get("backdrop", {}).get(bvar), envs, "BackdropVisual")
            bg = rng.uniform(*c["backdrop_gain"])
            s0 = self._get_nom(stage, f"{LOOKS}/Backdrop/Tex", "inputs:scale")
            s0 = np.asarray(s0 if s0 is not None else (1, 1, 1, 1), float)
            self._set(stage, f"{mat}/Tex", "inputs:scale", Gf.Vec4f(*map(float, s0[:3] * bg), float(s0[3])))
            out["backdrop"] = {"variant": bvar, "gain": _r(bg)}
        return out

    def _variant(self, stage, kind: str, var: str, file: str | None, envs, mesh: str) -> str:
        """Bind texture variant `var` of the twin's `kind` material (TableTop / Backdrop) to every env's `mesh`;
        'original' = the twin's own material. Variant materials are created once (a copy of the original's shader
        settings with another texture file) and never change file, so no texture is ever reloaded."""
        from sim.twin.visuals.materials import preview_material

        nom = f"{LOOKS}/{kind}"
        path = nom if var == "original" or not file else f"{DR_LOOKS}/{kind}_{var}"
        if path != nom and not stage.GetPrimAtPath(path):
            sh = stage.GetPrimAtPath(f"{nom}/Shader")
            tx = stage.GetPrimAtPath(f"{nom}/Tex")
            sc = tx.GetAttribute("inputs:scale").Get() if tx else None
            col = tuple(float(v) for v in (sc if sc is not None else (1, 1, 1, 1)))[:3]
            spec = sh.GetAttribute("inputs:specular").Get() if sh and sh.GetAttribute("inputs:specular") else None
            preview_material(stage, path, col, float(sh.GetAttribute("inputs:roughness").Get()), linear=True,
                             texture=file, specular=0.5 if spec is None else float(spec))
        for e in envs:
            self._bind(stage, f"/World/envs/env_{e}/{mesh}", path)
        return path

    def _tex(self) -> dict:
        if self._textures is None:
            try:
                self._textures = build_textures()
            except Exception as ex:  # photos / cv2 missing: original textures only
                print(f"[TwinRandomizer] photo texture variants unavailable: {ex}", flush=True)
                self._textures = {}
        return self._textures

    # ---------------------------------------------------------------- 4. clutter (visual only)
    def _clutter(self, env, stage, envs, rng) -> dict:
        from pxr import Gf, UsdGeom

        from sim.twin.visuals.twin_scene import _rscene

        c = self.cfg["clutter"]
        z0 = _rscene().TABLE_TOP_Z
        top = getattr(env.profile, "top", None)
        objs = self._object_xy(env)
        out = []
        for e in envs:
            root = f"/World/envs/env_{e}/TwinDR"
            if not stage.GetPrimAtPath(root):
                self._build_clutter(stage, e, c)
            UsdGeom.Imageable(stage.GetPrimAtPath(root)).MakeVisible()
            rec = {}
            cab = stage.GetPrimAtPath(f"{root}/Cable")
            pts = self._cable_path(rng, c, top, objs.get(e, []))
            if pts is not None and rng.uniform() < c["p_cable"]:
                UsdGeom.Imageable(cab).MakeVisible()
                r = c["cable_radius"]
                for k in range(c["cable_segments"]):
                    a, b = pts[k], pts[k + 1]
                    seg = stage.GetPrimAtPath(f"{root}/Cable/s{k:02d}")
                    d = b - a
                    L = float(np.linalg.norm(d))
                    cap = UsdGeom.Capsule(seg)
                    cap.GetHeightAttr().Set(L)
                    cap.GetExtentAttr().Set([Gf.Vec3f(-r, -r, -L / 2 - r), Gf.Vec3f(r, r, L / 2 + r)])
                    rot = Gf.Rotation(Gf.Vec3d(0, 0, 1), Gf.Vec3d(*map(float, d / max(L, 1e-9))))
                    m = Gf.Matrix4d().SetRotate(rot) * Gf.Matrix4d().SetTranslate(
                        Gf.Vec3d(*map(float, (a + b) / 2)) + Gf.Vec3d(0, 0, z0 + r))
                    _set_xf_plain(seg, m)
                rec["cable"] = _r(pts[[0, -1]], 3)
            else:
                UsdGeom.Imageable(cab).MakeInvisible()
            blob = stage.GetPrimAtPath(f"{root}/Blob")
            xy = self._free_xy(rng, top, objs.get(e, []), 0.06)
            if xy is not None and rng.uniform() < c["p_blob"]:
                UsdGeom.Imageable(blob).MakeVisible()
                sz = rng.uniform(*c["blob_size"])
                yaw = rng.uniform(0, 360)
                m = Gf.Matrix4d().SetScale(Gf.Vec3d(*map(float, sz))) * \
                    Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw)) * \
                    Gf.Matrix4d().SetTranslate(Gf.Vec3d(float(xy[0]), float(xy[1]), z0 + float(sz[2]) * 0.6))
                _set_xf_plain(blob, m)
                col = [(18, 18, 20), (45, 38, 34), (40, 48, 70), (70, 70, 72)][int(rng.integers(4))]
                self._material(stage, f"{DR_LOOKS}/env_{e}/Blob", col, rng.uniform(0.4, 0.8))
                rec["blob"] = {"xy": _r(xy, 3), "size": _r(sz, 3), "rgb": col}
            else:
                UsdGeom.Imageable(blob).MakeInvisible()
            out.append(rec)
        return out

    def _build_clutter(self, stage, e, c):
        from pxr import Gf, UsdGeom

        from sim.twin.visuals.materials import bind

        root = f"/World/envs/env_{e}/TwinDR"
        UsdGeom.Xform.Define(stage, root)
        UsdGeom.Xform.Define(stage, f"{root}/Cable")
        self._material(stage, f"{DR_LOOKS}/env_{e}/Cable", (22, 22, 24), 0.45)
        mat = _usd_material(stage, f"{DR_LOOKS}/env_{e}/Cable")
        for k in range(c["cable_segments"]):
            cap = UsdGeom.Capsule.Define(stage, f"{root}/Cable/s{k:02d}")
            cap.CreateAxisAttr("Z")
            cap.CreateRadiusAttr(float(c["cable_radius"]))
            cap.CreateHeightAttr(0.03)
            cap.CreateExtentAttr([Gf.Vec3f(-0.02, -0.02, -0.02), Gf.Vec3f(0.02, 0.02, 0.02)])
            bind(cap.GetPrim(), mat)
        sph = UsdGeom.Sphere.Define(stage, f"{root}/Blob")
        sph.CreateRadiusAttr(1.0)
        sph.CreateExtentAttr([Gf.Vec3f(-1, -1, -1), Gf.Vec3f(1, 1, 1)])
        self._material(stage, f"{DR_LOOKS}/env_{e}/Blob", (20, 20, 22), 0.6)
        bind(sph.GetPrim(), _usd_material(stage, f"{DR_LOOKS}/env_{e}/Blob"))

    def _object_xy(self, env) -> dict:
        """env -> [(x, y, clearance)] of plates, bricks (env frame) and the robot base."""
        cl = self.cfg["clutter"]["clearance"]
        sc = env.scene
        org = sc.env_origins.detach().cpu().numpy()
        out = {e: [(0.0, 0.0, cl["robot"])] for e in range(sc.num_envs)}
        for k, o in sc.rigid_objects.items():
            p = o.data.root_pos_w.detach().cpu().numpy() - org
            r = cl["bowl"] if k.startswith("bowl") else cl["brick"]
            for e in range(sc.num_envs):
                if p[e, 2] > -0.5:  # parked objects sit far below the table
                    out[e].append((float(p[e, 0]), float(p[e, 1]), r))
        return out

    @staticmethod
    def _visible(top, xy, margin=24) -> bool:
        if top is None:
            return True
        rows = np.asarray(top.rows, float)
        pc = (np.array([xy[0], xy[1], -0.002]) - rows[3, :3]) @ rows[:3, :3].T
        if pc[2] >= -1e-3:
            return False
        f = top.intr["focal_length"] / top.intr["horizontal_aperture"] * 640.0
        u, v = 320 + f * pc[0] / -pc[2], 240 - f * pc[1] / -pc[2]
        return margin < u < 640 - margin and margin < v < 480 - margin

    @staticmethod
    def _clear(xy, objs, extra=0.0) -> bool:
        return all(math.hypot(xy[0] - x, xy[1] - y) > r + extra for x, y, r in objs)

    def _free_xy(self, rng, top, objs, extra):
        for _ in range(60):
            xy = rng.uniform([-0.05, -0.4], [0.7, 0.4])
            if self._visible(top, xy) and self._clear(xy, objs, extra):
                return xy
        return None

    def _cable_path(self, rng, c, top, objs):
        n = c["cable_segments"]
        for _ in range(40):
            a = self._free_xy(rng, top, objs, 0.0)
            if a is None:
                return None
            th = rng.uniform(0, 2 * math.pi)
            L = rng.uniform(*c["cable_len"])
            d = np.array([math.cos(th), math.sin(th)])
            b = a + d * L
            ctrl = (a + b) / 2 + np.array([-d[1], d[0]]) * rng.uniform(-0.4, 0.4) * L
            t = np.linspace(0, 1, n + 1)[:, None]
            p = (1 - t) ** 2 * a + 2 * (1 - t) * t * ctrl + t ** 2 * b
            if all(self._clear(q, objs, 0.0) for q in p):
                return np.c_[p, np.zeros(n + 1)]
        return None

    # ---------------------------------------------------------------- 5. image-space parameters (per call)
    def _image(self, rng) -> dict:
        out = {}
        for cam in CAMS:
            p = self.cfg["image"].get(cam)
            if not p:
                continue
            d = {"gain": rng.uniform(*p["gain"]), "contrast": rng.uniform(*p["contrast"]),
                 "wb": (1.0 + rng.uniform(-p["wb"], p["wb"], 3)).tolist(),
                 "blur_sigma": rng.uniform(*p["blur_sigma"]) if rng.uniform() < p["p_blur"] else 0.0,
                 "motion_len": int(rng.integers(p["motion_len"][0], p["motion_len"][1] + 1))
                 if rng.uniform() < p["p_motion"] else 0,
                 "motion_deg": rng.uniform(0, 180), "noise": rng.uniform(*p["noise"]),
                 "jpeg_q": int(rng.integers(*p["jpeg_q"])) if rng.uniform() < p["p_jpeg"] else 0,
                 "seed": int(rng.integers(2 ** 31))}
            d["wb"] = [v / float(np.mean(d["wb"])) for v in d["wb"]]  # white balance only, no brightness
            out[cam] = _r(d)
        return out


# ------------------------------------------------------------------------------------------------ module helpers
def _usd_material(stage, path):
    from pxr import UsdShade

    return UsdShade.Material(stage.GetPrimAtPath(path))


def _set_xf_plain(prim, m):
    """Our own clutter prims: one full transform op (keeps scale)."""
    from pxr import UsdGeom

    xf = UsdGeom.Xformable(prim)
    ops = xf.GetOrderedXformOps()
    if len(ops) == 1 and ops[0].GetOpType() == UsdGeom.XformOp.TypeTransform:
        ops[0].Set(m)
    else:
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(m)


def _find_sync(env):
    """The env's VisualSync (env.sync / env.visual_sync / env.visuals['sync'])."""
    for name in ("sync", "visual_sync", "vsync"):
        s = getattr(env, name, None)
        if s is not None and hasattr(s, "cams"):
            return s
    v = getattr(env, "visuals", None)
    if isinstance(v, dict) and hasattr(v.get("sync"), "cams"):
        return v["sync"]
    return None


def postprocess(img, cam: str, draw: dict | None, rng=None):
    """Image-space DR for one rendered frame of camera `cam` ((H, W, 3|4) or (N, H, W, 3|4) uint8, numpy or torch),
    with the parameters drawn on reset (draw['image'][cam]). Apply after twin_scene.color_correct. Returns the same
    type (numpy uint8 / torch uint8 on the input device), 3 channels. Noise is fresh per call (rng or a generator
    seeded from the draw)."""
    p = (draw or {}).get("image", {}).get(cam)
    if not p:
        return img
    import torch
    import torch.nn.functional as F

    is_np = not isinstance(img, torch.Tensor)
    x = torch.from_numpy(np.ascontiguousarray(img)) if is_np else img
    single = x.dim() == 3
    x = (x[None] if single else x)[..., :3].float().div(255.0).permute(0, 3, 1, 2)  # N 3 H W
    dev = x.device
    wb = torch.tensor(p["wb"], device=dev, dtype=torch.float32).view(1, 3, 1, 1)
    x = x * wb * float(p["gain"])
    x = (x - 0.5) * float(p["contrast"]) + 0.5
    s = float(p["blur_sigma"])
    if s > 0.05:
        r = max(1, int(math.ceil(3 * s)))
        t = torch.arange(-r, r + 1, device=dev, dtype=torch.float32)
        k = torch.exp(-0.5 * (t / s) ** 2)
        k = k / k.sum()
        x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).repeat(3, 1, 1, 1), groups=3)
        x = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).repeat(3, 1, 1, 1), groups=3)
    L = int(p.get("motion_len", 0))
    if L >= 2:
        n = L if L % 2 else L + 1
        k = np.zeros((n, n), np.float32)
        a = math.radians(p["motion_deg"])
        for tt in np.linspace(-(L - 1) / 2, (L - 1) / 2, 4 * L):
            k[int(round(n // 2 + tt * math.sin(a))), int(round(n // 2 + tt * math.cos(a)))] = 1.0
        k /= k.sum()
        kk = torch.from_numpy(k).to(dev).view(1, 1, n, n).repeat(3, 1, 1, 1)
        x = F.conv2d(F.pad(x, (n // 2,) * 4, mode="reflect"), kk, groups=3)
    if rng is None:
        rng = np.random.default_rng(int(p.get("seed", 0)) + int(time.perf_counter_ns() % 1_000_003))
    sig = float(p["noise"]) / 255.0
    if sig > 0:
        g = torch.Generator(device=dev).manual_seed(int(rng.integers(2 ** 31)))
        x = x + torch.randn(x.shape, generator=g, device=dev) * sig
    x = (x.clamp(0, 1) * 255.0 + 0.5).to(torch.uint8).permute(0, 2, 3, 1)
    q = int(p.get("jpeg_q", 0))
    if q > 0:
        import cv2

        arr = x.cpu().numpy()
        for i in range(len(arr)):
            ok, buf = cv2.imencode(".jpg", arr[i][..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])
            if ok:
                arr[i] = cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1]
        x = torch.from_numpy(arr).to(dev)
    x = x[0] if single else x
    return x.cpu().numpy() if is_np else x


# ------------------------------------------------------------------------------------------------ photo textures
def _lab(img_rgb):
    import cv2

    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)


def _match_lab(img_rgb, mask_src, ref_rgb, mask_ref):
    """Reinhard colour transfer in Lab: src pixels (mask) get the reference pixels' (mask) mean / std."""
    import cv2

    a, b = _lab(img_rgb), _lab(ref_rgb)
    ma, sa = a[mask_src].mean(0), a[mask_src].std(0) + 1e-3
    mb, sb = b[mask_ref].mean(0), b[mask_ref].std(0) + 1e-3
    out = (a - ma) * np.clip(sb / sa, 0.7, 1.3) + mb
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def _patch_fill(img, hole, block=64, rng=None):
    """Fill `hole` pixels with wood copied from hole-free places of the same image: overlapping source blocks
    (stride block / 2) blended with a raised-cosine window, so the copies have no hard seams."""
    import cv2

    rng = rng or np.random.default_rng(0)
    H, W = hole.shape
    ii = cv2.integral(hole.astype(np.uint8))
    num = np.zeros((H, W, 3), np.float32)
    den = np.zeros((H, W), np.float32)
    win = np.outer(np.hanning(block + 2)[1:-1], np.hanning(block + 2)[1:-1]).astype(np.float32) + 1e-4
    src = img.astype(np.float32)
    for y in range(-block // 2, H, block // 2):
        for x in range(-block // 2, W, block // 2):
            ya, xa, yb, xb = max(0, y), max(0, x), min(H, y + block), min(W, x + block)
            if yb <= ya or xb <= xa or not hole[ya:yb, xa:xb].any():
                continue
            for _ in range(300):
                sy, sx = int(rng.integers(0, H - block)), int(rng.integers(0, W - block))
                if ii[sy + block, sx + block] - ii[sy, sx + block] - ii[sy + block, sx] + ii[sy, sx] == 0:
                    w = win[ya - y:yb - y, xa - x:xb - x]
                    num[ya:yb, xa:xb] += src[sy + ya - y:sy + yb - y, sx + xa - x:sx + xb - x] * w[..., None]
                    den[ya:yb, xa:xb] += w
                    break
    filled = np.where(den[..., None] > 0, num / np.maximum(den, 1e-6)[..., None], src)
    soft = cv2.GaussianBlur(hole.astype(np.float32), (0, 0), 3)[..., None]
    return (soft * filled + (1 - soft) * src).clip(0, 255).astype(np.uint8)


def _photo_wood(path: Path, ref_rgb, ref_mask, debug: dict):
    """Top-down phone photo -> wood-only image (RGB), metres per pixel. The two fully visible plates (0.184 m) give
    the scale; the photo is cropped to the table (left 84 %, below the cable / box strip at the top), its lighting
    gradient flattened, and plates / bricks / arm / cables / tripods (colour far from the wood) patch-filled with
    wood from elsewhere in the photo; colour matched (Lab mean / std) to the wood the top camera saw."""
    import cv2

    img = cv2.imread(str(path))[..., ::-1].copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    H, W = hsv.shape[:2]
    diam, plate_mask = [], np.zeros((H, W), np.uint8)
    for lo, hi in (((0, 170, 120), (14, 255, 255)), ((20, 150, 150), (34, 255, 255)), ((115, 40, 120), (150, 255, 255))):
        m = cv2.inRange(hsv, lo, hi)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in [c for c in cs if cv2.contourArea(c) > 0.02 * H * W]:
            (cx, cy), (a, b), _ = cv2.fitEllipse(c)
            x, y, w, h = cv2.boundingRect(c)
            if x > 2 and y > 2 and x + w < W - 2 and y + h < H - 2:  # fully inside the photo: a usable scale
                diam.append(max(a, b))
            cv2.ellipse(plate_mask, ((cx, cy), (max(a, b) * 1.25, max(a, b) * 1.25), 0), 255, -1)
    m_per_px = PLATE_D / float(np.mean(diam)) if diam else PLATE_D / 520.0
    debug.update(plate_diam_px=[round(d, 1) for d in diam], m_per_px=m_per_px)
    x0, x1, y0, y1 = 0, int(0.84 * W), int(0.10 * H), H  # the table (right of 84 %: floor / tripods)
    img, hsv, plate_mask = img[y0:y1, x0:x1], hsv[y0:y1, x0:x1], plate_mask[y0:y1, x0:x1]
    lab = _lab(img)
    wood = np.median(lab[lab.shape[0] // 4:3 * lab.shape[0] // 4, lab.shape[1] // 3:2 * lab.shape[1] // 3]
                     .reshape(-1, 3), 0)
    non = (np.linalg.norm(lab - wood, axis=2) > 34) | (hsv[..., 2] < 60) | (plate_mask > 0)
    # lighting gradient: wood-only normalised blur, divided out
    f = img.astype(np.float32)
    w = (~non).astype(np.float32)
    sig = 0.08 / m_per_px
    illum = cv2.GaussianBlur(f * w[..., None], (0, 0), sig) / (cv2.GaussianBlur(w, (0, 0), sig)[..., None] + 1e-3)
    f = np.clip(f / np.maximum(illum, 1.0) * f[w > 0].mean(0), 0, 255).astype(np.uint8)
    lab = _lab(f)
    wood = np.median(lab[w > 0], 0)
    dist = np.linalg.norm(lab - wood, axis=2)
    non = (dist > 42) | (hsv[..., 2] < 45) | (plate_mask > 0)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(non.astype(np.uint8))
    small = st[:, cv2.CC_STAT_AREA] < 300  # small dark specks are wood marks (knots, scratches)
    small[0] = False
    non &= ~(small[lbl] & (dist < 55))
    non = cv2.dilate(non.astype(np.uint8), np.ones((13, 13), np.uint8)) > 0
    debug.update(crop_xyxy=[x0, y0, x1, y1], hole_frac=round(float(non.mean()), 3))
    crop = _patch_fill(f, non, block=max(32, int(0.04 / m_per_px)))
    crop = _match_lab(crop, np.ones(crop.shape[:2], bool), ref_rgb, ref_mask)
    return crop, m_per_px


def _tile_to(img, out_hw, origin=(0, 0)):
    """Mirror-tile img to cover out_hw (seamless at the joins), one unflipped copy with its corner at origin."""
    H, W = out_hw
    t = np.concatenate([img, img[:, ::-1]], 1)
    t = np.concatenate([t, t[::-1]], 0)
    ry, rx = -(-H // t.shape[0]) + 2, -(-W // t.shape[1]) + 2
    big = np.tile(t, (ry, rx, 1))
    oy, ox = (-origin[0]) % t.shape[0], (-origin[1]) % t.shape[1]
    return big[oy:oy + H, ox:ox + W]


def _blur_people(img):
    """Blur skin-coloured regions (faces, arms; YCrCb skin box, dilated) of the room photo heavily. The OpenCV build
    in the image has no Haar cascades (cv2 5), and wood / cardboard that also passes the skin test only gets blurred."""
    import cv2

    ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    m = cv2.inRange(ycc, (40, 138, 80), (240, 175, 125))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.dilate(m, np.ones((41, 41), np.uint8)).astype(np.float32) / 255.0
    m = cv2.GaussianBlur(m, (0, 0), 6)[..., None]
    blur = cv2.GaussianBlur(img, (0, 0), 9).astype(np.float32)
    return (m * blur + (1 - m) * img.astype(np.float32)).astype(np.uint8), float(m.mean())


def build_textures(force: bool = False) -> dict:
    """Photo-derived DR texture variants (git-ignored, sim/twin/out/dr/textures/); returns {kind: {variant: file}}.

    table_photo_full.jpg / table_photo_outside.jpg (same uv layout as visuals/textures/table_top.jpg, 0.5 mm/px):
        wood from the top-down phone photo (plates / bricks / arm / cables patch-filled with wood, colour matched in Lab
        to the camera-projected texture's wood), mirror-tiled. 'outside' keeps the camera-projected texture where the
        top camera actually saw the table (out/visuals/table_texture_valid.png, feathered) and fixes only the
        smeared mirror fill elsewhere (what the wrist / inspect cameras see); 'full' replaces all the wood but keeps
        non-wood fixed features of the original (dark cloth beyond the table edge, the inspection stand).
    backdrop_pano{0,1,2}.jpg: the room above the table line of the station panorama (faces blurred), colour matched
        to the median inspect backdrop, as-is / mirrored / rolled by half.
    dome_pano.jpg + gain: lat-long dome map with the panorama on the horizon band; gain normalises its upper
        hemisphere irradiance to the plain white dome's, so the exposure calibration still holds."""
    import cv2

    TEX_OUT.mkdir(parents=True, exist_ok=True)
    meta_p = TEX_OUT / "textures.json"
    if meta_p.exists() and not force:
        meta = _abs(json.loads(meta_p.read_text()))
        if all(Path(f).exists() for kind in ("table", "backdrop") for f in meta.get(kind, {}).values()):
            return meta
    meta: dict = {"table": {}, "backdrop": {}, "debug": {}}
    td, pano = PHOTOS / "table_topdown.jpg", PHOTOS / "station_panorama.jpg"
    # ---- table
    orig = cv2.imread(str(VIS / "textures/table_top.jpg"))
    if td.exists() and orig is not None:
        orig = orig[..., ::-1].copy()
        Ho, Wo = orig.shape[:2]
        ext = json.loads((VIS / "textures/table_top.json").read_text())
        vp = REPO / "sim/twin/out/visuals/table_texture_valid.png"
        valid = (cv2.imread(str(vp), cv2.IMREAD_GRAYSCALE) > 0) if vp.exists() else np.ones((Ho, Wo), bool)
        ohsv = cv2.cvtColor(orig, cv2.COLOR_RGB2HSV)
        o_wood = valid & (ohsv[..., 1] < 70) & (ohsv[..., 2] > 80)  # grey-beige wood the top camera saw
        dbg: dict = {}
        wood, mpp = _photo_wood(td, orig, o_wood, dbg)
        res = 0.0005  # m / px of the variant
        wood = cv2.resize(wood, None, fx=mpp / res, fy=mpp / res, interpolation=cv2.INTER_AREA)
        wood = np.ascontiguousarray(np.rot90(wood))  # photo's long (vertical) axis along world x (texture cols)
        Hn, Wn = int(round(ext["height"] / res)), int(round(ext["width"] / res))
        ys, xs = np.nonzero(o_wood)
        cy, cx = (int(ys.mean() * Hn / Ho), int(xs.mean() * Wn / Wo)) if len(ys) else (Hn // 2, Wn // 2)
        tiled = _tile_to(wood, (Hn, Wn), (cy - wood.shape[0] // 2, cx - wood.shape[1] // 2))
        up = cv2.resize(orig, (Wn, Hn), interpolation=cv2.INTER_CUBIC)
        # table rectangle (the physical cube, env frame): x -0.425..0.925, y -0.65..0.65 -> texture px
        tx = lambda x: int(round((x - ext["x0"]) / res))  # noqa: E731
        ty = lambda y: int(round((ext["y0"] + ext["height"] - y) / res))  # noqa: E731  row 0 = +y edge
        table = np.zeros((Hn, Wn), np.float32)
        table[ty(0.65):ty(-0.65), tx(-0.425):tx(0.925)] = 1.0
        v_up = cv2.resize(valid.astype(np.float32), (Wn, Hn), interpolation=cv2.INTER_LINEAR)
        v_soft = cv2.GaussianBlur(cv2.erode(v_up, np.ones((41, 41), np.uint8)), (0, 0), 25)
        nonwood = cv2.resize((valid & ~o_wood).astype(np.uint8) * 255, (Wn, Hn), interpolation=cv2.INTER_NEAREST)
        n, lbl, st, _ = cv2.connectedComponentsWithStats((nonwood > 0).astype(np.uint8))
        big = st[:, cv2.CC_STAT_AREA] > 4000
        big[0] = False
        nw_soft = cv2.GaussianBlur(cv2.dilate(big[lbl].astype(np.float32), np.ones((31, 31), np.uint8)), (0, 0), 8)
        table_s = cv2.GaussianBlur(table, (0, 0), 6)
        for name, a in (("photo_outside", table_s * (1 - v_soft)), ("photo_full", table_s * (1 - nw_soft))):
            out = (a[..., None] * tiled + (1 - a[..., None]) * up).astype(np.uint8)
            f = TEX_OUT / f"table_{name}.jpg"
            cv2.imwrite(str(f), out[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
            meta["table"][name] = str(f.resolve())
        cv2.imwrite(str(TEX_OUT / "debug_wood.jpg"), wood[..., ::-1])
        meta["debug"]["table"] = dbg
    # ---- backdrop + dome
    bd = cv2.imread(str(VIS / "textures/backdrop_inspect.jpg"))
    if pano.exists() and bd is not None:
        bd = bd[..., ::-1].copy()
        p = cv2.imread(str(pano))[..., ::-1].copy()
        Hp, Wp = p.shape[:2]
        band, blurred = _blur_people(p[: int(0.53 * Hp)])  # the room above the far table line
        p[: int(0.53 * Hp)] = band
        band = _match_lab(band, np.ones(band.shape[:2], bool), bd, np.ones(bd.shape[:2], bool))
        band = cv2.resize(band, (bd.shape[1], bd.shape[0]), interpolation=cv2.INTER_AREA)
        for k, v in enumerate((band, band[:, ::-1], np.roll(band, band.shape[1] // 2, axis=1))):
            f = TEX_OUT / f"backdrop_pano{k}.jpg"
            cv2.imwrite(str(f), np.ascontiguousarray(v)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
            meta["backdrop"][f"pano{k}"] = str(f.resolve())
        # lat-long dome: 2048 x 1024; panorama (assumed ~200 deg wide, ~45 deg tall) centred on the horizon
        LW, LH = 2048, 1024
        pw = int(LW * 200 / 360)
        ph = int(pw * Hp / Wp)
        pr = cv2.resize(p, (pw, ph), interpolation=cv2.INTER_AREA)
        sky = np.median(p[: Hp // 8].reshape(-1, 3), 0)
        sky = 0.5 * sky + 0.5 * sky.mean()  # desaturated ceiling
        floor = np.array([38.0, 38.0, 42.0])  # dark floor: the wrist camera sees it past the table quad's edge
        dome = np.zeros((LH, LW, 3), np.float32)
        hz = LH // 2
        dome[:hz] = sky
        dome[hz:] = floor
        y0 = hz - int(ph * 0.55)
        x0 = (LW - pw) // 2
        dome[y0:y0 + ph, x0:x0 + pw] = pr
        dome = cv2.GaussianBlur(dome, (0, 0), 1.5).clip(0, 255).astype(np.uint8)
        f = TEX_OUT / "dome_pano.jpg"
        cv2.imwrite(str(f), dome[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
        lin = srgb_to_lin(dome[:hz].astype(np.float32) / 255.0) @ np.array([0.2126, 0.7152, 0.0722])
        el = (0.5 - (np.arange(hz) + 0.5) / LH) * math.pi  # elevation of each upper row
        w = (np.sin(el) * np.cos(el))[:, None]  # cosine-weighted, solid angle ~ cos(el)
        irr = float((lin * w).sum() / (w.sum() * lin.shape[1]))
        meta["dome"] = {"file": str(f.resolve()), "gain": round(1.0 / max(irr, 1e-3), 4)}
        meta["debug"]["skin_blur_frac"] = round(blurred, 3)
    meta_p.write_text(json.dumps(_rel(meta), indent=1))  # repo-relative: valid on the host and in the container
    return meta


def _rel(meta):
    r = str(REPO.resolve()) + "/"
    return json.loads(json.dumps(meta).replace(r, ""))


def _abs(meta):
    out = copy.deepcopy(meta)
    for kind in ("table", "backdrop"):
        out[kind] = {k: str(REPO / v) for k, v in out.get(kind, {}).items()}
    if "dome" in out:
        out["dome"]["file"] = str(REPO / out["dome"]["file"])
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--build-textures", action="store_true")
    a = ap.parse_args()
    if a.build_textures:
        print(json.dumps(build_textures(force=True), indent=1))
