"""Light-basis renders of the twin at the recorded states of the visual evaluation set (Isaac).

    NAME=dlsj-twin-light sim/twin/run.sh sim/twin/lighting/basis_render.py [--eps 2 49 ...] [--angle 8] [--passes 2]

The renderer is linear in the light intensities (linear tonemap, sim/twin/visuals/twin_scene.set_render_settings), so
an image under any rig of distant lights + dome is a non-negative combination of images rendered with one light at a
time. Per evaluation sample (posed by visuals/fit/harness.Harness: recorded joint state, plates and bricks where the
real frame-0 shows them) this renders
  zero    every light off (what the unlit hall cylinder / backdrop emit by themselves),
  dome    the untextured dome alone,
  D<k>    one distant light from each direction of a hemisphere grid (elevation x azimuth, `--angle` deg soft),
  S<k>    one small sphere light at `--side-dist` m from the workspace centre at low elevations (side lamps; a distant
          light that low would be blocked by the hall cylinder),
with the nominal rig's lights and the RTX flat ambient term (/rtx/sceneDb/ambientLightIntensity) switched off.
lighting/fit_rig.py fits the real frames with these.
Writes sim/twin/out/lighting/basis_<tag>.npz: top (S, K, 240, 320, 3), inspect (S, K, 120, 160, 3) uint8 raw renders,
ids (S,), names (K,), dirs (K, 3) (unit vectors TO the light, from CENTER for S<k>; dome / zero = 0), intensity (K,).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
OUT = REPO / "sim/twin/out/lighting"
FIT_EPS = (2, 49, 95, 142, 189, 235, 282, 329)
HELD_EPS = (33, 126, 220, 313)


CENTER = np.array([0.30, 0.0, 0.0])  # workspace centre (env frame, table top ~ z 0)


def grid(els=(45.0, 60.0, 75.0), n_az: int = 12, zenith: bool = True) -> list[np.ndarray]:
    """Unit directions TO the light: n_az azimuths per elevation (staggered by half a step per ring) + the zenith."""
    out = []
    for r, el in enumerate(els):
        for k in range(n_az):
            az = math.radians((k + 0.5 * (r % 2)) * 360.0 / n_az)
            e = math.radians(el)
            out.append(np.array([math.cos(e) * math.cos(az), math.cos(e) * math.sin(az), math.sin(e)]))
    if zenith:
        out.append(np.array([0.0, 0.0, 1.0]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=int, nargs="*", default=list(FIT_EPS + HELD_EPS))
    ap.add_argument("--angle", type=float, default=8.0, help="angular diameter of the basis distant lights (deg)")
    ap.add_argument("--intensity", type=float, default=600.0, help="basis distant-light intensity (before exposure)")
    ap.add_argument("--side-dist", type=float, default=1.8)
    ap.add_argument("--side-radius", type=float, default=0.1)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args()

    from sim.twin.visuals.fit.harness import Harness

    h = Harness(passes=args.passes)
    import carb
    import cv2
    from pxr import Gf, UsdGeom, UsdLux

    from sim.twin.lighting.rig import quat_z_to

    carb.settings.get_settings().set("/rtx/sceneDb/ambientLightIntensity", 0.0)
    stage = h.env.stage
    s = h.env.look.scale()
    nominal = [p for p in stage.Traverse() if p.GetTypeName().endswith("Light")]
    dome = next(p for p in nominal if p.GetTypeName() == "DomeLight")
    dome_i = float(dome.GetAttribute("inputs:intensity").Get())
    for p in nominal:
        p.GetAttribute("inputs:intensity").Set(0.0)
    dirs = grid()
    lights = []
    for k, d in enumerate(dirs):
        lt = UsdLux.DistantLight.Define(stage, f"/World/LightBasis/D{k:02d}")
        lt.CreateAngleAttr(args.angle)
        lt.CreateIntensityAttr(0.0)
        lt.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        q = quat_z_to(d)
        UsdGeom.Xformable(lt.GetPrim()).AddOrientOp().Set(Gf.Quatf(q[3], q[0], q[1], q[2]))
        lights.append(lt.GetPrim())
    side = grid(els=(15.0, 30.0), n_az=8, zenith=False)
    e_side = args.intensity * s * args.side_dist ** 2 / (math.pi * args.side_radius ** 2)  # same irradiance at CENTER
    for k, d in enumerate(side):
        lt = UsdLux.SphereLight.Define(stage, f"/World/LightBasis/S{k:02d}")
        lt.CreateRadiusAttr(args.side_radius)
        lt.CreateIntensityAttr(0.0)
        lt.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        p = CENTER + args.side_dist * d
        UsdGeom.Xformable(lt.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*map(float, p)))
        lights.append(lt.GetPrim())
    names = ["zero", "dome"] + [f"D{k:02d}" for k in range(len(dirs))] + [f"S{k:02d}" for k in range(len(side))]
    inten = [0.0, dome_i] + [args.intensity * s] * len(dirs) + [e_side] * len(side)
    all_dirs = np.array([[0, 0, 0], [0, 0, 0]] + [list(d) for d in dirs + side], float)

    def only(k: int):
        dome.GetAttribute("inputs:intensity").Set(dome_i if k == 1 else 0.0)
        for j, p in enumerate(lights):
            p.GetAttribute("inputs:intensity").Set(inten[j + 2] if k == j + 2 else 0.0)

    ids = [i for i, smp in enumerate(h.samples) if smp["ep"] in set(args.eps)]
    top = np.zeros((len(ids), len(names), 240, 320, 3), np.uint8)
    ins = np.zeros((len(ids), len(names), 120, 160, 3), np.uint8)
    t0 = time.time()
    drift = []
    for n, i in enumerate(ids):
        h.pose(h.samples[i])
        only(0)
        h.render()
        for k in range(len(names)):
            only(k)
            img = h.render()
            top[n, k] = cv2.resize(img["top"], (320, 240), interpolation=cv2.INTER_AREA)
            ins[n, k] = cv2.resize(img["inspect"], (160, 120), interpolation=cv2.INTER_AREA)
        if n % 10 == 0:  # temporal history check: re-render the dome after the last basis light
            only(1)
            again = cv2.resize(h.render()["top"], (320, 240), interpolation=cv2.INTER_AREA)
            drift.append(float(np.abs(again.astype(float) - top[n, 1]).mean()))
            print(f"[basis] {n + 1}/{len(ids)} {time.time() - t0:.0f} s; dome re-render |diff| {drift[-1]:.2f}",
                  flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"basis_{args.tag}.npz", top=top, inspect=ins, ids=np.array(ids), names=np.array(names),
                        dirs=all_dirs, intensity=np.array(inten), exposure_scale=s)
    (OUT / f"basis_{args.tag}.json").write_text(json.dumps(dict(
        eps=args.eps, angle_deg=args.angle, side_dist=args.side_dist, side_radius=args.side_radius,
        center=CENTER.tolist(), passes=args.passes, n_samples=len(ids), names=names,
        intensity=inten, exposure_scale=s, history_drift_mean_abs=drift, seconds=round(time.time() - t0)), indent=1))
    print("BASIS_OK", args.tag, len(ids), "samples", round(time.time() - t0), "s", flush=True)
    h.close()


if __name__ == "__main__":
    main()
