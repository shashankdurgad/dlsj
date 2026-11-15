"""Measure how RTX scales the USD light types against a distant light of the same nominal irradiance (Isaac).

    NAME=dlsj-twin-light sim/twin/run.sh sim/twin/lighting/calib_units.py

Every light is set up to give E = 400 on the table at the workspace centre by the formulas in rig.py
(rect_radiance / sphere_radiance, normalize off). Renders the top camera with only that light (RTX ambient off) and
reports the mean linear luminance of a table patch under the workspace centre, minus the all-off image, relative to
the distant light. rig.py divides its radiances by these factors (RTX_UNIT). Writes sim/twin/out/lighting/units.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
OUT = REPO / "sim/twin/out/lighting"


def main():
    import carb

    from sim.twin.visuals.fit.harness import Harness

    h = Harness(passes=3)
    from pxr import Gf, UsdGeom, UsdLux

    from sim.twin.lighting import rig

    carb.settings.get_settings().set(rig.AMBIENT_KEY, 0.0)
    st = h.env.stage
    for p in st.Traverse():
        if p.GetTypeName().endswith("Light"):
            p.GetAttribute("inputs:intensity").Set(0.0)
    h.pose(h.samples[0])
    E = 400.0
    c = rig.CENTER

    def put(prim, pos, q=(0.0, 0.0, 0.0, 1.0)):
        rig._pose(st, str(prim.GetPath()), pos, q)

    lights = {}
    d = UsdLux.DistantLight.Define(st, "/World/Calib/Distant")
    d.CreateAngleAttr(5.0)
    lights["distant"] = (d, E)
    s = UsdLux.SphereLight.Define(st, "/World/Calib/Sphere")
    s.CreateRadiusAttr(0.1)
    put(s.GetPrim(), c + [0, 0, 1.0])
    lights["sphere_r0.1_d1"] = (s, rig.sphere_radiance(E, 1.0, 0.1))
    s2 = UsdLux.SphereLight.Define(st, "/World/Calib/Sphere2")
    s2.CreateRadiusAttr(0.03)
    put(s2.GetPrim(), c + [0, 0, 2.0])
    lights["sphere_r0.03_d2"] = (s2, rig.sphere_radiance(E, 2.0, 0.03))
    sp = UsdLux.SphereLight.Define(st, "/World/Calib/Spot")
    sp.CreateRadiusAttr(0.05)
    sh = UsdLux.ShapingAPI.Apply(sp.GetPrim())
    sh.CreateShapingConeAngleAttr(40.0)
    sh.CreateShapingConeSoftnessAttr(0.2)
    put(sp.GetPrim(), c + [0, 0, 1.0])
    lights["spot_r0.05_d1_cone40"] = (sp, rig.sphere_radiance(E, 1.0, 0.05))
    r = UsdLux.RectLight.Define(st, "/World/Calib/Rect")
    r.CreateWidthAttr(1.2)
    r.CreateHeightAttr(0.15)
    put(r.GetPrim(), c + [0, 0, 3.0])
    lights["rect_1.2x0.15_d3"] = (r, rig.rect_radiance(E, 3.0, (1.2, 0.15)))
    for lt, _ in lights.values():
        lt.CreateIntensityAttr(0.0)
        lt.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    q = rig.quat_z_to([0, 0, 1])
    UsdGeom.Xformable(d.GetPrim()).AddOrientOp().Set(Gf.Quatf(q[3], q[0], q[1], q[2]))

    def lum():
        h.render()
        img = h.render()["top"].astype(np.float32) / 255.0
        lin = np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)
        return lin @ np.array([0.2126, 0.7152, 0.0722])

    base = lum()
    res = {}
    for name, (lt, i) in lights.items():
        lt.GetIntensityAttr().Set(float(i))
        y = lum() - base
        lt.GetIntensityAttr().Set(0.0)
        res[name] = y
    H, W = base.shape
    ref = res["distant"]
    win = (slice(H // 2 - 40, H // 2 + 40), slice(W // 2 - 60, W // 2 + 60))
    out = {k: round(float(np.median(v[win]) / max(np.median(ref[win]), 1e-6)), 4) for k, v in res.items()}
    print("UNITS", json.dumps(out), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "units.json").write_text(json.dumps(dict(relative_to_distant=out, E=E, window="central 80x120 px of top"),
                                               indent=1))
    h.close()


if __name__ == "__main__":
    main()
