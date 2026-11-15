"""Write the nominal light rig (visuals/lighting.json light fields) from the fitted / measured evidence (plain Python).

    python sim/twin/lighting/make_rig.py [--out sim/twin/visuals/lighting.json] [--dome-share 0.745] [--lamp-height 1.0]

Evidence (log.md): plate-highlight directions (visual-fit estimate_lights.py: az ~125 deg / el ~62 and az ~180 /
el 50-65, behind the robot), the light-basis fit (fit_rig.py: top camera prefers light from az -170 / el 64 and
az 135 / el 60, little from the front), the real arm shadows (falling to env +x +y), the user's facts (fluorescent
strips ~10 m up, parallel to the table's long edge, weak; warm 2700-3200 K lamps ~6 m out along the inspect camera's
view direction; auto exposure / white balance) and the user's verdict on the old rig (too strong, too hard).
Shares are of the nominal top-camera table brightness (light-basis increments, basis_v1.npz), whose total is kept at
the old rig's (so visuals/fit's exposure calibration still holds): dome (broad hall / ceiling light) carries most of
it, a very soft overhead key (the ensemble of strips), two faint strips (the plate highlights), three faint strips
far across the hall at ~12 deg elevation (the grazing sheen of the lacquered table in the inspect view) and two warm
lamps low (1 m) and 6 m out along the inspect camera's view.
Colours: strips / key 4500 K with a slight green (fluorescent), lamps 3000 K, dome 5000 K; all divided by their
share-weighted mix so the scene average stays white (the cameras white-balanced on it; materials are fitted to them).
Other lighting.json keys (exposure_scale, backdrop_gain, hall_gain: visual-fit / sensor) are kept as they are.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from sim.twin.lighting.rig import CENTER  # noqa: E402
from sim.twin.lighting.spotlights import kelvin_rgb, tint_rgb  # noqa: E402

LIGHTING_JSON = REPO / "sim/twin/visuals/lighting.json"
# brightness increment of the top image (mean linear luma) per unit of light (actual prim intensity, after the
# exposure scale), from basis_v1: dome 0.079 / 715, distant lights ~0.000198 per unit at the zenith x sin(elevation)
DOME_PER_UNIT = 0.079 / 715.0
DISTANT_PER_UNIT = 0.000198
OLD_TOTAL = 0.215  # old rig (dome 715 + key 250 + 3 x 107 ceiling + RTX ambient), emitters excluded
TABLE_EDGE = [-0.2762, -0.9611, 0.0]  # table long edge, env frame (visuals/textures/table_top.json)
INSPECT_VIEW = [-0.2106, -0.9683]  # inspect camera optical axis, horizontal part (calib/cameras_measured.json)
INSPECT_POS = [0.3794, 0.1803]


def unit(az_deg: float, el_deg: float) -> list:
    """Unit vector TO a light at azimuth / elevation (env frame)."""
    a, e = math.radians(az_deg), math.radians(el_deg)
    return [math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(LIGHTING_JSON))
    ap.add_argument("--dome-share", type=float, default=0.745)
    ap.add_argument("--key-share", type=float, default=0.08)
    ap.add_argument("--strip-share", type=float, default=0.025, help="per strip")
    ap.add_argument("--strip-size", type=float, nargs=2, default=[4.5, 0.25], help="long, short (m)")
    ap.add_argument("--lamp-share", type=float, default=0.06, help="per lamp")
    ap.add_argument("--lamp-dist", type=float, default=6.0)
    ap.add_argument("--lamp-height", type=float, default=1.0,
                    help="m above the table: the real warm grazing sheen on the far table in the inspect view needs "
                         "the lamps' mirror point ~0.6 m in front of the inspect camera (z 0.10): h ~ 6 * 0.1 / 0.6")
    ap.add_argument("--lamp-radius", type=float, default=0.25)
    ap.add_argument("--far-share", type=float, default=0.015, help="per far strip")
    ap.add_argument("--far-el", type=float, default=12.0)
    ap.add_argument("--ceiling-height", type=float, default=10.0)
    args = ap.parse_args()

    cur = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else {}
    s = float(cur.get("exposure_scale", 0.715))
    fl = kelvin_rgb(4500.0) * tint_rgb(0.03)
    warm = kelvin_rgb(3000.0)
    dome_c = kelvin_rgb(5000.0)
    key_dir = unit(-170.0, 72.0)
    strips = [(-170.0, 62.0), (128.0, 62.0)]
    v = np.array(INSPECT_VIEW) / np.linalg.norm(INSPECT_VIEW)
    side_axis = np.array([-v[1], v[0]])
    lamp_xy = [np.array(INSPECT_POS) + args.lamp_dist * v + o * side_axis for o in (-1.5, 1.5)]

    def per_unit_distant(d):
        return DISTANT_PER_UNIT * max(d[2], 0.0)

    view_az = math.degrees(math.atan2(v[1], v[0]))
    far = [(view_az + o, args.far_el) for o in (-35.0, 0.0, 35.0)]  # strips across the hall, seen at grazing angles
    shares = {"dome": args.dome_share, "key": args.key_share, "strip": args.strip_share, "lamp": args.lamp_share,
              "far": args.far_share}
    w_fl = shares["key"] + 2 * shares["strip"] + len(far) * shares["far"]
    mix = shares["dome"] * dome_c + w_fl * fl + 2 * shares["lamp"] * warm
    mix = mix / (shares["dome"] + w_fl + 2 * shares["lamp"])
    col = {k: (c / mix).round(4).tolist() for k, c in (("dome", dome_c), ("fl", fl), ("warm", warm))}

    rig = dict(cur)
    rig["dome_intensity"] = round(OLD_TOTAL * shares["dome"] / DOME_PER_UNIT / s, 1)
    rig["dome_rgb"] = col["dome"]
    rig["key_intensity"] = round(OLD_TOTAL * shares["key"] / per_unit_distant(key_dir) / s, 1)
    rig["key_angle_deg"] = 40.0
    # key: rotateXYZ that turns +z into key_dir (x then y rotation)
    rig["key_rot_xyz_deg"] = [round(math.degrees(math.atan2(-key_dir[1], key_dir[2])), 2),
                              round(math.degrees(math.asin(key_dir[0])), 2), 0.0]
    rig["key_rgb"] = col["fl"]
    rig["ceiling_lights"] = []
    for az, el in strips:
        d = unit(az, el)
        rig["ceiling_lights"].append(dict(
            type="rect", dir=[round(x, 4) for x in d], dist=round(args.ceiling_height / math.sin(math.radians(el)), 2),
            size=list(args.strip_size), axis=TABLE_EDGE, rgb=col["fl"], angle_deg=4.0,
            intensity=round(OLD_TOTAL * shares["strip"] / per_unit_distant(d) / s, 1)))
    for az, el in far:
        d = unit(az, el)
        rig["ceiling_lights"].append(dict(
            type="rect", dir=[round(x, 4) for x in d], dist=12.0, size=[2.4, 0.25], axis=TABLE_EDGE, rgb=col["fl"],
            angle_deg=4.0, intensity=round(OLD_TOTAL * shares["far"] / per_unit_distant(d) / s, 1)))
    rig["side_lights"] = []
    for xy in lamp_xy:
        pos = np.array([xy[0], xy[1], args.lamp_height])
        d = (pos - CENTER) / np.linalg.norm(pos - CENTER)
        rig["side_lights"].append(dict(pos=pos.round(3).tolist(), radius=args.lamp_radius,
                                       intensity=round(OLD_TOTAL * shares["lamp"] / per_unit_distant(d) / s, 1),
                                       rgb=col["warm"]))
    rig["ambient_intensity"] = 0.0
    rig["note"] = ("linear tonemap; flat TwinLook field names. Light fields written by sim/twin/lighting/make_rig.py "
                   "(evidence and shares in sim/twin/lighting/log.md): dome carries most light, very soft overhead key "
                   "(fluorescent ensemble), two faint 4.5 m strips 10 m up (plate highlights), three faint far strips "
                   "across the hall (grazing table sheen), two warm lamps 1 m up, 6 m out along the inspect view, "
                   "RTX flat ambient off. ceiling 'intensity' = irradiance at the workspace "
                   "centre (distant-light units); exposure_scale / backdrop_gain / hall_gain are other agents'.")
    rig["shares"] = shares
    Path(args.out).write_text(json.dumps(rig, indent=1) + "\n")
    print(json.dumps(rig, indent=1))


if __name__ == "__main__":
    main()
