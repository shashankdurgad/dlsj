"""Quick look test (Isaac): one recorded episode frame, top / wrist / inspect with the presentation table and the
RTX wireframe pass on a duplicate top camera, for choosing settings.

    sim/twin/run.sh sim/twin/presentation/look_test.py --episode 18 --tick 150 [--no-pres-table] [--no-normal]
Output: out/presentation/look_test/<tag>_<cam>.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
DATA = REPO / "sim/twin/data/canonical/duplo_teleop_cal2_rgb_456"
OUT = REPO / "sim/twin/out/presentation/look_test"

ap = argparse.ArgumentParser()
ap.add_argument("--episode", type=int, default=18)
ap.add_argument("--tick", type=int, default=150)
ap.add_argument("--no-pres-table", action="store_true")
ap.add_argument("--no-normal", action="store_true")
ap.add_argument("--tag", default="t")
ap.add_argument("--wire-scale", type=int, default=1, help="wire camera resolution = 640x480 x this")
ap.add_argument("--thickness", type=float, nargs="*", default=[], help="/rtx/wireframe/wireframeThickness values")
args = ap.parse_args()

from sim.twin.env.twin_env import launch_app  # noqa: E402

launch_app(headless=True, cameras=True)
from sim.twin.presentation.pres_scene import add_wire_camera, pres_look, set_wireframe  # noqa: E402

add_wire_camera(width=640 * args.wire_scale, height=480 * args.wire_scale)
from sim.twin.env import TwinEnv, TwinEnvCfg  # noqa: E402

env = TwinEnv(TwinEnvCfg(meas_noise=False, max_relative_target=1000.0,
                         look=pres_look(table=not args.no_pres_table, normal=not args.no_normal)))
import cv2  # noqa: E402
import omni.usd  # noqa: E402
from bases import load_episodes  # noqa: E402

from sim.twin.scene.layout import layout_from_episode  # noqa: E402

d = load_episodes(DATA, episodes=[args.episode])[args.episode]
lay = layout_from_episode(args.episode, DATA)
obs = env.reset(seed=args.episode, joint_deg=d["state"][args.tick], layout=lay)
for _ in range(2):
    obs["images"] = env.render()
OUT.mkdir(parents=True, exist_ok=True)
for c, img in obs["images"].items():
    cv2.imwrite(str(OUT / f"{args.tag}_{c}.png"), img[..., ::-1])
stage = omni.usd.get_context().get_stage()
wc = env.scene.sensors["wire_cam"]
import carb  # noqa: E402

st = carb.settings.get_settings()
print("default thickness", st.get("/rtx/wireframe/wireframeThickness"), st.get("/rtx/wireframe/mode"), flush=True)
for th in [None, *args.thickness]:
    if th is not None:
        st.set("/rtx/wireframe/wireframeThickness", float(th))
    for mode in ("shaded",) if th is not None else ("instance", "shaded", "emissive"):
        paths = set_wireframe(stage, "WireCam", mode)
        for _ in range(3):
            env.sync()
            env.sim.render()
            env.scene.update(0.0)
        img = wc.data.output["rgb"][0, ..., :3].detach().cpu().numpy()
        if args.wire_scale > 1:
            img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(OUT / f"{args.tag}_wire_{mode}_th{th}.png"), img[..., ::-1])
        print("WIRE", mode, th, paths, float(img.mean()), flush=True)
import os  # noqa: E402

os._exit(0)
