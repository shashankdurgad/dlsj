"""Diagnostics for the striped table in long replays (Isaac): replay an episode to a tick without rendering, then
render the top camera with candidate surfaces hidden and list large flat prims near the table plane.

    sim/twin/run.sh sim/twin/presentation/stripe_diag.py --episode 184 --tick 222
Output: out/presentation/look_test/diag_<variant>.png
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
ap.add_argument("--episode", type=int, default=184)
ap.add_argument("--tick", type=int, default=222)
args = ap.parse_args()

from sim.twin.env.twin_env import launch_app  # noqa: E402

launch_app(headless=True, cameras=True)
from sim.twin.env import TwinEnv, TwinEnvCfg  # noqa: E402
from sim.twin.presentation.pres_scene import pres_look  # noqa: E402

env = TwinEnv(TwinEnvCfg(meas_noise=False, max_relative_target=1000.0, look=pres_look(), color_correct=False))
import cv2  # noqa: E402
import omni.usd  # noqa: E402
from bases import load_episodes  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

from sim.twin.scene.layout import layout_from_episode  # noqa: E402

d = load_episodes(DATA, episodes=[args.episode])[args.episode]
lay = layout_from_episode(args.episode, DATA)
import torch  # noqa: E402
from env import _quat_apply_xyzw  # noqa: E402

S = d["state"]
insp = int(d["inspection_frame"])
g = S[:insp, 5]
gf = next((f for f in range(1, len(g)) if g[f - 1] > 12 >= g[f] and (g[f:] <= 15).all()), None)
R = env.residual
R.noise.zero_()
env.reset(seed=args.episode, joint_deg=S[gf], brick_pose=(0.8, 0.5, 0.0), render=False)
tip = (R.tip_w() - env.scene.env_origins)[0]
q = R.robot.data.body_quat_w[:1, R.grip_body]
xa = _quat_apply_xyzw(q, torch.tensor([[1.0, 0, 0]], device=env.device))[0]
brick = (float(tip[0]), float(tip[1]), float(torch.atan2(xa[1], xa[0])))
env.reset(seed=args.episode, joint_deg=S[0], brick_pose=brick, render=False, layout=lay)
stage = omni.usd.get_context().get_stage()
OUT.mkdir(parents=True, exist_ok=True)


def shot(tag):
    img = env.render()["top"]
    cv2.imwrite(str(OUT / f"diag_{tag}.png"), img[..., ::-1])


def flat_prims(label):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    for p in stage.Traverse():
        if not p.IsA(UsdGeom.Gprim):
            continue
        b = cache.ComputeWorldBound(p).ComputeAlignedRange()
        if b.IsEmpty():
            continue
        lo, hi = b.GetMin(), b.GetMax()
        if (hi[0] - lo[0]) > 0.3 and (hi[1] - lo[1]) > 0.3 and -0.02 < hi[2] < 0.01:
            vis = UsdGeom.Imageable(p).ComputeVisibility()
            print(f"FLAT[{label}] {p.GetPath()} top z {hi[2]:.5f} min z {lo[2]:.5f} vis {vis}", flush=True)


flat_prims("t0")
shot("t0")
for t in range(args.tick):
    env.step(d["action"][t], render=True)
flat_prims(f"t{args.tick}")
shot(f"t{args.tick}")
root = stage.GetPrimAtPath("/World/envs/env_0")
names = [c.GetName() for c in root.GetChildren()]
print("CHILDREN", names, flush=True)
print("WORLD", [c.GetName() for c in stage.GetPrimAtPath("/World").GetChildren()], flush=True)
skip = {"TableTopVisual", "TwinProps", "Table"}
world = ["/World/" + c.GetName() for c in stage.GetPrimAtPath("/World").GetChildren() if c.GetName() != "envs"]
for n in names + world:
    if n in skip:
        continue
    path = n if n.startswith("/") else f"/World/envs/env_0/{n}"
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsA(UsdGeom.Imageable):
        continue
    UsdGeom.Imageable(prim).MakeInvisible()
    shot(f"hide_{path.strip('/').replace('/', '_')}")
    UsdGeom.Imageable(prim).MakeVisible()
import os  # noqa: E402

os._exit(0)
