"""Beauty stills of the twin mid-task (Isaac; never PyAV here): a recorded cal2 episode is replayed with physics
(env/replay_video.py protocol, so the brick in the fingers is really grasped by the sim) and paused at chosen ticks,
then a free camera renders hero / close-up / top-down / low views at high resolution.

    sim/twin/run.sh sim/twin/presentation/beauty.py --episode 18 --ticks 120 --views hero_a closeup_a top_down
    ... --mode pt --spp 64 --passes 16 --width 2560 --height 1440      # path tracing (accumulated) instead of RTX RT

Views: live.py's searched HEROES ("hero_a" ...), plus views placed relative to the fingertip at that tick
("closeup_*"), and fixed station views ("top_down", "low_*", "wide_*"); any 'ex,ey,ez,tx,ty,tz[,focal]' also works.
The look is the twin's own (fitted lights / materials); background = live.py's dark room (inspect billboard hidden).
Output: sim/twin/out/presentation/beauty/<prefix>ep<NNN>_t<tick>_<view>_<mode>.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
DATA = REPO / "sim/twin/data/canonical/duplo_teleop_cal2_rgb_456"
OUT = REPO / "sim/twin/out/presentation/beauty"

ap = argparse.ArgumentParser()
ap.add_argument("--episode", type=int, required=True)
ap.add_argument("--ticks", type=int, nargs="+", required=True)
ap.add_argument("--views", nargs="+", default=["hero_a"])
ap.add_argument("--width", type=int, default=1920)
ap.add_argument("--height", type=int, default=1080)
ap.add_argument("--mode", choices=("rt", "pt"), default="rt", help="rt: RTX real-time + DLAA | pt: path tracing")
ap.add_argument("--spp", type=int, default=32, help="pt: samples per pass")
ap.add_argument("--passes", type=int, default=12, help="render passes per still (RT settles / PT accumulates)")
ap.add_argument("--bounces", type=int, default=6)
ap.add_argument("--upside-down", type=int, nargs="*", default=[],
                help="episodes whose target lies upside down in the real footage (scene/layout.py does not detect it)")
ap.add_argument("--bg", choices=("dark", "hall"), default="dark")
ap.add_argument("--prefix", default="")
ap.add_argument("--jpg", action="store_true", help="also write a JPEG (quick look)")
ap.add_argument("--pres-table", action="store_true", help="presentation table texture (pres_texture.py)")
args = ap.parse_args()
if "av" in sys.modules:
    raise SystemExit("PyAV must not be imported in the Isaac process")

from sim.twin.env.twin_env import launch_app  # noqa: E402

launch_app(headless=True, cameras=True)
from sim.twin.stream.live import HEROES, apply_background, hero_pose, patch_scene, set_hero  # noqa: E402

eye0, tgt0, f0 = HEROES["hero_a"]
patch_scene(args.width, args.height, eye0, tgt0, f0)
from sim.twin.env import TwinEnv, TwinEnvCfg  # noqa: E402
from sim.twin.presentation.pres_scene import pres_look  # noqa: E402

env = TwinEnv(TwinEnvCfg(cameras=(), color_correct=False, randomize=False, meas_noise=False,
                         max_relative_target=1000.0, look=pres_look() if args.pres_table else None))
import carb  # noqa: E402
import cv2  # noqa: E402
import omni.usd  # noqa: E402
import torch  # noqa: E402
from bases import load_episodes  # noqa: E402
from env import _quat_apply_xyzw  # noqa: E402

from sim.twin.scene.layout import layout_from_episode  # noqa: E402

stage = omni.usd.get_context().get_stage()
apply_background(stage, args.bg)
st = carb.settings.get_settings()
st.set("/rtx/post/aa/op", 4)  # DLAA
if args.mode == "pt":
    st.set("/rtx/rendermode", "PathTracing")
    st.set("/rtx/pathtracing/spp", int(args.spp))
    st.set("/rtx/pathtracing/totalSpp", int(args.spp * args.passes))
    st.set("/rtx/pathtracing/maxBounces", int(args.bounces))
    st.set("/rtx/pathtracing/maxSpecularAndTransmissionBounces", int(args.bounces))
    st.set("/rtx/pathtracing/optixDenoiser/enabled", True)
    st.set("/rtx/pathtracing/clampSpp", 0)
else:
    st.set("/rtx/rendermode", "RaytracedLighting")
cam = env.scene.sensors["hero_cam"]
R = env.residual
o0 = env.scene.env_origins[0]


def view(name: str, tip: np.ndarray, jaw: np.ndarray):
    """(eye, target, focal) for a view name at the current fingertip `tip` / jaw x-axis `jaw` (env frame)."""
    t = tip + np.array([0.0, 0.0, -0.01])
    close = {  # fingertip close-ups: (azimuth deg, elevation deg, distance m, focal mm)
        "closeup_a": (40, 18, 0.20, 30.0), "closeup_b": (75, 10, 0.19, 30.0), "closeup_c": (10, 25, 0.21, 30.0),
        "closeup_d": (-30, 15, 0.21, 30.0), "closeup_e": (130, 20, 0.22, 30.0), "closeup_f": (60, 35, 0.17, 24.0),
        "closeup_g": (20, 15, 0.32, 35.0), "closeup_h": (45, 22, 0.34, 35.0), "closeup_i": (5, 30, 0.30, 30.0),
    }
    fixed = {
        "top_down": ((0.10, -0.07, 1.30), (0.33, -0.07, 0.0), 16.0),
        "top_down_b": ((0.02, -0.05, 1.05), (0.30, -0.05, 0.0), 12.0),
        "top_tight": ((0.20, -0.06, 0.95), (0.30, -0.06, 0.0), 20.0),
        "hero_low": ((0.75, 0.35, 0.12), (0.22, -0.02, 0.12), 22.0),
        "low_d": ((0.48, -0.30, 0.03), (0.22, 0.0, 0.16), 18.0),
        "low_e": ((0.62, -0.10, 0.025), (0.22, 0.02, 0.15), 20.0),
        "low_f": ((0.42, -0.18, 0.05), (0.24, 0.03, 0.13), 16.0),
        "low_a": ((0.62, 0.30, 0.035), (0.18, -0.05, 0.17), 16.0),
        "low_b": ((0.58, -0.42, 0.04), (0.15, 0.0, 0.17), 16.0),
        "low_c": ((0.70, 0.02, 0.03), (0.15, -0.05, 0.18), 18.0),
        "wide_a": ((1.05, 0.70, 0.55), (0.25, -0.08, 0.08), 18.0),
        "wide_b": ((0.95, -0.75, 0.60), (0.25, -0.05, 0.08), 18.0),
    }
    if name in close:
        az, el, dist, f = close[name]
        a, e = np.radians(az), np.radians(el)
        eye = t + dist * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        return tuple(eye), tuple(t), f
    if name in fixed:
        return fixed[name]
    return hero_pose(name)


def render(passes: int) -> np.ndarray:
    img = None
    for _ in range(passes):
        env.sync()
        env.sim.render()
        env.scene.update(0.0)
    img = cam.data.output["rgb"][0, ..., :3]
    return img.detach().cpu().numpy().astype(np.uint8)


def _flip_targets(episodes):
    """The layout detector does not see brick orientation: flip the target of these episodes (checked by eye on the
    real top / wrist frames) through scene/bricks.py's own upside-down placement."""
    import sim.twin.scene.bricks as sb

    orig = sb.plan_from_layout

    def plan_from_layout(lay):
        plan = orig(lay)
        plan["target"]["upside_down"] = int(lay.get("episode", -1)) in episodes
        return plan

    sb.plan_from_layout = plan_from_layout


_flip_targets(args.upside_down)

d = load_episodes(DATA, episodes=[args.episode])[args.episode]
A, S = d["action"], d["state"]
T, insp = len(A), int(d["inspection_frame"])
g = S[:insp, 5]
gf = next((f for f in range(1, len(g)) if g[f - 1] > 12 >= g[f] and (g[f:] <= 15).all()), None)
R.noise.zero_()
env.reset(seed=args.episode, joint_deg=S[gf], brick_pose=(0.8, 0.5, 0.0), render=False)
tip = (R.tip_w() - env.scene.env_origins)[0]
q = R.robot.data.body_quat_w[:1, R.grip_body]
xa = _quat_apply_xyzw(q, torch.tensor([[1.0, 0, 0]], device=env.device))[0]
brick = (float(tip[0]), float(tip[1]), float(torch.atan2(xa[1], xa[0])))
lay = layout_from_episode(args.episode, DATA)
env.reset(seed=args.episode, joint_deg=S[0], brick_pose=brick, render=False, layout=lay)
OUT.mkdir(parents=True, exist_ok=True)
log = []
t = 0
for tick in sorted(args.ticks):
    while t < min(tick, T):
        env.step(A[t], render=False)
        t += 1
    tipn = (R.tip_w()[0] - o0).detach().cpu().numpy()
    qn = R.robot.data.body_quat_w[:1, R.grip_body]
    jaw = _quat_apply_xyzw(qn, torch.tensor([[1.0, 0, 0]], device=env.device))[0].detach().cpu().numpy()
    bpos = R.brick_local()[0].detach().cpu().numpy()
    for i, name in enumerate(args.views):
        eye, tgt, f = view(name, tipn, jaw)
        set_hero(stage, eye, tgt, f)
        t0 = time.perf_counter()
        img = render(args.passes + (4 if i == 0 else 0))
        dt = time.perf_counter() - t0
        stem = f"{args.prefix}ep{args.episode:03d}_t{tick:03d}_{name}_{args.mode}"
        cv2.imwrite(str(OUT / f"{stem}.png"), img[..., ::-1])
        if args.jpg:
            cv2.imwrite(str(OUT / f"{stem}.jpg"), img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
        rec = dict(file=f"{stem}.png", tick=tick, view=name, eye=np.round(eye, 4).tolist(),
                   target=np.round(tgt, 4).tolist(), focal=f, tip=np.round(tipn, 4).tolist(),
                   brick=np.round(bpos, 4).tolist(), brick_tip_dist=round(float(np.linalg.norm(bpos - tipn)), 4),
                   mode=args.mode, render_s=round(dt, 1), mean=round(float(img.mean()), 1))
        log.append(rec)
        print("BEAUTY", json.dumps(rec), flush=True)
with open(OUT / "beauty_log.jsonl", "a") as fh:
    for rec in log:
        fh.write(json.dumps(rec) + "\n")
sys.stdout.flush()
import os  # noqa: E402

os._exit(0)  # skip Kit's slow shutdown
