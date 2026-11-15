"""Open-loop twin replays of recorded cal2 episodes for the presentation (Isaac; never PyAV here).

    sim/twin/run.sh sim/twin/presentation/twin_replay.py --episodes 18 76 122            # screen: physics only
    sim/twin/run.sh sim/twin/presentation/twin_replay.py --episodes 18 --render         # + every frame, 3 cameras
    sim/twin/run.sh sim/twin/presentation/twin_replay.py --episodes 18 --render --dr-seeds 1 2 3 --cams top

The protocol is env/replay_video.py's sim stage (the current twin defaults: twin_profile cameras, W9 joint offsets,
fin-ray B25C gripper and Duplo shell colliders from physics/params.json, the fitted servo): scene from the real first
top frame (scene/layout.py), the target brick at the fingertip of the recorded gripper-close frame, the recorded
actions replayed with the follower clamp off and no measurement noise. Per tick it logs the brick, the fingertip and
the gripper so the captions can say exactly what the sim did (work/epNNN/telemetry<tag>.json, work/screen.jsonl).
--dr-seeds: one replay per seed with the RL run's visual DR only (rl/train_image.py --dr_scale 0.5 --spot_scale 1.0;
physics stays nominal).
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
WORK = REPO / "sim/twin/out/presentation/work"
CAMS = ("top", "wrist", "inspect")

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, nargs="+", required=True)
ap.add_argument("--render", action="store_true", help="save every rendered frame (work/epNNN/sim_<cam><tag>.npy)")
ap.add_argument("--width", type=int, default=640)
ap.add_argument("--height", type=int, default=480)
ap.add_argument("--cams", nargs="+", default=list(CAMS))
ap.add_argument("--dr-seeds", type=int, nargs="*", default=[], help="visual DR replays (one per seed)")
ap.add_argument("--dr-scale", type=float, default=0.5, help="visual DR scale (rl/visual_dr.scaled_cfg; RL used 0.5)")
ap.add_argument("--upside-down", type=int, nargs="*", default=[],
                help="episodes whose target lies upside down in the real footage (scene/layout.py does not detect it)")
ap.add_argument("--spot-scale", type=float, default=1.0, help="spotlight / light DR scale (RL used 1.0)")
ap.add_argument("--render-passes", type=int, default=1, help="sim.render() calls per frame (TwinEnvCfg.render_passes)")
ap.add_argument("--table-normal", type=int, default=1, help="--pres-table: 0 = no table normal map")
ap.add_argument("--table-clearcoat", type=float, default=None, help="--pres-table: table clear-coat weight override")
ap.add_argument("--hide-table-body", action="store_true",
                help="hide the physics table's own visual (the textured quad 0.3 mm above it stays): shadow-acne test")
ap.add_argument("--lights-off", nargs="*", default=[], help="diagnostics: light prim path prefixes set to intensity 0")
ap.add_argument("--carb", nargs="*", default=[], help="RTX carb settings key=value (bool / int / float / str)")
ap.add_argument("--raw", action="store_true", help="no camera sensor model (TwinEnvCfg.color_correct=False)")
ap.add_argument("--out-tag", default="", help="suffix of the saved files (frames / telemetry)")
ap.add_argument("--pres-table", action="store_true",
                help="presentation table texture (pres_texture.py; the training texture is unchanged)")
ap.add_argument("--wire-scale", type=int, default=2, help="wire pass resolution = width x height x this")
ap.add_argument("--wire-thickness", type=float, default=1.0, help="/rtx/wireframe/wireframeThickness (Kit default 0.3)")
ap.add_argument("--wire", action="store_true", help="--render: also an RTX wireframe pass of the top camera (sim_wire)")
args = ap.parse_args()
if "av" in sys.modules:
    raise SystemExit("PyAV must not be imported in the Isaac process")

from sim.twin.env.twin_env import launch_app  # noqa: E402

launch_app(headless=True, cameras=True)
from sim.twin.presentation.pres_scene import add_wire_camera, pres_look, set_wireframe  # noqa: E402

if args.wire:
    add_wire_camera("top_cam", width=args.width * args.wire_scale, height=args.height * args.wire_scale)
from sim.twin.env import TwinEnv, TwinEnvCfg  # noqa: E402

kw = {"look": pres_look(normal=bool(args.table_normal), clearcoat=args.table_clearcoat)} if args.pres_table else {}
if args.dr_seeds:
    from sim.twin.env.randomize import TwinRandomizer
    from sim.twin.rl.visual_dr import scaled_cfg

    dr = TwinRandomizer(scaled_cfg(args.dr_scale, codec=None))  # the RL's visual DR (rl/train_image.py)
    dr.set_light_scale(args.spot_scale)
    kw["randomizers"] = [dr]
W, H = (args.width, args.height) if args.render else (160, 120)
kw["render_passes"] = args.render_passes
kw["color_correct"] = not args.raw
env = TwinEnv(TwinEnvCfg(width=W, height=H, cameras=tuple(args.cams), meas_noise=False, max_relative_target=1000.0,
                         **kw))
if args.wire:
    import carb
    import omni.usd

    carb.settings.get_settings().set("/rtx/wireframe/wireframeThickness", float(args.wire_thickness))

    print("wireframe render products", set_wireframe(omni.usd.get_context().get_stage(), "WireCam", "shaded"),
          flush=True)
if args.hide_table_body:
    import omni.usd
    from pxr import UsdGeom

    _st = omni.usd.get_context().get_stage()
    for _e in range(env.num_envs):
        UsdGeom.Imageable(_st.GetPrimAtPath(f"/World/envs/env_{_e}/Table")).MakeInvisible()
if args.carb:
    import carb

    def _val(v):
        if v.lower() in ("true", "false"):
            return v.lower() == "true"
        for f in (int, float):
            try:
                return f(v)
            except ValueError:
                pass
        return v

    for kv in args.carb:
        k, v = kv.split("=", 1)
        carb.settings.get_settings().set(k, _val(v))
        print("carb", k, "=", carb.settings.get_settings().get(k), flush=True)
if args.lights_off:
    import omni.usd

    for _p in omni.usd.get_context().get_stage().Traverse():
        if any(_p.GetPath().pathString.startswith(x) for x in args.lights_off) and _p.GetAttribute("inputs:intensity"):
            _p.GetAttribute("inputs:intensity").Set(0.0)
            print("light off", _p.GetPath(), flush=True)
import torch  # noqa: E402
from bases import load_episodes  # noqa: E402  (residual env; on sys.path once TwinEnv is built)
from env import _quat_apply_xyzw  # noqa: E402

from sim.twin.scene.layout import layout_from_episode  # noqa: E402


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

eps = load_episodes(DATA, episodes=args.episodes)
R = env.residual
o0 = env.scene.env_origins[0]


def tip_local():
    return (R.tip_w()[0] - o0).detach().cpu().numpy()


runs = [(ep, s) for ep in args.episodes for s in (args.dr_seeds or [None])]
for ep, seed in runs:
    d = eps[ep]
    A, S = d["action"], d["state"]
    T, insp = len(A), int(d["inspection_frame"])
    g = S[:insp, 5]
    gf = next((f for f in range(1, len(g)) if g[f - 1] > 12 >= g[f] and (g[f:] <= 15).all()), None)
    rel = next((t for t in range(insp, T) if S[t, 5] > 15), None)  # real release (gripper opens after inspect)
    R.noise.zero_()
    env.reset(seed=ep, joint_deg=S[gf], brick_pose=(0.8, 0.5, 0.0), render=False)
    tip = (R.tip_w() - env.scene.env_origins)[0]
    q = R.robot.data.body_quat_w[:1, R.grip_body]
    xa = _quat_apply_xyzw(q, torch.tensor([[1.0, 0, 0]], device=env.device))[0]
    brick = (float(tip[0]), float(tip[1]), float(torch.atan2(xa[1], xa[0])))
    lay = layout_from_episode(ep, DATA)
    env.reset(seed=ep if seed is None else seed, joint_deg=S[0], brick_pose=brick, render=False,
              randomize=seed is not None, layout=lay)
    tag = ("" if seed is None else f"_dr{seed}") + args.out_tag
    frames = {c: np.zeros((T, H, W, 3), np.uint8) for c in args.cams} if args.render else None
    if frames is not None and args.wire:
        frames["wire"] = np.zeros((T, H * args.wire_scale, W * args.wire_scale, 3), np.uint8)
    tel = {k: [] for k in ("brick", "tip", "grip_sim", "lifted", "grasped", "placed", "phase")}
    t0 = time.perf_counter()
    for t in range(T):
        obs, _, _, _, info = env.step(A[t], render=args.render)
        tel["brick"].append([round(float(v), 4) for v in info["brick_pos"]])
        tel["tip"].append(np.round(tip_local(), 4).tolist())
        tel["grip_sim"].append(round(float(env.state()[5]), 2))
        for k in ("lifted", "grasped", "placed"):
            tel[k].append(bool(info[k]))
        tel["phase"].append(int(info.get("phase", -1)))
        if frames is not None:
            for c in args.cams:
                frames[c][t] = obs["images"][c]
            if "wire" in frames:  # same sim.render() as the frame above (raw RGB, no camera model)
                frames["wire"][t] = env.scene.sensors["wire_cam"].data.output["rgb"][0, ..., :3].cpu().numpy()
    wall = time.perf_counter() - t0
    b, tp = np.array(tel["brick"]), np.array(tel["tip"])
    dist = np.linalg.norm(b - tp, axis=1)
    w0, w1 = gf + 15, (rel or T) - 2
    hold = float((dist[w0:w1] < 0.035).mean()) if w1 > w0 else 0.0
    plate = (lay.get("plates") or {}).get(lay.get("bowl"))
    end_plate = float(np.hypot(b[-1, 0] - plate["x"], b[-1, 1] - plate["y"])) if plate else None
    res = dict(episode=int(ep), dr_seed=seed, T=T, grasp_frame=gf, inspection_frame=insp, real_release=rel,
               lifted=bool(any(tel["lifted"])), brick_max_z=round(float(b[:, 2].max()), 3), hold_frac=round(hold, 3),
               first_lift=next((t for t, v in enumerate(tel["lifted"]) if v), None),
               last_held=int(np.flatnonzero(dist < 0.035).max()) if (dist < 0.035).any() else None,
               end_brick=b[-1].tolist(), end_dist_to_bowl=None if end_plate is None else round(end_plate, 3),
               in_bowl_at_end=bool(end_plate is not None and end_plate < 0.08 and b[-1, 2] < 0.06),
               placed_flag=bool(any(tel["placed"])), bowl=lay.get("bowl"), target=lay.get("target"),
               n_distractors=len(lay.get("distractors", [])), wall_s=round(wall, 1))
    out = WORK / f"ep{ep:03d}"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"telemetry{tag}.json").write_text(json.dumps(dict(res, tel=tel, layout=lay)))
    with open(WORK / "screen.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")
    if frames is not None:
        for c in frames:
            np.save(out / f"sim_{c}{tag}.npy", frames[c])
    print("REPLAY_EP", json.dumps(res), flush=True)
env.close()
