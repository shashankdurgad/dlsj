"""Domain-randomisation stills (Isaac; never PyAV here): one recorded episode's scene (real first-frame layout, the
target brick where the gripper closed) with the arm at a recorded joint state, rendered once nominal and once per
DR seed, all three cameras.

    sim/twin/run.sh sim/twin/presentation/dr_stills.py --episode 18 --tick 85 --seeds 0 1 2 ... [--dr-scale 0.5]

DR = the RL run's visual DR at the end of its ramp (rl/train_image.py --dr_scale 0.5 --spot_scale 1.0:
TwinRandomizer(rl.visual_dr.scaled_cfg(0.5)) + set_light_scale(1.0)) with the full H.264 sensor round-trip (the RL
used an in-process JPEG stand-in for throughput). Output: out/presentation/dr/<cam>_nominal.png, <cam>_s<seed>.png,
draws.json (what each seed drew).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
DATA = REPO / "sim/twin/data/canonical/duplo_teleop_cal2_rgb_456"
OUT = REPO / "sim/twin/out/presentation/dr"
CAMS = ("top", "wrist", "inspect")

ap = argparse.ArgumentParser()
ap.add_argument("--episode", type=int, required=True)
ap.add_argument("--tick", type=int, required=True, help="recorded frame whose joint state poses the arm")
ap.add_argument("--seeds", type=int, nargs="+", default=list(range(16)))
ap.add_argument("--dr-scale", type=float, default=0.5)
ap.add_argument("--spot-scale", type=float, default=1.0)
args = ap.parse_args()
if "av" in sys.modules:
    raise SystemExit("PyAV must not be imported in the Isaac process")

from sim.twin.env import TwinEnv, TwinEnvCfg  # noqa: E402
from sim.twin.env.randomize import TwinRandomizer  # noqa: E402
from sim.twin.rl.visual_dr import scaled_cfg  # noqa: E402

dr = TwinRandomizer(scaled_cfg(args.dr_scale, codec=None))
dr.set_light_scale(args.spot_scale)
env = TwinEnv(TwinEnvCfg(cameras=CAMS, meas_noise=False, max_relative_target=1000.0, randomizers=[dr]))
import cv2  # noqa: E402
import torch  # noqa: E402
from bases import load_episodes  # noqa: E402
from env import _quat_apply_xyzw  # noqa: E402

from sim.twin.scene.layout import layout_from_episode  # noqa: E402

d = load_episodes(DATA, episodes=[args.episode])[args.episode]
S, insp = d["state"], int(d["inspection_frame"])
g = S[:insp, 5]
gf = next((f for f in range(1, len(g)) if g[f - 1] > 12 >= g[f] and (g[f:] <= 15).all()), None)
R = env.residual
R.noise.zero_()
env.reset(seed=args.episode, joint_deg=S[gf], brick_pose=(0.8, 0.5, 0.0), render=False)
tip = (R.tip_w() - env.scene.env_origins)[0]
q = R.robot.data.body_quat_w[:1, R.grip_body]
xa = _quat_apply_xyzw(q, torch.tensor([[1.0, 0, 0]], device=env.device))[0]
brick = (float(tip[0]), float(tip[1]), float(torch.atan2(xa[1], xa[0])))
lay = layout_from_episode(args.episode, DATA)
OUT.mkdir(parents=True, exist_ok=True)
draws = {}
for seed in [None, *args.seeds]:
    obs = env.reset(seed=args.episode if seed is None else seed, joint_deg=S[args.tick], brick_pose=brick,
                    layout=lay, randomize=seed is not None)
    for _ in range(3):  # let RTX settle after the light / material changes
        obs["images"] = env.render()
    name = "nominal" if seed is None else f"s{seed}"
    for c in CAMS:
        cv2.imwrite(str(OUT / f"{c}_{name}.png"), obs["images"][c][..., ::-1])
    draws[name] = json.loads(json.dumps(env.dr_draw, default=str)) if seed is not None else {}
    print("DR_STILL", name, flush=True)
(OUT / "draws.json").write_text(json.dumps(dict(episode=args.episode, tick=args.tick, dr_scale=args.dr_scale,
                                                spot_scale=args.spot_scale, draws=draws), indent=1))
sys.stdout.flush()
import os  # noqa: E402

os._exit(0)
