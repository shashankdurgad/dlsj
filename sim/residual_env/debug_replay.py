"""Render one replayed demo (top + wrist frames every 3 ticks) to inspect the grasp."""
import argparse
import json
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--episode", type=int, default=9)
ap.add_argument("--data", default=str(Path.home() / "residual_env/data/duplo_teleop_3phase"))
ap.add_argument("--out", default=str(Path.home() / "residual_env/out/debug_replay"))
args = ap.parse_args()
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import launch  # noqa: E402

app = launch(cameras=True)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from app import make  # noqa: E402
from bases import load_episodes  # noqa: E402
from env import EnvCfg, _quat_apply_xyzw  # noqa: E402

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
cfg = EnvCfg(num_envs=1, cameras=True, max_relative_target=1000.0, meas_noise=False, episode_s=60)
sim, scene, layout, env = make(cfg)
ep = load_episodes(args.data, [args.episode])[args.episode]
dev = env.device
gs = torch.tensor(ep["state"][ep["grasp_frame"]], device=dev).unsqueeze(0)
env.reset(joint_deg=gs)
env.physics()
env.physics()
tip = (env.tip_w() - scene.env_origins)[0]
q = env.robot.data.body_quat_w[:1, env.grip_body]
xa = _quat_apply_xyzw(q, torch.tensor([[1.0, 0, 0]], device=dev))[0]
brick = torch.tensor([[float(tip[0]), float(tip[1]), float(torch.atan2(xa[1], xa[0]))]], device=dev)
env.reset(joint_deg=torch.tensor(ep["state"][0], device=dev).unsqueeze(0), brick_pose=brick)
rows = []
for t in range(len(ep["action"])):
    _, _, _, info = env.step(goal_override=torch.tensor(ep["action"][t], device=dev).unsqueeze(0))
    s = env.measured(noisy=False)[0].cpu().numpy()
    bz = float((env.brick.data.root_pos_w - scene.env_origins)[0, 2])
    from env import _quat_conj_xyzw
    qg = env.robot.data.body_quat_w[:1, env.grip_body]
    rel = _quat_apply_xyzw(_quat_conj_xyzw(qg), (env.brick.data.root_pos_w - env.robot.data.body_pos_w[:, env.grip_body]))[0]
    rows.append({"t": t, "grip_meas": round(float(s[5]), 2), "grip_cmd": round(float(ep["action"][t][5]), 2),
                 "grip_ref": round(float(env.r[0, 5]), 2), "brick_z": round(bz, 4),
                 "dist": round(float(info["dist"][0]), 4), "grasped": bool(info["grasped"][0]),
                 "brick_in_gripper_frame": [round(float(v), 4) for v in rel],
                 "grip_link_z": round(float((env.robot.data.body_pos_w[0, env.grip_body] - scene.env_origins[0])[2]), 4)})
    if t % 3 == 0:
        env.sim.render()
        env.scene.update(0.0)
        top = env.scene["top_cam"].data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
        wr = env.scene["wrist_cam"].data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
        Image.fromarray(np.concatenate([top, wr], 1)).save(out / f"{t:04d}.jpg")
(out / "trace.json").write_text(json.dumps(rows))
g, i = ep["grasp_frame"], ep["inspection_frame"]
print("TRACE", json.dumps([r for r in rows if g - 15 <= r["t"] <= min(i, g + 45) and r["t"] % 3 == 0]))
print("DEBUG_DONE", g, i)
app.close()
