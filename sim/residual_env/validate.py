"""Realism gate for the residual env (run before any RL).

(a) step response per joint vs A1's fitted first-order model; (b) gripper hold position on a brick vs the real 5.75;
(c) open-loop replay of recorded teleop episodes (brick placed under the gripper at the recorded grasp frame):
joint tracking RMSE vs recorded states, and grasp / lift / present success; (d) throughput.

    python validate.py --episodes 20 --num_envs 20 --out ~/residual_env/out/validate
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=20)
ap.add_argument("--num_envs", type=int, default=20)
ap.add_argument("--data", default="/home/joe-spark-2/desmond/datasets/duplo-cal2-456-3stage-20261114/duplo_teleop_cal2_rgb_3stage",
                help="cal2 3-stage build (current calibration); old-calibration data: --calib fit_old --data ~/residual_env/data/duplo_teleop_3phase")
ap.add_argument("--out", default=str(Path.home() / "residual_env/out/validate"))
ap.add_argument("--clamp", type=float, default=1000.0, help="replay clamp (teleop had none on the follower)")
ap.add_argument("--fps_envs", default="8,16,32")
ap.add_argument("--skip", default="")
ap.add_argument("--joint_offsets", default="fit_old", help="fit_old (default) | joe | fit_cal2")
ap.add_argument("--calib", default="joe_cal2", help="calibration profile (calib.py): joe | fit_old | fit_cal2 | joe_cal2; "
                "overrides --joint_offsets")
ap.add_argument("--grasp_rule", default="gripper", choices=["labels", "gripper"],
                help="where to place the brick: FK at the labels' grasp_frame, or at the frame the gripper closes "
                     "(cal2 3-stage labels mark 'pickup done', ~0.8 s after the close)")
args = ap.parse_args()
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import launch  # noqa: E402

app = launch(cameras=False)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from app import make  # noqa: E402
from bases import load_episodes  # noqa: E402
from env import EnvCfg, _quat_apply_xyzw  # noqa: E402

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
report = {}
cfg = EnvCfg(num_envs=args.num_envs, max_relative_target=args.clamp, meas_noise=False, episode_s=60,
             joint_offsets=args.joint_offsets, calib=args.calib)
sim, scene, layout, env = make(cfg)
dev = env.device
J = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def settle(n=10, goal=None):
    for _ in range(n):
        env.step(goal_override=goal if goal is not None else env.r.clone())


# ---------------------------------------------------------------- (a) step response
if "step" not in args.skip:
    env.reset()
    start = torch.tensor(layout.init_joint_deg, device=dev).expand(env.n, 6).clone()
    env.reset(joint_deg=start)
    # keep the brick out of the way for the step test
    settle(30, start)
    res = {}
    for j in range(6):
        env.reset(joint_deg=start)
        settle(15, start)
        g = start.clone()
        g[:, j] += 15.0
        meas, model = [], []
        m = env.measured(noisy=False)[0, j].item()
        mm = m
        for t in range(30):
            env.step(goal_override=g)
            meas.append(env.measured(noisy=False)[0, j].item())
            mm = mm + max(-env.vmax[j].item(), min(env.vmax[j].item(), env.K[j].item() * (g[0, j].item() - mm)))
            model.append(mm + env.bias[j].item())
        meas, model = np.array(meas), np.array(model)
        # effective first-order K from the sim trace (least squares on the increments)
        x = g[0, j].item() + env.bias[j].item() - meas[:-1]
        dy = meas[1:] - meas[:-1]
        k_eff = float((x * dy).sum() / max((x * x).sum(), 1e-9))
        res[J[j]] = {"K_model": round(env.K[j].item(), 4), "K_sim": round(k_eff, 4),
                     "rmse_vs_model_deg": round(float(np.sqrt(((meas - model) ** 2).mean())), 3),
                     "final_err_deg": round(float(g[0, j].item() - meas[-1]), 3)}
    report["step_response"] = res
    print("STEP", json.dumps(res), flush=True)

# ---------------------------------------------------------------- (c) replay (incl. (b) gripper hold on the brick)
eps = load_episodes(args.data)
if args.grasp_rule == "gripper":
    def _close_frame(grip, insp, thr=12.0, stay=15.0, open_min=15.0):
        g = grip[:insp] if insp else grip
        if len(g) == 0 or g.max() < open_min:
            return None
        for f in range(1, len(g)):
            if g[f - 1] > thr >= g[f] and (g[f:] <= stay).all():
                return f
        return None
    for d in eps.values():
        d["grasp_frame"] = _close_frame(d["state"][:, 5], d["inspection_frame"])
ok = [e for e, d in eps.items() if not d["excluded"] and (d["grasp_frame"] or 0) > 5 and d["inspection_frame"]]
rng = np.random.default_rng(0)
chosen = sorted(rng.choice(ok, size=min(args.episodes, len(ok), env.n), replace=False).tolist())
n = len(chosen)
print("REPLAY episodes", chosen, flush=True)
ids = torch.arange(n, device=dev)
# 1) FK at the recorded grasp frame -> fingertip xy + jaw axis yaw -> brick pose
gstate = torch.tensor(np.stack([eps[e]["state"][eps[e]["grasp_frame"]] for e in chosen]), device=dev)
full = torch.tensor(layout.init_joint_deg, device=dev).expand(env.n, 6).clone()
full[:n] = gstate
env.reset(joint_deg=full)
for _ in range(2):
    env.physics()
tip = (env.tip_w() - scene.env_origins)[:n]
q = env.robot.data.body_quat_w[:n, env.grip_body]
xaxis = _quat_apply_xyzw(q, torch.tensor([1.0, 0, 0], device=dev).expand(n, 3))
yaw = torch.atan2(xaxis[:, 1], xaxis[:, 0])
brick = torch.zeros(env.n, 3, device=dev)
brick[:, :2] = torch.tensor(layout.target.pos, device=dev)
brick[:n, 0], brick[:n, 1], brick[:n, 2] = tip[:, 0], tip[:, 1], yaw
report["replay_tip_height_at_grasp_m"] = [round(float(v), 4) for v in tip[:, 2]]
# 2) replay from the recorded start
s0 = torch.tensor(layout.init_joint_deg, device=dev).expand(env.n, 6).clone()
s0[:n] = torch.tensor(np.stack([eps[e]["state"][0] for e in chosen]), device=dev)
env.reset(joint_deg=s0, brick_pose=brick)
print("REPLAY_START", json.dumps({"s0_env0": np.round(s0[0].cpu().numpy(), 1).tolist(),
      "meas_env0_after_reset": np.round(env.measured(noisy=False)[0].cpu().numpy(), 1).tolist(),
      "brick_xyyaw_env0": np.round(brick[0].cpu().numpy(), 3).tolist(),
      "tip_at_grasp_env0": np.round(tip[0].cpu().numpy(), 3).tolist()}), flush=True)
T = max(len(eps[e]["action"]) for e in chosen)
acts = torch.zeros(env.n, T, 6, device=dev)
states = np.full((n, T, 6), np.nan, np.float32)
for k, e in enumerate(chosen):
    a = eps[e]["action"]
    acts[k, :len(a)] = torch.tensor(a, device=dev)
    acts[k, len(a):] = torch.tensor(a[-1], device=dev)
    states[k, :len(a)] = eps[e]["state"]
sim_s = np.zeros((n, T, 6), np.float32)
g_any = np.zeros(n, bool)
lifted_any = np.zeros(n, bool)
view_at_insp = np.zeros(n, bool)
hold = [[] for _ in range(n)]
max_z = np.zeros(n)
t0 = time.perf_counter()
for t in range(T):
    goal = s0.clone()
    goal[:] = acts[:, t]
    _, _, _, info = env.step(goal_override=goal)
    s = env.measured(noisy=False)[:n].cpu().numpy()
    sim_s[:, t] = s
    bad = ~np.isfinite(s).all(1) | (np.abs(s[:, :5]) > 200).any(1) | (np.abs(s[:, 5]) > 150)
    if bad.any() and not getattr(env, "_reported_bad", False):
        k = int(np.nonzero(bad)[0][0])
        env._reported_bad = True
        print("BLOWUP", json.dumps({"tick": t, "env": k, "episode": chosen[k], "state": np.round(s[k], 1).tolist(),
                                    "goal": np.round(acts[k, t].cpu().numpy(), 1).tolist(),
                                    "ref_r": np.round(env.r[k].cpu().numpy(), 1).tolist(),
                                    "recorded_state": np.round(states[k, min(t, T - 1)], 1).tolist(),
                                    "n_bad": int(bad.sum())}), flush=True)
    g = info["grasped"][:n].cpu().numpy()
    g_any |= g
    lifted_any |= info["lifted"][:n].cpu().numpy()
    z = (env.brick.data.root_pos_w - scene.env_origins)[:n, 2].cpu().numpy()
    max_z = np.maximum(max_z, z)
    for k, e in enumerate(chosen):
        ins = eps[e]["inspection_frame"]
        if t == ins:
            view_at_insp[k] = bool(info["in_view"][k])
        if eps[e]["grasp_frame"] + 10 <= t <= ins:
            hold[k].append(float(s[k, 5]))
wall = time.perf_counter() - t0
err = []
for k, e in enumerate(chosen):
    L = len(eps[e]["action"])
    err.append(sim_s[k, :L] - states[k, :L])
err = np.concatenate(err, 0)
rmse = np.sqrt(np.nanmean(err ** 2, 0))
report["replay"] = {
    "episodes": chosen, "n": n,
    "tracking_rmse_deg_sim_vs_real": dict(zip(J, np.round(rmse, 2).tolist())),
    "a1_model_rmse_deg_real_heldout": {"shoulder_pan": 1.54, "shoulder_lift": 2.38, "elbow_flex": 1.82,
                                       "wrist_flex": 0.61, "wrist_roll": 0.54, "gripper": 1.60},
    "grasp_detected_frac": float(g_any.mean()), "lifted_3cm_frac": float(lifted_any.mean()),
    "brick_max_height_m": np.round(max_z, 3).tolist(),
    "in_inspect_view_at_mark_frac": float(view_at_insp.mean()),
    "gripper_hold_median_sim": float(np.median([np.median(h) for h in hold if h])) if any(hold) else None,
    "gripper_hold_median_real": 5.75, "calib": env.profile.name, "data": args.data, "grasp_rule": args.grasp_rule,
    "ticks": T, "wall_s": round(wall, 1), "ticks_per_s_all_envs": round(T * env.n / wall, 1)}
print("REPLAY", json.dumps(report["replay"]), flush=True)
(out / "validate.json").write_text(json.dumps(report, indent=1))
np.savez(out / "replay_traces.npz", sim=sim_s, real=states, episodes=np.array(chosen))
print("VALIDATE_DONE", flush=True)
app.close()
