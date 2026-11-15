"""Validate the twin env (TwinEnv + TwinGym).

    sim/twin/run.sh sim/twin/env/validate_twin.py [--only motion,replay,grid,fps,gym] [--episodes 12 76 190]

(a) motion: the cameras see the arm move (image change after a large move), and no render lag (a second render of
    the same physics state is identical to the first, also mid-motion);
(b) replay: open-loop replay of recorded cal2 actions (the residual env's validate.py protocol: brick placed under
    the fingertip at the frame the gripper closes, replay clamp off, no measurement noise): joint tracking RMSE vs
    the recorded states, grasp / lift success, real | sim contact sheets per camera;
(c) grid: 8 resets with randomisation on (top + wrist); restore: a DR-off reset after DR matches the nominal render;
(d) fps: env steps/s with all cameras rendered every step (and without rendering);
(e) gym: TwinGym reset / step, Observation fields and shapes.
Outputs: sim/twin/out/env/ (validate_twin.json, motion.jpg, replay_ep<NNN>.jpg, reset_grid.jpg).
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
OUT = REPO / "sim/twin/out/env"
DATA = REPO / "sim/twin/data/canonical/duplo_teleop_cal2_rgb_456"

ap = argparse.ArgumentParser()
ap.add_argument("--only", default="motion,replay,grid,restore,fps,gym")
ap.add_argument("--episodes", type=int, nargs="*", default=None, help="replay episodes (default: 3 drawn, seed 0)")
ap.add_argument("--n_episodes", type=int, default=8)
ap.add_argument("--sheets", type=int, default=3, help="contact sheets for the first N replayed episodes")
ap.add_argument("--sheet_frames", type=int, default=6)
ap.add_argument("--no-park", dest="park", action="store_false", help="keep the sampled plates / distractors in replay")
ap.add_argument("--fps_steps", type=int, default=150)
ap.add_argument("--no-physics", dest="physics", action="store_false", help="without sim/twin/physics/apply.py")
ap.add_argument("--tag", default="", help="suffix for the report / image names")
ap.add_argument("--width", type=int, default=640)
ap.add_argument("--height", type=int, default=480)
args = ap.parse_args()
only = set(args.only.split(","))

from sim.twin.env.twin_env import TwinEnv, TwinEnvCfg  # noqa: E402

env = TwinEnv(TwinEnvCfg(width=args.width, height=args.height, physics_hooks=args.physics))  # launches Isaac
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

OUT.mkdir(parents=True, exist_ok=True)
J = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
report_path = OUT / f"validate_twin{args.tag}.json"
report = json.loads(report_path.read_text()) if report_path.exists() else {}
report["config"] = {"width": args.width, "height": args.height, "cameras": list(env.cfg.cameras),
                    "randomizers": [type(r).__name__ for r in env.randomizers],
                    "physics_hooks": env._physics.__name__ if env._physics else None, "profile": env.profile.name}


def label(img, text):
    img = np.ascontiguousarray(img)
    cv2.putText(img, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
    cv2.putText(img, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return img


def small(img, w=320):
    return cv2.resize(img, (w, int(img.shape[0] * w / img.shape[1])), interpolation=cv2.INTER_AREA)


def save(path, rgb):
    cv2.imwrite(str(path), np.ascontiguousarray(rgb)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])


def diff(a, b):
    d = np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(-1)
    return {"mean_abs": round(float(d.mean()), 3), "pct_px_gt20": round(float((d > 20).mean() * 100), 3)}


def dump():
    report_path.write_text(json.dumps(report, indent=1))


# ------------------------------------------------------------------ (a) motion + render lag
if "motion" in only:
    from bases import load_episodes

    e0 = load_episodes(DATA, episodes=[0])[0]  # cal2 episode 0: start -> held at the inspect camera
    s_ep, insp0 = e0["state"], int(e0["inspection_frame"])
    obs0 = env.reset(seed=0, joint_deg=s_ep[0])
    s0 = obs0["state"].copy()
    goal = s_ep[insp0]
    lag_mid = {}
    for t in range(60):
        obs, *_ = env.step(goal, render=(t == 10))
        if t == 10:  # mid-motion: is a re-render of the same physics state identical (no one-frame lag / ghosting)?
            again = env.render()
            lag_mid = {c: diff(obs["images"][c], again[c]) for c in again}
    obs1 = env.step(goal, render=True)[0]  # one render after 50 unrendered ticks (a big jump)
    again = env.render()
    report["motion"] = {
        "state_before": np.round(s0, 1).tolist(), "goal": np.round(goal, 1).tolist(),
        "state_after": np.round(obs1["state"], 1).tolist(),
        "image_change_after_move": {c: diff(obs0["images"][c], obs1["images"][c]) for c in obs1["images"]},
        "rerender_same_state_after_jump": {c: diff(obs1["images"][c], again[c]) for c in again},
        "rerender_same_state_mid_motion": lag_mid}
    rows = [np.concatenate([label(small(obs0["images"][c]), f"{c} before"),
                            label(small(obs1["images"][c]), f"{c} after move")], 1) for c in obs1["images"]]
    save(OUT / "motion.jpg", np.concatenate(rows, 0))
    print("MOTION", json.dumps(report["motion"]), flush=True)
    dump()

# ------------------------------------------------------------------ (b) open-loop replay
if "replay" in only:
    from bases import load_episodes
    from env import _quat_apply_xyzw
    import subprocess
    import tempfile

    def real_frames(ep, idxs):
        """Recorded frames, decoded in a subprocess: importing PyAV in the Isaac process blacks out the RTX
        cameras (measured: all-zero renders when `av` is imported before the first render)."""
        with tempfile.TemporaryDirectory() as td:
            code = ("import sys, numpy as np; sys.path.insert(0, %r); from sim.twin.visuals.realdata import Dataset; "
                    "ds = Dataset(%r); np.savez(%r, **{c: np.stack(ds.frames(%d, c, %r)) for c in "
                    "('top', 'wrist', 'inspect')})" % (str(REPO), str(DATA), td + "/f.npz", ep, list(idxs)))
            subprocess.run([sys.executable, "-c", code], check=True)
            z = np.load(td + "/f.npz")
            return {c: list(z[c]) for c in z.files}

    eps = load_episodes(DATA)

    def close_frame(grip, insp, thr=12.0, stay=15.0, open_min=15.0):  # residual validate.py --grasp_rule gripper
        g = grip[:insp] if insp else grip
        if len(g) == 0 or g.max() < open_min:
            return None
        for f in range(1, len(g)):
            if g[f - 1] > thr >= g[f] and (g[f:] <= stay).all():
                return f
        return None

    for d in eps.values():
        d["grasp_frame"] = close_frame(d["state"][:, 5], d["inspection_frame"])
    ok = sorted(e for e, d in eps.items() if not d["excluded"] and (d["grasp_frame"] or 0) > 5 and d["inspection_frame"])
    chosen = args.episodes or sorted(np.random.default_rng(0).choice(ok, size=args.n_episodes, replace=False).tolist())
    R = env.residual
    saved = (R.cfg.max_relative_target, R.noise.clone())
    R.cfg.max_relative_target = 1000.0  # teleop had no clamp on the follower
    R.noise.zero_()
    per_ep, all_err = [], []
    for ep in chosen:
        d = eps[ep]
        A, S = d["action"], d["state"]
        T, gf, insp = len(A), int(d["grasp_frame"]), int(d["inspection_frame"])
        # 1) fingertip + jaw yaw at the recorded close frame -> brick pose
        env.reset(seed=ep, joint_deg=S[gf], brick_pose=(0.8, 0.5, 0.0), render=False)
        tip = (R.tip_w() - env.scene.env_origins)[0]
        q = R.robot.data.body_quat_w[:1, R.grip_body]
        xa = _quat_apply_xyzw(q, torch.tensor([[1.0, 0, 0]], device=env.device))[0]
        brick = (float(tip[0]), float(tip[1]), float(torch.atan2(xa[1], xa[0])))
        # 2) replay from the recorded start
        env.reset(seed=ep, joint_deg=S[0], brick_pose=brick, render=False)
        if args.park:  # the recorded actions were for a different layout: clear plates / distractors off the path
            o = env.scene.env_origins[:1]
            for k2, obj in enumerate(R.distractors + R.bowl_objs):
                pose = torch.tensor([[0.88, -0.55 + 0.12 * k2, 0.02, 0, 0, 0, 1.0]], device=env.device)
                pose[:, :3] += o
                obj.write_root_pose_to_sim(pose)
                obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device))
            env.settle(2)
        k = args.sheet_frames
        sheet_t = sorted(set(int(v) for v in np.clip([0, gf - 20, gf + 15, (gf + insp) // 2, insp,
                                                      insp + (T - insp) // 2][:k], 0, T - 1)))
        if len(per_ep) >= args.sheets:
            sheet_t = []
        sim_imgs, sim_s = {}, np.zeros((T, 6), np.float32)
        g_any = l_any = False
        dmin, grip_after = 1e9, []
        zmax, in_view_insp = 0.0, None
        t0 = time.perf_counter()
        for t in range(T):
            obs, rew, term, trunc, info = env.step(A[t], render=t in sheet_t)
            sim_s[t] = env.state(noisy=False)
            g_any |= bool(info["grasped"])
            l_any |= bool(info["lifted"])
            zmax = max(zmax, info["brick_pos"][2])
            dmin = min(dmin, float(info["dist"]))
            if gf + 10 <= t <= insp:
                grip_after.append(float(sim_s[t, 5]))
            if t == insp:
                in_view_insp = bool(info["in_view"])
            if t in sheet_t:
                sim_imgs[t] = obs["images"]
        wall = time.perf_counter() - t0
        err = sim_s - S[:T]
        all_err.append(err)
        rmse = np.sqrt((err ** 2).mean(0))
        res = {"episode": int(ep), "ticks": T, "grasp_frame": gf, "inspection_frame": insp,
               "brick_xy_yaw": np.round(brick, 3).tolist(), "tip_z_at_grasp": round(float(tip[2]), 4),
               "rmse_deg": dict(zip(J, np.round(rmse, 2).tolist())), "grasped": g_any, "lifted_3cm": l_any,
               "brick_max_z_m": round(zmax, 3), "min_tip_brick_dist_m": round(dmin, 4),
               "gripper_median_while_held_sim": round(float(np.median(grip_after)), 2) if grip_after else None,
               "gripper_median_while_held_real": round(float(np.median(S[gf + 10:insp + 1, 5])), 2), "in_inspect_view_at_mark": in_view_insp,
               "final_info": {k2: env.last_info.get(k2) for k2 in ("placed", "phase", "bowl")},
               "recorded_bowl": d.get("bowl"), "wall_s": round(wall, 1)}
        per_ep.append(res)
        print("REPLAY_EP", json.dumps(res), flush=True)
        if not sheet_t:
            continue
        # contact sheet: per camera, real row over sim row
        real = real_frames(ep, sheet_t)
        blocks = []
        for c in ("top", "wrist", "inspect"):
            if c not in sim_imgs[sheet_t[0]]:
                continue
            rr = np.concatenate([label(small(real[c][i], 256), f"real {c} t={t}") for i, t in enumerate(sheet_t)], 1)
            ss = np.concatenate([label(small(sim_imgs[t][c], 256), f"sim {c} t={t}") for t in sheet_t], 1)
            blocks += [rr, ss, np.full((6, rr.shape[1], 3), 255, np.uint8)]
        save(OUT / f"replay_ep{ep:03d}{args.tag}.jpg", np.concatenate(blocks, 0))
    R.cfg.max_relative_target, R.noise = saved[0], saved[1]
    e = np.concatenate(all_err, 0)
    report["replay"] = {
        "episodes": [int(v) for v in chosen], "per_episode": per_ep,
        "rmse_deg_all": dict(zip(J, np.round(np.sqrt((e ** 2).mean(0)), 2).tolist())),
        "grasp_frac": float(np.mean([r["grasped"] for r in per_ep])),
        "lift_frac": float(np.mean([r["lifted_3cm"] for r in per_ep])),
        "residual_env_reference_20ep_joe_cal2": {"rmse": [1.06, 1.55, 1.31, 1.02, 0.74, 2.30], "lift": 0.8},
        "protocol": "residual validate.py: brick at FK fingertip of the gripper-close frame, clamp off, no noise"
                    + ("; plates + distractors parked at the far table edge" if args.park else "")}
    print("REPLAY", json.dumps({k: report["replay"][k] for k in ("episodes", "rmse_deg_all", "grasp_frac",
                                                                  "lift_frac")}), flush=True)
    dump()

# ------------------------------------------------------------------ (c) randomised reset grid
if "grid" in only:
    tiles = []
    for s in range(8):
        obs = env.reset(seed=100 + s, randomize=True)
        tiles.append(np.concatenate([label(small(obs["images"]["top"]), f"seed {100 + s} top  bowl "
                                           f"{env.residual.bowl_names[int(env.residual.bowl[0])]}"),
                                     label(small(obs["images"]["wrist"]), "wrist")], 0))
    grid = np.concatenate([np.concatenate(tiles[:4], 1), np.concatenate(tiles[4:], 1)], 0)
    save(OUT / "reset_grid.jpg", grid)
    obs_n = env.reset(seed=100, randomize=False)  # evaluation reset after DR: randomizers restored
    save(OUT / "reset_nominal_after_dr.jpg", np.concatenate([label(small(obs_n["images"]["top"]), "seed 100 DR off"),
                                                             label(small(obs_n["images"]["wrist"]), "wrist")], 1))
    report["grid"] = {"image": str(OUT / "reset_grid.jpg"), "randomizers": report["config"]["randomizers"],
                      "randomizer_errors": env.randomizer_errors}
    print("GRID ok", flush=True)
    dump()

# ------------------------------------------------------------------ (c2) DR on -> off restores the nominal look
if "restore" in only:
    a = env.reset(seed=100, randomize=False)["images"]
    env.reset(seed=100, randomize=True)
    b = env.reset(seed=100, randomize=False)["images"]
    report["restore"] = {c: diff(a[c], b[c]) for c in a}
    save(OUT / "restore_check.jpg", np.concatenate([np.concatenate([label(small(a[c]), f"{c} nominal"),
                                                                    label(small(b[c]), f"{c} after DR + restore")], 1)
                                                    for c in a], 0))
    print("RESTORE", json.dumps(report["restore"]), flush=True)
    dump()

# ------------------------------------------------------------------ (d) throughput
if "fps" in only:
    obs = env.reset(seed=1)
    s0 = obs["state"]
    res = {}
    for name, rend in (("render_all_cams", True), ("no_render", False)):
        n = args.fps_steps if rend else args.fps_steps * 2
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for t in range(n):
            a = s0 + np.array([10, 8, -8, 6, 15, 20], np.float32) * np.sin(t / 15.0)
            env.step(a, render=rend)
        torch.cuda.synchronize()
        res[name] = round(n / (time.perf_counter() - t0), 2)
    t0 = time.perf_counter()
    for _ in range(30):
        env.render()
    res["render_only_per_s"] = round(30 / (time.perf_counter() - t0), 2)
    res["note"] = f"num_envs 1, {args.width}x{args.height}, cameras {list(env.cfg.cameras)}, 4 PhysX substeps/tick"
    report["fps"] = res
    print("FPS", json.dumps(res), flush=True)
    dump()

# ------------------------------------------------------------------ (e) Studio gym
if "gym" in only:
    from sim.twin.env.studio_gym import TwinGym

    gym = TwinGym(env=env)
    ob, info = gym.reset(seed=3)
    a = ob.state[0].clone()
    a[5] += 20
    ob2, r, term, trunc, info2 = gym.step(a)
    ob3, *_ = gym.step(gym.sample_action().clamp(-30, 30))
    report["gym"] = {"images": {k: [list(v.shape), str(v.dtype), round(float(v.min()), 3), round(float(v.max()), 3)]
                                for k, v in ob2.images.items()},
                     "state": [list(ob2.state.shape), str(ob2.state.dtype)], "task": ob2.task,
                     "reward": r, "terminated": term, "truncated": trunc, "info_keys": sorted(info2),
                     "max_episode_steps": gym.get_max_episode_steps()}
    print("GYM", json.dumps(report["gym"]), flush=True)
    dump()

print("VALIDATE_TWIN_DONE", flush=True)
env.close()
