"""Closed-loop MolmoAct2 (frozen 8k fine-tune) pick episodes in TwinEnv with a per-tick log, for the close forensics
that real_forensics.py runs on the real controller traces (analyse with twin_forensics.py).

    sim/twin/run.sh sim/twin/failure_match/twin_eval.py --episodes 20 --start ready --horizon_s 12 --out fm_ready_s0
    sim/twin/run.sh sim/twin/failure_match/twin_eval.py --episodes 20 --start protocol --out fm_protocol_s0

Controller = the deploy config (image_task.ImageGraspTask: one MolmoAct2 pass per chunk (K = 1), 30-action chunks,
RTC re-plan when 15 remain, inference delay 10 ticks, LeRobot clamp 10, top crop 0,0,560,480, the trained
"pick up the <colour> brick" prompt). No residual.

--start ready    : the deploy start. The arm starts at the loop's ready pose (26.3,-51.8,30.1,73.1,15.9,37.7) and
                   the scene is TwinEnv's data-sampled one (target / distractor kinds, colours, counts, on-plate,
                   upside-down, plates from scene_stats.json). The prompt names the target's colour.
--start protocol : the RL agent's protocol (image_task.reset: a held-out cal2 episode, start 0.7-4 s before its
                   recorded close or at frame 0, target at the close-frame fingertip +-2.5 cm).
--scene_flip F   : override the upside-down fraction of the sampled scene (default: the scene stats').
Episodes end at success (target lifted >= 3 cm near the fingertip for 15 ticks, as image_task), a lifted distractor
(wrong brick, same rule) or the horizon; a push or a tipped brick does not end it (the real loop goes on too).

Writes sim/twin/out/failure_match/<out>/ep<NNN>.npz (per tick: measured, goal, gripper-link pose, target pose,
distractor positions; per request: tick, latency) and episodes.jsonl.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "sim/residual_env"))

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=2)
ap.add_argument("--snap", default="sim/twin/data/models/molmoact2_ft_8k/step008000.ckpt")
ap.add_argument("--start", default="ready", choices=("ready", "protocol"))
ap.add_argument("--horizon_s", type=float, default=12.0)
ap.add_argument("--delay", type=int, default=10)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--scene_flip", type=float, default=-1.0, help="upside-down fraction (<0: scene stats)")
ap.add_argument("--max_distractors", type=int, default=-1, help="cap the sampled distractor count (<0: stats)")
ap.add_argument("--min_distractors", type=int, default=3,
                help="floor of the sampled distractor count (clutter; 3 = the deploy-matched default, 0 = scene stats)")
ap.add_argument("--target_xy", default="sampled", choices=("sampled", "deploy"),
                help="sampled: the residual env's grasp-position distribution; deploy: the real deploy bricks "
                     "(out/failure_match/real_brick_xy.json from real_forensics.py) +- --xy_jitter")
ap.add_argument("--xy_jitter", type=float, default=0.01)
ap.add_argument("--target_flip", type=float, default=-1.0,
                help="--target_xy deploy: probability the target lies upside down (<0: scene stats' 0.4)")
ap.add_argument("--first_skip", type=int, default=0,
                help="deploy controller: the first chunk of an episode starts at this index (the real loop's first "
                     "inference runs while the arm waits, then drops its `stale` ~12 actions); 0 = image_task as is")
ap.add_argument("--max_step", type=float, default=0.0,
                help="deploy controller slew limit on the command, deg / tick (real: 4, gripper 8); 0 = off")
ap.add_argument("--play_cfg", default="", help='TwinEnvCfg.play_cfg as JSON, e.g. \'{"controller_delay": true}\'')
ap.add_argument("--drift_joint_deg", type=float, default=0.0,
                help="deployment drift: per-episode calibration error N(0, s) deg on pan / lift / elbow / wrist flex; "
                     "the prior sees measured + delta and its command lands at command - delta")
ap.add_argument("--drift_joints", default="0,1,2,3", help="joints the calibration drift applies to")
ap.add_argument("--drift_top_cm", type=float, default=0.0, help="per-episode top-camera shift U(+-v) cm per axis")
ap.add_argument("--drift_top_deg", type=float, default=0.0, help="per-episode top-camera tilt U(+-v) deg per axis")
ap.add_argument("--drift_wrist_deg", type=float, default=0.0, help="per-episode wrist-camera mount tilt U(+-v) deg")
ap.add_argument("--out", default="fm_eval")
args = ap.parse_args()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sim.twin.rl.image_task import ImageGraspTask  # noqa: E402
from sim.twin.rl.priors import FT_BRICK_COLOURS, FT_PICK, TWIN_PALETTE  # noqa: E402

READY = [26.3, -51.8, 30.1, 73.1, 15.9, 37.7]
INV_PALETTE = {v: k for k, v in TWIN_PALETTE.items()}
OUT = REPO / "sim/twin/out/failure_match" / args.out
OUT.mkdir(parents=True, exist_ok=True)
DEPLOY_XY = None
if args.target_xy == "deploy":
    DEPLOY_XY = [np.array(r["xy_m"]) for r in
                 json.loads((REPO / "sim/twin/out/failure_match/real_brick_xy.json").read_text())]


def patch_scene_stats(env):
    """Optional overrides of the sampled scene (upside-down fraction, distractor cap)."""
    st = getattr(env, "scene_stats", None)
    if st is None:
        return
    if args.scene_flip >= 0:
        st["upside_down_frac"] = float(args.scene_flip)
    if args.max_distractors >= 0 or args.min_distractors > 0:
        hi = args.max_distractors if args.max_distractors >= 0 else 99
        dc = {k: v for k, v in st["distractor_count"].items() if args.min_distractors <= int(k) <= hi}
        st["distractor_count"] = dc or {str(args.min_distractors): 1}


def reset_ready(task: ImageGraspTask, n_ep: int, seed: int):
    """Deploy start: ready pose, sampled scene, prompt = the target's colour (image_task.reset's bookkeeping)."""
    env, R = task.env, task.R
    task.reset("test", n_ep, start="frame0")  # every per-episode field of the current image_task, then our scene
    pose = None
    if args.target_xy == "deploy":
        xy = DEPLOY_XY[int(task.rng.integers(len(DEPLOY_XY)))] + task.rng.uniform(-args.xy_jitter, args.xy_jitter, 2)
        pose = (float(xy[0]), float(xy[1]), float(task.rng.uniform(-math.pi, math.pi)))
    env.reset(seed=seed, joint_deg=np.array(READY, np.float32), brick_pose=pose, render=False, randomize=False)
    if pose is not None:  # explicit pose: TwinEnv made the target a light-blue 2x2; recolour / flip it here
        from sim.twin.visuals.twin_scene import set_brick_color

        colour = FT_BRICK_COLOURS[int(task.rng.integers(len(FT_BRICK_COLOURS)))]
        set_brick_color(0, "Brick", TWIN_PALETTE[colour])
        env.scene_plan["target"]["color"] = TWIN_PALETTE[colour]
        flip = 0.4 if args.target_flip < 0 else args.target_flip
        if task.rng.random() < flip:
            flip_target(env)
            env.scene_plan["target"]["upside_down"] = True
        env.settle(4)
    if getattr(env, "play", None) is not None and hasattr(env.play, "sample"):
        env.play.sample(torch.zeros(1, dtype=torch.long, device=task.dev), R.gen)
    plan = env.scene_plan or {}
    tgt = plan.get("target", {})
    colour = INV_PALETTE.get(tgt.get("color"))
    if colour is None:  # orange / unknown target colour: not a trained prompt colour -> recolour the target
        from sim.twin.visuals.twin_scene import set_brick_color

        colour = FT_BRICK_COLOURS[int(task.rng.integers(len(FT_BRICK_COLOURS)))]
        set_brick_color(0, tgt.get("prim") or "Brick", TWIN_PALETTE[colour])
        plan["target"]["color"] = TWIN_PALETTE[colour]
    task.task = FT_PICK.format(colour=colour)
    task.pick = True
    task.ep, task.n_ep, task.start = seed, n_ep, "ready"
    task.T = int(args.horizon_s * 30)
    task.b0 = R.brick_local()[0].clone()
    task.plan = []
    task.t = task.hold = task.c_streak = task.fade = 0
    task.flags = dict.fromkeys(("success", "centred", "pushed", "tipped", "ever_grasp", "lost"), False)
    for k in ("o", "o_tgt", "o_hand", "prev_r", "sum_abs_o"):
        setattr(task, k, torch.zeros(6, device=task.dev))
    task.phi, task.latent, task.lat, task.black, task.n_pick = 0.0, None, [], 0, 0
    task.inj = None  # no failure injection: the prior's own behaviour
    o0 = env.scene.env_origins[0]
    if hasattr(task, "nb0"):
        nb0 = [(d.data.root_pos_w[0] - o0).clone() for d in R.distractors]
        task.nb0 = [p for p in nb0 if abs(float(p[0])) < 1.0 and abs(float(p[1])) < 1.0]


def flip_target(env):
    """Lay the target upside down (studs on the table) in place, as BrickPool.place does for a planned flip."""
    from sim.twin.scene.bricks import _quat

    r = env.residual
    p = r.brick.data.root_pose_w[:1].clone() if hasattr(r.brick.data, "root_pose_w") else \
        r.brick.data.root_state_w[:1, :7].clone()
    q = p[0, 3:7].detach().cpu().numpy()
    yaw = 2.0 * math.atan2(float(q[2]), float(q[3]))
    p[0, 3:7] = torch.tensor(_quat(yaw, True), device=env.device)
    r.brick.write_root_pose_to_sim(p)
    r.brick.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device))


def distractor_objs(env):
    cur = getattr(getattr(env, "bricks", None), "current", None) or {}
    return [(name, obj) for name, _d, obj in cur.get("distractors", [])]


if args.play_cfg:  # image_task builds its own TwinEnvCfg: inject play_cfg through the module attribute it imports
    import sim.twin.env.twin_env as _te

    _BaseCfg, _play = _te.TwinEnvCfg, json.loads(args.play_cfg)

    def _cfg(*a, **k):
        k.setdefault("play_cfg", _play)
        return _BaseCfg(*a, **k)

    _te.TwinEnvCfg = _cfg
task = ImageGraspTask(args.snap, 1, args.delay, 3.0, args.seed, None, "auto", detector="sim")
patch_scene_stats(task.env)
R, env, dev = task.R, task.env, task.dev
DRIFT = torch.zeros(6, device=dev)
_state0 = env.state


def _state_drift(noisy=None):  # what the prior (and the RTC state) sees: the drifted calibration
    return _state0(noisy) + DRIFT.cpu().numpy()


env.state = _state_drift


def _usd_cam(prim_path):
    import omni.usd
    from pxr import Usd, UsdGeom

    prim = omni.usd.get_context().get_stage().GetPrimAtPath(prim_path)
    return prim, UsdGeom.Xformable(prim).GetLocalTransformation(Usd.TimeCode.Default()) if prim else None


def _rot(rng, deg):
    from pxr import Gf

    m = Gf.Matrix4d(1.0)
    for ax in (Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1)):
        m = m * Gf.Matrix4d().SetRotate(Gf.Rotation(ax, float(rng.uniform(-deg, deg))))
    return m


TOP = None
if args.drift_top_cm > 0 or args.drift_top_deg > 0:
    _p = env.scene.sensors["top_cam"].cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
    _p = _p.replace("env_[^/]+", "env_0").replace("env_.*", "env_0")
    TOP = _usd_cam(_p)
    print("drift: top camera prim", _p, "found" if TOP[1] is not None else "MISSING", flush=True)
WRIST0 = env.sync.cams.get(0) if (args.drift_wrist_deg > 0 and hasattr(env, "sync")) else None


def apply_drift(rng):
    """Per-episode deployment drift (joint calibration, top-camera extrinsics, wrist mount); returns its draw."""
    from pxr import Gf, UsdGeom

    d = np.zeros(6)
    if args.drift_joint_deg > 0:
        for j in (int(x) for x in args.drift_joints.split(",")):
            d[j] = rng.normal(0, args.drift_joint_deg)
    DRIFT[:] = torch.tensor(d, dtype=torch.float32, device=dev)
    out = {"joint_deg": np.round(d[:4], 2).tolist()}
    if TOP is not None and TOP[1] is not None:
        m = _rot(rng, args.drift_top_deg) * TOP[1]
        t = m.ExtractTranslation() + Gf.Vec3d(*rng.uniform(-args.drift_top_cm, args.drift_top_cm, 3) / 100)
        m.SetTranslateOnly(t)
        UsdGeom.Xformable(TOP[0]).MakeMatrixXform().Set(m)
        out["top_dt_cm"] = np.round((np.array(t) - np.array(TOP[1].ExtractTranslation())) * 100, 2).tolist()
    if WRIST0 is not None:
        cam, bi, local0, inv = WRIST0
        env.sync.cams[0] = (cam, bi, _rot(rng, args.drift_wrist_deg) * local0, inv)
    return out


print("prior ready", json.dumps(task.prior.client.info), flush=True)
rows = []
for n_ep in range(args.episodes):
    seed = 100000 * (args.seed + 1) + n_ep
    if args.start == "ready":
        reset_ready(task, n_ep, seed)
    else:
        task.reset("test", n_ep)  # keeps the protocol's own horizon
    drift = apply_drift(task.rng) if (args.drift_joint_deg or TOP is not None or WRIST0 is not None) else {}
    if drift and n_ep == 0 and (TOP is not None or WRIST0 is not None):  # sanity: the cameras really moved
        im1 = env.render()
        TOPM = TOP[0].GetAttribute("xformOp:transform").Get() if TOP is not None else None
        if TOP is not None:
            from pxr import UsdGeom

            UsdGeom.Xformable(TOP[0]).MakeMatrixXform().Set(TOP[1])
        if WRIST0 is not None:
            wr = env.sync.cams[0]
            env.sync.cams[0] = WRIST0
        im0 = env.render()
        print("drift sanity: mean |image change| top %.2f wrist %.2f" % tuple(
            float(np.abs(im1[c].astype(float) - im0[c].astype(float)).mean()) for c in ("top", "wrist")), flush=True)
        if TOP is not None:
            UsdGeom.Xformable(TOP[0]).MakeMatrixXform().Set(TOPM)
        if WRIST0 is not None:
            env.sync.cams[0] = wr
    o = env.scene.env_origins[0]
    dis = distractor_objs(env)
    d0 = [obj.data.root_pos_w[0] - o for _, obj in dis]
    log = {k: [] for k in ("meas", "goal", "grip_p", "grip_q", "brick_p", "brick_q", "dis_p", "grasped")}
    req = []
    t_wall = time.perf_counter()
    outcome, hold, dhold = "timeout", 0, [0] * len(dis)
    pushed_t = None
    cmd_prev = None
    while task.t < task.T:
        if task.need_request():
            first = not task.plan
            C = task.request()
            task.install(C[0][args.first_skip:] if first and args.first_skip else C[0])
            req.append((task.t, task.prior.last_info[0]["latency_s"]))
        goal = task._apply(task.plan.pop(0))
        if args.max_step > 0:  # smolvla_control --max-step: cmd moves towards the target at <= max_step per tick
            if cmd_prev is None:
                cmd_prev = R.measured(noisy=False)[0].clone()
            lim = torch.tensor([args.max_step] * 5 + [2 * args.max_step], device=dev)
            goal = cmd_prev + (goal - cmd_prev).clamp(-lim, lim)
            cmd_prev = goal.clone()
        env.step((goal - DRIFT).cpu().numpy(), render=False)  # the commanded pose lands drift-shifted
        task.t += 1
        log["meas"].append(R.measured(noisy=False)[0].cpu().numpy())
        log["goal"].append(goal.cpu().numpy())
        log["grip_p"].append((R.robot.data.body_pos_w[0, R.grip_body] - o).cpu().numpy())
        log["grip_q"].append(R.robot.data.body_quat_w[0, R.grip_body].cpu().numpy())
        bp = R.brick_local()[0]
        log["brick_p"].append(bp.cpu().numpy())
        log["brick_q"].append(R.brick.data.root_quat_w[0].cpu().numpy())
        dp = [obj.data.root_pos_w[0] - o for _, obj in dis]
        log["dis_p"].append(np.array([p.cpu().numpy() for p in dp]).reshape(-1, 3))
        log["grasped"].append(bool(R.grasped()[0]))
        tip = R.tip_w()[0] - o
        if pushed_t is None and float(torch.linalg.norm(bp[:2] - task.b0[:2])) > 0.015 and \
                not (float(bp[2] - task.b0[2]) > 0.01):
            pushed_t = task.t
        lifted = float(bp[2] - task.b0[2]) > 0.03 and float(torch.linalg.norm(bp - tip)) < 0.03
        hold = hold + 1 if lifted else 0
        if hold >= 15:
            outcome = "success"
            break
        wrong = False
        for k, p in enumerate(dp):
            up = float(p[2] - d0[k][2]) > 0.03 and float(torch.linalg.norm(p - tip)) < 0.03
            dhold[k] = dhold[k] + 1 if up else 0
            wrong |= dhold[k] >= 15
        if wrong:
            outcome = "wrong_brick"
            break
    plan = env.scene_plan or {}
    q0 = log["brick_q"][0]
    up0 = 1 - 2 * (q0[0] ** 2 + q0[1] ** 2)  # body z . world z (xyzw)
    r = {"n": n_ep, "seed": seed, "start": task.start, "target_xy": args.target_xy,
         "first_skip": args.first_skip, "max_step": args.max_step, "play_cfg": args.play_cfg,
         "drift": drift, "episode": int(task.ep),
         "task": task.task, "outcome": outcome,
         "ticks": task.t, "pushed_t": pushed_t, "black_frames": task.black,
         "target": plan.get("target"), "n_distractors": len(dis), "distractor_prims": [n for n, _ in dis],
         "plates_on_table": plan.get("plates_on_table"), "upside_down": bool(up0 < 0),
         "brick0": [round(float(v), 4) for v in log["brick_p"][0]],
         "min_dist_distractor_m": round(min([float(np.linalg.norm(log["dis_p"][0][k][:2] - log["brick_p"][0][:2]))
                                             for k in range(len(dis))] or [9.9]), 4),
         "prior_calls": len(req), "prior_latency_s_median": round(float(np.median([x[1] for x in req])), 3),
         "wall_s": round(time.perf_counter() - t_wall, 1)}
    # plate distance from the target's start (the "near a plate" scenes)
    try:
        dpl = [float(torch.linalg.norm(R.bowl_xy_env[0, k] - task.b0[:2])) - float(R.bowl_r[k])
               for k in range(len(R.bowl_objs)) if abs(float(R.bowl_xy_env[0, k, 0])) < 1.0]
        r["plate_rim_dist_m"] = round(min(dpl), 4) if dpl else None
    except Exception as e:  # noqa: BLE001
        r["plate_rim_dist_m"] = f"n/a ({type(e).__name__})"
    rows.append(r)
    np.savez_compressed(OUT / f"ep{n_ep:03d}.npz", **{k: np.asarray(v) for k, v in log.items() if k != "dis_p"},
                        dis_p=np.asarray(log["dis_p"], dtype=np.float32), req=np.asarray(req, np.float32),
                        meta=json.dumps(r))
    with open(OUT / "episodes.jsonl", "a") as fh:
        fh.write(json.dumps(r) + "\n")
    print("FM_EP", json.dumps(r), flush=True)
oc = [r["outcome"] for r in rows]
print("FM_SUMMARY", json.dumps({"out": args.out, "n": len(oc), "success": oc.count("success"),
                                "outcomes": {k: oc.count(k) for k in set(oc)}}), flush=True)
task.close()
