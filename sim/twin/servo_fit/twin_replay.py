"""Open-loop replay of real commanded sequences (real.py's windows.npz) in the twin, batched: env e replays window
e % W with parameter set e // W, from the window's first real reading. Each tick: read env.measured(noisy=True)
(what the controller sees), then env.apply_goal(cmd[t]) (the env's own LeRobot clamp + servo model) and physics.
A 2x2 brick sits at the FK fingertip of the real reading where the real window has a brick close (else parked).

    NAME=dlsj-twin-servo sim/twin/run.sh sim/twin/servo_fit/twin_replay.py --env grasp --play none --tag g_v1
    ... --env grasp --play nominal --sets sim/twin/servo_fit/sets.json:sweep_grip --tag g_grip
    ... --env twin --n_windows 8 --tag twin

--env grasp: sim/twin/rl/grasp_env.make (plates off, no distractors) with --play none (GraspCfg default v1) |
  nominal (PlayCfg(), the v3 / v4 presets without per-env DR) | dr (PlayCfg(dr=True), as the presets).
--env twin: TwinEnv (num_envs = W x sets, legacy bricks, no props, its own play install).
Parameter sets (OPT-IN overrides, JSON list of dicts; {} = the env as is), per joint lists of 6 (null = keep):
  K (first-order K / tick), vmax_s (units / s), delay (ticks, command FIFO before the clamp), stiffness (N m/rad),
  damping (N m s/rad), effort (N m), vel_limit_deg_s (PhysX joint velocity limit), clamp (max_relative_target).
-> sim/twin/out/servo_fit/<tag>/replay.npz + result.json (per set: per-joint reading RMSE vs real, paired gripper
   close timings, metrics.py on the twin readings with --fit).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "sim/residual_env"))

ap = argparse.ArgumentParser()
ap.add_argument("--env", default="grasp", choices=["grasp", "twin"])
ap.add_argument("--play", default="none", choices=["none", "nominal", "dr"])
ap.add_argument("--play_delay", type=int, default=0, help="PlayCfg.controller_delay (1 tick on params delay_joints)")
ap.add_argument("--meas_noise", type=int, default=-1, help="-1: the env default (grasp v1 on; play off)")
ap.add_argument("--windows", default=str(REPO / "sim/twin/out/servo_fit/windows.npz"))
ap.add_argument("--n_windows", type=int, default=0)
ap.add_argument("--sets", default="", help="file.json[:key] or inline JSON list")
ap.add_argument("--tag", required=True)
ap.add_argument("--fit", action="store_true")
ap.add_argument("--T", type=int, default=0)
args = ap.parse_args()

if args.env == "grasp":
    from sim.twin.rl import grasp_env as G

    G.launch()
else:
    from sim.twin.env.twin_env import launch_app

    launch_app(True, cameras=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sim.twin.servo_fit import metrics as M  # noqa: E402

OUT = REPO / "sim/twin/out/servo_fit" / args.tag
OUT.mkdir(parents=True, exist_ok=True)


def load_sets(spec: str) -> list[dict]:
    if not spec:
        return [{}]
    if spec.strip().startswith("["):
        return json.loads(spec)
    f, _, key = spec.partition(":")
    d = json.loads(Path(f).read_text())
    return d[key] if key else d


Z = np.load(args.windows)
names, srcs, CMD, MEAS = Z["name"], Z["src"], Z["cmd"].astype(np.float64), Z["meas"].astype(np.float64)
if args.n_windows:
    # spread over sources: take every k-th
    idx = np.linspace(0, len(names) - 1, args.n_windows).round().astype(int)
    names, srcs, CMD, MEAS = names[idx], srcs[idx], CMD[idx], MEAS[idx]
if args.T:
    CMD, MEAS = CMD[:, :args.T], MEAS[:, :args.T]
W, T = CMD.shape[:2]
SETS = load_sets(args.sets)
P = len(SETS)
n = W * P
print(f"[servo_fit] {W} windows x {P} sets = {n} envs, {T} ticks, env {args.env} play {args.play}", flush=True)

# ------------------------------------------------------------------ env
if args.env == "grasp":
    cfg = G.GraspCfg(num_envs=n, plates=False, distractors=0, play=False)
    if args.meas_noise >= 0:
        cfg.meas_noise = bool(args.meas_noise)
    elif args.play != "none":
        cfg.meas_noise = False
    env = G.make(cfg)
    if args.play != "none":
        from sim.twin.dynamics.play import PlayCfg, install

        install(env, PlayCfg(dr=args.play == "dr", controller_delay=bool(args.play_delay)))
    kin = env.kin
else:
    from sim.twin.env.twin_env import TwinEnv, TwinEnvCfg
    from sim.twin.rl.kin import Kin

    tcfg = TwinEnvCfg(num_envs=n, bricks="legacy", props=False, randomize=False)
    if args.meas_noise >= 0:
        tcfg.meas_noise = bool(args.meas_noise)
    tw = TwinEnv(tcfg)
    env = tw.residual
    kin = Kin(tw.profile.joint_offsets, env.device)
d = env.device
rob = env.robot
jid = list(env.jidx)
print("[servo_fit] drive: stiffness", rob.data.joint_stiffness[0, jid].tolist() if hasattr(rob.data, "joint_stiffness")
      else "?", "damping", rob.data.joint_damping[0, jid].tolist() if hasattr(rob.data, "joint_damping") else "?",
      "effort", rob.data.joint_effort_limits[0, jid].tolist() if hasattr(rob.data, "joint_effort_limits") else "?",
      "K", env.K.tolist(), "vmax/tick", env.vmax.tolist(), "bias", env.bias.tolist(),
      "play", getattr(env, "play", None) is not None, "noise", env.noise.tolist(), flush=True)


def _vec(v):
    return torch.as_tensor(v, dtype=torch.float32, device=d)


def _tensor_attr(name):
    t = getattr(rob.data, name, None)
    if t is None:
        return None
    try:
        import warp as wp

        if not isinstance(t, torch.Tensor):
            t = wp.to_torch(t)
    except Exception:  # noqa: BLE001
        pass
    return t


# per-env servo parameters (env.K / env.vmax become (n, 6); env.apply_goal / play read them per call)
K = env.K.expand(n, 6).clone()
V = env.vmax.expand(n, 6).clone()
DEL = torch.zeros(n, 6, dtype=torch.long, device=d)
CL = torch.full((n, 6), float(env.cfg.max_relative_target), device=d)
TELE = np.array([str(x) == "teleop" for x in srcs])
CL[torch.as_tensor(np.tile(TELE, P), device=d)] = 1000.0  # the teleop recordings ran without LeRobot's clamp
drive = {k: _tensor_attr(a) for k, a in (("stiffness", "joint_stiffness"), ("damping", "joint_damping"),
                                         ("effort", "joint_effort_limits"), ("vel", "joint_vel_limits"))}
print("[servo_fit] drive tensors:", {k: (None if v is None else v[0, jid].tolist()) for k, v in drive.items()},
      flush=True)
writes = {"stiffness": [], "damping": [], "effort": [], "vel_limit_deg_s": []}
for p, s in enumerate(SETS):
    e = slice(p * W, (p + 1) * W)
    for j in range(6):
        if s.get("K") and s["K"][j] is not None:
            K[e, j] = s["K"][j]
        if s.get("vmax_s") and s["vmax_s"][j] is not None:
            V[e, j] = s["vmax_s"][j] / 30.0
        if s.get("delay") and s["delay"][j] is not None:
            DEL[e, j] = int(s["delay"][j])
        if s.get("clamp") and s["clamp"][j] is not None:
            CL[e, j] = s["clamp"][j]
    for k in writes:
        if s.get(k):
            writes[k].append((p, s[k]))
env.K, env.vmax = K, V
if bool((CL != env.cfg.max_relative_target).any()):
    env.cfg.max_relative_target = CL  # broadcast in torch.minimum / maximum
for k, lst in writes.items():
    if not lst:
        continue
    base = drive[{"stiffness": "stiffness", "damping": "damping", "effort": "effort", "vel_limit_deg_s": "vel"}[k]]
    full = base[:, jid].clone() if base is not None else torch.zeros(n, 6, device=d)
    for p, v in lst:
        for j in range(6):
            if v[j] is not None:
                full[p * W:(p + 1) * W, j] = math.radians(v[j]) if k == "vel_limit_deg_s" else v[j]
    ids = torch.arange(n, device=d)
    if k == "stiffness":
        rob.write_joint_stiffness_to_sim_index(stiffness=full, joint_ids=jid, env_ids=ids)
    elif k == "damping":
        rob.write_joint_damping_to_sim_index(damping=full, joint_ids=jid, env_ids=ids)
    elif k == "effort":
        rob.write_joint_effort_limit_to_sim_index(limits=full, joint_ids=jid, env_ids=ids)
    else:
        rob.write_joint_velocity_limit_to_sim_index(limits=full, joint_ids=jid, env_ids=ids)
    print(f"[servo_fit] wrote {k} for sets {[p for p, _ in lst]}", flush=True)

# ------------------------------------------------------------------ reset: real start reading, brick at a brick close
wid = np.arange(n) % W
cmd = torch.tensor(CMD[wid], dtype=torch.float32, device=d)  # (n, T, 6)
j0 = torch.tensor(MEAS[wid, 0], dtype=torch.float32, device=d)
real_ev = [M.gripper_events(CMD[w], MEAS[w]) for w in range(W)]
bpose = torch.tensor([[0.85, 0.45, 0.0]] * n, dtype=torch.float32, device=d)
has_brick = np.zeros(W, bool)
for w in range(W):
    br = [e for e in real_ev[w] if e["kind"] == "brick"]
    if br:
        e = br[0]
        k = min(T - 1, e["on"] + (e["t_settled"] or 20))
        p, R = kin.fk(torch.tensor(MEAS[w, k][None], dtype=torch.float32, device=d))
        yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0])
        for q in range(P):
            bpose[q * W + w] = torch.stack([p[0, 0], p[0, 1], yaw[0]])
        has_brick[w] = True
ids = torch.arange(n, device=d)
env.reset(env_ids=ids, brick_pose=bpose, joint_deg=j0)
park = torch.zeros(n, 7, device=d)
for k, obj in enumerate(getattr(env, "distractors", [])):  # no distractors in the replays
    park[:, 0], park[:, 1], park[:, 2], park[:, 6] = 0.9 + 0.08 * k, -0.6, 0.02, 1.0
    p7 = park.clone()
    p7[:, :3] += env.scene.env_origins
    obj.write_root_pose_to_sim(p7, env_ids=ids)
    obj.write_root_velocity_to_sim(torch.zeros(n, 6, device=d), env_ids=ids)

# ------------------------------------------------------------------ replay
Dmax = int(DEL.max().item())
fifo = cmd[:, :1].expand(n, Dmax + 1, 6).clone()  # fifo[:, k] = command of k ticks ago
ys = torch.zeros(n, T, 6, device=d)
qs = torch.zeros(n, T, 6, device=d)
rs = torch.zeros(n, T, 6, device=d)
gsel = DEL.unsqueeze(1)  # (n, 1, 6)
for t in range(T):
    ys[:, t] = env.measured(noisy=True)
    qs[:, t] = env.measured(noisy=False)
    fifo = torch.cat([cmd[:, t:t + 1], fifo[:, :-1]], 1)
    g = torch.gather(fifo, 1, gsel).squeeze(1)
    env.apply_goal(g)
    env.physics()
    rs[:, t] = env.r
    if t % 100 == 0:
        print(f"[servo_fit] tick {t}/{T}", flush=True)
Y, Q, Rr = ys.cpu().numpy().astype(np.float64), qs.cpu().numpy(), rs.cpu().numpy()
np.savez_compressed(OUT / "replay.npz", meas=Y.astype(np.float32), q=Q.astype(np.float32), r=Rr.astype(np.float32),
                    wid=wid, names=names, srcs=srcs, has_brick=has_brick)


# ------------------------------------------------------------------ comparison
def paired_closes(w, y):
    out = []
    for e in real_ev[w]:
        on, end = e["on"], e["end"]
        seg = y[on:end + 1, 5]
        fin = float(np.median(y[max(on, end - 5):end + 1, 5]))
        b15 = np.nonzero(seg < 15)[0]
        ch = np.nonzero(seg < seg[0] - 1.0)[0]
        ok = np.abs(seg - fin) <= 0.3
        st = next((i for i in range(len(seg) - 4) if ok[i:i + 5].all()), None)
        out.append({"kind": e["kind"], "real": [e["t_first"], e["t_lt15"], e["t_settled"], e["final"]],
                    "twin": [int(ch[0]) if len(ch) else None, int(b15[0]) if len(b15) else None, st, round(fin, 2)]})
    return out


def med(v):
    v = [x for x in v if x is not None]
    return round(float(np.median(v)), 2) if v else None


res = {"args": vars(args), "sets": SETS, "W": W, "T": T, "per_set": []}
for p, s in enumerate(SETS):
    e = np.arange(p * W, (p + 1) * W)
    Ys = Y[e]
    err = Ys - MEAS
    vel = np.diff(Ys, axis=1) - np.diff(MEAS, axis=1)
    r = {"set": s, "rmse": dict(zip(M.JOINTS, np.sqrt((err ** 2).mean((0, 1))).round(3).tolist(), strict=True)),
         "rmse_vel": dict(zip(M.JOINTS, np.sqrt((vel ** 2).mean((0, 1))).round(3).tolist(), strict=True)),
         "bias": dict(zip(M.JOINTS, err.mean((0, 1)).round(3).tolist(), strict=True))}
    by_src = {}
    for sname in sorted(set(srcs.tolist())):
        k = srcs == sname
        by_src[sname] = dict(zip(M.JOINTS, np.sqrt((err[k] ** 2).mean((0, 1))).round(3).tolist(), strict=True))
    r["rmse_by_src"] = by_src
    pc = []
    for w in range(W):
        pc += paired_closes(w, Ys[w])
    gc = {}
    for kind in ("empty", "brick"):
        q = [x for x in pc if x["kind"] == kind]
        if not q:
            continue
        gc[kind] = {"n": len(q)}
        for i, nm in enumerate(("t_first", "t_lt15", "t_settled", "final")):
            gc[kind][nm] = {"real": med([x["real"][i] for x in q]), "twin": med([x["twin"][i] for x in q]),
                            "twin_minus_real_p50": med([x["twin"][i] - x["real"][i] for x in q
                                                        if x["twin"][i] is not None and x["real"][i] is not None])}
    r["gripper_close_paired"] = gc
    if args.fit:
        tr_, te_ = ~TELE, TELE
        r["twin_metrics"] = {"trace": M.all_metrics(list(CMD[tr_]), list(Ys[tr_]), fit=True),
                             "teleop": M.all_metrics(list(CMD[te_]), list(Ys[te_]), fit=True, clamp=1e9)}
    res["per_set"].append(r)
    print(f"[servo_fit] set {p} {json.dumps(s)}\n  rmse {r['rmse']}\n  rmse_vel {r['rmse_vel']}", flush=True)
    print(f"  grip {json.dumps(gc)}", flush=True)
if args.fit:
    res["real_metrics"] = {"trace": M.all_metrics(list(CMD[~TELE]), list(MEAS[~TELE]), fit=True),
                           "teleop": M.all_metrics(list(CMD[TELE]), list(MEAS[TELE]), fit=True, clamp=1e9)}
(OUT / "result.json").write_text(json.dumps(res, indent=1, default=float))
print("[servo_fit] wrote", OUT / "result.json", flush=True)
import os  # noqa: E402

os._exit(0)
