"""Grasp gate: do real demos that picked the brick on the robot also pick it when replayed in the twin?

    sim/twin/run.sh sim/twin/grasp_gate/gate.py --rung 3 --n 60          # the benchmark (PASS at >= 80 % lifts)
    sim/twin/run.sh sim/twin/grasp_gate/gate.py --rung 1 --n 60 --tag r1  # see the ladder below

Same held-out cal2 demos at every rung (GraspEnv's test pool, seeded draw of --n). One env per demo, the demo runs
from its first frame to 2.5 s after the gripper-close frame. Success = the brick is >= 3 cm above its start height
and within 4 cm of the grasp point for >= 0.5 s.

Ladder (what each rung isolates):
  rung 1  arm joints written to the recorded MEASURED states (no servo model), gripper = recorded action through the
          servo model (the jaw must stall on the brick); brick at the rung-0 pose (top camera, rung0.py).
          -> finger geometry + contact, with the best available brick pose.
  rung 2  as rung 1, brick at the FK grasp point of the close frame (the replay protocol used so far).
          -> rung 1 - rung 2 = brick-placement artefact.
  rung 3  recorded ACTIONS through the servo model (clamp off, as validate.py), brick as --place (default rung0).
          -> rung 1 - rung 3 = servo / drift mismatch.

Candidate fixes (prototypes, applied in this harness only):
  --set key=json        physics overrides (sim/twin/physics/params.json keys), e.g. gripper.collider=pads_cad,
                        gripper.finray.material.static_friction=1.2, gripper.finray.effort_nm=0.5
  --tip_dz -0.008       lower the whole arm trajectory by 8 mm at the grasp point (IK on pan / lift / elbow)
  --tip_z_target 0.002  oracle height: per demo, the lowest fin-tip point around the close frame is put 2 mm above
                        the table; --dz_reach c0,c1 lowers the arm by c0 + c1 * reach (a sag / offset model)
  --tip_dxy_grip a,b    shift the trajectory in the gripper's x (jaw axis) / y at the close frame (m)
  --joint_offsets f     joint offsets json (align/joint_fit) instead of the profile's
  --grip_shift k        gripper action k ticks later (+) / earlier (-); --grip_scale, --servo_k, --arm_bias
  --brick_off_grip a,b  move the brick in the gripper's x / y of the close frame (m): placement sensitivity
Outputs: sim/twin/out/grasp_gate/<tag>/{result.json, trace.npz}; one GATE line on stdout.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

ap = argparse.ArgumentParser()
ap.add_argument("--rung", type=int, default=3, choices=[1, 2, 3])
ap.add_argument("--n", type=int, default=60)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tag", default="")
ap.add_argument("--place", default="", choices=["", "fk", "rung0"], help="default: rung 2 fk, rungs 1 and 3 rung0")
ap.add_argument("--rung0", default=str(REPO / "sim/twin/out/grasp_gate/rung0/rung0.json"))
ap.add_argument("--kind", default="auto", choices=["auto", "auto3", "2x2"],
                help="auto = the rung-0 brick kind (rung0 place; 2x4 as the 2x4 box, others 2x2); auto3 also puts the "
                     "curved 2x3 as the env's curved 2x2")
ap.add_argument("--labels", default="", help="brick kind / resting pose labels json (brick_labels.json): the target is "
                                             "spawned in the labelled pose (2x4 as the 2x4 box, curved as 2x2)")
ap.add_argument("--label_map", default="", help="'a=b,...': treat labelled pose a as b (e.g. side_x=side_y)")
ap.add_argument("--label_poses", default="up,down,side_x,side_y", help="which labelled poses to apply (others stay up)")
ap.add_argument("--rung0_cam", default="joe", choices=["joe", "fit"], help="top-camera model of the rung-0 brick pose")
ap.add_argument("--sag", default="",
                help="'a,b,c' rad per N m: lift / elbow / wrist_flex += sag * static gravity torque "
                                          "(joint fit's model; torques from sag_tau.npz, sagtau.py)")
ap.add_argument("--drive", default="", choices=["", "kin", "dyn"])
ap.add_argument("--set", action="append", default=[], help="physics override key=json (repeatable)")
ap.add_argument("--joint_offsets", default="")
ap.add_argument("--tip_dz", type=float, default=0.0)
ap.add_argument("--tip_z_target", type=float, default=None,
                help="oracle: per demo, shift the trajectory so the FK fin tip's lowest point around the close frame "
                     "is this high above the table (m)")
ap.add_argument("--dz_reach", default="", help="'c0,c1': lower the grasp point by c0 + c1 * reach (m, m/m) every frame")
ap.add_argument("--tip_dxy_grip", default="0,0")
ap.add_argument("--brick_off_grip", default="0,0")
ap.add_argument("--grip_shift", type=int, default=0)
ap.add_argument("--grip_scale", type=float, default=1.0, help="scale the gripper action below 15 units")
ap.add_argument("--servo_k", type=float, default=1.0, help="scale the servo model's first-order gain")
ap.add_argument("--arm_bias", type=float, default=1.0, help="scale the servo model's steady-state bias")
ap.add_argument("--play", type=int, default=0, help="1 = GraspEnv's fitted joint-play model (sim/twin/dynamics/play.py)")
ap.add_argument("--distractors", type=int, default=0, help="GraspEnv clutter bricks (0 = the gate's single target)")
ap.add_argument("--plates", type=int, default=0, help="1 = plates on the table")
ap.add_argument("--clamp", type=float, default=1000.0)
ap.add_argument("--post", type=int, default=75, help="ticks after the close frame")
ap.add_argument("--pre", type=int, default=0, help="start this many ticks before the close frame (0 = first frame)")
ap.add_argument("--num_envs", type=int, default=64)
ap.add_argument("--pass_at", type=float, default=0.8)
args = ap.parse_args()
if args.set:
    os.environ["TWIN_PHYSICS_SET"] = ";".join([o for o in (os.environ.get("TWIN_PHYSICS_SET", ""), *args.set) if o])
place = args.place or ("fk" if args.rung == 2 else "rung0")
drive = args.drive or ("dyn" if args.rung == 3 else "kin")

from sim.twin.rl import grasp_env as G

app = G.launch()

import numpy as np
import torch

from sim.twin.grasp_gate import common as C
from sim.twin.grasp_gate.metrics import analyse, summarise

demos = C.load_demos()
eps = C.pick_test(demos, args.n, args.seed)
N = min(len(eps), args.num_envs)
# the gate's protocol: one target, no clutter / plates, no play model unless asked (GraspEnv's defaults for RL training
# moved to clutter + plates + the fitted play model on 2026-11-15; those are factors here, not the baseline)
gkw = {k: v for k, v in (("play", bool(args.play)), ("distractors", args.distractors), ("plates", bool(args.plates)))
       if k in G.GraspCfg.__dataclass_fields__}
cfg = G.GraspCfg(num_envs=N, max_relative_target=args.clamp, meas_noise=False, errors=False,
                 joint_offsets_json=args.joint_offsets, horizon_s=60.0, **gkw)
env = G.make(cfg)
dev = env.device
assert env.ep_ids == demos["keep"], "demo list differs from GraspEnv's pool"
row = {e: i for i, e in enumerate(env.ep_ids)}
env.K = env.K * args.servo_k
env.bias[:5] = env.bias[:5] * args.arm_bias
phys = sys.modules.get("sim.twin.physics.apply")
finray = bool(phys and phys.finray())
if finray:
    from sim.twin.gripper import model as fin

    fin_tip = [float(v) for v in fin.geometry()["fixed_tip"]]
    jaw_tip = [float(v) for v in fin.geometry()["jaw_tip"]]
else:
    fin_tip = [-0.0079, 0.0, -0.1044]
    jaw_tip = [0.0, 0.0, 0.0]
jaw_body = env.robot.body_names.index("moving_jaw_so101_v1_link")
jaw_tip_t = torch.tensor(jaw_tip, device=dev)
fin_tip_t = torch.tensor(fin_tip, device=dev)
r0 = json.loads(Path(args.rung0).read_text())["episodes"] if place == "rung0" else {}
dxy_tip = [float(v) for v in args.tip_dxy_grip.split(",")]
boff = [float(v) for v in args.brick_off_grip.split(",")]
ARM = env.jidx[:5]
labels = json.loads(Path(args.labels).read_text())["episodes"] if args.labels else {}
use_poses = set(args.label_poses.split(","))
pose_map = dict(kv.split("=") for kv in args.label_map.split(",") if kv)
_S = math.sqrt(0.5)
# brick-local rotation (xyzw) and centre height above the table (m) per (kind index, pose); the 2x2 hull is
# body-centred with 4.4 mm studs on +z, the 2x4 is a plain box (local x = long side, across the jaw after reset_envs)
POSE = {(0, "up"): ((0, 0, 0, 1), 0.0096), (0, "down"): ((1, 0, 0, 0), 0.0140),
        (0, "side_x"): ((0, _S, 0, _S), 0.01585), (0, "side_y"): ((-_S, 0, 0, _S), 0.01585),
        (1, "up"): ((0, 0, 0, 1), 0.0096), (1, "down"): ((1, 0, 0, 0), 0.0096),
        (1, "side_x"): ((-_S, 0, 0, _S), 0.0159), (1, "side_y"): ((-_S, 0, 0, _S), 0.0159)}


def qmul(a, b):
    """xyzw quaternion product a * b (torch, (..., 4))."""
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], -1)


def apply_labels(ids, pad, kind, yaw):
    """Re-place the target in its labelled resting pose (after reset_envs)."""
    o = env.scene.env_origins
    for k, e in enumerate(pad):
        lab = labels.get(str(e))
        pose_name = pose_map.get(lab["pose"], lab["pose"]) if lab else None
        if lab is None or pose_name not in use_poses or pose_name == "up":
            continue
        kk = int(kind[k])
        ql, h = POSE[(kk, pose_name)]
        qy = torch.tensor([0.0, 0.0, math.sin(float(yaw[k]) / 2), math.cos(float(yaw[k]) / 2)], device=dev)
        q = qmul(qy, torch.tensor(ql, dtype=torch.float32, device=dev))
        z = float(env.table_z) + h + 0.0005
        pose = torch.cat([env.b0[k, :2] + o[k, :2], torch.tensor([z + float(o[k, 2])], device=dev), q])[None]
        env.objs[kk].write_root_pose_to_sim(pose, env_ids=ids[k:k + 1])
        env.objs[kk].write_root_velocity_to_sim(torch.zeros(1, 6, device=dev), env_ids=ids[k:k + 1])
        env.b0[k, 2] = z
        n_labelled[0] += k < len(pad)
n_fallback, n_labelled = [0], [0]
sag = None
if args.sag:
    sag = torch.tensor([float(v) for v in args.sag.split(",")], device=dev)
    st = np.load(C.OUT / "sag_tau.npz")
    assert st["episodes"].tolist() == env.ep_ids, "sag_tau.npz: demo list differs"
    L0 = env.A.shape[1]
    tau_a, tau_s = (torch.tensor(st[k][:, :L0], device=dev) for k in ("tau_action", "tau_state"))
kin_fin = type(env.kin)(env.profile.joint_offsets, dev, tip_local=tuple(fin_tip))


def wrap(a, period):
    return (a + period / 2) % period - period / 2


def shift_traj(X, R_close, dz_env):
    """Move the grasp point of every frame: tip_dz + the per-demo dz_env + the reach model (world z), and
    tip_dxy_grip (close-frame gripper x / y)."""
    if args.tip_dz == 0 and dxy_tip == [0.0, 0.0] and dz_env is None and not args.dz_reach:
        return X
    d = (R_close[:, :, 0] * dxy_tip[0] + R_close[:, :, 1] * dxy_tip[1])[:, None, :].expand(-1, X.shape[1], -1).clone()
    d[..., 2] += args.tip_dz
    if dz_env is not None:
        d[..., 2] += dz_env[:, None]
    p, _ = env.kin.fk(X)
    if args.dz_reach:
        c0, c1 = (float(v) for v in args.dz_reach.split(","))
        d[..., 2] -= c0 + c1 * torch.linalg.norm(p[..., :2], dim=-1)
    return env.kin.ik_tip(X, p + d, iters=4)


def run_batch(batch):
    m = len(batch)
    pad = batch + [batch[0]] * (N - m)
    ei = torch.tensor([row[e] for e in pad], device=dev)
    cf = env.CF[ei]
    start = (cf - args.pre).clamp(min=0) if args.pre else torch.zeros_like(cf)
    end = torch.minimum(cf + args.post, env.LEN[ei] - 1)
    T = int((end - start).max()) + 1
    kind = torch.zeros(N, dtype=torch.long, device=dev)
    dxy = torch.zeros(N, 2, device=dev)
    dyaw = torch.zeros(N, device=dev)
    yc = env.yaw_close[ei]
    if place == "rung0":
        for k, e in enumerate(pad):
            r = r0.get(str(e))
            if r is None:  # no brick region in the top image: FK placement
                n_fallback[0] += k < m
                continue
            r = dict(r, **{q: r[args.rung0_cam][q] for q in ("kind", "xy", "yaw_long")}) if args.rung0_cam in r else r
            if args.kind in ("auto", "auto3") and r["kind"] == "2x4":
                kind[k] = 1
            if args.kind == "auto3" and r["kind"] == "2x3" and env.objs[2] is not None:
                kind[k] = 2
            dxy[k] = torch.tensor(r["xy"], device=dev) - env.tip_close[ei[k], :2]
            if kind[k] == 1:  # reset_envs adds pi/2 for the 2x4 (long side across the jaw axis)
                dyaw[k] = wrap(r["yaw_long"] - float(yc[k]) - math.pi / 2, math.pi)
            else:
                dyaw[k] = wrap(r["yaw_long"] - float(yc[k]), math.pi / 2)
    if labels:
        for k, e in enumerate(pad):
            lab = labels.get(str(e))
            if lab is not None and args.kind != "2x2":
                kind[k] = 1 if lab["kind"] == "2x4" else 0
    dxy[:, 0] += torch.cos(yc) * boff[0] - torch.sin(yc) * boff[1]
    dxy[:, 1] += torch.sin(yc) * boff[0] + torch.cos(yc) * boff[1]
    z = torch.zeros(N, device=dev)
    ids = torch.arange(N, device=dev)
    params = {"ep": ei, "pre": cf - start, "dxy": dxy, "dyaw": dyaw, "kind": kind,
              "err_p": torch.zeros(N, 3, device=dev),
              "err_yaw": z, "hover": z.bool(), "gshift": z.long(), "errors": z.bool()}
    _, R_close = env.kin.fk(env.S[ei, cf])
    dz_env = None
    if args.tip_z_target is not None:  # FK fin-tip height (above the table) around the close frame
        w = (cf[:, None] + torch.arange(-10, 25, device=dev)).clamp(min=0)
        w = torch.minimum(w, env.LEN[ei, None] - 1)
        zf = kin_fin.fk(env.S[ei[:, None], w])[0][..., 2] - env.table_z
        dz_env = args.tip_z_target - zf.min(1).values
    A = shift_traj(env.A[ei].clone(), R_close, dz_env)
    S = shift_traj(env.S[ei].clone(), R_close, dz_env)
    if sag is not None:  # recorded joints -> sagged joints (deg)
        A[..., 1:4] += torch.rad2deg(sag * tau_a[ei])
        S[..., 1:4] += torch.rad2deg(sag * tau_s[ei])
    shifted = args.tip_dz != 0 or dxy_tip != [0.0, 0.0] or dz_env is not None or bool(args.dz_reach) or sag is not None
    L = A.shape[1]
    gi = (torch.arange(L, device=dev)[None, :] - args.grip_shift).clamp(0, L - 1).expand(N, L)
    g = torch.gather(env.A[ei][..., 5], 1, gi)
    A[..., 5] = torch.where(g < 15, g * args.grip_scale, g)
    env.reset_envs(ids, params)
    if labels:
        apply_labels(ids, batch, kind, yc + dyaw + (kind == 1).float() * (math.pi / 2))
    if shifted:  # start from the shifted recorded state
        q = env.robot.data.default_joint_pos.clone()
        q[:, env.jidx] = env.to_sim(S[ids, start])
        env.robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        env.robot.set_joint_position_target(q)
        env.r[:] = S[ids, start]
    rec = {k: [] for k in ("q", "bp", "bq", "rel", "tip", "fin", "jfin", "goal")}
    for t in range(T):
        f = torch.minimum(start + t, end)
        f0 = torch.minimum(start + max(t - 1, 0), end)
        goal = A[ids, f]
        if drive == "dyn":
            env.apply_goal(goal)
            env.physics()
        else:
            s0, s1 = env.to_sim(S[ids, f0])[:, :5], env.to_sim(S[ids, f])[:, :5]
            env.apply_goal(torch.cat([S[ids, f][:, :5], goal[:, 5:]], -1))
            tgt = env.robot.data.joint_pos_target.clone()
            tgt[:, ARM] = s1
            env.robot.set_joint_position_target(tgt)
            vel = (s1 - s0) * env.cfg.control_hz
            for k in range(env.substeps):
                q = env.robot.data.joint_pos.clone()
                qd = env.robot.data.joint_vel.clone()
                q[:, ARM] = s0 + (s1 - s0) * (k + 1) / env.substeps
                qd[:, ARM] = vel
                env.robot.write_joint_state_to_sim(q, qd)
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(1.0 / env.cfg.physics_hz)
        env.t += 1
        o = env.scene.env_origins
        gp, gq = env.robot.data.body_pos_w[:, env.grip_body], env.robot.data.body_quat_w[:, env.grip_body]
        rec["q"].append(env.measured(noisy=False)[:m].cpu())
        rec["bp"].append((env.bpos_w() - o)[:m].cpu())
        rec["bq"].append(env.bquat_w()[:m].cpu())
        rec["rel"].append(env.priv()[:m].cpu())
        rec["tip"].append((env.tip_w() - o)[:m].cpu())
        rec["fin"].append((gp + G._quat_apply_xyzw(gq, fin_tip_t.expand(N, 3)) - o)[:m].cpu())
        jp, jq = env.robot.data.body_pos_w[:, jaw_body], env.robot.data.body_quat_w[:, jaw_body]
        rec["jfin"].append((jp + G._quat_apply_xyzw(jq, jaw_tip_t.expand(N, 3)) - o)[:m].cpu())
        rec["goal"].append(goal[:m].cpu())
    out = {k: torch.stack(v, 1).numpy() for k, v in rec.items()}
    out.update(start=start[:m].cpu().numpy(), cf=cf[:m].cpu().numpy(), end=end[:m].cpu().numpy(),
               kind=kind[:m].cpu().numpy(), b0=env.b0[:m].cpu().numpy(), yaw_close=yc[:m].cpu().numpy(),
               tip_close=env.tip_close[ei[:m]].cpu().numpy(),
               s_real=S[ids[:, None], (start[:, None] + torch.arange(T, device=dev)).clamp(max=L - 1)][:m]
               .cpu().numpy())
    return out


t_wall = time.perf_counter()
parts = [run_batch(eps[i:i + N]) for i in range(0, len(eps), N)]
T = max(p["q"].shape[1] for p in parts)
tr = {}
for k in parts[0]:
    arrs = [p[k] for p in parts]
    if arrs[0].ndim >= 2 and k not in ("b0", "tip_close"):
        arrs = [np.concatenate([a, np.repeat(a[:, -1:], T - a.shape[1], 1)], 1) for a in arrs]
    tr[k] = np.concatenate(arrs, 0)
tr["episodes"] = np.array(eps)
tr["fin_tip_local"] = np.array(fin_tip)
tr["table_z"] = np.array(env.table_z)
tag = args.tag or f"rung{args.rung}"
out = C.OUT / tag
out.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out / "trace.npz", **tr)
per = analyse(tr)
res = summarise(per)
res.update(grasp_cfg={k: getattr(cfg, k, None) for k in ("play", "distractors", "plates", "max_horizon_s")},
           rung=args.rung, drive=drive, place=place, n_fk_fallback=n_fallback[0], n_labelled=n_labelled[0],
           joint_offsets_deg=[round(math.degrees(float(v)), 2) for v in env.profile.joint_offsets[:5]], tag=tag,
           finray=finray,
           profile=env.profile.name,
           physics_set=os.environ.get("TWIN_PHYSICS_SET", ""), tip_local=[round(float(v), 4) for v in env.tip_local],
           args={k: v for k, v in vars(args).items() if k not in ("tag",)},
           wall_s=round(time.perf_counter() - t_wall, 1))
if phys is not None:  # the effective physics (other agents change params.json / the gripper model while sweeps run)
    import hashlib

    pp = phys._p()
    res["physics"] = {"gripper": pp.get("gripper"), "table": pp.get("table"), "bricks_material": pp["bricks"]["material"],
                      "md5": {f: hashlib.md5((REPO / "sim/twin" / f).read_bytes()).hexdigest()[:10]
                              for f in ("physics/params.json", "physics/apply.py", "gripper/model.py")}}
C.dump(out / "result.json", {"summary": res, "episodes": per})
print("GATE", json.dumps(res), flush=True)
print(f"GATE_{'PASS' if res['success'] >= args.pass_at else 'FAIL'} rung {args.rung} success {res['success']:.1%} "
      f"(95% CI {res['success_ci95']}) of {res['n']} held-out demos, threshold {args.pass_at:.0%}", flush=True)
app.close()
