"""Checks for the grasp env: kin.fk vs PhysX gripper / fingertip poses, IK accuracy, recorded lift timing, one
base-only batch with throughput.

    sim/twin/run.sh sim/twin/rl/check_kin.py [--num_envs 64]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from sim.twin.rl import grasp_env as G  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--num_envs", type=int, default=64)
args = ap.parse_args()
app = G.launch()

import torch  # noqa: E402

cfg = G.GraspCfg(num_envs=args.num_envs)
t0 = time.time()
env = G.make(cfg)
print(f"build {time.time() - t0:.1f} s", flush=True)
n, d = env.n, env.device
out = {}

# 1) FK vs PhysX at recorded states (zero velocity, after a physics step at the held targets)
gen = torch.Generator(device=d).manual_seed(0)
idx = torch.randint(len(env.ep_ids), (n,), device=d, generator=gen)
tt = (torch.rand(n, device=d, generator=gen) * (env.LEN[idx] - 1).float()).long()
q = env.S[idx, tt]
env.reset(joint_deg=q)
env.physics()
meas = env.measured(noisy=False)
p_fk, R_fk = env.kin.fk(meas)
p_sim = env.tip_w() - env.scene.env_origins
from env import _quat_apply_xyzw  # noqa: E402

qg = env.robot.data.body_quat_w[:, env.grip_body]
x_sim = _quat_apply_xyzw(qg, torch.tensor([1.0, 0, 0], device=d).expand(n, 3))
err = torch.linalg.norm(p_fk - p_sim, dim=-1)
ang = torch.rad2deg(torch.acos((R_fk[:, :, 0] * x_sim).sum(-1).clamp(-1, 1)))
out["fk_vs_physx_tip_err_mm"] = {"median": round(float(err.median() * 1e3), 3), "max": round(float(err.max() * 1e3), 3)}
out["fk_vs_physx_xaxis_deg"] = {"median": round(float(ang.median()), 3), "max": round(float(ang.max()), 3)}

# 2) IK: move the tip by a random 2 cm offset and check
dp = (torch.rand(n, 3, device=d, generator=gen) * 2 - 1) * 0.02
q2 = env.kin.ik_tip(meas, p_fk + dp)
e2 = torch.linalg.norm(env.kin.fk(q2)[0] - (p_fk + dp), dim=-1)
out["ik_2cm_residual_mm"] = {"median": round(float(e2.median() * 1e3), 3),
                             "p95": round(float(e2.quantile(0.95) * 1e3), 3)}

# 3) recorded data: fingertip height at the close frame, lift timing after it (FK of recorded states)
E = len(env.ep_ids)
z_close = env.tip_close[:, 2]
lift_t = []
for e in range(E):
    L, cf = int(env.LEN[e]), int(env.CF[e])
    pz = env.kin.fk(env.S[e, cf:L])[0][:, 2]
    k = torch.nonzero(pz > pz[0] + 0.035).flatten()
    lift_t.append(int(k[0]) if len(k) else 999)
lt = torch.tensor(lift_t).float()
out["recorded"] = {"episodes": E, "tip_z_at_close_mm_median": round(float(z_close.median() * 1e3), 1),
                   "tip_z_at_close_mm_p10_p90": [round(float(z_close.quantile(.1) * 1e3), 1),
                                                 round(float(z_close.quantile(.9) * 1e3), 1)],
                   "ticks_close_to_lift35mm_median": float(lt.median()),
                   "ticks_close_to_lift35mm_p90": float(lt.quantile(0.9)),
                   "frac_lift_within_45_ticks": float((lt <= 45).float().mean())}
print("CHECK", json.dumps(out), flush=True)

# 4) base-only batch (clean / with errors) + throughput
for errors in (False, True):
    gen = torch.Generator(device=d).manual_seed(1)
    P = env.sample_params(n, gen, "test", errors=errors)
    env.reset_envs(torch.arange(n, device=d), P)
    t1 = time.time()
    while not bool(env.done.all()):
        env.obs()
        env.candidates(1)
        env.step_chunk(torch.zeros(n, dtype=torch.long, device=d), torch.zeros(n, 6, device=d))
    wall = time.time() - t1
    oc = env.outcome()
    res = {G.FAIL[k]: round(float((oc == k).float().mean()), 3) for k in range(5)}
    res.update(errors=errors, centred=round(float(env.centred.float().mean()), 3),
               env_ticks_per_s=round(env.max_ticks * n / wall, 1))
    print("BASE", json.dumps(res), flush=True)
print("CHECK_DONE", flush=True)
app.close()
