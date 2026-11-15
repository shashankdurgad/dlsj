"""Diagnostic: clean replay base on unperturbed bricks; fingertip vs brick at the recorded close tick.

    sim/twin/run.sh sim/twin/rl/diag_base.py [--clamp 10] [--num_envs 128]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from sim.twin.rl import grasp_env as G  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--num_envs", type=int, default=128)
ap.add_argument("--clamp", type=float, default=10.0)
ap.add_argument("--perturb", type=int, default=0)
args = ap.parse_args()
app = G.launch()
import torch  # noqa: E402

cfg = G.GraspCfg(num_envs=args.num_envs, max_relative_target=args.clamp, brick_mix=(1.0, 0.0, 0.0))
env = G.make(cfg)
n, d = env.n, env.device
gen = torch.Generator(device=d).manual_seed(1)
P = env.sample_params(n, gen, "test", errors=False)
if not args.perturb:
    P["dxy"].zero_()
    P["dyaw"].zero_()
env.reset_envs(torch.arange(n, device=d), P)
pre = P["pre"]
rel_at_close = torch.zeros(n, 3, device=d)
while not bool(env.done.all()):
    env.obs()
    env.candidates(1)
    for _ in range(1):
        pass
    t_before = env.t.clone()
    env.step_chunk(torch.zeros(n, dtype=torch.long, device=d), torch.zeros(n, 6, device=d))
    hit = (t_before < pre) & (env.t >= pre)  # close tick inside this chunk (approx: at the chunk end)
    rel = env.priv()
    rel_at_close = torch.where(hit[:, None], rel, rel_at_close)
oc = env.outcome()
res = {G.FAIL[k]: round(float((oc == k).float().mean()), 3) for k in range(5)}
r = rel_at_close * 1e3
res["brick_in_gripper_frame_at_close_mm"] = {"median": r.median(0).values.round().tolist(),
                                            "p10": r.quantile(0.1, 0).round().tolist(),
                                            "p90": r.quantile(0.9, 0).round().tolist()}
ok = oc == 0
res["success_rel_median_mm"] = r[ok].median(0).values.round().tolist() if ok.any() else None
res["fail_rel_median_mm"] = r[~ok].median(0).values.round().tolist()
res.update(clamp=args.clamp, perturb=args.perturb, centred=float(env.centred.float().mean()))
print("DIAG", json.dumps(res), flush=True)
app.close()
