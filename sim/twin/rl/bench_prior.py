"""Cost of best-of-K with MolmoAct2 (one prefix forward, K flow seeds batched) on real cal2 frames, the pooled latent,
and the CPU latency of the residual / critic MLPs. Plain Python (no Isaac app: PyAV decodes the frames here).

    sim/twin/run.sh sim/twin/rl/bench_prior.py [--snap PATH] [--episodes 3 4] [--reps 5]
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--snap", default=None)
ap.add_argument("--episodes", type=int, nargs="+", default=[3, 120])
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--out", default=str(REPO / "sim/twin/out/rl/bench_prior.json"))
ap.add_argument("--skip_model", type=int, default=0)
ap.add_argument("--task", default="pick up the blue brick", help="a prompt the model was trained on")
args = ap.parse_args()
res = {}

if not args.skip_model:
    from sim.twin.policies.molmoact2 import MolmoAct2Client
    from sim.twin.visuals.realdata import Dataset

    ds = Dataset(REPO / "sim/twin/data/canonical/duplo_teleop_cal2_rgb_456")
    obs_list = []
    for ep in args.episodes:
        S = ds.states(ep)
        idx = max(0, int(np.argmax(S[:, 5] > 0)) + 60)  # an approach frame
        obs_list.append({"images": {c: ds.frame(ep, c, idx) for c in ("wrist", "top")}, "state": S[idx],
                         "task": args.task})
    crops = {"top": (0, 0, 560, 480)} if args.snap and args.snap.endswith(".ckpt") else None
    pol = MolmoAct2Client(snap=args.snap, crops=crops)
    res["model"] = pol.info
    o = obs_list[0]
    for _ in range(2):
        pol.act(o, seed=0)
        pol.act_k(o, 4, [0, 1, 2, 3])
    t1 = []
    for _ in range(args.reps):
        pol.act(o, seed=0)
        t1.append(pol.last_latency_s)
    res["act_single_s"] = round(float(np.median(t1)), 4)
    for k in (1, 2, 4, 8):
        ts, tp = [], []
        for r in range(args.reps):
            a, z = pol.act_k(o, k, list(range(10 * r, 10 * r + k)))
            ts.append(pol.last_latency_s)
            tp.append(pol.last_info["prefix_s"])
        res[f"act_k{k}_s"] = round(float(np.median(ts)), 4)
        res[f"act_k{k}_prefix_s"] = round(float(np.median(tp)), 4)
        res["latent_dim"] = int(z.shape[0])
    # k = 1 seed s from act_k vs act(seed=s): same model, different noise generator -> compare spreads instead
    a4, _ = pol.act_k(o, 4, [0, 1, 2, 3])
    res["k4_candidate_spread_deg"] = np.round(a4[:, :15].std(0).mean(0), 3).tolist()
    tail = a4[0, 15:]
    ar, _ = pol.act_k(o, 4, [5, 6, 7, 8], prev_chunk=tail, inference_delay=10)
    res["rtc_k4_s"] = round(pol.last_latency_s, 4)
    res["rtc_k4_prefix_match_deg"] = round(float(np.abs(ar[:, :10] - tail[None, :10]).mean()), 3)
    res["rtc_k4_spread_deg"] = np.round(ar[:, :15].std(0).mean(0), 3).tolist()
    pol.close()

# CPU latency of the deployment MLPs (XPS-like: 4 threads)
import torch  # noqa: E402

sys.path.insert(0, str(REPO))
from sim.twin.rl.dice import Dice, DiceCfg  # noqa: E402

torch.set_num_threads(4)
cpu = {}
for name, s_dim, hidden, ens in (("state-based (obs 25 + 4 cand)", 25 + 4 * 30, (256, 256, 256), 10),
                                 ("MolmoAct2 (obs 25 + latent 128 + 4 cand x 15 steps)", 25 + 128 + 4 * 90,
                                  (256, 256), 10)):
    a_dim = 30 if "state" in name else 90
    m = Dice(s_dim, a_dim, DiceCfg(hidden=hidden, ensemble=ens), "cpu")
    for K in (1, 4):
        s = torch.randn(1, s_dim)
        A = torch.randn(1, K, a_dim)
        with torch.no_grad():
            for _ in range(20):
                m.pi(s, A[:, 0])
            t = time.perf_counter()
            for _ in range(200):
                m.pi(s, A[:, 0])
            t_res = (time.perf_counter() - t) / 200
            sub = Dice(s_dim, a_dim, DiceCfg(hidden=hidden, ensemble=2), "cpu")
            for nm, net in (("full10", m), ("heads2", sub)):
                for _ in range(10):
                    net.select(s, A)
                t = time.perf_counter()
                for _ in range(100):
                    net.select(s, A)
                cpu[f"{name} | K={K} | residual+select ({nm})"] = round((time.perf_counter() - t) / 100 * 1e3, 3)
        cpu[f"{name} | residual only"] = round(t_res * 1e3, 3)
res["cpu_ms_4threads"] = cpu
Path(args.out).write_text(json.dumps(res, indent=1))
print("BENCH", json.dumps(res), flush=True)
