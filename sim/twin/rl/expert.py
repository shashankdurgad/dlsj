"""RLPD expert rows for the state-based grasp residual, from the reviewed post-train set (pick segments only).

Rows have exactly the online rows' features (GraspEnv.obs + candidate features), computed without Isaac from the
recorded joint states / actions with the twin FK (sim/twin/rl/kin.py):
* candidates = the demo's own next-5 actions (a_base = a, z = 0 in the DICE-RL template), copied K_state times;
  stored residual r = 0;
* detector = the residual env's model (4 mm noise, 5 % dropout, FOV gate) on the brick pose implied by the demo:
  at the FK fingertip of the close frame until the close, then carried rigidly by the gripper;
* load proxy = recorded gripper state - previous commanded gripper; offsets / previous residual = 0;
* reward = the env's potential shaping on the carried brick height + 1 at a held 3 cm lift (15 ticks), terminal there.
Episodes: post-train v3.1 cal2 demos whose source episode is in the grasp env's TRAIN split (held-out test episodes
never enter the replay) + the pick-only takes; optional --min_accuracy on deep_review.json. Starts 21/26/31/36 ticks
before the close frame (the env's 0.7-1.2 s).

Image path (--latent_snap, the 8k .ckpt): every decision (5 ticks) carries the prior's pooled latent from its last
request (every 15 ticks, as the controller), computed on the REAL wrist / top frames. With --prior_chunk 1 the base
action is the PRIOR's chunk on those frames (deploy noise, candidate 0) and the stored residual is the correction
that turns it into the demo: r = clip(demo - prior, bounds) in the residual's own coordinates (fingertip offset in the
prior goal's gripper frame / res_xy_m, res_z_m; wrist flex / roll / 10 deg; gripper / 30), averaged over the 5
goals. The test pool (the 8k's validation episodes, image_task.FT_VALIDATION) never enters. --pre all: the whole pick
phase from frame 0. --shard i/n: every n-th episode from i (merge the .pt files with --merge).

    sim/twin/run.sh sim/twin/rl/expert.py --data sim/twin/data/posttrain_v3_1 --out sim/twin/out/rl/expert_v31.pt
    python sim/twin/rl/expert.py --latent_snap sim/twin/data/models/molmoact2_ft_8k/step008000.ckpt --prior_chunk 1 \
        --pre all --shard 0/8 --out sim/twin/out/rl/expert_img8k_s0.pt          # x 8 shards, then
    python sim/twin/rl/expert.py --merge "sim/twin/out/rl/expert_img8k_s*.pt" --out sim/twin/out/rl/expert_img8k.pt
"""
import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
for p in (str(REPO), str(REPO / "sim/residual_env")):
    sys.path.insert(0, p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sim.twin.rl.grasp_env import DATA, GraspCfg, close_frame, load_profile  # noqa: E402
from sim.twin.rl.kin import Kin  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(REPO / "sim/twin/data/posttrain_v3_1"))
ap.add_argument("--out", default=str(REPO / "sim/twin/out/rl/expert_v31.pt"))
ap.add_argument("--min_accuracy", type=int, default=0)
ap.add_argument("--k_state", type=int, default=4)
ap.add_argument("--pre", default="21,26,31,36")
ap.add_argument("--sources", default="duplo_teleop_cal2,duplo_pick")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--latent_snap", default="", help="visual prior (.ckpt): rows for train_image.py with the prior's "
                "pooled latent on the REAL wrist / top frames (state = obs + projected latent, no candidates)")
ap.add_argument("--proj_dim", type=int, default=128)
ap.add_argument("--max_episodes", type=int, default=0)
ap.add_argument("--prior_chunk", type=int, default=0, help="1 (with --latent_snap): a_base = the prior's chunk on the "
                "real frames, r = clip(demo - prior) (real corrective supervision); 0: a_base = demo, r = 0")
ap.add_argument("--res_xy_m", type=float, default=0.035)
ap.add_argument("--res_z_m", type=float, default=0.02)
ap.add_argument("--shard", default="0/1", help="i/n: episodes i, i+n, ... (parallel builds)")
ap.add_argument("--merge", default="", help="glob of shard .pt files to concatenate into --out (no model)")
args = ap.parse_args()


def near_summary(dl, res_xy_m, res_z_m) -> dict:
    """|demo - prior| at decisions within 0.5 s of the close: (m, 6) unclipped residual coordinates (m / deg / units)
    -> percentiles of the lateral fingertip offset (mm), z (mm), wrist (deg), gripper, and the clipped fraction."""
    import torch

    if len(dl) == 0:
        return {"n": 0}
    lat = dl[:, :2].norm(dim=-1) * 1000
    q = torch.tensor([0.5, 0.9, 0.99])

    def pct(x):
        return [round(float(v), 2) for v in torch.quantile(x.float(), q)]

    bound = torch.tensor([res_xy_m, res_xy_m, res_z_m, 10.0, 10.0, 30.0])
    return {"n": len(dl), "lateral_mm_p50_p90_p99": pct(lat), "z_mm_p50_p90_p99": pct(dl[:, 2].abs() * 1000),
            "wrist_flex_deg": pct(dl[:, 3].abs()), "wrist_roll_deg": pct(dl[:, 4].abs()),
            "gripper_units": pct(dl[:, 5].abs()),
            "clipped_frac_per_dim": [round(float(v), 3) for v in (dl.abs() > bound).float().mean(0)],
            "lateral_beyond_bound_frac": round(float((lat > res_xy_m * 1000).float().mean()), 3)}


if args.merge:
    import glob

    import torch

    parts = [torch.load(f, weights_only=False) for f in sorted(glob.glob(args.merge))]
    keys = [k for k in parts[0] if k not in ("stats", "nxt")]
    out, off, nx = {k: torch.cat([p[k] for p in parts]) for k in keys}, 0, []
    for p in parts:
        n = p["nxt"].clone()
        n[n >= 0] += off
        nx.append(n)
        off += len(p["s"])
    out["nxt"] = torch.cat(nx)
    out["stats"] = {"shards": [p["stats"] for p in parts], "rows": off}
    if "near_delta" in out:
        out["stats"]["near_close"] = near_summary(out["near_delta"], args.res_xy_m, args.res_z_m)
    torch.save(out, args.out)
    print("EXPERT_MERGED", json.dumps({"rows": off, "files": len(parts),
                                       "near_close": out["stats"].get("near_close")}), flush=True)
    sys.exit(0)
SHARD_I, SHARD_N = (int(v) for v in args.shard.split("/"))

from bases import load_episodes  # noqa: E402

cfg = GraspCfg()
prof = load_profile(cfg)
dev = "cpu"
if cfg.physics:  # the same fingertip point as the env (physics layer: fin-ray grasp point)
    from sim.twin.physics import apply as _phys

    _phys.load_params()
    TIP = tuple(float(v) for v in _phys.tip_local())
else:
    from sim.twin.rl.kin import TIP_LOCAL as TIP
kin = Kin(prof.joint_offsets, dev, tip_local=TIP)
H, T = cfg.chunk, int(round(cfg.horizon_s * 30))
KS = args.k_state
TABLE_Z, BRICK_H = -0.002, 0.0192
REST_Z = TABLE_Z + BRICK_H / 2 + 0.0005
gen = torch.Generator().manual_seed(args.seed)
client = frames = proj = None
if args.latent_snap:  # plain Python process: PyAV decodes the real frames, the model runs in its worker
    from sim.twin.policies.molmoact2 import MolmoAct2Client
    from sim.twin.rl.image_task import DEPLOY_NOISE_SEED, FT_VALIDATION
    from sim.twin.rl.priors import FT_PICK, FT_TOP_CROP, LatentProjector, brick_colour_word
    from sim.twin.visuals.realdata import Dataset

    if not args.latent_snap.endswith(".ckpt"):
        raise ValueError("--latent_snap must be the cal2 fine-tune .ckpt (top crop + fine-tune prompts)")
    client = MolmoAct2Client(snap=args.latent_snap, crops={"top": FT_TOP_CROP})
    frames = Dataset(args.data)
    KS = 0
    # image path: the test pool is the 8k's validation set (post-train source_episode = canonical cal2 index, checked:
    # all 441 cal2 source episodes have the canonical episode's length)
    test_src = set(FT_VALIDATION)
else:
    # held-out split of the canonical set (same rule as GraspEnv._load_pool)
    canon = load_episodes(DATA)
    pre_max = int(math.ceil(cfg.pre_close_s[1] * 30))
    keep = []
    for e, d in sorted(canon.items()):
        cf = close_frame(d["state"][:, 5], d["inspection_frame"])
        if not d["excluded"] and cf is not None and cf >= pre_max + 1 and d["inspection_frame"]:
            keep.append(e)
    perm = np.random.default_rng(0).permutation(len(keep))
    test_src = {keep[i] for i in perm[int(round(len(keep) * 2 / 3)):]}
pre_max = int(math.ceil(cfg.pre_close_s[1] * 30))
if args.prior_chunk and client is None:
    raise ValueError("--prior_chunk needs --latent_snap")
RES_SCALE = torch.tensor([args.res_xy_m, args.res_xy_m, args.res_z_m, 10.0, 10.0, 30.0])

root = Path(args.data)
labels = {}
for line in (root / "meta/dlsj/labels.jsonl").read_text().splitlines():
    if line.strip():
        lab = json.loads(line)
        labels[int(lab["episode_index"])] = lab
rv_file = root / "meta/dlsj/deep_review.json"
review = json.loads(rv_file.read_text()) if rv_file.exists() else {}
eps = load_episodes(root)
srcs = set(args.sources.split(","))
stats = {"considered": 0, "used": 0, "skipped_test_split": 0, "skipped_no_close": 0, "skipped_accuracy": 0,
         "no_lift": 0, "by_source": {}}


def review_of(lab):
    sd, se = lab.get("source_dataset"), lab.get("source_episode")
    key = {"duplo_teleop_cal2": f"cal2_{se:03d}", "duplo_pick": f"pick_{se:03d}"}.get(sd)
    return review.get(key, {}) if key else {}


def det_model(rel, g):
    dist = rel.norm(dim=-1)
    cosang = (-rel[:, 2]) / dist.clamp(min=1e-4)
    valid = ((dist < cfg.det_max_range_m) & (cosang > math.cos(math.radians(cfg.det_fov_deg / 2)))) | (dist < 0.04)
    valid &= torch.rand(len(rel), generator=g) > cfg.det_dropout
    rel = rel + torch.randn(rel.shape, generator=g) * cfg.det_noise_m
    return torch.where(valid[:, None], rel, torch.zeros_like(rel)), valid.float()


def correction(P5, D5):
    """Residual (6,) that maps the prior's 5 goals P5 onto the demo's D5 (image_task._apply's coordinates), unclipped:
    fingertip offset in the prior goal's gripper frame (m), wrist flex / roll (deg), gripper (units), 5-goal mean."""
    pp, Rp = kin.fk(P5)
    pd, _ = kin.fk(D5)
    dx = torch.einsum("tji,tj->ti", Rp, pd - pp)  # R^T (p_demo - p_prior)
    return torch.cat([dx, D5[:, 3:5] - P5[:, 3:5], D5[:, 5:6] - P5[:, 5:6]], -1).mean(0)


rows = {k: [] for k in ("s", "s2", "A", "A2", "a_sel", "r", "rew", "term", "end", "epi", "G")}
near_delta = []
n_prior_calls = 0
nxt_all = []
n_rows = 0
epi = 0
for i_ep, (e, d) in enumerate(sorted(eps.items())):
    if i_ep % SHARD_N != SHARD_I:
        continue
    lab = labels.get(e, {})
    if lab.get("source_dataset") not in srcs:
        continue
    stats["considered"] += 1
    if lab.get("source_dataset") == "duplo_teleop_cal2" and lab.get("source_episode") in test_src:
        stats["skipped_test_split"] += 1
        continue
    rv = review_of(lab)
    if args.min_accuracy and int(rv.get("accuracy") or 0) < args.min_accuracy:
        stats["skipped_accuracy"] += 1
        continue
    S, A = torch.tensor(d["state"]), torch.tensor(d["action"])
    cf = close_frame(d["state"][:, 5], d["inspection_frame"])
    if cf is None and d.get("grasp_frame"):
        cf = int(d["grasp_frame"])
    if cf is None or cf < pre_max + 1:
        stats["skipped_no_close"] += 1
        continue
    L = len(S)
    p, R = kin.fk(S)
    b0 = torch.tensor([float(p[cf, 0]), float(p[cf, 1]), REST_Z])
    rel_c = R[cf].T @ (b0 - p[cf])
    # brick world position per tick: static until 3 ticks after the close, then carried by the gripper
    carried = p + (R @ rel_c)
    tt = torch.arange(L)
    bw = torch.where((tt >= cf + 3)[:, None], carried, b0.expand(L, 3))
    bw[:, 2] = torch.clamp(bw[:, 2], min=REST_Z)
    height = bw[:, 2] - REST_Z
    lifted = height > cfg.lift_m
    hold = torch.zeros(L, dtype=torch.long)
    for t in range(1, L):
        hold[t] = hold[t - 1] + 1 if lifted[t] else 0
    succ_t = torch.nonzero(hold >= cfg.hold_ticks).flatten()
    if not len(succ_t):
        stats["no_lift"] += 1
        continue
    t_s = int(succ_t[0])
    if args.max_episodes and stats["used"] >= args.max_episodes:
        break
    task = None
    if client is not None:
        word = brick_colour_word(frames.frame(e, "wrist", min(cf + 8, L - 1)))
        task = FT_PICK.format(colour=word)
        stats.setdefault("colours", {})[word] = stats.get("colours", {}).get(word, 0) + 1
    phi = 0.2 * (height / cfg.lift_m).clamp(0, 1) + 0.1 * (tt >= cf + 3).float()
    req = {}  # request tick -> (prior chunk (30, 6), pooled latent): every 15 ticks from the start, as the controller
    for pre in ([cf] if args.pre == "all" else [int(v) for v in args.pre.split(",")]):
        t0 = cf - pre
        dec = list(range(t0, t_s + 1 if args.pre == "all" else min(t0 + T, t_s + 1), H))
        ss, AA, RR, rw, tm = [], [], [], [], []
        for j, t in enumerate(dec):
            idx = torch.clamp(torch.arange(t, t + H), max=L - 1)
            feat = ((A[idx] - S[t]) / 20.0).flatten()
            r_lab = torch.zeros(6)
            if client is not None:  # the prior on the real frames at its last request tick
                t_rq = t0 + 15 * ((t - t0) // 15)
                if t_rq not in req:
                    img = {c: frames.frame(e, c, t_rq) for c in ("wrist", "top")}
                    ch, z = client.act_k({"images": img, "state": S[t_rq].numpy(), "task": task}, 1,
                                         [DEPLOY_NOISE_SEED])
                    req[t_rq] = (torch.as_tensor(np.asarray(ch)[0], dtype=torch.float32), torch.as_tensor(z))
                    n_prior_calls += 1
                Pc, z = req[t_rq]
                if args.prior_chunk:
                    pidx = torch.clamp(torch.arange(t - t_rq, t - t_rq + H), max=len(Pc) - 1)
                    feat = ((Pc[pidx] - S[t]) / 20.0).flatten()
                    delta = correction(Pc[pidx], A[idx])
                    r_lab = (delta / RES_SCALE).clamp(-1, 1)
                    if abs(t - cf) <= 15:
                        near_delta.append(delta)
            rel = R[t].T @ (bw[t] - p[t])
            det, valid = det_model(rel[None], gen)
            load = (S[t, 5] - A[max(t - 1, 0), 5]) / 10.0
            ob = torch.cat([S[t] / 100.0, det[0] * 10.0, valid, load[None], torch.tensor([(t - t0) / T]),
                            torch.zeros(12), torch.ones(1)])
            if client is not None:  # the prior's pooled prefix latent (from its last request)
                if proj is None:
                    proj = torch.as_tensor(LatentProjector(len(z), args.proj_dim).W)
                    stats["latent_dim"] = len(z)
                ob = torch.cat([ob, torch.as_tensor(z) @ proj])
            ss.append(torch.cat([ob, feat.repeat(KS)]))
            AA.append(feat.expand(max(KS, 4), -1).clone())
            RR.append(r_lab)
            t_end = t + H
            r = float(phi[min(t_end, L - 1)] - phi[t])
            done = t <= t_s < t_end
            if done:
                r += cfg.w_success
            rw.append(r)
            tm.append(done or j == len(dec) - 1)
        m = len(ss)
        s_t = torch.stack(ss)
        A_t = torch.stack(AA)
        rows["s"].append(s_t)
        rows["A"].append(A_t)
        rows["a_sel"].append(A_t[:, 0])
        rows["s2"].append(torch.cat([s_t[1:], s_t[-1:]]))
        rows["A2"].append(torch.cat([A_t[1:], A_t[-1:]]))
        rows["r"].append(torch.stack(RR))
        rw_t = torch.tensor(rw)
        rows["rew"].append(rw_t)
        rows["term"].append(torch.tensor(tm))
        rows["end"].append(torch.tensor(tm))
        rows["epi"].append(torch.full((m,), epi, dtype=torch.long))
        G = torch.zeros(m)
        acc = 0.0
        for j in range(m - 1, -1, -1):
            acc = float(rw_t[j]) + 0.99 * acc
            G[j] = acc
        rows["G"].append(G)
        nx = torch.arange(n_rows + 1, n_rows + m + 1)
        nx[-1] = -1
        nxt_all.append(nx)
        n_rows += m
        epi += 1
    stats["used"] += 1
    stats["by_source"][lab.get("source_dataset")] = stats["by_source"].get(lab.get("source_dataset"), 0) + 1

out = {k: torch.cat(v) for k, v in rows.items()}
out["nxt"] = torch.cat(nxt_all)
stats.update(prior_calls=n_prior_calls, prior_chunk=bool(args.prior_chunk), shard=args.shard,
             residual_bounds={"res_xy_m": args.res_xy_m, "res_z_m": args.res_z_m})
if near_delta:
    out["near_delta"] = torch.stack(near_delta)
    stats["near_close"] = near_summary(out["near_delta"], args.res_xy_m, args.res_z_m)
out["stats"] = stats
torch.save(out, args.out)
if client is not None:
    client.close()
starts = torch.cat([torch.tensor([0]), (out["nxt"] < 0).nonzero().flatten()[:-1] + 1])
stats.update(rows=int(n_rows), sequences=epi, mean_return=round(float(out["G"][starts].mean()), 3))
print("EXPERT", json.dumps(stats), flush=True)
