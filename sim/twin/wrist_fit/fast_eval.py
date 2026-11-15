"""Fast wrist-gap loop: MolmoAct2 (8k fine-tune) close timing on real vs (edited) sim frames, reach + pregrasp only.

    sim/twin/wrist_fit/py.sh sim/twin/wrist_fit/fast_eval.py --sim <run>/sim.npz --tag <name> \
        --variants sim: simw: simt: sim:sat14 sim:fins_real ...

Reuses the audit set (sim/twin/out/audit/set) and the audit's prompts / metrics (audit/policy_eval.py). The real-side
conditions (real, nbr = next real frame, ref = other episode same phase) are cached per (phases, K) in
sim/twin/out/wrist_fit/. Per variant: close step (first chunk step with the gripper commanded < 12, 30 = none; mean
over seeds and samples), sim - real, |sim - real|, later / earlier fractions, and the gripper equivalence index
grip10 = (x - nbr) / (ref - nbr) over the first 10 actions (0 = like the next real frame, 1 = like a wrong scene).
--sim may hold all 63 audit samples (audit run) or only the selected ones in set order (wrist_fit/render_sub.py).
Writes sim/twin/out/wrist_fit/eval_<tag>.json. Plain python with the GPU (no Isaac).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from sim.twin.audit.common import SET, load_set
from sim.twin.audit.policy_eval import delta, ft_ckpt, prompt
from sim.twin.wrist_fit.edits import apply

OUT = REPO / "sim/twin/out/wrist_fit"


def close_step(x):
    c = np.asarray(x)[..., 5] < 12.0
    return np.where(c.any(-1), c.argmax(-1), 30)  # (..., K) per seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--variants", nargs="+", default=["sim:", "simw:", "simt:"])
    ap.add_argument("--phases", default="reach,pregrasp")
    ap.add_argument("--k", type=int, default=8)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    meta = load_set()
    phases = a.phases.split(",")
    idx = [i for i, s in enumerate(meta["samples"]) if s["phase"] in phases]
    S = [meta["samples"][i] for i in idx]
    lays = json.loads((SET / "layouts_first.json").read_text())
    P = [prompt("ft", s, lays.get(str(s["ep"]))) for s in S]
    fr = {k: np.load(SET / f"{k}.npz") for k in ("real", "nbr", "ref")}
    sz = np.load(a.sim)
    full = sz["wrist"].shape[0] == len(meta["samples"])
    if not full and sz["wrist"].shape[0] != len(idx):
        sys.exit(f"--sim has {sz['wrist'].shape[0]} samples, need 63 or {len(idx)}")
    sim = {k: sz[k][idx] if full else sz[k] for k in ("top", "wrist", "mask_wrist")}
    real = {c: {cam: fr[c][cam][idx] for cam in ("top", "wrist")} for c in fr}

    from sim.twin.policies.molmoact2 import MolmoAct2Client

    pol = MolmoAct2Client(snap=str(ft_ckpt()), cams=("top", "wrist"), crops={"top": (0, 0, 560, 480)})
    seeds = list(range(a.k))

    def run(get) -> np.ndarray:
        acts = []
        for n, s in enumerate(S):
            top, wrist = get(n)
            x, _ = pol.act_k({"images": {"top": top, "wrist": wrist}, "state": np.asarray(s["state"], np.float32)},
                             a.k, seeds=seeds, task=P[n])
            acts.append(x)
        return np.stack(acts)  # (N, K, 30, 6)

    cache = OUT / f"real_cache_{'_'.join(phases)}_k{a.k}.npz"
    if cache.exists() and list(np.load(cache)["prompts"]) == P:
        z = np.load(cache)
        A = {c: z[c] for c in ("real", "nbr", "ref")}
    else:
        A = {c: run(lambda n, c=c: (real[c]["top"][n], real[c]["wrist"][n])) for c in ("real", "nbr", "ref")}
        np.savez_compressed(cache, prompts=np.array(P), **A)
    R = A["real"]
    cs_real = close_step(R).mean(-1)
    g = {c: delta(A[c], R)["grip10"] for c in ("nbr", "ref")}
    j = {c: delta(A[c], R)["joint10"] for c in ("nbr", "ref")}
    f_g, c_g = g["nbr"].mean(), g["ref"].mean()
    f_j, c_j = j["nbr"].mean(), j["ref"].mean()
    res = {"tag": a.tag, "sim": a.sim, "k": a.k, "phases": phases, "idx": idx, "real_close": cs_real.mean(),
           "nbr_close": close_step(A["nbr"]).mean(), "floor_grip10": f_g, "ceil_grip10": c_g, "variants": {}}
    print(f"[wf] {len(idx)} samples K={a.k}: real close {cs_real.mean():.2f}, nbr close "
          f"{close_step(A['nbr']).mean():.2f}; grip10 floor {f_g:.2f} ceil {c_g:.2f}", flush=True)

    def ctx(n):
        return {"real": real["real"]["wrist"][n], "sim": sim["wrist"][n], "mask": sim["mask_wrist"][n], "s": S[n],
                "lay": lays.get(str(S[n]["ep"]))}

    bases = {"sim": lambda n: (sim["top"][n], sim["wrist"][n]),
             "simw": lambda n: (real["real"]["top"][n], sim["wrist"][n]),
             "simt": lambda n: (sim["top"][n], real["real"]["wrist"][n]),
             "real": lambda n: (real["real"]["top"][n], real["real"]["wrist"][n])}
    for v in a.variants:
        base, _, ed = v.partition(":")
        t0 = time.time()

        def get(n, base=base, ed=ed):
            top, wrist = bases[base](n)
            return top, apply(ed, wrist, ctx(n))

        X = run(get)
        cs = close_step(X).mean(-1)
        gi = delta(X, R)["grip10"].mean()
        ji = delta(X, R)["joint10"].mean()
        m = {"close": float(cs.mean()), "d_close": float((cs - cs_real).mean()),
             "abs_d_close": float(np.abs(cs - cs_real).mean()),
             "later": float((cs > cs_real + 1).mean()), "earlier": float((cs < cs_real - 1).mean()),
             "grip10": float(gi), "grip_index": float((gi - f_g) / max(c_g - f_g, 1e-9)),
             "joint_index": float((ji - f_j) / max(c_j - f_j, 1e-9)),
             "bias_grip10": float((X - R)[:, :, :10, 5].mean()), "per_sample_close": cs.round(2).tolist()}
        res["variants"][v] = m
        print(f"[wf] {v:34s} close {m['close']:5.2f} (real {cs_real.mean():.2f}, d {m['d_close']:+5.2f}, "
              f"|d| {m['abs_d_close']:4.2f}, later {m['later']:.2f}) grip idx {m['grip_index']:.2f} "
              f"joint idx {m['joint_index']:.2f} bias {m['bias_grip10']:+5.2f} [{time.time() - t0:.0f}s]", flush=True)
        (OUT / f"eval_{a.tag}.json").write_text(json.dumps(res, indent=1, default=float))
    pol.close()
    print("WF_OK", a.tag, flush=True)


if __name__ == "__main__":
    main()
