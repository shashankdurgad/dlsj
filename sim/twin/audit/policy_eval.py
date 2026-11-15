"""Policy equivalence (the audit's headline): MolmoAct2 on real vs sim frames at the same state, prompt and noise.

    python sim/twin/audit/policy_eval.py --run sim/twin/out/audit/runs/<stamp> [--model zero|ft]   # plain python, GPU

Per sample the model gets the recorded state and prompt and the wrist + top images of one condition:
    real       the recorded frames (reference)
    sim        the twin's renders at the same joint state
    nbr        the NEXT recorded frames (33 ms later; same state fed): the real-vs-real floor of a benign image change
    ref        recorded frames of a different episode at the same phase (wrong scene): the ceiling
    sim_top    sim top + real wrist       sim_wrist    real top + sim wrist     (which camera carries the gap)
Each condition: K flow-noise seeds from one prefix forward (MolmoAct2Client.act_k), the same seeds everywhere, so
`x vs real, same seed` isolates the image effect; `real seed a vs real seed b` is the flow-noise floor.
Metrics (runtime units: degrees for the 5 joints, 0-100 for the gripper): mean |delta| of the 5 joints over the first
10 / 30 actions, gripper |delta| over the first 10 / 30, cosine of the pooled prefix feature (raw and centred on the
mean real feature). Equivalence index = (sim - nbr) / (ref - nbr): 0 = as close as the next real frame, 1 = as far as
a wrong scene. The real-side results (real / nbr / ref) are cached in set/ (identical across audits).
Models: zero = allenai/MolmoAct2-SO100_101 as deployed; ft = the team's 8k-step fine-tune (top cropped to 560 px,
three stage prompts). Writes <run>/policy_<model>.json and policy_<model>.npz.
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
from sim.twin.audit.common import FT_DIR, PHASES, SET, jdump, load_set

K = 4
REAL_CONDS = ("real", "nbr", "ref")
SIM_CONDS = ("sim", "sim_top", "sim_wrist")
FT_BRICK = {"blue": "blue", "red": "red", "yellow": "yellow", "white": "white", "cream": "white", "brown": "brown",
            "tan": "brown", "orange": "brown", "green": "blue"}


def ft_ckpt():
    c = sorted(FT_DIR.glob("*.ckpt")) if FT_DIR.exists() else []
    return c[0] if c else None


def brick_colour(lay: dict | None) -> str:
    name = str(((lay or {}).get("target") or {}).get("color") or "").lower()
    for k, v in FT_BRICK.items():
        if k in name:
            return v
    return "blue"


def prompt(model: str, s: dict, lay: dict | None) -> str:
    from sim.twin.policies.molmoact2 import DEPLOY

    if model == "zero":
        if s["phase"] == "place":
            return DEPLOY["place_prompt"].format(colour=s["bowl"])
        return DEPLOY["pick_prompt"]
    if s["frame"] < s["grasp_frame"]:
        return f"pick up the {brick_colour(lay)} brick"
    if s["frame"] < s["inspection_frame"]:
        return "bring the brick to the inspection camera"
    return f"place the brick in the {s['bowl']} bowl"


def delta(a, b) -> dict:
    """a, b (..., 30, 6) -> per-sample metrics averaged over the leading axes after the first."""
    d = np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))
    ax = tuple(range(1, d.ndim - 2))
    return {"joint10": d[..., :10, :5].mean((-1, -2)).mean(ax) if ax else d[..., :10, :5].mean((-1, -2)),
            "joint30": d[..., :, :5].mean((-1, -2)).mean(ax) if ax else d[..., :, :5].mean((-1, -2)),
            "grip10": d[..., :10, 5].mean(-1).mean(ax) if ax else d[..., :10, 5].mean(-1),
            "grip30": d[..., :, 5].mean(-1).mean(ax) if ax else d[..., :, 5].mean(-1),
            "jointmax10": d[..., :10, :5].max(-1).mean(-1).mean(ax) if ax else d[..., :10, :5].max(-1).mean(-1)}


def cosine(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)


def summarise(v) -> dict:
    v = np.asarray(v, np.float64)
    return {"mean": float(v.mean()), "median": float(np.median(v)), "p90": float(np.percentile(v, 90))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default="zero", choices=("zero", "ft"))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    run = Path(a.run)
    meta = load_set()
    S = meta["samples"][: a.limit] if a.limit else meta["samples"]
    N = len(S)
    lp = SET / "layouts_first.json"
    lays = json.loads(lp.read_text()) if lp.exists() else {}
    prompts = [prompt(a.model, s, lays.get(str(s["ep"]))) for s in S]
    fr = {k: np.load(SET / f"{k}.npz") for k in REAL_CONDS}
    sim = np.load(run / "sim.npz")
    dark = [i for i in range(N) if min(float(sim["top"][i].mean()), float(sim["wrist"][i].mean())) <= 2.0]
    if dark:
        sys.exit(f"POLICY_ABORT {len(dark)} black sim samples (first {dark[:5]}): not auditing black frames")
    img = {"real": lambda i: (fr["real"]["top"][i], fr["real"]["wrist"][i]),
           "nbr": lambda i: (fr["nbr"]["top"][i], fr["nbr"]["wrist"][i]),
           "ref": lambda i: (fr["ref"]["top"][i], fr["ref"]["wrist"][i]),
           "sim": lambda i: (sim["top"][i], sim["wrist"][i]),
           "sim_top": lambda i: (sim["top"][i], fr["real"]["wrist"][i]),
           "sim_wrist": lambda i: (fr["real"]["top"][i], sim["wrist"][i])}

    from sim.twin.policies.molmoact2 import MolmoAct2Client

    if a.model == "ft":
        ck = ft_ckpt()
        if ck is None:
            print("POLICY_SKIP no fine-tuned checkpoint in", FT_DIR)
            return
        pol = MolmoAct2Client(snap=str(ck), cams=("top", "wrist"), crops={"top": (0, 0, 560, 480)})
    else:
        pol = MolmoAct2Client(cams=("top", "wrist"))
    print("client:", pol.info, flush=True)

    def run_cond(c: str):
        acts, lats, lat_s = [], [], []
        for i, s in enumerate(S):
            top, wrist = img[c](i)
            obs = {"images": {"top": top, "wrist": wrist}, "state": np.asarray(s["state"], np.float32)}
            x, lat = pol.act_k(obs, K, seeds=list(range(K)), task=prompts[i])
            acts.append(x)
            lats.append(lat if lat is not None else np.zeros(1, np.float32))
            lat_s.append(pol.last_latency_s)
        print(f"[policy {a.model}] {c}: {N} samples, {np.mean(lat_s):.2f} s / call", flush=True)
        return np.stack(acts), np.stack(lats)

    cache_p = SET / f"policy_cache_{a.model}.npz"
    A, Lt = {}, {}
    t0 = time.time()
    if cache_p.exists() and not a.limit:
        z = np.load(cache_p, allow_pickle=False)
        if list(z["prompts"]) == prompts and int(z["k"]) == K:
            for c in REAL_CONDS:
                A[c], Lt[c] = z[f"a_{c}"], z[f"l_{c}"]
    for c in REAL_CONDS:
        if c not in A:
            A[c], Lt[c] = run_cond(c)
    if not cache_p.exists() and not a.limit:
        np.savez_compressed(cache_p, prompts=np.array(prompts), k=K,
                            **{f"a_{c}": A[c] for c in REAL_CONDS}, **{f"l_{c}": Lt[c] for c in REAL_CONDS})
    # determinism of the reference (same image, same seeds, a second call): should be ~0
    chk = []
    for i in range(min(3, N)):
        top, wrist = img["real"](i)
        x, _ = pol.act_k({"images": {"top": top, "wrist": wrist}, "state": np.asarray(S[i]["state"], np.float32)},
                         K, seeds=list(range(K)), task=prompts[i])
        chk.append(float(np.abs(x - A["real"][i]).max()))
    for c in SIM_CONDS:
        A[c], Lt[c] = run_cond(c)
    pol.close()

    real = A["real"]  # (N, K, 30, 6)
    per = {}
    for c in ("sim", "nbr", "ref", "sim_top", "sim_wrist"):
        m = delta(A[c], real)  # same seed, averaged over seeds
        m.update({"mean_" + k: v for k, v in delta(A[c].mean(1), real.mean(1)).items()})
        per[c] = m
    pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
    nz = [delta(real[:, i], real[:, j]) for i, j in pairs]
    per["seed_noise"] = {k: np.mean([d[k] for d in nz], 0) for k in nz[0]}
    has_lat = Lt["real"].shape[-1] > 1
    cos = {}
    if has_lat:
        mu = Lt["real"].mean(0, keepdims=True)
        for c in ("sim", "nbr", "ref", "sim_top", "sim_wrist"):
            cos[c] = {"cos": cosine(Lt[c], Lt["real"]), "cos_centred": cosine(Lt[c] - mu, Lt["real"] - mu)}
    fut = np.asarray([s["future_actions"] for s in S], np.float64)
    demo = {c: delta(A[c].mean(1), fut) for c in ("real", "sim")}
    hold = delta(np.repeat(np.asarray([s["state"] for s in S], np.float64)[:, None], 30, 1), fut)

    def table(idx) -> dict:
        out = {}
        for c, m in per.items():
            out[c] = {k: summarise(v[idx]) for k, v in m.items()}
        for c, m in cos.items():
            out[c].update({k: summarise(v[idx]) for k, v in m.items()})
        eq = {}
        for k in ("joint10", "joint30", "grip10", "grip30"):
            f, ce, x = per["nbr"][k][idx].mean(), per["ref"][k][idx].mean(), per["sim"][k][idx].mean()
            eq[k] = float((x - f) / max(ce - f, 1e-9))
        if has_lat:
            f, ce, x = (1 - cos["nbr"]["cos"][idx]).mean(), (1 - cos["ref"]["cos"][idx]).mean(), \
                (1 - cos["sim"]["cos"][idx]).mean()
            eq["feature"] = float((x - f) / max(ce - f, 1e-9))
        out["equivalence_index"] = eq
        out["demo_error"] = {c: {k: float(v[idx].mean()) for k, v in m.items()} for c, m in demo.items()}
        out["demo_error"]["hold"] = {k: float(v[idx].mean()) for k, v in hold.items()}
        return out

    def close_step(x):
        """First chunk step with the gripper commanded below 12 (30 = not within the chunk), mean over seeds."""
        c = np.asarray(x)[..., 5] < 12.0
        return np.where(c.any(-1), c.argmax(-1), 30).mean(-1)

    # signed effect of the sim images (sim - real, same seed): a systematic bias would move the grasp point / timing
    bias = (A["sim"] - real)[:, :, :10].mean((1, 2))  # (N, 6)
    closes = {c: close_step(A[c]) for c in ("real", "sim", "nbr")}
    pg = np.array([i for i, s in enumerate(S) if s["phase"] in ("pregrasp", "reach")])
    timing = {"note": "reach + pregrasp samples; chunk step (30 Hz) of the first gripper command < 12, 30 = none",
              "real": float(closes["real"][pg].mean()), "sim": float(closes["sim"][pg].mean()),
              "next_real_frame": float(closes["nbr"][pg].mean()),
              "abs_diff_sim": float(np.abs(closes["sim"] - closes["real"])[pg].mean()),
              "abs_diff_next_real_frame": float(np.abs(closes["nbr"] - closes["real"])[pg].mean()),
              "sim_closes_earlier_frac": float((closes["sim"][pg] < closes["real"][pg] - 1).mean()),
              "sim_closes_later_frac": float((closes["sim"][pg] > closes["real"][pg] + 1).mean())}
    allidx = np.arange(N)
    res = dict(model=pol.info.get("model"), model_key=a.model, k=K, n=N, seconds=round(time.time() - t0, 1),
               determinism_max_abs=chk, overall=table(allidx),
               bias_sim_minus_real_first10={"overall": bias.mean(0).round(3),
                                            **{p: bias[[i for i, s in enumerate(S) if s["phase"] == p]].mean(0).round(3)
                                               for p in PHASES if any(s["phase"] == p for s in S)}},
               close_timing=timing,
               by_phase={p: table(np.array([i for i, s in enumerate(S) if s["phase"] == p]))
                         for p in PHASES if any(s["phase"] == p for s in S)},
               per_sample={c: {k: np.round(v, 4) for k, v in m.items() if not k.startswith("mean_")}
                           for c, m in per.items()},
               per_sample_cos={c: {k: np.round(v, 5) for k, v in m.items()} for c, m in cos.items()},
               prompts=prompts)
    jdump(run / f"policy_{a.model}.json", res)
    np.savez_compressed(run / f"policy_{a.model}.npz", **{f"a_{c}": A[c] for c in SIM_CONDS},
                        **{f"l_{c}": Lt[c] for c in SIM_CONDS})
    o = res["overall"]
    j10 = {c: o[c]["joint10"]["mean"] for c in ("sim", "nbr", "ref", "seed_noise")}
    print(f"POLICY_OK {a.model}: joint10 deg {j10} | equivalence index {o['equivalence_index']}", flush=True)


if __name__ == "__main__":
    main()
