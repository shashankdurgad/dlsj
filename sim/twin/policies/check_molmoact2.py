"""MolmoAct2 runner check on the GB10: (a) load + latency, (b) sanity on REAL recorded cal2 frames.

    docker run --rm --init -v ~/Documents/dlsj:/repo -w /repo -e PYTHONPATH=/repo \
        -e HF_HOME=/repo/sim/twin/data/hf --device nvidia.com/gpu=all \
        --entrypoint /opt/isaaclab-venv/bin/python dlsj/twin:latest sim/twin/policies/check_molmoact2.py

(b) feeds recorded top / wrist / inspect frames + states of 16 timesteps (4 episodes x 4 points: start, mid-reach,
just before the inspection frame, mid-place) to the client and compares the predicted chunk with the recorded next
actions (`action` column, t .. t+29): mean |pred - rec| per joint over the first 10 and 30 steps, against the
"hold" baseline (repeat the current state). Variants differ only in instruction / cameras / noise / steps (one model
load). Writes sim/twin/out/policies/check_molmoact2.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from sim.twin.align import frames as F
from sim.twin.policies.molmoact2 import DEPLOY, JOINTS, MolmoAct2Client

THREE = {"pick": "grasp the brick", "present": "lift the brick and hold it still in front of the inspection camera",
         "place": "place the brick in the {bowl} bowl and release it"}


def prompt(kind: str, phase: str, lab: dict) -> str:
    bowl = lab.get("bowl_final") or lab.get("bowl") or "yellow"
    if kind == "deploy":  # deploy/smolvla/run_molmo.sh
        return DEPLOY["pick_prompt"] if phase != "place" else DEPLOY["place_prompt"].format(colour=bowl)
    if kind == "data":  # the recorded task strings (= sim/isaac_molmoact2 rollout's instruction for pick)
        return lab.get("pick_task") if phase != "place" else lab.get("place_task")
    if kind == "three":  # deploy/molmoact2/harness.py THREE_PHASE (fine-tune strings)
        return THREE[{"start": "pick", "reach": "pick"}.get(phase, phase)].format(bowl=bowl)
    raise KeyError(kind)


VARIANTS = {  # name: (prompt kind, cams in model order, noise, steps)
    "deploy": ("deploy", ("wrist", "top"), "random", 10),
    "deploy_zeros": ("deploy", ("wrist", "top"), "zeros", 10),
    "deploy_seed2": ("deploy", ("wrist", "top"), "random", 10),
    "deploy_5steps": ("deploy", ("wrist", "top"), "random", 5),
    "data_prompt": ("data", ("wrist", "top"), "random", 10),
    "three_phase": ("three", ("wrist", "top"), "random", 10),
    "cams_top_wrist": ("deploy", ("top", "wrist"), "random", 10),
    "cams_top_inspect": ("deploy", ("top", "inspect"), "random", 10),
    "data_top_inspect": ("data", ("top", "inspect"), "random", 10),
    "cams_wrist_top_inspect": ("deploy", ("wrist", "top", "inspect"), "random", 10),
    "cams_top_only": ("deploy", ("top",), "random", 10),
}


def load_points(root: str, n_eps: int, seed: int) -> list[dict]:
    labs = F.labels(root)
    rng = np.random.default_rng(seed)
    eps = sorted(int(e) for e in rng.choice(sorted(e for e, l in labs.items() if l.get("success", True)
                                                   and l.get("inspection_frame")), n_eps, replace=False))
    tab = F.frames_table(root)
    pts = []
    for e in eps:
        lab = labs[e]
        s = F.states(e, root).astype(np.float32)
        a = np.stack(tab[tab.episode_index == e].sort_values("frame_index")["action"].to_numpy()).astype(np.float32)
        n, insp = len(s), int(lab["inspection_frame"])
        ts = {"start": 10, "reach": insp // 2, "present": max(0, insp - 15), "place": insp + (n - insp) // 3}
        imgs = {c: F.video_frames(e, c, list(ts.values()), root) for c in F.CAMS}
        for phase, t in ts.items():
            rec = a[np.minimum(np.arange(t, t + 30), n - 1)]
            pts.append({"ep": e, "t": t, "phase": phase, "lab": lab, "state": s[t], "rec": rec,
                        "images": {c: imgs[c][t] for c in F.CAMS}})
    return pts


def err(pred: np.ndarray, rec: np.ndarray, k: int) -> np.ndarray:
    return np.abs(pred[:k] - rec[:k]).mean(0)  # (6,)


def summarise(per: list[np.ndarray]) -> dict:
    e = np.stack(per)  # (n, 6)
    return {"per_joint": dict(zip(JOINTS, e.mean(0).round(2).tolist())), "joints_mean": round(float(e[:, :5].mean()), 2),
            "gripper": round(float(e[:, 5].mean()), 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(F.DEFAULT_ROOT))
    ap.add_argument("--snap", default=None)
    ap.add_argument("--eps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--latency-n", type=int, default=10)
    ap.add_argument("--out", default="sim/twin/out/policies/check_molmoact2.json")
    args = ap.parse_args()
    res: dict = {}

    t0 = time.perf_counter()
    pts = load_points(args.root, args.eps, args.seed)
    print(f"loaded {len(pts)} recorded timesteps from eps {sorted({p['ep'] for p in pts})} "
          f"in {time.perf_counter() - t0:.1f}s", flush=True)

    # ---------------------------------------------------------------- (a) load + latency
    t0 = time.perf_counter()
    pol = MolmoAct2Client(snap=args.snap)  # defaults = the real deployment
    res["load"] = pol.info | {"client_init_s": round(time.perf_counter() - t0, 1)}
    print("ready:", res["load"], flush=True)
    p0 = pts[0]
    obs = {"images": p0["images"], "state": p0["state"], "task": DEPLOY["pick_prompt"]}
    lat = {}
    for steps in (10, 5):
        ls, rt = [], []
        for k in range(args.latency_n + 1):
            pol.act(obs, steps=steps)
            ls.append(pol.last_latency_s)
            rt.append(pol.last_info["roundtrip_s"])
        lat[f"steps{steps}"] = {"first_s": round(ls[0], 3), "median_s": round(float(np.median(ls[1:])), 3),
                                "p90_s": round(float(np.percentile(ls[1:], 90)), 3), "min_s": round(min(ls[1:]), 3),
                                "roundtrip_median_s": round(float(np.median(rt[1:])), 3)}
    a = pol.act(obs)
    res["latency"] = lat | {"chunk_shape": list(a.shape)}
    print("latency:", res["latency"], flush=True)

    # ---------------------------------------------------------------- (b) real frames
    hold10 = [err(np.repeat(p["state"][None], 30, 0), p["rec"], 10) for p in pts]
    hold30 = [err(np.repeat(p["state"][None], 30, 0), p["rec"], 30) for p in pts]
    res["hold_baseline"] = {"first10": summarise(hold10), "first30": summarise(hold30)}
    res["variants"] = {}
    preds = {}
    for name in args.variants.split(","):
        kind, cams, noise, steps = VARIANTS[name]
        e10, e30, e10p, ex, b10 = [], [], {}, [], []
        P = []
        for p in pts:
            o = {"images": p["images"], "state": p["state"]}
            pr = pol.act(o, cams=cams, task=prompt(kind, p["phase"], p["lab"]), noise=noise, steps=steps)
            P.append(pr)
            e10.append(err(pr, p["rec"], 10))
            b10.append((pr[:10] - p["rec"][:10]).mean(0))
            e30.append(err(pr, p["rec"], 30))
            e10p.setdefault(p["phase"], []).append(err(pr, p["rec"], 10))
            # did it move the right way? cos between predicted and recorded displacement at step 10 (joints)
            d_pred, d_rec = pr[9, :5] - p["state"][:5], p["rec"][9, :5] - p["state"][:5]
            ex.append(float(d_pred @ d_rec / (np.linalg.norm(d_pred) * np.linalg.norm(d_rec) + 1e-6)))
        preds[name] = np.stack(P)
        res["variants"][name] = {"prompt": kind, "cams": list(cams), "noise": noise, "steps": steps,
                                 "first10": summarise(e10), "first30": summarise(e30),
                                 "first10_by_phase": {k: summarise(v)["joints_mean"] for k, v in e10p.items()},
                                 "signed_bias10": dict(zip(JOINTS, np.stack(b10).mean(0).round(2).tolist())),
                                 "dir_cos10_mean": round(float(np.mean(ex)), 2),
                                 "a0_minus_state_joints": round(float(np.mean([np.abs(q[0, :5] - p["state"][:5]).mean()
                                                                               for q, p in zip(P, pts)])), 2)}
        v = res["variants"][name]
        print(time.strftime("%H:%M:%S"), f"{name:24s} j10 {v['first10']['joints_mean']:6.2f} g10 {v['first10']['gripper']:6.2f} "
              f"j30 {v['first30']['joints_mean']:6.2f} cos {v['dir_cos10_mean']:5.2f} by-phase {v['first10_by_phase']} bias {v['signed_bias10']}",
              flush=True)
    if "deploy" in preds and "deploy_seed2" in preds:
        d = np.abs(preds["deploy"] - preds["deploy_seed2"])
        res["sampling_spread_first10"] = {"joints": round(float(d[:, :10, :5].mean()), 2),
                                          "gripper": round(float(d[:, :10, 5].mean()), 2)}

    # ---------------------------------------------------------------- RTC hand-off (deployed h=15, g=3)
    p = pts[1]
    o = {"images": p["images"], "state": p["state"], "task": DEPLOY["pick_prompt"]}
    c1 = pol.act(o)
    tail, delay = c1[15:], 10  # 15 executed, ~0.34 s * 30 fps of the tail passes during the next inference
    plain = pol.act(o)
    guided = pol.act(o, prev_chunk=tail, inference_delay=delay)
    res["rtc"] = {"plain_vs_tail_first5": round(float(np.abs(plain[:5, :5] - tail[:5, :5]).mean()), 2),
                  "rtc_vs_tail_first5": round(float(np.abs(guided[:5, :5] - tail[:5, :5]).mean()), 2),
                  "rtc_latency_s": round(pol.last_latency_s, 3)}
    print("rtc:", res["rtc"], flush=True)
    pol.close()

    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, np.generic):
            return x.item()
        return x

    res["points"] = [{"ep": p["ep"], "t": p["t"], "phase": p["phase"]} for p in pts]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(clean(res), indent=1))
    print("wrote", out)
    print(json.dumps({k: res[k] for k in ("hold_baseline", "latency", "rtc")}, indent=1))


if __name__ == "__main__":
    main()
