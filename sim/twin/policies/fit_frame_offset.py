"""Is zero-shot MolmoAct2's bias a joint-frame mismatch? Fit a per-joint frame offset on real cal2 frames.

    docker run --rm --init -v ~/Documents/dlsj:/repo -w /repo -e PYTHONPATH=/repo \
        -e HF_HOME=/repo/sim/twin/data/hf --device nvidia.com/gpu=all \
        --entrypoint /opt/isaaclab-venv/bin/python dlsj/twin:latest sim/twin/policies/fit_frame_offset.py

Offset delta (runtime cal2 degrees): the model sees state + delta, its actions come back - delta
(MolmoAct2Client(frame_offset_deg=...)). Deployed settings otherwise (wrist+top, deployed prompts, 10 steps, random
start noise), with a FIXED noise seed per timestep (ep * 1000 + t) so every delta sees the same draws.

Sets (4 timesteps per episode: start, mid-reach, just before inspection, mid-place; check_molmoact2.load_points):
  A    = the 16 of check_molmoact2 (4 eps, seed 0)       } HELD-OUT (40 timesteps)
  B    = the 24 of the second check run (6 eps, seed 1)  }
  FIT  = 6 other episodes (seed 2, disjoint), 24 timesteps -- delta is fitted only here.
Fit: coordinate descent on the mean |pred - recorded| over the first 10 actions (5 body joints): shoulder_lift,
wrist_roll, then pan, elbow, wrist_flex on a -20..20 grid in 2.5 deg steps, then a +-2 deg / 0.5 refine pass.
Writes sim/twin/out/policies/fit_frame_offset.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from sim.twin.align import frames as F
from sim.twin.policies.check_molmoact2 import load_points, pick_episodes, prompt
from sim.twin.policies.molmoact2 import FRAME_OFFSETS, JOINTS, MolmoAct2Client

BODY = 5
ORDER = ("shoulder_lift", "wrist_roll", "shoulder_pan", "elbow_flex", "wrist_flex")


def predict(pol, pts, delta, seed_off=0) -> np.ndarray:
    out = []
    for p in pts:
        o = {"images": p["images"], "state": p["state"]}
        out.append(pol.act(o, task=prompt("deploy", p["phase"], p["lab"]), frame_offset_deg=delta,
                           seed=p["ep"] * 1000 + p["t"] + seed_off))
    return np.stack(out)  # (n, 30, 6)


def metrics(P: np.ndarray, pts) -> dict:
    R = np.stack([p["rec"] for p in pts])
    S0 = np.stack([p["state"] for p in pts])[:, None]
    e10, e30 = np.abs(P[:, :10] - R[:, :10]), np.abs(P - R)
    return {"j10": float(e10[..., :BODY].mean()), "j30": float(e30[..., :BODY].mean()),
            "g10": float(e10[..., 5].mean()), "g30": float(e30[..., 5].mean()),
            "err10": e10.mean((0, 1)).tolist(),
            "bias10": (P[:, :10] - R[:, :10]).mean((0, 1)).tolist(),
            "a0_minus_state": (P[:, 0] - S0[:, 0]).mean(0).tolist()}


def hold(pts) -> dict:
    return metrics(np.stack([np.repeat(p["state"][None], 30, 0) for p in pts]), pts) | {
        "rec_a0_minus_state": np.mean([p["rec"][0] - p["state"] for p in pts], 0).tolist()}


def pooled(a: dict, b: dict, na: int, nb: int) -> dict:
    """Metrics of A + B from the per-set means (all metrics are per-timestep means)."""
    return {k: ((np.asarray(a[k]) * na + np.asarray(b[k]) * nb) / (na + nb)).tolist() for k in a}


def fmt(m: dict) -> str:
    return (f"j10 {m['j10']:5.2f} j30 {m['j30']:5.2f} g10 {m['g10']:5.2f} | bias10 "
            + " ".join(f"{v:+6.1f}" for v in m["bias10"]) + " | a0-s " + " ".join(f"{v:+6.1f}" for v in m["a0_minus_state"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(F.DEFAULT_ROOT))
    ap.add_argument("--fit-eps", type=int, default=6)
    ap.add_argument("--span", type=float, default=20.0)
    ap.add_argument("--step", type=float, default=2.5)
    ap.add_argument("--out", default="sim/twin/out/policies/fit_frame_offset.json")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res: dict = {"joints": JOINTS}

    eps_a = pick_episodes(args.root, 4, 0)
    eps_b = pick_episodes(args.root, 6, 1)
    eps_f = pick_episodes(args.root, args.fit_eps, 2, exclude=eps_a + eps_b)
    sets = {"A": load_points(args.root, eps=eps_a), "B": load_points(args.root, eps=eps_b),
            "FIT": load_points(args.root, eps=eps_f)}
    sets["HELDOUT"] = sets["A"] + sets["B"]
    res["episodes"] = {"A": eps_a, "B": eps_b, "FIT": eps_f}
    print("episodes", res["episodes"], {k: len(v) for k, v in sets.items()}, flush=True)
    res["hold"] = {k: hold(v) for k, v in sets.items()}
    for k in ("A", "B", "HELDOUT", "FIT"):
        print(f"hold {k:8s}", fmt(res["hold"][k]), "| rec a0-s",
              " ".join(f"{v:+6.1f}" for v in res["hold"][k]["rec_a0_minus_state"]), flush=True)

    pol = MolmoAct2Client(verbose=False)
    print("model ready", pol.info.get("load_s"), "s", flush=True)

    # ---------------------------------------------------------------- fit on FIT
    cache: dict = {}

    def obj(d) -> float:
        key = tuple(np.round(d, 3))
        if key not in cache:
            cache[key] = metrics(predict(pol, sets["FIT"], np.asarray(d, np.float32)), sets["FIT"])
        return cache[key]["j10"]

    t0 = time.perf_counter()
    d = np.zeros(6, np.float32)
    best = obj(d)
    print(f"fit start delta=0 j10 {best:.3f}", flush=True)
    trace = []
    for rnd, (span, step) in enumerate(((args.span, args.step), (2.0, 0.5))):
        for j in ORDER:
            ji = JOINTS.index(j)
            c = float(d[ji])
            grid = np.round(np.arange(c - span, c + span + 1e-6, step), 3)
            vals = []
            for g in grid:
                dd = d.copy()
                dd[ji] = g
                vals.append(obj(dd))
            k = int(np.argmin(vals))
            d[ji], best = grid[k], vals[k]
            trace.append({"round": rnd, "joint": j, "grid": grid.tolist(), "j10": [round(v, 3) for v in vals]})
            print(f"[{time.perf_counter() - t0:6.0f}s] round {rnd} {j:13s} -> {grid[k]:+6.2f}  j10 {best:.3f}  "
                  f"(curve {' '.join(f'{v:.2f}' for v in vals)})  delta {np.round(d, 2).tolist()}", flush=True)
            res["fit_trace"] = trace
            out.write_text(json.dumps(res, indent=1))
    res["fitted_delta"] = d.tolist()
    res["fit_s"] = round(time.perf_counter() - t0, 1)
    print("FITTED delta", np.round(d, 2).tolist(), "FIT j10", round(best, 3), flush=True)

    # ---------------------------------------------------------------- evaluate variants
    dl = np.array(FRAME_OFFSETS["old_cal"], np.float32)
    du = np.array(FRAME_OFFSETS["urdf"], np.float32)
    lr = np.zeros(6, np.float32)
    for j in ("shoulder_lift", "wrist_roll"):
        lr[JOINTS.index(j)] = d[JOINTS.index(j)]
    variants = {"none": np.zeros(6, np.float32), "old_cal": dl, "old_cal_no_roll": dl * [1, 1, 1, 1, 0, 1],
                "urdf": du, "urdf_no_roll": du * [1, 1, 1, 1, 0, 1], "fitted": d, "fitted_lift_roll_only": lr}
    res["variants"] = {}
    for name, dv in variants.items():
        res["variants"][name] = {"delta": np.asarray(dv, float).round(2).tolist()}
        for k in ("A", "B", "FIT"):
            if k == "FIT" and name in ("none", "fitted"):
                res["variants"][name][k] = cache.get(tuple(np.round(dv, 3))) or metrics(predict(pol, sets[k], dv), sets[k])
            else:
                res["variants"][name][k] = metrics(predict(pol, sets[k], dv), sets[k])
        va, vb = res["variants"][name]["A"], res["variants"][name]["B"]
        res["variants"][name]["HELDOUT"] = pooled(va, vb, len(sets["A"]), len(sets["B"]))
        for k in ("A", "B", "HELDOUT", "FIT"):
            print(f"{name:22s} {k:8s}", fmt(res["variants"][name][k]), flush=True)
        out.write_text(json.dumps(res, indent=1))
    # sampling-noise floor: deployed (delta 0) with another set of seeds on held-out
    m2 = metrics(predict(pol, sets["HELDOUT"], np.zeros(6, np.float32), seed_off=7), sets["HELDOUT"])
    res["none_other_seeds_HELDOUT"] = m2
    print(f"{'none (other seeds)':22s} HELDOUT ", fmt(m2), flush=True)
    pol.close()
    out.write_text(json.dumps(res, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
