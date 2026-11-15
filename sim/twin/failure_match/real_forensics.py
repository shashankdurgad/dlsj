"""Every gripper close of the real MolmoAct2 fine-tune runs (controller traces, 30 Hz), classified and measured.

    python sim/twin/failure_match/real_forensics.py [--offsets W9]
        -> sim/twin/out/failure_match/real_closes.jsonl (one row per close), real_episodes.jsonl, real_summary.json,
           real_brick_xy.json

Episodes as data/tools/episodes.py: operator / loop `start` key to the next start / q; only the `running` phase;
episodes with < 3 s of running are operator aborts and are counted but not analysed. Models: ft8k (loop_control,
15:18-15:29), ft5.5k with the trained prompts, ft5.5k with untrained prompts (14:58-15:01). Pick-stage closes only
for the pick statistics (bring / place closes are listed with their stage).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from collections import Counter

import numpy as np
from closes import TWIN, Geometry, describe_close, segment_closes, t_reach

DATA = TWIN / "data/real_rollouts/home"
OUT = TWIN / "out/failure_match"
EXPORTS = {"ft_cal2_456/rtc_int8": "ft5.5k", "ft_cal2_456_8k/rtc_int8": "ft8k"}
KEYS = ("low_mm", "tip_z_mm", "t_s", "dwell_before_close_s", "descent_during_close_mm", "tilt_deg")
TRAINED = set(json.loads((DATA / "inference/deploy/molmoact2/finetune_cal2_456.json").read_text())["tasks"])


def tag(export: str | None) -> str | None:
    for k, v in EXPORTS.items():
        if export and export.endswith(k):
            return v
    return None


def episodes(path: str):
    recs = [json.loads(line) for line in open(path) if line.strip()]
    if not recs or recs[0].get("kind") != "run":
        return None, []
    eps, cur = [], None
    for r in recs[1:]:
        if r["kind"] == "key" and r["key"] in ("start", "q"):
            if cur and cur["ticks"]:
                eps.append(cur)
            cur = {"start": r["wall"], "ticks": [], "infer": [], "keys": []} if r["key"] == "start" else None
            continue
        if cur is None:
            continue
        if r["kind"] == "tick":
            cur["ticks"].append(r)
        elif r["kind"] == "infer":
            cur["infer"].append(r)
        elif r["kind"] == "key":
            cur["keys"].append(r["key"])
    if cur and cur["ticks"]:
        eps.append(cur)
    return tag(recs[0]["args"].get("export")), eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offsets", default="W9")
    args = ap.parse_args()
    geo = Geometry(args.offsets)
    OUT.mkdir(parents=True, exist_ok=True)
    close_rows, ep_rows, aborted = [], [], Counter()
    for f in sorted(glob.glob(str(DATA / "desmond/pi05_traces/*.jsonl"))):
        model, eps = episodes(f)
        if not model:
            continue
        for i, e in enumerate(eps):
            run = [t for t in e["ticks"] if t["phase"] == "running"]
            prompts = [x.get("prompt") for x in e["infer"] if x.get("prompt")]
            pick_prompts = sorted({p for p in prompts if p.startswith("pick")})
            trained = all(p in TRAINED for p in prompts)
            group = model if model == "ft8k" else ("ft5.5k" if trained else "ft5.5k_untrained")
            if len(run) < 90:
                aborted[group] += 1
                continue
            M = np.array([t["measured"] for t in run], float)
            C = np.array([t["cmd"] for t in run], float)
            st = [t["stage"] for t in run]
            wall = np.array([t["wall"] for t in run])
            # recorder gaps: ticks are nominally 1/30 s apart
            gaps = np.diff(wall)
            ep_id = f"{os.path.basename(f)[5:-6]}#{i}"
            cl = []
            for i0, i1 in segment_closes(M[:, 5]):
                d = describe_close(M, C, i0, i1, geo)
                ots = round(float(wall[d["shut"]] - wall[d["onset"]]), 2) if d["onset"] is not None else None
                d.update(ep=ep_id, group=group, stage=st[i0], t_s=round(float(wall[i0] - wall[0]), 2),
                         onset_to_shut_s=ots)
                # did the arm still descend while closing? tip z at command onset minus at the shut instant
                if d["onset"] is not None:
                    d["descent_during_close_mm"] = round(geo.at(M[d["onset"]])["tip_z_mm"] - d["tip_z_mm"], 1)
                # dwell: how long the tip had been within 10 mm of its shut height before the command onset
                if d["onset"] is not None:
                    zs = [geo.at(M[j])["tip_z_mm"] for j in range(max(0, d["onset"] - 90), d["onset"] + 1)]
                    dw = 0
                    for z in reversed(zs):
                        if abs(z - d["tip_z_mm"]) <= 10:
                            dw += 1
                        else:
                            break
                    d["dwell_before_close_s"] = round(dw / 30.0, 2)
                # reached bring while still closed on this close (loop grasp check / operator)
                d["to_bring"] = any(s != "pick" for s in st[i0:i1])
                cl.append(d)
            pick = [c for c in cl if c["stage"] == "pick"]
            for k, c in enumerate(pick):
                c["attempt"] = k + 1
                c["outcome"] = ("empty" if c["empty"] else "lifted" if (c["lifted_30mm_0p5s"] or c["to_bring"])
                                else "brick_not_lifted")
            close_rows += cl
            first = pick[0] if pick else None
            success = any(c["outcome"] == "lifted" for c in pick)
            k_succ = next((c["attempt"] for c in pick if c["outcome"] == "lifted"), None)
            ep_rows.append(dict(
                ep=ep_id, group=group, start=time.strftime("%H:%M:%S", time.gmtime(wall[0] - 8 * 3600)),
                dur_s=round(float(wall[-1] - wall[0]), 1), stages="/".join(dict.fromkeys(st)), prompts=pick_prompts,
                keys=e["keys"], n_pick_closes=len(pick), n_empty=sum(c["empty"] for c in pick),
                first=(first["outcome"] if first else "no_close"), success=success, success_attempt=k_succ,
                t_first_close_s=first["t_s"] if first else None, t_reach30_s=t_reach(M, geo),
                max_gap_s=round(float(gaps.max()), 3) if len(gaps) else 0,
                frac_gap_gt_0p1=round(float((gaps > 0.1).mean()), 4)))
    with open(OUT / "real_closes.jsonl", "w") as fh:
        for r in close_rows:
            fh.write(json.dumps(r, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o)) + "\n")
    with open(OUT / "real_episodes.jsonl", "w") as fh:
        for r in ep_rows:
            fh.write(json.dumps(r) + "\n")
    summ = {"offsets": args.offsets, "aborted_lt_3s": dict(aborted)}
    for grp in ("ft8k", "ft5.5k", "ft5.5k_untrained"):
        E = [e for e in ep_rows if e["group"] == grp]
        P = [c for c in close_rows if c["group"] == grp and c["stage"] == "pick"]
        F = [c for c in P if c["attempt"] == 1]
        R2 = [c for c in P if c["attempt"] > 1]
        if not E:
            continue

        def stats(rows, key):
            v = np.array([r[key] for r in rows if r.get(key) is not None], float)
            return {"n": len(v), "median": round(float(np.median(v)), 1), "p10": round(float(np.percentile(v, 10)), 1),
                    "p90": round(float(np.percentile(v, 90)), 1)} if len(v) else None

        summ[grp] = {
            "episodes": len(E),
            "episodes_with_a_pick_close": sum(e["n_pick_closes"] > 0 for e in E),
            "first_attempt_success": sum(e["first"] == "lifted" for e in E),
            "episode_success": sum(e["success"] for e in E),
            "first_close": dict(Counter(e["first"] for e in E)),
            "pick_closes": len(P), "pick_closes_empty": sum(c["empty"] for c in P),
            "pick_close_outcomes": dict(Counter(c["outcome"] for c in P)),
            "closes_per_episode": stats([{"v": e["n_pick_closes"]} for e in E], "v"),
            "episodes_ge3_closes": sum(e["n_pick_closes"] >= 3 for e in E),
            "t_reach30_s": stats(E, "t_reach30_s"), "t_first_close_s": stats(E, "t_first_close_s"),
            "first_close_empty": {k: stats([c for c in F if c["empty"]], k) for k in KEYS},
            "first_close_brick": {k: stats([c for c in F if not c["empty"]], k) for k in KEYS},
            "all_empty": {k: stats([c for c in P if c["empty"]], k) for k in KEYS[:2] + KEYS[3:4]},
            "all_brick": {k: stats([c for c in P if not c["empty"]], k) for k in KEYS[:2] + KEYS[3:4]},
            "regrasp_empty_frac": round(sum(c["empty"] for c in R2) / max(1, len(R2)), 3),
        }
    (OUT / "real_summary.json").write_text(json.dumps(summ, indent=1))
    # where the real target bricks were (brick-position proxy per episode: the tip at the lifting close, else the
    # median tip of its pick closes); twin_eval.py --target_xy deploy samples from these
    xy = []
    for e in ep_rows:
        P = [c for c in close_rows if c["ep"] == e["ep"] and c["stage"] == "pick"]
        L = [c for c in P if c["outcome"] == "lifted"]
        if P and e["group"] != "ft5.5k_untrained":
            v = L[0]["tip_xy_mm"] if L else np.median([c["tip_xy_mm"] for c in P], 0).tolist()
            xy.append({"ep": e["ep"], "group": e["group"], "lifted": bool(L),
                       "xy_m": [round(float(u) / 1e3, 4) for u in v]})
    (OUT / "real_brick_xy.json").write_text(json.dumps(xy, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
