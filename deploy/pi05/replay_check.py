"""Replay check: recorded observations -> `OpenVINOPi05Policy` -> compare with the recorded future actions.

For ~N frames spread over episodes (half from the pick segment, half from the place segment, by task string):

1. builds an `Observation` from the recorded frame (wrist/top/inspect RGB uint8 + state in degrees) and runs
   `OpenVINOPi05Policy.predict_chunk(obs, <frame's recorded task>, noise=<seeded>)`;
2. calls `InferenceModel.predict_action_chunk` directly with an independently built input dict (the
   `studio_bench.py` layout) and the same noise -> parity (max |diff|) of the connector against the runtime;
3. compares the chunk with the recorded actions a[t .. t+49] (padding masked): MAE per joint over the first 10
   and all 50 steps, next to a "hold still" baseline (repeat the current state) so the numbers mean something.

Read-only on the dataset; GPU inference only (run niced). Studio library env or the loop env on the XPS:

    nice -n 5 ~/inference/.venv-loop/bin/python deploy/pi05/replay_check.py \
        ~/inference/exports/pi05_junction_int8 --dataset ~/desmond/dlsj/data/duplo_teleop_rgb \
        --frames 50 --max-episode 212 --json ~/inference/logs/pi05/replay_check_int8.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dlsj.contracts import Observation  # noqa: E402
from dlsj.loop.policies import OpenVINOPi05Policy  # noqa: E402
from dlsj.loop.poses import MOTORS  # noqa: E402
from dlsj.loop.replay import episode_range, open_dataset, task_names, to_uint8_hwc  # noqa: E402

CAMS = ("wrist", "top", "inspect")
PICK_PREFIX = "pick"


def select_frames(ds, episodes: list[int], n: int, rng: np.random.Generator, margin: int = 10) -> list[tuple[int, str]]:
    """(local dataset row, segment) pairs: n/2 pick frames and n/2 place frames, one random frame per episode
    segment, cycling over a shuffled episode list."""
    hf = ds.hf_dataset.with_format(None)
    index = np.asarray(hf["index"])
    task_idx = np.asarray(hf["task_index"])
    pos = {int(a): i for i, a in enumerate(index)}
    names = task_names(ds)
    picks, places = [], []
    for ep in rng.permutation(episodes):
        lo, hi = episode_range(ds, int(ep))
        rows = [pos[a] for a in range(lo, hi - margin) if a in pos]  # leave room for the future actions
        if not rows:
            continue
        tasks = [names[int(task_idx[r])] for r in rows]
        seg = {"pick": [r for r, t in zip(rows, tasks) if t.startswith(PICK_PREFIX)],
               "place": [r for r, t in zip(rows, tasks) if not t.startswith(PICK_PREFIX)]}
        if seg["pick"] and len(picks) < n // 2:
            picks.append((int(rng.choice(seg["pick"])), "pick"))
        if seg["place"] and len(places) < n - n // 2:
            places.append((int(rng.choice(seg["place"])), "place"))
        if len(picks) >= n // 2 and len(places) >= n - n // 2:
            break
    return picks + places


def direct_inputs(item, cams, task: str, noise: np.ndarray) -> dict:
    """Independent of the connector: the studio_bench.py observation layout from the raw dataset item."""
    d = {f"images.{c}": to_uint8_hwc(item[f"observation.images.{c}"]).transpose(2, 0, 1)[None].copy() for c in cams}
    d["state"] = np.asarray(item["observation.state"], np.float32)[None]
    d["task"] = [task]
    d["noise"] = noise
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export_dir", type=Path)
    ap.add_argument("--dataset", type=Path, default=Path("~/desmond/dlsj/data/duplo_teleop_rgb").expanduser())
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--max-episode", type=int, default=212, help="use episodes < this (the training split)")
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--ablate", action="store_true",
                    help="also run each frame with deliberately wrong inputs (BGR, wrist/top swapped, the other "
                         "segment's task, other noise): the correct mapping should score best")
    args = ap.parse_args()

    t0 = time.perf_counter()
    policy = OpenVINOPi05Policy(args.export_dir.expanduser(), args.device, cache_dir=args.cache_dir)
    print(f"loaded {args.export_dir} on {args.device} in {time.perf_counter() - t0:.1f}s; cameras {policy.cameras}, "
          f"chunk {policy.chunk_len}, noise {policy._noise_shape}", flush=True)
    policy.warmup(2)
    n = policy.chunk_len
    probe = open_dataset(args.dataset)
    fps = float(probe.fps)
    episodes = [e for e in range(probe.meta.total_episodes) if e < args.max_episode]
    ds = open_dataset(args.dataset, delta_timestamps={"action": [i / fps for i in range(n)]})
    rng = np.random.default_rng(args.seed)
    frames = select_frames(ds, episodes, args.frames, rng)
    print(f"{len(frames)} frames from {len(episodes)} episodes (fps {fps})", flush=True)

    rows = []
    for k, (r, seg) in enumerate(frames):
        item = ds[r]
        task = str(item["task"])
        state = np.asarray(item["observation.state"], np.float32)
        obs = Observation(state=state, images={c: to_uint8_hwc(item[f"observation.images.{c}"]) for c in CAMS})
        noise = np.random.default_rng(1000 + k).standard_normal((1, *policy._noise_shape), dtype=np.float32)
        chunk = policy.predict_chunk(obs, task, noise=noise)
        ms = policy.latency_ms
        direct = policy.model.predict_action_chunk(direct_inputs(item, policy.cameras, task, noise.copy()))
        direct = np.asarray(direct, np.float32).reshape(-1, direct.shape[-1])[:, :6]
        gt = np.asarray(item["action"], np.float32)  # (n, 6) future actions a[t..t+n-1]
        pad = np.asarray(item.get("action_is_pad", np.zeros(n, bool)), bool)
        valid = ~pad
        err = np.abs(chunk - gt)
        hold = np.abs(state[None] - gt)
        rows.append({
            "row": int(r), "episode": int(item["episode_index"]), "frame": int(item["frame_index"]), "segment": seg,
            "task": task, "ms": round(ms, 1), "parity_max_abs": float(np.max(np.abs(chunk - direct))),
            "mae10": err[:10][valid[:10]].mean(0).tolist(), "mae50": err[valid].mean(0).tolist(),
            "hold10": hold[:10][valid[:10]].mean(0).tolist(), "hold50": hold[valid].mean(0).tolist(),
            "first_step_err": err[0].tolist(), "valid": int(valid.sum()),
        })
        if args.ablate:
            other = ("place the brick in the yellow bowl" if task.startswith(PICK_PREFIX)
                     else "pick up the brick and hold it in front of the inspection camera")
            imgs = obs.images
            variants = {
                "bgr": (Observation(state, {c: np.ascontiguousarray(v[..., ::-1]) for c, v in imgs.items()}), task, noise),
                "swap_wrist_top": (Observation(state, {**imgs, "wrist": imgs["top"], "top": imgs["wrist"]}), task, noise),
                "other_task": (obs, other, noise),
                "other_noise": (obs, task, np.random.default_rng(5000 + k).standard_normal(noise.shape, dtype=np.float32)),
            }
            rows[-1]["ablate"] = {}
            for name, (o, t, nz) in variants.items():
                e = np.abs(policy.predict_chunk(o, t, noise=nz) - gt)
                rows[-1]["ablate"][name] = {"mae10": float(e[:10][valid[:10]].mean()), "mae50": float(e[valid].mean())}
        print(f"[{k + 1}/{len(frames)}] ep {rows[-1]['episode']:3d} f {rows[-1]['frame']:4d} {seg:5s} "
              f"mae10 {np.mean(rows[-1]['mae10']):5.2f} mae50 {np.mean(rows[-1]['mae50']):5.2f} "
              f"(hold {np.mean(rows[-1]['hold50']):5.2f}) parity {rows[-1]['parity_max_abs']:.2e} {ms:.0f} ms",
              flush=True)

    def agg(sel, key):
        return np.mean([x[key] for x in sel], axis=0) if sel else np.full(6, np.nan)

    summary = {"export": str(args.export_dir), "dataset": str(args.dataset), "frames": len(rows),
               "device": args.device, "joints": MOTORS}
    print(f"\nMAE per joint (deg; gripper 0-100) vs recorded actions; export {args.export_dir.name}")
    print(f"{'segment':8s} {'metric':8s} " + " ".join(f"{m[:11]:>11s}" for m in MOTORS) + "        mean")
    for name, sel in (("all", rows), ("pick", [x for x in rows if x["segment"] == "pick"]),
                      ("place", [x for x in rows if x["segment"] == "place"])):
        summary[name] = {}
        for key in ("mae10", "mae50", "hold10", "hold50"):
            v = agg(sel, key)
            summary[name][key] = [round(float(x), 3) for x in v]
            print(f"{name:8s} {key:8s} " + " ".join(f"{x:11.2f}" for x in v) + f"  {np.mean(v):10.2f}")
    if args.ablate:
        summary["ablation"] = {"correct": {"mae10": float(np.mean([np.mean(x["mae10"]) for x in rows])),
                                           "mae50": float(np.mean([np.mean(x["mae50"]) for x in rows]))}}
        for name in rows[0]["ablate"]:
            summary["ablation"][name] = {m: float(np.mean([x["ablate"][name][m] for x in rows])) for m in ("mae10", "mae50")}
        print("\nablation (mean over joints and frames, same frames):")
        for name, v in summary["ablation"].items():
            print(f"  {name:15s} mae10 {v['mae10']:6.2f}  mae50 {v['mae50']:6.2f}")
    par = [x["parity_max_abs"] for x in rows]
    lat = [x["ms"] for x in rows]
    summary["parity_max_abs"] = float(np.max(par))
    summary["latency_ms"] = {"median": float(np.median(lat)), "p90": float(np.percentile(lat, 90))}
    print(f"\nparity vs direct InferenceModel (same inputs + noise): max |diff| = {np.max(par):.3e}")
    print(f"connector latency: median {np.median(lat):.0f} ms, p90 {np.percentile(lat, 90):.0f} ms (niced, shared GPU)")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
        print("wrote", args.json)


if __name__ == "__main__":
    main()
