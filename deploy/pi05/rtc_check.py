"""Offline RTC check on recorded episodes: does the RTC export join consecutive chunks smoothly?

For frames spread over the training episodes, simulate the controller's hand-off:

1. chunk A from the observation at row t (unguided: first chunk);
2. after `e = 50 - (horizon + delay)` executed actions, chunk B from the observation at row t + e, given A's
   unexecuted tail A[e:] as `prev_chunk_left_over` and `delay` actions of expected latency;
3. the robot switches to B at its action `delay` (the actions before it passed during inference), so the jump the
   arm sees is |B[delay] - A[e + delay]| per joint.

Each pair is run with the configured guidance (RTC) and with guidance ~0 (the same export behaving like a plain
re-plan), with identical noise, so the difference is RTC alone. Also reports B's error against the recorded
actions (RTC must not make the new chunk ignore the newer observation) and the latency of both calls.

    nice -n 5 ~/inference/.venv-loop/bin/python deploy/pi05/rtc_check.py \
        ~/inference/exports/pi05_ft9060_rtc_tok64_5step_int8 --frames 40 --json ~/inference/logs/pi05/rtc/check.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dlsj.loop.poses import MOTORS  # noqa: E402
from dlsj.loop.replay import open_dataset, task_names, to_uint8_hwc  # noqa: E402
from replay_check import select_frames  # noqa: E402
from rtc_control import rtc_inputs  # noqa: E402

CAMS = ("wrist", "top", "inspect")


def obs_from(item, task: str, noise: np.ndarray) -> dict:
    d = {f"images.{c}": to_uint8_hwc(item[f"observation.images.{c}"]).transpose(2, 0, 1)[None].copy() for c in CAMS}
    d["state"] = np.asarray(item["observation.state"], np.float32)[None]
    d["task"] = [task]
    d["noise"] = noise
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export_dir", type=Path)
    ap.add_argument("--dataset", type=Path, default=Path("~/desmond/dlsj/data/duplo_teleop_rgb").expanduser())
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--max-episode", type=int, default=212)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--delay", type=int, default=5, help="actions of latency (~150 ms at 30 fps)")
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--cache-dir", default=str(Path("~/inference/ov_cache").expanduser()))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    from physicalai.inference import InferenceModel

    model = InferenceModel(str(args.export_dir.expanduser()), backend="openvino", device=args.device,
                           PERFORMANCE_HINT="LATENCY", NUM_STREAMS="1", CACHE_DIR=args.cache_dir)
    shapes = {f.name: tuple(f.shape) for f in model.input_features}
    noise_shape = (1, *shapes["noise"]) if "noise" in shapes else None
    ds = open_dataset(args.dataset.expanduser())
    names = task_names(ds)
    rng = np.random.default_rng(args.seed)
    frames = select_frames(ds, list(range(min(args.max_episode, ds.meta.total_episodes))), args.frames, rng, margin=60)
    e = 50 - (args.horizon + args.delay)
    d = args.delay

    def run(obs) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        a = np.asarray(model.predict_action_chunk(obs), np.float32)
        return a, (time.perf_counter() - t0) * 1e3

    rows, lat = [], {"first": [], "rtc": [], "plain": []}
    for k, (row, seg) in enumerate(frames):
        it_a, it_b = ds[row], ds[row + e]
        if int(it_a["episode_index"]) != int(it_b["episode_index"]):
            continue
        task_a, task_b = names[int(it_a["task_index"])], names[int(it_b["task_index"])]
        if task_a != task_b:  # the hand-off within one instruction (a prompt switch starts unguided anyway)
            continue
        nA = rng.standard_normal(noise_shape).astype(np.float32) if noise_shape else None
        nB = rng.standard_normal(noise_shape).astype(np.float32) if noise_shape else None
        A, ms = run(obs_from(it_a, task_a, nA) | rtc_inputs(None, 0, args.horizon, args.guidance, 50, 6))
        lat["first"].append(ms)
        tail = A[e:]
        out = {}
        for mode, g in (("rtc", args.guidance), ("plain", 1e-8)):
            B, ms = run(obs_from(it_b, task_b, nB) | rtc_inputs(tail, d, args.horizon, g, 50, 6))
            lat[mode].append(ms)
            jump = np.abs(B[d] - A[e + d])  # what the arm sees at the switch
            # recorded future actions from the switch tick
            rec = np.stack([np.asarray(ds[row + e + d + j]["action"], np.float32) for j in range(10)])
            out[mode] = {"jump": jump.tolist(), "mae10": np.abs(B[d:d + 10] - rec).mean(0).tolist(),
                         "seam_vel": float(np.abs((B[d] - A[e + d - 1]) - (A[e + d - 1] - A[e + d - 2])).mean())}
        rows.append({"row": int(row), "segment": seg, "task": task_b, **out})
        print(f"{k:3d} {seg:5s} jump rtc {np.mean(out['rtc']['jump']):6.2f}  plain {np.mean(out['plain']['jump']):6.2f}"
              f"  | gripper rtc {out['rtc']['jump'][5]:5.2f} plain {out['plain']['jump'][5]:5.2f}", flush=True)

    def agg(mode, key):
        return np.asarray([r[mode][key] for r in rows], np.float32)

    summary = {"export": str(args.export_dir), "pairs": len(rows), "horizon": args.horizon, "delay": d,
               "guidance": args.guidance, "executed_before_switch": e, "joints": MOTORS}
    for mode in ("rtc", "plain"):
        j = agg(mode, "jump")
        summary[mode] = {"jump_mean_deg": round(float(j.mean()), 3), "jump_p90_deg": round(float(np.percentile(j, 90)), 3),
                         "jump_per_joint": np.round(j.mean(0), 3).tolist(),
                         "seam_accel_mean": round(float(agg(mode, "seam_vel").mean()), 3),
                         "mae10_vs_recorded": round(float(agg(mode, "mae10").mean()), 3),
                         "latency_median_ms": round(float(np.median(lat[mode])), 1)}
    summary["first_chunk_latency_median_ms"] = round(float(np.median(lat["first"])), 1)
    print(json.dumps(summary, indent=1))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"summary": summary, "pairs": rows}, indent=1))


if __name__ == "__main__":
    main()
