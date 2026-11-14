"""MolmoAct2 INT8 vs FP16 (OpenVINO, XPS GPU) on recorded cal2 frames: latency, action agreement, gripper behaviour.

For ~N frames spread over the three phases (labels from the 3-phase relabel), each build predicts the chunk R times
(the export draws its own flow noise, so repeats measure sampling spread). Reports per build: latency; per phase:
  * fp16_self   = mean |FP16 run i - FP16 run j|        (FP16's own sampling spread)
  * int8_vs_fp16 = mean |INT8 run i - FP16 run j|        (if ~= fp16_self, INT8 costs nothing measurable)
  * mae_vs_recorded per build (first 10 steps) and the gripper channel separately.
One model in memory at a time.

    ~/inference/.venv-loop/bin/python precision_compare.py --frames 36 --repeats 3
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

CAMS = ("wrist", "top")
PICK = "pick up the brick and lift it up"
PLACE = "put the brick in the {bowl} bowl"


def frames_by_phase(root: Path, n: int, seed: int):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    labels = {l["episode_index"]: l for l in map(json.loads, open(root / "meta/dlsj/labels.jsonl"))}
    eps = [e for e, l in labels.items() if not l.get("excluded") and l.get("grasp_frame") and l.get("inspection_frame")]
    rng = np.random.default_rng(seed)
    eps = list(rng.choice(eps, size=min(len(eps), n), replace=False))
    ds = LeRobotDataset("local/" + root.name, root=root, episodes=[int(e) for e in eps], video_backend="pyav")
    starts = {int(ds.hf_dataset[i]["episode_index"]): i for i in range(len(ds)) if int(ds.hf_dataset[i]["frame_index"]) == 0}
    out = []
    for k, e in enumerate(eps):
        l = labels[int(e)]
        g, insp, n_fr = l["grasp_frame"], l["inspection_frame"], l["n_frames"]
        phase = ("grasp", "present", "place")[k % 3]
        fr = {"grasp": max(0, g - 15), "present": (g + insp) // 2, "place": min(n_fr - 12, insp + 10)}[phase]
        idx = starts[int(e)] + fr
        item = ds[idx]
        rec = np.stack([np.asarray(ds.hf_dataset[min(idx + j, starts[int(e)] + n_fr - 1)]["action"], np.float32)
                        for j in range(10)])
        task = PICK if phase != "place" else PLACE.format(bowl=l.get("bowl_final") or "yellow")
        imgs = {c: (np.asarray(item[f"observation.images.{c}"]) * 255).clip(0, 255).astype(np.uint8) for c in CAMS}
        out.append({"ep": int(e), "frame": int(fr), "phase": phase, "task": task, "imgs": imgs,
                    "state": np.asarray(item["observation.state"], np.float32), "rec": rec})
    return out


def run_build(export: str, frames, repeats: int, cache: str):
    from physicalai.inference import InferenceModel

    t = time.perf_counter()
    m = InferenceModel(export, backend="openvino", device="GPU", PERFORMANCE_HINT="LATENCY", NUM_STREAMS="1",
                       CACHE_DIR=cache)
    load_s = time.perf_counter() - t
    preds, lat = [], []
    for f in frames:
        obs = {"state": f["state"][None], "task": [f["task"]]}
        for c in CAMS:
            obs[f"images.{c}"] = np.ascontiguousarray(f["imgs"][c][None])  # (1,3,H,W) uint8
        runs = []
        for r in range(repeats + (1 if f is frames[0] else 0)):  # one extra warm-up on the first frame
            t = time.perf_counter()
            a = np.asarray(m.predict_action_chunk(obs), np.float32)
            dt = (time.perf_counter() - t) * 1e3
            if not (f is frames[0] and r == 0):
                runs.append(a)
                lat.append(dt)
        preds.append(np.stack(runs))
    del m
    return preds, {"load_s": round(load_s, 1), "median_ms": round(float(np.median(lat)), 1),
                   "p90_ms": round(float(np.percentile(lat, 90)), 1)}


def pair_mae(a, b, steps=10, same=False):
    """mean |a_i - b_j| over run pairs (i != j when same), first `steps` actions; returns (all joints, gripper)."""
    vals, grip = [], []
    for i, j in itertools.product(range(len(a)), range(len(b))):
        if same and i >= j:
            continue
        d = np.abs(a[i][:steps] - b[j][:steps])
        vals.append(d[:, :5].mean())
        grip.append(d[:, 5].mean())
    return float(np.mean(vals)), float(np.mean(grip))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path("~/inference/data/duplo_teleop_cal2_3phase").expanduser()))
    ap.add_argument("--int8", default=str(Path("~/inference/exports/molmoact2_so101_rtc_int8").expanduser()))
    ap.add_argument("--fp16", default=str(Path("~/inference/exports/molmoact2_so101_rtc_fp16").expanduser()))
    ap.add_argument("--frames", type=int, default=36)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--cache", default=str(Path("~/inference/ov_cache").expanduser()))
    ap.add_argument("--json", default=str(Path("~/inference/logs/molmo/precision_compare.json").expanduser()))
    a = ap.parse_args()

    frames = frames_by_phase(Path(a.data), a.frames, 0)
    print(f"{len(frames)} frames:", {p: sum(f['phase'] == p for f in frames) for p in ('grasp', 'present', 'place')},
          flush=True)
    res = {}
    preds = {}
    for name, path in (("int8", a.int8), ("fp16", a.fp16)):
        preds[name], res[name] = run_build(path, frames, a.repeats, a.cache)
        print(name, res[name], flush=True)
    for phase in ("grasp", "present", "place", "all"):
        idx = [k for k, f in enumerate(frames) if phase == "all" or f["phase"] == phase]
        self16 = np.mean([pair_mae(preds["fp16"][k], preds["fp16"][k], same=True) for k in idx], 0)
        self8 = np.mean([pair_mae(preds["int8"][k], preds["int8"][k], same=True) for k in idx], 0)
        cross = np.mean([pair_mae(preds["int8"][k], preds["fp16"][k]) for k in idx], 0)
        rec = {n: np.mean([[np.abs(preds[n][k][:, :10, :5] - frames[k]["rec"][None, :, :5]).mean(),
                            np.abs(preds[n][k][:, :10, 5] - frames[k]["rec"][None, :, 5]).mean()] for k in idx], 0)
               for n in ("int8", "fp16")}
        res[phase] = {"n": len(idx),
                      "fp16_self_joints_deg": round(self16[0], 2), "fp16_self_gripper": round(self16[1], 2),
                      "int8_self_joints_deg": round(self8[0], 2), "int8_self_gripper": round(self8[1], 2),
                      "int8_vs_fp16_joints_deg": round(cross[0], 2), "int8_vs_fp16_gripper": round(cross[1], 2),
                      "mae_vs_recorded_joints": {n: round(float(rec[n][0]), 2) for n in rec},
                      "mae_vs_recorded_gripper": {n: round(float(rec[n][1]), 2) for n in rec}}
        print(phase, json.dumps(res[phase]), flush=True)
    Path(a.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
