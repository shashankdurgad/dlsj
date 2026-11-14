"""Shared MolmoAct2 (OpenVINO) evaluation harness: recorded frames -> predictions -> latency/quality vs a reference.

* `load_frames(root, n, prompts, cams)`: n frames spread evenly over the 3 phases (grasp / present / place) of a
  3-phase-labelled LeRobot dataset (labels.jsonl with grasp_frame, inspection_frame, bowl_final), plus the next 10
  recorded actions. Frames are cached to an .npz next to the logs so every variant sees identical inputs.
* `predict(export_dir, frames, device)`: loads one export with physicalai InferenceModel, returns predictions
  (n, chunk, 6) and latency stats (warm-up excluded). One model in memory at a time.
* `compare(pred, ref, frames)`: per phase + overall mean |pred - ref| (joints, gripper; first 10 actions) and the
  error vs the recorded demo, plus the "no quality loss" verdict.

Prompts: a dict {"grasp": str (may contain {brick}), "present": str, "place": str with {bowl}}. {brick} comes from
the episode's label (pick_task "pick up the <colour> brick") or the `brick` default. `crops` ({cam: (x, y, w, h)})
crops cameras the way the training set did (MolmoAct2 fine-tune: top 0,0,560,480). Two presets: ZEROSHOT (what the arm used
with the released checkpoint) and THREE_PHASE (the relabelled-dataset strings the LoRA fine-tune will use).
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

ZEROSHOT = {"grasp": "pick up the brick and lift it up", "present": "pick up the brick and lift it up",
            "place": "put the brick in the {bowl} bowl"}
THREE_PHASE = {"grasp": "grasp the brick", "present": "lift the brick and hold it still in front of the inspection camera",
               "place": "place the brick in the {bowl} bowl and release it"}
PHASES = ("grasp", "present", "place")
# "no quality loss" = within the INT8-vs-FP16 noise floor measured on 14 Nov (0.26 deg joints, 0.08 gripper),
# with margin; and the error vs the human demo must not get worse by more than DEMO_TOL.
JOINT_TOL, GRIP_TOL, DEMO_TOL = 0.5, 0.3, 0.2


def load_frames(root: str | Path, n: int = 60, prompts: dict | None = None, cams=("wrist", "top"), seed: int = 0,
                cache: str | Path | None = None, crops: dict | None = None, brick: str = "red") -> list[dict]:
    prompts = prompts or ZEROSHOT
    crops = {k: tuple(v) for k, v in (crops or {}).items()}
    cache = Path(cache).expanduser() if cache else None
    if cache and cache.exists():
        z = np.load(cache, allow_pickle=True)
        frames = list(z["frames"])
        if len(frames) == n and all(f["task"] == _task(prompts, f["phase"], f["bowl"], f.get("brick", brick))
                                    and f.get("crops", {}) == crops for f in frames):
            return frames
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(root).expanduser()
    labels = {l["episode_index"]: l for l in map(json.loads, open(root / "meta/dlsj/labels.jsonl"))}
    eps = sorted(e for e, l in labels.items()
                 if not l.get("excluded") and l.get("grasp_frame") and l.get("inspection_frame"))
    rng = np.random.default_rng(seed)
    eps = [int(e) for e in rng.choice(eps, size=min(len(eps), n), replace=False)]
    ds = LeRobotDataset("local/" + root.name, root=root, episodes=eps, video_backend="pyav")
    ep_col = np.asarray(ds.hf_dataset["episode_index"])
    fr_col = np.asarray(ds.hf_dataset["frame_index"])
    frames = []
    for k, e in enumerate(eps):
        lab = labels[e]
        g, insp, n_fr = lab["grasp_frame"], lab["inspection_frame"], lab.get("n_frames") or int((ep_col == e).sum())
        phase = PHASES[k % 3]
        fr = {"grasp": max(0, g - 15), "present": (g + insp) // 2, "place": min(n_fr - 12, insp + 10)}[phase]
        base = int(np.flatnonzero((ep_col == e) & (fr_col == 0))[0])
        item = ds[base + fr]
        rec = np.stack([np.asarray(ds.hf_dataset[min(base + fr + j, base + n_fr - 1)]["action"], np.float32)
                        for j in range(10)])
        bowl = lab.get("bowl_final") or lab.get("bowl") or "yellow"
        m = re.fullmatch(r"pick up the ([a-z]+) brick", str(lab.get("pick_task", "")))
        ep_brick = m.group(1) if m else brick
        imgs = {}
        for c in cams:
            img = np.asarray(item[f"observation.images.{c}"])  # (3, H, W) float in [0, 1]
            if c in crops:
                x, y, w, h = crops[c]
                img = img[:, y:y + h, x:x + w]
            imgs[c] = (img * 255).clip(0, 255).astype(np.uint8)
        frames.append({"ep": e, "frame": int(fr), "phase": phase, "bowl": bowl, "brick": ep_brick, "crops": crops,
                       "task": _task(prompts, phase, bowl, ep_brick),
                       "imgs": imgs, "state": np.asarray(item["observation.state"], np.float32), "rec": rec})
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, frames=np.asarray(frames, dtype=object))
    return frames


def _task(prompts: dict, phase: str, bowl: str, brick: str = "red") -> str:
    return prompts[phase].replace("{bowl}", bowl).replace("{brick}", brick)


def predict(export_dir: str | Path, frames: list[dict], device: str = "GPU", cache_dir: str | None = None,
            extra_inputs: dict | None = None, **config) -> tuple[np.ndarray, dict]:
    from physicalai.inference import InferenceModel

    cfg = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", **config}
    if cache_dir:
        cfg["CACHE_DIR"] = str(Path(cache_dir).expanduser())
    t = time.perf_counter()
    m = InferenceModel(str(Path(export_dir).expanduser()), backend="openvino", device=device, **cfg)
    load_s = time.perf_counter() - t
    names = set(m.adapter.input_names or [])
    rtc = "prev_chunk_left_over" in names
    if rtc:  # RTC-patched export: inputs the runtime preprocessors drop -> add after preprocessing; guidance ~0
        chunk = 30
        extra_inputs = {"prev_chunk_left_over": np.zeros((1, chunk, 6), np.float32),
                        "inference_delay": np.int64(0), "execution_horizon": np.int64(1),
                        "max_guidance_weight": np.float32(1e-8), **(extra_inputs or {})}

    def call(obs):
        if not rtc:
            return m.predict_action_chunk(obs)
        x = {k: v for k, v in obs.items() if k not in extra_inputs}
        for p in m.preprocessors:
            x = p(x)
        x.update({k: np.asarray(v) for k, v in extra_inputs.items()})
        out = m.runner.run(m.adapter, m._prepare_inputs(x))
        for p in m.postprocessors:
            out = p(out)
        a = np.asarray(out["action"])
        return a[0] if a.ndim == 3 else a

    preds, lat = [], []
    for k, f in enumerate(frames):
        obs = {"state": f["state"][None], "task": [f["task"]], **(extra_inputs or {})}
        for c, img in f["imgs"].items():
            obs[f"images.{c}"] = np.ascontiguousarray(img[None])
        if k == 0:
            call(obs)  # warm-up
        t = time.perf_counter()
        a = np.asarray(call(obs), np.float32)
        lat.append((time.perf_counter() - t) * 1e3)
        preds.append(a)
    del m
    return np.stack(preds), {"load_s": round(load_s, 1), "median_ms": round(float(np.median(lat)), 1),
                             "p90_ms": round(float(np.percentile(lat, 90)), 1)}


def compare(pred: np.ndarray, ref: np.ndarray, frames: list[dict], steps: int = 10) -> dict:
    out = {}
    for phase in (*PHASES, "all"):
        idx = [k for k, f in enumerate(frames) if phase == "all" or f["phase"] == phase]
        if not idx:
            continue
        d = np.abs(pred[idx, :steps] - ref[idx, :steps])
        rec = np.stack([frames[k]["rec"][:steps] for k in idx])
        e_pred = np.abs(pred[idx, :steps] - rec)
        e_ref = np.abs(ref[idx, :steps] - rec)
        out[phase] = {"n": len(idx), "vs_ref_joints": round(float(d[..., :5].mean()), 3),
                      "vs_ref_gripper": round(float(d[..., 5].mean()), 3),
                      "demo_err_joints": round(float(e_pred[..., :5].mean()), 3),
                      "demo_err_ref_joints": round(float(e_ref[..., :5].mean()), 3),
                      "demo_err_gripper": round(float(e_pred[..., 5].mean()), 3),
                      "demo_err_ref_gripper": round(float(e_ref[..., 5].mean()), 3)}
    a = out["all"]
    a["pass"] = bool(a["vs_ref_joints"] <= JOINT_TOL and a["vs_ref_gripper"] <= GRIP_TOL
                     and a["demo_err_joints"] - a["demo_err_ref_joints"] <= DEMO_TOL)
    return out
