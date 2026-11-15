"""Offline replay of the grasp residual's robot path on a recorded LeRobot episode (no robot, no policy).

Per frame of the episode: the recorded state is the "measured" reading, the recorded actions from this frame on are
the "plan", the recorded task string drives the gate. The wrist frame goes through the same detector + tracker as on
the robot (`residual.WristDetector.process`), then `GraspResidual.step`. Reports detector / tracker / residual timing
against the 30 Hz budget, detection rate while picking, the size of the corrections and two geometry checks:
* the wrist target at the gripper-close frame, relative to the fingertip (a grasped brick sits near (0, 0, ~0));
* with --top: the top-camera target (detector on the first top frame, nearest the close-frame fingertip) vs that
  fingertip (the top camera -> table geometry check).

    python deploy/smolvla/residual_replay.py --dataset <lerobot root> --episode 3 \
        --detector ~/inference/models/brick-detector-v1 --device CPU [--weights zero|<npz>] [--out replay.jsonl]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import residual as RES  # noqa: E402


def load_episode(root: str | Path, episode: int, cams=("wrist", "top")) -> dict:
    """LeRobot v3 episode -> {"state", "action", "tasks" (per frame), "<cam>": [BGR frames]}."""
    import cv2
    import pandas as pd

    root = Path(root).expanduser()
    eps = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(root / "meta/episodes/*/*.parquet")))])
    e = eps[eps.episode_index == episode].iloc[0]
    data = pd.read_parquet(root / f"data/chunk-{int(e['data/chunk_index']):03d}/file-{int(e['data/file_index']):03d}.parquet")
    d = data[data.episode_index == episode].sort_values("frame_index")
    tasks = pd.read_parquet(root / "meta/tasks.parquet")
    names = {int(i): t for t, i in zip(tasks.index, tasks.task_index, strict=True)}
    out = {"state": np.stack(d["observation.state"].to_numpy()).astype(float),
           "action": np.stack(d["action"].to_numpy()).astype(float),
           "tasks": [names[int(i)] for i in d["task_index"]], "fps": 30}
    n = len(d)
    for cam in cams:
        key = f"videos/observation.images.{cam}"
        path = root / f"videos/observation.images.{cam}/chunk-{int(e[key + '/chunk_index']):03d}/" \
                      f"file-{int(e[key + '/file_index']):03d}.mp4"
        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_POS_MSEC, float(e[key + "/from_timestamp"]) * 1e3)
        frames = []
        while len(frames) < n:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        out[cam] = frames
    return out


def close_frame(state, action, pick, thr: float = 12.0) -> int | None:
    """First pick frame whose commanded gripper falls below `thr` (validate.py's close rule, simplified)."""
    for i in range(len(action)):
        if pick[i] and action[i, 5] < thr and state[i, 5] > thr:
            return i
    return None


def top_target(ep: dict, kin: RES.Kin, detector, close: int | None):
    """Detector on the first top frame (zoomed tiles, as the loop's survey); the box whose table point is nearest the
    close-frame fingertip."""
    if detector is None or not ep.get("top") or close is None:
        return None, None, None
    top = RES.TopCamera()
    tip = kin.fk(ep["state"][close])[0]
    best = None
    import perception as P

    for d in P.detect_tiled(detector, ep["top"][0], (0, 60, 560, 480)):
        x0, y0, x1, y1 = d.box
        X = top.pixel_to_table((x0 + x1) / 2, (y0 + y1) / 2)
        if X is None:
            continue
        dist = float(np.linalg.norm(X[:2] - tip[:2]))
        if best is None or dist < best[0]:
            best = (dist, X, d.box)
    if best is None:
        return None, None, None
    return best[1], best[2], best[0]


def replay(ep: dict, policy, detector, top_seed: bool = False, log=None) -> dict:
    res = RES.GraspResidual(policy)
    cam = RES.WT.WristCamera(offsets_rad=res.kin.off.tolist(), tip_local=res.meta["tip_local"])
    RES.warm_fin_cache(cam)
    wd = RES.WristDetector(None, detector, RES.SeededTracker(cam), lambda: None)
    pick = [RES.is_pick_prompt(t) for t in ep["tasks"]]
    close = close_frame(ep["state"], ep["action"], pick)
    xyz, top_box, top_err = top_target(ep, res.kin, detector, close) if top_seed else (None, None, None)
    res.reset(xyz)
    wd.reset(None if xyz is None else xyz[:2])
    n = len(ep["state"])
    tick_ms, det_ms, track_ms, deltas, valid, rel_close = [], [], [], [], [], None
    cmd_prev = None
    for i in range(n):
        m, goal = ep["state"][i], ep["action"][i]
        det = RES.Det()
        rec = None
        if pick[i] and i < len(ep.get("wrist", [])):
            rec = wd.process(ep["wrist"][i], m)
            det_ms.append(rec["det_ms"])
            track_ms.append(rec["track_ms"])
            t = rec["target"]
            det = RES.Det(bool(t.valid), np.asarray(t.rel, float))
            valid.append(bool(t.valid))
            if i == close:
                rel_close = None if not t.valid else np.asarray(t.rel, float)
        t0 = time.perf_counter()
        g2, info = res.step(goal, m, cmd_prev, ep["action"][i:i + RES.PLAN_STEPS], det, pick[i])
        tick_ms.append((time.perf_counter() - t0) * 1e3)
        deltas.append(np.abs(np.asarray(g2) - goal))
        cmd_prev = np.asarray(g2, float)
        if log is not None:
            row = {"i": i, "pick": pick[i], "goal": goal, "goal_res": g2, **{k: v for k, v in info.items()}}
            if rec is not None:
                t = rec["target"]
                row.update(det_valid=bool(t.valid), det_rel=t.rel, det_box=t.box, n_boxes=rec["n_boxes"],
                           det_ms=rec["det_ms"])
            log.write(json.dumps({k: (np.round(np.asarray(v, float), 4).tolist() if isinstance(v, np.ndarray) else v)
                                  for k, v in row.items()}, default=str) + "\n")
    deltas = np.stack(deltas)

    def pct(x, q):
        return round(float(np.percentile(x, q)), 3) if len(x) else None

    return {"frames": n, "pick_frames": int(sum(pick)), "close_frame": close,
            "det_rate_pick": round(float(np.mean(valid)), 3) if valid else None,
            "det_ms_p50": pct(det_ms, 50), "det_ms_p99": pct(det_ms, 99),
            "track_ms_p50": pct(track_ms, 50), "track_ms_p99": pct(track_ms, 99),
            "residual_ms_p50": pct(tick_ms, 50), "residual_ms_p99": pct(tick_ms, 99),
            "residual_ms_max": round(float(np.max(tick_ms)), 3),
            "max_abs_goal_change": np.round(deltas.max(0), 3).tolist(),
            "max_change_outside_pick": round(float(deltas[~np.asarray(pick)].max()), 6) if not all(pick) else 0.0,
            "wrist_rel_at_close_mm": None if rel_close is None else np.round(rel_close * 1e3, 1).tolist(),
            "top_target_xyz": None if xyz is None else np.round(xyz, 4).tolist(), "top_box": top_box,
            "top_target_minus_close_tip_mm": None if top_err is None else round(top_err * 1e3, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--episode", type=int, action="append", help="repeatable; default 0")
    ap.add_argument("--detector", default=RES.DEFAULT_DETECTOR)
    ap.add_argument("--device", default="CPU", help="detector device (NPU on the XPS)")
    ap.add_argument("--weights", default="zero")
    ap.add_argument("--residual-device", default="numpy")
    ap.add_argument("--top", action="store_true", help="seed from the top camera (as --residual-top-seed)")
    ap.add_argument("--out", default="", help="per-frame JSON lines")
    a = ap.parse_args()
    det = RES.make_wrist_detector(a.detector, a.device)
    pol = RES.load_policy(a.weights, a.residual_device)
    with open(a.out or os.devnull, "w") as log:
        for e in a.episode or [0]:
            ep = load_episode(a.dataset, e)
            print(json.dumps({"episode": e, **replay(ep, pol, det, a.top, log if a.out else None)}), flush=True)


if __name__ == "__main__":
    main()
