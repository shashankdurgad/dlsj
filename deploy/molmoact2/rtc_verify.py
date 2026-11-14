"""Verify an RTC-patched MolmoAct2 export (export.py --rtc) on recorded cal2 frames.

1. Equivalence: guidance ~0 must reproduce the plain export (same checkpoint/steps/precision) on the same frames.
2. Hand-off: for frame pairs (row t, row t + e) of one episode, chunk A from t (unguided), chunk B from t + e with A's
   unexecuted tail A[e:] (runtime degrees), delay d. The arm switches to B at action d, so it sees |B[d] - A[e + d]|.
   Reported for guidance g (RTC) and ~0 (plain re-plan), identical inputs otherwise.

    python rtc_verify.py RTC_EXPORT --plain PLAIN_EXPORT [--pairs 24 --horizon 15 --delay 10 --guidance 3]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "pi05"))
sys.path.insert(0, str(Path("~/inference/deploy/pi05").expanduser()))
from rtc_control import rtc_inputs  # noqa: E402

CAMS = ("wrist", "top")
PROMPT = "pick up the brick and lift it up"


def load_pairs(root: Path, n: int, e: int, seed: int = 0):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    labels = {l["episode_index"]: l for l in map(json.loads, open(root / "meta/dlsj/labels.jsonl"))}
    eps = sorted(k for k, l in labels.items() if not l.get("excluded") and l.get("grasp_frame"))
    rng = np.random.default_rng(seed)
    eps = [int(x) for x in rng.choice(eps, size=min(n, len(eps)), replace=False)]
    ds = LeRobotDataset("local/" + root.name, root=root, episodes=eps, video_backend="pyav")
    ep_col = np.asarray(ds.hf_dataset["episode_index"])
    fr_col = np.asarray(ds.hf_dataset["frame_index"])
    pairs = []
    for ep in eps:
        g = labels[ep]["grasp_frame"]
        t = max(0, g - 40)  # approach phase, where hand-off jumps matter most
        base = int(np.flatnonzero((ep_col == ep) & (fr_col == 0))[0])
        items = []
        for fr in (t, t + e):
            it = ds[base + fr]
            items.append({"state": np.asarray(it["observation.state"], np.float32),
                          "imgs": {c: (np.asarray(it[f"observation.images.{c}"]) * 255).clip(0, 255).astype(np.uint8)
                                   for c in CAMS}})
        pairs.append(items)
    return pairs


class Runner:
    def __init__(self, export: str, device: str, cache: str):
        from physicalai.inference import InferenceModel

        self.m = InferenceModel(export, backend="openvino", device=device, PERFORMANCE_HINT="LATENCY", NUM_STREAMS="1",
                                CACHE_DIR=str(Path(cache).expanduser()))
        self.rtc = "prev_chunk_left_over" in set(self.m.adapter.input_names or [])

    def __call__(self, item, extra=None):
        obs = {"state": item["state"][None], "task": [PROMPT]}
        for c, img in item["imgs"].items():
            obs[f"images.{c}"] = np.ascontiguousarray(img[None])
        if not self.rtc:
            return np.asarray(self.m.predict_action_chunk(obs), np.float32)
        x = obs
        for p in self.m.preprocessors:
            x = p(x)
        x.update({k: np.asarray(v) for k, v in (extra or rtc_inputs(None, 0, 1, 1e-8, 30, 6)).items()})
        out = self.m.runner.run(self.m.adapter, self.m._prepare_inputs(x))
        for p in self.m.postprocessors:
            out = p(out)
        a = np.asarray(out["action"], np.float32)
        return a[0] if a.ndim == 3 else a


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rtc_export")
    ap.add_argument("--plain", help="plain export of the same checkpoint/steps/precision (equivalence check)")
    ap.add_argument("--data", default="~/inference/data/duplo_teleop_cal2_3phase")
    ap.add_argument("--pairs", type=int, default=24)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--delay", type=int, default=10, help="actions of latency (~340 ms at 30 fps)")
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--cache-dir", default="~/inference/ov_cache")
    ap.add_argument("--json")
    a = ap.parse_args()
    e = 30 - (a.horizon + a.delay)  # actions executed from A before B is requested (controller threshold h + d)
    e = max(e, 1)
    pairs = load_pairs(Path(a.data).expanduser(), a.pairs, e)
    res = {"rtc_export": a.rtc_export, "horizon": a.horizon, "delay": a.delay, "guidance": a.guidance, "executed": e}
    r = Runner(a.rtc_export, a.device, a.cache_dir)
    if not r.rtc:
        raise SystemExit("export has no RTC inputs")
    A = [r(p[0]) for p in pairs]
    lat = []
    jumps = {"rtc": [], "plain": []}
    for k, p in enumerate(pairs):
        tail = A[k][e:]
        for mode, g in (("rtc", a.guidance), ("plain", 1e-8)):
            t = time.perf_counter()
            B = r(p[1], rtc_inputs(tail, a.delay, a.horizon, g, 30, 6))
            lat.append((time.perf_counter() - t) * 1e3)
            jumps[mode].append(np.abs(B[a.delay] - A[k][e + a.delay]))
    for mode in jumps:
        j = np.stack(jumps[mode])
        res[mode] = {"jump_mean_deg": round(float(j[:, :5].mean()), 3), "jump_p90_deg": round(float(np.percentile(j[:, :5], 90)), 3),
                     "gripper_jump": round(float(j[:, 5].mean()), 3), "per_joint": np.round(j.mean(0), 3).tolist()}
    res["latency_median_ms"] = round(float(np.median(lat)), 1)
    if a.plain:
        del r
        p = Runner(a.plain, a.device, a.cache_dir)
        diffs = [np.abs(p(pr[0]) - A[k]).mean() for k, pr in enumerate(pairs)]
        res["equivalence_vs_plain_mean_abs"] = round(float(np.mean(diffs)), 4)
    print(json.dumps(res, indent=1))
    if a.json:
        Path(a.json).expanduser().write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
