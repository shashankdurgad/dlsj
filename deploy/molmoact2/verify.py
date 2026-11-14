"""Load a MolmoAct2 OpenVINO export with InferenceModel; one unguided and one RTC-guided call; frame sanity + latency.

    python verify.py <export_dir> [--device CPU|GPU] [--iters 10]
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pi05"))
sys.path.insert(0, str(Path("~/inference/deploy/pi05").expanduser()))
from rtc_control import rtc_inputs  # noqa: E402
from physicalai.inference import InferenceModel  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("export_dir")
ap.add_argument("--device", default="CPU")
ap.add_argument("--iters", type=int, default=3)
ap.add_argument("--cache-dir", default=None)
a = ap.parse_args()
cfg = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1"}
if a.cache_dir:
    cfg["CACHE_DIR"] = str(Path(a.cache_dir).expanduser())
t = time.perf_counter()
m = InferenceModel(a.export_dir, backend="openvino", device=a.device, **cfg)
load_s = time.perf_counter() - t
feats = {f.name: tuple(f.shape) for f in m.input_features}
print("inputs:", feats)
cams = [n.removeprefix("images.") for n in feats if n.startswith("images.")]
rng = np.random.default_rng(0)
state = np.array([-2.0, -59.0, 56.0, 60.0, 17.5, 29.0], np.float32)  # our degrees, a typical work pose
obs = {"state": state[None], "task": ["pick up the brick"]}
for c in cams:
    obs[f"images.{c}"] = rng.integers(0, 255, (1, 3, 480, 640), dtype=np.uint8)
chunk = feats.get("prev_chunk_left_over", (30, 6))[0]
res = {"device": a.device, "load_s": round(load_s, 1), "cams": cams, "chunk": chunk}
for mode in ("unguided", "guided"):
    if mode == "unguided":
        extra = rtc_inputs(None, 0, 10, 3.0, chunk, 6)
    else:
        extra = rtc_inputs(prev[10:], 4, 10, 3.0, chunk, 6)
    o = obs | extra if "prev_chunk_left_over" in feats else obs
    out = np.asarray(m.predict_action_chunk(o))
    ts = []
    for _ in range(a.iters):
        t = time.perf_counter(); out = np.asarray(m.predict_action_chunk(o)); ts.append((time.perf_counter() - t) * 1e3)
    if mode == "unguided":
        prev = out
    res[mode] = {"shape": list(out.shape), "finite": bool(np.isfinite(out).all()),
                 "first_action": np.round(out[0], 1).tolist(), "first_minus_state": np.round(out[0] - state, 1).tolist(),
                 "median_ms": round(float(np.median(ts)), 1), "p90_ms": round(float(np.percentile(ts, 90)), 1)}
    if mode == "guided":
        res[mode]["switch_jump"] = np.round(np.abs(out[4] - prev[14]), 2).tolist()
print(json.dumps(res, indent=1))
