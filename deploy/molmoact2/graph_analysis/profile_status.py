"""Execution status of the step-invariant nodes + which kernels the action expert's attention/transposes use."""
import json, sys
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path("~/inference/deploy/molmoact2").expanduser()))
from harness import load_frames  # noqa: E402
from optimize import wait_arm_free  # noqa: E402
import openvino as ov
from physicalai.inference import InferenceModel

exp = Path(sys.argv[1]).expanduser()
frames = load_frames("~/inference/data/duplo_teleop_cal2_3phase", 60,
                     cache="~/inference/exports/molmo_opt/zeroshot/frames_zeroshot_60.npz")
f = frames[0]
obs = {"state": f["state"][None], "task": [f["task"]]}
for c in ("wrist", "top"):
    obs[f"images.{c}"] = np.ascontiguousarray(f["imgs"][c][None])
m = InferenceModel(str(exp), backend="openvino", device="GPU", PERFORMANCE_HINT="LATENCY", NUM_STREAMS="1",
                   CACHE_DIR=str(Path("~/inference/ov_cache").expanduser()))
cap = {}
orig = m.adapter.predict
m.adapter.predict = lambda inputs: (cap.setdefault("inputs", inputs), orig(inputs))[1]
m.predict_action_chunk(obs)
inputs = cap["inputs"]
del m
cm = ov.Core().compile_model(str(next(p for p in exp.glob("*.xml") if "tokenizer" not in p.name)), "GPU",
                             {"PERFORMANCE_HINT": "LATENCY", "PERF_COUNT": "YES"})
req = cm.create_infer_request()
req.infer(inputs)
wait_arm_free(lambda s: print(s, flush=True))
iters = 5
inv_status = Counter(); ae_type = defaultdict(float); ae_cnt = Counter(); sdpa_kernels = Counter(); sdpa_ms = defaultdict(float)
for _ in range(iters):
    req.infer(inputs)
    for p in req.profiling_info:
        n = p.node_name; us = p.real_time.total_seconds() * 1e6
        if "action_expert" in n and ("time_embed" in n or "modulation" in n):
            inv_status[str(p.status).split(".")[-1]] += 1
        if "action_expert" in n:
            t = p.node_type
            ae_type[t] += us; ae_cnt[t] += 1
            if "scaled_dot_product" in n.lower() or t.lower().startswith("scaled") or "sdpa" in (p.exec_type or "").lower():
                key = f"{t} | {p.exec_type}"
                sdpa_kernels[key] += 1; sdpa_ms[key] += us
res = {"invariant_node_status_counts": {k: v // iters for k, v in inv_status.items()},
       "action_expert_ms_by_node_type": {k: round(v / iters / 1e3, 2) for k, v in sorted(ae_type.items(), key=lambda kv: -kv[1])[:12]},
       "action_expert_kernel_counts": {k: v // iters for k, v in ae_cnt.most_common(12)},
       "expert_attention_kernels": {k: {"count": sdpa_kernels[k] // iters, "ms": round(sdpa_ms[k] / iters / 1e3, 2)} for k in sdpa_kernels}}
print(json.dumps(res, indent=1))
