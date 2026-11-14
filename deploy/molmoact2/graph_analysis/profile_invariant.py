"""How much GPU time do the step-invariant time-conditioning nodes (time_embed, per-block adaLN modulation, final-layer
modulation) take in a MolmoAct2 OpenVINO export, and do they survive OpenVINO's compile-time transformations?"""
import json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path("~/inference/deploy/molmoact2").expanduser()))
from harness import ZEROSHOT, load_frames  # noqa: E402
from optimize import wait_arm_free  # noqa: E402
import openvino as ov
from physicalai.inference import InferenceModel

exp = Path(sys.argv[1]).expanduser()
KEYS = ("time_embed", "modulation")
def invariant(name: str) -> bool:
    return "action_expert" in name and any(k in name for k in KEYS)

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

core = ov.Core()
xml = next(p for p in exp.glob("*.xml") if "tokenizer" not in p.name)
cm = core.compile_model(str(xml), "GPU", {"PERFORMANCE_HINT": "LATENCY", "PERF_COUNT": "YES"})
rt = cm.get_runtime_model()
rt_inv = [op for op in rt.get_ordered_ops() if invariant(op.get_friendly_name())]
rt_types = defaultdict(int)
for op in rt_inv:
    rt_types[op.get_rt_info()["layerType"].astype(str) if "layerType" in op.get_rt_info() else op.get_type_name()] += 1
req = cm.create_infer_request()
req.infer(inputs)
wait_arm_free(lambda s: print(s, flush=True))
tot = defaultdict(float); n = defaultdict(int); allt = 0.0; iters = 5
for _ in range(iters):
    req.infer(inputs)
    for p in req.profiling_info:
        us = p.real_time.total_seconds() * 1e6
        allt += us
        if invariant(p.node_name):
            k = "time_embed" if "time_embed" in p.node_name else ("final_modulation" if "final_layer" in p.node_name else "block_modulation")
            tot[k] += us; n[k] += 1
res = {"export": str(exp), "runtime_invariant_nodes": len(rt_inv), "runtime_invariant_node_types": dict(rt_types),
       "per_chunk_ms": {k: round(v / iters / 1e3, 2) for k, v in tot.items()},
       "per_chunk_kernels": {k: n[k] // iters for k in n},
       "invariant_total_ms": round(sum(tot.values()) / iters / 1e3, 2),
       "all_kernels_ms_with_profiling": round(allt / iters / 1e3, 1)}
res["invariant_share"] = round(res["invariant_total_ms"] / res["all_kernels_ms_with_profiling"], 3)
print(json.dumps(res, indent=1))
