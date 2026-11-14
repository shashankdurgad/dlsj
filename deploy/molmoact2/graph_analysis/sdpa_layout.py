"""Expert attention as in the real graph: q from (1,30,H,D) and K/V from (1,L,H,D) transposed to (1,H,L,D) right before
SDPA (OpenVINO fuses the transposes into the SDPA input order) vs contiguous (1,H,L,D) inputs."""
import json, time
import numpy as np
import openvino as ov
import openvino.opset13 as ops
core = ov.Core(); H, D = 8, 96
def build(lq, lk, transposed, mask_rows):
    shp_q = [1, lq, H, D] if transposed else [1, H, lq, D]
    shp_k = [1, lk, H, D] if transposed else [1, H, lk, D]
    pq = ops.parameter(shp_q, ov.Type.f16, name="q"); pk = ops.parameter(shp_k, ov.Type.f16, name="k")
    pv = ops.parameter(shp_k, ov.Type.f16, name="v"); pm = ops.parameter([1, 1, mask_rows, lk], ov.Type.f16, name="m")
    q, k, v = pq, pk, pv
    if transposed:
        perm = ops.constant(np.int64([0, 2, 1, 3]))
        q, k, v = ops.transpose(pq, perm), ops.transpose(pk, perm), ops.transpose(pv, perm)
    out = ops.scaled_dot_product_attention(q, k, v, pm, ops.constant(np.float16(1 / np.sqrt(D))), causal=False)
    return ov.Model([out], [pq, pk, pv, pm], "attn")
rng = np.random.default_rng(0); res = {}
for name, lq, lk, mrows in (("cross_30x650", 30, 650, 1), ("self_30x30", 30, 30, 30)):
    for transposed in (False, True):
        shp_q = (1, lq, H, D) if transposed else (1, H, lq, D); shp_k = (1, lk, H, D) if transposed else (1, H, lk, D)
        data = {"q": rng.standard_normal(shp_q).astype(np.float16), "k": rng.standard_normal(shp_k).astype(np.float16),
                "v": rng.standard_normal(shp_k).astype(np.float16), "m": np.zeros((1, 1, mrows, lk), np.float16)}
        cfg = {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": "f16", "PERF_COUNT": "YES"}
        cm = core.compile_model(build(lq, lk, transposed, mrows), "GPU", cfg)
        r = cm.create_infer_request()
        for _ in range(50): r.infer(data)
        dev = []; kern = set()
        for _ in range(200):
            r.infer(data)
            dev.append(sum(p.real_time.total_seconds() for p in r.profiling_info) * 1e6)
            kern |= {f"{p.node_type}:{p.exec_type}" for p in r.profiling_info if p.real_time.total_seconds() > 0}
        res[f"{name}/{'transposed_in' if transposed else 'contiguous'}"] = {"device_us_median": round(float(np.median(dev)), 1), "kernels": sorted(kern)}
print(json.dumps(res, indent=1))
