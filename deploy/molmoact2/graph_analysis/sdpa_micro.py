"""Micro-benchmark MolmoAct2 expert attention shapes on the XPS GPU: fused SDPA op vs decomposed MatMul-Softmax-MatMul
(fusion disabled for the decomposed model), and a manual-softmax variant that the SDPA fusion pattern can't match."""
import json, time
import numpy as np
import openvino as ov
import openvino.opset13 as ops

core = ov.Core()
H, D = 8, 96
def build(kind, lq, lk, mask):
    q = ops.parameter([1, H, lq, D], ov.Type.f16, name="q")
    k = ops.parameter([1, H, lk, D], ov.Type.f16, name="k")
    v = ops.parameter([1, H, lk, D], ov.Type.f16, name="v")
    m = ops.parameter([1, 1, lq if not mask == "cross" else 1, lk], ov.Type.f16, name="m")
    scale = ops.constant(np.float16(1.0 / np.sqrt(D)))
    if kind == "sdpa":
        out = ops.scaled_dot_product_attention(q, k, v, m, scale, causal=False)
    else:
        s = ops.matmul(q, k, False, True)
        s = ops.add(ops.multiply(s, scale), m)
        if kind == "decomp":
            p = ops.softmax(s, -1)
        else:  # manual softmax: exp(s - max) / sum  (not matched by SDPA fusion)
            mx = ops.reduce_max(s, ops.constant(np.int64([-1])), keep_dims=True)
            e = ops.exp(ops.subtract(s, mx))
            p = ops.divide(e, ops.reduce_sum(e, ops.constant(np.int64([-1])), keep_dims=True))
        out = ops.matmul(p, v, False, False)
    return ov.Model([out], [q, k, v, m], kind)

res = {}
rng = np.random.default_rng(0)
for name, lq, lk, mask in (("cross_30x650", 30, 650, "cross"), ("self_30x30", 30, 30, "self")):
    data = {"q": rng.standard_normal((1, H, lq, D)).astype(np.float16), "k": rng.standard_normal((1, H, lk, D)).astype(np.float16),
            "v": rng.standard_normal((1, H, lk, D)).astype(np.float16),
            "m": np.zeros((1, 1, 1 if mask == "cross" else lq, lk), np.float16)}
    outs = {}
    for kind in ("sdpa", "decomp", "manual"):
        cfg = {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": "f16"}
        if kind != "sdpa":
            cfg["GPU_ENABLE_SDPA_OPTIMIZATION"] = "NO"
        cm = core.compile_model(build(kind, lq, lk, mask), "GPU", cfg)
        req = cm.create_infer_request()
        for _ in range(50): req.infer(data)
        ts = []
        for _ in range(500):
            t = time.perf_counter(); req.infer(data); ts.append((time.perf_counter() - t) * 1e6)
        outs[kind] = np.asarray(req.get_output_tensor(0).data, np.float32).copy()
        # device-side kernel time via profiling
        cmp = core.compile_model(build(kind, lq, lk, mask), "GPU", dict(cfg, PERF_COUNT="YES"))
        rp = cmp.create_infer_request(); rp.infer(data)
        dev = []
        for _ in range(50):
            rp.infer(data); dev.append(sum(p.real_time.total_seconds() for p in rp.profiling_info) * 1e6)
        kern = sorted({p.exec_type for p in rp.profiling_info if p.real_time.total_seconds() > 0})
        res[f"{name}/{kind}"] = {"wall_us_median": round(float(np.median(ts)), 1), "device_us_median": round(float(np.median(dev)), 1),
                                 "kernels": kern[:6]}
    res[f"{name}/max_abs_diff_decomp_vs_sdpa"] = float(np.abs(outs["decomp"] - outs["sdpa"]).max())
    res[f"{name}/max_abs_diff_manual_vs_sdpa"] = float(np.abs(outs["manual"] - outs["sdpa"]).max())
print(json.dumps(res, indent=1))
