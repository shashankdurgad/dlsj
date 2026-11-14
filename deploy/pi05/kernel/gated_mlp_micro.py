"""Micro-benchmark of the GPU GatedMLP fusion on one synthetic MLP block (debug-caps build only).

Builds `down(act(x @ Wg) * (x @ Wu))` with INT8-symmetric per-channel compressed weights (the same
`Convert(i8) * scale` decompression NNCF/Studio produce), for Swish (SwiGLU: what the stock FuseGatedMLP pass
matches) and Gelu-tanh / Gelu-erf (Gemma's GeGLU, matched only with `patch_gated_mlp_gelu.py` +
`OV_GPU_GATED_MLP_GELU=1`), at π0.5's expert (50×1024→4096) and VLM (968×2048→16384) shapes. Each case is compiled
with the fusion off and on, in a fresh subprocess (env options are read at compile time). Reports which oneDNN
implementation the fused primitive got and the GPU time per block.

    source ~/inference/ovbuild/env.sh
    ~/inference/.venv-ovdbg/bin/python gated_mlp_micro.py --out gmlp_micro.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SHAPES = {"expert": (50, 1024, 4096), "vlm": (968, 2048, 16384)}
ACTS = ("swish", "gelu_tanh", "gelu_erf")


def build(m: int, k: int, n: int, act: str, wtype: str = "i8"):
    import openvino as ov
    import openvino.opset13 as ops

    rng = np.random.default_rng(0)

    def weight(rows: int, cols: int):  # [out, in], transpose_b=True like the exported Linear layers
        q = rng.integers(-127, 128, size=(rows, cols), dtype=np.int8)
        s = (rng.random((rows, 1)) * 0.01 + 1e-3).astype(np.float16)
        if wtype == "f16":
            return ops.constant((q.astype(np.float32) * s.astype(np.float32)).astype(np.float16))
        return ops.multiply(ops.convert(ops.constant(q), "f16"), ops.constant(s))

    x = ops.parameter([1, m, k], np.float32, name="x")
    xh = ops.convert(x, "f16")
    g = ops.matmul(xh, weight(n, k), False, True)
    u = ops.matmul(xh, weight(n, k), False, True)
    if act == "swish":
        a = ops.swish(g, ops.constant(np.array(1.0, np.float16)))  # beta must match the f16 input
    else:
        a = ops.gelu(g, "tanh" if act == "gelu_tanh" else "erf")
    d = ops.matmul(ops.multiply(a, u), weight(k, n), False, True)
    return ov.Model([ops.convert(d, "f32")], [x], f"mlp_{act}_{m}")


def run_case(m: int, k: int, n: int, act: str, iters: int, wtype: str = "i8") -> dict:
    import openvino as ov

    core = ov.Core()
    model = build(m, k, n, act, wtype)
    c = core.compile_model(model, "GPU", {"PERFORMANCE_HINT": "LATENCY", "PERF_COUNT": "YES"})
    req = c.create_infer_request()
    x = np.random.default_rng(1).standard_normal((1, m, k)).astype(np.float32) * 0.5
    req.set_tensor("x", ov.Tensor(x))
    for _ in range(5):
        req.infer()
    wall, gpu, prims = [], [], {}
    for _ in range(iters):
        t = time.perf_counter()
        req.infer()
        wall.append((time.perf_counter() - t) * 1e3)
        tot = 0.0
        for p in req.get_profiling_info():
            if str(p.status).endswith("EXECUTED") and p.real_time.total_seconds() > 0:
                tot += p.real_time.total_seconds() * 1e3
                key = f"{p.node_type}|{p.exec_type}"
                prims.setdefault(key, []).append(p.real_time.total_seconds() * 1e6)
        gpu.append(tot)
    out = req.get_output_tensor(0).data.copy()
    return {"wall_ms": float(np.median(wall)), "gpu_ms": float(np.median(gpu)),
            "prims_us": {k2: float(np.median(v)) for k2, v in prims.items()},
            "out_abs_mean": float(np.abs(out).mean()), "out_head": out.ravel()[:64].tolist()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--_case", nargs=5)
    ap.add_argument("--wtypes", default="i8,f16", help="weight types: i8 (INT8-sym compressed) and/or f16")
    a = ap.parse_args()
    if a._case:
        part, act, fused, iters, wtype = a._case
        print(json.dumps(run_case(*SHAPES[part], act, int(iters), wtype)))
        return
    res = []
    for part, act, wtype in [(p, ac, w) for p in SHAPES for ac in ACTS for w in a.wtypes.split(",")]:
        row = {"part": part, "act": act, "weights": wtype}
        for fused in ("off", "on"):
            env = dict(os.environ)
            env["OV_GPU_DISABLE_GATED_MLP_FUSION"] = "1" if fused == "off" else "0"
            env["OV_GPU_GATED_MLP_GELU"] = "1"
            p = subprocess.run([sys.executable, __file__, "--_case", part, act, fused, str(a.iters), wtype],
                               env=env, capture_output=True, text=True)
            line = next((ln for ln in p.stdout.splitlines() if ln.startswith("{")), None)
            row[fused] = json.loads(line) if line else {"error": p.stderr.strip().splitlines()[-3:]}
        on, off = row["on"], row["off"]
        if "out_head" in on and "out_head" in off:
            h1, h0 = np.array(on.pop("out_head")), np.array(off.pop("out_head"))
            row["max_rel_diff_head"] = float(np.abs(h1 - h0).max() / (np.abs(h0).max() + 1e-9))
        gm = {k: v for k, v in on.get("prims_us", {}).items() if "GatedMLP" in k or "gated" in k.lower()}
        print(f"{part:6s} {act:9s} {wtype:3s} off {off.get('gpu_ms', float('nan')):7.3f} ms | on {on.get('gpu_ms', float('nan')):7.3f} ms"
              f" | fused prims {gm or 'none'} | rel diff {row.get('max_rel_diff_head', 'n/a')}", flush=True)
        res.append(row)
    if a.out:
        a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
