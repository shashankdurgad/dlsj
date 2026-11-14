"""Per-kernel GPU profile of a Studio π0.5 IR (`pi05.xml`) with OpenVINO's own counters.

Compiles the IR once without and once with `PERF_COUNT=YES` (LATENCY hint, 1 stream), runs synthetic static-shape
inputs, then joins `InferRequest.get_profiling_info()` (GPU event time per primitive) with the compiled runtime model
(primitive type, implementation, tensor shapes, topological order). Writes one JSON with every executed primitive and
prints the aggregated tables (`summarize_kernel.py profile` renders them as Markdown).

    PY=~/inference/physical-ai-studio/library/.venv/bin/python      # release 2026.4.1
    $PY profile_kernels.py ~/inference/exports/pi05_junction_int8/pi05.xml --out prof_int8.json
    # debug-caps build (GatedMLP): source ~/inference/ovbuild/env.sh; OV_GPU_DISABLE_GATED_MLP_FUSION=0 \
    #   OV_GPU_GATED_MLP_GELU=1 ~/inference/.venv-ovdbg/bin/python profile_kernels.py ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kernel_lib as K  # noqa: E402


def make_inputs(model, seed: int = 0, prompt_tokens: int = 30) -> dict[str, np.ndarray]:
    """Synthetic inputs for every (static) model input. Values do not change kernel choice for a static graph."""
    import openvino as ov

    rng = np.random.default_rng(seed)
    out = {}
    for p in model.inputs:
        name = p.get_any_name()
        shape = list(p.get_partial_shape().to_shape())
        et = p.get_element_type()
        if et == ov.Type.boolean:
            a = np.ones(shape, bool)
            if "prompt" in name and len(shape) == 2:
                a[:, prompt_tokens:] = False
        elif et.is_integral():
            a = rng.integers(2, 1000, size=shape).astype(et.to_dtype())
        else:
            a = rng.standard_normal(shape).astype(et.to_dtype())
            if "image" in name:
                a = np.clip(a, -1, 1)
        out[name] = a
    return out


def time_wall(compiled, inputs, iters: int, warmup: int) -> list[float]:
    req = compiled.create_infer_request()
    for k, v in inputs.items():
        req.set_tensor(k, __import__("openvino").Tensor(v))
    for _ in range(warmup):
        req.infer()
    ms = []
    for _ in range(iters):
        t = time.perf_counter()
        req.infer()
        ms.append((time.perf_counter() - t) * 1e3)
    return ms


def runtime_nodes(compiled) -> list[dict]:
    """Primitive metadata from the compiled (execution) graph, in topological order."""
    nodes = []
    for i, op in enumerate(compiled.get_runtime_model().get_ordered_ops()):
        rt = op.get_rt_info()

        def g(key: str) -> str:
            try:
                return str(rt[key].astype(str)) if hasattr(rt[key], "astype") else str(rt[key])
            except Exception:
                return ""

        shapes_out = [list(op.get_output_partial_shape(j).to_shape()) if op.get_output_partial_shape(j).is_static
                      else [] for j in range(op.get_output_size())]
        shapes_in = [list(op.get_input_partial_shape(j).to_shape()) if op.get_input_partial_shape(j).is_static
                     else [] for j in range(op.get_input_size())]
        nodes.append({"order": i, "name": op.get_friendly_name(), "layer_type": g("layerType"),
                      "exec_type": g("execType"), "orig": g("originalLayersNames")[:300],
                      "out_shapes": shapes_out, "in_shapes": shapes_in})
    return nodes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("xml", type=Path)
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--steps", type=int, default=10, help="denoising steps unrolled in the IR (expert attribution)")
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--config", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--cache_dir", default="", help="CACHE_DIR (default none: profile what is compiled now)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import openvino as ov

    core = ov.Core()
    model = core.read_model(str(args.xml.expanduser()))
    inputs = make_inputs(model)
    cfg = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", **dict(kv.split("=", 1) for kv in args.config)}
    if args.cache_dir:
        cfg["CACHE_DIR"] = args.cache_dir

    t0 = time.perf_counter()
    compiled = core.compile_model(model, args.device, cfg)
    compile_s = time.perf_counter() - t0
    wall_nopc = time_wall(compiled, inputs, args.iters, args.warmup)
    del compiled

    compiled = core.compile_model(model, args.device, {**cfg, "PERF_COUNT": "YES"})
    nodes = runtime_nodes(compiled)
    req = compiled.create_infer_request()
    for k, v in inputs.items():
        req.set_tensor(k, ov.Tensor(v))
    for _ in range(args.warmup):
        req.infer()
    acc: dict[str, list[float]] = {}
    meta: dict[str, tuple[str, str, str]] = {}
    wall_pc = []
    for _ in range(args.iters):
        t = time.perf_counter()
        req.infer()
        wall_pc.append((time.perf_counter() - t) * 1e3)
        for p in req.get_profiling_info():
            acc.setdefault(p.node_name, []).append(p.real_time.total_seconds() * 1e3)
            meta[p.node_name] = (str(p.status).replace("Status.", ""), p.node_type, p.exec_type)

    by_name = {n["name"]: n for n in nodes}
    prefix_rows = K.infer_prefix_rows(n["out_shapes"] for n in nodes)
    rows = []
    for name, ts in acc.items():
        st, lt, et = meta[name]
        n = by_name.get(name, {})
        rows.append({"name": name, "status": st, "layer_type": lt, "exec_type": et, "real_ms": float(np.median(ts)),
                     "order": n.get("order", -1), "subgraph": K.subgraph_of(n.get("out_shapes", []), chunk=args.chunk,
                                                                prefix_rows=prefix_rows),
                     "backend": K.backend_of(et), "xmx": K.uses_xmx(et, lt), "family": K.op_family(lt, et),
                     "out_shapes": n.get("out_shapes", []), "orig": n.get("orig", "")})
    kern = sorted((r for r in rows if K.is_kernel(r)), key=lambda r: r["order"])
    kernel_ms = sum(r["real_ms"] for r in kern)
    expert = [r for r in kern if r["subgraph"] == "expert"]
    per_step = [sum(r["real_ms"] for r in grp) for grp in K.split_steps(expert, args.steps)] if expert else []
    res = {
        "xml": str(args.xml), "ov_version": ov.get_version(), "device": args.device, "config": cfg,
        "env": {k: v for k, v in os.environ.items() if k.startswith("OV_")},
        "full_device_name": core.get_property(args.device, "FULL_DEVICE_NAME"),
        "compile_s": round(compile_s, 1), "wall_no_perfcount": K.stats(wall_nopc), "wall_perfcount": K.stats(wall_pc),
        "dispatch": K.dispatch_estimate(float(np.median(wall_pc)), kernel_ms, len(kern),
                                        kernel_times_us=[1e3 * r["real_ms"] for r in kern]),
        "expert_step_ms": per_step,
        "by_family": K.aggregate(kern, lambda r: r["family"]),
        "by_subgraph": K.aggregate(kern, lambda r: r["subgraph"]),
        "by_backend": K.aggregate(kern, lambda r: r["backend"] + (" (XMX)" if r["xmx"] else "")),
        "by_exec_type": K.aggregate(kern, lambda r: f'{r["layer_type"]} | {K.strip_precision(r["exec_type"])}')[:40],
        "by_subgraph_family": K.aggregate(kern, lambda r: f'{r["subgraph"]} | {r["family"]}')[:40],
        "layer_type_counts": K.aggregate(rows, lambda r: r["layer_type"]),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res))
    d = res["dispatch"]
    print(f'{ov.get_version()} compile {compile_s:.1f}s | wall no-PC median {res["wall_no_perfcount"]["median_ms"]:.1f} ms '
          f'| PC {d["wall_ms"]:.1f} ms, kernels {d["kernel_ms"]:.1f} ms in {int(d["n_kernels"])} launches, '
          f'gap {d["gap_ms"]:.1f} ms ({d["gap_us_per_kernel"]:.1f} µs/launch)')
    for key in ("by_subgraph", "by_family", "by_backend"):
        print(key)
        for a in res[key][:12]:
            print(f'  {a["key"]:<40} {a["ms"]:8.2f} ms {100 * a["share"]:5.1f}%  n={a["count"]:5d}  {a["mean_us"]:7.1f} µs')
    print("expert per step ms:", [round(x, 2) for x in per_step])
    gm = [r for r in rows if r["family"] == "GatedMLP" and K.is_kernel(r)]
    print("GatedMLP primitives executed:", len(gm))


if __name__ == "__main__":
    main()
