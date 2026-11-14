"""Pure helpers (stdlib only) for the π0.5 GPU kernel study: parse OpenVINO per-kernel counters, attribute kernels
to π0.5 sub-graphs, aggregate, estimate dispatch overhead, summarise latency sweeps. Tested in
`tests/test_pi05_kernel.py`; used by `profile_kernels.py`, `sweep.py` and `summarize_kernel.py`.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Sequence
from typing import Any

# Not real GPU work: optimised-out / fused-away nodes, host-side waits, graph inputs.
NON_KERNEL_EXEC = ("undef", "unknown", "wait_for_events", "")


def parse_counters_csv(text: str) -> tuple[list[dict[str, Any]], float | None]:
    """Parse benchmark_app's `benchmark_detailed_counters_report.csv` (`;`-separated).

    Returns (rows, total_real_ms). Rows: name, status, layer_type, exec_type, real_ms, cpu_ms. The `Total` line is
    returned separately. Multi-request reports (`Performance counts for N-th infer request`) are concatenated.
    """
    rows: list[dict[str, Any]] = []
    total = None
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(";")]
        if len(parts) < 6 or parts[0] in ("layerName", ""):
            if parts and parts[0] == "Total" and len(parts) >= 5:
                total = _f(parts[4])
            continue
        if parts[0] == "Total":
            total = _f(parts[4])
            continue
        real, cpu = _f(parts[4]), _f(parts[5])
        if real is None:
            continue
        rows.append({"name": parts[0], "status": parts[1].replace("Status.", ""), "layer_type": parts[2],
                     "exec_type": parts[3], "real_ms": real, "cpu_ms": cpu or 0.0})
    return rows, total


def _f(s: str) -> float | None:
    try:
        return float(s)
    except ValueError:
        return None


def strip_precision(exec_type: str) -> str:
    """`jit:gemm:any__i8` -> `jit:gemm:any`; `fully_connected_gpu_bf_tiled__f16` -> `fully_connected_gpu_bf_tiled`."""
    return exec_type.split("__", 1)[0]


def is_kernel(row: dict[str, Any]) -> bool:
    """A real GPU kernel launch (executed, with an implementation)."""
    st = str(row.get("status", "EXECUTED")).replace("Status.", "")
    et = strip_precision(str(row.get("exec_type", "")))
    return st == "EXECUTED" and not any(et == x or (x and et.startswith(x)) for x in NON_KERNEL_EXEC)


def backend_of(exec_type: str) -> str:
    """Which library implements a GPU primitive.

    oneDNN implementation names are `<kind>:<impl>` with a single colon (`jit:gemm:any`, `jit:ir`, `ocl:ref:any`,
    `ocl:micro:...`); these are the ones that can use XMX (`jit:*` and micro-kernels). OpenVINO's own OpenCL kernels
    are either legacy names (`fully_connected_gpu_bf_tiled`, `rms_gpu_bfyx_opt`) or new-style `ocl::<prim>::<impl>`
    (`ocl::sdpa::opt`, `ocl::rope::opt`).
    """
    et = strip_precision(exec_type)
    if et in NON_KERNEL_EXEC or et.startswith("wait_for_events"):
        return "none"
    if et.startswith("ocl::") or et.startswith("cm::") or et.startswith("sycl::"):
        return "openvino-" + et.split("::", 1)[0]
    if re.match(r"^(jit|ocl|ref|gemm|brg|xe)[:_][^:]", et) and ":" in et:
        return "onednn"
    return "openvino-ocl"


def uses_xmx(exec_type: str, layer_type: str = "") -> bool:
    """Heuristic: oneDNN jit/micro GEMM-class kernels, OV micro SDPA and dpas/XMX-tagged OV kernels run on XMX."""
    et = strip_precision(exec_type).lower()
    if backend_of(exec_type) == "onednn":
        return et.startswith("jit:") or "micro" in et
    return "micro" in et or "dpas" in et or "xmx" in et or "systolic" in et


def subgraph_of(shapes: Sequence[Sequence[int]], chunk: int = 50, vision_rows: int = 768,
                prefix_min_rows: int | None = None, image_tokens: int = 256, n_cams: int = 3,
                prefix_rows: int | None = None) -> str:
    """Attribute a primitive to a π0.5 part from its tensor shapes (outputs first, then inputs).

    Studio's IR has anonymous op names, so shapes identify the parts (see `opt_lib.classify_matmul`):
    - expert: any tensor whose token axis (dim -2, or dim 1 of a `[1,T,H,D]` SDPA output) is the action chunk
      (`[1,50,1024]`, scores `[1,8,50,1018]`, SDPA `[1,50,8,256]`), or, if `prefix_rows` is known, the expert's
      KV concat of the prefix cache with its own 50 tokens (`[1,1,prefix_rows+50,256]`);
    - vision: SigLIP runs the cameras as a batch (`[3,256,1152]`, SDPA `[3,256,16,72]`), 2-D `[768,1152]`, images;
    - vlm: the Gemma-2B prefix, token axis >= 3x256 image tokens + 16 (`[1,968,2048]`, `[1,8,968,968]`);
    - other: everything else (prompt embedding, state, small glue).
    The first shape that matches decides; expert is checked first on every shape, then vision, then vlm.
    """
    prefix_min = prefix_min_rows if prefix_min_rows is not None else n_cams * image_tokens + 16
    good = [list(s) for s in shapes if s is not None and len(s) > 0 and all(isinstance(d, int) and d >= 0 for d in s)]
    for s in good:
        if len(s) >= 2 and s[-2] == chunk:
            return "expert"
        if len(s) == 4 and s[0] == 1 and s[1] == chunk:
            return "expert"
        if prefix_rows is not None and len(s) >= 2 and s[-2] == prefix_rows + chunk:
            return "expert"
    for s in good:
        if len(s) >= 3 and s[0] == n_cams and (image_tokens in s[1:] or s[-1] == 224 or s[-1] == 16):
            return "vision"
        if len(s) == 5 and s[0] == n_cams and s[-1] == 224:
            return "vision"
        if len(s) == 2 and s[0] == vision_rows:
            return "vision"
    for s in good:
        if len(s) >= 2 and s[-2] >= prefix_min:
            return "vlm"
        if len(s) == 4 and s[0] == 1 and s[1] >= prefix_min:
            return "vlm"
    return "other"


def infer_prefix_rows(all_shapes: Iterable[Sequence[Sequence[int]]], n_cams: int = 3,
                      image_tokens: int = 256) -> int | None:
    """Most common token count of `[1, T, H]` tensors with T >= cams x image tokens + 16 (the Gemma prefix length)."""
    pm = n_cams * image_tokens + 16
    cnt: dict[int, int] = {}
    for shapes in all_shapes:
        for s in shapes or []:
            if len(s) == 3 and s[0] == 1 and isinstance(s[1], int) and s[1] >= pm:
                cnt[s[1]] = cnt.get(s[1], 0) + 1
    return max(cnt, key=cnt.get) if cnt else None


def op_family(layer_type: str, exec_type: str) -> str:
    """Coarse op family for the per-op-type table."""
    lt = layer_type.lower()
    et = strip_precision(exec_type).lower()
    if "gatedmlp" in lt or "gated_mlp" in et:
        return "GatedMLP"
    if lt in ("fullyconnected", "fullyconnectedcompressed", "matmul", "gemm") or "gemm" in et or "fully_connected" in et:
        return "MatMul/FC"
    if "scaleddotproduct" in lt or "sdpa" in et:
        return "SDPA"
    if "dynamicquantize" in lt or "dynamic_quantize" in et:
        return "DynamicQuantize"
    if lt in ("rms", "rmsnorm") or "rms" in et:
        return "RMSNorm"
    if lt in ("mvn",) or "mvn" in et:
        return "LayerNorm(MVN)"
    if "rope" in lt or "rope" in et:
        return "RoPE"
    if lt in ("softmax",) or "softmax" in et:
        return "Softmax"
    if lt in ("reorder", "convert") or "reorder" in et:
        return "Reorder/Convert"
    if lt in ("eltwise", "add", "multiply", "subtract", "divide", "power") or "eltwise" in et:
        return "Eltwise"
    if lt in ("activation", "gelu", "swish", "glu", "swiglu") or "activation" in et or "glu" in et:
        return "Activation/GLU"
    if lt in ("concatenation", "concat") or "concat" in et:
        return "Concat"
    if lt in ("convolution",) or "conv" in et:
        return "Convolution"
    if lt in ("crop", "slice", "stridedslice", "strided_slice", "gather", "permute", "transpose", "reshape") \
            or any(k in et for k in ("crop", "slice", "gather", "permute")):
        return "DataMovement"
    return layer_type or "other"


def aggregate(rows: Iterable[dict[str, Any]], key: Callable[[dict[str, Any]], str],
              time_key: str = "real_ms") -> list[dict[str, Any]]:
    """Group kernel rows by `key(row)`: total time, count, share of the total, mean µs per kernel. Sorted by time."""
    tot: dict[str, float] = {}
    cnt: dict[str, int] = {}
    for r in rows:
        k = key(r)
        tot[k] = tot.get(k, 0.0) + float(r.get(time_key) or 0.0)
        cnt[k] = cnt.get(k, 0) + 1
    grand = sum(tot.values()) or 1.0
    out = [{"key": k, "ms": tot[k], "count": cnt[k], "share": tot[k] / grand,
            "mean_us": 1e3 * tot[k] / cnt[k]} for k in tot]
    return sorted(out, key=lambda d: -d["ms"])


def split_steps(items: Sequence[Any], steps: int) -> list[list[Any]]:
    """Split a topologically ordered list of expert primitives into `steps` contiguous, near-equal groups."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    n = len(items)
    bounds = [round(i * n / steps) for i in range(steps + 1)]
    return [list(items[bounds[i]:bounds[i + 1]]) for i in range(steps)]


def dispatch_estimate(wall_ms: float, kernel_ms: float, n_kernels: int, small_us: float = 20.0,
                      kernel_times_us: Sequence[float] | None = None) -> dict[str, float]:
    """How much of one inference is not covered by GPU kernel time.

    gap_ms = wall - sum(kernel GPU time): host enqueue, queue/launch gaps, sync. Divided by the kernel count it gives
    a per-launch overhead figure comparable to Intel's "~104 µs per unfused kernel". Also counts tiny kernels.
    """
    gap = max(wall_ms - kernel_ms, 0.0)
    out = {"wall_ms": wall_ms, "kernel_ms": kernel_ms, "gap_ms": gap, "gap_share": gap / wall_ms if wall_ms else 0.0,
           "n_kernels": float(n_kernels), "gap_us_per_kernel": 1e3 * gap / n_kernels if n_kernels else 0.0}
    if kernel_times_us is not None:
        small = [t for t in kernel_times_us if t < small_us]
        out["n_small"] = float(len(small))
        out["small_ms"] = sum(small) / 1e3
    return out


def stats(ms: Sequence[float]) -> dict[str, float]:
    """median / p90 / p99 / min / mean of a latency list (linear-interpolated percentiles, like numpy)."""
    xs = sorted(float(x) for x in ms)
    if not xs:
        raise ValueError("empty latency list")
    return {"median_ms": _pct(xs, 50), "p90_ms": _pct(xs, 90), "p99_ms": _pct(xs, 99), "min_ms": xs[0],
            "mean_ms": sum(xs) / len(xs), "n": float(len(xs))}


def _pct(xs: Sequence[float], q: float) -> float:
    pos = (len(xs) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def pooled_compare(results: Iterable[dict[str, Any]], baseline: str = "default",
                   metric: str = "runtime_call") -> list[dict[str, Any]]:
    """Summarise an interleaved sweep: per variant, the median of per-round medians and the paired delta vs baseline.

    `results` are `sweep.py` rows: {"variant", "round", metric: {"median_ms", "p90_ms"}}. The paired delta compares
    each variant with the baseline from the same round, which cancels slow drift (thermal, other jobs).
    """
    by: dict[str, dict[int, dict[str, Any]]] = {}
    for r in results:
        if metric not in r:
            continue
        by.setdefault(r["variant"], {})[int(r["round"])] = r[metric]
    base = by.get(baseline, {})
    out = []
    for v, rounds in by.items():
        med = [x["median_ms"] for x in rounds.values()]
        p90 = [x["p90_ms"] for x in rounds.values()]
        deltas = [rounds[k]["median_ms"] - base[k]["median_ms"] for k in rounds if k in base]
        out.append({"variant": v, "rounds": len(rounds), "median_ms": _median(med), "p90_ms": _median(p90),
                    "delta_ms": _median(deltas) if deltas else None,
                    "delta_pct": (100 * _median(deltas) / _median([base[k]["median_ms"] for k in rounds if k in base]))
                    if deltas else None})
    order = {baseline: -1}
    return sorted(out, key=lambda d: (order.get(d["variant"], 0), d["median_ms"]))


def _median(xs: Sequence[float]) -> float:
    return _pct(sorted(xs), 50)


def md_table(rows: Sequence[dict[str, Any]], cols: Sequence[tuple[str, str]], fmt: dict[str, str] | None = None) -> str:
    """Markdown table from dict rows. `cols` = [(key, header)], `fmt` = {key: format spec}."""
    fmt = fmt or {}
    head = "| " + " | ".join(h for _, h in cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    body = []
    for r in rows:
        cells = []
        for k, _ in cols:
            v = r.get(k)
            cells.append("" if v is None else (format(v, fmt[k]) if k in fmt and isinstance(v, (int, float)) else str(v)))
        body.append("| " + " | ".join(cells) + " |")
    return head + "\n".join(body) + "\n"
