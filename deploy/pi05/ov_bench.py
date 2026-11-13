"""Benchmark a raw π0.5 OpenVINO IR (`model.xml`) with plain `openvino`, no pre/post-processing.

Works for both export paths: Studio (`physicalai export`) and Intel's robotics-ai-suite
`convert_ov_rtc.py`. Inputs are synthesised from the model's own ports (static shapes): images
uniform in [-1, 1], masks all-true, token ids random, RTC inputs (`prev_chunk_left_over`,
`prefix_weights`) zero, which disables guidance exactly as Intel's README recommends for a first chunk.
This is the number to compare across exports: one OpenVINO call = one 50-action chunk.

Needs only `openvino` + `numpy`, so any of the XPS envs works:

    ~/inference/.venv-pi05ov/bin/python ~/inference/deploy/pi05/ov_bench.py \
        ~/inference/exports/pi05_rtc_3c_INT8/model.xml --device GPU --iters 30
    ~/inference/physical-ai-studio/library/.venv/bin/python ~/inference/deploy/pi05/ov_bench.py \
        ~/inference/exports/pi05_studio_fp16/pi05.xml --device GPU,CPU --json results.jsonl
"""

import argparse
import json
import resource
import time
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np


def _input_for(port: Any, rng: np.random.Generator) -> np.ndarray:
    """A plausible random value for one model input, chosen by name and element type."""
    import openvino as ov

    name = port.get_any_name().lower()
    shape = tuple(port.get_partial_shape().to_shape())
    etype = port.get_element_type()
    if etype == ov.Type.boolean or "mask" in name:
        return np.ones(shape, dtype=bool if etype == ov.Type.boolean else etype.to_dtype())
    if etype.is_integral():
        hi = 1000 if "token" in name else 2
        return rng.integers(0, hi, size=shape).astype(etype.to_dtype())
    if "prev_chunk" in name or "prefix_weight" in name:
        return np.zeros(shape, dtype=np.float32)
    if "image" in name or len(shape) == 5:  # Intel's IR leaves the image input unnamed ("292")
        return rng.uniform(-1, 1, size=shape).astype(np.float32)
    return rng.standard_normal(shape).astype(np.float32)


def _stats(ms: list[float]) -> dict[str, float]:
    t = np.asarray(ms)
    return {"median_ms": round(float(np.median(t)), 1), "p90_ms": round(float(np.percentile(t, 90)), 1),
            "p99_ms": round(float(np.percentile(t, 99)), 1), "min_ms": round(float(t.min()), 1), "n": len(ms)}


def benchmark(xml: Path, device: str, iters: int, warmup: int, cache_dir: str | None,
              precision_hint: str | None, extra: dict[str, str] | None = None) -> dict[str, Any]:
    import openvino as ov

    core = ov.Core()
    if cache_dir:
        core.set_property({"CACHE_DIR": cache_dir})
    config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1"}
    if precision_hint:
        config["INFERENCE_PRECISION_HINT"] = precision_hint
    config.update(extra or {})
    t0 = time.perf_counter()
    compiled = core.compile_model(str(xml), device, config)
    compile_s = time.perf_counter() - t0
    request = compiled.create_infer_request()
    rng = np.random.default_rng(0)
    inputs = {p.get_any_name(): _input_for(p, rng) for p in compiled.inputs}
    t0 = time.perf_counter()
    request.infer(inputs)
    first_s = time.perf_counter() - t0
    for _ in range(warmup):
        request.infer(inputs)
    ms = []
    for _ in range(iters):
        t = time.perf_counter()
        request.infer(inputs)
        ms.append((time.perf_counter() - t) * 1e3)
    out = request.get_output_tensor(0).data
    return {
        "model": str(xml), "device": device, "precision_hint": precision_hint or "device default", "extra_config": extra or {},
        "date": date.today().isoformat(), "compile_s": round(compile_s, 1), "first_call_s": round(first_s, 2),
        "inputs": {k: list(v.shape) for k, v in inputs.items()}, "output_shape": list(out.shape),
        "output_finite": bool(np.isfinite(out).all()), **_stats(ms),
        "peak_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("xml", type=Path)
    ap.add_argument("--device", default="GPU", help="comma-separated OpenVINO devices, e.g. GPU,CPU")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--cache_dir", default=str(Path("~/inference/ov_cache").expanduser()))
    ap.add_argument("--precision_hint", help="INFERENCE_PRECISION_HINT, e.g. f16 or f32 (default: device default)")
    ap.add_argument("--config", action="append", default=[], metavar="KEY=VALUE",
                    help="extra compile property, repeatable, e.g. GPU_ENABLE_SDPA_OPTIMIZATION=NO")
    ap.add_argument("--json", type=Path, help="append results as one JSON line to this file")
    args = ap.parse_args()
    extra = dict(kv.split("=", 1) for kv in args.config)
    for dev in args.device.split(","):
        res = benchmark(args.xml.expanduser(), dev.strip(), args.iters, args.warmup, args.cache_dir,
                        args.precision_hint, extra)
        print(json.dumps(res))
        if args.json:
            with args.json.open("a") as f:
                f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
