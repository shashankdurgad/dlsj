"""Benchmark a Studio π0.5 OpenVINO export through the `physicalai` runtime (`InferenceModel`).

Times the full runtime call (`predict_action_chunk`: normalise + image prep + OV tokenizer +
model + denormalise) and, separately, the OpenVINO model call alone (`adapter.predict`), so
pre/post-processing overhead is visible. Inputs are random but correctly shaped, built from the
export's manifest (3 cameras, uint8 640x480 frames by default, resized to 224² by the runtime).

Run on the XPS in the Studio library env:

    cd ~/inference/physical-ai-studio/library
    uv run python ~/inference/deploy/pi05/studio_bench.py ~/inference/exports/pi05_studio_fp16 --device GPU
    uv run python ~/inference/deploy/pi05/studio_bench.py ~/inference/exports/pi05_studio_int8 --device GPU \
        --iters 30 --json results.jsonl
"""

import argparse
import json
import resource
import time
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

LATENCY_CONFIG = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1"}


def make_observation(model: Any, frame_hw: tuple[int, int], rng: np.random.Generator) -> dict[str, Any]:
    """Random observation matching the manifest's input features (state, cameras, task)."""
    obs: dict[str, Any] = {}
    for feat in model.input_features:
        name, shape = feat.name, tuple(feat.shape or ())
        if name.startswith("images"):
            obs[name] = rng.integers(0, 256, size=(1, 3, *frame_hw), dtype=np.uint8)
        elif name == "state":
            obs[name] = rng.uniform(-1, 1, size=(1, *shape)).astype(np.float32)
        elif name == "task":
            obs[name] = ["place the brick on the blue plate"]
    return obs


def _stats(ms: list[float]) -> dict[str, float]:
    t = np.asarray(ms)
    return {"median_ms": round(float(np.median(t)), 1), "p90_ms": round(float(np.percentile(t, 90)), 1),
            "p99_ms": round(float(np.percentile(t, 99)), 1), "min_ms": round(float(t.min()), 1), "n": len(ms)}


def wrap_adapter_timer(model: Any, sink: list[float]) -> None:
    """Record the time of each raw OpenVINO call made by the runtime."""
    inner = model.adapter.predict

    def timed(inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        t = time.perf_counter()
        out = inner(inputs)
        sink.append((time.perf_counter() - t) * 1e3)
        return out

    model.adapter.predict = timed


def benchmark(export_dir: Path, device: str, iters: int, warmup: int, cache_dir: str | None,
              precision_hint: str | None, extra: dict[str, str] | None = None) -> dict[str, Any]:
    from physicalai.inference import InferenceModel

    config = dict(LATENCY_CONFIG)
    if cache_dir:
        config["CACHE_DIR"] = cache_dir
    if precision_hint:
        config["INFERENCE_PRECISION_HINT"] = precision_hint
    config.update(extra or {})
    t0 = time.perf_counter()
    model = InferenceModel(str(export_dir), backend="openvino", device=device, **config)
    load_s = time.perf_counter() - t0
    rng = np.random.default_rng(0)
    obs = make_observation(model, (480, 640), rng)
    t0 = time.perf_counter()
    chunk = model.predict_action_chunk(dict(obs))
    first_s = time.perf_counter() - t0
    for _ in range(warmup):
        model.predict_action_chunk(dict(obs))
    model_ms: list[float] = []
    wrap_adapter_timer(model, model_ms)
    total_ms = []
    for _ in range(iters):
        t = time.perf_counter()
        model.predict_action_chunk(dict(obs))
        total_ms.append((time.perf_counter() - t) * 1e3)
    return {
        "export": str(export_dir), "device": device, "precision_hint": precision_hint or "device default", "extra_config": extra or {},
        "date": date.today().isoformat(), "load_compile_s": round(load_s, 1), "first_call_s": round(first_s, 2),
        "chunk_shape": list(chunk.shape), "runtime_call": _stats(total_ms), "ov_model_only": _stats(model_ms),
        "peak_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export_dir", type=Path)
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
        res = benchmark(args.export_dir.expanduser(), dev.strip(), args.iters, args.warmup, args.cache_dir,
                        args.precision_hint, extra)
        print(json.dumps(res))
        if args.json:
            with args.json.open("a") as f:
                f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
