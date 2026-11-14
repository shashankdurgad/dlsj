"""Interleaved GPU-plugin config sweep for a Studio π0.5 export through `physicalai.InferenceModel`.

Each (round, variant) is a fresh `studio_bench.py` process (compile properties and OV_* env variables only take
effect at compile time, and a fresh process avoids state leaking between variants). Variants run in a shuffled order
per round, and `summarize_kernel.py sweep` reports the median of per-round medians plus the paired delta vs the
`default` variant of the same round, which cancels slow drift.

    PY=~/inference/physical-ai-studio/library/.venv/bin/python
    $PY sweep.py ~/inference/exports/pi05_junction_int8 --rounds 3 --iters 40 --out sweep.jsonl
    $PY sweep.py ... --only default,dq0,dq32        # subset
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# One CACHE_DIR per variant: OV_* env options are not part of OpenVINO's cache key, so a shared cache would silently
# load a blob compiled without them.
CACHE_ROOT = Path(os.environ.get("SWEEP_CACHE_ROOT", Path("~/inference/ovcache_kernel").expanduser()))
STUDIO_BENCH = Path(os.environ.get("STUDIO_BENCH", HERE.parent / "studio_bench.py"))

# name -> {"config": compile properties added on top of studio_bench's LATENCY/NUM_STREAMS=1/CACHE_DIR,
#          "env": process env, "prefix": command prefix}
VARIANTS: dict[str, dict] = {
    "default": {},
    "dq0": {"config": {"DYNAMIC_QUANTIZATION_GROUP_SIZE": "0"}},
    "dq32": {"config": {"DYNAMIC_QUANTIZATION_GROUP_SIZE": "32"}},
    "dq64": {"config": {"DYNAMIC_QUANTIZATION_GROUP_SIZE": "64"}},
    "dq128": {"config": {"DYNAMIC_QUANTIZATION_GROUP_SIZE": "128"}},
    # RELEASE_INTERNAL option (env only): dynamic INT8 activation quantisation also for FCs with <=64 rows,
    # i.e. the 50-token action expert
    "dq_thr16": {"env": {"OV_GPU_DYNAMIC_QUANTIZATION_THRESHOLD": "16"}},
    "f16": {"config": {"INFERENCE_PRECISION_HINT": "f16"}},
    "sdpa_off": {"config": {"GPU_ENABLE_SDPA_OPTIMIZATION": "NO"}},
    "queue_prio_high": {"config": {"GPU_QUEUE_PRIORITY": "HIGH"}},
    "host_prio_high": {"config": {"GPU_HOST_TASK_PRIORITY": "HIGH"}},
    "throttle_high": {"config": {"GPU_QUEUE_THROTTLE": "HIGH"}},
    "throttle_low": {"config": {"GPU_QUEUE_THROTTLE": "LOW"}},
    "all_high": {"config": {"GPU_QUEUE_PRIORITY": "HIGH", "GPU_HOST_TASK_PRIORITY": "HIGH",
                            "GPU_QUEUE_THROTTLE": "HIGH"}},
    "tput_1stream": {"config": {"PERFORMANCE_HINT": "THROUGHPUT", "NUM_STREAMS": "1"}},
    "pcores": {"prefix": ["taskset", "-c", "0-3"]},
    # debug-caps build (run with --python ~/inference/.venv-ovdbg/bin/python after `source ~/inference/ovbuild/env.sh`)
    "gm_stock": {"env": {"OV_GPU_DISABLE_GATED_MLP_FUSION": "0"}},
    "gm_gelu": {"env": {"OV_GPU_DISABLE_GATED_MLP_FUSION": "0", "OV_GPU_GATED_MLP_GELU": "1"}},
    # the release wheel, same session (Studio env python)
    # (LD_LIBRARY_PATH from env.sh is dropped so the wheel loads its own libopenvino, not the debug build's)
    "release": {"python": str(Path("~/inference/physical-ai-studio/library/.venv/bin/python").expanduser()),
                "unset": ["LD_LIBRARY_PATH"]},
}


def run_one(py: str, export: Path, name: str, spec: dict, iters: int, warmup: int, rnd: int, out: Path,
            extra_cfg: list[str]) -> dict | None:
    cmd = list(spec.get("prefix", [])) + [spec.get("python", py), str(STUDIO_BENCH), str(export), "--device", "GPU", "--iters", str(iters),
                                          "--warmup", str(warmup), "--cache_dir", str(CACHE_ROOT / name)]
    for k, v in {**spec.get("config", {}), **dict(kv.split("=", 1) for kv in extra_cfg)}.items():
        cmd += ["--config", f"{k}={v}"]
    env = {k: v for k, v in {**os.environ, **spec.get("env", {})}.items() if k not in spec.get("unset", [])}
    t = time.time()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    line = next((ln for ln in p.stdout.splitlines() if ln.startswith("{")), None)
    if p.returncode != 0 or line is None:
        print(f"[{name} r{rnd}] FAILED rc={p.returncode}\n{p.stderr[-2000:]}", flush=True)
        res = {"variant": name, "round": rnd, "error": p.stderr[-2000:]}
    else:
        res = json.loads(line)
        res.update({"variant": name, "round": rnd, "env": spec.get("env", {}), "prefix": spec.get("prefix", []),
                    "python": spec.get("python", py),
                    "wall_s": round(time.time() - t, 1), "loadavg": os.getloadavg()[0]})
        print(f'[{name} r{rnd}] runtime {res["runtime_call"]["median_ms"]} / p90 {res["runtime_call"]["p90_ms"]} ms, '
              f'ov {res["ov_model_only"]["median_ms"]} ms, load {res["loadavg"]:.1f}', flush=True)
    with out.open("a") as f:
        f.write(json.dumps(res) + "\n")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export", type=Path)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--only", default="", help="comma-separated variant names (default: all)")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--config", action="append", default=[], help="extra KEY=VALUE applied to every variant")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    dbg_only = ("gm_stock", "gm_gelu", "release")
    names = [n for n in (args.only.split(",") if args.only else [v for v in VARIANTS if v not in dbg_only]) if n]
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants {unknown}; known: {list(VARIANTS)}")
    rng = random.Random(0)
    for r in range(args.rounds):
        order = [n for n in names if n != "default"]
        rng.shuffle(order)
        for n in ["default"] + order:  # baseline first in every round, rest shuffled
            run_one(args.python, args.export.expanduser(), n, VARIANTS[n], args.iters, args.warmup, r, args.out,
                    args.config)


if __name__ == "__main__":
    main()
