"""Interleaved latency benchmark of several Studio π0.5 exports through `InferenceModel` (GPU).

Each round loads every variant in turn (one at a time, so only one compiled model is resident), runs
warmup + `--iters` timed `predict_action_chunk` calls and records the runtime-call and OV-only latency
(median / p90 / p99). Interleaving rounds spreads background load from the shared laptop evenly over
the variants. `--compile_test` instead measures cold compile (fresh empty CACHE_DIR) vs cached compile.

    PY=~/inference/physical-ai-studio/library/.venv/bin/python
    nice -n 5 $PY bench_variants.py ~/inference/exports/pi05_junction_{fp16,int8} --rounds 2 \
        --json ~/inference/logs/pi05/opt_bench.jsonl
    nice -n 5 $PY bench_variants.py ~/inference/exports/pi05_junction_int8 --compile_test
    # compile-config A/B on one export: name=KEY=VALUE[,KEY=VALUE]
    nice -n 5 $PY bench_variants.py ~/inference/exports/pi05_junction_int8 \
        --variant default= --variant f16=INFERENCE_PRECISION_HINT=f16 --variant nolat=PERFORMANCE_HINT=THROUGHPUT
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # studio_bench.py lives in deploy/pi05/
import studio_bench  # noqa: E402


def load_avg() -> float:
    return round(os.getloadavg()[0], 1)


def mem_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return round(int(line.split()[1]) / 2**20, 1)
    return float("nan")


def parse_variant(s: str) -> tuple[str, dict[str, str]]:
    name, _, rest = s.partition("=")
    cfg = dict(kv.split("=", 1) for kv in rest.split(",") if kv)
    return name, cfg


def compile_test(export_dir: Path, device: str) -> dict:
    from physicalai.inference import InferenceModel

    tmp = Path(tempfile.mkdtemp(prefix="ovcache_", dir=Path("~/inference/tmp").expanduser()))
    out = {"export": str(export_dir), "device": device}
    try:
        for label, cfg in (("no_cache", {}), ("cold_cache", {"CACHE_DIR": str(tmp)}),
                           ("warm_cache", {"CACHE_DIR": str(tmp)})):
            t = time.perf_counter()
            m = InferenceModel(str(export_dir), backend="openvino", device=device,
                               PERFORMANCE_HINT="LATENCY", NUM_STREAMS="1", **cfg)
            out[f"{label}_compile_s"] = round(time.perf_counter() - t, 1)
            del m  # (compile_test loads 3 times in one process; run it on its own)
        out["cache_bytes"] = sum(f.stat().st_size for f in tmp.rglob("*") if f.is_file())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exports", type=Path, nargs="+")
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--variant", action="append", default=[], help="name=KEY=VALUE,... compile-config variant")
    ap.add_argument("--compile_test", action="store_true")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--min_free_gib", type=float, default=25.0)
    ap.add_argument("--one", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()

    def emit(res: dict) -> None:
        print(json.dumps(res), flush=True)
        if a.json:
            with a.json.expanduser().open("a") as f:
                f.write(json.dumps(res) + "\n")

    if a.compile_test:
        for e in a.exports:
            emit(compile_test(e.expanduser(), a.device))
        return
    variants = [parse_variant(v) for v in a.variant] or [("default", {})]
    cache = str(Path("~/inference/ov_cache").expanduser())
    if a.one:  # child mode: one export, one config, result on stdout
        name, cfg = variants[0]
        res = studio_bench.benchmark(a.exports[0].expanduser(), a.device, a.iters, a.warmup, cache, None, cfg)
        print("RESULT " + json.dumps(res), flush=True)
        return
    # Each measurement runs in its own process. Loading several compiled π0.5 models one after another in
    # one process kept iGPU memory allocated (it does not show in RSS): on 13 Nov a 9-model loop exhausted
    # the 64 GB and the kernel OOM storm rebooted the laptop.
    for r in range(a.rounds):
        for e in a.exports:
            for name, cfg in variants:
                while mem_available_gib() < a.min_free_gib:  # shared laptop: never start a load into a full RAM
                    print(f"waiting for {a.min_free_gib} GiB free (now {mem_available_gib()})", flush=True)
                    time.sleep(30)
                la = load_avg()
                cmd = [sys.executable, __file__, str(e), "--one", "--device", a.device, "--iters", str(a.iters),
                       "--warmup", str(a.warmup), "--variant", f"{name}=" + ",".join(f"{k}={v}" for k, v in cfg.items())]
                out = subprocess.run(cmd, capture_output=True, text=True)
                lines = [ln for ln in out.stdout.splitlines() if ln.startswith("RESULT ")]
                if not lines:
                    emit({"export": str(e), "variant": name, "round": r, "error": out.stderr[-2000:]})
                    continue
                res = json.loads(lines[-1][7:])
                res.update({"round": r, "variant": name, "load_avg_before": la, "load_avg_after": load_avg(),
                            "time": time.strftime("%H:%M:%S"), "mem_available_gib": mem_available_gib()})
                emit(res)


if __name__ == "__main__":
    main()
