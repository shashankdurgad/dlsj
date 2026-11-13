"""Latency of the exported `inspect` detector on NPU vs GPU vs CPU (PLAYBOOK §8.1 OPTIMIZE table).

    python -m dlsj.inspect.bench --manifest ~/inference/exports/inspect/padim_synth/manifest.json \
        --data ~/inference/inspect_data/synth_test

For each device and batch mode (3 frames in one static-batch call vs 3 sequential batch-1 calls) it reports
median / p90 / p99 over `--iters` inspections after `--warmup`, end to end (crop + resize + inference + median
+ band + region + heatmap, i.e. `AnomalibInspector.inspect`), plus the inference-only share and compile time.
Frames are real files from `--data` (cycled), 3 per inspection.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import time
from pathlib import Path

import numpy as np

from dlsj.inspect import core
from dlsj.inspect.detector import AnomalibInspector

log = logging.getLogger("inspect.bench")


def load_frame_sets(data: Path, n_frames: int, limit: int = 20) -> list[list[np.ndarray]]:
    sets = []
    for kind in ("good", "bad"):
        for paths in list(core.group_by_brick(core.list_images(data / kind)).values())[: limit // 2]:
            frames = [core.read_rgb(p) for p in paths[:n_frames]]
            while len(frames) < n_frames:
                frames.append(frames[-1])
            sets.append(frames)
    if not sets:
        raise SystemExit(f"no frames in {data}/good or {data}/bad")
    return sets


def pct(a, q) -> float:
    return float(np.percentile(np.asarray(a), q))


def bench_one(manifest: Path, device: str, batch: int, sets, warmup: int, iters: int) -> dict | None:
    t0 = time.perf_counter()
    try:
        insp = AnomalibInspector.from_manifest(manifest, device=device, batch=batch, fallback=(device,))
    except Exception as e:  # noqa: BLE001
        return {"device": device, "batch": batch, "error": str(e)[-300:]}
    compile_s = time.perf_counter() - t0
    m = insp.manifest
    crops = [np.stack([core.prepare(f, m.crop, m.input_size) for f in s]) for s in sets]
    for i in range(warmup):
        insp.inspect(sets[i % len(sets)])
    e2e, infer = [], []
    for i in range(iters):
        s = sets[i % len(sets)]
        t = time.perf_counter()
        insp.inspect(s)
        e2e.append((time.perf_counter() - t) * 1e3)
        c = crops[i % len(crops)]
        t = time.perf_counter()
        insp.score_crops(c)
        infer.append((time.perf_counter() - t) * 1e3)
    return {"device": insp.device, "batch": insp.batch, "compile_s": compile_s,
            "e2e_med": pct(e2e, 50), "e2e_p90": pct(e2e, 90), "e2e_p99": pct(e2e, 99),
            "inf_med": pct(infer, 50), "inf_p90": pct(infer, 90)}


def gpu_busy_note() -> str:
    try:
        out = subprocess.run(["ps", "-eo", "pcpu,args", "--sort=-pcpu"], capture_output=True, text=True, timeout=5)
        busy = [ln.strip()[:110] for ln in out.stdout.splitlines()[1:6] if "python" in ln and "dlsj.inspect" not in ln]
        return "; ".join(busy) or "none"
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True, help="folder with good/ and bad/ inspect frames")
    ap.add_argument("--devices", default="NPU,GPU,CPU")
    ap.add_argument("--batches", default="3,1", help="static batch sizes to compare (3 = one call for 3 frames)")
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    sets = load_frame_sets(args.data.expanduser(), args.frames)
    print(f"other busy python processes at start: {gpu_busy_note()}")
    rows = []
    for dev in args.devices.split(","):
        for b in (int(x) for x in args.batches.split(",")):
            r = bench_one(args.manifest.expanduser(), dev, b, sets, args.warmup, args.iters)
            rows.append(r)
            if "error" in r:
                print(f"{dev} batch={b}: FAILED {r['error']}")
            else:
                print(f"{r['device']} batch={r['batch']}: e2e median {r['e2e_med']:.2f} ms", flush=True)
    m = core.Manifest.load(args.manifest.expanduser())
    print(f"\n{m.model_name}/{m.backbone} {m.input_size[0]}x{m.input_size[1]}, {args.frames} frames per inspection, "
          f"{args.iters} iters after {args.warmup} warmup")
    print(f"{'device':6s} {'batch':>5s} {'e2e med':>8s} {'e2e p90':>8s} {'e2e p99':>8s} {'infer med':>9s} "
          f"{'infer p90':>9s} {'compile':>8s}")
    for r in rows:
        if "error" in r:
            print(f"{r['device']:6s} {r['batch']:5d}  failed: {r['error'][:80]}")
            continue
        print(f"{r['device']:6s} {r['batch']:5d} {r['e2e_med']:8.2f} {r['e2e_p90']:8.2f} {r['e2e_p99']:8.2f} "
              f"{r['inf_med']:9.2f} {r['inf_p90']:9.2f} {r['compile_s']:7.1f}s")
    print("(ms; e2e = AnomalibInspector.inspect on full 640x480 frames; infer = model calls only)")
    print(f"other busy python processes at end: {gpu_busy_note()}")


if __name__ == "__main__":
    main()
