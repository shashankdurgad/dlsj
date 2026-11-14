"""Baseline vs optimised benchmark for MolmoAct2 on the XPS: PyTorch (Studio, unoptimised) vs OpenVINO variants.

Every row runs in its own subprocess (one model in memory), under the shared heavy lock with the 6 GB RAM watchdog,
and only while no controller drives the arm. All rows see the SAME frames (harness cache: 60 recorded frames over
grasp / present / place, wrist + top) and are compared against the same FP16 reference predictions.

Row specs (--rows, comma separated):
  torch:xpu:bf16 | torch:xpu:fp32 | torch:cpu:bf16 | torch:cpu:fp32
      stock physicalai.policies.MolmoAct2 (Studio defaults: text 256, 10 flow steps, plain chunk prediction),
      built exactly like export.py (pretrained + unit-scale SO-101 frame) or from --ckpt (LoRA checkpoint)
  ov:<export dir>[@GPU|@CPU|@NPU]   an OpenVINO export (RTC exports get a guided call: delay 5, horizon 15, w 3)

Per row: cold/warm load time, latency per chunk median/p90/p99 over --iters warm calls (load average recorded
before each timed block; blocks with load > --max-load are redone), peak RSS, GPU-resident memory (xe drm fdinfo),
and quality on the 60 frames vs the reference and vs the recorded demos (harness.compare). Power: package RAPL is
read when permitted (root-only on the XPS, so usually reported as unavailable).

    python baseline_bench.py --rows torch:xpu:bf16,ov:~/inference/exports/molmoact2_so101_rtc_fp16,... \
        --ref ~/inference/exports/molmo_opt/zeroshot/pred_fp16_ref.npz --out ~/inference/logs/molmo/baseline
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness as H  # noqa: E402

CHUNK, HZ, HORIZON = 30, 30.0, 15


# ----------------------------------------------------------------------------- measurement helpers
def gpu_resident_mb() -> float | None:
    """xe driver per-client memory from /proc/self/fdinfo (drm-resident-* / drm-total-*), MB."""
    tot, seen = 0, False
    for fd in Path("/proc/self/fdinfo").iterdir():
        try:
            txt = fd.read_text()
        except OSError:
            continue
        if "drm-driver" not in txt:
            continue
        for line in txt.splitlines():
            if line.startswith(("drm-resident-", "drm-total-")) and line.startswith("drm-resident-"):
                k, v = line.split(":", 1)
                parts = v.split()
                val = float(parts[0]) * {"KiB": 1 / 1024, "MiB": 1, "GiB": 1024}.get(parts[1] if len(parts) > 1 else "", 1 / 1048576)
                tot += val
                seen = True
    return round(tot, 1) if seen else None


def rapl_uj() -> int | None:
    try:
        return int(Path("/sys/class/powercap/intel-rapl:0/energy_uj").read_text())
    except OSError:
        return None


def stats(ms: list[float]) -> dict:
    a = np.asarray(ms)
    return {"median_ms": round(float(np.median(a)), 1), "p90_ms": round(float(np.percentile(a, 90)), 1),
            "p99_ms": round(float(np.percentile(a, 99)), 1), "n": len(a)}


# ----------------------------------------------------------------------------- torch row
def torch_worker(spec: str, frames: list[dict], args) -> dict:
    import torch

    from export import build

    _, device, dtype_s = spec.split(":")
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype_s]
    torch.set_grad_enabled(False)
    t = time.perf_counter()
    policy = build(args.pretrained, False, ["wrist", "top"], 480, 640, ckpt=args.ckpt)
    policy = policy.to(device=device, dtype=dtype).eval()
    load_s = time.perf_counter() - t
    from physicalai.data.observation import Observation

    def call(f):
        obs = Observation(
            state=torch.from_numpy(f["state"][None]).to(device),
            images={c: (torch.from_numpy(np.asarray(img)[None]).to(device).float() / 255.0).to(dtype)
                    for c, img in f["imgs"].items()},
            task=[f["task"]])
        a = policy.predict_action_chunk(obs)
        if device == "xpu":
            torch.xpu.synchronize()
        a = a.float().cpu().numpy()
        return a[0] if a.ndim == 3 else a

    call(frames[0])  # warm-up (first call compiles kernels)
    call(frames[1 % len(frames)])
    preds = [call(f) for f in frames]  # quality pass on all frames
    lat = timed(lambda k: call(frames[k % len(frames)]), args)
    extra = {}
    if device == "xpu":
        extra["xpu_peak_alloc_mb"] = round(torch.xpu.max_memory_allocated() / 1048576, 1)
    return {"load_s": round(load_s, 1), **lat, "preds": np.stack(preds), **extra}


# ----------------------------------------------------------------------------- OpenVINO row
def ov_worker(spec: str, frames: list[dict], args) -> dict:
    from physicalai.inference import InferenceModel

    path, _, device = spec[3:].partition("@")
    device = device or "GPU"
    path = str(Path(path).expanduser())
    cfg = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": str(Path(args.cache_dir).expanduser())}
    t = time.perf_counter()
    m = InferenceModel(path, backend="openvino", device=device, **cfg)
    load_s = time.perf_counter() - t
    names = set(m.adapter.input_names or [])
    rtc = "prev_chunk_left_over" in names
    prev = np.zeros((1, CHUNK, 6), np.float32)

    def call(f, guided=False):
        obs = {"state": f["state"][None], "task": [f["task"]]}
        for c, img in f["imgs"].items():
            obs[f"images.{c}"] = np.ascontiguousarray(np.asarray(img)[None])
        if not rtc:
            return np.asarray(m.predict_action_chunk(obs), np.float32)
        x = dict(obs)
        for p in m.preprocessors:
            x = p(x)
        x.update({"prev_chunk_left_over": prev if guided else np.zeros_like(prev),
                  "inference_delay": np.int64(5 if guided else 0), "execution_horizon": np.int64(HORIZON if guided else 1),
                  "max_guidance_weight": np.float32(3.0 if guided else 1e-8)})
        out = m.runner.run(m.adapter, m._prepare_inputs(x))
        for p in m.postprocessors:
            out = p(out)
        a = np.asarray(out["action"], np.float32)
        return a[0] if a.ndim == 3 else a

    call(frames[0])
    preds = [call(f) for f in frames]
    if rtc:  # time the guided call (what the controller runs every chunk after the first)
        prev[0, : CHUNK - 10] = preds[0][10:]
    lat = timed(lambda k: call(frames[k % len(frames)], guided=rtc), args)
    return {"load_s": round(load_s, 1), **lat, "preds": np.stack(preds), "rtc": rtc}


def timed(fn, args) -> dict:
    """>= args.iters warm calls in blocks of 10; a block that starts with load1 > max_load is waited out and redone."""
    from optimize import wait_arm_free

    ms, loads, redone = [], [], 0
    k = 0
    e0, t0 = rapl_uj(), time.perf_counter()
    while len(ms) < args.iters:
        wait_arm_free(lambda m: print(m, flush=True))
        load1 = os.getloadavg()[0]
        if load1 > args.max_load and redone < 20:
            redone += 1
            print(f"load {load1:.1f} > {args.max_load}: waiting 30 s", flush=True)
            time.sleep(30)
            continue
        loads.append(round(load1, 2))
        for _ in range(10):
            t = time.perf_counter()
            fn(k)
            ms.append((time.perf_counter() - t) * 1e3)
            k += 1
    e1, t1 = rapl_uj(), time.perf_counter()
    out = {**stats(ms), "load1_per_block": loads, "blocks_redone": redone}
    if e0 is not None and e1 is not None and e1 > e0:
        w = (e1 - e0) / 1e6 / (t1 - t0)
        out |= {"pkg_power_w": round(w, 1), "pkg_j_per_chunk": round(w * out["median_ms"] / 1e3, 2)}
    else:
        out["pkg_power_w"] = None  # RAPL is root-only on the XPS
    return out


def worker(args) -> None:
    frames = list(np.load(Path(args.frames_cache).expanduser(), allow_pickle=True)["frames"])
    res = torch_worker(args.worker, frames, args) if args.worker.startswith("torch") else ov_worker(args.worker, frames, args)
    res["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    res["gpu_resident_mb"] = gpu_resident_mb()
    np.save(args.worker_out + ".npy", res.pop("preds"))
    Path(args.worker_out + ".json").write_text(json.dumps(res, indent=1))


# ----------------------------------------------------------------------------- orchestration
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", default="torch:xpu:bf16")
    ap.add_argument("--pretrained", default="allenai/MolmoAct2-SO100_101")
    ap.add_argument("--ckpt", help="Studio Lightning checkpoint (LoRA fine-tune) instead of --pretrained")
    ap.add_argument("--frames-cache", default="~/inference/exports/molmo_opt/zeroshot/frames_zeroshot_60.npz")
    ap.add_argument("--ref", default="~/inference/exports/molmo_opt/zeroshot/pred_fp16_ref.npz",
                    help="reference predictions (.npz with 'pred', or .npy) on the same frames")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--max-load", type=float, default=3.0)
    ap.add_argument("--cache-dir", default="~/inference/ov_cache")
    ap.add_argument("--out", default="~/inference/logs/molmo/baseline")
    ap.add_argument("--label", action="append", default=[], help="row=label (display names)")
    ap.add_argument("--worker")
    ap.add_argument("--worker-out")
    args = ap.parse_args()
    if args.worker:
        return worker(args)

    from optimize import run_heavy

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    labels = dict(x.split("=", 1) for x in args.label)
    frames = list(np.load(Path(args.frames_cache).expanduser(), allow_pickle=True)["frames"])
    refp = Path(args.ref).expanduser()
    ref = np.load(refp)["pred"] if refp.suffix == ".npz" else np.load(refp)
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)  # noqa: E731
    results = json.loads((out / "results.json").read_text()) if (out / "results.json").exists() else {}
    for row in [r.strip() for r in args.rows.split(",") if r.strip()]:
        tag = row.replace("/", "_").replace("~", "").replace(":", "-").replace("@", "-")
        wo = str(out / f"row_{tag}")
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", row, "--worker-out", wo,
               "--frames-cache", args.frames_cache, "--iters", str(args.iters), "--max-load", str(args.max_load),
               "--cache-dir", args.cache_dir, "--pretrained", args.pretrained] + (["--ckpt", args.ckpt] if args.ckpt else [])
        try:
            run_heavy(cmd, log)
            r = json.loads(Path(wo + ".json").read_text())
            pred = np.load(wo + ".npy")
            q = H.compare(pred, ref, frames)
            r["quality"] = {"vs_ref_joints": q["all"]["vs_ref_joints"], "vs_ref_gripper": q["all"]["vs_ref_gripper"],
                            "demo_err_joints": q["all"]["demo_err_joints"], "demo_err_ref_joints": q["all"]["demo_err_ref_joints"]}
        except Exception as e:  # noqa: BLE001
            r = {"error": f"{type(e).__name__}: {e}"[:400]}
        r["label"] = labels.get(row, row)
        results[row] = r
        (out / "results.json").write_text(json.dumps(results, indent=1))
        log(f"{row}: {json.dumps({k: v for k, v in r.items() if k not in ('load1_per_block',)})[:400]}")
    write_table(results, out / "table.md")


def write_table(results: dict, path: Path) -> None:
    base = next((r for k, r in results.items() if k.startswith("torch") and "median_ms" in r), None)
    lines = ["| row | load s | median ms | p90 | p99 | speed-up vs baseline | actions/s (30-step chunk) | peak RSS MB | GPU MB | "
             "vs FP16 ref joints / gripper (deg) | demo err (ref) | power W |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in results.items():
        if "median_ms" not in r:
            lines.append(f"| {r.get('label', k)} | error: {r.get('error', '?')[:120]} |||||||||||")
            continue
        sp = f"{base['median_ms'] / r['median_ms']:.2f}x" if base else "n/a"
        q = r.get("quality", {})
        lines.append(f"| {r['label']} | {r['load_s']} | {r['median_ms']} | {r['p90_ms']} | {r['p99_ms']} | {sp} | "
                     f"{CHUNK * 1000 / r['median_ms']:.0f} | {r.get('peak_rss_mb')} | {r.get('gpu_resident_mb') or r.get('xpu_peak_alloc_mb')} | "
                     f"{q.get('vs_ref_joints')} / {q.get('vs_ref_gripper')} | {q.get('demo_err_joints')} ({q.get('demo_err_ref_joints')}) | "
                     f"{r.get('pkg_power_w') or 'n/a'} |")
    rtc = [r for r in results.values() if r.get("rtc") and "median_ms" in r]
    if rtc:
        r = rtc[0]
        lines.append("")
        lines.append(f"RTC control-loop proxy: chunk {CHUNK} at {HZ:.0f} Hz, horizon {HORIZON}: a new chunk is requested "
                     f"every {HORIZON} ticks ({HORIZON / HZ * 1000:.0f} ms) and arrives after {r['median_ms']} ms "
                     f"(p99 {r['p99_ms']}), i.e. {min(1.0, (HORIZON / HZ * 1000) / max(r['p99_ms'], 1)):.0%} of requests "
                     f"land before the queue runs dry at p99; ticks are never idle while latency < {CHUNK / HZ * 1000:.0f} ms.")
    path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
