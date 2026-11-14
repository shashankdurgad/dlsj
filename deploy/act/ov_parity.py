"""Parity and latency of a `dlsj.deploy.act_export` OpenVINO export against the PyTorch reference.

Feeds the full 640x480 recorded frames (top camera uncropped) to `OpenVINOACT`, which must apply the manifest's
crop, and compares the action chunks with `torch_reference.py`'s output. Also checks that a pre-cropped top frame
gives the same result as the full frame (crop applied exactly once), and times the call on each device.

    nice -n 5 python deploy/act/ov_parity.py ~/inference/exports/act_warp_417_fp16 \
        ~/inference/exports/act_warp_417_ref.npz --device CPU GPU NPU
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dlsj.deploy.act_export import OpenVINOACT  # noqa: E402
from dlsj.deploy.act_studio import diff_summary  # noqa: E402

KEYS = {"observation.images.wrist": "wrist", "observation.images.top": "top", "observation.images.inspect": "inspect"}


def run(export_dir: str, ref_path: str, device: str, precision: str | None, iters: int) -> dict:
    ref = dict(np.load(ref_path))  # materialise: NpzFile decompresses on every access
    t0 = time.perf_counter()
    runner = OpenVINOACT(export_dir, device, cache_dir=str(Path(export_dir) / "ov_cache"),
                         inference_precision=precision)
    compile_s = time.perf_counter() - t0
    n = len(ref["state"])
    frames = [{k: ref[v][i] for k, v in KEYS.items()} for i in range(n)]
    preds, times = [], []
    for i in range(n):
        t = time.perf_counter()
        preds.append(runner(ref["state"][i], frames[i]))
        times.append((time.perf_counter() - t) * 1e3)
    for _ in range(max(0, iters - n)):
        t = time.perf_counter()
        runner(ref["state"][0], frames[0])
        times.append((time.perf_counter() - t) * 1e3)
    pred = np.stack(preds)
    crop = runner.crops["observation.images.top"]
    x, y, w, h = crop["box"]
    pre_cropped = dict(frames[0], **{"observation.images.top": np.ascontiguousarray(frames[0]["observation.images.top"][y:y + h, x:x + w])})
    crop_same = float(np.abs(runner(ref["state"][0], pre_cropped) - preds[0]).max())
    t = np.array(times[3:])
    return {"export": Path(export_dir).name, "device": device, "precision_hint": precision or "device default",
            "compile_s": round(compile_s, 2), "frames": n,
            **{k: round(v, 4) for k, v in diff_summary(pred, ref["ref"]).items()},
            "full_vs_precropped_top_max_abs": crop_same,
            "median_ms": round(float(np.median(t)), 2), "p90_ms": round(float(np.percentile(t, 90)), 2)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export_dir")
    ap.add_argument("ref")
    ap.add_argument("--device", nargs="+", default=["CPU"])
    ap.add_argument("--precision", default=None, help="INFERENCE_PRECISION_HINT, e.g. f32 (default: device's)")
    ap.add_argument("--iters", type=int, default=60)
    args = ap.parse_args()
    for dev in args.device:
        try:
            print(json.dumps(run(args.export_dir, args.ref, dev, args.precision, args.iters)), flush=True)
        except Exception as e:  # report and carry on with the other devices
            print(json.dumps({"export": Path(args.export_dir).name, "device": dev, "error": f"{type(e).__name__}: {e}"}),
                  flush=True)


if __name__ == "__main__":
    main()
