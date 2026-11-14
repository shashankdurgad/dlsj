"""PyTorch baseline latency (LeRobot 0.7, `preprocessor -> predict_action_chunk -> postprocessor`) on recorded frames.

    nice -n 5 ~/inference/.venv/bin/python deploy/act/torch_bench.py \
        ~/inference/models/act-warp-duplo-417-20261114 ~/inference/exports/act_warp_417_ref.npz --device xpu cpu
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from torch_reference import CAMS, load_policy, reference_chunk  # noqa: E402

from dlsj.deploy.act_studio import Crop, diff_summary  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("policy_path")
    ap.add_argument("ref")
    ap.add_argument("--device", nargs="+", default=["xpu"])
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()
    ref = dict(np.load(args.ref))
    meta = json.loads((Path(args.policy_path) / "checkpoint-metadata.json").read_text())
    crop = Crop.from_metadata(meta["top_camera_crop"])
    n = min(args.iters, len(ref["state"]))
    for dev in args.device:
        policy, pre, post = load_policy(args.policy_path, dev)
        sync = torch.xpu.synchronize if dev == "xpu" else (lambda: None)
        outs, times = [], []
        for i in range(n):
            frames = {k: ref[k.split(".")[-1]][i] for k in CAMS}
            t = time.perf_counter()
            outs.append(reference_chunk(policy, pre, post, ref["state"][i], frames, crop, dev))
            sync()
            times.append((time.perf_counter() - t) * 1e3)
        t = np.array(times[3:])
        print(json.dumps({"device": dev, "torch": torch.__version__, "iters": n,
                          "median_ms": round(float(np.median(t)), 1), "p90_ms": round(float(np.percentile(t, 90)), 1),
                          "vs_cpu_reference": {k: round(v, 4) for k, v in diff_summary(np.stack(outs), ref["ref"][:n]).items()}}),
              flush=True)


if __name__ == "__main__":
    main()
