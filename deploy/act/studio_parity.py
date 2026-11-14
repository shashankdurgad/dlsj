"""Run a Studio ACT export through Studio's own runtime and compare with the PyTorch reference.

Uses `physicalai.inference.InferenceModel`, the class Studio's model worker loads for "Run model", and builds the
observation exactly like `workers/model_worker.py` does: `state` (1,S) float32 plus, for each `images.<camera>`
input, that camera's latest full frame with a batch axis (1,480,640,3) uint8. So the top camera gets the full
640x480 frame and the export must crop it itself.

    cd ~/inference/physical-ai-studio/library
    nice -n 5 .venv/bin/python ~/inference/dlsj-act/deploy/act/studio_parity.py \
        <studio model dir>/exports/openvino ~/inference/exports/act_warp_417_ref.npz --device CPU GPU
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dlsj.deploy.act_studio import diff_summary  # noqa: E402


def run(export_dir: str, ref_path: str, device: str, iters: int) -> dict:
    from physicalai.inference import InferenceModel

    ref = dict(np.load(ref_path))  # materialise: NpzFile decompresses on every access
    manifest = json.loads((Path(export_dir) / "manifest.json").read_text())
    cams = [f["init_args"]["name"] for f in manifest["model"]["input_features"]
            if f["init_args"]["name"].startswith("images.")]
    t0 = time.perf_counter()
    model = InferenceModel(export_dir, backend="openvino", device=device)
    load_s = time.perf_counter() - t0

    def obs(i: int) -> dict:
        o = {"state": ref["state"][i:i + 1].astype(np.float32)}
        for name in cams:
            o[name] = ref[name.split(".", 1)[1]][i][np.newaxis]  # full 640x480 frame, uint8 HWC
        return o

    observations = [obs(i) for i in range(len(ref["state"]))]
    outs, times = [], []
    for o in observations:
        model.reset()
        t = time.perf_counter()
        a = np.asarray(model.predict_action_chunk(o))
        times.append((time.perf_counter() - t) * 1e3)
        outs.append(a.reshape(a.shape[-2], a.shape[-1]))
    for _ in range(max(0, iters - len(times))):  # extra timing iterations on frame 0
        model.reset()
        t = time.perf_counter()
        model.predict_action_chunk(observations[0])
        times.append((time.perf_counter() - t) * 1e3)
    pred = np.stack(outs)
    t = np.array(times[3:])
    return {"device": device, "load_s": round(load_s, 1), "chunk_shape": list(outs[0].shape),
            "inputs": cams, "frames": len(outs), **{k: round(v, 4) for k, v in diff_summary(pred, ref["ref"]).items()},
            "median_ms": round(float(np.median(t)), 2), "p90_ms": round(float(np.percentile(t, 90)), 2)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export_dir")
    ap.add_argument("ref")
    ap.add_argument("--device", nargs="+", default=["CPU"])
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()
    for dev in args.device:
        print(json.dumps(run(args.export_dir, args.ref, dev, args.iters)), flush=True)


if __name__ == "__main__":
    main()
