"""Run a π0.5 OpenVINO policy package (from `dlsj.deploy.pi05_export`) on the laptop.

Thin wrapper over `physicalai.inference.InferenceModel`, which does the whole pipeline from the manifest:
quantile-normalise the state, resize-with-pad each camera to 224², tokenise the prompt (with the state
discretised into it), draw fresh noise, run the graph, de-normalise. Needs only `physicalai` (runtime commit
8381d43), `openvino` and `openvino-tokenizers`; no torch.

    policy = Pi05Policy("exports/pi05_ft_int8", device="GPU")
    chunk = policy(state, {"wrist": rgb, "top": rgb, "inspect": rgb}, "place the brick in the orange bowl")
    # chunk: (50, 6) joint targets in the robot's calibrated units, 30 fps

Benchmark on the XPS (synthetic inputs, 640x480 frames like the cameras):

    python -m dlsj.deploy.pi05_runtime ~/desmond/exports/pi05_ft_int8 --device GPU,CPU
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

PICK = "pick up the brick and hold it in front of the inspection camera"
PLACE = "place the brick in the {colour} bowl"
LATENCY = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1"}


class Pi05Policy:
    def __init__(self, export_dir: str | Path, device: str = "GPU", cache_dir: str | None = "ov_cache", **config):
        from physicalai.inference import InferenceModel

        cfg = dict(LATENCY, **config)
        if cache_dir:
            cfg["CACHE_DIR"] = str(Path(cache_dir).expanduser())
        self.model = InferenceModel(str(export_dir), backend="openvino", device=device, **cfg)
        self.cameras = [f.name.removeprefix("images.") for f in self.model.input_features if f.name.startswith("images.")]
        self.device = device

    def __call__(self, state: np.ndarray, images: dict[str, np.ndarray], task: str) -> np.ndarray:
        """state (6,) float; images {camera: (H, W, 3) uint8 RGB}; task text -> (chunk, 6) actions."""
        obs = {"state": np.asarray(state, np.float32)[None], "task": [task]}
        for cam in self.cameras:
            obs[f"images.{cam}"] = np.ascontiguousarray(np.asarray(images[cam], np.uint8).transpose(2, 0, 1)[None])
        return self.model.predict_action_chunk(obs)


def benchmark(export_dir: str, device: str, iters: int = 20, warmup: int = 3) -> dict:
    t0 = time.perf_counter()
    policy = Pi05Policy(export_dir, device)
    compile_s = time.perf_counter() - t0
    rng = np.random.default_rng(0)
    imgs = {c: rng.integers(0, 256, (480, 640, 3), dtype=np.uint8) for c in policy.cameras}
    state = rng.uniform(-50, 50, 6).astype(np.float32)
    for _ in range(warmup):
        chunk = policy(state, imgs, PLACE.format(colour="orange"))
    ms = []
    for _ in range(iters):
        t = time.perf_counter()
        policy(state, imgs, PLACE.format(colour="orange"))
        ms.append((time.perf_counter() - t) * 1e3)
    med = float(np.median(ms))
    return {"export": str(export_dir), "device": device, "compile_s": round(compile_s, 1),
            "median_ms": round(med, 1), "p90_ms": round(float(np.percentile(ms, 90)), 1),
            "chunk": list(chunk.shape), "chunks_per_s": round(1e3 / med, 2),
            "actions_per_s": round(chunk.shape[0] * 1e3 / med, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export_dir")
    ap.add_argument("--device", default="GPU", help="comma-separated: GPU, CPU, NPU")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    for dev in args.device.split(","):
        try:
            print(json.dumps(benchmark(args.export_dir, dev.strip(), args.iters)), flush=True)
        except Exception as e:  # noqa: BLE001  e.g. a device that can't compile the graph
            print(json.dumps({"export": args.export_dir, "device": dev, "error": f"{type(e).__name__}: {e}"[:400]}),
                  flush=True)


if __name__ == "__main__":
    main()
