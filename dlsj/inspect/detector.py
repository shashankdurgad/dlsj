"""`AnomalibInspector`: the Anomalib model exported by `train.py`, run with OpenVINO on the NPU (GPU/CPU fallback).

    from dlsj.inspect.detector import AnomalibInspector
    insp = AnomalibInspector.from_manifest("~/inference/exports/inspect/padim/manifest.json", device="NPU")
    result = insp.inspect([f0, f1, f2])        # inspect-camera frames, HxWx3 uint8 RGB

Evaluate on a folder of frames (`good/`, `bad/`, files `<brick>_f<k>.png` grouped per brick):

    python -m dlsj.inspect.detector --manifest .../manifest.json --data ~/inference/inspect_data/synth_test --device NPU

The exported IR (see `train.py`) takes a static uint8 NHWC RGB batch: the u8→f32 conversion, /255 and layout
change are folded into the graph with `ov.preprocess.PrePostProcessor`, the Anomalib pre-processor (resize +
ImageNet normalisation) is in the graph too. On the CPU we only crop the ROI and resize it (cv2, uint8).
Outputs: `pred_score` (B,) raw anomaly score and `anomaly_map` (B,1,H,W) at the model input resolution.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from dlsj.contracts import InspectionResult, Label
from dlsj.inspect import core
from dlsj.inspect.core import Manifest

log = logging.getLogger("inspect.detector")

DEFAULT_FALLBACK = ("NPU", "GPU", "CPU")


class OpenVINOBackend:
    """Compiled OpenVINO model: (B,H,W,3) uint8 RGB → (scores (B,), maps (B,h,w) float32)."""

    def __init__(self, xml: str | Path, input_hw: tuple[int, int], batch: int, device: str,
                 fallback: tuple[str, ...] = DEFAULT_FALLBACK, cache_dir: str | Path | None = None):
        import openvino as ov

        core_ = ov.Core()
        if cache_dir:
            core_.set_property({"CACHE_DIR": str(cache_dir)})
        model = core_.read_model(str(xml))
        h, w = input_hw
        try:
            model.reshape({model.input(0).get_any_name(): [batch, h, w, 3]})  # static: NPU rejects dynamic shapes
        except Exception as e:  # noqa: BLE001  some graphs (e.g. PatchCore) only work at their exported batch
            model = core_.read_model(str(xml))
            exported = model.input(0).get_partial_shape()
            if not exported.is_static:
                raise
            log.warning("cannot reshape %s to batch %d (%s); using its exported shape %s", Path(xml).name, batch,
                        str(e).strip().splitlines()[0][:120], exported)
            batch = exported[0].get_length()
        names = [o.get_any_name() for o in model.outputs]
        self.score_idx = next(i for i, n in enumerate(names) if "score" in n)
        self.map_idx = next(i for i, n in enumerate(names) if "map" in n)
        order = [device] + [d for d in fallback if d != device]
        available = core_.available_devices
        errors = {}
        for dev in order:
            if not any(a.startswith(dev) for a in available):
                errors[dev] = "not available"
                continue
            try:
                cfg = {"PERFORMANCE_HINT": "LATENCY"}
                self.compiled = core_.compile_model(model, dev, cfg)
                self.device = dev
                break
            except Exception as e:  # noqa: BLE001
                errors[dev] = str(e).strip().splitlines()[-1][:300]
                log.warning("compile on %s failed: %s", dev, errors[dev])
        else:
            raise RuntimeError(f"could not compile {xml} on any of {order}: {errors}")
        if self.device != device:
            log.warning("requested %s, running on %s (%s)", device, self.device, errors.get(device))
        self.batch = batch
        self.request = self.compiled.create_infer_request()

    def __call__(self, batch_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.request.infer({0: batch_u8})
        score = self.request.get_output_tensor(self.score_idx).data.reshape(len(batch_u8))
        amap = self.request.get_output_tensor(self.map_idx).data
        return score.astype(np.float32, copy=True), amap.reshape(len(batch_u8), *amap.shape[-2:]).astype(np.float32, copy=True)


class AnomalibInspector:
    """Implements `dlsj.contracts.Inspector`.

    `backend` maps a (B,H,W,3) uint8 batch to (scores, maps); tests inject a fake one.
    `batch` is the static batch the backend was compiled for: frames are run in chunks of that size
    (one call for 3 frames when batch=3; a short last chunk is padded by repeating its last frame).
    """

    def __init__(self, manifest: Manifest, backend, batch: int = 1, device: str = "", heatmap: bool = True):
        self.manifest = manifest
        self.backend = backend
        self.batch = batch
        self.device = device
        self.heatmap = heatmap

    @classmethod
    def from_manifest(cls, path: str | Path, device: str = "NPU", batch: int = 3,
                      fallback: tuple[str, ...] = DEFAULT_FALLBACK, heatmap: bool = True,
                      cache_dir: str | Path | None = None) -> "AnomalibInspector":
        path = Path(path).expanduser()
        m = Manifest.load(path)
        backend = OpenVINOBackend(path.parent / m.model, m.input_size, batch, device, fallback, cache_dir)
        return cls(m, backend, batch=backend.batch, device=backend.device, heatmap=heatmap)

    def score_crops(self, crops: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(N,H,W,3) uint8 model-size crops → scores (N,), maps (N,h,w)."""
        n = len(crops)
        scores, maps = [], []
        for i in range(0, n, self.batch):
            chunk = crops[i:i + self.batch]
            k = len(chunk)
            if k < self.batch:
                chunk = np.concatenate([chunk, np.repeat(chunk[-1:], self.batch - k, axis=0)])
            s, m = self.backend(np.ascontiguousarray(chunk))
            scores.append(s[:k])
            maps.append(m[:k])
        return np.concatenate(scores), np.concatenate(maps)

    def inspect(self, frames: list[np.ndarray]) -> InspectionResult:
        if not frames:
            raise ValueError("inspect() needs at least one frame")
        t0 = time.perf_counter()
        m = self.manifest
        crops = np.stack([core.prepare(f, m.crop, m.input_size) for f in frames])
        scores, maps = self.score_crops(crops)
        result = core.build_result(scores, maps, m, frames[0].shape[:2], device=self.device, with_heatmap=self.heatmap)
        latency = (time.perf_counter() - t0) * 1e3
        return InspectionResult(**{**result.__dict__, "latency_ms": latency})


# ----------------------------------------------------------------------------- evaluation CLI
def evaluate(insp: AnomalibInspector, data: Path, overlays: Path | None = None) -> dict:
    """Per-brick inspection over `data/good` and `data/bad`; prints a confusion table."""
    rows = []
    for kind in ("good", "bad"):
        for brick, paths in sorted(core.group_by_brick(core.list_images(data / kind)).items()):
            frames = [core.read_rgb(p) for p in paths]
            r = insp.inspect(frames)
            rows.append((kind, brick, r))
            if overlays is not None:
                overlays.mkdir(parents=True, exist_ok=True)
                core.write_rgb(overlays / f"{kind}_{brick}_{r.label.value}.jpg", core.overlay(frames[0], r))
    expect = {"good": Label.PERFECT, "bad": Label.DEFECTIVE}
    table = {k: {lab: 0 for lab in Label} for k in expect}
    for kind, _, r in rows:
        table[kind][r.label] += 1
    n = len(rows)
    correct = sum(r.label is expect[k] for k, _, r in rows)
    wrong = sum(r.label not in (expect[k], Label.UNCERTAIN) for k, _, r in rows)
    unc = sum(r.label is Label.UNCERTAIN for _, _, r in rows)
    # Accuracy if UNCERTAIN falls back to DEFECTIVE (the loop's safe default after re-inspection fails).
    safe = sum((Label.DEFECTIVE if r.label is Label.UNCERTAIN else r.label) is expect[k] for k, _, r in rows)
    lat = np.array([r.latency_ms for _, _, r in rows])
    print(f"device={insp.device} bricks={n} threshold={insp.manifest.threshold:.4g} "
          f"band=[{insp.manifest.band_low:.4g}, {insp.manifest.band_high:.4g}]")
    print(f"{'true':6s} {'perfect':>8s} {'defective':>10s} {'uncertain':>10s}")
    for k in expect:
        print(f"{k:6s} {table[k][Label.PERFECT]:8d} {table[k][Label.DEFECTIVE]:10d} {table[k][Label.UNCERTAIN]:10d}")
    print(f"correct {correct}/{n} ({100 * correct / max(n, 1):.1f}%), wrong {wrong}, uncertain {unc}; "
          f"uncertain→defective accuracy {100 * safe / max(n, 1):.1f}%")
    print(f"inspect() latency over {n} calls ({len(rows and rows[0][2].frame_scores)} frames each): "
          f"median {np.median(lat):.2f} ms, p90 {np.percentile(lat, 90):.2f} ms")
    gs = [r.score for k, _, r in rows if k == "good"]
    bs = [r.score for k, _, r in rows if k == "bad"]
    if gs and bs:
        print(f"scores: good max {max(gs):.4g} (median {np.median(gs):.4g}), bad min {min(bs):.4g} "
              f"(median {np.median(bs):.4g})")
    return {"table": table, "correct": correct, "n": n, "uncertain": unc, "wrong": wrong, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True, help="folder with good/ and bad/ inspect frames")
    ap.add_argument("--device", default="NPU", choices=["NPU", "GPU", "CPU"])
    ap.add_argument("--batch", type=int, default=3, help="static batch compiled into the model (1 = sequential)")
    ap.add_argument("--overlays", type=Path, help="write heatmap/region overlays here")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    insp = AnomalibInspector.from_manifest(args.manifest, device=args.device, batch=args.batch)
    evaluate(insp, args.data.expanduser(), args.overlays)


if __name__ == "__main__":
    main()
