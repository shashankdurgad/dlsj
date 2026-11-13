"""`AnomalibInspector`: the Anomalib model exported by `train.py`, run with OpenVINO on the NPU (GPU/CPU fallback).

    from dlsj.inspect.detector import AnomalibInspector
    insp = AnomalibInspector.from_manifest("~/inference/exports/inspect/padim/manifest.json", device="NPU")
    result = insp.inspect([f0, f1, f2])        # inspect-camera frames, HxWx3 uint8 RGB

Stage 2 (only when stage 1 says marked): the mark classifier (`classify.py`) runs on the stage-1 region crop of every
frame and sets `mark` (CIRCLE / CROSS / UNKNOWN) and `mark_confidence`. The back-end comes from the `stage2`
argument ("shape" | "none" | a CNN classifier.json) or the manifest's `stage2` entry; default: shape rules.
`stage_latency_ms` holds {"stage1", "stage2"}; `latency_ms` is the total.

Evaluate on a folder of frames (`good/` plain, `bad/` cross, `minor/` circle; `<brick>_f<k>.png` grouped per brick):

    python -m dlsj.inspect.detector --manifest .../manifest.json --data ~/inference/inspect_data/synth3_test --device NPU
    python -m dlsj.inspect.detector ... --stage2 ~/inference/exports/inspect/mark_cnn/classifier.json --stage2-device NPU

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

from dlsj.contracts import InspectionResult, Label, MarkType, mark_outcome
from dlsj.inspect import classify, core
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
        try:
            self.request.infer({0: batch_u8})
        except RuntimeError as e:  # seen once on the NPU: "zeFenceHostSynchronize ... ZE_RESULT_ERROR_UNKNOWN"
            log.warning("%s inference failed (%s); retrying with a new infer request", self.device,
                        str(e).strip().splitlines()[-1][:200])
            self.request = self.compiled.create_infer_request()
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

    def __init__(self, manifest: Manifest, backend, batch: int = 1, device: str = "", heatmap: bool = True,
                 classifier: "classify.MarkClassifier | None" = None):
        self.manifest = manifest
        self.backend = backend
        self.batch = batch
        self.device = device
        self.heatmap = heatmap
        self.classifier = classifier  # stage 2; None = two-plate behaviour (mark stays None)

    @classmethod
    def from_manifest(cls, path: str | Path, device: str = "NPU", batch: int = 3,
                      fallback: tuple[str, ...] = DEFAULT_FALLBACK, heatmap: bool = True,
                      cache_dir: str | Path | None = None, stage2: str | dict | None = None,
                      stage2_device: str | None = None) -> "AnomalibInspector":
        """`stage2`: "shape" | "none" | path to a CNN classifier.json; None = the manifest's `stage2` (default shape)."""
        path = Path(path).expanduser()
        m = Manifest.load(path)
        backend = OpenVINOBackend(path.parent / m.model, m.input_size, batch, device, fallback, cache_dir)
        spec = stage2 if stage2 is not None else (m.stage2 or "shape")
        clf = classify.load_classifier(spec, device=stage2_device or device, base_dir=path.parent, cache_dir=cache_dir)
        return cls(m, backend, batch=backend.batch, device=backend.device, heatmap=heatmap, classifier=clf)

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
        t1 = time.perf_counter()
        extra = {"stage_latency_ms": {"stage1": (t1 - t0) * 1e3}}
        if self.classifier is not None and result.label is Label.DEFECTIVE:
            extra.update(self.classify_mark(frames, result))
            extra["stage_latency_ms"]["stage2"] = (time.perf_counter() - t1) * 1e3
        latency = (time.perf_counter() - t0) * 1e3
        return InspectionResult(**{**result.__dict__, "latency_ms": latency, **extra})

    def classify_mark(self, frames: list[np.ndarray], result: InspectionResult) -> dict:
        """Stage 2 on the region crop of every frame -> InspectionResult fields."""
        clf = self.classifier
        if result.region is None:
            return {"mark": MarkType.UNKNOWN, "mark_confidence": 0.0, "mark_device": clf.device}
        box = classify.crop_box(result.region, frames[0].shape[:2], clf.pad_frac)
        x0, y0, x1, y1 = box
        crops = [f[y0:y1, x0:x1] for f in frames]
        heats = None
        if result.heatmap is not None:
            heats = [result.heatmap[y0:y1, x0:x1]] * len(crops)
        pred = clf.classify(crops, heats)
        return {"mark": pred.mark, "mark_confidence": float(pred.confidence), "mark_device": clf.device}


# ----------------------------------------------------------------------------- evaluation CLI
FOLDERS = {"good": MarkType.NONE, "minor": MarkType.CIRCLE, "bad": MarkType.CROSS}  # data folder -> truth


def evaluate(insp: AnomalibInspector, data: Path, overlays: Path | None = None, min_mark_confidence: float = 0.0
             ) -> dict:
    """Per-brick inspection over `data/good` (plain), `data/minor` (circle) and `data/bad` (cross).

    Prints the stage-1 table (plain vs marked) and the final 3-class confusion matrix (`contracts.mark_outcome`,
    UNKNOWN = some stage uncertain -> the loop re-inspects, then falls back to the defective plate).
    """
    rows = []
    for kind in FOLDERS:
        for brick, paths in sorted(core.group_by_brick(core.list_images(data / kind)).items()):
            frames = [core.read_rgb(p) for p in paths]
            r = insp.inspect(frames)
            rows.append((kind, brick, r))
            if overlays is not None:
                overlays.mkdir(parents=True, exist_ok=True)
                tag = r.label.value if r.mark is None else f"{r.label.value}_{r.mark.value}"
                core.write_rgb(overlays / f"{kind}_{brick}_{tag}.jpg", core.overlay(frames[0], r))
    kinds = [k for k in FOLDERS if any(k == kk for kk, _, _ in rows)]
    n = len(rows)
    # stage 1
    expect1 = {"good": Label.PERFECT, "minor": Label.DEFECTIVE, "bad": Label.DEFECTIVE}
    t1 = {k: {lab: 0 for lab in (Label.PERFECT, Label.DEFECTIVE, Label.UNCERTAIN)} for k in kinds}
    for kind, _, r in rows:
        t1[kind][r.label] += 1
    clf = insp.classifier
    print(f"device={insp.device} stage2={clf.name if clf else 'none'}@{clf.device if clf else '-'} bricks={n} "
          f"threshold={insp.manifest.threshold:.4g} band=[{insp.manifest.band_low:.4g}, {insp.manifest.band_high:.4g}]")
    print("stage 1 (Anomalib): plain vs marked")
    print(f"{'true':6s} {'perfect':>8s} {'marked':>8s} {'uncertain':>10s}")
    for k in kinds:
        print(f"{k:6s} {t1[k][Label.PERFECT]:8d} {t1[k][Label.DEFECTIVE]:8d} {t1[k][Label.UNCERTAIN]:10d}")
    c1 = sum(r.label is expect1[k] for k, _, r in rows)
    print(f"stage 1 correct {c1}/{n} ({100 * c1 / max(n, 1):.1f}%)")
    # final 3-class
    order = (MarkType.NONE, MarkType.CIRCLE, MarkType.CROSS, MarkType.UNKNOWN)
    table = {k: {o: 0 for o in order} for k in kinds}
    for kind, _, r in rows:
        table[kind][mark_outcome(r, min_mark_confidence)] += 1
    print("final 3-class (stage 1 + stage 2): rows = truth, unknown = re-inspect, then defective plate")
    print(f"{'true':14s} {'plain':>6s} {'circle':>7s} {'cross':>6s} {'unknown':>8s}")
    for k in kinds:
        print(f"{k + '/' + FOLDERS[k].value:14s} {table[k][MarkType.NONE]:6d} {table[k][MarkType.CIRCLE]:7d} "
              f"{table[k][MarkType.CROSS]:6d} {table[k][MarkType.UNKNOWN]:8d}")
    correct = sum(mark_outcome(r, min_mark_confidence) is FOLDERS[k] for k, _, r in rows)
    unc = sum(mark_outcome(r, min_mark_confidence) is MarkType.UNKNOWN for _, _, r in rows)
    wrong = n - correct - unc

    def safe(o):  # the loop's fallback: unknown -> defective plate (cross)
        return MarkType.CROSS if o is MarkType.UNKNOWN else o

    safe_ok = sum(safe(mark_outcome(r, min_mark_confidence)) is FOLDERS[k] for k, _, r in rows)
    print(f"correct {correct}/{n} ({100 * correct / max(n, 1):.1f}%), wrong {wrong}, unknown {unc}; "
          f"unknown->defective accuracy {100 * safe_ok / max(n, 1):.1f}%")
    lat = np.array([r.latency_ms for _, _, r in rows])
    s1 = np.array([r.stage_latency_ms.get("stage1", np.nan) for _, _, r in rows])
    s2 = np.array([r.stage_latency_ms["stage2"] for _, _, r in rows if "stage2" in r.stage_latency_ms])
    nf = len(rows and rows[0][2].frame_scores)
    print(f"latency over {n} inspect() calls ({nf} frames each): total median {np.median(lat):.2f} ms "
          f"p90 {np.percentile(lat, 90):.2f}; stage1 median {np.nanmedian(s1):.2f} p90 {np.nanpercentile(s1, 90):.2f}"
          + (f"; stage2 (n={len(s2)}) median {np.median(s2):.2f} p90 {np.percentile(s2, 90):.2f} ms" if len(s2) else ""))
    gs = [r.score for k, _, r in rows if k == "good"]
    bs = [r.score for k, _, r in rows if k != "good"]
    if gs and bs:
        print(f"scores: good max {max(gs):.4g} (median {np.median(gs):.4g}), marked min {min(bs):.4g} "
              f"(median {np.median(bs):.4g})")
    return {"table": table, "stage1": t1, "correct": correct, "n": n, "uncertain": unc, "wrong": wrong, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True, help="folder with good/ and bad/ inspect frames")
    ap.add_argument("--device", default="NPU", choices=["NPU", "GPU", "CPU"])
    ap.add_argument("--batch", type=int, default=3, help="static batch compiled into the model (1 = sequential)")
    ap.add_argument("--overlays", type=Path, help="write heatmap/region overlays here")
    ap.add_argument("--stage2", default=None, help='"shape" | "none" | CNN classifier.json (default: manifest, else shape)')
    ap.add_argument("--stage2-device", default=None, help="OpenVINO device of the stage-2 CNN (default: --device)")
    ap.add_argument("--min-mark-confidence", type=float, default=0.0, help="reasoner's extra stage-2 threshold")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    insp = AnomalibInspector.from_manifest(args.manifest, device=args.device, batch=args.batch, stage2=args.stage2,
                                           stage2_device=args.stage2_device)
    evaluate(insp, args.data.expanduser(), args.overlays, args.min_mark_confidence)


if __name__ == "__main__":
    main()
