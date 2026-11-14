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
from dataclasses import replace
from pathlib import Path

import numpy as np

from dlsj.contracts import FrameInspection, InspectionResult, Label, MarkType, mark_outcome
from dlsj.inspect import aggregate, classify, core
from dlsj.inspect.core import Manifest

log = logging.getLogger("inspect.detector")

DEFAULT_FALLBACK = ("NPU", "GPU", "CPU")


class OpenVINOBackend:
    """Compiled OpenVINO model: (B,H,W,3) uint8 RGB → (scores (B,), maps (B,h,w) float32).

    Takes our own IR (u8 NHWC, pre-processing in the graph) or a standard Anomalib export (float NCHW [0, 1]; the
    conversion is folded in at load time, and `score_space="raw"` cuts it before its normaliser; see `model_io`).
    `last_labels` holds the export's own `pred_label` output of the last call (None when the model has none).
    """

    def __init__(self, xml: str | Path, input_hw: tuple[int, int], batch: int, device: str,
                 fallback: tuple[str, ...] = DEFAULT_FALLBACK, cache_dir: str | Path | None = None,
                 score_space: str = "raw", input_format: str | None = None):
        import openvino as ov

        from dlsj.inspect import model_io

        core_ = ov.Core()
        if cache_dir:
            core_.set_property({"CACHE_DIR": str(cache_dir)})
        model, info = model_io.prepare_model(core_, xml, input_hw, batch, score_space, input_format)
        batch = info["batch"]
        self.info = info
        self.score_idx, self.map_idx = info["outputs"]["score"], info["outputs"]["map"]
        self.label_idx = info["outputs"]["label"]
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
        self.last_labels: np.ndarray | None = None

    def __call__(self, batch_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        try:
            self.request.infer({0: batch_u8})
        except RuntimeError as e:  # seen once on the NPU: "zeFenceHostSynchronize ... ZE_RESULT_ERROR_UNKNOWN"
            log.warning("%s inference failed (%s); retrying with a new infer request", self.device,
                        str(e).strip().splitlines()[-1][:200])
            self.request = self.compiled.create_infer_request()
            self.request.infer({0: batch_u8})
        n = len(batch_u8)
        score = self.request.get_output_tensor(self.score_idx).data.reshape(n)
        amap = self.request.get_output_tensor(self.map_idx).data
        if self.label_idx is not None:
            self.last_labels = self.request.get_output_tensor(self.label_idx).data.reshape(n).astype(np.int32)
        return score.astype(np.float32, copy=True), amap.reshape(n, *amap.shape[-2:]).astype(np.float32, copy=True)


class AnomalibInspector(aggregate.SweepInspectorMixin):
    """Implements `dlsj.contracts.Inspector` (3-frame `inspect`) and the sweep API (`inspect_sweep`, `start_sweep`,
    `inspect_frame`) of `aggregate.SweepInspectorMixin`.

    `backend` maps a (B,H,W,3) uint8 batch to (scores, maps); tests inject a fake one.
    `batch` is the static batch the backend was compiled for: frames are run in chunks of that size
    (one call for 3 frames when batch=3; a short last chunk is padded by repeating its last frame).
    `agg` holds the sweep aggregation strategy and its knobs (`aggregate.AggregateConfig`).
    """

    def __init__(self, manifest: Manifest, backend, batch: int = 1, device: str = "", heatmap: bool = True,
                 classifier: "classify.MarkClassifier | None" = None, agg: aggregate.AggregateConfig | None = None,
                 base_dir: Path | None = None):
        self.manifest = manifest
        self.backend = backend
        self.batch = batch
        self.device = device
        self.heatmap = heatmap
        self.classifier = classifier  # stage 2; None = two-plate behaviour (mark stays None)
        self.agg = agg or aggregate.AggregateConfig()
        self.base_dir = base_dir or Path(".")

    @classmethod
    def from_manifest(cls, path: str | Path, device: str = "NPU", batch: int = 3,
                      fallback: tuple[str, ...] = DEFAULT_FALLBACK, heatmap: bool = True,
                      cache_dir: str | Path | None = None, stage2: str | dict | None = None,
                      stage2_device: str | None = None, **kw) -> "AnomalibInspector":
        """`stage2`: "shape" | "none" | path to a CNN classifier.json; None = the manifest's `stage2` (default shape).

        `path` may also be a standard Anomalib export folder or .xml (see `from_model`)."""
        return cls.from_model(path, device=device, batch=batch, fallback=fallback, heatmap=heatmap,
                              cache_dir=cache_dir, stage2=stage2, stage2_device=stage2_device, **kw)

    @classmethod
    def from_model(cls, path: str | Path, device: str = "NPU", batch: int = 1,
                   fallback: tuple[str, ...] = DEFAULT_FALLBACK, heatmap: bool = True,
                   cache_dir: str | Path | None = None, stage2: str | dict | None = None,
                   stage2_device: str | None = None, crop=None, input_size=None, score_space: str | None = None,
                   band: float | None = None, agg: aggregate.AggregateConfig | None = None) -> "AnomalibInspector":
        """Any stage-1 model: our manifest.json, an Anomalib export folder, or a .xml (`model_io.resolve`).

        For an export without a manifest the threshold comes from its sidecars (Studio `evaluation.json`), stage 2
        defaults to the shape rules, and `crop` (inspect-frame ROI) should be given: exports know nothing about it.
        """
        from dlsj.inspect import model_io

        m, base = model_io.resolve(path, crop, input_size, score_space,
                                   band if band is not None else model_io.DEFAULT_BAND)
        xml = Path(m.model) if Path(m.model).is_absolute() else base / m.model
        backend = OpenVINOBackend(xml, m.input_size, batch, device, fallback, cache_dir, m.score_space, m.input_format)
        spec = stage2 if stage2 is not None else (m.stage2 or "shape")
        clf = classify.load_classifier(spec, device=stage2_device or device, base_dir=base, cache_dir=cache_dir)
        return cls(m, backend, batch=backend.batch, device=backend.device, heatmap=heatmap, classifier=clf, agg=agg,
                   base_dir=base)

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

    def score_frames(self, frames: list[np.ndarray]) -> np.ndarray:
        """Stage-1 scores only (calibration)."""
        m = self.manifest
        return self.score_crops(np.stack([core.prepare(f, m.crop, m.input_size) for f in frames]))[0]

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

    # ------------------------------------------------------------------ sweep (per-frame) API
    def sweep_meta(self) -> dict:
        m = self.manifest
        return {"threshold": m.threshold, "band_low": m.band_low, "band_high": m.band_high, "device": self.device,
                "mark_device": self.classifier.device if self.classifier is not None else ""}

    def _roi(self, frame_hw):
        roi = self.manifest.crop
        if roi is None:
            return None, tuple(frame_hw)
        fh, fw = frame_hw
        roi = (max(0, roi[0]), max(0, roi[1]), min(fw, roi[2]), min(fh, roi[3]))
        return roi, (roi[3] - roi[1], roi[2] - roi[0])

    def _frame_stage1(self, index: int, angle, frame_hw, score: float, amap: np.ndarray, ms: float,
                      pred_label: int | None = None):
        m = self.manifest
        label = aggregate.frame_label(float(score), m.band_low, m.band_high)
        roi, crop_hw = self._roi(frame_hw)
        region = None
        if label is not Label.PERFECT:
            pthr = m.pixel_threshold if m.pixel_threshold is not None else m.threshold
            region = core.region_from_map(amap, pthr, roi, crop_hw)
        heat = core.to_heatmap(amap, m.map_min, m.threshold, frame_hw, roi) if self.heatmap else None
        return FrameInspection(index, None if angle is None else float(angle), float(score), label, region,
                               latency_ms=ms, pred_label=pred_label), heat

    def _stage2_crops(self, frames, recs, heats):
        """Stage 2 on every marked frame's own region crop (one classifier call). Returns updated records."""
        clf = self.classifier
        todo = [i for i, r in enumerate(recs) if r.label is Label.DEFECTIVE and clf is not None]
        if not todo:
            return recs
        t0 = time.perf_counter()
        crops, hcrops, ok = [], [], []
        for i in todo:
            if recs[i].region is None:
                continue
            x0, y0, x1, y1 = classify.crop_box(recs[i].region, frames[i].shape[:2], clf.pad_frac)
            crops.append(frames[i][y0:y1, x0:x1])
            hcrops.append(heats[i][y0:y1, x0:x1] if heats[i] is not None else None)
            ok.append(i)
        ps = clf.p_cross(crops, hcrops) if crops else []
        per = (time.perf_counter() - t0) * 1e3 / max(len(todo), 1)
        out = list(recs)
        pmap = dict(zip(ok, ps))
        for i in todo:
            p = pmap.get(i)
            if p is None or (isinstance(p, float) and np.isnan(p)):
                mark, conf, p = MarkType.UNKNOWN, 0.0, None
            else:
                mark = MarkType.CROSS if p >= 0.5 else MarkType.CIRCLE
                conf = max(p, 1 - p)
                if conf < clf.min_confidence:
                    mark = MarkType.UNKNOWN
            out[i] = replace(out[i], mark=mark, mark_confidence=float(conf), p_cross=None if p is None else float(p),
                             latency_ms=out[i].latency_ms + per)
        return out

    def inspect_frame(self, frame: np.ndarray, index: int = 0, angle: float | None = None):
        """One sweep frame: stage 1 (batch 1 call), its region, stage 2 if marked. -> (FrameInspection, heatmap)."""
        t0 = time.perf_counter()
        m = self.manifest
        scores, maps = self.score_crops(core.prepare(frame, m.crop, m.input_size)[None])
        lab = getattr(self.backend, "last_labels", None)
        ms = (time.perf_counter() - t0) * 1e3
        rec, heat = self._frame_stage1(index, angle, frame.shape[:2], scores[0], maps[0], ms,
                                       None if lab is None else int(lab[0]))
        rec = self._stage2_crops([frame], [rec], [heat])[0]
        return rec, heat

    def inspect_frames(self, frames, angles=None):
        """All frames of a sweep, stage 1 batched by the compiled batch size, stage 2 in one call."""
        angles = list(angles) if angles is not None else [None] * len(frames)
        t0 = time.perf_counter()
        m = self.manifest
        crops = np.stack([core.prepare(f, m.crop, m.input_size) for f in frames])
        labels = []
        scores, maps = [], []
        for i in range(0, len(crops), self.batch):  # chunked here too, to collect pred_label per chunk
            s, mp = self.score_crops(crops[i:i + self.batch])
            scores.append(s)
            maps.append(mp)
            lab = getattr(self.backend, "last_labels", None)
            labels += [None] * len(s) if lab is None else [int(v) for v in lab[:len(s)]]
        scores, maps = np.concatenate(scores), np.concatenate(maps)
        per = (time.perf_counter() - t0) * 1e3 / len(frames)
        out = [self._frame_stage1(i, a, f.shape[:2], s, mp, per, lb)
               for i, (f, a, s, mp, lb) in enumerate(zip(frames, angles, scores, maps, labels))]
        recs, heats = [o[0] for o in out], [o[1] for o in out]
        recs = self._stage2_crops(list(frames), recs, heats)
        return list(zip(recs, heats))


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
    ap.add_argument("--manifest", "--model", dest="manifest", type=Path, required=True,
                    help="manifest.json, or a standard Anomalib OpenVINO export folder / .xml")
    ap.add_argument("--crop", default=None, help="x0,y0,x1,y1 ROI (overrides the manifest; needed for exports)")
    ap.add_argument("--score-space", choices=["raw", "normalized"], default=None,
                    help="exports only: cut the graph to raw scores, or keep the normalised ones (default)")
    ap.add_argument("--data", type=Path, required=True, help="folder with good/ and bad/ inspect frames")
    ap.add_argument("--device", default="NPU", choices=["NPU", "GPU", "CPU"])
    ap.add_argument("--batch", type=int, default=3, help="static batch compiled into the model (1 = sequential)")
    ap.add_argument("--overlays", type=Path, help="write heatmap/region overlays here")
    ap.add_argument("--stage2", default=None, help='"shape" | "none" | CNN classifier.json (default: manifest, else shape)')
    ap.add_argument("--stage2-device", default=None, help="OpenVINO device of the stage-2 CNN (default: --device)")
    ap.add_argument("--min-mark-confidence", type=float, default=0.0, help="reasoner's extra stage-2 threshold")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    insp = AnomalibInspector.from_model(args.manifest, device=args.device, batch=args.batch, stage2=args.stage2,
                                        stage2_device=args.stage2_device, crop=core.parse_box(args.crop),
                                        score_space=args.score_space)
    evaluate(insp, args.data.expanduser(), args.overlays, args.min_mark_confidence)


if __name__ == "__main__":
    main()
