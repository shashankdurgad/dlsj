"""YOLO-only brick inspector: plain / circle / cross per frame with the released YOLO26n classifier, one decision per
sweep. No Anomalib.

Per frame (`YoloInspector.classify`):

    inspect frame (RGB, contract) -> BGR -> brick crop -> 320x320 letterbox -> JPEG q95 round trip -> Tan's
    `dlsj.inspect.yolo_classifier.YoloClassifier.predict` (OpenVINO, RGB float32 NCHW /255) -> {plain, circle, cross}

* crop: the automatic brick detector (`dlsj.inspect.brick_detector`, only when `detector_weights` is given; no
  detector model is released yet) inside the ROI, else the configurable fixed inspect ROI. Both go through the
  shared `inspection_crop` (padded, aspect-preserving letterbox), the release's crop path. The JPEG q95 round trip
  matches the live tester (`live_web`: canvas crop -> toDataURL('image/jpeg', .95) -> /predict); `jpeg_quality=0`
  skips it.
* device: NPU by default, falling back to GPU then CPU if a device fails to compile. Tan's loader requests FP32
  (`INFERENCE_PRECISION_HINT f32`); if a device rejects that property it is compiled once more without it and
  `precision` says so.
* per-frame gate (the release's own, applied to one frame): confident = top probability >= 0.8 and top-2 margin
  >= 0.2. A frame's `FrameInspection` has score = 1 - P(plain), label PERFECT / DEFECTIVE (confident circle or
  cross) / UNCERTAIN, and the full probability vector in `probs`.

Sweep decision: `dlsj.inspect.aggregate` strategies `yolo_topk` (default) and `yolo_mean` (the release's naive mean).

Benchmark (per-frame latency on each device):

    python -m dlsj.inspect.yolo_inspector bench --model-dir ~/inference/models/brick-yolo-classifier-v1 \\
        --devices NPU,GPU,CPU --n 200 [--image some_inspect_frame.png --roi 160,80,480,400]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from dlsj.contracts import FrameInspection, InspectionResult, Label, MarkType
from dlsj.inspect.aggregate import AggregateConfig, SweepInspectorMixin, YOLO_CLASSES, gate

log = logging.getLogger(__name__)

DEVICE_FALLBACK = ("NPU", "GPU", "CPU")
Box = tuple[int, int, int, int]


def fallback_chain(device: str) -> list[str]:
    """`device` first, then the remaining NPU/GPU/CPU in that order (CPU always last)."""
    device = device.upper()
    return [device] + [d for d in DEVICE_FALLBACK if d != device]


def load_classifier(model_dir: str | Path, device: str = "NPU", fallback: bool = True):
    """Tan's `YoloClassifier` on the first device that compiles. Returns (classifier, notes)."""
    from dlsj.inspect.yolo_classifier import YoloClassifier

    notes: list[str] = []
    for dev in (fallback_chain(device) if fallback else [device.upper()]):
        try:
            clf = YoloClassifier(model_dir, dev)
            clf.precision = "f32"
            return clf, notes
        except Exception as e:  # noqa: BLE001 - a device may be missing or reject the FP32 hint
            notes.append(f"{dev} (f32 hint): {_first_line(e)}")
            log.warning("YOLO classifier on %s with the f32 hint failed: %s", dev, e)
        if dev == "CPU":
            continue
        try:
            clf = _compile_default_precision(YoloClassifier, model_dir, dev)
            notes.append(f"{dev}: compiled without INFERENCE_PRECISION_HINT (device default precision)")
            return clf, notes
        except Exception as e:  # noqa: BLE001
            notes.append(f"{dev}: {_first_line(e)}")
            log.warning("YOLO classifier on %s failed: %s", dev, e)
    raise RuntimeError("could not compile the YOLO classifier on any device:\n  " + "\n  ".join(notes))


def _first_line(e: Exception) -> str:
    """OpenVINO errors are multi-line stacks; keep the line that says what is wrong."""
    lines = [ln.strip() for ln in str(e).splitlines() if ln.strip() and not ln.strip().startswith("Exception from")]
    return f"{type(e).__name__}: {lines[-1] if lines else str(e).strip()}"


def _compile_default_precision(cls, model_dir, device):
    """Same object as `YoloClassifier(model_dir, device)` (Tan's `predict` and preprocessing are reused), compiled
    without the FP32 precision hint, for devices that reject it."""
    import threading

    import openvino as ov

    d = Path(model_dir)
    clf = cls.__new__(cls)
    clf.config = json.loads((d / "classifier.json").read_text())
    if set(clf.config["classes"]) != set(YOLO_CLASSES):
        raise ValueError("expected plain/cross/circle classes")
    clf.classes, clf.size, clf.lock, clf.device = clf.config["classes"], clf.config["input_size"], threading.Lock(), device
    clf.compiled = ov.Core().compile_model(str(d / clf.config["model"]), device,
                                           {"PERFORMANCE_HINT": "LATENCY", "CACHE_DIR": str(d / f"runtime-cache-{device}")})
    clf.precision = "device default"
    return clf


def jpeg_roundtrip(bgr: np.ndarray, quality: int) -> np.ndarray:
    if not quality:
        return bgr
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError("could not encode the crop")
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def default_roi(w: int = 640, h: int = 480, frac: float = 0.6) -> Box:
    """A centred square ROI covering `frac` of the shorter side (the live tester's default view)."""
    side = int(min(w, h) * frac)
    x0, y0 = (w - side) // 2, (h - side) // 2
    return x0, y0, x0 + side, y0 + side


class YoloInspector(SweepInspectorMixin):
    """`Inspector` (+ sweep API) backed only by the YOLO brick classifier. Frames are HxWx3 uint8 **RGB**."""

    def __init__(self, model_dir: str | Path | None = None, device: str = "NPU", roi: Box | None = None,
                 padding: float = 0.0, jpeg_quality: int = 95, detector_weights: str | None = None,
                 detector_device: str = "cpu", agg: AggregateConfig | None = None, fallback: bool = True,
                 classifier=None):
        self.model_dir = Path(model_dir).expanduser() if model_dir else None
        self.roi, self.padding, self.jpeg_quality, self.fallback = roi, float(padding), int(jpeg_quality), fallback
        self.agg = agg or AggregateConfig(strategy="yolo_topk")
        self.requested_device = device.upper()
        self.notes: list[str] = []
        self._cache: dict[str, object] = {}
        if classifier is not None:  # tests / pre-built
            self.classifier = classifier
        else:
            if self.model_dir is None:
                raise ValueError("model_dir is required")
            self.classifier, self.notes = load_classifier(self.model_dir, device, fallback)
        self._cache[self.device] = self.classifier
        self.detector = None
        self._prev_box = None
        if detector_weights:
            from dlsj.inspect.brick_detector import BrickDetector

            self.detector = BrickDetector(detector_weights, detector_device)
        self.min_conf = float(getattr(self.classifier, "config", {}).get("min_confidence", self.agg.yolo_min_conf))
        self.min_margin = float(getattr(self.classifier, "config", {}).get("min_margin", self.agg.yolo_min_margin))

    # ------------------------------------------------------------------ device
    @property
    def device(self) -> str:
        return str(getattr(self.classifier, "device", "CPU"))

    def set_device(self, device: str) -> str:
        """Switch the OpenVINO device (compiled models are cached per device). Returns the device in use."""
        device = device.upper()
        if device == self.device:
            return device
        if device in self._cache:
            self.classifier = self._cache[device]
            return device
        if self.model_dir is None:
            raise RuntimeError("no model_dir to recompile from")
        clf, notes = load_classifier(self.model_dir, device, fallback=False)
        self.notes += notes
        self._cache[clf.device] = clf
        self.classifier = clf
        return clf.device

    def health(self) -> dict:
        h = self.classifier.health() if hasattr(self.classifier, "health") else {}
        return {**h, "device": self.device, "requested_device": self.requested_device,
                "precision": getattr(self.classifier, "precision", "?"), "notes": self.notes,
                "roi": list(self.roi) if self.roi else None, "padding": self.padding,
                "jpeg_quality": self.jpeg_quality, "detector": self.detector is not None,
                "model_dir": str(self.model_dir) if self.model_dir else None,
                "min_confidence": self.min_conf, "min_margin": self.min_margin}

    # ------------------------------------------------------------------ crop + classify
    def crop_box(self, bgr: np.ndarray) -> tuple[Box, str]:
        h, w = bgr.shape[:2]
        roi = self.roi or default_roi(w, h)
        if self.detector is not None:
            from dlsj.inspect.brick_detector import select_target

            target = select_target(self.detector.detect(bgr), roi, self._prev_box)
            if target is not None:
                self._prev_box = target.box
                return tuple(int(v) for v in target.box), "detector"
        return roi, "roi"

    def crop(self, frame_rgb: np.ndarray) -> tuple[np.ndarray, Box, str]:
        """(320x320 BGR model crop, source bounds, "detector" | "roi")."""
        from dlsj.inspect.brick_detector import inspection_crop

        bgr = np.ascontiguousarray(frame_rgb[..., ::-1])
        box, source = self.crop_box(bgr)
        crop, bounds = inspection_crop(bgr, box, padding=self.padding, size=320)
        return jpeg_roundtrip(crop, self.jpeg_quality), tuple(int(v) for v in bounds), source

    def classify(self, frame_rgb: np.ndarray) -> dict:
        """One frame: {probs, confident (class | None), top, conf, margin, box, source, crop (BGR), ms, model_ms}."""
        t0 = time.perf_counter()
        crop, box, source = self.crop(frame_rgb)
        t1 = time.perf_counter()
        probs = self.classifier.predict(crop)
        t2 = time.perf_counter()
        probs = {c: float(probs[c]) for c in YOLO_CLASSES}
        confident, top, conf, margin = gate(probs, self.min_conf, self.min_margin)
        return {"probs": probs, "confident": confident, "top": top, "conf": conf, "margin": margin, "box": box,
                "source": source, "crop": crop, "ms": (t2 - t0) * 1e3, "model_ms": (t2 - t1) * 1e3,
                "device": self.device}

    @staticmethod
    def to_frame(index: int, angle: float | None, c: dict) -> FrameInspection:
        """The `FrameInspection` of one classified frame (see the module docstring for the mapping)."""
        p = c["probs"]
        if c["confident"] == "plain":
            label = Label.PERFECT
        elif c["confident"] in ("circle", "cross"):
            label = Label.DEFECTIVE
        else:
            label = Label.UNCERTAIN
        marked = p["circle"] + p["cross"]
        p_cross = p["cross"] / marked if marked > 0 else 0.5
        mark = MarkType(c["confident"]) if label is Label.DEFECTIVE else None
        return FrameInspection(index, None if angle is None else float(angle), 1.0 - p["plain"], label,
                               tuple(c["box"]) if label is not Label.PERFECT else None, mark,
                               float(c["conf"]) if mark is not None else 0.0,
                               p_cross if label is Label.DEFECTIVE else None, float(c["ms"]), None, dict(p))

    # ------------------------------------------------------------------ Inspector protocol + sweep API
    def inspect_frame(self, frame: np.ndarray, index: int = 0, angle: float | None = None):
        return self.to_frame(index, angle, self.classify(frame)), None

    def sweep_meta(self) -> dict:
        # score = 1 - P(plain); the band mirrors the per-frame gate so the anomaly strategies stay comparable
        return {"threshold": 0.5, "band_low": 1.0 - self.min_conf, "band_high": self.min_conf,
                "device": self.device, "mark_device": self.device}

    def inspect(self, frames: list[np.ndarray]) -> InspectionResult:
        """The classic few-frame inspection with the release's rule (mean of the probability vectors)."""
        return self.inspect_sweep(frames, None, "yolo_mean")


# ----------------------------------------------------------------------------- benchmark
def bench(model_dir: str, devices: list[str], n: int = 200, warmup: int = 10, image: str | None = None,
          roi: Box | None = None) -> list[dict]:
    """Per-frame latency on each device: model only (`predict` on a 320 crop) and end to end (crop + JPEG +
    predict from a 640x480 frame)."""
    if image:
        from dlsj.inspect.core import read_rgb

        frame = read_rgb(image)
    else:
        frame = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    out = []
    for dev in devices:
        row: dict = {"device": dev}
        try:
            t0 = time.perf_counter()
            insp = YoloInspector(model_dir, dev, roi=roi, fallback=False)
            row["compile_s"] = round(time.perf_counter() - t0, 2)
            row["precision"] = getattr(insp.classifier, "precision", "?")
            row["notes"] = insp.notes
            crop, _, _ = insp.crop(frame)
            for _ in range(warmup):
                insp.classifier.predict(crop)
            m, e = [], []
            for _ in range(n):
                t = time.perf_counter()
                insp.classifier.predict(crop)
                m.append((time.perf_counter() - t) * 1e3)
            for _ in range(n):
                e.append(insp.classify(frame)["ms"])
            c = insp.classify(frame)
            for name, v in (("model", m), ("e2e", e)):
                v = np.array(v)
                row[name] = {"median_ms": round(float(np.median(v)), 3), "p90_ms": round(float(np.percentile(v, 90)), 3),
                             "p99_ms": round(float(np.percentile(v, 99)), 3), "mean_ms": round(float(v.mean()), 3)}
            row["sample"] = {"probs": {k: round(x, 4) for k, x in c["probs"].items()}, "confident": c["confident"]}
        except Exception as ex:  # noqa: BLE001
            row["error"] = f"{type(ex).__name__}: {ex}"
        out.append(row)
        print(json.dumps(row), flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench", help="per-frame latency per device")
    b.add_argument("--model-dir", required=True)
    b.add_argument("--devices", default="NPU,GPU,CPU")
    b.add_argument("--n", type=int, default=200)
    b.add_argument("--image", default=None, help="an inspect frame (PNG/JPEG); default random noise")
    b.add_argument("--roi", default=None, help="x0,y0,x1,y1")
    b.add_argument("--json", default=None, help="write the rows here")
    c = sub.add_parser("classify", help="classify image files")
    c.add_argument("--model-dir", required=True)
    c.add_argument("--device", default="NPU")
    c.add_argument("--roi", default=None)
    c.add_argument("images", nargs="+")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from dlsj.inspect.core import parse_box, read_rgb

    if args.cmd == "bench":
        rows = bench(args.model_dir, [d.strip().upper() for d in args.devices.split(",") if d.strip()], args.n,
                     image=args.image, roi=parse_box(args.roi))
        if args.json:
            Path(args.json).write_text(json.dumps(rows, indent=1) + "\n")
        return 0
    insp = YoloInspector(args.model_dir, args.device, roi=parse_box(args.roi))
    for p in args.images:
        r = insp.classify(read_rgb(p))
        print(p, r["confident"] or f"unsure({r['top']})", {k: round(v, 3) for k, v in r["probs"].items()},
              f"{r['ms']:.1f} ms on {r['device']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
