"""One-class brick detection and shared, aspect-preserving inspection crops."""
from dataclasses import dataclass
import math
from pathlib import Path
import threading

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    box: tuple[int, int, int, int]
    confidence: float


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x1-x0) * max(0, y1-y0)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0.


def select_target(detections, zone, previous=None):
    """Select only an unambiguous target; never pick a different brick by size."""
    candidates = [d for d in detections if zone[0] <= (d.box[0]+d.box[2])/2 <= zone[2]
                  and zone[1] <= (d.box[1]+d.box[3])/2 <= zone[3]]
    if previous is not None:
        matches = [d for d in candidates if iou(d.box, previous) >= .3]
        if len(matches) == 1:
            return matches[0]
    return candidates[0] if len(candidates) == 1 else None


def inspection_crop(frame, box, padding=.05, size=320):
    """Return padded rectangular ROI letterboxed to square, plus source coordinates.

    Used identically for training preparation and live inference. Padding is a
    fraction of each box dimension; letterboxing avoids stretching the brick.
    """
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError('Expected a three-channel camera frame')
    h, w = frame.shape[:2]
    if len(box) != 4 or not all(math.isfinite(v) for v in box) or not 0 <= padding <= .5:
        raise ValueError('Invalid brick box or padding')
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        raise ValueError('Brick box must have positive area')
    dx, dy = (x1-x0)*padding, (y1-y0)*padding
    bounds = (max(0, math.floor(x0-dx)), max(0, math.floor(y0-dy)),
              min(w, math.ceil(x1+dx)), min(h, math.ceil(y1+dy)))
    a, b, c, d = bounds
    if c <= a or d <= b:
        raise ValueError('Brick box is outside the frame')
    crop = frame[b:d, a:c]
    scale = size / max(crop.shape[:2])
    rw, rh = max(1, round(crop.shape[1]*scale)), max(1, round(crop.shape[0]*scale))
    resized = cv2.resize(crop, (rw, rh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    result = np.full((size, size, 3), 127, dtype=np.uint8)
    x, y = (size-rw)//2, (size-rh)//2
    result[y:y+rh, x:x+rw] = resized
    return result, bounds


class BrickDetector:
    """Ultralytics handles detector preprocessing/NMS and exported OpenVINO input."""
    def __init__(self, weights, device='cpu', confidence=.5):
        from ultralytics import YOLO
        self.model = YOLO(str(weights), task='detect')
        # Initialize once with the intended device (names alone initializes a
        # separate default predictor for exported models).
        self.model.predict(np.zeros((640, 640, 3), np.uint8), imgsz=640,
                           conf=confidence, device=device, rect=False, verbose=False)
        if self.model.names != {0: 'brick'}:
            raise ValueError('Expected a custom one-class brick detector, not generic COCO weights')
        path = Path(weights)
        if str(device).lower() == 'cpu' and (path.is_dir() or path.suffix == '.xml'):
            # Ultralytics 8.4.212 leaves macOS CPU inference precision automatic.
            # Force FP32 for reproducible parity with the selected PyTorch model.
            import openvino as ov
            xml = next(path.glob('*.xml')) if path.is_dir() else path
            backend = self.model.predictor.model.backend
            backend.ov_compiled_model = ov.Core().compile_model(
                str(xml), 'CPU', {'PERFORMANCE_HINT': 'LATENCY', 'INFERENCE_PRECISION_HINT': 'f32'})
        self.device, self.confidence = device, confidence
        self.lock = threading.Lock()

    def detect(self, frame):
        with self.lock:
            result = self.model.predict(frame, imgsz=640, conf=self.confidence,
                                        device=self.device, rect=False, verbose=False)[0]
        h, w = frame.shape[:2]
        detections = []
        for box, confidence in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
            x0, y0, x1, y1 = box
            coords = (max(0, int(x0)), max(0, int(y0)), min(w, math.ceil(x1)), min(h, math.ceil(y1)))
            if coords[2]-coords[0] >= 12 and coords[3]-coords[1] >= 12:
                detections.append(Detection(coords, float(confidence)))
        return detections
