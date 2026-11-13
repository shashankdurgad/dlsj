"""Brick-in-view detection with the trained YOLO11n-cls model, run through OpenVINO.

Same interface as `dlsj.inspect_detect.BrickInViewDetector`: `update(bgr) -> Detection`, `reset()`, `last`,
`debug_image(bgr)`. The recorder and the deployment controller can use either.

Preprocessing matches Ultralytics' classify transforms:
* resize the shorter side to `imgsz`, antialiased;
* centre crop to `imgsz` x `imgsz`;
* RGB, scaled to 0-1, CHW.

"In view" means P(in_view) >= `threshold` for `stable_frames` consecutive frames.

Typical cost: ~0.15 ms per frame on the XPS NPU, ~0.6 ms on its CPU.
"""

from pathlib import Path

import cv2
import numpy as np

from dlsj.inspect_detect import Detection

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models/brick_in_view_v1/best.xml"


def preprocess(bgr: np.ndarray, imgsz: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    s = imgsz / min(h, w)
    nh, nw = max(imgsz, round(h * s)), max(imgsz, round(w * s))
    img = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    y0, x0 = (nh - imgsz) // 2, (nw - imgsz) // 2
    img = img[y0:y0 + imgsz, x0:x0 + imgsz, ::-1]  # crop, BGR -> RGB
    return np.ascontiguousarray(img.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0


class InViewClassifier:
    def __init__(self, model: str | Path = DEFAULT_MODEL, device: str = "CPU", threshold: float = 0.6,
                 stable_frames: int = 3, names: tuple[str, ...] = ("in_view", "not_in_view")):
        import openvino as ov

        core = ov.Core()
        compiled = core.compile_model(str(model), device, {"PERFORMANCE_HINT": "LATENCY"})
        self.request = compiled.create_infer_request()
        self.imgsz = int(compiled.inputs[0].get_partial_shape()[2].get_length())
        self.in_view_idx = names.index("in_view")
        self.threshold, self.stable_frames = threshold, stable_frames
        self.model, self.device = str(model), device
        self._stable = 0
        self.prob = 0.0
        self.last = Detection(False, False, "no frame yet")

    def reset(self) -> None:
        self._stable = 0

    def probability(self, bgr: np.ndarray) -> float:
        out = np.asarray(next(iter(self.request.infer([preprocess(bgr, self.imgsz)]).values())))[0]
        if not np.isclose(out.sum(), 1.0, atol=1e-3) or out.min() < 0:  # logits rather than probabilities
            out = np.exp(out - out.max())
            out = out / out.sum()
        return float(out[self.in_view_idx])

    def update(self, bgr: np.ndarray) -> Detection:
        self.prob = self.probability(bgr)
        candidate = self.prob >= self.threshold
        self._stable = self._stable + 1 if candidate else 0
        in_view = self._stable >= self.stable_frames
        reason = "in view" if in_view else ("settling" if candidate else "no brick in view")
        self.last = Detection(in_view, candidate, reason, area=self.prob, bbox=None, stable=self._stable)
        return self.last

    def debug_image(self, bgr: np.ndarray) -> np.ndarray:
        """`bgr` with the classifier's view (the centre crop) outlined and a P(in view) bar."""
        out = bgr.copy()
        h, w = out.shape[:2]
        side = min(h, w)
        x0, y0 = (w - side) // 2, (h - side) // 2
        d = self.last
        colour = (0, 220, 0) if d.in_view else ((0, 190, 255) if d.candidate else (0, 0, 255))
        cv2.rectangle(out, (x0, y0), (x0 + side - 1, y0 + side - 1), colour, 2)
        bar_w = int((w - 20) * self.prob)
        cv2.rectangle(out, (10, h - 22), (w - 10, h - 10), (60, 60, 60), -1)
        cv2.rectangle(out, (10, h - 22), (10 + bar_w, h - 10), colour, -1)
        tx = int(10 + (w - 20) * self.threshold)
        cv2.line(out, (tx, h - 26), (tx, h - 6), (255, 255, 255), 1)
        label = f"P(in view) {self.prob:.2f}  {d.reason}  stable {d.stable}/{self.stable_frames}"
        cv2.putText(out, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(out, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        return out
