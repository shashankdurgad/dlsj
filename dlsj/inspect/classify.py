"""Stage 2 of the inspection: classify the mark that stage 1 (Anomalib) localised as a CIRCLE or a CROSS.

Two back-ends behind one interface (`MarkClassifier.classify(crops, heats) -> MarkPrediction`):

* `ShapeRuleClassifier` (default, no training): segment the mark inside the region crop by its colour difference
  to the brick's dominant colour (dark-on-light or light-on-dark), pick the component the anomaly heatmap points
  at, then contour features: hole (enclosed area), circularity 4piA/P^2, solidity, convexity defects, centre fill,
  radial spread and long Hough lines. A circle is a closed ring with a hole, high circularity, empty centre and a
  narrow radial profile; a cross is two crossing strokes, low solidity, a filled centre and two line directions.
  The features vote into a logistic score; `fit_platt` calibrates it on labelled crops.
* `CNNMarkClassifier`: a MobileNetV3-small / ResNet18 fine-tuned by `train_classifier.py`, exported to OpenVINO
  with a static u8 NHWC input (pre-processing in the graph) so it compiles on the NPU (GPU/CPU fallback).

Both see the same crop: `crop_for_stage2(frame, region)` (the stage-1 box, padded and made square). Per inspection,
every frame's crop is classified and the P(cross) values are averaged; the result is UNKNOWN when no mark is found
in most frames, when frames clearly disagree, or when the confidence is below `min_confidence`.

    python -m dlsj.inspect.classify eval --crops data/mark_crops_test                 # shape rules on cross/ circle/
    python -m dlsj.inspect.classify calibrate --crops data/mark_crops --out shape_rules.json
    python -m dlsj.inspect.classify eval --crops data/mark_crops_test --classifier exports/mark_cnn/classifier.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from dlsj.contracts import MarkType

log = logging.getLogger("inspect.classify")

Box = tuple[int, int, int, int]
CLASSES = ("circle", "cross")  # index 1 = cross: every back-end reports P(cross)


# ----------------------------------------------------------------------------- crop geometry (shared everywhere)
def crop_box(region: Box, frame_hw: tuple[int, int], pad_frac: float = 0.3, min_pad: int = 6,
             min_side: int = 24) -> Box:
    """The stage-1 region grown by `pad_frac` of its longer side (at least `min_pad` px), made square, clamped."""
    x0, y0, x1, y1 = region
    fh, fw = frame_hw
    side = max(x1 - x0, y1 - y0)
    side = max(int(round(side + 2 * max(pad_frac * side, min_pad))), min_side)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    nx0, ny0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    nx0, ny0 = min(max(0, nx0), max(0, fw - side)), min(max(0, ny0), max(0, fh - side))
    return nx0, ny0, min(fw, nx0 + side), min(fh, ny0 + side)


def crop_for_stage2(frame: np.ndarray, region: Box, pad_frac: float = 0.3, min_pad: int = 6) -> np.ndarray:
    x0, y0, x1, y1 = crop_box(region, frame.shape[:2], pad_frac, min_pad)
    return frame[y0:y1, x0:x1]


def to_input(crop: np.ndarray, size: int | tuple[int, int] = 128) -> np.ndarray:
    hw = (size, size) if isinstance(size, int) else tuple(size)
    if crop.shape[:2] == hw:
        return np.ascontiguousarray(crop)
    interp = cv2.INTER_AREA if crop.shape[0] > hw[0] else cv2.INTER_LINEAR
    return cv2.resize(crop, (hw[1], hw[0]), interpolation=interp)


# ----------------------------------------------------------------------------- prediction + aggregation
@dataclass(frozen=True)
class MarkPrediction:
    mark: MarkType  # CIRCLE / CROSS / UNKNOWN
    confidence: float  # probability of the more likely class
    p_cross: float  # mean P(cross) over the frames where a mark was found (nan if none)
    per_crop: tuple[float | None, ...] = ()  # P(cross) per crop, None = no mark found in that crop
    reason: str = ""
    features: dict = field(default_factory=dict)  # shape rules: features of the first crop (for logs)


def aggregate(p_cross: list[float | None], min_confidence: float, disagree: float = 0.35) -> MarkPrediction:
    """Mean P(cross) over crops; UNKNOWN when most crops show no mark, crops disagree, or confidence is low."""
    valid = [p for p in p_cross if p is not None and not math.isnan(p)]
    n = len(p_cross)
    if not valid or len(valid) * 2 < n:
        return MarkPrediction(MarkType.UNKNOWN, 0.0, float("nan"), tuple(p_cross),
                              f"no mark found in {n - len(valid)}/{n} crops")
    p = float(np.mean(valid))
    conf = max(p, 1 - p)
    if len(valid) > 1 and max(valid) > 0.5 + disagree and min(valid) < 0.5 - disagree:
        return MarkPrediction(MarkType.UNKNOWN, conf, p, tuple(p_cross), "crops disagree")
    mark = MarkType.CROSS if p >= 0.5 else MarkType.CIRCLE
    if conf < min_confidence:
        return MarkPrediction(MarkType.UNKNOWN, conf, p, tuple(p_cross),
                              f"{mark.value}? confidence {conf:.2f} < {min_confidence:.2f}")
    return MarkPrediction(mark, conf, p, tuple(p_cross), f"{mark.value} {conf:.2f}")


class MarkClassifier:
    """Interface: `classify(crops, heats)`. Sub-classes implement `p_cross(crops, heats) -> [P(cross) | None]`."""

    name = "base"
    device = ""
    min_confidence = 0.7
    pad_frac = 0.3

    def p_cross(self, crops: list[np.ndarray], heats: list[np.ndarray | None] | None = None) -> list[float | None]:
        raise NotImplementedError

    def classify(self, crops: list[np.ndarray], heats: list[np.ndarray | None] | None = None) -> MarkPrediction:
        if not crops:
            return MarkPrediction(MarkType.UNKNOWN, 0.0, float("nan"), (), "no crops")
        pred = aggregate(self.p_cross(crops, heats), self.min_confidence)
        feats = getattr(self, "last_features", None)
        return pred if not feats else MarkPrediction(pred.mark, pred.confidence, pred.p_cross, pred.per_crop,
                                                     pred.reason, feats)


# ----------------------------------------------------------------------------- shape rules
WORK = 96  # crops are resized so the longer side is WORK px before segmentation


@dataclass
class ShapeFeatures:
    found: bool
    area_frac: float = 0.0  # mark pixels / crop pixels
    contrast: float = 0.0  # Lab distance threshold that separated mark and brick
    hole_frac: float = 0.0  # enclosed area / filled area (ring: large, cross: ~0)
    circularity: float = 0.0  # 4 pi A / P^2 of the filled outer contour (disc 1, cross ~0.3)
    solidity: float = 0.0  # mark area / convex hull area
    filled_solidity: float = 0.0  # filled outer contour area / hull area (ring ~0.95, cross ~0.5)
    defects: int = 0  # convexity defects deeper than 15 % of the equivalent radius (cross 4, ring 0)
    center_fill: float = 0.0  # mark fraction in a small disc at the hull centre (cross ~1, ring 0)
    radial_cv: float = 0.0  # std/mean of mark-pixel distances to the hull centre (ring small, cross ~0.5)
    line_dirs: int = 0  # distinct directions of long Hough lines (cross 2, ring 0)

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def _dominant_lab(lab: np.ndarray, sigma: float = 0.3) -> np.ndarray:
    """Brick colour: the densest bin of a coarse, centre-weighted Lab histogram (Gaussian weight, `sigma` x side).

    The stage-1 box is centred on the mark, so the brick around it outweighs background/gripper at the crop edges.
    """
    h, w = lab.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    wgt = np.exp(-(((xx - (w - 1) / 2) / w) ** 2 + ((yy - (h - 1) / 2) / h) ** 2) / (2 * sigma ** 2)).ravel()
    q = (lab // 16).astype(np.int32).reshape(-1, 3)
    keys = q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2]
    k = np.bincount(keys, weights=wgt).argmax()
    sel = keys == k
    return np.median(lab.reshape(-1, 3)[sel], axis=0)


def segment_mark(crop_rgb: np.ndarray, heat: np.ndarray | None = None, min_contrast: float = 22.0
                 ) -> tuple[np.ndarray | None, float]:
    """Binary mask (WORK-scale) of the mark component, or None. Returns (mask, Lab contrast threshold)."""
    h, w = crop_rgb.shape[:2]
    s = WORK / max(h, w)
    img = cv2.resize(crop_rgb, (max(8, round(w * s)), max(8, round(h * s))), interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    ref = _dominant_lab(lab)
    d = np.linalg.norm(lab - ref, axis=2)
    d8 = np.clip(d * 2, 0, 255).astype(np.uint8)  # Lab units x2 (Otsu works on 8 bit)
    t, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(t / 2.0, min_contrast)
    mask = (d > thr).astype(np.uint8)
    # Brick face = convex hull of the largest brick-coloured component (the mark sits inside it). Cutting the mask
    # to the slightly eroded face drops background, gripper and the brick's edge outline, which otherwise connect
    # a mark drawn near the edge to the background.
    nb, bl, bst, _ = cv2.connectedComponentsWithStats(1 - mask, connectivity=4)
    if nb > 1:
        kb = 1 + int(np.argmax(bst[1:, cv2.CC_STAT_AREA]))
        pts = cv2.findNonZero((bl == kb).astype(np.uint8))
        face = np.zeros_like(mask)
        cv2.fillConvexPoly(face, cv2.convexHull(pts), 1)
        face = cv2.erode(face, np.ones((3, 3), np.uint8), iterations=2)
        if face.sum() > 0.2 * mask.size:
            mask &= face
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab_cc, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None, thr
    hh, ww = mask.shape
    if heat is not None and heat.size:
        hm = cv2.resize(heat.astype(np.float32), (ww, hh), interpolation=cv2.INTER_LINEAR)
    else:
        hm = None
    yy, xx = np.mgrid[0:hh, 0:ww]
    centre_w = np.exp(-(((xx - ww / 2) / (0.35 * ww)) ** 2 + ((yy - hh / 2) / (0.35 * hh)) ** 2))
    best, best_score = 0, 0.0
    for k in range(1, n):
        x, y, bw, bh, area = stats[k]
        if area < 0.01 * hh * ww:
            continue
        sel = lab_cc == k
        score = float(area) * float(centre_w[sel].mean())
        if hm is not None:
            score *= 0.25 + float(hm[sel].mean())
        sides = int(x == 0) + int(y == 0) + int(x + bw >= ww) + int(y + bh >= hh)
        score *= (1.0, 0.8, 0.3, 0.1, 0.05)[sides]  # background and gripper touch several crop sides
        if score > best_score:
            best, best_score = k, score
    if best == 0:
        return None, thr
    # Keep other big components close to the chosen one (a cross drawn as two strokes that barely touch).
    keep = lab_cc == best
    x, y, bw, bh, area = stats[best]
    for k in range(1, n):
        if k == best:
            continue
        kx, ky, kw, kh, ka = stats[k]
        ccx, ccy = cent[k]
        sides = int(kx == 0) + int(ky == 0) + int(kx + kw >= ww) + int(ky + kh >= hh)
        if ka >= 0.3 * area and sides <= 1 and x <= ccx <= x + bw and y <= ccy <= y + bh:
            keep |= lab_cc == k
    return keep.astype(np.uint8), thr


def shape_features(mask: np.ndarray) -> ShapeFeatures:
    m = (mask > 0).astype(np.uint8)
    area = float(m.sum())
    contours, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours or area < 10:
        return ShapeFeatures(False)
    outer = [c for c, h in zip(contours, hier[0]) if h[3] < 0]
    pts = np.concatenate(outer)
    hull = cv2.convexHull(pts)
    hull_area = max(float(cv2.contourArea(hull)), 1.0)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, outer, -1, 1, -1)
    filled_area = float(filled.sum())
    big = max(outer, key=cv2.contourArea)
    perim = max(float(cv2.arcLength(big, True)), 1.0)
    circ = float(4 * np.pi * cv2.contourArea(big) / perim ** 2)

    r_eq = math.sqrt(hull_area / math.pi)
    defects = 0
    hull_idx = cv2.convexHull(big, returnPoints=False)
    if hull_idx is not None and len(hull_idx) > 3 and len(big) > 3:
        try:
            dd = cv2.convexityDefects(big, np.sort(hull_idx, axis=0))
        except cv2.error:
            dd = None
        if dd is not None:
            defects = int((dd.reshape(-1, 4)[:, 3] / 256.0 > 0.15 * r_eq).sum())

    mu = cv2.moments(hull)
    cx, cy = (mu["m10"] / mu["m00"], mu["m01"] / mu["m00"]) if mu["m00"] else (m.shape[1] / 2, m.shape[0] / 2)
    ys, xs = np.nonzero(m)
    r = np.hypot(xs - cx, ys - cy)
    rad_cv = float(r.std() / max(r.mean(), 1e-6))
    disc = np.zeros_like(m)
    cv2.circle(disc, (int(round(cx)), int(round(cy))), max(2, int(round(0.15 * r_eq))), 1, -1)
    center_fill = float((m & disc).sum() / max(disc.sum(), 1))

    edges = cv2.Canny(m * 255, 50, 150)
    diam = 2 * r_eq
    lines = cv2.HoughLinesP(edges, 1, np.pi / 90, threshold=max(8, int(0.3 * diam)),
                            minLineLength=max(6, int(0.55 * diam)), maxLineGap=max(2, int(0.08 * diam)))
    dirs: list[float] = []
    if lines is not None:
        for x0, y0, x1, y1 in np.asarray(lines).reshape(-1, 4):
            a = math.degrees(math.atan2(y1 - y0, x1 - x0)) % 180
            if all(min(abs(a - b), 180 - abs(a - b)) > 25 for b in dirs):
                dirs.append(a)
    return ShapeFeatures(True, area / m.size, 0.0, max(0.0, (filled_area - area) / max(filled_area, 1.0)), circ,
                         area / hull_area, filled_area / hull_area, defects, center_fill, rad_cv, len(dirs))


# Evidence per feature in [-1, 1], positive = cross; weights sum the votes into a logit.
EVIDENCE = {  # feature: (pivot, scale, sign) -> clip(sign * (value - pivot) / scale, -1, 1)
    "hole_frac": (0.12, 0.10, -1),
    "center_fill": (0.35, 0.25, +1),
    "filled_solidity": (0.78, 0.12, -1),
    "radial_cv": (0.33, 0.10, +1),
    "circularity": (0.55, 0.20, -1),
    "defects": (1.5, 1.5, +1),
    "line_dirs": (0.8, 1.0, +1),
}
DEFAULT_WEIGHTS = {"hole_frac": 1.2, "center_fill": 2.0, "filled_solidity": 1.0, "radial_cv": 1.5,
                   "circularity": 0.8, "defects": 0.8, "line_dirs": 0.5}


def evidence(f: ShapeFeatures) -> dict[str, float]:
    return {k: float(np.clip(sg * (getattr(f, k) - pv) / sc, -1, 1)) for k, (pv, sc, sg) in EVIDENCE.items()}


class ShapeRuleClassifier(MarkClassifier):
    """No-training stage 2. `platt` = (a, b): P(cross) = sigmoid(a * logit + b), fitted by `fit_platt`."""

    name = "shape"
    device = "SHAPE-CPU"

    def __init__(self, min_confidence: float = 0.7, weights: dict | None = None, platt: tuple[float, float] = (1.0, 0.0),
                 pad_frac: float = 0.3, min_contrast: float = 22.0, blob_solidity: float = 0.85):
        self.min_confidence = min_confidence
        self.blob_solidity = blob_solidity
        self.weights = dict(DEFAULT_WEIGHTS, **(weights or {}))
        self.platt = tuple(platt)
        self.pad_frac = pad_frac
        self.min_contrast = min_contrast
        self.last_features: dict = {}

    @classmethod
    def from_config(cls, cfg: dict | str | Path | None) -> "ShapeRuleClassifier":
        if cfg is None:
            return cls()
        if not isinstance(cfg, dict):
            cfg = json.loads(Path(cfg).expanduser().read_text())
        keys = ("min_confidence", "weights", "platt", "pad_frac", "min_contrast", "blob_solidity")
        return cls(**{k: cfg[k] for k in keys if k in cfg})

    def logit(self, f: ShapeFeatures) -> float:
        e = evidence(f)
        return float(sum(self.weights[k] * e[k] for k in e))

    def features(self, crop: np.ndarray, heat: np.ndarray | None = None) -> ShapeFeatures:
        mask, thr = segment_mark(crop, heat, self.min_contrast)
        if mask is None:
            return ShapeFeatures(False, contrast=thr)
        f = shape_features(mask)
        f.contrast = thr
        return f

    def p_cross(self, crops, heats=None):
        heats = heats or [None] * len(crops)
        out: list[float | None] = []
        self.last_features = {}
        for crop, heat in zip(crops, heats):
            f = self.features(crop, heat)
            if not self.last_features:
                self.last_features = f.as_dict()
            if not f.found or f.area_frac < 0.02:
                out.append(None)
                continue
            if f.solidity > self.blob_solidity:  # a solid blob (smudge, sticker, crop inside a thick ring):
                out.append(0.5)  # neither a ring nor two strokes -> ambiguous
                continue
            a, b = self.platt
            out.append(1.0 / (1.0 + math.exp(-(a * self.logit(f) + b))))
        return out

    def config(self) -> dict:
        return {"backend": "shape", "min_confidence": self.min_confidence, "weights": self.weights,
                "platt": list(self.platt), "pad_frac": self.pad_frac, "min_contrast": self.min_contrast,
                "blob_solidity": self.blob_solidity}


def fit_platt(logits, labels, iters: int = 100, l2: float = 1e-3) -> tuple[float, float]:
    """Logistic calibration P = sigmoid(a * logit + b): damped Newton on the log-loss (labels: 1 = cross)."""
    x = np.asarray(logits, np.float64)
    y = np.asarray(labels, np.float64)
    X = np.stack([x, np.ones_like(x)], 1)

    def loss(w):
        z = X @ w
        return float(np.sum(np.logaddexp(0, z) - y * z) + 0.5 * l2 * w @ w)

    w = np.array([1.0, 0.0])
    cur = loss(w)
    for _ in range(iters):
        p = 0.5 * (1 + np.tanh(0.5 * (X @ w)))  # stable sigmoid
        g = X.T @ (p - y) + l2 * w
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(2)
        step = np.linalg.solve(H, g)
        t = 1.0
        while t > 1e-6 and loss(w - t * step) > cur:  # backtracking: never increase the loss
            t *= 0.5
        w, prev, cur = w - t * step, cur, loss(w - t * step)
        if prev - cur < 1e-10:
            break
    return float(w[0]), float(w[1])


# ----------------------------------------------------------------------------- CNN (OpenVINO)
class CNNMarkClassifier(MarkClassifier):
    """OpenVINO IR from `train_classifier.py`: static [B,H,W,3] u8 RGB in, (B,2) class probabilities out."""

    name = "cnn"

    def __init__(self, xml: str | Path, input_hw: tuple[int, int] = (128, 128), classes=CLASSES, device: str = "NPU",
                 fallback: tuple[str, ...] = ("NPU", "GPU", "CPU"), batch: int = 1, min_confidence: float = 0.7,
                 pad_frac: float = 0.3, cache_dir: str | Path | None = None):
        import openvino as ov

        self.input_hw, self.classes = tuple(input_hw), tuple(classes)
        if "cross" not in self.classes:
            raise ValueError(f"classes {self.classes} need a 'cross' entry")
        self.cross_idx = self.classes.index("cross")
        self.min_confidence, self.pad_frac = min_confidence, pad_frac
        core = ov.Core()
        if cache_dir:
            core.set_property({"CACHE_DIR": str(cache_dir)})
        model = core.read_model(str(xml))
        model.reshape({model.input(0).get_any_name(): [batch, *self.input_hw, 3]})  # static: the NPU needs it
        order = [device] + [d for d in fallback if d != device]
        errors = {}
        for dev in order:
            if not any(a.startswith(dev) for a in core.available_devices):
                errors[dev] = "not available"
                continue
            try:
                self.compiled = core.compile_model(model, dev, {"PERFORMANCE_HINT": "LATENCY"})
                self.device = dev
                break
            except Exception as e:  # noqa: BLE001
                errors[dev] = str(e).strip().splitlines()[-1][:300]
                log.warning("stage-2 CNN compile on %s failed: %s", dev, errors[dev])
        else:
            raise RuntimeError(f"could not compile {xml} on any of {order}: {errors}")
        if self.device != device:
            log.warning("stage-2 CNN: requested %s, running on %s (%s)", device, self.device, errors.get(device))
        self.batch = batch
        self.request = self.compiled.create_infer_request()

    @classmethod
    def from_config(cls, path: str | Path, device: str = "NPU", **kw) -> "CNNMarkClassifier":
        path = Path(path).expanduser()
        cfg = json.loads(path.read_text())
        return cls(path.parent / cfg["model"], tuple(cfg["input_size"]), tuple(cfg["classes"]), device,
                   min_confidence=cfg.get("min_confidence", 0.7), pad_frac=cfg.get("pad_frac", 0.3), **kw)

    def probs(self, batch_u8: np.ndarray) -> np.ndarray:
        n = len(batch_u8)
        out = []
        for i in range(0, n, self.batch):
            chunk = batch_u8[i:i + self.batch]
            k = len(chunk)
            if k < self.batch:
                chunk = np.concatenate([chunk, np.repeat(chunk[-1:], self.batch - k, 0)])
            try:
                self.request.infer({0: np.ascontiguousarray(chunk)})
            except RuntimeError as e:  # NPU hiccup seen with stage 1: retry once with a new request
                log.warning("stage-2 %s inference failed (%s); retrying", self.device, str(e).strip()[-200:])
                self.request = self.compiled.create_infer_request()
                self.request.infer({0: np.ascontiguousarray(chunk)})
            out.append(self.request.get_output_tensor(0).data.reshape(self.batch, -1)[:k].copy())
        return np.concatenate(out)

    def p_cross(self, crops, heats=None):
        x = np.stack([to_input(c, self.input_hw) for c in crops])
        return [float(p) for p in self.probs(x)[:, self.cross_idx]]


class NullClassifier(MarkClassifier):
    """Stage 2 off: the inspector leaves `mark` None (a marked brick counts as a cross: two-plate behaviour)."""

    name = "none"


def load_classifier(spec: str | dict | Path | None, device: str = "NPU", base_dir: Path | None = None,
                    **kw) -> MarkClassifier | None:
    """"shape" | "none" | "cnn:<classifier.json>" | <classifier.json or shape_rules.json path> | manifest dict."""
    if spec is None or spec == "" or spec == "none":
        return None
    if isinstance(spec, dict):
        backend = spec.get("backend", "shape")
        if backend == "none":
            return None
        if backend == "shape":
            return ShapeRuleClassifier.from_config(spec.get("config") or spec)
        p = Path(spec["manifest"]).expanduser()
        p = p if p.is_absolute() or base_dir is None else base_dir / p
        clf = CNNMarkClassifier.from_config(p, device=spec.get("device", device), **kw)
        if "min_confidence" in spec:
            clf.min_confidence = spec["min_confidence"]
        return clf
    spec = str(spec)
    if spec == "shape":
        return ShapeRuleClassifier()
    if spec.startswith("cnn:"):
        spec = spec[4:]
    p = Path(spec).expanduser()
    cfg = json.loads(p.read_text())
    if cfg.get("backend") == "shape":
        return ShapeRuleClassifier.from_config(cfg)
    return CNNMarkClassifier.from_config(p, device=device, **kw)


# ----------------------------------------------------------------------------- crop folders: eval / calibrate
def labelled_crops(folder: str | Path) -> list[tuple[Path, int]]:
    """`folder/circle/*`, `folder/cross/*` (recursive) -> [(path, 1 if cross else 0)]."""
    from dlsj.inspect.core import IMAGE_EXTS

    out = []
    for cls_i, name in enumerate(CLASSES):
        d = Path(folder).expanduser() / name
        out += [(p, cls_i) for p in sorted(d.rglob("*")) if p.suffix.lower() in IMAGE_EXTS]
    return out


def evaluate(clf: MarkClassifier, items, group_views: bool = True) -> dict:
    """Per-brick classification (views `<brick>_f<k>` grouped like at inspection). Prints a confusion table."""
    from dlsj.inspect.core import group_by_brick, read_rgb

    rows = []
    by_label: dict[int, list[Path]] = {}
    for p, y in items:
        by_label.setdefault(y, []).append(p)
    for y, paths in sorted(by_label.items()):
        groups = group_by_brick(paths) if group_views else {p.stem: [p] for p in paths}
        for brick, ps in sorted(groups.items()):
            crops = [read_rgb(p) for p in ps]
            t0 = time.perf_counter()
            pred = clf.classify(crops)
            rows.append((y, brick, pred, (time.perf_counter() - t0) * 1e3))
    names = {0: "circle", 1: "cross"}
    table = {names[y]: {m: 0 for m in ("circle", "cross", "unknown")} for y in names}
    for y, _, pred, _ in rows:
        table[names[y]][pred.mark.value] += 1
    n = len(rows)
    correct = sum(pred.mark.value == names[y] for y, _, pred, _ in rows)
    unk = sum(pred.mark is MarkType.UNKNOWN for _, _, pred, _ in rows)
    lat = np.array([r[3] for r in rows]) if rows else np.zeros(1)
    print(f"stage 2 = {clf.name} on {clf.device}  bricks={n}  min_confidence={clf.min_confidence}")
    print(f"{'true':7s} {'circle':>7s} {'cross':>7s} {'unknown':>8s}")
    for k, v in table.items():
        print(f"{k:7s} {v['circle']:7d} {v['cross']:7d} {v['unknown']:8d}")
    print(f"correct {correct}/{n} ({100 * correct / max(n, 1):.1f}%), wrong {n - correct - unk}, unknown {unk}")
    print(f"classify() latency per brick ({len(rows and [1])} call each): median {np.median(lat):.2f} ms, "
          f"p90 {np.percentile(lat, 90):.2f} ms")
    return {"table": table, "correct": correct, "n": n, "unknown": unk, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval", help="classify a labelled crop folder (circle/, cross/)")
    e.add_argument("--crops", type=Path, required=True)
    e.add_argument("--classifier", default="shape", help='"shape", a shape_rules.json or a CNN classifier.json')
    e.add_argument("--device", default="NPU")
    e.add_argument("--min-confidence", type=float)
    e.add_argument("--no-group", action="store_true", help="one crop per decision (default: group <brick>_f<k>)")
    c = sub.add_parser("calibrate", help="fit the shape rules' Platt scaling on a labelled crop folder")
    c.add_argument("--crops", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True, help="shape_rules.json to write")
    c.add_argument("--min-confidence", type=float, default=0.7)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    items = labelled_crops(args.crops)
    if not items:
        raise SystemExit(f"no crops under {args.crops}/circle or {args.crops}/cross")
    if args.cmd == "eval":
        clf = load_classifier(args.classifier, device=args.device)
        if args.min_confidence is not None:
            clf.min_confidence = args.min_confidence
        evaluate(clf, items, group_views=not args.no_group)
        return
    from dlsj.inspect.core import read_rgb

    clf = ShapeRuleClassifier(min_confidence=args.min_confidence)
    xs, ys, missing = [], [], 0
    for p, y in items:
        f = clf.features(read_rgb(p))
        if not f.found:
            missing += 1
            continue
        xs.append(clf.logit(f))
        ys.append(y)
    clf.platt = fit_platt(xs, ys)
    args.out.write_text(json.dumps(clf.config(), indent=2) + "\n")
    print(f"fitted platt a={clf.platt[0]:.3f} b={clf.platt[1]:.3f} on {len(xs)} crops ({missing} without a mark) "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
