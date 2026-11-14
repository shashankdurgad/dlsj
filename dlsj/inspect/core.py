"""Hardware-free logic of the `inspect` defect detector: manifest, ROI crop, aggregation, uncertainty band,
region extraction, heatmap and threshold calibration.

Only numpy + OpenCV, so tests and the robot loop can use it without OpenVINO or Anomalib.

Decision rule (PLAYBOOK §8 DETECT/REASON):
* per-frame anomaly scores are aggregated with the median;
* score < band_low → PERFECT, score > band_high → DEFECTIVE, otherwise UNCERTAIN (band_low <= threshold <= band_high);
* also UNCERTAIN when the frames clearly disagree (one frame below band_low and another above band_high).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from dlsj.contracts import InspectionResult, Label, MarkType

Box = tuple[int, int, int, int]  # (x0, y0, x1, y1), x1/y1 exclusive

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


# ----------------------------------------------------------------------------- manifest
@dataclass
class Manifest:
    """What `train.py` writes next to the exported model and `detector.py` reads back."""

    model: str  # OpenVINO .xml, relative to the manifest's folder
    model_name: str  # "Padim" / "Patchcore"
    input_size: tuple[int, int]  # (H, W) the model was exported with (static)
    crop: Box | None  # ROI in inspect-frame pixels; None = whole frame
    threshold: float  # on the raw score the exported model outputs
    band_low: float  # below: PERFECT
    band_high: float  # above: DEFECTIVE; in between: UNCERTAIN
    map_min: float = 0.0  # typical anomaly-map value on good bricks; heatmap 0 here, 0.5 at the threshold
    pixel_threshold: float | None = None  # region threshold on the anomaly map (defaults to `threshold`)
    trained_at: str = ""
    backbone: str = ""
    calibration: dict = field(default_factory=dict)  # score stats and the method used
    # Stage-2 mark classifier: {"backend": "shape"} (default when empty), {"backend": "none"}, or
    # {"backend": "cnn", "manifest": "../mark_cnn/classifier.json", "min_confidence": 0.7} (path relative to here).
    stage2: dict = field(default_factory=dict)
    # Which IR this is (see `model_io`): "u8_nhwc" = ours (pre-processing folded in by train.py), "f32_nchw" = a
    # standard Anomalib export (u8 -> f32 /255 folded in at load time). `score_space`: "raw" scores (ours, or an
    # export cut before its normaliser) or the export's "normalized" [0, 1] scores. `normalisation`: the export's
    # {"score": [a, b], "map": [a, b]} with normalised = clamp(raw * a + b, 0, 1).
    input_format: str = "u8_nhwc"
    score_space: str = "raw"
    source: str = "dlsj"
    normalisation: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "Manifest":
        d = json.loads(Path(path).read_text())
        d["input_size"] = tuple(d["input_size"])
        d["crop"] = tuple(d["crop"]) if d.get("crop") else None
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")


# ----------------------------------------------------------------------------- pre-processing
def parse_box(text: str | None) -> Box | None:
    if not text:
        return None
    x0, y0, x1, y1 = (int(v) for v in text.split(","))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"empty box {text!r}")
    return x0, y0, x1, y1


def crop_roi(frame: np.ndarray, roi: Box | None) -> np.ndarray:
    """Crop `roi` out of an HxWx3 frame, clamped to the frame (a view, no copy)."""
    if roi is None:
        return frame
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = max(0, roi[0]), max(0, roi[1]), min(w, roi[2]), min(h, roi[3])
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"ROI {roi} is outside the {w}x{h} frame")
    return frame[y0:y1, x0:x1]


def prepare(frame: np.ndarray, roi: Box | None, input_hw: tuple[int, int]) -> np.ndarray:
    """RGB uint8 frame → RGB uint8 crop resized to the model input (HxWx3). Normalisation happens in-graph."""
    crop = crop_roi(frame, roi)
    if crop.shape[:2] != tuple(input_hw):
        crop = cv2.resize(crop, (input_hw[1], input_hw[0]), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(crop)


# ----------------------------------------------------------------------------- decision
def aggregate(frame_scores) -> float:
    """Median over frames (robust to one bad frame: motion blur, a reflection)."""
    s = np.asarray(frame_scores, dtype=np.float64)
    if s.size == 0:
        raise ValueError("no frame scores")
    return float(np.median(s))


def frames_disagree(frame_scores, band_low: float, band_high: float) -> bool:
    """True when at least one frame is clearly perfect and another clearly defective."""
    s = np.asarray(frame_scores, dtype=np.float64)
    return bool(s.size > 1 and s.min() < band_low and s.max() > band_high)


def decide(score: float, frame_scores, band_low: float, band_high: float) -> Label:
    if frames_disagree(frame_scores, band_low, band_high):
        return Label.UNCERTAIN
    if score > band_high:
        return Label.DEFECTIVE
    if score < band_low:
        return Label.PERFECT
    return Label.UNCERTAIN


# ----------------------------------------------------------------------------- localisation
def region_from_map(
    amap: np.ndarray, pixel_threshold: float, roi: Box | None, crop_hw: tuple[int, int]
) -> Box | None:
    """Bounding box (inspect-frame pixels) of the anomaly-map component holding the map's peak.

    Thresholds the low-resolution map and only scales the box: no full-resolution upsampling.
    If no pixel passes `pixel_threshold`, falls back to the pixels within 50 % of the peak's excess over
    the map median, so an UNCERTAIN result still points at the most suspicious spot.
    """
    amap = np.asarray(amap, dtype=np.float32)
    mh, mw = amap.shape
    mask = amap > pixel_threshold
    if not mask.any():
        med, peak = float(np.median(amap)), float(amap.max())
        if peak <= med:
            return None
        mask = amap >= med + 0.5 * (peak - med)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    py, px = np.unravel_index(int(np.argmax(np.where(mask, amap, -np.inf))), amap.shape)
    k = labels[py, px]
    x, y, w, h = (int(v) for v in stats[k, :4])
    ch, cw = crop_hw
    sx, sy = cw / mw, ch / mh
    ox, oy = (roi[0], roi[1]) if roi is not None else (0, 0)
    return (
        ox + int(np.floor(x * sx)),
        oy + int(np.floor(y * sy)),
        ox + int(np.ceil((x + w) * sx)),
        oy + int(np.ceil((y + h) * sy)),
    )


def to_heatmap(
    amap: np.ndarray, map_min: float, threshold: float, frame_hw: tuple[int, int], roi: Box | None
) -> np.ndarray:
    """Anomaly map → HxW float32 in [0, 1] aligned to the inspect frame (0 outside the ROI).

    Linear scale with `map_min` → 0 and `threshold` → 0.5.
    """
    span = max(threshold - map_min, 1e-6)
    norm = np.clip((np.asarray(amap, np.float32) - map_min) * (0.5 / span), 0.0, 1.0)
    fh, fw = frame_hw
    x0, y0, x1, y1 = roi if roi is not None else (0, 0, fw, fh)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(fw, x1), min(fh, y1)
    out = np.zeros((fh, fw), np.float32)
    out[y0:y1, x0:x1] = cv2.resize(norm, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    return out


def build_result(
    frame_scores,
    maps,
    manifest: Manifest,
    frame_hw: tuple[int, int],
    latency_ms: float = 0.0,
    device: str = "",
    with_heatmap: bool = True,
) -> InspectionResult:
    """Aggregate per-frame scores/maps into the shared `InspectionResult`."""
    scores = tuple(float(s) for s in frame_scores)
    score = aggregate(scores)
    label = decide(score, scores, manifest.band_low, manifest.band_high)
    # Localise on the frame whose score is the median (the frame the decision is based on).
    mid = int(np.argsort(scores)[(len(scores) - 1) // 2])
    amap = maps[mid]
    roi = manifest.crop
    if roi is not None:
        fh, fw = frame_hw
        roi = (max(0, roi[0]), max(0, roi[1]), min(fw, roi[2]), min(fh, roi[3]))
    crop_hw = (roi[3] - roi[1], roi[2] - roi[0]) if roi is not None else tuple(frame_hw)
    region = None
    if label is not Label.PERFECT:
        pthr = manifest.pixel_threshold if manifest.pixel_threshold is not None else manifest.threshold
        region = region_from_map(amap, pthr, roi, crop_hw)
    heat = to_heatmap(amap, manifest.map_min, manifest.threshold, frame_hw, roi) if with_heatmap else None
    return InspectionResult(
        label=label,
        score=score,
        threshold=manifest.threshold,
        region=region,
        heatmap=heat,
        frame_scores=scores,
        latency_ms=latency_ms,
        device=device,
    )


# ----------------------------------------------------------------------------- calibration
def calibrate(good, bad=None, band_frac: float = 0.25, sigma: float = 3.0) -> dict:
    """Threshold and uncertainty band from held-out scores (higher = more anomalous).

    * good + bad, separable (max good < min bad): threshold in the middle of the gap; the band covers the
      middle half of the gap by default (`band_frac` = 0.25 of the gap kept clear on each side).
    * good + bad, overlapping: threshold maximising balanced accuracy; the band spans the overlap
      [min bad, max good], so every sample the threshold can't separate comes out UNCERTAIN.
    * good only: threshold = max(max good, mean + `sigma`·std); band = ±1 std around it.
    """
    g = np.sort(np.asarray(good, dtype=np.float64))
    if g.size == 0:
        raise ValueError("need held-out good scores to calibrate")
    b = np.sort(np.asarray(bad, dtype=np.float64)) if bad is not None and len(bad) else np.empty(0)
    stats = {
        "n_good": int(g.size),
        "n_bad": int(b.size),
        "good_mean": float(g.mean()),
        "good_std": float(g.std()),
        "good_max": float(g.max()),
    }
    if b.size:
        stats.update(bad_min=float(b.min()), bad_median=float(np.median(b)))
    if b.size == 0:
        std = float(g.std()) or 1e-3 * max(abs(float(g.max())), 1.0)
        thr = max(float(g.max()), float(g.mean()) + sigma * std)
        lo, hi, method = thr - std, thr + std, "good-only"
    elif g.max() < b.min():
        gap = float(b.min() - g.max())
        thr = float(g.max()) + gap / 2
        lo, hi, method = float(g.max()) + band_frac * gap, float(b.min()) - band_frac * gap, "separable-gap"
    else:
        cands = np.unique(np.concatenate([g, b]))
        bal = [0.5 * ((g <= t).mean() + (b > t).mean()) for t in cands]
        thr = float(cands[int(np.argmax(bal))])
        lo, hi, method = min(float(b.min()), thr), max(float(g.max()), thr), "overlap-balanced-acc"
    stats["method"] = method
    return {"threshold": thr, "band_low": lo, "band_high": hi, "stats": stats}


# ----------------------------------------------------------------------------- datasets
_VIEW_RE = re.compile(r"^(?P<brick>.+)_f\d+$")


def list_images(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def group_by_brick(paths) -> dict[str, list[Path]]:
    """`<brick>_f<k>.png` files are views of one brick; anything else is its own group."""
    groups: dict[str, list[Path]] = {}
    for p in paths:
        m = _VIEW_RE.match(Path(p).stem)
        groups.setdefault(m["brick"] if m else Path(p).stem, []).append(Path(p))
    return groups


def read_rgb(path: str | Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def write_rgb(path: str | Path, rgb: np.ndarray) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def overlay(frame_rgb: np.ndarray, result: InspectionResult, alpha: float = 0.45) -> np.ndarray:
    """Demo view: heatmap blended over the frame, region box and label (RGB in, RGB out)."""
    out = frame_rgb.copy()
    if result.heatmap is not None:
        heat = cv2.applyColorMap((result.heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        m = result.heatmap[..., None] > 0.02
        out = np.where(m, (out * (1 - alpha) + heat * alpha).astype(np.uint8), out)
    color = {Label.PERFECT: (0, 200, 0), Label.DEFECTIVE: (230, 0, 0), Label.UNCERTAIN: (255, 170, 0),
             Label.MINOR: (240, 200, 0)}
    col = color[result.label]
    if result.label is Label.DEFECTIVE and result.mark is not None:
        col = {MarkType.CIRCLE: color[Label.MINOR], MarkType.CROSS: color[Label.DEFECTIVE]}.get(
            result.mark, color[Label.UNCERTAIN])
    if result.region is not None:
        x0, y0, x1, y1 = result.region
        cv2.rectangle(out, (x0, y0), (x1, y1), col, 2)
    txt = f"{result.label.value} {result.score:.3g} / thr {result.threshold:.3g}"
    if result.mark is not None:
        txt += f"  {result.mark.value} {result.mark_confidence:.2f}"
    if result.latency_ms:
        txt += f"  {result.latency_ms:.1f} ms {result.device}"
    cv2.putText(out, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
    return out
