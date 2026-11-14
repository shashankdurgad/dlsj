"""Sweep aggregation: one decision from the per-frame results of a brick spun through N `wrist_roll` angles.

The mark may be on one face only, so it shows in a few frames of the sweep, and a glint can make one frame of a
plain brick look anomalous. The rule therefore weighs evidence instead of averaging it away. Strategies
(`AggregateConfig.strategy`, CLI `--aggregate`):

* `topk` (default, evidence-aware)
    stage 1: score = mean of the top-k frame scores (k = 3).
             MARKED  if score > band_high and at least `min_marked_frames` (2) frames are above band_high
                     (so one glare frame can't flip a plain brick);
             PLAIN   if at most `max_outlier_frames` (1) frame is above band_high and at least `min_clean_frac`
                     (75 %) of the frames are below band_low ("nearly all angles look clean");
             otherwise UNCERTAIN.
    stage 2: confidence-weighted vote over the marked frames: P(cross) = sum(w_i * p_i) / sum(w_i), with
             w_i = (score_i - threshold) / (band_high - threshold) clipped to [1, `max_mark_weight`], so frames
             where the mark is clearly visible count more. Argmax class; UNKNOWN when the classifier found a mark
             in less than `min_mark_found_frac` of the marked frames, or the winner's probability is below
             `min_mark_confidence`, or the margin P(win) - P(lose) is below `min_mark_margin`.
* `majority` (naive): each frame votes clean / marked / uncertain; the majority wins (tie / no majority ->
  UNCERTAIN). Stage 2: majority of the per-frame argmax votes. Fails on one-face marks (most frames are clean).
* `max` (naive): score = the single most anomalous frame, judged against the band; stage 2 = that frame's
  class. Fails on glare (one bright frame flips a plain brick).
* `median`: the classic 3-frame rule (median score, UNCERTAIN when frames disagree; mean P(cross)). On a sweep
  of a one-face mark the frames always "disagree", so it mostly says UNCERTAIN.

Every result carries `summary` (counts, top-k mean, class probabilities, which frame localised the mark) and the
outcome each other strategy would have given (`summary["alternatives"]`), so sweeps can be compared offline
(`python -m dlsj.loop.sweep_test eval`). Pure NumPy; no OpenCV / OpenVINO.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Sequence

import numpy as np

from dlsj.contracts import FrameInspection, InspectionResult, Label, MarkType, mark_outcome

STRATEGIES = ("topk", "majority", "max", "median")


@dataclass(frozen=True)
class AggregateConfig:
    strategy: str = "topk"
    k: int = 3  # topk: frames averaged
    min_marked_frames: int = 2  # topk: frames above band_high needed to call the brick marked
    max_outlier_frames: int = 1  # topk: frames above band_high a plain brick may still have (glare)
    min_clean_frac: float = 0.75  # topk: fraction of frames below band_low needed to call the brick plain
    min_mark_confidence: float = 0.65  # stage 2: probability of the winning class
    min_mark_margin: float = 0.3  # stage 2: P(winner) - P(runner-up)
    min_mark_found_frac: float = 0.5  # stage 2: marked frames in which the classifier found a mark
    max_mark_weight: float = 4.0  # topk: cap on one frame's vote weight

    def __post_init__(self):
        if self.strategy not in STRATEGIES:
            raise ValueError(f"unknown aggregation strategy {self.strategy!r}; choose from {STRATEGIES}")


@dataclass
class Aggregate:
    """What one strategy concludes from a sweep's frames."""

    strategy: str
    label: Label  # stage 1: PERFECT / DEFECTIVE / UNCERTAIN
    score: float
    mark: MarkType | None  # None = stage 2 did not run (no classifier, or not marked)
    mark_confidence: float
    best_index: int | None  # frame used for the region / heatmap
    summary: dict = field(default_factory=dict)

    def outcome(self) -> MarkType:
        """NONE / CIRCLE / CROSS / UNKNOWN, the same mapping as `contracts.mark_outcome`."""
        if self.label is Label.PERFECT:
            return MarkType.NONE
        if self.label is not Label.DEFECTIVE:
            return MarkType.UNKNOWN
        if self.mark is None:
            return MarkType.CROSS
        return self.mark


# ----------------------------------------------------------------------------- per frame
def frame_label(score: float, band_low: float, band_high: float) -> Label:
    if score > band_high:
        return Label.DEFECTIVE
    if score < band_low:
        return Label.PERFECT
    return Label.UNCERTAIN


def _counts(frames: Sequence[FrameInspection]) -> dict:
    labs = [f.label for f in frames]
    return {"n": len(frames), "n_marked": labs.count(Label.DEFECTIVE), "n_clean": labs.count(Label.PERFECT),
            "n_uncertain": labs.count(Label.UNCERTAIN)}


def _weight(score: float, threshold: float, band_high: float, cap: float) -> float:
    span = max(band_high - threshold, 1e-6)
    return float(np.clip((score - threshold) / span, 1.0, cap))


# ----------------------------------------------------------------------------- stage 1
def _stage1(strategy: str, frames, cfg: AggregateConfig, threshold: float, band_low: float, band_high: float
            ) -> tuple[Label, float, str]:
    s = np.array([f.score for f in frames], dtype=np.float64)
    c = _counts(frames)
    n = c["n"]
    if strategy == "topk":
        k = max(1, min(cfg.k, n))
        score = float(np.sort(s)[::-1][:k].mean())
        need = max(1, min(cfg.min_marked_frames, n))
        if score > band_high and c["n_marked"] >= need:
            return Label.DEFECTIVE, score, f"top-{k} mean {score:.4g} > {band_high:.4g} and {c['n_marked']}/{n} frames marked"
        if c["n_marked"] <= cfg.max_outlier_frames and c["n_clean"] >= cfg.min_clean_frac * n:
            why = f"{c['n_clean']}/{n} frames clean"
            if c["n_marked"]:
                why += f", {c['n_marked']} outlier frame(s) ignored"
            return Label.PERFECT, score, why
        return Label.UNCERTAIN, score, (f"{c['n_marked']} marked / {c['n_clean']} clean / {c['n_uncertain']} "
                                        f"uncertain of {n}, top-{k} mean {score:.4g}")
    if strategy == "majority":
        score = float(np.median(s))
        if c["n_marked"] * 2 > n:
            return Label.DEFECTIVE, score, f"majority marked ({c['n_marked']}/{n})"
        if c["n_clean"] * 2 > n:
            return Label.PERFECT, score, f"majority clean ({c['n_clean']}/{n})"
        return Label.UNCERTAIN, score, "no majority"
    if strategy == "max":
        score = float(s.max())
        return frame_label(score, band_low, band_high), score, f"max frame score {score:.4g}"
    # median: the classic rule (core.decide), kept dependency-free here
    score = float(np.median(s))
    if n > 1 and s.min() < band_low and s.max() > band_high:
        return Label.UNCERTAIN, score, "frames disagree"
    return frame_label(score, band_low, band_high), score, f"median score {score:.4g}"


# ----------------------------------------------------------------------------- stage 2
def _stage2(strategy: str, frames, label: Label, cfg: AggregateConfig, threshold: float, band_high: float
            ) -> tuple[MarkType | None, float, dict]:
    """(mark, confidence, info). None when stage 2 did not run on any frame (no classifier configured)."""
    marked = [f for f in frames if f.label is Label.DEFECTIVE]
    ran = [f for f in frames if f.mark is not None]
    if label is not Label.DEFECTIVE or not ran:
        return None, 0.0, {}
    pool = [f for f in marked if f.mark is not None] or ran
    valid = [f for f in pool if f.p_cross is not None and not math.isnan(f.p_cross)]
    votes = {"circle": sum(f.p_cross < 0.5 for f in valid), "cross": sum(f.p_cross >= 0.5 for f in valid),
             "none": len(pool) - len(valid)}
    info: dict = {"mark_votes": votes, "mark_frames": [f.index for f in pool]}
    if strategy == "max":
        best = max(frames, key=lambda f: f.score)
        if best.mark is None or best.p_cross is None:
            return MarkType.UNKNOWN, 0.0, info | {"mark_reason": "no mark found in the max frame"}
        p = float(best.p_cross)
        weights = None
    else:
        found = len(valid) / max(len(pool), 1)
        info["mark_found_frac"] = round(found, 3)
        if not valid or found < cfg.min_mark_found_frac:
            return MarkType.UNKNOWN, 0.0, info | {"mark_reason": f"mark found in {len(valid)}/{len(pool)} frames"}
        if strategy == "majority":
            if votes["cross"] == votes["circle"]:
                return MarkType.UNKNOWN, 0.5, info | {"mark_reason": "tied vote"}
            win = MarkType.CROSS if votes["cross"] > votes["circle"] else MarkType.CIRCLE
            conf = max(votes["cross"], votes["circle"]) / len(valid)
            info["mark_probs"] = {"circle": round(votes["circle"] / len(valid), 4),
                                  "cross": round(votes["cross"] / len(valid), 4)}
            return win, conf, info | {"mark_reason": f"{win.value} {max(votes['cross'], votes['circle'])}/{len(valid)} votes"}
        if strategy == "topk":
            weights = np.array([_weight(f.score, threshold, band_high, cfg.max_mark_weight) for f in valid])
        else:  # median: plain mean
            weights = np.ones(len(valid))
        p = float(np.sum(weights * np.array([f.p_cross for f in valid])) / np.sum(weights))
    probs = {"circle": round(1.0 - p, 4), "cross": round(p, 4)}
    win = MarkType.CROSS if p >= 0.5 else MarkType.CIRCLE
    conf, margin = max(p, 1.0 - p), abs(2.0 * p - 1.0)
    info |= {"mark_probs": probs, "mark_margin": round(margin, 4)}
    if weights is not None:
        info["mark_weights"] = {int(f.index): round(float(w), 3) for f, w in zip(valid, weights)}
    if conf < cfg.min_mark_confidence or margin < cfg.min_mark_margin:
        return MarkType.UNKNOWN, conf, info | {
            "mark_reason": f"{win.value}? P={conf:.2f} margin {margin:.2f} (need {cfg.min_mark_confidence:.2f} / "
                           f"{cfg.min_mark_margin:.2f})"}
    return win, conf, info | {"mark_reason": f"{win.value} P={conf:.2f} margin {margin:.2f}"}


# ----------------------------------------------------------------------------- public API
def aggregate(frames: Sequence[FrameInspection], threshold: float, band_low: float, band_high: float,
              cfg: AggregateConfig = AggregateConfig(), strategy: str | None = None) -> Aggregate:
    """Decide one brick from its per-frame results with `strategy` (default `cfg.strategy`)."""
    if not frames:
        raise ValueError("no frames to aggregate")
    strategy = strategy or cfg.strategy
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown aggregation strategy {strategy!r}; choose from {STRATEGIES}")
    label, score, why = _stage1(strategy, frames, cfg, threshold, band_low, band_high)
    mark, conf, info = _stage2(strategy, frames, label, cfg, threshold, band_high)
    s = np.array([f.score for f in frames], dtype=np.float64)
    c = _counts(frames)
    k = max(1, min(cfg.k, len(frames)))
    best = None
    if label is not Label.PERFECT:
        marked = [f for f in frames if f.label is Label.DEFECTIVE and f.region is not None]
        pool = marked or [f for f in frames if f.region is not None] or list(frames)
        best = max(pool, key=lambda f: f.score).index
    summary = {
        "strategy": strategy, "reason": why, **c,
        "topk": k, "topk_mean": round(float(np.sort(s)[::-1][:k].mean()), 5), "median": round(float(np.median(s)), 5),
        "max": round(float(s.max()), 5), "min": round(float(s.min()), 5),
        "stage1_probs": {"clean": round(c["n_clean"] / c["n"], 4), "marked": round(c["n_marked"] / c["n"], 4),
                         "uncertain": round(c["n_uncertain"] / c["n"], 4)},
        "best_index": best, **info,
    }
    return Aggregate(strategy, label, score, mark, float(conf), best, summary)


def aggregate_all(frames: Sequence[FrameInspection], threshold: float, band_low: float, band_high: float,
                  cfg: AggregateConfig = AggregateConfig(), strategies: Sequence[str] = STRATEGIES
                  ) -> dict[str, Aggregate]:
    return {s: aggregate(frames, threshold, band_low, band_high, cfg, s) for s in strategies}


def sweep_result(frames: Sequence[FrameInspection], threshold: float, band_low: float, band_high: float,
                 cfg: AggregateConfig = AggregateConfig(), strategy: str | None = None,
                 heatmaps: Sequence[np.ndarray | None] | None = None, latency_ms: float = 0.0, device: str = "",
                 mark_device: str = "", stage_latency_ms: dict | None = None) -> InspectionResult:
    """The `InspectionResult` of a sweep: the chosen strategy's decision, every frame, and the alternatives."""
    results = aggregate_all(frames, threshold, band_low, band_high, cfg)
    strategy = strategy or cfg.strategy
    agg = results[strategy]
    summary = dict(agg.summary)
    summary["alternatives"] = {s: a.outcome().value for s, a in results.items()}
    summary["config"] = asdict(replace(cfg, strategy=strategy))
    best = agg.best_index
    region = frames[best].region if best is not None else None
    heat = heatmaps[best if best is not None else int(np.argmax([f.score for f in frames]))] if heatmaps else None
    return InspectionResult(
        label=agg.label, score=agg.score, threshold=threshold, region=region, heatmap=heat,
        frame_scores=tuple(float(f.score) for f in frames), latency_ms=latency_ms, device=device,
        mark=agg.mark, mark_confidence=agg.mark_confidence, mark_device=mark_device if agg.mark is not None else "",
        stage_latency_ms=dict(stage_latency_ms or {}), frames=tuple(frames), aggregation=strategy, summary=summary)


def frames_record(frames: Sequence[FrameInspection]) -> list[dict]:
    """JSON-friendly per-frame list (for events, sweep.json and the status page)."""
    return [{"i": f.index, "angle": None if f.angle_deg is None else round(float(f.angle_deg), 2),
             "score": round(float(f.score), 5), "label": f.label.value,
             "region": None if f.region is None else [int(v) for v in f.region],
             "mark": None if f.mark is None else f.mark.value, "mark_confidence": round(float(f.mark_confidence), 4),
             "p_cross": None if f.p_cross is None else round(float(f.p_cross), 4),
             "latency_ms": round(float(f.latency_ms), 3),
             **({"pred_label": int(f.pred_label)} if f.pred_label is not None else {})} for f in frames]


def frames_from_record(rec: list[dict]) -> list[FrameInspection]:
    return [FrameInspection(int(d["i"]), d.get("angle"), float(d["score"]), Label(d["label"]),
                            tuple(d["region"]) if d.get("region") else None,
                            MarkType(d["mark"]) if d.get("mark") else None, float(d.get("mark_confidence", 0.0)),
                            d.get("p_cross"), float(d.get("latency_ms", 0.0)), d.get("pred_label")) for d in rec]


# ----------------------------------------------------------------------------- inspectors: sweep support
class SweepSession:
    """Streams a sweep through an inspector: `add(frame, angle)` queues the frame on one background worker
    (stage 1 + region + stage 2 for that frame) and returns at once, so inference of frame i overlaps the wrist
    moving to stop i+1; `result()` waits for the last frame and aggregates. OpenVINO releases the GIL during
    inference, so the control loop keeps its tick rate.
    """

    def __init__(self, inspector, strategy: str | None = None):
        from concurrent.futures import ThreadPoolExecutor

        self.insp, self.strategy = inspector, strategy
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sweep-infer")
        self.futures: list = []
        self.t_start = time.perf_counter()

    def add(self, frame: np.ndarray, angle: float | None = None) -> None:
        self.futures.append(self.pool.submit(self.insp.inspect_frame, frame, len(self.futures), angle))

    def cancel(self) -> None:
        for f in self.futures:
            f.cancel()
        self.pool.shutdown(wait=True)

    def result(self) -> InspectionResult:
        t0 = time.perf_counter()
        try:
            out = [f.result() for f in self.futures]
        finally:
            self.pool.shutdown(wait=True)
        if not out:
            raise ValueError("sweep produced no frames")
        tail = (time.perf_counter() - t0) * 1e3
        self.records, self.heats = [o[0] for o in out], [o[1] for o in out]
        return self.insp.sweep_from_frames(self.records, self.heats, self.strategy, extra_latency={"tail": tail})


class SweepInspectorMixin:
    """`inspect_sweep` / `start_sweep` for an inspector that implements `inspect_frame(frame, index, angle)`
    -> (FrameInspection, heatmap | None) and `sweep_meta()` -> {threshold, band_low, band_high, device, mark_device}.
    `agg` (AggregateConfig) is the default strategy and its knobs."""

    agg: AggregateConfig = AggregateConfig()

    def inspect_frames(self, frames: Sequence[np.ndarray], angles: Sequence[float | None] | None = None):
        angles = list(angles) if angles is not None else [None] * len(frames)
        return [self.inspect_frame(f, i, a) for i, (f, a) in enumerate(zip(frames, angles))]

    def inspect_sweep(self, frames: Sequence[np.ndarray], angles: Sequence[float | None] | None = None,
                      strategy: str | None = None) -> InspectionResult:
        out = self.inspect_frames(frames, angles)
        return self.sweep_from_frames([o[0] for o in out], [o[1] for o in out], strategy)

    def start_sweep(self, strategy: str | None = None) -> SweepSession:
        return SweepSession(self, strategy)

    def sweep_from_frames(self, frames: Sequence[FrameInspection], heats, strategy: str | None = None,
                          extra_latency: dict | None = None) -> InspectionResult:
        meta = self.sweep_meta()
        s1 = sum(f.latency_ms for f in frames)
        lat = {"stage1+2": s1, **(extra_latency or {})}
        return sweep_result(frames, meta["threshold"], meta["band_low"], meta["band_high"], self.agg,
                            strategy or self.agg.strategy, heats, latency_ms=s1 + lat.get("tail", 0.0),
                            device=meta["device"], mark_device=meta["mark_device"], stage_latency_ms=lat)


__all__ = ["SweepSession", "SweepInspectorMixin", "STRATEGIES", "AggregateConfig", "Aggregate", "aggregate", "aggregate_all", "sweep_result", "frame_label",
           "frames_record", "frames_from_record", "mark_outcome"]
