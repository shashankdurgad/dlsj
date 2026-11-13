"""REASON: the three-plate decision tree + uncertainty handling. Pure functions, no I/O.

    stage 1 (Anomalib)      stage 2 (mark classifier)      ->  plate / action
    PERFECT (plain)         -                                  good plate       (PLACE_BLUE skill id)
    DEFECTIVE (marked)      CIRCLE, conf >= threshold          minor plate      (PLACE_MINOR)
    DEFECTIVE (marked)      CROSS,  conf >= threshold          defective plate  (PLACE_PINK)
    DEFECTIVE (marked)      not run (mark None)                defective plate  (legacy two-plate behaviour)
    DEFECTIVE (marked)      UNKNOWN / below threshold          uncertain
    UNCERTAIN               -                                  uncertain

    uncertain -> re-inspect (RESHOW skill if enabled, else re-grab frames); after `max_uncertain` uncertain
                 inspections -> the defective plate (safest), logged as forced.

Each stage has its own uncertainty threshold: stage 1 its band (from the manifest; `extra_margin` widens it here),
stage 2 its classifier's `min_confidence` (UNKNOWN) plus `ReasonConfig.min_mark_confidence` on top.
The decision label is PERFECT / MINOR / DEFECTIVE (Label.MINOR exists only here, stage 1 never emits it).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from dlsj.contracts import InspectionResult, Label, MarkType, mark_outcome
from dlsj.loop.skills import PLACE_BLUE, PLACE_MINOR, PLACE_PINK, RESHOW


@dataclass(frozen=True)
class ReasonConfig:
    max_uncertain: int = 2  # uncertain inspections before falling back to DEFECTIVE
    reshow_available: bool = False  # RESHOW skill trained and enabled
    extra_margin: float = 0.0  # stage 1: widen the inspector's uncertainty band by this much around the threshold
    min_mark_confidence: float = 0.0  # stage 2: below this a CIRCLE/CROSS counts as uncertain
    fallback_label: Label = Label.DEFECTIVE


@dataclass(frozen=True)
class Decision:
    kind: Literal["place", "reinspect"]
    label: Label  # the label acted on (after any fallback): PERFECT / MINOR / DEFECTIVE, or UNCERTAIN to re-inspect
    skill: str | None  # skill to run next; None for a plain re-grab
    reason: str
    params: dict = field(default_factory=dict)
    forced: bool = False  # True when the label came from the uncertainty fallback
    stage: str = ""  # which stage was uncertain ("stage1" / "stage2"), "" otherwise


def effective_label(result: InspectionResult, cfg: ReasonConfig) -> Label:
    """Stage-1 label after the optional extra margin."""
    if result.label is Label.UNCERTAIN:
        return Label.UNCERTAIN
    if cfg.extra_margin > 0 and abs(result.score - result.threshold) < cfg.extra_margin:
        return Label.UNCERTAIN
    return result.label


def grade(result: InspectionResult, cfg: ReasonConfig) -> tuple[Label, str]:
    """(PERFECT | MINOR | DEFECTIVE | UNCERTAIN, uncertain stage or "")."""
    if effective_label(result, cfg) is Label.UNCERTAIN:
        return Label.UNCERTAIN, "stage1"
    mark = mark_outcome(result, cfg.min_mark_confidence)
    return {MarkType.NONE: (Label.PERFECT, ""), MarkType.CIRCLE: (Label.MINOR, ""),
            MarkType.CROSS: (Label.DEFECTIVE, ""), MarkType.UNKNOWN: (Label.UNCERTAIN, "stage2")}[mark]


def _describe(result: InspectionResult) -> str:
    s = f"score {result.score:.3f} vs threshold {result.threshold:.3f}"
    if result.region is not None:
        s += f", region {tuple(int(v) for v in result.region)}"
    if result.mark is not None:
        s += f", mark {result.mark.value} {result.mark_confidence:.2f}"
    return s


def decide(result: InspectionResult, attempt: int, cfg: ReasonConfig = ReasonConfig(),
           perfect_skill: str = PLACE_BLUE, defective_skill: str = PLACE_PINK,
           perfect_params: dict | None = None, defective_params: dict | None = None,
           minor_skill: str = PLACE_MINOR, minor_params: dict | None = None) -> Decision:
    """Next step after inspection number `attempt` (1-based) of the current brick."""
    rules = {
        Label.PERFECT: (perfect_skill, perfect_params or {}, "plain brick"),
        Label.MINOR: (minor_skill, minor_params or {}, "circle: minor defect"),
        Label.DEFECTIVE: (defective_skill, defective_params or {},
                          "cross detected" if result.mark is not None else "marked (cross / no stage 2)"),
    }
    label, stage = grade(result, cfg)
    if label is not Label.UNCERTAIN:
        skill, params, why = rules[label]
        return Decision("place", label, skill, f"{label.value.upper()} ({why}; {_describe(result)}) -> {skill}",
                        params)

    which = "stage 1 (Anomalib)" if stage == "stage1" else "stage 2 (mark type)"
    if attempt < cfg.max_uncertain:
        how = RESHOW if cfg.reshow_available else None
        verb = "turn and re-show the brick" if how else "re-grab the inspect frames"
        return Decision("reinspect", Label.UNCERTAIN, how,
                        f"UNCERTAIN at {which} ({_describe(result)}; attempt {attempt}/{cfg.max_uncertain}) -> {verb}",
                        stage=stage)

    fb = cfg.fallback_label
    skill, params, _ = rules[fb]
    return Decision("place", fb, skill,
                    f"UNCERTAIN at {which} after {attempt} inspections ({_describe(result)}) -> treat as "
                    f"{fb.value.upper()} (safe default) -> {skill}", params, forced=True, stage=stage)


def inspection_record(result: InspectionResult) -> dict:
    """JSON-friendly view of an InspectionResult for the event log (heatmap summarized, not dumped)."""
    rec = {"label": result.label.value, "score": round(float(result.score), 4),
           "threshold": round(float(result.threshold), 4),
           "region": None if result.region is None else [int(v) for v in result.region],
           "frame_scores": [round(float(s), 4) for s in result.frame_scores],
           "latency_ms": round(float(result.latency_ms), 3), "device": result.device,
           "mark": None if result.mark is None else result.mark.value,
           "mark_confidence": round(float(result.mark_confidence), 4), "mark_device": result.mark_device,
           "stage_latency_ms": {k: round(float(v), 3) for k, v in result.stage_latency_ms.items()},
           "outcome": mark_outcome(result).value}
    if result.heatmap is not None:
        rec["heatmap_shape"] = list(result.heatmap.shape)
        rec["heatmap_max"] = round(float(result.heatmap.max()), 4)
    return rec
