"""REASON: rule table + uncertainty handling. Pure functions, no I/O.

    PERFECT   (plain brick)  -> the task's "perfect" skill (phase 1: PLACE_BLUE)
    DEFECTIVE (cross)        -> the task's "defective" skill (PLACE_PINK)
    UNCERTAIN                -> re-inspect (RESHOW skill if enabled, else re-grab frames), and after
                                `max_uncertain` uncertain inspections treat it as DEFECTIVE (safe default)

`ReasonConfig.extra_margin` optionally widens the inspector's own uncertainty band: a PERFECT/DEFECTIVE
whose score lies within that margin of the threshold is treated as UNCERTAIN too (0 = trust the inspector).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from dlsj.contracts import InspectionResult, Label
from dlsj.loop.skills import PLACE_BLUE, PLACE_PINK, RESHOW


@dataclass(frozen=True)
class ReasonConfig:
    max_uncertain: int = 2  # uncertain inspections before falling back to DEFECTIVE
    reshow_available: bool = False  # RESHOW skill trained and enabled
    extra_margin: float = 0.0
    fallback_label: Label = Label.DEFECTIVE


@dataclass(frozen=True)
class Decision:
    kind: Literal["place", "reinspect"]
    label: Label  # the label acted on (after any fallback)
    skill: str | None  # skill to run next; None for a plain re-grab
    reason: str
    params: dict = field(default_factory=dict)
    forced: bool = False  # True when the label came from the uncertainty fallback


def effective_label(result: InspectionResult, cfg: ReasonConfig) -> Label:
    if result.label is Label.UNCERTAIN:
        return Label.UNCERTAIN
    if cfg.extra_margin > 0 and abs(result.score - result.threshold) < cfg.extra_margin:
        return Label.UNCERTAIN
    return result.label


def _describe(result: InspectionResult) -> str:
    s = f"score {result.score:.3f} vs threshold {result.threshold:.3f}"
    if result.region is not None:
        s += f", region {tuple(int(v) for v in result.region)}"
    return s


def decide(result: InspectionResult, attempt: int, cfg: ReasonConfig = ReasonConfig(),
           perfect_skill: str = PLACE_BLUE, defective_skill: str = PLACE_PINK,
           perfect_params: dict | None = None, defective_params: dict | None = None) -> Decision:
    """Next step after inspection number `attempt` (1-based) of the current brick."""
    rules = {
        Label.PERFECT: (perfect_skill, perfect_params or {}, "plain brick"),
        Label.DEFECTIVE: (defective_skill, defective_params or {}, "cross detected"),
    }
    label = effective_label(result, cfg)
    if label is not Label.UNCERTAIN:
        skill, params, why = rules[label]
        return Decision("place", label, skill, f"{label.value.upper()} ({why}; {_describe(result)}) -> {skill}",
                        params)

    if attempt < cfg.max_uncertain:
        how = RESHOW if cfg.reshow_available else None
        verb = "turn and re-show the brick" if how else "re-grab the inspect frames"
        return Decision("reinspect", Label.UNCERTAIN, how,
                        f"UNCERTAIN ({_describe(result)}; attempt {attempt}/{cfg.max_uncertain}) -> {verb}")

    fb = cfg.fallback_label
    skill, params, _ = rules[fb]
    return Decision("place", fb, skill,
                    f"UNCERTAIN after {attempt} inspections ({_describe(result)}) -> treat as {fb.value.upper()} "
                    f"(safe default) -> {skill}", params, forced=True)


def inspection_record(result: InspectionResult) -> dict:
    """JSON-friendly view of an InspectionResult for the event log (heatmap summarized, not dumped)."""
    rec = {"label": result.label.value, "score": round(float(result.score), 4),
           "threshold": round(float(result.threshold), 4),
           "region": None if result.region is None else [int(v) for v in result.region],
           "frame_scores": [round(float(s), 4) for s in result.frame_scores],
           "latency_ms": round(float(result.latency_ms), 3), "device": result.device}
    if result.heatmap is not None:
        rec["heatmap_shape"] = list(result.heatmap.shape)
        rec["heatmap_max"] = round(float(result.heatmap.max()), 4)
    return rec
