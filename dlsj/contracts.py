"""Shared contracts between perception, the closed-loop controller and policy back-ends.

Kept dependency-free (numpy only) so every module and test can import it. The loop
(`dlsj.loop`) depends only on these types, never on Anomalib, OpenVINO or LeRobot directly, so each
back-end can be swapped or faked in tests.

Task (see docs/PLAYBOOK.md): pick a brick, present it to the `inspect` camera, then two perception stages:

* stage 1 (Anomalib, trained on plain bricks): plain vs marked, plus the `region` of the mark;
* stage 2 (mark classifier on the region crop, only when stage 1 says marked): circle vs cross.

The reasoner maps plain -> good plate, circle -> minor-defect plate, cross -> defective plate; anything uncertain
-> re-inspect, then the defective plate (safe default). `mark_outcome` is that mapping without the retry logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

import numpy as np


class Label(str, Enum):
    """Stage-1 label (Anomalib) and, with MINOR, the final grade the reasoner acts on.

    Stage 1 only ever emits PERFECT / DEFECTIVE / UNCERTAIN, where DEFECTIVE means "marked" (anomalous): the
    mark type comes from stage 2 (`InspectionResult.mark`). The reasoner's decision label is PERFECT (plain),
    MINOR (circle) or DEFECTIVE (cross, or the uncertainty fallback).
    """

    PERFECT = "perfect"  # plain brick
    DEFECTIVE = "defective"  # stage 1: marked; final: marked with a cross (or the safe fallback)
    UNCERTAIN = "uncertain"  # score inside the uncertainty band, or the frames disagree
    MINOR = "minor"  # final grade only: marked with a circle (minor defect)


class MarkType(str, Enum):
    """Stage-2 result: what the stage-1 region contains."""

    NONE = "none"  # plain brick (stage 1 PERFECT; stage 2 did not run)
    CIRCLE = "circle"  # minor defect
    CROSS = "cross"  # defective
    UNKNOWN = "unknown"  # stage 2 ran but is unsure (low confidence, no mark found, or the frames disagree)


@dataclass(frozen=True)
class InspectionResult:
    """Structured defect signal: the brief asks for a label, a score and a localized region."""

    label: Label
    score: float  # aggregated anomaly score (e.g. median over frames); higher = more anomalous
    threshold: float  # decision threshold the label was derived from
    region: tuple[int, int, int, int] | None = None  # (x0, y0, x1, y1) of the anomaly in the inspect frame
    heatmap: np.ndarray | None = None  # HxW float32 anomaly map in [0, 1], aligned to the inspect frame
    frame_scores: tuple[float, ...] = ()  # per-frame scores that were aggregated
    latency_ms: float = 0.0  # inference time for the whole inspection (all frames, both stages)
    device: str = ""  # OpenVINO device that ran the model, e.g. "NPU"
    # Stage 2 (mark classifier). `mark` None = stage 2 did not run on a marked brick (no classifier configured):
    # the legacy two-plate behaviour, a marked brick counts as a cross.
    mark: MarkType | None = None
    mark_confidence: float = 0.0  # probability of `mark` (for UNKNOWN: of the more likely class)
    mark_device: str = ""  # "SHAPE-CPU" for the shape rules, else the OpenVINO device of the CNN
    stage_latency_ms: dict[str, float] = field(default_factory=dict)  # {"stage1": ms, "stage2": ms}


def mark_outcome(result: InspectionResult, min_mark_confidence: float = 0.0) -> MarkType:
    """Three-class outcome of one inspection: NONE (plain), CIRCLE, CROSS, or UNKNOWN (some stage is unsure).

    Stage 1 PERFECT -> NONE; UNCERTAIN -> UNKNOWN; DEFECTIVE -> the stage-2 mark, UNKNOWN when stage 2 is unsure
    or below `min_mark_confidence`, CROSS when stage 2 did not run (legacy: marked = defective).
    """
    if result.label is Label.PERFECT:
        return MarkType.NONE
    if result.label is not Label.DEFECTIVE:
        return MarkType.UNKNOWN
    if result.mark is None:
        return MarkType.CROSS
    if result.mark in (MarkType.CIRCLE, MarkType.CROSS) and result.mark_confidence >= min_mark_confidence:
        return result.mark
    return MarkType.UNKNOWN


class Inspector(Protocol):
    """Anything that turns `inspect`-camera frames (HxWx3 uint8 RGB) into an InspectionResult."""

    def inspect(self, frames: list[np.ndarray]) -> InspectionResult: ...


@dataclass(frozen=True)
class Observation:
    """One robot observation in the dataset's units (LeRobot SO-101 follower conventions)."""

    state: np.ndarray  # (6,) float32 joint positions: shoulder_pan .. wrist_roll in degrees, gripper 0-100
    images: dict[str, np.ndarray] = field(default_factory=dict)  # "wrist" / "top" / "inspect" -> HxWx3 uint8 RGB
    timestamp: float = 0.0


class Policy(Protocol):
    """A language-conditioned action-chunk policy (pi0.5 via OpenVINO, LeRobot, or a fake)."""

    def reset(self) -> None: ...

    def predict_chunk(self, obs: Observation, instruction: str) -> np.ndarray:
        """Return an (n, 6) float32 chunk of joint targets in the same units as `Observation.state`."""
        ...
