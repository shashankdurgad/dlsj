"""Shared contracts between perception, the closed-loop controller and policy back-ends.

Kept dependency-free (numpy only) so every module and test can import it. The loop
(`dlsj.loop`) depends only on these types, never on Anomalib, OpenVINO or LeRobot directly, so each
back-end can be swapped or faked in tests.

Task (see docs/PLAYBOOK.md): pick a brick, present it to the `inspect` camera, Anomalib decides
plain ("perfect") vs cross-marked ("defective"), then pi0.5 places it on the blue or pink plate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

import numpy as np


class Label(str, Enum):
    PERFECT = "perfect"  # plain brick
    DEFECTIVE = "defective"  # brick marked with a cross
    UNCERTAIN = "uncertain"  # score inside the uncertainty band, or the frames disagree


@dataclass(frozen=True)
class InspectionResult:
    """Structured defect signal: the brief asks for a label, a score and a localized region."""

    label: Label
    score: float  # aggregated anomaly score (e.g. median over frames); higher = more anomalous
    threshold: float  # decision threshold the label was derived from
    region: tuple[int, int, int, int] | None = None  # (x0, y0, x1, y1) of the anomaly in the inspect frame
    heatmap: np.ndarray | None = None  # HxW float32 anomaly map in [0, 1], aligned to the inspect frame
    frame_scores: tuple[float, ...] = ()  # per-frame scores that were aggregated
    latency_ms: float = 0.0  # inference time for the whole inspection (all frames)
    device: str = ""  # OpenVINO device that ran the model, e.g. "NPU"


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
