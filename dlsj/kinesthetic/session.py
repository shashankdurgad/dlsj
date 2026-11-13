"""Hardware-free logic for kinesthetic recording on the SO-101.

The recording workflow (two-pedal go/reject keys, a review phase that holds the arm after a take,
replaying the unsaved take before deciding, a saved start pose) is adapted from Flash ALOHA
(https://github.com/JohnYao93/flash-aloha, MIT License, Copyright (c) 2026 John Yao). See
THIRD_PARTY_NOTICES.md. The implementation here is a re-write for Feetech STS3215 servos and LeRobot.

This module is kept separate from the serial bus and cameras so it can be unit-tested:

* `KeyMap` / `SessionMachine`: which key does what in which phase.
* `ActionLookahead`: action[t] = measured_position[t + k], which compensates position-controller lag.
* `GripperCommand`: open/close/nudge target for a torque-on gripper driven by keys or a pedal.
* `approach_waypoints` / `tracking_report`: move-to-first-pose and replay tracking statistics.
"""

from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np


class Phase(Enum):
    IDLE = auto()  # between takes: arm held (or released with h), operator resets the scene
    RECORDING = auto()  # arm free, frames are written
    REVIEW = auto()  # take stopped, arm held at its last pose; accept, fail, replay or reject


class Command(Enum):
    START = auto()
    STOP = auto()
    SAVE_SUCCESS = auto()
    SAVE_FAILURE = auto()
    DISCARD = auto()
    REPLAY = auto()
    GRIPPER_TOGGLE = auto()
    GRIPPER_OPEN_MORE = auto()
    GRIPPER_CLOSE_MORE = auto()
    TOGGLE_HOLD = auto()
    SAVE_START_POSE = auto()
    QUIT = auto()


@dataclass(frozen=True)
class KeyMap:
    """Canonical key names from `lerobot.utils.keyboard_input.create_key_listener`.

    A two-pedal foot switch maps to `go` (left pedal, types '1') and `reject` (right pedal, types 'h'),
    as in Flash ALOHA. The two pedals alone run the whole loop: 1 start, 1 stop, 1 accept, h reject.
    """

    go: tuple[str, ...] = ("1", "s", "enter")
    reject: tuple[str, ...] = ("h", "x", "backspace", "left")
    success: tuple[str, ...] = ("n", "right")  # LeRobot's "next" key: save straight away as success
    failure: tuple[str, ...] = ("f",)
    replay: tuple[str, ...] = ("r",)
    gripper_toggle: tuple[str, ...] = ("space", "g")
    gripper_open_more: tuple[str, ...] = ("up",)
    gripper_close_more: tuple[str, ...] = ("down",)
    start_pose: tuple[str, ...] = ("p",)
    quit: tuple[str, ...] = ("q", "esc")

    def lookup(self, key: str, phase: Phase) -> Command | None:
        key = key.lower()
        if key in self.quit:
            return Command.QUIT
        if key in self.go:
            return {Phase.IDLE: Command.START, Phase.RECORDING: Command.STOP, Phase.REVIEW: Command.SAVE_SUCCESS}[phase]
        if key in self.reject:
            return Command.TOGGLE_HOLD if phase is Phase.IDLE else Command.DISCARD
        if key in self.success:
            return Command.SAVE_SUCCESS
        if key in self.failure:
            return Command.SAVE_FAILURE
        if key in self.replay:
            return Command.REPLAY
        if key in self.gripper_toggle:
            return Command.GRIPPER_TOGGLE
        if key in self.gripper_open_more:
            return Command.GRIPPER_OPEN_MORE
        if key in self.gripper_close_more:
            return Command.GRIPPER_CLOSE_MORE
        if key in self.start_pose:
            return Command.SAVE_START_POSE
        return None


_GRIPPER = {Command.GRIPPER_TOGGLE, Command.GRIPPER_OPEN_MORE, Command.GRIPPER_CLOSE_MORE}
_ALLOWED: dict[Phase, set[Command]] = {
    Phase.IDLE: {Command.START, Command.TOGGLE_HOLD, Command.SAVE_START_POSE, Command.QUIT} | _GRIPPER,
    Phase.RECORDING: {Command.STOP, Command.SAVE_SUCCESS, Command.SAVE_FAILURE, Command.DISCARD, Command.QUIT}
    | _GRIPPER,
    Phase.REVIEW: {Command.SAVE_SUCCESS, Command.SAVE_FAILURE, Command.DISCARD, Command.REPLAY, Command.QUIT},
}


class SessionMachine:
    """Phase transitions. `handle` returns the command to execute, or None if it is ignored."""

    def __init__(self) -> None:
        self.phase = Phase.IDLE

    def handle(self, cmd: Command) -> Command | None:
        if cmd not in _ALLOWED[self.phase]:
            return None
        if cmd is Command.START:
            self.phase = Phase.RECORDING
        elif cmd is Command.STOP:
            self.phase = Phase.REVIEW
        elif cmd in (Command.SAVE_SUCCESS, Command.SAVE_FAILURE, Command.DISCARD):
            self.phase = Phase.IDLE
        return cmd

    def time_limit_reached(self) -> bool:
        """Move a running take into review. Returns True if it did."""
        if self.phase is Phase.RECORDING:
            self.phase = Phase.REVIEW
            return True
        return False


class ActionLookahead:
    """Delay frames by `k` steps so each frame's action is the measured position `k` steps later.

    `push(frame, state)` returns the frames that are now complete. `flush()` completes the last `k`
    frames using the final measured state (the arm is at rest at the end of a demo).
    Joints passed in `cmds` (e.g. a key-driven gripper) take the action from the command issued
    *at that frame* instead of from future measurements: the command is the true action there.
    """

    def __init__(self, k: int, joint_names: list[str]):
        if k < 0:
            raise ValueError("lookahead must be >= 0")
        self.k = k
        self.joint_names = list(joint_names)
        self._pending: deque[tuple[dict[str, Any], list[float], dict[str, float]]] = deque()
        self._last_state: list[float] | None = None

    def reset(self) -> None:
        self._pending.clear()
        self._last_state = None

    def _complete(self, frame: dict[str, Any], future_state: list[float], cmds: dict[str, float]) -> dict:
        action = list(future_state)
        for name, value in cmds.items():
            action[self.joint_names.index(name)] = value
        out = dict(frame)
        out["action"] = action
        return out

    def push(self, frame: dict[str, Any], state: list[float], cmds: dict[str, float] | None = None) -> list[dict]:
        self._pending.append((frame, list(state), dict(cmds or {})))
        self._last_state = list(state)
        ready = []
        while len(self._pending) > self.k:
            f, _, c = self._pending.popleft()
            # The newest state is exactly k steps after the frame being completed.
            ready.append(self._complete(f, state, c))
        return ready

    def flush(self) -> list[dict]:
        if self._last_state is None:
            return []
        ready = [self._complete(f, self._last_state, c) for f, _, c in self._pending]
        self._pending.clear()
        return ready


@dataclass
class GripperCommand:
    """Target for a torque-on gripper, in LeRobot's normalised 0-100 gripper units."""

    open_pos: float
    closed_pos: float
    step: float = 5.0
    target: float = field(init=False)

    def __post_init__(self) -> None:
        self.target = self.open_pos
        self._lo, self._hi = sorted((self.open_pos, self.closed_pos))

    @property
    def is_open(self) -> bool:
        return abs(self.target - self.open_pos) <= abs(self.target - self.closed_pos)

    def toggle(self) -> float:
        self.target = self.closed_pos if self.is_open else self.open_pos
        return self.target

    def nudge(self, towards_open: bool) -> float:
        direction = 1.0 if self.open_pos > self.closed_pos else -1.0
        if not towards_open:
            direction = -direction
        self.target = min(self._hi, max(self._lo, self.target + direction * self.step))
        return self.target


def approach_waypoints(start: np.ndarray, goal: np.ndarray, max_speed: float, fps: float) -> np.ndarray:
    """Linear waypoints from `start` to `goal` so no joint moves faster than `max_speed` units/s.

    Always ends exactly at `goal`. Used to move the held arm to a replay's first pose or a start pose.
    """
    start, goal = np.asarray(start, dtype=np.float64), np.asarray(goal, dtype=np.float64)
    if max_speed <= 0 or fps <= 0:
        raise ValueError("max_speed and fps must be positive")
    dist = float(np.max(np.abs(goal - start))) if start.size else 0.0
    n = max(1, int(np.ceil(dist / (max_speed / fps))))
    alphas = np.arange(1, n + 1, dtype=np.float64) / n
    return start[None, :] + alphas[:, None] * (goal - start)[None, :]


@dataclass
class TrackingReport:
    joint_names: list[str]
    median_abs: np.ndarray
    max_abs: np.ndarray

    def worst(self) -> tuple[str, float]:
        i = int(np.argmax(self.max_abs))
        return self.joint_names[i], float(self.max_abs[i])

    def format(self, units: str = "") -> str:
        parts = [f"{n.removesuffix('.pos')} {m:.1f}/{x:.1f}" for n, m, x in
                 zip(self.joint_names, self.median_abs, self.max_abs, strict=True)]
        name, val = self.worst()
        return (f"tracking error median/max{(' ' + units) if units else ''}: " + ", ".join(parts)
                + f" | worst {name.removesuffix('.pos')} {val:.1f}")


def tracking_report(commanded: np.ndarray, measured: np.ndarray, joint_names: list[str], delay: int = 1) -> TrackingReport:
    """Compare commanded targets with positions measured `delay` ticks later (the servo needs a tick to act).

    Large errors on the arm joints mean the demo was too fast for the servos to reproduce: the policy
    would be learning motions the SO-101 cannot execute.
    """
    commanded, measured = np.asarray(commanded, dtype=np.float64), np.asarray(measured, dtype=np.float64)
    if commanded.shape != measured.shape or commanded.ndim != 2:
        raise ValueError("commanded and measured must have the same (T, J) shape")
    if delay < 0 or delay >= len(commanded):
        delay = 0
    err = np.abs(commanded[: len(commanded) - delay] - measured[delay:])
    return TrackingReport(list(joint_names), np.median(err, axis=0), np.max(err, axis=0))


@dataclass
class EpisodeStats:
    """Running duration stats so operators can see whether they are hitting the tempo target."""

    durations_s: list[float] = field(default_factory=list)

    def add(self, duration_s: float) -> None:
        self.durations_s.append(duration_s)

    def summary(self) -> str:
        if not self.durations_s:
            return "no saved successes yet"
        d = sorted(self.durations_s)
        fastest_quarter = d[: max(1, len(d) // 4)]
        return (
            f"{len(d)} ok | median {statistics.median(d):.1f}s | fastest {d[0]:.1f}s | "
            f"fastest-25% median {statistics.median(fastest_quarter):.1f}s"
        )
