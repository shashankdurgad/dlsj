"""Hardware-free logic for kinesthetic recording on the SO-101.

Kept separate from the serial bus and cameras so it can be unit-tested:

* `KeyMap` / `SessionMachine`: which key does what in which phase (idle, recording, labelling).
* `ActionLookahead`: turns a stream of measured joint positions into (frame, action) pairs where
  action[t] = measured_position[t + k]. Recording the *future* position as the action compensates for
  the position controller lagging its target when the policy is later replayed (k=0 records the
  position itself, the Flash ALOHA convention).
* `GripperCommand`: open/close/nudge target for a torque-on gripper driven by keys or a foot pedal,
  so both hands stay on the arm.
"""

from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class Phase(Enum):
    IDLE = auto()  # between episodes: operator resets the scene
    RECORDING = auto()  # frames are written
    LABELLING = auto()  # episode hit the time limit; waiting for success / failure / discard


class Command(Enum):
    START = auto()  # begin a new episode (release arm torque if held)
    SAVE_SUCCESS = auto()
    SAVE_FAILURE = auto()
    DISCARD = auto()
    GRIPPER_TOGGLE = auto()
    GRIPPER_OPEN_MORE = auto()
    GRIPPER_CLOSE_MORE = auto()
    TOGGLE_HOLD = auto()  # idle only: torque the arm on/off where it is, to rest it while resetting
    QUIT = auto()


@dataclass(frozen=True)
class KeyMap:
    """Canonical key names as produced by `lerobot.utils.keyboard_input.create_key_listener`.

    Foot pedals usually present as a keyboard; map the pedal to `space` (gripper) and `s`/`n`.
    """

    start: tuple[str, ...] = ("s", "enter")
    success: tuple[str, ...] = ("n", "right")
    failure: tuple[str, ...] = ("f",)
    discard: tuple[str, ...] = ("r", "left", "backspace")
    gripper_toggle: tuple[str, ...] = ("space", "g")
    gripper_open_more: tuple[str, ...] = ("up",)
    gripper_close_more: tuple[str, ...] = ("down",)
    hold: tuple[str, ...] = ("h",)
    quit: tuple[str, ...] = ("q", "esc")

    def lookup(self, key: str) -> Command | None:
        key = key.lower()
        table = {
            Command.START: self.start,
            Command.SAVE_SUCCESS: self.success,
            Command.SAVE_FAILURE: self.failure,
            Command.DISCARD: self.discard,
            Command.GRIPPER_TOGGLE: self.gripper_toggle,
            Command.GRIPPER_OPEN_MORE: self.gripper_open_more,
            Command.GRIPPER_CLOSE_MORE: self.gripper_close_more,
            Command.TOGGLE_HOLD: self.hold,
            Command.QUIT: self.quit,
        }
        for cmd, keys in table.items():
            if key in keys:
                return cmd
        return None


# Commands that make sense in each phase; anything else is ignored (e.g. `s` while recording).
_ALLOWED: dict[Phase, set[Command]] = {
    Phase.IDLE: {Command.START, Command.TOGGLE_HOLD, Command.GRIPPER_TOGGLE, Command.GRIPPER_OPEN_MORE,
                 Command.GRIPPER_CLOSE_MORE, Command.DISCARD, Command.QUIT},
    Phase.RECORDING: {Command.SAVE_SUCCESS, Command.SAVE_FAILURE, Command.DISCARD, Command.GRIPPER_TOGGLE,
                      Command.GRIPPER_OPEN_MORE, Command.GRIPPER_CLOSE_MORE, Command.QUIT},
    Phase.LABELLING: {Command.SAVE_SUCCESS, Command.SAVE_FAILURE, Command.DISCARD, Command.QUIT},
}


class SessionMachine:
    """Phase transitions. `handle` returns the command to execute, or None if it is ignored."""

    def __init__(self) -> None:
        self.phase = Phase.IDLE

    def handle(self, cmd: Command) -> Command | None:
        if cmd not in _ALLOWED[self.phase]:
            return None
        if cmd is Command.DISCARD and self.phase is Phase.IDLE:
            return None  # nothing to discard
        if cmd is Command.START:
            self.phase = Phase.RECORDING
        elif cmd in (Command.SAVE_SUCCESS, Command.SAVE_FAILURE, Command.DISCARD):
            self.phase = Phase.IDLE
        return cmd

    def time_limit_reached(self) -> None:
        if self.phase is Phase.RECORDING:
            self.phase = Phase.LABELLING


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


@dataclass
class EpisodeStats:
    """Running duration stats so operators can see whether they are hitting the tempo target."""

    durations_s: list[float] = field(default_factory=list)

    def add(self, duration_s: float) -> None:
        self.durations_s.append(duration_s)

    def summary(self) -> str:
        if not self.durations_s:
            return "no saved episodes yet"
        d = sorted(self.durations_s)
        fastest_quarter = d[: max(1, len(d) // 4)]
        return (
            f"{len(d)} saved | median {statistics.median(d):.1f}s | fastest {d[0]:.1f}s | "
            f"fastest-25% median {statistics.median(fastest_quarter):.1f}s"
        )
