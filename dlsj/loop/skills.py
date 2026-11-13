"""Skill library: named pi0.5 skills = a fixed instruction string + when to consider the skill finished.

A VLA never says "done", so each skill carries a `Termination` that the executor evaluates every control
tick on the measured joint state. A skill ends with:

* DONE     when every configured success condition holds at the same tick (and `min_duration_s` passed);
* TIMEOUT  after `max_duration_s`;
* ABORTED  when the operator/controller aborts.

Success conditions (each optional, all configured ones must hold together):

* `gripper`: a `GripperCondition`. PICK uses "closed" (min_value < gripper position <= threshold for
  `hold_ticks` consecutive ticks, then latched, and it must still be closed now: "closed then held"; the
  `min_value` floor means jaws closed on nothing never count as holding). PLACE uses
  "opened" (position >= threshold for `hold_ticks` ticks: the brick was released).
* `still_ticks`: the arm joints (gripper excluded) moved slower than `still_deg_s` for N consecutive ticks.
  Velocity is |delta q| * nominal fps, the same formula as the PLAYBOOK's idle-frame filter.
* `near_pose`: max |arm joint - saved pose| <= `near_pose_tol_deg`, for a pose saved in calibration/
  (`dlsj.loop.poses`). A missing pose file makes the condition a no-op (with a warning), so the loop still
  runs before anyone saved an `inspect` pose.

Instruction strings must be byte-identical to the training demos' task strings (PLAYBOOK 3.1b).
Phase-2 skills (`PICK_MARKED`, `PLACE_SLOT`) are placeholders whose strings take `{slot}`-style params.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Literal

import numpy as np

from dlsj.loop.poses import ARM, GRIPPER

log = logging.getLogger(__name__)

PICK_AND_SHOW = "PICK_AND_SHOW"
PLACE_BLUE = "PLACE_BLUE"
PLACE_PINK = "PLACE_PINK"
RESHOW = "RESHOW"
PICK_MARKED = "PICK_MARKED"  # phase 2
PLACE_SLOT = "PLACE_SLOT"  # phase 2


@dataclass(frozen=True)
class GripperCondition:
    mode: Literal["closed", "opened"]
    threshold: float  # gripper position (0 = closed .. 100 = open)
    hold_ticks: int = 5  # consecutive ticks the condition must hold before it latches
    must_persist: bool = True  # "closed" only: once latched, it must still hold now (brick not dropped)
    min_value: float | None = None  # "closed" only: below this the jaws closed on nothing (EMPTY), not holding

    def holds(self, gripper: float) -> bool:
        if self.mode == "opened":
            return gripper >= self.threshold
        return gripper <= self.threshold and (self.min_value is None or gripper > self.min_value)


@dataclass(frozen=True)
class Termination:
    max_duration_s: float = 20.0
    min_duration_s: float = 0.5
    gripper: GripperCondition | None = None
    still_ticks: int = 0  # 0 = stillness not required
    still_deg_s: float = 3.0
    near_pose: str | None = None
    near_pose_tol_deg: float = 8.0

    @staticmethod
    def from_dict(d: dict) -> "Termination":
        d = dict(d)
        if isinstance(d.get("gripper"), dict):
            d["gripper"] = GripperCondition(**d["gripper"])
        return Termination(**d)


@dataclass(frozen=True)
class Skill:
    name: str
    instruction: str  # may contain {placeholders}, e.g. "place the brick in slot {slot}"
    termination: Termination = field(default_factory=Termination)
    description: str = ""

    def render(self, **params) -> str:
        try:
            return self.instruction.format(**params)
        except KeyError as e:
            raise ValueError(f"skill {self.name} needs parameter {e} for {self.instruction!r}") from None


def default_skills() -> dict[str, Skill]:
    """Phase-1 defaults. Tune the numbers on hardware; override per run with --skills-config JSON."""
    # holding a brick stalls the jaws well above 0; <= 4 means they closed on nothing (= grasp_empty_pos)
    holding = GripperCondition("closed", threshold=40.0, hold_ticks=8, min_value=4.0)
    released = GripperCondition("opened", threshold=40.0, hold_ticks=5)  # well above the holding stall
    show = Termination(max_duration_s=25.0, min_duration_s=1.0, gripper=holding, still_ticks=15, still_deg_s=3.0,
                       near_pose="inspect", near_pose_tol_deg=10.0)
    place = Termination(max_duration_s=20.0, min_duration_s=1.0, gripper=released)
    return {
        PICK_AND_SHOW: Skill(PICK_AND_SHOW, "pick up the brick and show it to the inspection camera", show,
                             "brick on the table -> held still at the inspect spot"),
        PLACE_BLUE: Skill(PLACE_BLUE, "place the brick on the blue plate", place, "inspect spot -> blue plate"),
        PLACE_PINK: Skill(PLACE_PINK, "place the brick on the pink plate", place, "inspect spot -> pink plate"),
        RESHOW: Skill(RESHOW, "turn the brick and show it to the inspection camera again",
                      replace(show, max_duration_s=12.0), "optional recovery skill for uncertain inspections"),
        # phase 2 placeholders (PLAYBOOK 8.2); not trained yet
        PICK_MARKED: Skill(PICK_MARKED, "pick up the marked brick and show it to the inspection camera", show,
                           "phase 2: target brick marked in the top image"),
        PLACE_SLOT: Skill(PLACE_SLOT, "place the brick in slot {slot}", place, "phase 2: ordered placement"),
    }


class SkillLibrary:
    """Name -> Skill. The state machine only refers to skills by name."""

    def __init__(self, skills: dict[str, Skill] | None = None, enabled_optional: set[str] | None = None):
        self.skills = dict(skills or default_skills())
        # Optional skills only run when trained; RESHOW is off unless enabled.
        self.enabled_optional = set(enabled_optional or ())

    def __getitem__(self, name: str) -> Skill:
        return self.skills[name]

    def __contains__(self, name: str) -> bool:
        return name in self.skills

    def available(self, name: str) -> bool:
        return name in self.skills and (name != RESHOW or RESHOW in self.enabled_optional)

    def override(self, name: str, instruction: str | None = None, **termination) -> None:
        s = self.skills[name]
        if isinstance(termination.get("gripper"), dict):
            termination["gripper"] = GripperCondition(**termination["gripper"])
        self.skills[name] = replace(s, instruction=instruction or s.instruction,
                                    termination=replace(s.termination, **termination))

    def load_overrides(self, path: str | Path) -> None:
        """JSON: {"PICK_AND_SHOW": {"max_duration_s": 30, "gripper": {...}}, "RESHOW": {"enabled": true}}."""
        for name, cfg in json.loads(Path(path).read_text()).items():
            cfg = dict(cfg)
            if cfg.pop("enabled", False):
                self.enabled_optional.add(name)
            instr = cfg.pop("instruction", None)
            if name not in self.skills:
                self.skills[name] = Skill(name, instr or name.lower(), Termination.from_dict(cfg))
            else:
                self.override(name, instr, **cfg)

    def describe(self) -> dict:
        return {n: {"instruction": s.instruction, "termination": asdict(s.termination),
                    "enabled": self.available(n)} for n, s in self.skills.items()}


# ----------------------------------------------------------------------------- termination monitor
class Outcome(str, Enum):
    DONE = "done"
    TIMEOUT = "timeout"
    ABORTED = "aborted"
    ERROR = "error"


class TerminationMonitor:
    """Evaluates a `Termination` tick by tick. `update` returns an Outcome when the skill should end."""

    def __init__(self, term: Termination, fps: float, poses: dict[str, np.ndarray] | None = None):
        self.term, self.fps = term, fps
        self.pose = None
        if term.near_pose:
            self.pose = (poses or {}).get(term.near_pose)
            if self.pose is None:
                log.warning("termination: pose %r not saved; ignoring the near-pose condition", term.near_pose)
        self.t0: float | None = None
        self.prev: np.ndarray | None = None
        self.still_run = 0
        self.grip_run = 0
        self.grip_latched = False
        self.last_status: dict = {}

    def update(self, state: np.ndarray, t: float) -> Outcome | None:
        term = self.term
        if self.t0 is None:
            self.t0 = t
        elapsed = t - self.t0

        vel = 0.0 if self.prev is None else float(np.max(np.abs(state[ARM] - self.prev[ARM]))) * self.fps
        self.prev = state.copy()
        self.still_run = self.still_run + 1 if vel < term.still_deg_s else 0

        g = term.gripper
        if g is not None:
            self.grip_run = self.grip_run + 1 if g.holds(float(state[GRIPPER])) else 0
            if self.grip_run >= g.hold_ticks:
                self.grip_latched = True
            elif g.mode == "closed" and g.must_persist and self.grip_run == 0:
                self.grip_latched = False  # dropped it / reopened: wait for a new grasp

        conds = {}
        if g is not None:
            conds["gripper"] = self.grip_latched
        if term.still_ticks > 0:
            conds["still"] = self.still_run >= term.still_ticks
        if self.pose is not None:
            conds["near_pose"] = float(np.max(np.abs(state[ARM] - self.pose[ARM]))) <= term.near_pose_tol_deg
        self.last_status = {"elapsed_s": round(elapsed, 3), "vel_deg_s": round(vel, 2), **conds}

        if elapsed >= term.max_duration_s:
            return Outcome.TIMEOUT
        if elapsed >= term.min_duration_s and conds and all(conds.values()):
            return Outcome.DONE
        return None
