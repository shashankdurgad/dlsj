"""Task queue. The controller asks for the next Task and reports how it ended.

Phase 1 (`AnyBrickPlanner`): "whatever brick is there": an endless (or `max_tasks`-long) stream of
identical tasks: PICK_AND_SHOW, then PLACE_BLUE (good) / PLACE_MINOR (circle) / PLACE_PINK (cross). Stops after `max_consecutive_failures`
failed picks in a row (an empty table looks exactly like that).

Phase 2 (`OrderedPlanner`, PLAYBOOK 8.2): a number reader on the `top` image produces detections
`[{number, confidence, box, table_xy}]`; `replan(detections)` turns them into tasks sorted by number,
skipping numbers already placed. Each task picks with PICK_MARKED (the target is the `target` signal the
policy observation must be marked with) and places perfect bricks with PLACE_SLOT(slot=k); defective
bricks still go to the pink plate (and minor-defect bricks to the minor plate; they take no slot). `expected_number` is there for the inspect-camera number re-check.
"""

from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass, field

from dlsj.loop.skills import PICK_AND_SHOW, PICK_MARKED, PLACE_BLUE, PLACE_MINOR, PLACE_PINK, PLACE_SLOT


@dataclass
class Task:
    id: int
    pick_skill: str = PICK_AND_SHOW
    pick_params: dict = field(default_factory=dict)
    perfect_skill: str = PLACE_BLUE
    perfect_params: dict = field(default_factory=dict)
    defective_skill: str = PLACE_PINK
    defective_params: dict = field(default_factory=dict)
    minor_skill: str = PLACE_MINOR
    minor_params: dict = field(default_factory=dict)
    target: dict | None = None  # phase 2 perception signal: {number, confidence, box, table_xy}
    expected_number: int | None = None
    attempts: int = 0

    def brief(self) -> dict:
        return {"id": self.id, "pick": self.pick_skill, "perfect": self.perfect_skill, "defective": self.defective_skill,
                "minor": self.minor_skill, "target": self.target, "expected_number": self.expected_number, "attempts": self.attempts}


class TaskQueue:
    """FIFO of tasks plus a history of how each ended."""

    def __init__(self):
        self.pending: deque[Task] = deque()
        self.history: list[tuple[Task, str]] = []
        self._ids = itertools.count(1)
        self.consecutive_failures = 0

    def new_task(self, **kw) -> Task:
        return Task(id=next(self._ids), **kw)

    def enqueue(self, task: Task) -> Task:
        self.pending.append(task)
        return task

    def next(self) -> Task | None:
        return self.pending.popleft() if self.pending else None

    def complete(self, task: Task, outcome: str) -> None:
        self.history.append((task, outcome))
        self.consecutive_failures = 0

    def fail(self, task: Task, reason: str, requeue: bool = False) -> None:
        self.history.append((task, f"failed: {reason}"))
        self.consecutive_failures += 1
        if requeue:
            task.attempts += 1
            self.pending.appendleft(task)


class AnyBrickPlanner(TaskQueue):
    """Phase 1: one brick at a time, whatever brick is there."""

    def __init__(self, max_tasks: int | None = None, max_consecutive_failures: int = 3):
        super().__init__()
        self.max_tasks = max_tasks
        self.max_consecutive_failures = max_consecutive_failures
        self.issued = 0

    def next(self) -> Task | None:
        if self.pending:
            return self.pending.popleft()
        if self.consecutive_failures >= self.max_consecutive_failures:
            return None
        if self.max_tasks is not None and self.issued >= self.max_tasks:
            return None
        self.issued += 1
        return self.new_task()


class OrderedPlanner(TaskQueue):
    """Phase 2: bricks in ascending number order, re-planned from fresh detections after every brick."""

    def __init__(self, min_confidence: float = 0.5):
        super().__init__()
        self.min_confidence = min_confidence
        self.handled_numbers: list[int] = []  # placed anywhere (slot or reject plate)
        self.filled_slots = 0

    def replan(self, detections: list[dict]) -> list[Task]:
        self.pending.clear()
        good = [d for d in detections if d.get("confidence", 1.0) >= self.min_confidence
                and d["number"] not in self.handled_numbers]
        for d in sorted(good, key=lambda d: d["number"]):
            # provisional: assumes every brick ahead is perfect; replan after each brick fixes it up
            slot = self.filled_slots + 1 + len(self.pending)
            self.enqueue(self.new_task(pick_skill=PICK_MARKED, perfect_skill=PLACE_SLOT,
                                       perfect_params={"slot": slot}, target=d, expected_number=d["number"]))
        return list(self.pending)

    def complete(self, task: Task, outcome: str) -> None:
        """`outcome` is the label acted on ("perfect" fills the next slot; "minor"/"defective" go to their plates)."""
        super().complete(task, outcome)
        if task.expected_number is not None:
            self.handled_numbers.append(task.expected_number)
        if outcome == "perfect":
            self.filled_slots += 1
