"""The closed-loop state machine (PLAYBOOK Section 8): one program runs the whole demo.

Per task (brick) from the planner:

    PICK          run the task's pick skill (PICK_AND_SHOW) until the brick is held still at the inspect spot
    VERIFY_GRASP  gripper position (+ Present_Current if configured): EMPTY -> re-open and retry the pick
    DETECT        hold until the arm is still, then either grab N fresh `inspect` frames -> Inspector.inspect(frames)
                  or (sweep enabled) spin the brick on wrist_roll through N angles (`dlsj.loop.sweep`), each frame's
                  inference queued while the wrist moves on, aggregated per `aggregate` (topk / majority / max /
                  median, `dlsj.inspect.aggregate`); a grasp lost during the sweep ends the brick ("dropped")
    REASON        decision tree (reasoner.decide): plain -> good plate, circle -> minor plate, cross -> defective
                  plate; uncertain at either stage -> re-inspect (RESHOW / re-grab); after `max_uncertain`
                  uncertain inspections -> DEFECTIVE (safe default), logged as forced
    PLACE         run the chosen place skill until the gripper opens (released)
    -> planner.complete(task) and loop

Every stage is timed into the RunLog (timings.jsonl) and every result/decision/retry is an event
(events.jsonl). `abort()` stops at the next control tick; the arm holds, then ramps home on shutdown.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np

from dlsj.contracts import InspectionResult, Inspector, Label
from dlsj.loop.executor import Cameras, SkillExecutor
from dlsj.loop.planner import Task, TaskQueue
from dlsj.loop.poses import GRIPPER
from dlsj.loop.reasoner import Decision, ReasonConfig, decide, inspection_record
from dlsj.loop.skills import Outcome
from dlsj.loop.sweep import SweepConfig, Sweeper, grasp_state  # noqa: F401  (grasp_state re-exported)
from dlsj.loop.telemetry import RunLog, Stopwatch

log = logging.getLogger(__name__)


class LoopState(str, Enum):
    IDLE = "idle"
    HOMING = "homing"
    WAITING = "waiting_for_brick"
    PICK = "pick"
    VERIFY_GRASP = "verify_grasp"
    DETECT = "detect"
    REASON = "reason"
    RESHOW = "reshow"
    PLACE = "place"
    DONE = "done"
    ABORTED = "aborted"
    ERROR = "error"


@dataclass
class ControllerConfig:
    max_pick_attempts: int = 3  # first try + 2 retries (PLAYBOOK 8: "on EMPTY, retry up to 2")
    max_place_attempts: int = 2
    inspect_camera: str = "inspect"
    inspect_frames: int = 3
    settle_still_ticks: int = 10
    settle_still_deg_s: float = 2.0
    settle_timeout_s: float = 3.0
    grasp_empty_pos: float = 4.0  # gripper closed this far = nothing between the jaws
    grasp_current_min: float | None = None  # raw Present_Current that means "holding" (None = position only)
    reopen_gripper: float = 60.0  # open to this before retrying a pick
    go_home: bool = True  # ramp to the home pose at start and on shutdown (needs a saved home pose)
    home_between_bricks: bool = False
    reason: ReasonConfig = field(default_factory=ReasonConfig)
    sweep: SweepConfig | None = None  # None or frames == 0: the classic N-frame inspection
    aggregate: str = "topk"  # sweep aggregation strategy (dlsj.inspect.aggregate.STRATEGIES)
    sweep_async: bool = True  # infer each sweep frame while the wrist moves to the next stop


@dataclass
class BrickReport:
    task_id: int
    status: str  # "placed" | "pick_failed" | "dropped" (grasp lost in the sweep) | "place_failed" | "aborted" | "error"
    label: Label | None = None
    forced: bool = False
    place_skill: str | None = None
    inspections: list[dict] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    pick_attempts: int = 0
    duration_s: float = 0.0


class Controller:
    def __init__(self, executor: SkillExecutor, cameras: Cameras, inspector: Inspector, planner: TaskQueue,
                 runlog: RunLog | None = None, cfg: ControllerConfig | None = None, home: np.ndarray | None = None,
                 wait_for_brick: Callable[[Task], bool] | None = None):
        self.ex, self.cameras, self.inspector, self.planner = executor, cameras, inspector, planner
        self.log = runlog or executor.log
        self.cfg = cfg or ControllerConfig()
        self.home = home
        self.wait_for_brick = wait_for_brick
        self.abort_event = threading.Event()
        self.state = LoopState.IDLE
        self.task: Task | None = None
        self.reports: list[BrickReport] = []
        self.counts: Counter[str] = Counter()
        self.last_result: InspectionResult | None = None
        self.last_inspect_frame: np.ndarray | None = None
        self.last_decision: Decision | None = None
        self.message = ""

    # ------------------------------------------------------------------ control
    def abort(self) -> None:
        self.abort_event.set()

    @property
    def aborted(self) -> bool:
        return self.abort_event.is_set()

    def _set(self, state: LoopState, message: str = "") -> None:
        if state != self.state:
            log.info("state: %s%s", state.value, f" ({message})" if message else "")
            self.log.event("state", state=state.value, task=self.task.id if self.task else None, message=message)
        self.state, self.message = state, message

    def snapshot(self) -> dict:
        r = self.last_result
        return {
            "state": self.state.value,
            "message": self.message,
            "task": self.task.brief() if self.task else None,
            "skill": dict(self.ex.live),
            "last_inspection": inspection_record(r) if r else None,
            "last_decision": None if self.last_decision is None else {
                "kind": self.last_decision.kind, "label": self.last_decision.label.value,
                "skill": self.last_decision.skill, "reason": self.last_decision.reason,
                "forced": self.last_decision.forced, "stage": self.last_decision.stage},
            "counts": dict(self.counts),
            "latency": self.log.summary(),
        }

    # ------------------------------------------------------------------ main loop
    def run(self) -> list[BrickReport]:
        self.log.event("run_start", reason_config=self.cfg.reason.__dict__, library=self.ex.library.describe())
        try:
            if self.cfg.go_home and self.home is not None:
                self._set(LoopState.HOMING, "moving to home")
                self.ex.move_to(self.home, self.abort_event)
            while not self.aborted:
                task = self.planner.next()
                if task is None:
                    break
                self.task = task
                if self.wait_for_brick is not None:
                    self._set(LoopState.WAITING, "place a brick, then confirm")
                    if not self.wait_for_brick(task):
                        break
                report = self.run_task(task)
                self.reports.append(report)
                if report.status == "error":
                    self._set(LoopState.ERROR, "stopping after an error")
                    break
                if self.cfg.home_between_bricks and self.home is not None and not self.aborted:
                    self.ex.move_to(self.home, self.abort_event)
        finally:
            final = LoopState.ABORTED if self.aborted else (
                self.state if self.state is LoopState.ERROR else LoopState.DONE)
            if self.cfg.go_home and self.home is not None:
                self._set(LoopState.HOMING, "returning home")
                self.ex.move_to(self.home)  # also after an abort, like teleop_web's Ctrl+C
            self.task = None
            self._set(final)
            self.log.event("run_end", state=final.value, counts=dict(self.counts),
                           bricks=[r.__dict__ | {"label": r.label.value if r.label else None} for r in self.reports])
        return self.reports

    def run_task(self, task: Task) -> BrickReport:
        t0 = time.perf_counter()
        rep = BrickReport(task.id, "aborted")
        self.log.event("task_start", task=task.brief())
        try:
            rep.status = self._run_task(task, rep)
        finally:
            rep.duration_s = time.perf_counter() - t0
            self.log.timing("brick", rep.duration_s * 1000.0, task=task.id, status=rep.status)
            self.counts[rep.status if rep.status != "placed" else f"placed_{rep.label.value}"] += 1
            self.log.event("brick_done", task=task.id, status=rep.status,
                           label=rep.label.value if rep.label else None, forced=rep.forced,
                           place_skill=rep.place_skill, pick_attempts=rep.pick_attempts,
                           inspections=len(rep.inspections), duration_s=round(rep.duration_s, 3))
        return rep

    def _run_task(self, task: Task, rep: BrickReport) -> str:
        # ---------------- PICK + VERIFY_GRASP
        picked = False
        with Stopwatch(self.log, "pick", task=task.id) as _:
            for attempt in range(1, self.cfg.max_pick_attempts + 1):
                rep.pick_attempts = attempt
                self._set(LoopState.PICK, f"{task.pick_skill} attempt {attempt}/{self.cfg.max_pick_attempts}")
                res = self.ex.run(task.pick_skill, self.abort_event, **task.pick_params)
                if res.outcome is Outcome.ABORTED:
                    return "aborted"
                if res.outcome is Outcome.ERROR:
                    self.log.event("error", stage="pick", error=res.error)
                    return "error"
                self._set(LoopState.VERIFY_GRASP)
                state = res.final_state
                cur = self.ex.robot.read_gripper_current()
                grasp = grasp_state(float(state[GRIPPER]), cur, self.cfg.grasp_empty_pos, self.cfg.grasp_current_min)
                self.log.event("grasp_check", task=task.id, attempt=attempt, outcome=res.outcome.value, grasp=grasp,
                               gripper_pos=round(float(state[GRIPPER]), 2), gripper_current=cur)
                if res.outcome is Outcome.DONE and grasp != "EMPTY":
                    picked = True
                    break
                why = "timed out" if res.outcome is Outcome.TIMEOUT else "nothing in the gripper"
                self.log.event("retry", stage="pick", task=task.id, attempt=attempt, reason=why)
                if grasp == "EMPTY" or res.outcome is Outcome.TIMEOUT:
                    self.ex.set_gripper(self.cfg.reopen_gripper, self.abort_event)
        if not picked:
            self.planner.fail(task, "pick failed")
            return "pick_failed"

        # ---------------- DETECT + REASON (with the uncertainty loop)
        decision: Decision | None = None
        attempt = 0
        while decision is None or decision.kind == "reinspect":
            attempt += 1
            if self.aborted:
                return "aborted"
            self._set(LoopState.DETECT, f"inspection {attempt}")
            with Stopwatch(self.log, "settle", task=task.id):
                still = self.ex.hold_until_still(self.cfg.settle_still_ticks, self.cfg.settle_still_deg_s,
                                                 self.cfg.settle_timeout_s, self.abort_event)
            sweep = self.cfg.sweep
            with Stopwatch(self.log, "detect", task=task.id, attempt=attempt):
                if sweep is not None and sweep.frames > 0:
                    frames, result, status = self._sweep_inspect(task, attempt)
                    if status != "ok":
                        if status == "grasp_lost":
                            self.planner.fail(task, "brick dropped during the inspection sweep")
                            return "dropped"
                        if status == "aborted":
                            return "aborted"
                        self.log.event("error", stage="sweep", error=status)
                        return "error"
                else:
                    frames = self.cameras.fresh(self.cfg.inspect_camera, self.cfg.inspect_frames)
                    result = self.inspector.inspect(frames)
            self.log.timing("inspector", result.latency_ms, task=task.id, device=result.device)
            for stage, ms in result.stage_latency_ms.items():
                dev = result.device if stage == "stage1" else result.mark_device
                self.log.timing(f"inspector.{stage}", ms, task=task.id, device=dev)
            best = result.summary.get("best_index") if result.frames else None
            self.last_result = result
            self.last_inspect_frame = frames[best] if best is not None else frames[len(frames) // 2]
            rec = inspection_record(result)
            rep.inspections.append(rec)
            self.log.event("inspection", task=task.id, attempt=attempt, settled=still, **rec)

            self._set(LoopState.REASON)
            with Stopwatch(self.log, "reason", task=task.id):
                decision = decide(result, attempt, self.cfg.reason, task.perfect_skill, task.defective_skill,
                                  task.perfect_params, task.defective_params, task.minor_skill, task.minor_params)
            self.last_decision = decision
            rep.decisions.append(decision.reason)
            self.log.event("decision", task=task.id, attempt=attempt, kind=decision.kind, label=decision.label.value,
                           skill=decision.skill, forced=decision.forced, reason=decision.reason,
                           uncertain_stage=decision.stage or None)
            log.info("decision: %s", decision.reason)
            if decision.forced:
                self.log.event("uncertain_fallback", task=task.id, attempts=attempt, label=decision.label.value,
                               stage=decision.stage)
            if decision.kind == "reinspect":
                self.log.event("retry", stage="detect", task=task.id, attempt=attempt, reason=decision.reason)
                if decision.skill is not None:
                    self._set(LoopState.RESHOW, decision.skill)
                    res = self.ex.run(decision.skill, self.abort_event)
                    if res.outcome is Outcome.ABORTED:
                        return "aborted"
                    if res.outcome is Outcome.ERROR:
                        self.log.event("error", stage="reshow", error=res.error)
                        return "error"

        rep.label, rep.forced, rep.place_skill = decision.label, decision.forced, decision.skill
        return self._place(task, decision)

    def _sweep_inspect(self, task: Task, attempt: int) -> tuple[list[np.ndarray], InspectionResult | None, str]:
        """Spin the brick, inspect every angle. Returns (frames, result, sweep status)."""
        cfg = self.cfg
        insp = self.inspector
        session = insp.start_sweep(cfg.aggregate) if cfg.sweep_async and hasattr(insp, "start_sweep") else None
        sweeper = Sweeper(self.ex.robot, self.cameras, cfg.sweep, self.abort_event)
        res = sweeper.run(self.ex.cmd, on_frame=session.add if session is not None else None)
        if res.final_cmd is not None:
            self.ex.cmd = res.final_cmd.copy()  # the executor continues from the restored start pose
        for st in res.stops:
            self.log.timing("sweep.stop", st.move_ms + st.settle_ms + st.grab_ms, task=task.id, i=st.index,
                            move_ms=round(st.move_ms, 1), settle_ms=round(st.settle_ms, 1),
                            grab_ms=round(st.grab_ms, 1), settled=st.settled, angle=round(st.angle_deg, 2))
        self.log.timing("sweep", res.duration_ms, task=task.id, n=len(res.frames), status=res.status)
        self.log.timing("sweep.return", res.return_ms, task=task.id)
        self.log.event("sweep", task=task.id, attempt=attempt, **res.record())
        if res.status != "ok" or not res.frames:
            if session is not None:
                session.cancel()
            return res.frames, None, res.status if res.status != "ok" else "camera_error"
        t0 = time.perf_counter()
        if session is not None:
            result = session.result()
        elif hasattr(insp, "inspect_sweep"):
            result = insp.inspect_sweep(res.frames, res.angles, cfg.aggregate)
        else:
            result = insp.inspect(res.frames)
        self.log.timing("sweep.inference_tail", (time.perf_counter() - t0) * 1000.0, task=task.id,
                        mode="async" if session is not None else "after")
        return res.frames, result, "ok"

    def _place(self, task: Task, decision: Decision) -> str:
        # ---------------- PLACE
        with Stopwatch(self.log, "place", task=task.id, skill=decision.skill):
            for attempt in range(1, self.cfg.max_place_attempts + 1):
                self._set(LoopState.PLACE, f"{decision.skill} attempt {attempt}")
                res = self.ex.run(decision.skill, self.abort_event, **decision.params)
                if res.outcome is Outcome.DONE:
                    break
                if res.outcome is Outcome.ABORTED:
                    return "aborted"
                if res.outcome is Outcome.ERROR:
                    self.log.event("error", stage="place", error=res.error)
                    return "error"
                self.log.event("retry", stage="place", task=task.id, attempt=attempt, reason="place timed out")
            else:
                self.planner.fail(task, "place timed out")
                return "place_failed"
        self.planner.complete(task, decision.label.value)
        return "placed"
