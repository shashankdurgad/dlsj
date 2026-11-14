"""Spin-and-inspect: rotate the held brick on `wrist_roll` and grab one fresh `inspect` frame per angle.

Called in DETECT once the arm holds the brick still at the inspect spot (after the pick skill). Relative to the
current command (the pose the pick skill ended in):

    plan    N stops (default 12) evenly over `range_deg` (default 180 = +-90 deg) around the start roll, clipped to
            the calibrated wrist_roll limits minus a margin (`dlsj.arms.joint_limits_deg`); visited from the nearer
            end to the farther one, so the wrist travels range/2 + range + range/2 at most.
    move    only wrist_roll changes; per tick by at most max_speed/fps (Desmond's `step_toward` cap); every other
            joint holds its start command and the gripper holds its (closed) command or `gripper_hold`.
    settle  hold until at least `settle_ms` have passed since arrival AND the measured arm joints moved slower than
            `still_deg_s` for `still_ticks` ticks (timeout `settle_timeout_s`: the frame is still taken, flagged
            `settled=False`).
    grab    `cameras.fresh(camera, discard + 1)[-1]`: the first new frame may have started exposing before the arm
            settled, so it is discarded by default. The measured roll is read just before and after the grab
            (their mean is the frame's angle). `on_frame(frame, angle)` is called right away, e.g. to queue the
            frame's inference while the wrist moves on (`dlsj.inspect.aggregate.SweepSession`).
    grasp   every tick the gripper position must stay above `grasp_empty_pos` (closing on nothing); at every stop
            Present_Current is checked too when `grasp_current_min` is set. A lost grasp ends the sweep.
    return  the command goes back to the start pose (same wrist_roll) whatever happened, so the place skill starts
            from an in-distribution state.

`continuous=True` instead rolls through the range at `continuous_speed_deg_s` and grabs the latest frame each time
the measured roll passes a planned angle (faster, but motion blur; for comparison only).
Every stop's timing (move / settle / grab) is in `SweepResult.stops`.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

from dlsj.loop.poses import ARM, GRIPPER

log = logging.getLogger(__name__)

ROLL = 4  # wrist_roll index in poses.MOTORS


def grasp_state(grip_pos: float, grip_cur: float | None, empty_pos: float, cur_min: float | None) -> str:
    """PLAYBOOK 7.2: closed all the way with no resistance = EMPTY."""
    if cur_min is not None and grip_cur is not None:
        if grip_cur >= cur_min:
            return "HOLDING"
        return "EMPTY" if grip_pos <= empty_pos else "UNKNOWN"
    return "EMPTY" if grip_pos <= empty_pos else "HOLDING"


@dataclass
class SweepConfig:
    frames: int = 12  # stops; 0 = no sweep (the classic 3-frame inspection)
    range_deg: float = 180.0  # total roll range around the start angle
    settle_ms: float = 150.0  # minimum dwell after arriving at a stop
    still_ticks: int = 3  # ...and this many consecutive still ticks (0 = time only)
    still_deg_s: float = 3.0
    settle_timeout_s: float = 1.5
    max_speed_deg_s: float = 90.0  # wrist_roll speed cap (per tick: / fps)
    arrive_tol_deg: float = 1.5  # measured roll within this of the target = arrived
    move_timeout_s: float = 4.0
    discard_frames: int = 1  # new frames dropped before the one kept (exposure started before settling)
    limits: tuple[float, float] | None = None  # calibrated wrist_roll (min, max) deg; None = no clipping
    limit_margin_deg: float = 5.0
    gripper_hold: float | None = None  # gripper command during the sweep (None = keep the current command)
    grasp_empty_pos: float = 4.0
    grasp_current_min: float | None = None
    return_to_start: bool = True
    continuous: bool = False
    continuous_speed_deg_s: float = 30.0
    camera: str = "inspect"
    fps: float = 30.0


@dataclass
class SweepStop:
    index: int
    target_deg: float
    angle_deg: float  # measured wrist_roll at the grab
    move_ms: float
    settle_ms: float
    grab_ms: float
    settled: bool
    gripper_pos: float
    gripper_current: float | None = None


@dataclass
class SweepResult:
    status: str  # "ok" | "grasp_lost" | "aborted" | "camera_error" | "skipped"
    frames: list[np.ndarray] = field(default_factory=list)
    angles: list[float] = field(default_factory=list)
    stops: list[SweepStop] = field(default_factory=list)
    plan: list[float] = field(default_factory=list)
    start_roll: float = 0.0
    final_roll: float = 0.0
    final_cmd: np.ndarray | None = None
    duration_ms: float = 0.0
    return_ms: float = 0.0
    returned: bool = False
    error: str = ""

    def record(self) -> dict:
        """JSON-friendly summary (no images)."""
        return {"status": self.status, "n": len(self.frames), "plan": [round(a, 2) for a in self.plan],
                "angles": [round(a, 2) for a in self.angles], "start_roll": round(self.start_roll, 2),
                "final_roll": round(self.final_roll, 2), "returned": self.returned,
                "duration_ms": round(self.duration_ms, 1), "return_ms": round(self.return_ms, 1), "error": self.error,
                "stops": [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in asdict(s).items()}
                          for s in self.stops]}


def plan_angles(start: float, cfg: SweepConfig) -> list[float]:
    """The roll targets (deg), clipped to the calibrated limits, nearer end first."""
    n = int(cfg.frames)
    if n <= 0:
        return []
    lo, hi = start - cfg.range_deg / 2, start + cfg.range_deg / 2
    if cfg.limits is not None:
        llo, lhi = cfg.limits[0] + cfg.limit_margin_deg, cfg.limits[1] - cfg.limit_margin_deg
        if llo > lhi:
            raise ValueError(f"wrist_roll limits {cfg.limits} leave no room with a {cfg.limit_margin_deg} deg margin")
        if lo < llo or hi > lhi:
            log.info("sweep [%.1f, %.1f] clipped to the wrist_roll limits [%.1f, %.1f]", lo, hi, llo, lhi)
        lo, hi = float(np.clip(lo, llo, lhi)), float(np.clip(hi, llo, lhi))
    if n == 1:
        return [float(np.clip(start, lo, hi))]
    angles = [float(a) for a in np.linspace(lo, hi, n)]
    return angles if abs(start - lo) <= abs(start - hi) else angles[::-1]


class Sweeper:
    """Runs one sweep on a `Robot` + `Cameras` (dlsj.loop.executor protocols)."""

    def __init__(self, robot, cameras, cfg: SweepConfig, abort: threading.Event | None = None):
        self.robot, self.cameras, self.cfg = robot, cameras, cfg
        self.abort = abort or threading.Event()
        self.period = 1.0 / cfg.fps
        self.step = cfg.max_speed_deg_s / cfg.fps

    # ------------------------------------------------------------------ low-level
    def _tick(self, cmd: np.ndarray, t0: float) -> None:
        self.robot.send(cmd.copy())
        dt = time.perf_counter() - t0
        if dt < self.period:
            time.sleep(self.period - dt)

    def _grasp_lost(self, state: np.ndarray, current: float | None = None) -> bool:
        return grasp_state(float(state[GRIPPER]), current, self.cfg.grasp_empty_pos,
                           self.cfg.grasp_current_min) == "EMPTY"

    def _move_roll(self, cmd: np.ndarray, target: float, check_grasp: bool = True,
                   speed: float | None = None) -> tuple[str, np.ndarray]:
        """Step the roll command to `target`, wait for the measured roll. Returns (status, last state)."""
        step = (speed / self.cfg.fps) if speed else self.step
        t_end = time.perf_counter() + self.cfg.move_timeout_s
        state = np.asarray(self.robot.read_state(), np.float32)
        while True:
            t0 = time.perf_counter()
            state = np.asarray(self.robot.read_state(), np.float32)
            if self.abort.is_set():
                return "aborted", state
            if check_grasp and self._grasp_lost(state):
                return "grasp_lost", state
            if abs(float(cmd[ROLL]) - target) < 1e-3 and abs(float(state[ROLL]) - target) <= self.cfg.arrive_tol_deg:
                return "ok", state
            if t0 > t_end:
                log.warning("wrist_roll did not reach %.1f (at %.1f) in %.1fs", target, state[ROLL],
                            self.cfg.move_timeout_s)
                return "ok", state  # keep going: the stop is flagged by its measured angle
            d = target - float(cmd[ROLL])
            cmd[ROLL] = target if abs(d) <= step else float(cmd[ROLL]) + np.sign(d) * step
            self._tick(cmd, t0)

    def _settle(self, cmd: np.ndarray) -> tuple[str, bool]:
        cfg = self.cfg
        t_arrive = time.perf_counter()
        t_end = t_arrive + cfg.settle_timeout_s
        prev, prev_t, run = None, None, 0
        while True:
            t0 = time.perf_counter()
            s = np.asarray(self.robot.read_state(), np.float32)
            if self.abort.is_set():
                return "aborted", False
            if self._grasp_lost(s):
                return "grasp_lost", False
            if prev is not None:
                vel = float(np.max(np.abs(s[ARM] - prev[ARM]))) / max(t0 - prev_t, 1e-3)
                run = run + 1 if vel < cfg.still_deg_s else 0
            prev, prev_t = s, t0
            dwell = (t0 - t_arrive) * 1e3 >= cfg.settle_ms
            if dwell and run >= cfg.still_ticks:
                return "ok", True
            if t0 > t_end:
                return "ok", False
            self._tick(cmd, t0)

    def _grab(self) -> tuple[np.ndarray, float, float]:
        s0 = float(self.robot.read_state()[ROLL])
        frame = self.cameras.fresh(self.cfg.camera, self.cfg.discard_frames + 1)[-1]
        s1 = float(self.robot.read_state()[ROLL])
        return frame, (s0 + s1) / 2, s1

    # ------------------------------------------------------------------ public
    def run(self, cmd: np.ndarray | None = None, on_frame: Callable[[np.ndarray, float], None] | None = None
            ) -> SweepResult:
        cfg = self.cfg
        t_start = time.perf_counter()
        state = np.asarray(self.robot.read_state(), np.float32)
        start_cmd = (state if cmd is None else np.asarray(cmd, np.float32)).copy()
        if cfg.gripper_hold is not None:
            start_cmd[GRIPPER] = cfg.gripper_hold
        res = SweepResult("ok", start_roll=float(start_cmd[ROLL]))
        if cfg.frames <= 0:
            res.status, res.final_cmd = "skipped", start_cmd
            return res
        res.plan = plan_angles(float(start_cmd[ROLL]), cfg)
        work = start_cmd.copy()
        try:
            if cfg.continuous:
                self._run_continuous(work, res, on_frame)
            else:
                self._run_stepped(work, res, on_frame)
        except TimeoutError as e:
            res.status, res.error = "camera_error", str(e)
            log.warning("sweep camera error: %s", e)
        finally:
            res.duration_ms = (time.perf_counter() - t_start) * 1e3
            if cfg.return_to_start and not self.abort.is_set():
                t0 = time.perf_counter()
                st, s = self._move_roll(work, float(start_cmd[ROLL]), check_grasp=False)
                work[:] = start_cmd
                res.returned = st == "ok"
                res.final_roll = float(s[ROLL])
                res.return_ms = (time.perf_counter() - t0) * 1e3
            else:
                res.final_roll = float(self.robot.read_state()[ROLL])
            res.final_cmd = work.copy()
        log.info("sweep %s: %d frames over [%.1f, %.1f] deg in %.0f ms (+%.0f ms back)", res.status, len(res.frames),
                 min(res.angles, default=float("nan")), max(res.angles, default=float("nan")), res.duration_ms,
                 res.return_ms)
        return res

    def _accept(self, res: SweepResult, frame, angle, on_frame) -> None:
        res.frames.append(frame)
        res.angles.append(angle)
        if on_frame is not None:
            on_frame(frame, angle)

    def _run_stepped(self, cmd: np.ndarray, res: SweepResult, on_frame) -> None:
        cfg = self.cfg
        for i, target in enumerate(res.plan):
            t0 = time.perf_counter()
            st, state = self._move_roll(cmd, target)
            t1 = time.perf_counter()
            if st == "ok":
                st, settled = self._settle(cmd)
            if st != "ok":
                res.status = st
                return
            t2 = time.perf_counter()
            cur = self.robot.read_gripper_current() if cfg.grasp_current_min is not None else None
            state = np.asarray(self.robot.read_state(), np.float32)
            if self._grasp_lost(state, cur):
                res.status = "grasp_lost"
                return
            frame, angle, _ = self._grab()
            t3 = time.perf_counter()
            res.stops.append(SweepStop(i, float(target), angle, (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3,
                                       settled, float(state[GRIPPER]), cur))
            self._accept(res, frame, angle, on_frame)

    def _run_continuous(self, cmd: np.ndarray, res: SweepResult, on_frame) -> None:
        cfg = self.cfg
        if not res.plan:
            return
        st, _ = self._move_roll(cmd, res.plan[0])  # full speed to the first end, then a slow roll through
        if st == "ok":
            st, _ = self._settle(cmd)
        if st != "ok":
            res.status = st
            return
        sign = 1.0 if res.plan[-1] >= res.plan[0] else -1.0
        step = cfg.continuous_speed_deg_s / cfg.fps
        nxt, t_last = 0, time.perf_counter()
        t_end = time.perf_counter() + abs(res.plan[-1] - res.plan[0]) / cfg.continuous_speed_deg_s + cfg.move_timeout_s
        while nxt < len(res.plan):
            t0 = time.perf_counter()
            s = np.asarray(self.robot.read_state(), np.float32)
            if self.abort.is_set():
                res.status = "aborted"
                return
            if self._grasp_lost(s):
                res.status = "grasp_lost"
                return
            if sign * (float(s[ROLL]) - res.plan[nxt]) >= -cfg.arrive_tol_deg or t0 > t_end:
                frame = self.cameras.fresh(cfg.camera, 1)[-1]
                angle = float(self.robot.read_state()[ROLL])
                t1 = time.perf_counter()
                res.stops.append(SweepStop(nxt, res.plan[nxt], angle, (t0 - t_last) * 1e3, 0.0, (t1 - t0) * 1e3,
                                           False, float(s[GRIPPER])))
                self._accept(res, frame, angle, on_frame)
                nxt, t_last = nxt + 1, t1
                continue
            cmd[ROLL] = cmd[ROLL] + float(np.clip(res.plan[-1] - cmd[ROLL], -step, step))
            self._tick(cmd, t0)
