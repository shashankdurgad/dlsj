"""Runs one skill: observation -> pi0.5 action chunk -> 30 Hz joint targets, with safety and termination.

Control thread (one tick every 1/fps s):
    read joint state -> termination check -> abort check
    -> if the action queue is at or below `request_threshold` x chunk length and nothing is in flight:
         build an Observation (state + latest camera frames) and hand it to the inference worker
    -> pop the next joint target, move the command toward it by at most `max_step` per joint
       (Desmond's `step_toward`), send it.  An empty queue holds the last command ("starved" tick).

Inference worker (background thread, at most one request in flight), the same idea as Physical AI Studio's
`AsyncExecution(request_threshold=0.5)` / LeRobot's async inference: while the arm executes the second half
of chunk k, chunk k+1 is computed from a newer observation. When it lands, it replaces the queue, and the
actions whose time already passed while the model was thinking (ticks since the observation) are dropped,
so the new chunk starts "now". Real-Time Chunking (RTC) guidance, if any, lives inside the policy back-end;
this executor only needs `predict_chunk`. `async_inference=False` runs inference inline (the arm holds while
the model thinks): the baseline for the OPTIMIZE table (`starved_ticks` shows the difference).
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from dlsj.contracts import Observation, Policy
from dlsj.loop.poses import ARM, GRIPPER
from dlsj.loop.skills import Outcome, Skill, SkillLibrary, TerminationMonitor
from dlsj.loop.telemetry import RunLog

try:  # reuse Desmond's helper; teleop_web imports cv2 at module level, so fall back when it is missing
    from dlsj.teleop_web import step_toward
except ImportError:  # pragma: no cover - only without OpenCV

    def step_toward(cmd: np.ndarray, target: np.ndarray, max_step) -> np.ndarray:
        """Move `cmd` toward `target`, no joint by more than `max_step`."""
        return cmd + np.clip(target - cmd, -max_step, max_step)


log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------- hardware interfaces
class Robot(Protocol):
    """Joint-space SO-101 follower in dataset units (degrees, gripper 0-100), order = `poses.MOTORS`."""

    def read_state(self) -> np.ndarray: ...

    def send(self, target: np.ndarray) -> None: ...

    def read_gripper_current(self) -> float | None: ...


class Cameras(Protocol):
    names: list[str]

    def latest(self) -> dict[str, np.ndarray]:
        """Latest frame per camera, HxWx3 uint8 RGB (non-blocking)."""
        ...

    def fresh(self, name: str, n: int, timeout_s: float = 2.0) -> list[np.ndarray]:
        """`n` new, distinct frames from one camera captured after the call (RGB)."""
        ...


# ----------------------------------------------------------------------------- config / results
@dataclass
class ExecConfig:
    fps: float = 30.0
    max_speed_deg_s: float = 90.0  # arm joints, per-tick cap = this / fps
    gripper_max_speed: float = 200.0  # gripper units/s
    request_threshold: float = 0.5  # request the next chunk when queue <= this fraction of the chunk
    async_inference: bool = True
    skip_stale_actions: bool = True  # drop actions whose time passed while the model ran
    max_actions_per_chunk: int | None = None  # execute at most this many actions of each chunk (n_action_steps)
    camera_names: tuple[str, ...] | None = None  # cameras fed to the policy (None = all)
    ramp_speed_deg_s: float = 30.0  # homing / move_to speed, like teleop_web --ramp-speed


@dataclass
class SkillResult:
    skill: str
    instruction: str
    outcome: Outcome
    duration_s: float = 0.0
    ticks: int = 0
    chunks: int = 0
    starved_ticks: int = 0
    stall_ms: float = 0.0  # time the arm had no fresh target: starved ticks + (sync mode) inline inference
    dropped_actions: int = 0
    final_state: np.ndarray | None = None
    termination: dict = field(default_factory=dict)
    error: str | None = None

    def brief(self) -> dict:
        return {"skill": self.skill, "outcome": self.outcome.value, "duration_s": round(self.duration_s, 3),
                "ticks": self.ticks, "chunks": self.chunks, "starved_ticks": self.starved_ticks,
                "stall_ms": round(self.stall_ms, 1), "dropped_actions": self.dropped_actions, "termination": self.termination, "error": self.error}


# ----------------------------------------------------------------------------- inference worker
@dataclass
class _Request:
    obs: Observation
    instruction: str
    tick: int
    gen: int


@dataclass
class _Result:
    chunk: np.ndarray | None
    tick: int
    gen: int
    ms: float
    error: BaseException | None = None


class InferenceWorker:
    """One background thread that runs `policy.predict_chunk`, at most one request in flight."""

    def __init__(self, policy: Policy):
        self.policy = policy
        self._req: queue.Queue[_Request | None] = queue.Queue(maxsize=1)
        self._res: queue.Queue[_Result] = queue.Queue()
        self.busy = False
        self._thread = threading.Thread(target=self._run, name="inference", daemon=True)
        self._thread.start()

    def submit(self, req: _Request) -> None:
        self.busy = True
        self._req.put(req)

    def poll(self) -> _Result | None:
        try:
            r = self._res.get_nowait()
        except queue.Empty:
            return None
        self.busy = False
        return r

    def wait(self, timeout: float) -> _Result | None:
        try:
            r = self._res.get(timeout=timeout)
        except queue.Empty:
            return None
        self.busy = False
        return r

    def close(self) -> None:
        self._req.put(None)

    def _run(self) -> None:
        while (req := self._req.get()) is not None:
            t0 = time.perf_counter()
            try:
                chunk = np.asarray(self.policy.predict_chunk(req.obs, req.instruction), dtype=np.float32)
                err = None
            except BaseException as e:  # noqa: BLE001 - handed to the control thread
                chunk, err = None, e
            self._res.put(_Result(chunk, req.tick, req.gen, (time.perf_counter() - t0) * 1000.0, err))


# ----------------------------------------------------------------------------- executor
class SkillExecutor:
    def __init__(self, robot: Robot, cameras: Cameras, policy: Policy, library: SkillLibrary,
                 cfg: ExecConfig | None = None, poses: dict[str, np.ndarray] | None = None,
                 runlog: RunLog | None = None):
        self.robot, self.cameras, self.policy, self.library = robot, cameras, policy, library
        self.cfg = cfg or ExecConfig()
        self.poses = poses or {}
        self.log = runlog or RunLog(None)
        self.period = 1.0 / self.cfg.fps
        self.max_step = np.full(6, self.cfg.max_speed_deg_s / self.cfg.fps, dtype=np.float32)
        self.max_step[GRIPPER] = self.cfg.gripper_max_speed / self.cfg.fps
        self.cmd: np.ndarray | None = None
        self.worker = InferenceWorker(policy) if self.cfg.async_inference else None
        self._gen = 0
        self.live: dict = {}  # latest status for the status page (replaced atomically each tick)

    def close(self) -> None:
        if self.worker:
            self.worker.close()

    # ------------------------------------------------------------------ helpers
    def _state(self) -> np.ndarray:
        s = np.asarray(self.robot.read_state(), dtype=np.float32)
        if self.cmd is None:
            self.cmd = s.copy()
        return s

    def perceive(self, state: np.ndarray, **meta) -> Observation:
        t0 = time.perf_counter()
        images = self.cameras.latest()
        if self.cfg.camera_names is not None:
            images = {k: v for k, v in images.items() if k in self.cfg.camera_names}
        obs = Observation(state=state.astype(np.float32), images=images, timestamp=time.time())
        self.log.timing("perceive", (time.perf_counter() - t0) * 1000.0, **meta)
        return obs

    def _sleep_rest(self, t0: float) -> None:
        dt = time.perf_counter() - t0
        if dt < self.period:
            time.sleep(self.period - dt)

    def _send(self, target: np.ndarray | None) -> None:
        if target is not None:
            self.cmd = step_toward(self.cmd, target, self.max_step)
        self.robot.send(self.cmd.copy())

    # ------------------------------------------------------------------ skills
    def run(self, name: str, abort: threading.Event | None = None, **params) -> SkillResult:
        skill: Skill = self.library[name]
        instruction = skill.render(**params)
        abort = abort or threading.Event()
        cfg = self.cfg
        self._gen += 1
        gen = self._gen
        self.policy.reset()
        monitor = TerminationMonitor(skill.termination, cfg.fps, self.poses)
        res = SkillResult(name, instruction, Outcome.TIMEOUT)
        q: deque[np.ndarray] = deque()
        chunk_len, inflight, tick = 0, False, 0
        tick_ms: list[float] = []
        t_start = time.perf_counter()
        log.info("skill %s: %r", name, instruction)
        self.log.event("skill_start", skill=name, instruction=instruction)

        def install(r: _Result) -> None:
            nonlocal chunk_len, q
            if r.error is not None:
                raise r.error
            chunk = r.chunk
            if chunk.ndim != 2 or chunk.shape[1] != 6 or len(chunk) == 0:
                raise ValueError(f"policy returned a chunk of shape {chunk.shape}, expected (n, 6)")
            if cfg.max_actions_per_chunk:
                chunk = chunk[: cfg.max_actions_per_chunk]
            stale = max(0, tick - r.tick) if cfg.skip_stale_actions else 0
            drop = min(stale, len(chunk) - 1)
            res.dropped_actions += drop
            res.chunks += 1
            chunk_len = len(chunk)
            q = deque(chunk[drop:])
            self.log.timing("inference", r.ms, skill=name, chunk=res.chunks, n=len(chunk), dropped=drop,
                            mode="async" if self.worker else "sync")

        try:
            while True:
                t0 = time.perf_counter()
                state = self._state()
                if abort.is_set():
                    res.outcome = Outcome.ABORTED
                    break
                out = monitor.update(state, t0)
                if out is not None:
                    res.outcome = out
                    break
                if self.worker is not None:
                    r = self.worker.poll()
                    if r is not None:
                        inflight = False
                        if r.gen == gen:
                            install(r)
                if not inflight and len(q) <= cfg.request_threshold * chunk_len:
                    obs = self.perceive(state, skill=name)
                    if self.worker is not None:
                        self.worker.submit(_Request(obs, instruction, tick, gen))
                        inflight = True
                    else:
                        t_inf = time.perf_counter()
                        try:
                            chunk = np.asarray(self.policy.predict_chunk(obs, instruction), dtype=np.float32)
                            ms = (time.perf_counter() - t_inf) * 1000.0
                            res.stall_ms += ms  # the arm held still while the model ran
                            install(_Result(chunk, tick, gen, ms))
                        except BaseException as e:  # noqa: BLE001
                            install(_Result(None, tick, gen, 0.0, e))
                target = q.popleft() if q else None
                if target is None:
                    res.starved_ticks += 1
                    res.stall_ms += self.period * 1000.0
                self._send(target)
                tick += 1
                tick_ms.append((time.perf_counter() - t0) * 1000.0)
                self.live = {"skill": name, "instruction": instruction, "tick": tick, "queue": len(q),
                             "inflight": inflight, **monitor.last_status}
                self._sleep_rest(t0)
        except Exception as e:  # noqa: BLE001 - policy/robot failure ends the skill, the controller decides
            log.exception("skill %s failed", name)
            res.outcome, res.error = Outcome.ERROR, f"{type(e).__name__}: {e}"
        finally:
            self._gen += 1  # any chunk still in flight belongs to a finished skill and is ignored
            if inflight and self.worker is not None:
                self.worker.wait(timeout=5.0)  # drain so the next skill starts with an idle worker

        res.duration_s = time.perf_counter() - t_start
        res.ticks = tick
        res.final_state = self.robot.read_state()
        res.termination = monitor.last_status
        if tick_ms:
            a = np.asarray(tick_ms)
            self.log.timing("tick", float(np.median(a)), skill=name, p90=round(float(np.percentile(a, 90)), 3),
                            n=len(a), max=round(float(a.max()), 3))
        self.log.timing(f"skill.{name}", res.duration_s * 1000.0, outcome=res.outcome.value)
        self.log.timing("stall", res.stall_ms, skill=name, mode="async" if self.worker else "sync")
        self.log.event("skill_end", **res.brief())
        self.live = {"skill": name, "outcome": res.outcome.value}
        return res

    # ------------------------------------------------------------------ scripted moves
    def move_to(self, pose: np.ndarray, abort: threading.Event | None = None, timeout_s: float = 20.0,
                tol: float = 0.5) -> bool:
        """Ramp the command to `pose` at `ramp_speed_deg_s` (Desmond's homing). True when reached."""
        step = np.full(6, self.cfg.ramp_speed_deg_s / self.cfg.fps, dtype=np.float32)
        step[GRIPPER] = self.cfg.gripper_max_speed / self.cfg.fps
        self._state()
        t_end = time.perf_counter() + timeout_s
        pose = np.asarray(pose, dtype=np.float32)
        while time.perf_counter() < t_end and not (abort and abort.is_set()):
            t0 = time.perf_counter()
            self._state()
            self.cmd = step_toward(self.cmd, pose, step)
            self.robot.send(self.cmd.copy())
            if float(np.max(np.abs(self.cmd - pose))) < tol:
                return True
            self._sleep_rest(t0)
        return False

    def set_gripper(self, value: float, abort: threading.Event | None = None, timeout_s: float = 3.0) -> bool:
        self._state()
        pose = self.cmd.copy()
        pose[GRIPPER] = value
        return self.move_to(pose, abort, timeout_s)

    def hold_until_still(self, still_ticks: int = 10, still_deg_s: float = 2.0, timeout_s: float = 3.0,
                         abort: threading.Event | None = None) -> bool:
        """Hold the current command until the measured arm joints are still (the `Moving` = 0 check)."""
        prev, run = None, 0
        t_end = time.perf_counter() + timeout_s
        while time.perf_counter() < t_end and not (abort and abort.is_set()):
            t0 = time.perf_counter()
            s = self._state()
            self.robot.send(self.cmd.copy())
            if prev is not None:
                vel = float(np.max(np.abs(s[ARM] - prev[ARM]))) * self.cfg.fps
                run = run + 1 if vel < still_deg_s else 0
                if run >= still_ticks:
                    return True
            prev = s
            self._sleep_rest(t0)
        return False
