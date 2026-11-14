"""loop_control.LoopController: the full sort cycle and its recovery paths on a fake arm with a fake clock."""

import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy" / "smolvla"))
import auto_control as A  # noqa: E402
import loop_control as LC  # noqa: E402
import smolvla_control as S  # noqa: E402


class Clock:
    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now

    def time(self):
        return self.now

    def strftime(self, f):
        return "test"

    def sleep(self, s):
        import time
        time.sleep(min(s, 0.01))


class FakeArm:
    def read(self):
        return np.zeros(6)

    def send(self, t):
        pass


class FakeBase:
    """Minimal pi05_control.Controller + SmolController: home -> idle, start -> running/pick."""

    def __init__(self, arm, cameras, policy, *a, **kw):
        self.arm, self.cameras, self.policy, self.fps = arm, cameras, policy, 30
        self.phase, self.stage, self.bowl, self.message = "idle", "pick", None, ""
        self.cmd, self.measured = np.zeros(6), np.zeros(6)
        self.lock, self.bowls, self.brick = threading.RLock(), ["yellow", "purple", "orange"], ""

    def _replan_now(self):
        pass

    def _handle(self, c):
        if c in ("q", "stop"):
            self.phase = "stopped"
        elif c == "home":
            self.phase, self.stage = "homing", "pick"
        elif c in ("start", "space") and self.phase not in ("running", "readying"):
            self.phase, self.stage = "running", getattr(self, "start_stage", "pick")

    def tick(self):
        if self.phase == "homing":
            self.phase = "idle"

    def _step(self, target):
        self.cmd = np.asarray(target, float).copy()
        self.measured = self.cmd.copy()
        return self.measured

    def snapshot(self):
        return {"phase": self.phase}


class StubInspector:
    def __init__(self, outcome="cross", bowl="purple", status="ok"):
        self.r = SimpleNamespace(outcome=outcome, bowl=bowl, reason="r", status=status, attempts=1, frames=5,
                                 early_stop=True, seconds=1.2, saved=[])
        self.insp, self.listener, self.cameras, self.done = SimpleNamespace(detector=None), None, None, threading.Event()

    def inspect(self, arm, start_cmd):
        self.done.set()
        return self.r

    def stop(self):
        pass


@pytest.fixture
def build(monkeypatch, tmp_path):
    clock = Clock()

    def make(inspector=None):
        insp = inspector or StubInspector()
        monkeypatch.setattr(LC, "time", clock)
        monkeypatch.setattr(LC.LCFG, "log_dir", str(tmp_path))
        monkeypatch.setattr(S, "_manual_make_controller_class", lambda b: b, raising=False)
        monkeypatch.setattr(S, "BRING", "bring the brick to the inspection camera")
        fake_ai = types.ModuleType("auto_inspect")
        fake_ai.make_auto_inspector = lambda *a, **kw: insp
        fake_hw = types.ModuleType("dlsj.loop.hardware")
        fake_hw.StreamCameras = lambda c: c
        monkeypatch.setitem(sys.modules, "auto_inspect", fake_ai)
        monkeypatch.setitem(sys.modules, "dlsj.loop.hardware", fake_hw)
        A.CFG.__dict__.update(inspect_device="NPU", roi=None, inspect_frames=12, inspect_range=180.0,
                              inspect_speed=60.0)
        cls = LC.make_loop_controller_class(FakeBase)
        c = cls(FakeArm(), {"wrist": SimpleNamespace(frame=None), "top": SimpleNamespace(frame=None),
                            "inspect": SimpleNamespace(frame=None)}, None)
        c._perc_stop.set()
        return c, clock, insp
    return make


def run(c, clock, seconds, grip=None, cmd_grip=None, pose=None):
    for _ in range(int(seconds * 30)):
        clock.now += 1 / 30
        if grip is not None:
            c.measured[5] = grip
        if cmd_grip is not None:
            c.cmd[5] = cmd_grip
        if pose is not None:
            c.measured[:5] = pose
        c.tick()


def see_bricks(c, clock, n, colour="blue"):
    t = clock.now
    with c.lock:
        c.perc.update(t=t, target={"colour": colour, "box": [1, 2, 3, 4], "conf": 0.8} if n else None)
        for k in range(12):
            c._count_hist.append((t + k * 0.25, n))


def test_full_cycle_then_done(build):
    c, clock, insp = build()
    c.loop_command("start")
    run(c, clock, 0.1)
    assert c.L["step"] == "survey"
    run(c, clock, 1.1)
    see_bricks(c, clock, 2, "blue")
    clock.now += 3.0
    run(c, clock, 0.1)
    assert c.L["step"] == "pick" and c.brick == "blue" and c.stage == "pick"
    run(c, clock, 1.0, grip=30.0, cmd_grip=30.0)
    assert c.L["step"] == "pick"
    run(c, clock, 1.0, grip=5.6, cmd_grip=0.0)  # squeezing at brick width
    assert c.L["step"] == "bring" and c.stage == "bring"
    with c.lock:
        c.perc["inspect_seen"] = {"t": clock.now + 10, "conf": 0.7, "close": True}
    run(c, clock, 5.0, grip=5.6, cmd_grip=0.0, pose=np.ones(5))
    assert insp.done.wait(2.0)
    for _ in range(100):
        if c.phase != "inspecting":
            break
        threading.Event().wait(0.01)
    run(c, clock, 0.1, grip=5.6, cmd_grip=0.0)
    assert c.L["step"] == "place" and c.bowl == "purple" and c.stage == "place"
    run(c, clock, 1.5, grip=30.0, cmd_grip=30.0)
    run(c, clock, 1.0, grip=30.0, cmd_grip=30.0)
    assert c.L["tallies"] == {"purple": 1} and c.L["cycles"][-1]["result"] == "sorted"
    # no trip home: it holds above the plate and looks from there
    assert c.L["step"] == "survey" and c.L["survey_from"] == "plate" and c.phase == "holding"
    run(c, clock, 1.1)
    see_bricks(c, clock, 0)
    clock.now += 3.0
    run(c, clock, 0.1)
    # nothing seen from up there: back home (the fake arm arrives at once), and only the view from home may end it
    assert c.L["step"] in ("home", "survey") and c.L["survey_from"] == "home" and c.L["mode"] == "running"
    run(c, clock, 0.2)
    assert c.L["step"] == "survey" and c.L["survey_from"] == "home"
    run(c, clock, 1.1)
    see_bricks(c, clock, 0)
    clock.now += 3.0
    run(c, clock, 0.1)
    assert c.L["mode"] == "done" and c.L["step"] == "done"


def test_pick_timeout_goes_home_and_retries(build):
    c, clock, _ = build()
    c.loop_command("start")
    run(c, clock, 1.2)
    see_bricks(c, clock, 1)
    clock.now += 3.0
    run(c, clock, 0.1)
    assert c.L["step"] == "pick"
    run(c, clock, LC.LCFG.pick_timeout + 1, grip=1.0, cmd_grip=0.0)  # closing on nothing
    assert c.L["attempts"] == 1 and c.L["cycles"][-1]["result"] == "pick timeout"
    assert c.L["step"] in ("home", "survey")


def test_drop_while_bringing_goes_home(build):
    c, clock, _ = build()
    c.loop_command("start")
    run(c, clock, 1.2)
    see_bricks(c, clock, 1)
    clock.now += 3.0
    run(c, clock, 0.1)
    run(c, clock, 1.0, grip=5.6, cmd_grip=0.0)
    assert c.L["step"] == "bring"
    run(c, clock, 1.0, grip=1.0, cmd_grip=0.0)
    assert c.L["cycles"][-1]["result"] == "dropped while bringing" and c.L["step"] in ("home", "survey")


def test_operator_stop_pauses_and_resume_restarts_from_home(build):
    c, clock, _ = build()
    c.loop_command("start")
    run(c, clock, 1.2)
    see_bricks(c, clock, 1)
    clock.now += 3.0
    run(c, clock, 0.1)
    c._handle("q")  # Q from the console
    assert c.L["mode"] == "paused" and c.phase == "stopped"
    run(c, clock, 1.0)
    assert c.L["step"] == "paused"
    c.loop_command("resume")
    assert c.L["mode"] == "running" and c.L["step"] == "home" and c.phase == "homing"


def test_unclear_colour_pauses(build):
    c, clock, _ = build()
    c.loop_command("start")
    run(c, clock, 1.2)
    t = clock.now
    with c.lock:
        c.perc.update(t=t, target=None)
        for k in range(12):
            c._count_hist.append((t + k * 0.25, 1))
    clock.now += 9.0
    run(c, clock, 0.1)
    assert c.L["mode"] == "paused"


def test_snapshot_has_loop_and_perception(build):
    c, clock, _ = build()
    snap = c.snapshot()
    assert {"mode", "step", "grasp", "cycles"} <= set(snap["loop"]) and "table" in snap["perception"]


def _to_survey_from_plate(c, clock, insp):
    c.loop_command("start")
    run(c, clock, 1.2)
    see_bricks(c, clock, 2)
    clock.now += 3.0
    run(c, clock, 0.1)
    run(c, clock, 1.0, grip=5.6, cmd_grip=0.0)
    with c.lock:
        c.perc["inspect_seen"] = {"t": clock.now + 10, "conf": 0.7, "close": True}
    run(c, clock, 5.0, grip=5.6, cmd_grip=0.0, pose=np.ones(5))
    assert insp.done.wait(2.0)
    for _ in range(100):
        if c.phase != "inspecting":
            break
        threading.Event().wait(0.01)
    run(c, clock, 0.1, grip=5.6, cmd_grip=0.0)
    run(c, clock, 2.5, grip=30.0, cmd_grip=30.0)
    assert c.L["step"] == "survey" and c.L["survey_from"] == "plate"


def test_next_pick_straight_from_the_plate(build):
    c, clock, insp = build()
    _to_survey_from_plate(c, clock, insp)
    homes = []
    orig = c._handle
    c._handle = lambda k: (homes.append(k), orig(k))[1]
    run(c, clock, 1.1)
    see_bricks(c, clock, 1, "yellow")
    clock.now += 3.0
    run(c, clock, 0.1)
    assert c.L["step"] == "pick" and c.stage == "pick" and c.phase == "running" and c.brick == "yellow"
    assert "home" not in homes and "start" not in homes and c.bowl is None and c.L["cycle"] == 2
