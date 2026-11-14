"""auto_control.AutoController state machine on a fake base controller + stub inspector (no robot, no models)."""

import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy" / "smolvla"))
import auto_control as A  # noqa: E402
import smolvla_control as S  # noqa: E402


class FakeArm:
    def __init__(self):
        self.q, self.sent = np.zeros(6), []

    def read(self):
        return self.q.copy()

    def send(self, t):
        self.sent.append(np.asarray(t))


class FakeBase:
    def __init__(self, arm, cameras, policy, *a, **kw):
        self.arm, self.cameras, self.policy, self.fps = arm, cameras, policy, 30
        self.phase, self.stage, self.bowl, self.message, self.cmd = "idle", "pick", None, "", None
        self.lock, self.replans = threading.RLock(), 0
        self.bowls = ["yellow", "purple", "orange"]

    def _replan_now(self):
        self.replans += 1

    def _handle(self, c):
        if c in ("q", "stop"):
            self.phase = "stopped"
        elif c.startswith("bowl"):
            self.bowl = self.bowls[int(c[4:]) - 1]
        elif c in ("start", "space") and self.phase not in ("running", "readying"):
            self.stage, self.phase, self.cmd = "pick", "running", self.arm.read()

    def snapshot(self):
        return {"phase": self.phase}


class StubInspector:
    def __init__(self, result, delay=0.0):
        self.result, self.delay, self.calls, self.abort = result, delay, 0, threading.Event()

    def inspect(self, arm, start_cmd):
        self.calls += 1
        t_end = time.time() + self.delay
        while time.time() < t_end:
            if self.abort.is_set():
                return SimpleNamespace(**{**vars(self.result), "status": "aborted"})
            time.sleep(0.01)
        return self.result

    def stop(self):
        self.abort.set()


def res(outcome="cross", bowl="purple", status="ok"):
    return SimpleNamespace(outcome=outcome, bowl=bowl, reason="r", status=status, attempts=1, frames=6,
                           early_stop=True, seconds=1.0, saved=[])


@pytest.fixture
def make_ctl(monkeypatch):
    def build(inspector, bring="bring the brick to the inspection camera"):
        monkeypatch.setattr(S, "BRING", bring)
        monkeypatch.setattr(S, "_manual_make_controller_class", S.make_controller_class, raising=False)
        fake_ai = types.ModuleType("auto_inspect")
        fake_ai.make_auto_inspector = lambda *a, **kw: inspector
        fake_hw = types.ModuleType("dlsj.loop.hardware")
        fake_hw.StreamCameras = lambda c: c
        monkeypatch.setitem(sys.modules, "auto_inspect", fake_ai)
        monkeypatch.setitem(sys.modules, "dlsj.loop.hardware", fake_hw)
        A.CFG.__dict__.update(inspect_device="NPU", roi=None, inspect_frames=12, inspect_range=180.0,
                              inspect_speed=60.0)
        cls = A.make_auto_controller_class(FakeBase)
        return cls(FakeArm(), {"wrist": 0, "top": 0, "inspect": 0}, None)
    return build


def wait(pred, t=3.0):
    t_end = time.time() + t
    while time.time() < t_end and not pred():
        time.sleep(0.01)
    assert pred()


def test_pick_bring_inspect_place(make_ctl):
    insp = StubInspector(res("cross", "purple"))
    c = make_ctl(insp)
    c._handle("space")
    c._handle("m")
    assert c.stage == "bring" and insp.calls == 0
    c._handle("m")
    wait(lambda: c.stage == "place")
    assert (c.bowl, c.phase, insp.calls) == ("purple", "running", 1)
    assert c.snapshot()["inspection"]["outcome"] == "cross"


def test_keys_ignored_while_spinning_and_q_aborts(make_ctl):
    insp = StubInspector(res(), delay=2.0)
    c = make_ctl(insp)
    c._handle("space"); c._handle("m"); c._handle("m")
    assert c.phase == "inspecting"
    for k in ("space", "m", "bowl1"):
        c._handle(k)
    assert c.phase == "inspecting" and c.bowl is None
    c._handle("q")
    wait(lambda: c.phase == "stopped")
    assert c.stage == "pick" and c.bowl is None and c.arm.sent  # held in place


def test_grasp_lost_holds(make_ctl):
    c = make_ctl(StubInspector(res("unknown", "purple", "grasp_lost")))
    c._handle("space"); c._handle("m"); c._handle("m")
    wait(lambda: c.phase == "stopped")
    assert "dropped" in c.message and c.bowl is None


def test_next_brick_is_inspected_again(make_ctl):
    insp = StubInspector(res("none", "yellow"))
    c = make_ctl(insp)
    c._handle("space"); c._handle("m"); c._handle("m")
    wait(lambda: c.stage == "place")
    c._handle("q"); c._handle("space")
    assert c.bowl is None
    c._handle("m"); c._handle("m")
    wait(lambda: insp.calls == 2 and c.stage == "place")


def test_manual_override_skips_inspection(make_ctl):
    insp = StubInspector(res())
    c = make_ctl(insp)
    c._handle("space"); c._handle("bowl3"); c._handle("m"); c._handle("m")
    assert (c.stage, c.bowl, insp.calls) == ("place", "orange", 0)


def test_no_bring_stage(make_ctl):
    insp = StubInspector(res("circle", "orange"))
    c = make_ctl(insp, bring="")
    c._handle("space"); c._handle("m")
    wait(lambda: c.stage == "place")
    assert c.bowl == "orange"
