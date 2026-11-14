import json
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from dlsj.deploy.pi05_control import Controller, make_handler
from dlsj.deploy.pi05_runtime import PICK


class FakeArm:
    def __init__(self):
        self.pos = np.zeros(6)
        self.sent: list[np.ndarray] = []

    def read(self):
        return self.pos.copy()

    def send(self, target):
        self.sent.append(np.asarray(target, float).copy())
        self.pos = self.sent[-1].copy()  # perfect tracking


class FakeCam:
    def __init__(self):
        self.frame = np.full((480, 640, 3), 100, np.uint8)
        self.jpeg, self.seq, self.cond, self.error = b"jpg", 0, threading.Condition(), None

    def info(self):
        return {"name": "fake", "fps": 30.0, "error": None}


class FakePolicy:
    """Chunk heading to +30 on every joint; records prompts. `delay` simulates inference time."""

    def __init__(self, delay=0.0, target=30.0):
        self.delay, self.target, self.prompts, self.calls = delay, target, [], 0
        self.release = threading.Event()
        self.release.set()

    def __call__(self, state, images, task):
        assert set(images) == {"wrist", "top", "inspect"} and images["top"].shape == (480, 640, 3)
        self.prompts.append(task)
        self.calls += 1
        self.release.wait(5)
        time.sleep(self.delay)
        return np.linspace(state, np.full(6, self.target), 50)


def make(policy=None, sync=True, **kw):
    cams = {c: FakeCam() for c in ("wrist", "top", "inspect")}
    return Controller(FakeArm(), cams, policy or FakePolicy(), sync=sync, **kw)


def run_ticks(ctl, n):
    for _ in range(n):
        ctl.tick()


def test_start_drives_the_arm_with_rate_limited_targets():
    ctl = make(max_step=1.0, max_step_gripper=2.0)
    run_ticks(ctl, 5)
    assert ctl.arm.sent == []  # idle: nothing sent before Space
    ctl.command("start")
    run_ticks(ctl, 40)
    sent = np.array(ctl.arm.sent)
    assert len(sent) >= 35
    steps = np.abs(np.diff(np.vstack([np.zeros(6), sent]), axis=0))
    assert steps[:, :5].max() <= 1.0 + 1e-9 and steps[:, 5].max() <= 2.0 + 1e-9
    assert sent[-1, 0] > 10  # actually moving toward the policy's targets
    assert ctl.policy.prompts[0] == PICK


def test_q_holds_present_position_and_drops_queued_actions():
    ctl = make()
    ctl.command("start")
    run_ticks(ctl, 10)
    assert ctl.queue
    here = ctl.arm.pos.copy()
    ctl.command("q")  # immediate, no tick needed
    assert ctl.phase == "stopped" and not ctl.queue
    np.testing.assert_allclose(ctl.arm.sent[-1], here)  # last command = hold where it is
    n = len(ctl.arm.sent)
    run_ticks(ctl, 20)
    assert len(ctl.arm.sent) == n  # nothing more is sent while stopped


def test_in_flight_chunk_is_discarded_after_q():
    policy = FakePolicy()
    policy.release.clear()  # inference "hangs" until released
    ctl = make(policy, sync=False)
    ctl.command("start")
    ctl.tick()  # requests a chunk in a worker thread
    deadline = time.time() + 2
    while policy.calls == 0 and time.time() < deadline:
        time.sleep(0.01)
    ctl.command("q")
    n = len(ctl.arm.sent)
    policy.release.set()  # the late chunk arrives after the stop
    time.sleep(0.2)
    run_ticks(ctl, 10)
    assert not ctl.queue and len(ctl.arm.sent) == n


def test_prompt_switches_at_the_mark_with_the_chosen_plate():
    ctl = make()
    ctl.command("start")
    run_ticks(ctl, 3)
    ctl.command("mark")
    run_ticks(ctl, 1)
    assert ctl.stage == "pick" and "plate" in ctl.message  # no plate chosen yet
    ctl.command("bowl3")  # orange
    ctl.command("mark")
    run_ticks(ctl, 3)
    assert ctl.stage == "place" and ctl.policy.prompts[-1] == "place the brick in the orange bowl"
    ctl.command("bowl1")  # change of mind mid-place: replan with the new colour
    run_ticks(ctl, 3)
    assert ctl.policy.prompts[-1] == "place the brick in the yellow bowl"


def test_stale_actions_are_skipped_by_inference_latency():
    ctl = make(FakePolicy(delay=0.2), fps=30)  # 0.2 s at 30 fps -> first 6 actions already stale
    ctl.command("start")
    ctl.tick()
    assert len(ctl.queue) in (50 - 6 - 1, 50 - 7 - 1)  # ceil(0.2 s x 30) = 6 (+1 for overhead) skipped, 1 sent
    snap = ctl.snapshot()
    assert snap["latency_ms"] >= 200 and snap["phase"] == "running"


def test_inference_error_stops_and_holds():
    class Boom(FakePolicy):
        def __call__(self, *a):
            raise RuntimeError("GPU lost")

    ctl = make(Boom())
    ctl.command("start")
    run_ticks(ctl, 3)
    assert ctl.phase == "stopped" and "GPU lost" in ctl.message


def test_http_console():
    ctl = make()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(ctl.cameras, ctl))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        assert "Stop now" in urllib.request.urlopen(base + "/").read().decode()
        for c in ("bowl2", "start"):
            urllib.request.urlopen(urllib.request.Request(base + "/ctl/" + c, method="POST"))
        run_ticks(ctl, 3)
        urllib.request.urlopen(urllib.request.Request(base + "/ctl/q", method="POST"))
        st = json.loads(urllib.request.urlopen(base + "/ctl/state").read())
        assert st["phase"] == "stopped" and st["bowl"] == "purple"
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(urllib.request.Request(base + "/ctl/rm_rf", method="POST"))
    finally:
        srv.shutdown()
