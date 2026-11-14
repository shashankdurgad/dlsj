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
    # the fake chunk starts exactly at the present state, so replacing it every tick ("none") would never move;
    # a real policy's chunk leads the arm and stale actions are skipped. Rate limiting is blend-independent.
    ctl = make(max_step=1.0, max_step_gripper=2.0, blend="ensemble")
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
    assert ctl.chunks
    here = ctl.arm.pos.copy()
    ctl.command("q")  # immediate, no tick needed
    assert ctl.phase == "stopped" and not ctl.chunks
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
    assert not ctl.chunks and len(ctl.arm.sent) == n


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


class ClockPolicy(FakePolicy):
    """Action j of every chunk = j on all joints, so a sent value says which chunk index was used."""

    def __call__(self, state, images, task):
        super().__call__(state, images, task)
        return np.repeat(np.arange(50.0)[:, None], 6, 1)


def test_actions_stale_by_inference_time_are_skipped():
    policy = ClockPolicy()
    policy.release.clear()
    ctl = make(policy, sync=False, max_step=100, blend="none")
    ctl.command("start")
    ctl.tick()  # tick 0: request, observation taken now
    for _ in range(6):  # inference takes 6 ticks
        ctl.tick()
    assert ctl.arm.sent == []  # nothing planned yet: hold
    policy.release.set()
    deadline = time.time() + 2
    while not ctl.chunks and time.time() < deadline:
        time.sleep(0.01)
    ctl.tick()  # tick 7 -> chunk index 7
    assert ctl.arm.sent[0][0] == 7.0
    for _ in range(100):  # inference runs on a thread: give it a moment to record its latency
        if ctl.lat_ms:
            break
        time.sleep(0.01)
    assert len(ctl.lat_ms) > 0 and ctl.snapshot()["latency_ms"] is not None  # the fake policy can round to 0.0 ms


def test_ensemble_aligns_chunks_by_time_and_favours_the_oldest():
    ctl = make(blend="ensemble", ensemble_k=0.5)
    ctl.t = 10
    ctl._add_chunk(4, np.repeat(np.arange(4.0, 54.0)[:, None], 6, 1))  # value = the tick it is meant for
    ctl._add_chunk(8, np.repeat(np.arange(8.0, 58.0)[:, None], 6, 1))
    np.testing.assert_allclose(ctl._planned(10), 10.0)  # both agree on tick 10 only if aligned by time
    ctl.chunks.clear()
    ctl._add_chunk(10, np.zeros((50, 6)))  # older
    ctl._add_chunk(10, np.ones((50, 6)))  # newer
    w = np.exp(-0.5 * np.arange(2))
    np.testing.assert_allclose(ctl._planned(12), w[1] / w.sum())  # ACT: the oldest chunk weighs most
    assert ctl._planned(12)[0] < 0.5


class JitterPolicy(FakePolicy):
    """Consecutive chunks disagree by 12 units (+6 / -6): what made the arm jitter on the first live run."""

    def __call__(self, state, images, task):
        super().__call__(state, images, task)
        return np.full((50, 6), 6.0 * (-1) ** self.calls)


def _max_jump(blend):
    ctl = make(JitterPolicy(), blend=blend, max_step=100, replan=40, blend_steps=10)
    ctl.command("start")
    run_ticks(ctl, 80)
    assert ctl.policy.calls >= 6
    return float(np.abs(np.diff(np.array(ctl.arm.sent)[:, 0])).max())


def test_blending_smooths_chunk_switches():
    raw = _max_jump("none")
    assert raw >= 12.0  # without blending every switch jumps the whole disagreement
    assert _max_jump("crossfade") <= 12.0 / 11 * 2 + 1e-9  # <= 2 fade steps per tick even when fades overlap
    assert _max_jump("ensemble") <= raw / 2


def test_speed_keys_change_the_step_limit():
    ctl = make(max_step=8.0)
    ctl.command("faster")
    ctl.tick()
    assert ctl.max_step[0] == pytest.approx(10.0)
    ctl.command("slower")
    ctl.command("slower")
    ctl.tick()
    assert ctl.max_step[0] == pytest.approx(6.4)


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


class ShapePolicy(FakePolicy):
    def __call__(self, state, images, task):
        self.shapes = {k: v.shape for k, v in images.items()}
        self.top_right_col = images["top"][:, -1].copy()
        self.prompts.append(task)
        return np.linspace(state, np.full(6, self.target), 50)


def test_training_crop_is_applied_before_the_policy():
    from dlsj.deploy.pi05_control import parse_crops

    crops = parse_crops("top=0,0,560,480")
    assert crops == {"top": (0, 0, 560, 480)}
    policy = ShapePolicy()
    ctl = make(policy, crops=crops)
    top = ctl.cameras["top"].frame
    top[:, 559] = 7  # last column kept by the crop
    top[:, 560:] = 200  # dropped by the crop
    ctl.command("start")
    run_ticks(ctl, 2)
    assert policy.shapes["top"] == (480, 560, 3) and policy.shapes["wrist"] == (480, 640, 3)
    assert (policy.top_right_col == 7).all()


def test_space_eases_to_the_ready_pose_before_the_policy_runs():
    policy = FakePolicy()
    ready = [10.0, -50.0, 40.0, 60.0, 15.0, 30.0]
    ctl = make(policy, ready=ready)
    ctl.command("start")
    run_ticks(ctl, 1)
    assert ctl.phase == "readying" and policy.calls == 0
    for _ in range(60):
        if ctl.phase == "running":
            break
        run_ticks(ctl, 1)
    assert ctl.phase == "running" and np.abs(ctl.arm.pos - np.array(ready)).max() < 4.0  # handed over at the pose
    steps = np.diff(np.array(ctl.arm.sent), axis=0)
    assert np.abs(steps).max() <= ctl.max_step.max() + 1e-9  # rate-limited on the way
    run_ticks(ctl, 3)
    assert policy.calls >= 1


class LaggyArm(FakeArm):
    """Moves only a fraction of the way to each command, like a loaded joint."""

    def send(self, target):
        self.sent.append(np.asarray(target, float).copy())
        self.pos = self.pos + 0.2 * (self.sent[-1] - self.pos)


def test_default_blend_is_newest_chunk():
    assert make().blend == "none"


def test_command_stays_within_the_sync_band_of_the_measured_position():
    import io

    trace = io.StringIO()
    policy = FakePolicy(target=100.0)
    ctl = Controller(LaggyArm(), {c: FakeCam() for c in ("wrist", "top", "inspect")}, policy, sync=True,
                     max_step=8.0, sync_band=5.0, trace=trace)
    ctl.command("start")
    for _ in range(30):
        pos_before = ctl.arm.pos.copy()
        ctl.tick()
        if ctl.arm.sent:
            assert np.abs(ctl.arm.sent[-1] - pos_before).max() <= 5.0 + 1e-9  # never more than the band ahead
    recs = [json.loads(line) for line in trace.getvalue().splitlines()]
    kinds = {r["kind"] for r in recs}
    assert {"key", "infer", "tick"} <= kinds
    tick = next(r for r in recs if r["kind"] == "tick")
    assert {"measured", "target", "cmd", "queued", "gen"} <= set(tick) and len(tick["cmd"]) == 6
    infer = next(r for r in recs if r["kind"] == "infer")
    assert infer["kept"] and len(infer["chunk_head"]) == 10 and infer["prompt"]
