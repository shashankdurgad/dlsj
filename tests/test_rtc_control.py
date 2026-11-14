"""RTC wrapper for the π0.5 controller: graph inputs, delay estimate, hand-off of the plan's tail."""

import collections
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy" / "pi05"))

import rtc_control as R  # noqa: E402


def test_first_chunk_is_unguided():
    d = R.rtc_inputs(None, 7, 15, 5.0, 50, 6)
    assert d[R.PREV_CHUNK_LEFT_OVER].shape == (1, 50, 6) and not d[R.PREV_CHUNK_LEFT_OVER].any()
    assert d[R.RTC_INFERENCE_DELAY] == 0 and d[R.RTC_EXECUTION_HORIZON] == 1
    assert d[R.RTC_MAX_GUIDANCE_WEIGHT] == pytest.approx(1e-8)
    assert d[R.RTC_INFERENCE_DELAY].dtype == np.int64 and d[R.RTC_MAX_GUIDANCE_WEIGHT].dtype == np.float32


def test_tail_is_padded_and_delay_clamped_to_horizon():
    tail = np.arange(20 * 6, dtype=np.float32).reshape(20, 6)
    d = R.rtc_inputs(tail, 30, 15, 5.0, 50, 6)
    prev = d[R.PREV_CHUNK_LEFT_OVER]
    assert np.array_equal(prev[0, :20], tail) and not prev[0, 20:].any()
    assert d[R.RTC_INFERENCE_DELAY] == 15 and d[R.RTC_EXECUTION_HORIZON] == 15
    assert d[R.RTC_MAX_GUIDANCE_WEIGHT] == pytest.approx(5.0)
    assert R.rtc_inputs(np.zeros((60, 6)), 2, 15, 5.0, 50, 6)[R.PREV_CHUNK_LEFT_OVER].shape == (1, 50, 6)


def test_delay_is_ceil_of_worst_recent_latency():
    assert R.delay_actions([], 30, 15) == 0
    assert R.delay_actions([100, 140, 120], 30, 15) == 5  # 0.14 s * 30 = 4.2 -> 5
    assert R.delay_actions([2000], 30, 15) == 15  # clamped to the horizon
    assert R.delay_actions([900] + [100] * 10, 30, 15) == 3  # only the last 10 count


class FakeController:
    """The bits of pi05_control.Controller the RTC subclass relies on."""

    def __init__(self, policy, fps=30):
        self.policy, self.fps = policy, fps
        self.chunks, self.t, self.replan, self.blend = [], 0, 50, "ensemble"
        self.lat_ms = collections.deque(maxlen=30)
        self.requests = []

    def tick(self):
        if self.remaining() <= self.replan:
            self._request()
        self.t += 1

    def remaining(self):
        if not self.chunks:
            return 0
        s, c = self.chunks[-1]
        return max(0, s + len(c) - self.t)

    def _request(self):
        self.requests.append((self.t, self.policy.rtc_next))

    def snapshot(self):
        return {"blend": self.blend}


def test_controller_hands_the_unexecuted_tail_and_uses_the_rtc_threshold():
    class P:
        rtc_next = None
        guidance = 5.0

    cls = R.make_rtc_controller_class(FakeController)
    c = cls(P(), horizon=15)
    assert c.blend == "none"
    c.tick()  # no plan: first request is unguided
    assert c.requests[-1][1][0] is None
    plan = np.arange(50 * 6, dtype=float).reshape(50, 6)
    c.chunks = [(1, plan)]
    c.lat_ms.extend([150.0])  # 4.5 -> 5 actions of delay
    c.requests.clear()
    while not c.requests:
        c.tick()
    t, (tail, delay) = c.requests[-1]
    assert delay == 5 and c.replan == 20
    assert c.remaining() <= 20 and t == 1 + 50 - 20  # asked when fewer than horizon + delay actions remained
    assert np.array_equal(tail, plan[t - 1:])  # the plan from the current tick on
    assert "rtc" in c.snapshot()["blend"]
