"""inspect_pose: demo default, save/load round trip, nudge limits, and the waypoint mover on a fake controller."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy" / "smolvla"))
import inspect_pose as IP  # noqa: E402


class FakeCtl:
    def __init__(self, start):
        self.cmd, self.fps = np.asarray(start, float), 30
        self.sent = []

    def _step(self, target):  # the real controller rate-limits too; the fake arm follows exactly
        self.cmd = np.asarray(target, float).copy()
        self.sent.append(self.cmd.copy())
        return self.cmd


def test_demo_default_without_saved_pose(tmp_path):
    pose, src = IP.load("nobody", cal_dir=tmp_path)
    assert src == "demo" and np.allclose(pose, IP.DEMO_POSE)


def test_save_load_round_trip(tmp_path):
    pytest.importorskip("dlsj.loop.poses")
    IP.save([1, 2, 3, 4, 5], "follower", cal_dir=tmp_path)
    pose, src = IP.load("follower", cal_dir=tmp_path)
    assert src == "saved" and np.allclose(pose, [1, 2, 3, 4, 5])


def test_nudge_clipped_and_checked():
    p = IP.nudge(IP.DEMO_POSE, "elbow_flex", 50)
    assert p[2] == pytest.approx(IP.DEMO_POSE[2] + IP.NUDGE_LIMIT_DEG)
    with pytest.raises(ValueError):
        IP.nudge(IP.DEMO_POSE, "gripper", 1)


def test_mover_visits_waypoints_at_capped_speed_and_keeps_gripper():
    start = [30.0, 20.0, -2.0, 70.0, 28.0, 5.6]
    ctl = FakeCtl(start)
    mid, end = [26.3, -51.8, 30.1, 73.1, 15.9], list(IP.DEMO_POSE)
    m = IP.Mover([mid, end], speed=30.0)
    for _ in range(30 * 20):
        if m.step(ctl):
            break
    assert m.done and np.allclose(ctl.cmd[:5], end, atol=2.5)
    steps = np.abs(np.diff(np.stack(ctl.sent)[:, :5], axis=0)).max()
    assert steps <= 30.0 / 30 + 1e-9  # never faster than the speed cap
    assert all(s[5] == 5.6 for s in ctl.sent)  # the gripper keeps squeezing the brick
    # it really went through the raised waypoint first
    assert min(s[1] for s in ctl.sent) < -45


def test_mover_finishes_when_the_arm_sags_short_of_the_target():
    class Sagging(FakeCtl):
        def _step(self, target):
            self.cmd = np.asarray(target, float).copy()
            m = self.cmd.copy()
            m[1] += 4.0  # shoulder_lift hangs 4 deg below its command under load
            return m

    ctl = Sagging([0, 0, 0, 0, 0, 5.6])
    m = IP.Mover([[1, 1, 1, 1, 1]], speed=30.0)
    for _ in range(30 * 5):
        if m.step(ctl):
            break
    assert m.done
