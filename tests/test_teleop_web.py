import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from dlsj import arms
from dlsj.teleop_web import Teleop, step_toward

JOINTS = [f"{m}.pos" for m in arms.MOTORS]


def cal(**over):
    base = {m: {"range_min": 1000, "range_max": 3000} for m in arms.MOTORS}
    base["wrist_roll"] = {"range_min": 0, "range_max": 4095}
    base.update(over)
    return base


def test_good_calibration_passes():
    assert arms.calibration_problems(cal()) == []


def test_unmoved_gripper_flagged():
    (p,) = arms.calibration_problems(cal(gripper={"range_min": 2047, "range_max": 2056}))
    assert p.startswith("gripper: only moved 9")


def test_wrapped_joint_flagged():
    (p,) = arms.calibration_problems(cal(shoulder_lift={"range_min": 9, "range_max": 4095}))
    assert "wrapped" in p


def test_classify_ports_by_voltage():
    f, l = arms.classify_ports({"/dev/a": 5.1, "/dev/b": 12.8, "/dev/c": None})
    assert (f, l) == ("/dev/b", "/dev/a")
    assert arms.classify_ports({"/dev/a": 5.1}) == (None, "/dev/a")
    with pytest.raises(RuntimeError):
        arms.classify_ports({"/dev/a": 12.0, "/dev/b": 12.5})


def test_step_toward_caps_each_joint():
    out = step_toward(np.array([0.0, 0.0, 10.0]), np.array([5.0, -0.2, 0.0]), 1.0)
    np.testing.assert_allclose(out, [1.0, -0.2, 9.0])


class FakeArm:
    def __init__(self, pose):
        self.pose = dict(pose)
        self.sent = []

    def get_observation(self):
        return dict(self.pose)

    def get_action(self):
        return dict(self.pose)

    def send_action(self, a):
        self.sent.append(dict(a))
        self.pose = dict(a)


def make_teleop(tmp_path, monkeypatch, home):
    monkeypatch.setattr(arms, "FOLLOWER_CAL_DIR", tmp_path)
    t = Teleop.__new__(Teleop)
    monkeypatch.setattr(arms, "LEADER_CAL_DIR", tmp_path)
    t.args = SimpleNamespace(follower_id="f", leader_id="l", auto_engage=False)
    t.align = np.zeros(len(JOINTS))
    t.sign = np.ones(len(JOINTS))
    t.fps = 200
    t.max_step = 1.0
    t.joints = JOINTS
    t.robot = FakeArm({j: 0.0 for j in JOINTS})
    t.leader = FakeArm({j: 20.0 for j in JOINTS})
    t.home = home
    t.phase, t.message, t.cmd, t.state = "connecting", "", None, {}
    import queue

    t.commands = queue.Queue()
    t.lock = threading.Lock()
    t._stop = threading.Event()
    return t


def wait_for(cond, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.005)
    return False


def test_homes_then_waits_then_engages_smoothly_then_returns_home(tmp_path, monkeypatch):
    home = {j: 10.0 for j in JOINTS}
    t = make_teleop(tmp_path, monkeypatch, home)
    th = threading.Thread(target=t.run, daemon=True)
    th.start()
    assert wait_for(lambda: t.phase == "ready")
    assert t.robot.pose == home
    time.sleep(0.05)
    assert t.phase == "ready"  # does not follow the leader on its own

    t.command("engage")
    assert wait_for(lambda: t.phase == "live")
    steps = np.array([[a[j] for j in JOINTS] for a in t.robot.sent])
    assert np.max(np.abs(np.diff(steps, axis=0))) <= 1.0 + 1e-9  # never jumped

    t.stop()
    th.join(3)
    assert not th.is_alive()
    assert t.robot.pose == home


def test_save_home_writes_file(tmp_path, monkeypatch):
    t = make_teleop(tmp_path, monkeypatch, None)
    th = threading.Thread(target=t.run, daemon=True)
    th.start()
    assert wait_for(lambda: t.phase == "ready")
    t.command("save_home")
    assert wait_for(lambda: t.home is not None)
    assert arms.load_home("f") == {j: 0.0 for j in JOINTS}
    t.stop()
    th.join(3)
    assert not th.is_alive()


def test_page_javascript_parses(tmp_path):
    import re
    import shutil
    import subprocess

    from dlsj.teleop_web import PAGE

    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    js = tmp_path / "page.js"
    js.write_text(re.search(r"<script>(.*)</script>", PAGE, re.S).group(1))
    r = subprocess.run([node, "--check", str(js)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_unwrap_step_crosses_encoder_edge():
    assert arms.unwrap_step(4090, 5) == 11
    assert arms.unwrap_step(5, 4090) == -11
    assert arms.unwrap_step(1000, 1200) == 200


def test_centred_homing_handles_rest_at_end_of_travel():
    # Elbow folded at rest sits at raw 4000 and travels 2200 ticks "up", crossing 4095 -> 0.
    mid_raw, rmin, rmax = arms.centred_homing(4000, 0, 2200)
    assert mid_raw == (4000 + 1100) % 4096
    assert (rmin, rmax) == (2047 - 1100, 2047 + 1100)
    assert arms.calibration_problems(
        {m: {"range_min": rmin, "range_max": rmax} for m in arms.MOTORS}
    ) == []


def test_align_makes_engage_a_no_op(tmp_path, monkeypatch):
    t = make_teleop(tmp_path, monkeypatch, None)  # leader reads 20, follower sits at 0
    th = threading.Thread(target=t.run, daemon=True)
    th.start()
    assert wait_for(lambda: t.phase == "ready")
    t.command("align")
    assert wait_for(lambda: arms.load_align("l") is not None)
    # Once aligned the leader matches the follower, so it latches on by itself without moving the follower.
    assert wait_for(lambda: t.phase == "live")
    time.sleep(0.02)
    sent = np.array([[a[j] for j in JOINTS] for a in t.robot.sent])
    assert np.max(np.abs(sent)) < 1e-9  # follower never moved
    t.stop()
    th.join(3)


def test_inverted_joint_is_mirrored(tmp_path, monkeypatch):
    t = make_teleop(tmp_path, monkeypatch, None)
    t.sign = np.array([-1.0 if j == "wrist_roll.pos" else 1.0 for j in JOINTS])
    t.leader.pose = {j: 0.0 for j in JOINTS} | {"wrist_roll.pos": 30.0}
    th = threading.Thread(target=t.run, daemon=True)
    th.start()
    assert wait_for(lambda: t.phase == "ready")
    t.command("engage")
    assert wait_for(lambda: t.phase == "live")
    assert t.robot.pose["wrist_roll.pos"] == pytest.approx(-30.0)
    t.stop()
    th.join(3)


def test_engages_when_leader_matches_and_not_again_right_after_hold(tmp_path, monkeypatch):
    t = make_teleop(tmp_path, monkeypatch, None)  # follower at 0, leader at 20
    th = threading.Thread(target=t.run, daemon=True)
    th.start()
    assert wait_for(lambda: t.phase == "ready")
    time.sleep(0.05)
    assert t.phase == "ready"  # 20 deg off: waits
    t.leader.pose = {j: 3.0 for j in JOINTS}  # within 5 deg
    assert wait_for(lambda: t.phase == "live")

    t.command("hold")
    assert wait_for(lambda: t.phase == "ready")
    time.sleep(0.05)
    assert t.phase == "ready"  # leader still on the pose: no instant re-latch
    t.leader.pose = {j: 30.0 for j in JOINTS}  # away...
    time.sleep(0.05)
    t.leader.pose = {j: 4.0 for j in JOINTS}  # ...and back
    assert wait_for(lambda: t.phase == "live")
    t.stop()
    th.join(3)


def test_parse_cameras_accepts_indices_and_device_paths():
    from dlsj.teleop_web import parse_cameras

    assert parse_cameras("wrist=0, top=/dev/v4l/by-id/usb-cam-video-index0", 8) == {
        "wrist": 0,
        "top": "/dev/v4l/by-id/usb-cam-video-index0",
    }
