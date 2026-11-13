import numpy as np
import pytest

from dlsj.kinesthetic.session import (
    ActionLookahead,
    Command,
    EpisodeStats,
    GripperCommand,
    KeyMap,
    Phase,
    SessionMachine,
    approach_waypoints,
    tracking_report,
)

JOINTS = ["a.pos", "b.pos", "gripper.pos"]


def test_lookahead_zero_uses_current_state():
    la = ActionLookahead(0, JOINTS)
    out = la.push({"i": 0}, [1.0, 2.0, 3.0])
    assert out == [{"i": 0, "action": [1.0, 2.0, 3.0]}]
    assert la.flush() == []


def test_lookahead_k_uses_future_state_and_flush_pads_with_last():
    la = ActionLookahead(2, JOINTS)
    assert la.push({"i": 0}, [0.0, 0.0, 0.0]) == []
    assert la.push({"i": 1}, [1.0, 1.0, 1.0]) == []
    assert la.push({"i": 2}, [2.0, 2.0, 2.0]) == [{"i": 0, "action": [2.0, 2.0, 2.0]}]
    assert la.push({"i": 3}, [3.0, 3.0, 3.0]) == [{"i": 1, "action": [3.0, 3.0, 3.0]}]
    tail = la.flush()
    assert [f["i"] for f in tail] == [2, 3]
    assert all(f["action"] == [3.0, 3.0, 3.0] for f in tail)
    assert la.flush() == []


def test_lookahead_commanded_joint_uses_command_at_its_own_frame():
    la = ActionLookahead(1, JOINTS)
    la.push({"i": 0}, [0.0, 0.0, 50.0], {"gripper.pos": 50.0})
    out = la.push({"i": 1}, [1.0, 1.0, 40.0], {"gripper.pos": 0.0})
    assert out == [{"i": 0, "action": [1.0, 1.0, 50.0]}]
    assert la.flush() == [{"i": 1, "action": [1.0, 1.0, 0.0]}]


def test_lookahead_frames_count_preserved():
    la = ActionLookahead(3, JOINTS)
    got = []
    for i in range(10):
        got += la.push({"i": i}, [float(i)] * 3)
    got += la.flush()
    assert [f["i"] for f in got] == list(range(10))


def test_lookahead_rejects_negative():
    with pytest.raises(ValueError):
        ActionLookahead(-1, JOINTS)


def test_two_pedals_run_the_whole_loop():
    km, sm = KeyMap(), SessionMachine()

    def press(key):
        cmd = km.lookup(key, sm.phase)
        return sm.handle(cmd) if cmd else None

    assert press("1") is Command.START and sm.phase is Phase.RECORDING
    assert press("1") is Command.STOP and sm.phase is Phase.REVIEW
    assert press("1") is Command.SAVE_SUCCESS and sm.phase is Phase.IDLE
    assert press("1") is Command.START
    assert press("h") is Command.DISCARD and sm.phase is Phase.IDLE  # reject mid-take
    assert press("h") is Command.TOGGLE_HOLD  # idle: release / hold the arm


def test_review_allows_replay_and_failure_but_not_gripper():
    km, sm = KeyMap(), SessionMachine()
    sm.handle(Command.START)
    sm.handle(Command.STOP)
    assert sm.handle(km.lookup("r", sm.phase)) is Command.REPLAY and sm.phase is Phase.REVIEW
    assert sm.handle(km.lookup("space", sm.phase)) is None
    assert sm.handle(km.lookup("f", sm.phase)) is Command.SAVE_FAILURE and sm.phase is Phase.IDLE


def test_ignored_commands():
    sm = SessionMachine()
    assert sm.handle(Command.SAVE_SUCCESS) is None
    assert sm.handle(Command.REPLAY) is None
    sm.handle(Command.START)
    assert sm.handle(Command.START) is None
    assert sm.handle(Command.TOGGLE_HOLD) is None
    assert sm.handle(Command.REPLAY) is None


def test_time_limit_moves_to_review_once():
    sm = SessionMachine()
    assert not sm.time_limit_reached()
    sm.handle(Command.START)
    assert sm.time_limit_reached() and sm.phase is Phase.REVIEW
    assert not sm.time_limit_reached()


def test_keymap_quit_and_unknown():
    km = KeyMap()
    assert km.lookup("Q", Phase.RECORDING) is Command.QUIT
    assert km.lookup("z", Phase.IDLE) is None


def test_gripper_toggle_and_nudge_clamped():
    g = GripperCommand(open_pos=50.0, closed_pos=0.0, step=20.0)
    assert g.target == 50.0 and g.is_open
    assert g.toggle() == 0.0
    assert g.nudge(towards_open=False) == 0.0
    assert g.nudge(towards_open=True) == 20.0
    assert g.toggle() == 50.0  # 20 is nearer closed, so the toggle opens


def test_approach_waypoints_speed_limited_and_ends_at_goal():
    wp = approach_waypoints(np.array([0.0, 0.0]), np.array([30.0, -6.0]), max_speed=30.0, fps=30)
    assert len(wp) == 30
    assert np.allclose(wp[-1], [30.0, -6.0])
    assert np.max(np.abs(np.diff(np.vstack([[0.0, 0.0], wp]), axis=0))) <= 1.0 + 1e-9
    assert len(approach_waypoints(np.zeros(2), np.zeros(2), 10, 30)) == 1


def test_tracking_report_uses_delay():
    cmd = np.array([[0.0], [1.0], [2.0], [3.0]])
    meas = np.array([[0.0], [0.0], [1.0], [2.5]])  # follows one tick late, then lags by 0.5
    rep = tracking_report(cmd, meas, ["elbow.pos"], delay=1)
    assert rep.max_abs[0] == pytest.approx(0.5)
    assert rep.worst() == ("elbow.pos", pytest.approx(0.5))
    assert "elbow" in rep.format("deg")


def test_episode_stats_summary():
    s = EpisodeStats()
    assert "no saved" in s.summary()
    for d in [8.0, 6.0, 10.0, 7.0]:
        s.add(d)
    assert "fastest 6.0s" in s.summary()
