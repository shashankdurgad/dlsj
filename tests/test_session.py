import pytest

from dlsj.kinesthetic.session import (
    ActionLookahead,
    Command,
    EpisodeStats,
    GripperCommand,
    KeyMap,
    Phase,
    SessionMachine,
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
    out = la.push({"i": 2}, [2.0, 2.0, 2.0])
    assert out == [{"i": 0, "action": [2.0, 2.0, 2.0]}]
    out = la.push({"i": 3}, [3.0, 3.0, 3.0])
    assert out == [{"i": 1, "action": [3.0, 3.0, 3.0]}]
    tail = la.flush()
    assert [f["i"] for f in tail] == [2, 3]
    assert all(f["action"] == [3.0, 3.0, 3.0] for f in tail)
    assert la.flush() == []


def test_lookahead_commanded_joint_uses_command_at_its_own_frame():
    la = ActionLookahead(1, JOINTS)
    la.push({"i": 0}, [0.0, 0.0, 50.0], {"gripper.pos": 50.0})
    out = la.push({"i": 1}, [1.0, 1.0, 40.0], {"gripper.pos": 0.0})
    # frame 0: arm joints from frame 1's state, gripper from frame 0's command
    assert out == [{"i": 0, "action": [1.0, 1.0, 50.0]}]
    tail = la.flush()
    assert tail == [{"i": 1, "action": [1.0, 1.0, 0.0]}]


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


def test_session_flow_and_ignored_keys():
    sm = SessionMachine()
    assert sm.handle(Command.SAVE_SUCCESS) is None  # nothing recording
    assert sm.handle(Command.DISCARD) is None
    assert sm.handle(Command.START) is Command.START
    assert sm.phase is Phase.RECORDING
    assert sm.handle(Command.START) is None  # already recording
    assert sm.handle(Command.TOGGLE_HOLD) is None  # never hold mid-demo
    assert sm.handle(Command.SAVE_FAILURE) is Command.SAVE_FAILURE
    assert sm.phase is Phase.IDLE


def test_session_time_limit_then_label():
    sm = SessionMachine()
    sm.handle(Command.START)
    sm.time_limit_reached()
    assert sm.phase is Phase.LABELLING
    assert sm.handle(Command.GRIPPER_TOGGLE) is None
    assert sm.handle(Command.DISCARD) is Command.DISCARD
    assert sm.phase is Phase.IDLE


def test_keymap_lookup_case_insensitive():
    km = KeyMap()
    assert km.lookup("N") is Command.SAVE_SUCCESS
    assert km.lookup("space") is Command.GRIPPER_TOGGLE
    assert km.lookup("z") is None


def test_gripper_toggle_and_nudge_clamped():
    g = GripperCommand(open_pos=50.0, closed_pos=0.0, step=20.0)
    assert g.target == 50.0 and g.is_open
    assert g.toggle() == 0.0
    assert g.nudge(towards_open=False) == 0.0  # clamped
    assert g.nudge(towards_open=True) == 20.0
    assert g.toggle() == 50.0  # 20 is nearer closed, so the toggle opens
