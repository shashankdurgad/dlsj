"""End-to-end checks of the recorder against a fake SO-101 (no serial port, no real cameras)."""

import json

import numpy as np
import pytest

from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.robots.so_follower import SO101FollowerConfig

from dlsj.kinesthetic import record as rec
from dlsj.kinesthetic.session import Command, Phase

MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


class FakeBus:
    """Positions follow Goal_Position writes immediately (perfect tracking) when torque is on."""

    def __init__(self):
        self.motors = {m: None for m in MOTORS}
        self.pos = {m: 0.0 for m in MOTORS}
        self.torque = {m: True for m in MOTORS}
        self.goal = {}
        self.goal_log = []

    def sync_read(self, name, motors=None, normalize=True, **_):
        motors = motors or list(self.motors)
        if name == "Present_Position":
            return {m: self.pos[m] for m in motors}
        return {m: 1.0 for m in motors}

    def read(self, name, motor, **_):
        return self.pos[motor]

    def sync_write(self, name, values):
        self.goal.update(values)
        self.goal_log.append(dict(values))
        for m, v in values.items():
            if self.torque[m]:
                self.pos[m] = float(v)

    def write(self, name, motor, value, **_):
        self.sync_write(name, {motor: value})

    def enable_torque(self, motors=None, **_):
        for m in motors or self.motors:
            self.torque[m] = True

    def disable_torque(self, motors=None, **_):
        for m in motors or self.motors:
            self.torque[m] = False


class FakeRobot:
    robot_type = "so_follower"

    def __init__(self):
        self.bus = FakeBus()
        self.cameras = {"wrist": None}
        self.is_connected = False
        self.t = 0

    @property
    def action_features(self):
        return {f"{m}.pos": float for m in MOTORS}

    @property
    def observation_features(self):
        return {**self.action_features, "wrist": (8, 8, 3)}

    def connect(self):
        self.is_connected = True

    def disconnect(self):
        self.is_connected = False

    def get_observation(self):
        # each tick the hand-guided arm moves by +1 on every arm joint
        self.t += 1
        for m in MOTORS[:-1]:
            self.bus.pos[m] = float(self.t)
        obs = {f"{m}.pos": self.bus.pos[m] for m in MOTORS}
        obs["wrist"] = np.full((8, 8, 3), self.t % 255, dtype=np.uint8)
        return obs


@pytest.fixture
def recorder(tmp_path):
    cfg = rec.KinestheticRecordConfig(
        robot=SO101FollowerConfig(port="fake"),
        dataset=DatasetRecordConfig(
            repo_id="local/test_kinesthetic", single_task="sort duplo", root=tmp_path / "ds", fps=200,
            video=False, push_to_hub=False, no_stamp=True,
        ),
        action_lookahead=2, operator="tester", overlay=False, approach_speed=1000.0, sounds=False,
    )
    r = rec.KinestheticRecorder(cfg, robot=FakeRobot())
    r.dataset = r._open_dataset()
    r.robot.connect()
    return r


def press(r, key):
    """Simulate a key press through the same path as the live loop."""
    r.keys.put(key)
    return r._drain_keys()


def take(r, ticks, gripper_toggle_at=None):
    assert press(r, "1") and r.machine.phase is Phase.RECORDING
    for i in range(ticks):
        if i == gripper_toggle_at:
            press(r, "g")
        r._record_tick()


def test_pedal_loop_review_success_failure_reject_and_sidecars(recorder):
    r = recorder
    take(r, 10, gripper_toggle_at=5)
    assert not r.arm_held  # recording releases the arm
    press(r, "1")  # stop -> review, arm held
    assert r.machine.phase is Phase.REVIEW and r.arm_held
    press(r, "1")  # accept
    take(r, 6)
    press(r, "h")  # reject mid-take
    take(r, 7)
    press(r, "f")  # save as failure straight from recording
    r.dataset.finalize()

    assert r.dataset.num_episodes == 2
    labels = [json.loads(line) for line in (r.sidecar / "labels.jsonl").read_text().splitlines()]
    assert [(lab["episode_index"], lab["success"], lab["n_frames"]) for lab in labels] == [(0, True, 10), (1, False, 7)]
    tel = np.load(r.sidecar / "telemetry" / "episode_000000.npz")
    assert tel["present_current"].shape == (10, 6)

    from lerobot.datasets import LeRobotDataset

    ds = LeRobotDataset("local/test_kinesthetic", root=r.cfg.dataset.root)
    assert ds.num_frames == 17
    # lookahead=2: arm action at frame t equals the state at frame t+2; tail padded with the final state
    assert ds[0]["observation.state"][0].item() == 1.0
    assert ds[0]["action"][0].item() == 3.0
    assert ds[9]["action"][0].item() == 10.0
    # gripper action is the key command: open (50) before tick 5, closed (0) from tick 5 on
    assert ds[0]["action"][5].item() == 50.0
    assert ds[7]["action"][5].item() == 0.0


def test_review_replay_plays_every_action_and_records_tracking(recorder):
    r = recorder
    take(r, 8)
    press(r, "1")  # review
    n_writes = len(r.bus.goal_log)
    r.keys.put("r")  # replay, then answer "reset the scene, 1 = replay"
    r.keys.put("1")
    assert r._drain_keys()
    replayed = [g for g in r.bus.goal_log[n_writes:] if set(g) == set(MOTORS)]
    actions = np.stack(r._take_actions)
    # the approach ends on the first action, then every action is sent once, in order
    assert [g["elbow_flex"] for g in replayed[-len(actions):]] == list(actions[:, 2])
    assert r._last_tracking is not None and not r._last_tracking["stopped_early"]
    assert r._last_tracking["max_abs"]["elbow_flex"] == pytest.approx(1.0)  # perfect tracking, 1-tick ramp
    assert r.machine.phase is Phase.REVIEW and r.arm_held
    press(r, "n")
    label = json.loads((r.sidecar / "labels.jsonl").read_text().splitlines()[0])
    assert label["replay_tracking"]["replay_s"] >= 0


def test_replay_cancel_keeps_review(recorder):
    r = recorder
    take(r, 4)
    press(r, "1")
    r.keys.put("r")
    r.keys.put("h")  # cancel at the scene-reset prompt
    r._drain_keys()
    assert r.machine.phase is Phase.REVIEW and r._last_tracking is None
    press(r, "h")  # reject
    assert r.dataset.num_episodes == 0


def test_start_pose_saved_and_returned_to(recorder):
    r = recorder
    r.cfg.go_to_start = True
    r.bus.pos.update({"elbow_flex": 42.0, "gripper": 50.0})
    press(r, "p")
    assert json.loads((r.sidecar / "start_pose.json").read_text())["elbow_flex"] == 42.0
    take(r, 3)
    press(r, "n")
    assert r.bus.pos["elbow_flex"] == pytest.approx(42.0)


def test_arm_hold_writes_goal_before_torque(recorder):
    r = recorder
    r.bus.pos["elbow_flex"] = 42.0
    r._release_arm()
    assert not r.bus.torque["elbow_flex"] and r.bus.torque["gripper"]  # keys mode keeps gripper powered
    r._hold_arm()
    assert r.bus.goal["elbow_flex"] == 42.0 and r.bus.torque["elbow_flex"]


def test_quit_mid_take_discards(recorder):
    r = recorder
    take(r, 3)
    assert press(r, "q") is False
    assert r.dataset.num_episodes == 0 and r.machine.phase is Phase.IDLE


def test_space_starts_and_stops_with_distinct_cues(recorder):
    r = recorder
    played = []
    r.cues.play = played.append
    press(r, "space")
    assert r.machine.phase is Phase.RECORDING and played == ["start"]
    r._record_tick()
    press(r, "space")
    assert r.machine.phase is Phase.REVIEW and played == ["start", "stop"]
    press(r, "h")  # rejecting an already-stopped take does not beep again
    assert played == ["start", "stop"]
