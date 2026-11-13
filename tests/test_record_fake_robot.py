"""End-to-end check of the recorder against a fake SO-101 (no serial port, no cameras on disk)."""

import json

import numpy as np
import pytest

from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.robots.so_follower import SO101FollowerConfig

from dlsj.kinesthetic import record as rec
from dlsj.kinesthetic.session import Command

MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


class FakeBus:
    def __init__(self):
        self.motors = {m: None for m in MOTORS}
        self.pos = {m: 0.0 for m in MOTORS}
        self.torque = {m: True for m in MOTORS}
        self.goal = {}

    def sync_read(self, name, motors=None, normalize=True, **_):
        motors = motors or list(self.motors)
        if name == "Present_Position":
            return {m: self.pos[m] for m in motors}
        return {m: 1.0 for m in motors}

    def read(self, name, motor, **_):
        return self.pos[motor]

    def sync_write(self, name, values):
        self.goal.update(values)

    def write(self, name, motor, value, **_):
        self.goal[motor] = value

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
        # each tick the arm moves by +1 on every joint
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
            repo_id="local/test_kinesthetic", single_task="sort duplo", root=tmp_path / "ds", fps=30,
            video=False, push_to_hub=False, no_stamp=True,
        ),
        action_lookahead=2, operator="tester",
    )
    r = rec.KinestheticRecorder(cfg, robot=FakeRobot())
    r.dataset = r._open_dataset()
    r.robot.connect()
    return r


def _episode(r, ticks, end_cmd, gripper_toggle_at=None):
    r.machine.handle(Command.START)
    r._handle(Command.START)
    for i in range(ticks):
        if i == gripper_toggle_at:
            r.machine.handle(Command.GRIPPER_TOGGLE)
            r._handle(Command.GRIPPER_TOGGLE)
        r._record_tick()
    r.machine.handle(end_cmd)
    r._handle(end_cmd)


def test_success_failure_discard_and_sidecars(recorder):
    r = recorder
    _episode(r, 10, Command.SAVE_SUCCESS, gripper_toggle_at=5)
    assert not r.arm_held  # starting an episode releases the arm
    _episode(r, 6, Command.DISCARD)
    _episode(r, 7, Command.SAVE_FAILURE)
    r.dataset.finalize()

    assert r.dataset.num_episodes == 2
    labels = [json.loads(line) for line in (r.sidecar / "labels.jsonl").read_text().splitlines()]
    assert [(lab["episode_index"], lab["success"], lab["n_frames"]) for lab in labels] == [(0, True, 10), (1, False, 7)]
    assert (r.sidecar / "telemetry" / "episode_000000.npz").exists()
    tel = np.load(r.sidecar / "telemetry" / "episode_000000.npz")
    assert tel["present_current"].shape == (10, 6)

    from lerobot.datasets import LeRobotDataset

    ds = LeRobotDataset("local/test_kinesthetic", root=r.cfg.dataset.root)
    assert ds.num_frames == 17
    f0, f7 = ds[0], ds[7]
    # lookahead=2: arm action at frame t equals the state at frame t+2
    assert f0["observation.state"][0].item() == 1.0
    assert f0["action"][0].item() == 3.0
    # last frames are padded with the final state (10)
    assert ds[9]["action"][0].item() == 10.0
    # gripper action comes from the key command: open (50) before tick 5, closed (0) from tick 5 on
    assert f0["action"][5].item() == 50.0
    assert f7["action"][5].item() == 0.0


def test_arm_hold_writes_goal_before_torque(recorder):
    r = recorder
    r.bus.pos["elbow_flex"] = 42.0
    r._release_arm()
    assert not r.bus.torque["elbow_flex"] and r.bus.torque["gripper"]  # keys mode keeps gripper powered
    r._hold_arm()
    assert r.bus.goal["elbow_flex"] == 42.0 and r.bus.torque["elbow_flex"]
