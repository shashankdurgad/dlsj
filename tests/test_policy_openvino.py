"""Hardware-free tests of the pi0.5 OpenVINO connector (`dlsj.loop.policies`) and the replay sources.

A stub stands in for `physicalai.InferenceModel`: same attributes the connector reads (input/output features,
chunk_size, pre/postprocessors, adapter.input_names) and `predict_action_chunk(inputs)`. The real runtime is
checked against recorded episodes by `deploy/pi05/replay_check.py` on the XPS.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from dlsj.contracts import Observation
from dlsj.loop.policies import OpenVINOPi05Policy

CAMS = ["top", "wrist", "inspect"]  # deliberately not the Observation dict order


class StubNoise:
    def __init__(self, shape):
        self._shape = tuple(shape)


class StubModel:
    """Mimics InferenceModel for a 3-camera pi0.5 export with a 6-d state and a 32-d raw action head."""

    def __init__(self, cams=CAMS, state_dim=6, chunk=50, raw_dim=32, rtc=False, delay_s=0.0, out_dim=None):
        self.input_features = [SimpleNamespace(name=f"images.{c}", shape=[3, 480, 640]) for c in cams]
        self.input_features += [SimpleNamespace(name="state", shape=[state_dim]), SimpleNamespace(name="task", shape=None)]
        self.output_features = [SimpleNamespace(name="action", shape=[chunk, state_dim])]
        self.chunk_size = chunk
        self.preprocessors = [StubNoise((chunk, raw_dim))]
        self.postprocessors = []
        names = ["images", "image_masks", "tokenized_prompt", "tokenized_prompt_mask", "state", "noise"]
        if rtc:
            names += ["prev_chunk_left_over", "inference_delay", "max_guidance_weight", "execution_horizon"]
        self.adapter = SimpleNamespace(input_names=names)
        self.raw_dim, self.out_dim, self.delay_s = raw_dim, out_dim or raw_dim, delay_s
        self.calls: list[dict] = []
        self.resets = 0
        self._inside = 0
        self.overlap = False

    def predict_action_chunk(self, inputs):
        self._inside += 1
        if self._inside > 1:
            self.overlap = True
        try:
            time.sleep(self.delay_s)
            self.calls.append(inputs)
            n = self.chunk_size
            raw = np.tile(np.arange(n, dtype=np.float32)[:, None] / n, (1, self.raw_dim))[None]  # normalised
            outputs = {"action": raw}
            for p in self.postprocessors:
                outputs = p(outputs)
            state = inputs["state"][0]
            out = np.zeros((n, self.out_dim), np.float32)
            out[:, : len(state)] = state + np.arange(n, dtype=np.float32)[:, None]  # "denormalised" ramp
            return out
        finally:
            self._inside -= 1

    def reset(self):
        self.resets += 1


def make_obs(seed=0, h=480, w=640, state=None):
    rng = np.random.default_rng(seed)
    imgs = {c: rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for c in ("wrist", "top", "inspect")}
    st = np.array([-5.98, -92.79, 97.19, 4.26, -4.35, 1.46], np.float32) if state is None else state
    return Observation(state=st, images=imgs, timestamp=time.time())


def test_reads_export_metadata():
    p = OpenVINOPi05Policy(None, model=StubModel())
    assert p.cameras == CAMS and p.state_dim == 6 and p.chunk_len == 50 and not p.rtc


def test_input_dict_matches_export_layout():
    p = OpenVINOPi05Policy(None, model=StubModel())
    obs = make_obs()
    inp = p.build_inputs(obs, "pick up the brick and hold it in front of the inspection camera")
    assert [k for k in inp if k.startswith("images.")] == [f"images.{c}" for c in CAMS]  # manifest order
    for c in CAMS:
        a = inp[f"images.{c}"]
        assert a.shape == (1, 3, 480, 640) and a.dtype == np.uint8 and a.flags.c_contiguous
        np.testing.assert_array_equal(a[0].transpose(1, 2, 0), obs.images[c])  # RGB channels untouched
    assert inp["state"].shape == (1, 6) and inp["state"].dtype == np.float32
    np.testing.assert_array_equal(inp["state"][0], obs.state)  # degrees, no scaling (the export normalises)
    assert inp["task"] == ["pick up the brick and hold it in front of the inspection camera"]
    assert "noise" not in inp  # unseeded: the export's noise preprocessor draws fresh noise


def test_camera_map_and_errors():
    p = OpenVINOPi05Policy(None, model=StubModel(cams=["front", "wrist", "side"]),
                           camera_map={"front": "top", "side": "inspect"})
    inp = p.build_inputs(make_obs(), "x")
    np.testing.assert_array_equal(inp["images.front"][0].transpose(1, 2, 0), make_obs().images["top"])
    with pytest.raises(KeyError):
        OpenVINOPi05Policy(None, model=StubModel(cams=["front"])).build_inputs(make_obs(), "x")
    with pytest.raises(ValueError):
        p.build_inputs(make_obs(state=np.zeros(7, np.float32)), "x")
    with pytest.raises(ValueError):
        p.build_inputs(make_obs(state=np.array([0, 0, np.nan, 0, 0, 0], np.float32)), "x")
    bad = make_obs()
    bad.images["wrist"] = bad.images["wrist"][..., 0]
    with pytest.raises(ValueError):
        p.build_inputs(bad, "x")


def test_float_images_are_converted():
    p = OpenVINOPi05Policy(None, model=StubModel())
    obs = make_obs()
    obs.images["top"] = obs.images["top"].astype(np.float32)
    assert p.build_inputs(obs, "x")["images.top"].dtype == np.uint8


def test_chunk_postprocessing():
    m = StubModel()
    p = OpenVINOPi05Policy(None, model=m)
    obs = make_obs()
    chunk = p.predict_chunk(obs, "place the brick in the yellow bowl")
    assert chunk.shape == (50, 6) and chunk.dtype == np.float32  # 32-d padding dropped
    np.testing.assert_allclose(chunk[0], obs.state)
    np.testing.assert_allclose(chunk[49], obs.state + 49)
    assert p.latency_ms > 0 and p.calls == 1
    assert p.postprocess(np.zeros((1, 50, 6))).shape == (50, 6)
    with pytest.raises(ValueError):
        p.postprocess(np.full((50, 6), np.inf))
    with pytest.raises(ValueError):
        p.postprocess(np.zeros((50, 4)))
    with pytest.raises(ValueError):
        p.postprocess(np.zeros((2, 50, 6)))


def test_seeded_noise_reproducible_and_reset():
    a = OpenVINOPi05Policy(None, model=StubModel(), seed=7)
    b = OpenVINOPi05Policy(None, model=StubModel(), seed=7)
    obs = make_obs()
    a.predict_chunk(obs, "x")
    a.predict_chunk(obs, "x")
    b.predict_chunk(obs, "x")
    n0, n1 = a.model.calls[0]["noise"], a.model.calls[1]["noise"]
    assert n0.shape == (1, 50, 32) and n0.dtype == np.float32
    np.testing.assert_array_equal(n0, b.model.calls[0]["noise"])
    assert not np.array_equal(n0, n1)  # fresh noise every call
    a.reset()
    assert a.model.resets == 1
    a.predict_chunk(obs, "x")
    np.testing.assert_array_equal(a.model.calls[2]["noise"], n0)  # reset re-seeds
    explicit = np.ones((1, 50, 32), np.float32)
    a.predict_chunk(obs, "x", noise=explicit)
    np.testing.assert_array_equal(a.model.calls[3]["noise"], explicit)


def test_thread_safe_calls():
    m = StubModel(delay_s=0.01)
    p = OpenVINOPi05Policy(None, model=m)
    obs = make_obs()
    ts = [threading.Thread(target=lambda: [p.predict_chunk(obs, "x") for _ in range(3)]) for _ in range(4)]
    ts.append(threading.Thread(target=lambda: [p.reset() for _ in range(5)]))
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not m.overlap and len(m.calls) == 12


def test_warmup_uses_all_cameras_and_resets():
    m = StubModel()
    p = OpenVINOPi05Policy(None, model=m)
    p.warmup(2)
    assert len(m.calls) == 2 and m.resets == 1
    assert m.calls[0]["images.top"].shape == (1, 3, 480, 640)


def test_rtc_feeds_previous_raw_tail():
    m = StubModel(rtc=True)
    p = OpenVINOPi05Policy(None, model=m, fps=30.0)
    assert p.rtc
    obs = make_obs()
    p.predict_chunk(obs, "x")
    first = m.calls[0]
    assert first["prev_chunk_left_over"].shape == (1, 50, 32) and not first["prev_chunk_left_over"].any()
    assert float(first["max_guidance_weight"]) < 1e-6 and int(first["inference_delay"]) == 0
    obs2 = Observation(state=obs.state, images=obs.images, timestamp=obs.timestamp + 20 / 30.0)  # 20 actions later
    p.predict_chunk(obs2, "x")
    left = m.calls[1]["prev_chunk_left_over"][0]
    raw = np.arange(50, dtype=np.float32) / 50
    np.testing.assert_allclose(left[:30, 0], raw[20:])  # unexecuted tail of the raw (normalised) chunk
    assert not left[30:].any()
    assert int(m.calls[1]["execution_horizon"]) == 10
    p.reset()
    p.predict_chunk(obs, "x")
    assert not m.calls[2]["prev_chunk_left_over"].any()
    off = OpenVINOPi05Policy(None, model=StubModel(rtc=True), rtc=False)
    off.predict_chunk(obs, "x")
    assert "prev_chunk_left_over" not in off.model.calls[0]


def test_executor_drives_policy_from_background_thread():
    from dlsj.loop import fakes
    from dlsj.loop.executor import ExecConfig, SkillExecutor
    from dlsj.loop.skills import PICK_AND_SHOW, SkillLibrary

    world = fakes.FakeWorld(bricks=["plain"])
    robot = fakes.FakeRobot(world)
    m = StubModel(delay_s=0.02)
    policy = OpenVINOPi05Policy(None, model=m)
    ex = SkillExecutor(robot, fakes.FakeCameras(world, robot), policy, SkillLibrary(), ExecConfig(fps=100.0),
                       dict(world.poses))
    abort = threading.Event()
    threading.Timer(0.8, abort.set).start()
    res = ex.run(PICK_AND_SHOW, abort)
    ex.close()
    assert res.error is None and res.outcome.value == "aborted" and res.chunks >= 2
    assert m.resets >= 1  # executor resets the policy at skill start
    assert all(c["task"] == ["pick up the brick and hold it in front of the inspection camera"] for c in m.calls)


# ----------------------------------------------------------------------------- replay sources
def _episode():
    from dlsj.loop.replay import Episode

    n = 40
    states = np.zeros((n, 6), np.float32)
    states[:, 0] = np.linspace(0, 39, n)  # shoulder_pan sweeps 0..39 deg
    states[:, 5] = 60.0
    states[15:, 5] = 25.0  # gripper closed on the brick from frame 15
    tasks = ["pick up the brick and hold it in front of the inspection camera"] * 20 + \
            ["place the brick in the orange bowl"] * 20
    idx = np.arange(0, n, 2)
    imgs = {c: [np.full((4, 4, 3), int(i), np.uint8) for i in idx] for c in ("wrist", "top", "inspect")}
    return Episode(states, states.copy(), tasks, idx, imgs, 30.0)


def test_replay_poses_and_state_matched_frames():
    from dlsj.loop.replay import ReplayCameras, ReplayRobot, episode_poses, holding_grip, parse_replay_spec

    ep = _episode()
    assert ep.split == 20 and holding_grip(ep) == 25.0
    poses = episode_poses(ep)
    assert sorted(poses) == ["home", "inspect", "orange"] and poses["inspect"][0] == 19.0
    robot = ReplayRobot(ep.states[0], holding_grip(ep), lag=1.0)
    cams = ReplayCameras(ep, robot, fps=1000.0)
    robot.send(np.array([30.2, 0, 0, 0, 0, 60], np.float32))
    assert cams.latest()["top"][0, 0, 0] == 30  # recorded frame whose state is closest to the robot's
    assert len(cams.fresh("inspect", 3)) == 3
    assert parse_replay_spec("replay:~/d/x:5") == ("~/d/x", 5, 2)
    assert parse_replay_spec("replay:/a/b:12:3") == ("/a/b", 12, 3)
    with pytest.raises(ValueError):
        parse_replay_spec("replay:/a/b")


def test_replay_robot_gripper_stalls_on_brick():
    from dlsj.loop.replay import ReplayRobot

    r = ReplayRobot(np.array([0, 0, 0, 0, 0, 60], np.float32), hold_grip=25.0, lag=1.0)
    r.send(np.array([0, 0, 0, 0, 0, 2], np.float32))  # close hard: jaws stop at the brick
    assert r.read_state()[5] == 25.0 and r.holding and r.read_gripper_current() == 300.0
    r.send(np.array([0, 0, 0, 0, 0, 30], np.float32))  # slightly open: still holding
    assert r.holding
    r.send(np.array([0, 0, 0, 0, 0, 70], np.float32))  # open: released
    assert not r.holding and r.read_state()[5] == 70.0


def test_run_refuses_replay_on_real_arm():
    from dlsj.loop import run

    args = run.parse_args(["--robot", "so101", "--cameras", "replay:/nonexistent:0", "--status-port", "0"])
    with pytest.raises(SystemExit):
        run.build(args)
