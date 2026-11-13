import json
import re
import shutil
import subprocess
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.configs.video import RGBEncoderConfig
from lerobot.robots.so_follower import SO101FollowerConfig

from dlsj.kinesthetic.record import KinestheticRecordConfig, KinestheticRecorder
from dlsj.kinesthetic_web import PAGE, FollowerWithStreams, make_handler
from test_record_fake_robot import MOTORS, FakeBus  # noqa: E402  (pytest puts tests/ on sys.path)


class FakeFollower:
    """A LeRobot follower without cameras: joints only."""

    robot_type = "so_follower"

    def __init__(self):
        self.bus = FakeBus()
        self.t = 0

    @property
    def action_features(self):
        return {f"{m}.pos": float for m in MOTORS}

    def get_observation(self):
        self.t += 1
        for m in MOTORS[:-1]:
            self.bus.pos[m] = float(self.t)
        return {f"{m}.pos": self.bus.pos[m] for m in MOTORS}

    def disconnect(self):
        pass


class FakeStream:
    def __init__(self, name, value):
        self.name = name
        self.frame = np.full((48, 64, 3), value, dtype=np.uint8)  # BGR

    def info(self):
        return {"name": self.name, "index": 0, "fps": 30.0, "size": "64x48", "error": None}


@pytest.fixture
def setup(tmp_path):
    streams = {"wrist": FakeStream("wrist", 40), "top": FakeStream("top", 200)}
    robot = FollowerWithStreams(FakeFollower(), streams, wait_s=1)
    cfg = KinestheticRecordConfig(
        robot=SO101FollowerConfig(port="fake"),
        dataset=DatasetRecordConfig(repo_id="local/kin_web", single_task="pick", root=tmp_path / "ds", fps=200,
                                    video=True, push_to_hub=False, no_stamp=True,
                                    rgb_encoder=RGBEncoderConfig(vcodec="h264")),
        overlay=False, terminal_keys=False, sounds=False, approach_speed=1000.0, operator="tester",
    )
    rec = KinestheticRecorder(cfg, robot=robot)
    rec.dataset = rec._open_dataset()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(streams, rec, target_s=6))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield rec, f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def post(base, path):
    urllib.request.urlopen(urllib.request.Request(base + path, method="POST"))


def get_json(base, path):
    return json.loads(urllib.request.urlopen(base + path).read())


def test_browser_keys_drive_a_take_and_dataset_matches_teleop_layout(setup):
    rec, base = setup
    post(base, "/kin/key/space")
    rec._drain_keys()
    assert get_json(base, "/kin/state")["phase"] == "recording"
    assert not rec.arm_held  # torque released for hand guiding
    for _ in range(5):
        rec._record_tick()
    post(base, "/kin/key/space")  # stop -> review, arm held
    rec._drain_keys()
    s = get_json(base, "/kin/state")
    assert s["phase"] == "review" and s["arm_held"] and s["frames"] == 5

    img = urllib.request.urlopen(base + "/kin/overlay.jpg").read()  # replay alignment overlay
    assert img[:2] == b"\xff\xd8"

    post(base, "/kin/key/space")  # accept
    rec._drain_keys()
    rec.dataset.finalize()
    assert rec.dataset.num_episodes == 1
    feats = set(rec.dataset.features)
    # same keys as dlsj.teleop_record, so kinesthetic and teleop datasets can be merged
    assert {"observation.state", "action", "observation.images.wrist", "observation.images.top"} <= feats
    label = json.loads((rec.sidecar / "labels.jsonl").read_text().splitlines()[0])
    assert label["method"] == "kinesthetic" and label["success"]


def test_unknown_or_quit_keys_are_rejected(setup):
    rec, base = setup
    for k in ("q", "esc", "zzz"):
        with pytest.raises(urllib.error.HTTPError):
            post(base, f"/kin/key/{k}")
    assert rec.keys.empty()


def test_camera_frames_are_rgb(setup):
    rec, _ = setup
    obs = rec.robot.get_observation()
    assert obs["wrist"].shape == (48, 64, 3) and obs["shoulder_pan.pos"] == 1.0


def test_page_javascript_parses(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    js = tmp_path / "page.js"
    js.write_text(re.search(r"<script>(.*)</script>", PAGE.replace("__TARGET__", "6"), re.S).group(1))
    r = subprocess.run([node, "--check", str(js)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
