import json
import re
import shutil
import subprocess
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from dlsj.inspect_detect import BrickInViewDetector, DetectorConfig
from dlsj.teleop_record import PAGE, EpisodeRecorder, make_handler

JOINTS = [f"{m}.pos" for m in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]]


class FakeTeleop:
    joints = JOINTS

    def __init__(self):
        self.latest = None
        self.t = 0
        self.commands = []

    def go_live(self):
        self.latest = {"t": 0.0, "follower": np.zeros(6), "cmd": np.zeros(6), "phase": "live"}

    def step(self):
        self.t += 1
        self.latest = {"t": float(self.t), "follower": np.full(6, float(self.t)),
                       "cmd": np.full(6, float(self.t) + 0.5), "phase": self.latest["phase"]}

    def snapshot(self):
        return {"phase": self.latest["phase"] if self.latest else "ready", "hz": 60.0}

    def command(self, name):
        self.commands.append(name)


class FakeStream:
    def __init__(self, name, h=48, w=64):
        self.name = name
        self.frame = np.full((h, w, 3), 120, dtype=np.uint8)  # unsaturated grey

    def show_brick(self, on=True):
        self.frame = np.full(self.frame.shape, 120, dtype=np.uint8)
        if on:
            self.frame[14:34, 20:44] = (0, 0, 220)  # a red brick, fully inside the ROI

    def info(self):
        return {"name": self.name, "index": 0, "fps": 30.0, "size": "64x48", "error": None}


@pytest.fixture
def rec(tmp_path):
    teleop = FakeTeleop()
    streams = {"wrist": FakeStream("wrist"), "inspect": FakeStream("inspect")}
    det = BrickInViewDetector(DetectorConfig(stable_frames=2, downscale=1))
    r = EpisodeRecorder(teleop, streams, "local/test_teleop", fps=30, root=tmp_path / "ds", episode_time_s=60,
                        operator="tester", vcodec="h264", bowls=["red", "blue", "green"], detector=det)
    r.open_dataset()
    return r


def step(r, n=1):
    for _ in range(n):
        r.teleop.step()
        r.tick()


def press(r, *cmds):
    for c in cmds:
        r.command(c)
    r.tick()


def test_refuses_to_start_until_follower_is_live(rec):
    press(rec, "space")
    assert rec.phase == "idle" and "Engage" in rec.message


def test_detector_splits_take_and_tasks_follow_the_chosen_bowl(rec):
    rec.teleop.go_live()
    press(rec, "space")
    assert rec.phase == "recording" and rec.stage == "pick"
    step(rec, 3)  # picking, brick not yet at the camera
    rec.streams["inspect"].show_brick()
    step(rec, 3)  # detector needs 2 stable frames
    assert rec.stage == "place" and rec.split_frame is not None
    split = rec.split_frame
    rec.streams["inspect"].show_brick(False)
    step(rec, 3)  # placing; bowl not chosen yet
    press(rec, "space")  # stop -> review
    press(rec, "space")  # save refused: no bowl
    assert rec.phase == "review" and "bowl" in rec.message.lower()
    press(rec, "bowl3", "space")  # choose green, save
    rec.close()

    assert rec.dataset.num_episodes == 1
    lab = json.loads((rec.sidecar / "labels.jsonl").read_text().splitlines()[0])
    assert lab["bowl"] == "green" and lab["inspection_frame"] == split and lab["inspection_auto"]
    assert lab["place_task"] == "place the brick in the green bowl"

    from lerobot.datasets import LeRobotDataset

    ds = LeRobotDataset("local/test_teleop", root=rec.root)
    tasks = [ds[i]["task"] for i in range(ds.num_frames)]
    assert set(tasks[:split]) == {"pick up the brick and hold it in front of the inspection camera"}
    assert set(tasks[split:]) == {"place the brick in the green bowl"}  # rewritten from the placeholder
    f = ds[1]
    assert f["observation.state"][0].item() == 1.0 and f["action"][0].item() == 1.5  # state=follower, action=cmd


def test_manual_in_view_failure_and_discard(rec):
    rec.teleop.go_live()
    press(rec, "space")
    step(rec, 2)
    press(rec, "space", "space")  # stop, then try to save without an in-view moment
    assert rec.phase == "review" and "in-view" in rec.message
    press(rec, "fail")  # failures can be saved without one
    assert rec.saved == {"success": 0, "failure": 1}

    press(rec, "space")
    step(rec, 2)
    press(rec, "inview", "bowl1")
    assert rec.stage == "place" and not rec.split_auto
    step(rec, 2)
    press(rec, "discard")
    assert rec.phase == "idle" and rec.dataset.num_episodes == 1


def test_bowl_choice_resets_after_save(rec):
    rec.teleop.go_live()
    press(rec, "bowl2", "space")
    step(rec, 1)
    press(rec, "inview")
    step(rec, 1)
    press(rec, "space", "space")
    assert rec.saved["success"] == 1 and rec.bowl is None


def test_frames_skipped_while_not_live_and_time_limit(rec):
    rec.teleop.go_live()
    rec.episode_time_s = 0.0
    press(rec, "space")
    assert rec.phase == "review"  # time limit hit on the first tick
    press(rec, "discard")
    rec.episode_time_s = 60
    press(rec, "space")
    rec.teleop.latest["phase"] = "ready"  # operator pressed hold mid-take
    rec.tick()
    assert rec._frames == 1 and rec._paused_ticks == 1  # start tick recorded, held tick skipped


def test_http_endpoints(rec):
    rec.teleop.go_live()
    rec.streams["inspect"].show_brick()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(rec.streams, rec.teleop, rec, target_s=6))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        assert "<title>Record</title>" in urllib.request.urlopen(base + "/").read().decode()
        for c in ("space", "bowl2"):
            urllib.request.urlopen(urllib.request.Request(base + "/rec/" + c, method="POST"))
        rec.tick()
        state = json.loads(urllib.request.urlopen(base + "/rec/state").read())
        assert state["phase"] == "recording" and state["teleop_live"] and state["bowl"] == "blue"
        assert state["bowls"] == ["red", "blue", "green"]
        assert urllib.request.urlopen(base + "/rec/inspect.jpg").read()[:2] == b"\xff\xd8"
        urllib.request.urlopen(urllib.request.Request(base + "/cmd/hold", method="POST"))
        assert rec.teleop.commands == ["hold"]  # teleop commands still reach teleop_web's handler
    finally:
        srv.shutdown()


def test_place_task_needs_colour_placeholder(rec):
    with pytest.raises(ValueError):
        EpisodeRecorder(rec.teleop, rec.streams, "local/x", place_task="place it somewhere")


def test_page_javascript_parses(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    js = tmp_path / "page.js"
    js.write_text(re.search(r"<script>(.*)</script>", PAGE.replace("__TARGET__", "6"), re.S).group(1))
    r = subprocess.run([node, "--check", str(js)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
