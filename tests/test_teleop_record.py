import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from dlsj.teleop_record import EpisodeRecorder, make_handler

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
        self.frame = np.zeros((h, w, 3), dtype=np.uint8)
        self.frame[..., 0] = 200  # BGR blue channel

    def info(self):
        return {"name": self.name, "index": 0, "fps": 30.0, "size": "64x48", "error": None}


@pytest.fixture
def rec(tmp_path):
    teleop = FakeTeleop()
    streams = {"wrist": FakeStream("wrist"), "top": FakeStream("top")}
    r = EpisodeRecorder(teleop, streams, "local/test_teleop", "pick a brick", fps=30, root=tmp_path / "ds",
                        episode_time_s=60, operator="tester", vcodec="h264")
    r.open_dataset()
    return r


def run_take(r, n, end):
    r.command("space")
    r.tick()
    assert r.phase == "recording"
    for _ in range(n):
        r.teleop.step()
        r.tick()
    r.command(end)
    r.tick()


def test_refuses_to_start_until_follower_is_live(rec):
    rec.command("space")
    rec.tick()
    assert rec.phase == "idle" and "ENGAGE" in rec.message


def test_space_flow_saves_success_failure_and_discards(rec):
    rec.teleop.go_live()
    run_take(rec, 5, "space")  # space while recording -> review
    assert rec.phase == "review"
    rec.command("space")  # accept
    rec.tick()
    run_take(rec, 3, "discard")
    run_take(rec, 4, "fail")  # fail straight from recording
    rec.close()

    assert rec.dataset.num_episodes == 2
    labels = [json.loads(x) for x in (rec.sidecar / "labels.jsonl").read_text().splitlines()]
    assert [(lab["episode_index"], lab["success"], lab["n_frames"], lab["method"]) for lab in labels] == [
        (0, True, 6, "teleop"), (1, False, 5, "teleop")]  # the start tick records a frame too

    from lerobot.datasets import LeRobotDataset

    ds = LeRobotDataset("local/test_teleop", root=rec.root)
    f = ds[1]
    assert f["observation.state"][0].item() == 1.0 and f["action"][0].item() == 1.5  # state=follower, action=cmd
    img = f["observation.images.wrist"]  # (3,H,W) float, RGB: the BGR blue channel lands in channel 2
    assert img[2].mean() > 0.6 and img[0].mean() < 0.1


def test_frames_skipped_while_not_live_and_time_limit(rec):
    rec.teleop.go_live()
    rec.episode_time_s = 0.0
    rec.command("space")
    rec.tick()
    assert rec.phase == "review"  # time limit hit on the first tick
    rec.command("discard")
    rec.tick()
    rec.episode_time_s = 60
    rec.command("space")
    rec.tick()
    rec.teleop.latest["phase"] = "ready"  # operator pressed hold mid-take
    rec.tick()
    assert rec._frames == 1 and rec._paused_ticks == 1  # start tick recorded, held tick skipped


def test_http_endpoints(rec):
    rec.teleop.go_live()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(rec.streams, rec.teleop, rec, target_s=6))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        page = urllib.request.urlopen(base + "/").read().decode()
        assert "SO-101 record" in page and "6" in page
        urllib.request.urlopen(urllib.request.Request(base + "/rec/space", method="POST"))
        rec.tick()
        state = json.loads(urllib.request.urlopen(base + "/rec/state").read())
        assert state["phase"] == "recording" and state["teleop_live"]
        urllib.request.urlopen(urllib.request.Request(base + "/cmd/hold", method="POST"))
        assert rec.teleop.commands == ["hold"]  # teleop commands still reach teleop_web's handler
    finally:
        srv.shutdown()


def test_page_javascript_parses(tmp_path):
    import re
    import shutil
    import subprocess

    from dlsj.teleop_record import PAGE

    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    js = tmp_path / "page.js"
    js.write_text(re.search(r"<script>(.*)</script>", PAGE.replace("__TARGET__", "6"), re.S).group(1))
    r = subprocess.run([node, "--check", str(js)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
