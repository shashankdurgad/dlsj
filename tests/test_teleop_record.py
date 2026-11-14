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
    rec.auto_split = True  # --auto-split: the classifier marks the moment
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
    press(rec, "save")  # save refused: no plate chosen
    assert rec.phase == "review" and "plate" in rec.message.lower()
    press(rec, "bowl3", "save")  # choose green, save
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


def test_failure_without_marking_and_discard(rec):
    rec.teleop.go_live()
    press(rec, "space")
    step(rec, 2)
    press(rec, "fail")  # failures can be saved without an in-view moment
    assert rec.saved == {"success": 0, "failure": 1}
    assert rec.phase == "recording"  # continuous: the next take is already running
    step(rec, 2)
    press(rec, "inview", "bowl1")
    assert rec.stage == "place" and not rec.split_auto
    step(rec, 2)
    press(rec, "discard")
    assert rec.phase == "idle" and rec.dataset.num_episodes == 1


def test_space_marks_in_view_and_classifier_frame_is_kept_separately(rec):
    rec.teleop.go_live()
    press(rec, "space")  # start
    step(rec, 2)
    rec.streams["inspect"].show_brick()
    step(rec, 3)  # classifier fires, but without --auto-split it only records when
    assert rec.stage == "pick" and rec.classifier_frame is not None
    clf = rec.classifier_frame
    step(rec, 2)
    press(rec, "inview")  # operator presses m
    assert rec.stage == "place" and rec.split_frame is not None and not rec.split_auto
    marked = rec.split_frame
    rec.streams["inspect"].show_brick(False)
    press(rec, "bowl2")
    step(rec, 2)
    press(rec, "space")  # stop
    assert rec.phase == "review"
    press(rec, "space")  # space does nothing in review
    assert rec.phase == "review"
    press(rec, "save")  # s
    rec.close()
    lab = json.loads((rec.sidecar / "labels.jsonl").read_text().splitlines()[0])
    assert lab["inspection_frame"] == marked and lab["inspection_marked_by"] == "operator"
    assert lab["classifier_inview_frame"] == clf < marked

    from lerobot.datasets import LeRobotDataset

    ds = LeRobotDataset("local/test_teleop", root=rec.root)
    probs = [float(ds[i]["inspect.prob"].reshape(-1)[0]) for i in range(ds.num_frames)]
    assert probs[0] == 0.0 and max(probs) == 1.0  # colour detector: 0/1 in-view as the per-frame signal
    assert ds[marked]["task"] == "place the brick in the blue bowl"
    assert ds[marked - 1]["task"].startswith("pick up")


def test_plate_choice_persists_across_takes_and_restarts(rec):
    rec.teleop.go_live()
    press(rec, "bowl2", "space")
    step(rec, 1)
    press(rec, "inview")
    step(rec, 1)
    press(rec, "space", "save")
    assert rec.saved["success"] == 1 and rec.bowl == "blue"  # stays selected for the next take
    rec.close()
    again = EpisodeRecorder(FakeTeleop(), rec.streams, "local/test_teleop", root=rec.root, resume=True,
                            bowls=["red", "blue", "green"])
    again.open_dataset()
    assert again.bowl == "blue"  # and after a restart


def test_plate_chosen_after_stopping_is_for_the_next_take(rec):
    import json

    rec.continuous = False
    rec.teleop.go_live()
    press(rec, "bowl1", "space")  # take 0 recorded for red
    step(rec, 2)
    press(rec, "inview")
    step(rec, 2)
    press(rec, "space")  # stop -> review
    press(rec, "bowl2")  # operator picks the next take's plate before saving
    assert rec.bowl == "red" and rec.next_bowl == "blue"
    assert rec.snapshot()["next_bowl"] == "blue"
    press(rec, "save")
    assert rec.bowl == "blue" and rec.next_bowl is None
    press(rec, "space")  # take 1, for blue
    step(rec, 2)
    press(rec, "inview")
    step(rec, 2)
    press(rec, "space", "bowl3", "discard")  # discarding also hands over the queued plate
    assert rec.bowl == "green" and rec.next_bowl is None
    labels = [json.loads(x) for x in (rec.sidecar / "labels.jsonl").read_text().splitlines()]
    assert [x["bowl"] for x in labels] == ["red"]
    tasks = rec.dataset.meta.tasks.index.tolist()
    assert "place the brick in the red bowl" in tasks and "place the brick in the blue bowl" not in tasks


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


class FakeRealSense(FakeStream):
    use_depth = True

    def __init__(self, name, h=48, w=64):
        super().__init__(name, h, w)
        self.depth = np.full((h, w), 700, dtype=np.uint16)  # 0.7 m everywhere
        self.depth[:8, :8] = 0  # no reading


def test_realsense_depth_is_recorded_as_depth_video(tmp_path):
    teleop = FakeTeleop()
    streams = {"top": FakeRealSense("top"), "inspect": FakeStream("inspect")}
    r = EpisodeRecorder(teleop, streams, "local/test_depth", fps=30, root=tmp_path / "ds", vcodec="h264",
                        bowls=["yellow"], detector=BrickInViewDetector(DetectorConfig(stable_frames=2, downscale=1)))
    r.open_dataset()
    assert r.depth_cams == ["top"]
    ft = r.dataset.features["observation.images.top_depth"]
    assert ft["shape"] == (48, 64, 1) and ft["info"]["is_depth_map"]
    assert "observation.images.inspect_depth" not in r.dataset.features  # plain webcam: colour only
    teleop.go_live()
    press(r, "space")
    step(r, 3)
    press(r, "inview", "bowl1")
    step(r, 2)
    press(r, "space", "save")
    r.close()
    assert r.saved["success"] == 1

    from lerobot.datasets import LeRobotDataset

    ds = LeRobotDataset("local/test_depth", root=r.root)
    d = ds[2]["observation.images.top_depth"]
    assert tuple(d.shape[-2:]) == (48, 64)
    mid = float(d.flatten()[d.numel() // 2])
    assert 0.6 < mid < 0.8 or 600 < mid < 800  # 0.7 m back (metres or mm, depending on the decoder)


def test_depth_off_records_colour_only(tmp_path):
    streams = {"top": FakeRealSense("top")}
    r = EpisodeRecorder(FakeTeleop(), streams, "local/test_nodepth", root=tmp_path / "ds", record_depth=False)
    r.open_dataset()
    assert r.depth_cams == [] and "observation.images.top_depth" not in r.dataset.features


def test_continuous_chains_takes_and_enter_pauses(rec):
    assert rec.continuous and rec.streaming_encoding
    rec.teleop.go_live()
    press(rec, "space")  # start take 1
    step(rec, 2)
    press(rec, "inview", "bowl1")  # m, choose the first plate
    step(rec, 2)
    press(rec, "space")  # stop
    press(rec, "save")  # s: save and next
    assert rec.saved["success"] == 1 and rec.phase == "recording" and rec.stage == "pick"  # take 2 running
    step(rec, 2)
    press(rec, "inview", "bowl3")
    step(rec, 2)
    press(rec, "space")  # stop
    press(rec, "enter")  # save and pause
    assert rec.saved["success"] == 2 and rec.phase == "idle"
    press(rec, "fail")  # nothing to save while idle
    assert rec.saved["failure"] == 0
    rec.close()
    labels = [json.loads(x) for x in (rec.sidecar / "labels.jsonl").read_text().splitlines()]
    assert [lab["bowl"] for lab in labels] == ["red", "green"]  # this fixture's bowls


def test_keys_space_starts_and_stops_s_saves_d_discards(rec):
    rec.teleop.go_live()
    press(rec, "space")
    assert rec.phase == "recording" and rec.stage == "pick"
    step(rec, 2)
    press(rec, "space")  # space stops even before the mark
    assert rec.phase == "review" and rec.split_frame is None
    press(rec, "discard")
    assert rec.phase == "idle"
    press(rec, "space")
    step(rec, 2)
    press(rec, "save")  # s while recording stops and tries to save: needs the mark first
    assert rec.phase == "review" and "(m)" in rec.message
    press(rec, "fail")  # f saves as failed, then the next take starts
    assert rec.saved["failure"] == 1 and rec.phase == "recording"


def test_checkpoint_seals_earlier_takes_while_recording_continues(tmp_path):
    import pyarrow.parquet as pq

    streams = {"top": FakeRealSense("top"), "inspect": FakeStream("inspect")}
    r = EpisodeRecorder(FakeTeleop(), streams, "local/test_ckpt", root=tmp_path / "ds", bowls=["yellow"],
                        checkpoint_every=2, detector=BrickInViewDetector(DetectorConfig(stable_frames=2, downscale=1)))
    r.open_dataset()
    r.teleop.go_live()

    def take(n):
        if r.phase == "idle":
            press(r, "space")
        step(r, n)
        press(r, "inview", "bowl1")
        step(r, n)
        press(r, "space", "save")

    take(2)
    take(3)  # second save triggers a checkpoint
    assert r.checkpoints == 1 and r.dataset.num_episodes == 2
    first = tmp_path / "ds" / "data" / "chunk-000" / "file-000.parquet"
    sealed = pq.read_table(first, columns=["episode_index"]).to_pandas()  # readable while still recording
    assert sorted(sealed.episode_index.unique()) == [0, 1]
    take(4)  # continues into a new data file
    r.close()

    from lerobot.datasets import LeRobotDataset

    ds = LeRobotDataset("local/test_ckpt", root=tmp_path / "ds")
    assert ds.num_episodes == 3
    assert len(list((tmp_path / "ds" / "data").glob("*/*.parquet"))) == 2
    labels = (r.sidecar / "labels.jsonl").read_text().splitlines()
    assert len(labels) == 3
    assert float(ds[ds.num_frames - 1]["observation.images.top_depth"].float().median()) > 0


def test_pick_only_take_stops_after_the_hold_and_needs_no_plate(tmp_path):
    teleop = FakeTeleop()
    streams = {"wrist": FakeStream("wrist"), "inspect": FakeStream("inspect")}
    det = BrickInViewDetector(DetectorConfig(stable_frames=2, downscale=1))
    r = EpisodeRecorder(teleop, streams, "local/test_pick", fps=30, root=tmp_path / "ds", episode_time_s=60,
                        bowls=["red", "blue", "green"], detector=det, pick_only=True, pick_hold_s=0.1)
    r.open_dataset()
    teleop.go_live()
    press(r, "space")
    step(r, 5)  # picking
    press(r, "inview")
    mark = r.split_frame
    assert r.stage == "pick" and r.phase == "recording"  # still recording the hold, still the pick task
    press(r, "inview")  # a second m does not move the mark
    assert r.split_frame == mark
    step(r, 5)  # hold is 0.1 s = 3 frames, then the take stops itself
    assert r.phase == "review" and "s to save" in r.message
    frames = r._frames
    step(r, 3)  # nothing more is recorded in review
    assert r._frames == frames
    press(r, "save")  # no plate chosen, still saved
    assert r.phase == "idle" and r.saved["success"] == 1  # no chaining into a next take
    r.close()

    lab = json.loads((r.sidecar / "labels.jsonl").read_text().splitlines()[0])
    assert lab["mode"] == "pick_only" and lab["inspection_frame"] == mark
    assert lab["bowl"] is None and lab["place_task"] is None
    from lerobot.datasets import LeRobotDataset

    ds = LeRobotDataset("local/test_pick", root=r.root)
    assert ds.num_frames == frames == mark + 3
    assert {ds[i]["task"] for i in range(ds.num_frames)} == {
        "pick up the brick and hold it in front of the inspection camera"}


def test_full_takes_are_labelled_full(rec):
    rec.teleop.go_live()
    press(rec, "bowl1", "space")
    step(rec, 3)
    press(rec, "inview")
    step(rec, 3)
    press(rec, "space", "save")
    rec.close()
    lab = json.loads((rec.sidecar / "labels.jsonl").read_text().splitlines()[0])
    assert lab["mode"] == "full" and lab["bowl"] == "red"


def test_mode_toggles_only_between_takes(rec):
    assert not rec.pick_only
    press(rec, "toggle_pick")
    assert rec.pick_only and "Pick only" in rec.message
    rec.teleop.go_live()
    press(rec, "space")
    press(rec, "toggle_pick")  # refused mid-take
    assert rec.pick_only and "between takes" in rec.message
    step(rec, 3)
    press(rec, "inview")
    step(rec, 12)  # default 0.3 s hold = 9 frames, then review
    assert rec.phase == "review"
    press(rec, "save")
    press(rec, "toggle_pick")  # back to full demos
    assert not rec.pick_only and rec.snapshot()["pick_only"] is False
    rec.close()
    lab = json.loads((rec.sidecar / "labels.jsonl").read_text().splitlines()[0])
    assert lab["mode"] == "pick_only"


def test_two_marks_records_the_pickup_then_the_in_view_split(tmp_path):
    teleop = FakeTeleop()
    streams = {"wrist": FakeStream("wrist"), "inspect": FakeStream("inspect")}
    det = BrickInViewDetector(DetectorConfig(stable_frames=2, downscale=1))
    r = EpisodeRecorder(teleop, streams, "local/test_marks", fps=30, root=tmp_path / "ds", episode_time_s=60,
                        bowls=["red", "blue", "green"], detector=det, continuous=False)
    r.open_dataset()
    teleop.go_live()
    press(r, "toggle_marks")
    assert r.two_marks and r.snapshot()["two_marks"]
    press(r, "bowl1", "space")
    step(r, 4)  # reach and grasp
    press(r, "inview")  # first M: pickup done
    assert r.grasp_frame is not None and r.split_frame is None and r.stage == "pick"
    grasp = r.grasp_frame
    step(r, 4)  # carry to the camera
    press(r, "inview")  # second M: in view, as before
    assert r.split_frame is not None and r.split_frame > grasp and r.stage == "place"
    split = r.split_frame
    press(r, "inview")  # a third M changes nothing
    assert (r.grasp_frame, r.split_frame) == (grasp, split)
    step(r, 4)
    press(r, "space")
    press(r, "toggle_marks")  # can't switch mid-take
    assert r.two_marks and "between takes" in r.message
    press(r, "save")
    r.close()
    lab = json.loads((r.sidecar / "labels.jsonl").read_text().splitlines()[0])
    assert lab["grasp_frame"] == grasp and lab["grasp_marked_by"] == "operator" and lab["inspection_frame"] == split
    from lerobot.datasets import LeRobotDataset

    ds = LeRobotDataset("local/test_marks", root=tmp_path / "ds")
    tasks = [ds[i]["task"] for i in range(ds.num_frames)]
    assert all(t == r.pick_task for t in tasks[:split]) and all("red" in t for t in tasks[split:])  # unchanged


def test_one_mark_by_default_and_no_grasp_frame(tmp_path, rec):
    rec.teleop.go_live()
    assert not rec.two_marks
    press(rec, "bowl1", "space")
    step(rec, 2)
    press(rec, "inview")
    assert rec.split_frame is not None and rec.grasp_frame is None
