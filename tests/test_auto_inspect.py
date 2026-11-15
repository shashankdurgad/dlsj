"""Automatic spin-and-inspect used by the VLA controller (deploy/smolvla/auto_inspect.py), with the sweep page's fakes."""

import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy" / "smolvla"))

from auto_inspect import AutoInspector  # noqa: E402
from tests.test_yolo_sweep import FakeArm, FakeStream, StubClassifier  # noqa: E402


class Cams:
    """StreamCameras-like: fresh RGB frames from FakeStream threads."""

    def __init__(self, ref, mark_range):
        self.s = FakeStream("inspect", ref, mark_range=mark_range)

    def fresh(self, name, n, timeout_s=2.0):
        frames, seq = [], self.s.seq
        t_end = time.monotonic() + timeout_s
        while len(frames) < n:
            with self.s.cond:
                if not self.s.cond.wait_for(lambda: self.s.seq != seq, timeout=max(0.0, t_end - time.monotonic())):
                    raise TimeoutError("no frame")
                seq, f = self.s.seq, self.s.frame
            frames.append(np.ascontiguousarray(f[..., ::-1]))
        return frames

    def stop(self):
        self.s.stop()


def make(mark_range, tmp_path, **kw):
    from dlsj.inspect.yolo_inspector import YoloInspector

    arm = FakeArm()
    ref = {"arm": arm}
    cams = Cams(ref, mark_range)
    insp = YoloInspector(classifier=StubClassifier("CPU"))
    ai = AutoInspector(cams, insp, (-150.0, 150.0), frames=12, range_deg=180, speed_deg_s=600, settle_ms=0, fps=200,
                       out_root=tmp_path, **kw)
    return arm, cams, ai


def test_marked_brick_goes_to_purple_and_the_wrist_comes_back(tmp_path):
    arm, cams, ai = make((-200.0, 200.0), tmp_path)  # the mark shows on every angle (all-sided marks)
    start = arm.goal().copy()
    res = ai.inspect(arm, start)
    cams.stop()
    assert res.status == "ok" and res.outcome == "cross" and res.bowl == "purple"
    assert res.early_stop and res.frames < 12  # decisive after a few stops
    assert abs(float(arm.read_state()[4]) - float(start[4])) < 2.0  # wrist back at its start angle
    assert abs(float(arm.goal()[5]) - float(start[5])) < 1e-3  # gripper command untouched (keeps squeezing)
    assert res.saved and (Path(res.saved[0]) / "sweep.json").exists()


def test_plain_brick_goes_to_yellow(tmp_path):
    arm, cams, ai = make((999.0, 1000.0), tmp_path)  # never marked
    res = ai.inspect(arm, arm.goal().copy())
    cams.stop()
    assert res.status == "ok" and res.outcome == "none" and res.bowl == "yellow"


def test_lost_grasp_stops_and_reports(tmp_path):
    arm, cams, ai = make((999.0, 1000.0), tmp_path)
    arm.brick_at = 0.0
    start = arm.goal().copy()
    start[5] = 2.0  # gripper closed on nothing -> below the empty threshold
    arm.sim.state[5] = 2.0
    res = ai.inspect(arm, start)
    cams.stop()
    assert res.status == "grasp_lost" and res.bowl == "purple"


def test_abort_stops_the_sweep(tmp_path):
    arm, cams, ai = make((999.0, 1000.0), tmp_path, early_stop=False)
    ai.speed = 30.0
    threading.Timer(0.2, ai.stop).start()
    res = ai.inspect(arm, arm.goal().copy())
    cams.stop()
    assert res.status == "aborted"


def test_continuous_spin_classifies_many_frames_and_saves_every_crop(tmp_path):
    import json as _json

    arm, cams, ai = make((-200.0, 200.0), tmp_path)
    ai.continuous, ai.continuous_speed, ai.early_stop = True, 600.0, False  # 600 deg/s at 200 fps = 1 frame / 3 deg
    start = arm.goal().copy()
    res = ai.inspect(arm, start)
    cams.stop()
    assert res.status == "ok" and res.outcome == "cross" and res.bowl == "purple"
    assert res.frames > 40  # every planned angle, not 12 stops
    out = Path(res.saved[0])
    crops = sorted(out.glob("crop_*.jpg"))
    assert len(crops) == res.frames and len(_json.loads((out / "crops.json").read_text())) == res.frames
    assert len(list(out.glob("frame_*.png"))) <= 12  # full frames subsampled
    assert abs(float(arm.read_state()[4]) - float(start[4])) < 2.0


def test_scaled_counts():
    from dlsj.inspect.aggregate import AggregateConfig

    from auto_inspect import scaled_agg

    base = AggregateConfig(strategy="yolo_topk")
    assert scaled_agg(base, 12) is base
    big = scaled_agg(base, 180)
    assert big.yolo_min_marked_frames == base.yolo_min_marked_frames * 15
    assert big.yolo_max_outlier_frames == base.yolo_max_outlier_frames * 15
