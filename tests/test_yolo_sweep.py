"""YOLO-only spin-and-inspect: 3-class sweep aggregation (`yolo_topk`, `yolo_mean`), the YOLO inspector with a
stub classifier, and the live page's endpoints with fake cameras and a fake arm (no OpenVINO, no hardware)."""

import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from dlsj.contracts import FrameInspection, Label, MarkType
from dlsj.inspect import aggregate as A
from dlsj.inspect.aggregate import AggregateConfig, aggregate, aggregate_all, sweep_result

PLAIN = {"plain": 0.95, "circle": 0.03, "cross": 0.02}
CROSS = {"plain": 0.04, "circle": 0.06, "cross": 0.90}
CIRCLE = {"plain": 0.05, "circle": 0.88, "cross": 0.07}
UNSURE = {"plain": 0.45, "circle": 0.30, "cross": 0.25}
WEAK_CROSS = {"plain": 0.30, "circle": 0.10, "cross": 0.60}  # marked-looking but below the per-frame gate


def yframes(seq):
    out = []
    for i, p in enumerate(seq):
        conf, top, c, m = A.gate(p, 0.8, 0.2)
        lab = Label.PERFECT if conf == "plain" else Label.DEFECTIVE if conf in ("circle", "cross") else Label.UNCERTAIN
        out.append(FrameInspection(i, -90 + 15 * i, 1 - p["plain"], lab, None, MarkType(conf) if lab is Label.DEFECTIVE
                                   else None, c, None, 1.0, None, dict(p)))
    return out


def outcome(seq, strategy="yolo_topk", cfg=AggregateConfig()):
    return aggregate(yframes(seq), 0.5, 0.2, 0.8, cfg, strategy).outcome()


# ----------------------------------------------------------------------------- aggregation
def test_one_face_mark_is_found_by_yolo_topk_but_missed_by_the_release_mean():
    seq = [PLAIN] * 9 + [CROSS, CROSS, WEAK_CROSS]  # the mark faces the camera on 3 of 12 angles
    assert outcome(seq) is MarkType.CROSS
    # the mean of all frames is dominated by the plain angles: never cross (here: too weak -> uncertain)
    assert outcome(seq, "yolo_mean") is MarkType.UNKNOWN
    assert outcome([PLAIN] * 11 + [CROSS], "yolo_mean") is MarkType.NONE  # one marked angle: plain wins the mean
    a = aggregate(yframes(seq), 0.5, 0.2, 0.8, AggregateConfig(), "yolo_topk")
    assert a.summary["mark_frames"] == [9, 10] and a.summary["n_cross"] == 2
    assert len(a.summary["distribution"]) == 12 and a.summary["distribution"][9]["confident"] == "cross"


def test_single_glare_frame_does_not_flip_a_plain_brick():
    assert outcome([PLAIN] * 11 + [CROSS]) is MarkType.NONE
    assert outcome([PLAIN] * 10 + [UNSURE, PLAIN]) is MarkType.NONE


def test_all_plain_is_plain_for_both_rules():
    assert outcome([PLAIN] * 12) is MarkType.NONE
    assert outcome([PLAIN] * 12, "yolo_mean") is MarkType.NONE


def test_circle_cross_conflict_is_uncertain():
    seq = [PLAIN] * 8 + [CIRCLE, CIRCLE, CROSS, CROSS]
    a = aggregate(yframes(seq), 0.5, 0.2, 0.8, AggregateConfig(), "yolo_topk")
    assert a.label is Label.DEFECTIVE and a.mark is MarkType.UNKNOWN and a.outcome() is MarkType.UNKNOWN
    assert "conflict" in a.summary["mark_reason"]


def test_clear_majority_circle_wins_and_two_to_one_is_unknown():
    assert outcome([PLAIN] * 6 + [CIRCLE] * 5 + [CROSS]) is MarkType.CIRCLE
    assert outcome([PLAIN] * 9 + [CIRCLE, CIRCLE, CROSS]) is MarkType.UNKNOWN  # 2:1 is below the 0.4 margin


def test_mostly_unsure_is_uncertain_and_one_marked_frame_is_not_enough():
    assert outcome([UNSURE] * 8 + [PLAIN] * 4) is MarkType.UNKNOWN
    assert outcome([PLAIN] * 7 + [UNSURE] * 4 + [CROSS]) is MarkType.UNKNOWN  # 7/12 plain < 80 %


def test_release_mean_rule_marks_a_brick_shown_mostly_marked():
    assert outcome([CROSS] * 11 + [PLAIN], "yolo_mean") is MarkType.CROSS
    assert outcome([CROSS] * 10 + [PLAIN] * 2, "yolo_mean") is MarkType.UNKNOWN  # mean P(cross) 0.75 < 0.8


def test_sweep_result_carries_yolo_alternatives_and_probs_round_trip():
    fr = yframes([PLAIN] * 9 + [CROSS] * 3)
    r = sweep_result(fr, 0.5, 0.2, 0.8, AggregateConfig(strategy="yolo_topk"), "yolo_topk")
    assert r.aggregation == "yolo_topk" and r.label is Label.DEFECTIVE and r.mark is MarkType.CROSS
    assert set(r.summary["alternatives"]) == set(A.ALL_STRATEGIES)
    assert r.summary["alternatives"]["yolo_mean"] in ("none", "unknown")
    rec = A.frames_record(fr)
    assert rec[0]["probs"]["plain"] == pytest.approx(0.95)
    back = A.frames_from_record(json.loads(json.dumps(rec)))
    assert back[10].probs == pytest.approx(CROSS)


def test_anomaly_frames_keep_the_old_strategies_and_yolo_needs_probs():
    fr = [FrameInspection(i, None, 0.1, Label.PERFECT) for i in range(4)]
    assert set(aggregate_all(fr, 0.5, 0.4, 0.6)) == set(A.STRATEGIES)
    with pytest.raises(ValueError):
        aggregate(fr, 0.5, 0.4, 0.6, AggregateConfig(), "yolo_topk")
    with pytest.raises(ValueError):
        AggregateConfig(strategy="nope")


# ----------------------------------------------------------------------------- inspector (stub model)
class StubClassifier:
    """Stands in for Tan's YoloClassifier: BGR 320x320 in; 'cross' when the crop is blue (B channel), else plain."""

    def __init__(self, device="NPU"):
        self.device, self.precision = device, "f32"
        self.config = {"min_confidence": 0.8, "min_margin": 0.2, "classes": ["circle", "cross", "plain"]}
        self.calls = 0

    def health(self):
        return {"model": "stub", "device": self.device}

    def predict(self, bgr):
        assert bgr.shape == (320, 320, 3) and bgr.dtype == np.uint8
        self.calls += 1
        blue = bgr[..., 0].mean() > 128 and bgr[..., 2].mean() < 100
        return dict(CROSS) if blue else dict(PLAIN)


def blue_rgb(blue: bool, h=480, w=640):
    img = np.full((h, w, 3), 90, np.uint8)
    if blue:
        img[..., 2] = 230  # RGB blue
        img[..., 0] = 30
    return img


def test_inspector_converts_rgb_to_bgr_and_maps_frames():
    pytest.importorskip("cv2")
    from dlsj.inspect.yolo_inspector import YoloInspector

    insp = YoloInspector(classifier=StubClassifier(), roi=(160, 80, 480, 400))
    c = insp.classify(blue_rgb(True))
    assert c["confident"] == "cross" and c["box"] == (160, 80, 480, 400) and c["source"] == "roi"
    assert c["crop"].shape == (320, 320, 3)
    rec, heat = insp.inspect_frame(blue_rgb(True), 3, 42.0)
    assert heat is None and rec.index == 3 and rec.angle_deg == 42.0
    assert rec.label is Label.DEFECTIVE and rec.mark is MarkType.CROSS and rec.probs["cross"] == pytest.approx(0.9)
    rec2, _ = insp.inspect_frame(blue_rgb(False), 0, 0.0)
    assert rec2.label is Label.PERFECT and rec2.region is None and rec2.score == pytest.approx(0.05)


def test_inspector_sweep_and_classic_inspect():
    pytest.importorskip("cv2")
    from dlsj.inspect.yolo_inspector import YoloInspector

    insp = YoloInspector(classifier=StubClassifier())
    frames = [blue_rgb(k in (4, 5, 6)) for k in range(12)]
    r = insp.inspect_sweep(frames, list(range(-90, 90, 15)))
    assert r.aggregation == "yolo_topk" and r.mark is MarkType.CROSS and len(r.frames) == 12
    assert r.summary["alternatives"]["yolo_mean"] in ("none", "unknown")
    s = insp.start_sweep()
    for k, f in enumerate(frames):
        s.add(f, float(k))
    assert s.result().mark is MarkType.CROSS
    classic = insp.inspect([blue_rgb(True)] * 3)
    assert classic.aggregation == "yolo_mean" and classic.mark is MarkType.CROSS


def test_load_classifier_falls_back_from_npu(monkeypatch):
    pytest.importorskip("cv2")
    from dlsj.inspect import yolo_classifier, yolo_inspector

    class Fake(StubClassifier):
        def __init__(self, directory, device="CPU"):
            if device == "NPU":
                raise RuntimeError("NPU not available")
            super().__init__(device)

    monkeypatch.setattr(yolo_classifier, "YoloClassifier", Fake)
    monkeypatch.setattr(yolo_inspector, "_compile_default_precision",
                        lambda cls, d, dev: (_ for _ in ()).throw(RuntimeError(f"{dev} down")))
    clf, notes = yolo_inspector.load_classifier("/nowhere", "NPU")
    assert clf.device == "GPU" and any("NPU" in n for n in notes)
    assert yolo_inspector.fallback_chain("gpu") == ["GPU", "NPU", "CPU"]


# ----------------------------------------------------------------------------- web page (fake cams + arm)
class FakeStream:
    """A camera thread: BGR frames, blue while the arm's wrist_roll is inside `mark_range` (a one-face mark)."""

    def __init__(self, name, arm_ref, mark_range=(10.0, 60.0), hz=120.0):
        import cv2

        self.name, self.arm_ref, self.mark_range = name, arm_ref, mark_range
        self.cond, self.seq, self.frame, self.jpeg = threading.Condition(), 0, None, None
        self._cv2, self._stop = cv2, threading.Event()
        threading.Thread(target=self._run, args=(1.0 / hz,), daemon=True).start()

    def _run(self, period):
        from dlsj.loop.sweep import ROLL

        while not self._stop.is_set():
            arm = self.arm_ref.get("arm")
            roll = float(arm.read_state()[ROLL]) if arm is not None else 0.0
            blue = self.name == "inspect" and self.mark_range[0] <= roll <= self.mark_range[1]
            f = np.ascontiguousarray(blue_rgb(blue)[..., ::-1])
            ok, buf = self._cv2.imencode(".jpg", f)
            with self.cond:
                self.frame, self.jpeg, self.seq = f, buf.tobytes(), self.seq + 1
                self.cond.notify_all()
            time.sleep(period)

    def info(self):
        return {"name": self.name, "fps": 120, "error": None}

    def stop(self):
        self._stop.set()


class FakeArm:
    """A real-arm stand-in: records every command; follows it with a lag."""

    def __init__(self):
        from dlsj.inspect.sweep_live import SimArm

        self.sim = SimArm(np.array([0, -40, 50, 0, 0, 30], np.float32))
        self.sends = 0
        self.lock = threading.Lock()

    def read_state(self):
        with self.lock:
            return self.sim.read_state()

    def send(self, t):
        with self.lock:
            self.sends += 1
            self.sim.send(t)

    def read_gripper_current(self):
        return None

    def goal(self):
        with self.lock:
            return self.sim.goal()


@pytest.fixture
def web(tmp_path):
    pytest.importorskip("cv2")
    from dlsj.inspect.sweep_live import App, Cams, make_handler
    from dlsj.inspect.yolo_inspector import YoloInspector

    ref = {"arm": None}
    cams = Cams({n: FakeStream(n, ref) for n in ("inspect", "wrist", "top")})
    insp = YoloInspector(classifier=StubClassifier("CPU"))
    poses = {"inspect": np.array([5, -35, 45, 5, 0, 0], np.float32)}

    def factory():
        ref["arm"] = FakeArm()
        return ref["arm"]

    app = App(cams, insp, tmp_path / "sweeps", arm=None, fps=200.0, live_hz=50.0, arm_factory=factory,
              pose_loader=lambda fid: poses["inspect"], limits_loader=lambda fid: (-150.0, 150.0))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    yield app, base, ref, tmp_path / "sweeps"
    app.closing.set()
    srv.shutdown()
    cams.stop()


def get(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, r.headers.get("Content-Type"), r.read()


def post(url, body=None):
    req = urllib.request.Request(url, json.dumps(body or {}).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def wait_job(base, states=("done", "aborted", "error", "grasp_lost", "camera_error"), timeout=30):
    t_end = time.time() + timeout
    while time.time() < t_end:
        st = json.loads(get(base + "/api/state")[2])
        if st["job"] and st["job"]["state"] in states:
            return st
        time.sleep(0.05)
    raise AssertionError("job did not finish")


FAST = {"frames": 8, "range_deg": 140, "settle_ms": 0, "speed": 120}


def test_page_state_snapshot_and_stream_without_moving_anything(web):
    app, base, ref, _ = web
    code, ctype, body = get(base + "/")
    assert code == 200 and b"Spin &amp; inspect" in body and "text/html" in ctype
    st = json.loads(get(base + "/api/state")[2])
    assert st["settings"]["frames"] == 12 and st["settings"]["range_deg"] == 180 and st["settings"]["dry_run"]
    assert "yolo_topk" in st["strategies"] and st["job"] is None
    code, ctype, jpg = get(base + "/snapshot/inspect.jpg")
    assert code == 200 and ctype == "image/jpeg" and jpg[:2] == b"\xff\xd8"
    with urllib.request.urlopen(base + "/stream/wrist", timeout=5) as r:
        assert "multipart/x-mixed-replace" in r.headers["Content-Type"]
        assert b"--frame" in r.read(2000)
    arm = json.loads(get(base + "/api/arm")[2])
    assert arm["connected"] is False and ref["arm"] is None  # page load never connects or moves the arm


def test_classify_live(web):
    app, base, _, _ = web
    assert post(base + "/api/live", {"on": True}) == (200, {"live": True})
    t_end = time.time() + 5
    while time.time() < t_end:
        st = json.loads(get(base + "/api/state")[2])
        if st["live"] and st["live"]["mean3"].get("ready"):
            break
        time.sleep(0.05)
    assert st["live"]["confident"] == "plain" and st["live"]["mean3"]["decision"] == "plain"
    assert st["latency"]["CPU"]["n"] >= 3
    assert post(base + "/api/live", {"on": False})[1] == {"live": False}


def test_dry_run_sweep_without_an_arm_saves_and_decides(web):
    app, base, _, out = web
    code, r = post(base + "/api/sweep", {**FAST, "dry_run": True})
    assert code == 200
    st = wait_job(base)
    job = st["job"]
    assert job["state"] == "done" and len(job["frames"]) == 8
    assert job["decision"]["outcome"] == "none" and job["decision"]["bowl"] == "yellow"
    assert job["decision"]["naive"]["outcome"] == "none"
    d = out / job["id"]
    doc = json.loads((d / "sweep.json").read_text())
    assert doc["source"].startswith("sweep_live") and len(doc["frames"]) == 8
    assert (d / "frame_00.png").exists() and (d / "crop_07.jpg").exists()
    assert doc["inspection"]["per_frame"][0]["probs"]["plain"] > 0.9
    assert "yolo_topk" in doc["inspection"]["strategies"]
    hist = json.loads(get(base + "/api/history")[2])
    assert hist[0]["id"] == job["id"] and hist[0]["outcome"] == "none"
    assert get(base + f"/thumb/{job['id']}/3.jpg")[2][:2] == b"\xff\xd8"
    assert json.loads(get(base + f"/api/history/{job['id']}")[2])["decision"]["bowl"] == "yellow"


def test_real_sweep_needs_an_arm_then_finds_a_one_face_mark_and_returns(web):
    app, base, ref, _ = web
    code, r = post(base + "/api/sweep", {**FAST, "dry_run": False})
    assert code == 409 and "arm" in r["error"]
    assert post(base + "/api/arm/connect")[0] == 200
    arm = ref["arm"]
    assert arm.sends == 0  # connecting holds; nothing is commanded
    code, _ = post(base + "/api/sweep", {**FAST, "dry_run": False, "range_deg": 160})
    assert code == 200
    job = wait_job(base)["job"]
    assert job["state"] == "done", job["message"]
    assert arm.sends > 0
    marked = [f for f in job["frames"] if f["confident"] == "cross"]
    assert len(marked) >= 2 and len(marked) < len(job["frames"])  # the mark shows on a few angles only
    assert job["decision"]["outcome"] == "cross" and job["decision"]["bowl"] == "purple"
    assert job["decision"]["naive"]["outcome"] != "cross"  # the release mean misses the one-face mark
    assert abs(float(arm.read_state()[4]) - 0.0) < 2.0  # wrist back at its start angle
    assert job["sweep"]["returned"]


def test_dry_run_with_an_arm_connected_never_commands_it(web):
    app, base, ref, _ = web
    post(base + "/api/arm/connect")
    arm = ref["arm"]
    post(base + "/api/sweep", {**FAST, "dry_run": True})
    assert wait_job(base)["job"]["state"] == "done"
    assert arm.sends == 0


def test_stop_mid_sweep_returns_the_wrist(web):
    app, base, ref, _ = web
    post(base + "/api/arm/connect")
    arm = ref["arm"]
    post(base + "/api/sweep", {"frames": 12, "range_deg": 180, "settle_ms": 300, "speed": 60, "dry_run": False})
    t_end = time.time() + 10
    while time.time() < t_end and len(json.loads(get(base + "/api/state")[2])["job"]["frames"]) < 2:
        time.sleep(0.02)
    assert post(base + "/api/stop")[0] == 200
    job = wait_job(base)["job"]
    assert job["state"] == "aborted" and job["decision"] is None
    assert job["sweep"]["returned"] and abs(float(arm.read_state()[4])) < 2.0
    assert post(base + "/api/sweep", {**FAST, "dry_run": True})[0] == 200  # the motion lock was released
    wait_job(base)


def test_settings_are_clamped_and_validated(web):
    app, base, _, _ = web
    code, s = post(base + "/api/settings", {"speed": 999, "frames": 0, "range_deg": 1000})
    assert code == 200 and s["speed"] == 120 and s["frames"] == 1 and s["range_deg"] == 270
    assert post(base + "/api/settings", {"strategy": "bogus"})[0] == 400
    code, s = post(base + "/api/settings", {"roi": "100,50,420,370"})
    assert code == 200 and s["roi"] == [100, 50, 420, 370] and app.insp.roi == (100, 50, 420, 370)
    assert post(base + "/api/settings", {"roi": "5,5,1,1"})[0] == 400
    assert post(base + "/api/settings", {"roi": ""})[1]["roi"] is None and app.insp.roi is None


def test_goto_inspect_pose_ramps_and_keeps_the_gripper(web):
    app, base, ref, _ = web
    assert post(base + "/api/pose/inspect")[0] == 409  # no arm
    post(base + "/api/arm/connect")
    arm = ref["arm"]
    app.ramp_speed = 400.0
    assert post(base + "/api/pose/inspect")[0] == 200
    st = wait_job(base)
    assert st["job"]["state"] == "done"
    g = arm.goal()
    assert abs(g[0] - 5) < 0.6 and abs(g[5] - 30) < 1e-3  # pan reached; gripper command unchanged
