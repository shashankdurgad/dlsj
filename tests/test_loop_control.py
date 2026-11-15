"""loop_control.LoopController: the full sort cycle and its recovery paths on a fake arm with a fake clock."""

import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy" / "smolvla"))
import auto_control as A  # noqa: E402
import loop_control as LC  # noqa: E402
import smolvla_control as S  # noqa: E402


class Clock:
    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now

    def time(self):
        return self.now

    def strftime(self, f):
        return "test"

    def sleep(self, s):
        import time
        time.sleep(min(s, 0.01))


class FakeArm:
    def read(self):
        return np.zeros(6)

    def send(self, t):
        pass


class FakeBase:
    """Minimal pi05_control.Controller + SmolController: home -> idle, start -> running/pick."""

    def __init__(self, arm, cameras, policy, *a, **kw):
        self.arm, self.cameras, self.policy, self.fps = arm, cameras, policy, 30
        self.phase, self.stage, self.bowl, self.message = "idle", "pick", None, ""
        self.cmd, self.measured = np.zeros(6), np.zeros(6)
        self.lock, self.bowls, self.brick = threading.RLock(), ["yellow", "purple", "orange"], ""

    def _replan_now(self):
        pass

    def _handle(self, c):
        if c in ("q", "stop"):
            self.phase = "stopped"
        elif c == "home":
            self.phase, self.stage = "homing", "pick"
        elif c in ("start", "space") and self.phase not in ("running", "readying"):
            self.phase, self.stage = "running", getattr(self, "start_stage", "pick")

    def tick(self):
        if self.phase == "homing":
            self.phase = "idle"

    def _step(self, target):
        self.cmd = np.asarray(target, float).copy()
        self.measured = self.cmd.copy()
        return self.measured

    def snapshot(self):
        return {"phase": self.phase}


class StubInspector:
    def __init__(self, outcome="cross", bowl="purple", status="ok"):
        self.r = SimpleNamespace(outcome=outcome, bowl=bowl, reason="r", status=status, attempts=1, frames=5,
                                 early_stop=True, seconds=1.2, saved=[])
        self.insp, self.listener, self.cameras, self.done = SimpleNamespace(detector=None), None, None, threading.Event()

    def inspect(self, arm, start_cmd):
        self.done.set()
        return self.r

    def stop(self):
        pass


@pytest.fixture
def build(monkeypatch, tmp_path):
    clock = Clock()

    def make(inspector=None, fixed_pose=False):
        insp = inspector or StubInspector()
        monkeypatch.setattr(LC, "time", clock)
        monkeypatch.setattr(LC.LCFG, "log_dir", str(tmp_path))
        monkeypatch.setattr(LC.LCFG, "bring_mode", fixed_pose if isinstance(fixed_pose, str) else
                            ("fixed" if fixed_pose else "vla_offset"))
        monkeypatch.setattr(LC, "_make_top_detector", lambda: None)
        monkeypatch.setattr(S, "_manual_make_controller_class", lambda b: b, raising=False)
        monkeypatch.setattr(S, "BRING", "bring the brick to the inspection camera")
        fake_ai = types.ModuleType("auto_inspect")
        fake_ai.make_auto_inspector = lambda *a, **kw: insp
        fake_ai.BOWL = {"none": "yellow", "circle": "orange", "cross": "purple", "marked": "purple"}
        fake_ai.FALLBACK_BOWL = "purple"
        fake_hw = types.ModuleType("dlsj.loop.hardware")
        fake_hw.StreamCameras = lambda c: c
        monkeypatch.setitem(sys.modules, "auto_inspect", fake_ai)
        monkeypatch.setitem(sys.modules, "dlsj.loop.hardware", fake_hw)
        A.CFG.__dict__.update(inspect_device="NPU", roi=None, inspect_frames=12, inspect_range=180.0,
                              inspect_speed=60.0)
        cls = LC.make_loop_controller_class(FakeBase)
        c = cls(FakeArm(), {"wrist": SimpleNamespace(frame=None), "top": SimpleNamespace(frame=None),
                            "inspect": SimpleNamespace(frame=None)}, None)
        c._perc_stop.set()
        return c, clock, insp
    return make


def run(c, clock, seconds, grip=None, cmd_grip=None, pose=None):
    for _ in range(int(seconds * 30)):
        clock.now += 1 / 30
        if grip is not None:
            c.measured[5] = grip
        if cmd_grip is not None:
            c.cmd[5] = cmd_grip
        if pose is not None:
            c.measured[:5] = pose
        c.tick()


def see_bricks(c, clock, n, colour="blue", frames=12):
    """`n` bricks steady on the table (same place in every frame) over `frames` survey frames."""
    t = clock.now
    with c.lock:
        for k in range(frames):
            c._count_hist.append((t + k * 0.25, [_brick(100 + 60 * i, 300, 0.8, colour) for i in range(n)]))


def test_full_cycle_then_done(build):
    c, clock, insp = build()
    c.loop_command("start")
    run(c, clock, 0.1)
    assert c.L["step"] == "survey"
    run(c, clock, 1.1)
    see_bricks(c, clock, 2, "blue")
    clock.now += 3.0
    run(c, clock, 0.1)
    assert c.L["step"] == "pick" and c.brick == "blue" and c.stage == "pick"
    run(c, clock, 1.0, grip=30.0, cmd_grip=30.0)
    assert c.L["step"] == "pick"
    run(c, clock, 1.0, grip=5.6, cmd_grip=0.0)  # squeezing at brick width
    assert c.L["step"] == "bring" and c.stage == "bring"
    with c.lock:
        c.perc["inspect_seen"] = {"t": clock.now + 10, "conf": 0.7, "close": True}
    run(c, clock, 5.0, grip=5.6, cmd_grip=0.0, pose=np.ones(5))
    assert insp.done.wait(2.0)
    for _ in range(100):
        if c.phase != "inspecting":
            break
        threading.Event().wait(0.01)
    run(c, clock, 0.1, grip=5.6, cmd_grip=0.0)
    assert c.L["step"] == "place" and c.bowl == "purple" and c.stage == "place"
    run(c, clock, 1.5, grip=30.0, cmd_grip=30.0)
    run(c, clock, 1.0, grip=30.0, cmd_grip=30.0)
    assert c.L["tallies"] == {"purple": 1} and c.L["cycles"][-1]["result"] == "sorted"
    # no trip home: it holds above the plate and looks from there
    assert c.L["step"] == "survey" and c.L["survey_from"] == "plate" and c.phase == "holding"
    run(c, clock, 1.1)
    see_bricks(c, clock, 0)
    clock.now += 3.0
    run(c, clock, 0.1)
    # nothing seen from up there: back home (the fake arm arrives at once), and only the view from home may end it
    assert c.L["step"] in ("home", "survey") and c.L["survey_from"] == "home" and c.L["mode"] == "running"
    run(c, clock, 0.2)
    assert c.L["step"] == "survey" and c.L["survey_from"] == "home"
    run(c, clock, 1.1)
    see_bricks(c, clock, 0)
    clock.now += 3.0
    run(c, clock, 0.1)
    # the survey alone never ends the loop: the YOLO verification has to agree
    assert c.L["mode"] == "running" and c.L["step"] == "verify"
    feed_verify(c, [[]] * LC.LCFG.verify_frames)
    run(c, clock, 0.1)
    assert c.L["mode"] == "done" and c.L["step"] == "done" and c.perc["verify"]["state"] == "clear"


def feed_verify(c, tables):
    with c.lock:
        c._verify_frames = list(tables)


def _brick(x, y, conf=0.5, colour="blue"):
    return LC.P.Brick((x, y, x + 40, y + 40), conf, colour, 0.8)


def _to_verify(c, clock):
    c.loop_command("start")
    run(c, clock, 1.2)
    see_bricks(c, clock, 0)
    clock.now += 3.0
    run(c, clock, 0.1)
    assert c.L["step"] == "verify"


def test_verify_finds_a_brick_and_goes_back_to_work(build):
    c, clock, _ = build()
    _to_verify(c, clock)
    n = LC.LCFG.verify_frames
    feed_verify(c, [[_brick(200, 300)]] * (n // 2 + 1) + [[]] * (n - n // 2 - 1))
    run(c, clock, 0.1)
    assert c.L["mode"] == "running" and c.L["step"] == "survey" and c.L["survey_from"] == "home"
    assert c.perc["verify"]["state"] == "bricks" and len(c.perc["verify"]["bricks"]) == 1


def test_verify_ignores_a_one_frame_flicker(build):
    c, clock, _ = build()
    _to_verify(c, clock)
    feed_verify(c, [[_brick(200, 300)]] + [[]] * (LC.LCFG.verify_frames - 1))
    run(c, clock, 0.1)
    assert c.L["mode"] == "done"


def test_verify_keeps_finding_bricks_then_pauses(build):
    c, clock, _ = build()
    _to_verify(c, clock)
    for k in range(LC.LCFG.max_verify_fails + 1):
        assert c.L["step"] == "verify"
        feed_verify(c, [[_brick(200, 300)]] * LC.LCFG.verify_frames)
        run(c, clock, 0.1)
        if k < LC.LCFG.max_verify_fails:
            assert c.L["step"] == "survey"
            run(c, clock, 1.2)
            see_bricks(c, clock, 0)  # the survey still cannot see it
            clock.now += 3.0
            run(c, clock, 0.1)
    assert c.L["mode"] == "paused" and "Verification" in c.L["message"]


def test_verify_without_top_frames_pauses(build):
    c, clock, _ = build()
    _to_verify(c, clock)
    clock.now += LC.LCFG.verify_timeout + 1
    run(c, clock, 0.1)
    assert c.L["mode"] == "paused"


def test_pick_timeout_goes_home_and_retries(build):
    c, clock, _ = build()
    c.loop_command("start")
    run(c, clock, 1.2)
    see_bricks(c, clock, 1)
    clock.now += 3.0
    run(c, clock, 0.1)
    assert c.L["step"] == "pick"
    run(c, clock, LC.LCFG.pick_timeout + 1, grip=1.0, cmd_grip=0.0)  # closing on nothing
    assert c.L["attempts"] == 1 and c.L["cycles"][-1]["result"] == "pick timeout"
    assert c.L["step"] in ("home", "survey")


def test_five_missed_grasps_go_home_and_try_again(build):
    c, clock, _ = build()
    c.loop_command("start")
    run(c, clock, 1.2)
    see_bricks(c, clock, 1)
    clock.now += 3.0
    run(c, clock, 0.1)
    assert c.L["step"] == "pick"
    for _ in range(4):  # open, close on nothing: four misses, still picking
        run(c, clock, 1.0, grip=37.0, cmd_grip=37.0)
        run(c, clock, 1.0, grip=1.5, cmd_grip=0.0)
    assert c.L["step"] == "pick" and c._pick_watch.misses == 4
    run(c, clock, 1.0, grip=37.0, cmd_grip=37.0)
    run(c, clock, 0.2, grip=1.5, cmd_grip=0.0)  # fifth miss
    assert c.L["attempts"] == 1 and c.L["cycles"][-1]["result"] == "pick stopped: misses"
    assert c.L["step"] in ("home", "survey")
    run(c, clock, 1.2)
    see_bricks(c, clock, 1)
    clock.now += 3.0
    run(c, clock, 0.1, grip=37.0, cmd_grip=37.0)
    assert c.L["step"] == "pick" and c._pick_watch.misses == 0  # fresh count for the new try


def test_missed_grasps_off_keeps_the_timeout(build, monkeypatch):
    c, clock, _ = build()
    monkeypatch.setattr(LC.LCFG, "max_misses", 0)
    monkeypatch.setattr(LC.LCFG, "stuck_closed_s", 0.0)
    c.loop_command("start")
    run(c, clock, 1.2)
    see_bricks(c, clock, 1)
    clock.now += 3.0
    run(c, clock, 0.1)
    for _ in range(5):
        run(c, clock, 1.0, grip=37.0, cmd_grip=37.0)
        run(c, clock, 1.0, grip=1.5, cmd_grip=0.0)
    assert c.L["step"] == "pick" and c.L["attempts"] == 0


def test_drop_while_bringing_goes_home(build):
    c, clock, _ = build()
    c.loop_command("start")
    run(c, clock, 1.2)
    see_bricks(c, clock, 1)
    clock.now += 3.0
    run(c, clock, 0.1)
    run(c, clock, 1.0, grip=5.6, cmd_grip=0.0)
    assert c.L["step"] == "bring"
    run(c, clock, 1.0, grip=1.0, cmd_grip=0.0)
    assert c.L["cycles"][-1]["result"] == "dropped while bringing" and c.L["step"] in ("home", "survey")


def test_operator_stop_pauses_and_resume_restarts_from_home(build):
    c, clock, _ = build()
    c.loop_command("start")
    run(c, clock, 1.2)
    see_bricks(c, clock, 1)
    clock.now += 3.0
    run(c, clock, 0.1)
    c._handle("q")  # Q from the console
    assert c.L["mode"] == "paused" and c.phase == "stopped"
    run(c, clock, 1.0)
    assert c.L["step"] == "paused"
    c.loop_command("resume")
    assert c.L["mode"] == "running" and c.L["step"] == "home" and c.phase == "homing"


def test_unclear_colour_pauses(build):
    c, clock, _ = build()
    c.loop_command("start")
    run(c, clock, 1.2)
    t = clock.now
    with c.lock:
        for k in range(12):
            c._count_hist.append((t + k * 0.25, [_brick(200, 300, 0.8, "?")]))
    clock.now += 9.0
    run(c, clock, 0.1)
    assert c.L["mode"] == "paused"


def test_snapshot_has_loop_and_perception(build):
    c, clock, _ = build()
    snap = c.snapshot()
    assert {"mode", "step", "grasp", "cycles"} <= set(snap["loop"]) and "table" in snap["perception"]


def _to_survey_from_plate(c, clock, insp):
    c.loop_command("start")
    run(c, clock, 1.2)
    see_bricks(c, clock, 2)
    clock.now += 3.0
    run(c, clock, 0.1)
    run(c, clock, 1.0, grip=5.6, cmd_grip=0.0)
    with c.lock:
        c.perc["inspect_seen"] = {"t": clock.now + 10, "conf": 0.7, "close": True}
    run(c, clock, 5.0, grip=5.6, cmd_grip=0.0, pose=np.ones(5))
    assert insp.done.wait(2.0)
    for _ in range(100):
        if c.phase != "inspecting":
            break
        threading.Event().wait(0.01)
    run(c, clock, 0.1, grip=5.6, cmd_grip=0.0)
    run(c, clock, 2.5, grip=30.0, cmd_grip=30.0)
    assert c.L["step"] == "survey" and c.L["survey_from"] == "plate"


def test_next_pick_straight_from_the_plate(build):
    c, clock, insp = build()
    _to_survey_from_plate(c, clock, insp)
    homes = []
    orig = c._handle
    c._handle = lambda k: (homes.append(k), orig(k))[1]
    run(c, clock, 1.1)
    see_bricks(c, clock, 1, "yellow")
    clock.now += 3.0
    run(c, clock, 0.1)
    assert c.L["step"] == "pick" and c.stage == "pick" and c.phase == "running" and c.brick == "yellow"
    assert "home" not in homes and "start" not in homes and c.bowl is None and c.L["cycle"] == 2


class TileDetector:
    """Answers only for the tile whose frame contains a bright square: boxes in tile pixels."""

    def detect(self, tile):
        ys, xs = np.nonzero(tile[..., 0] > 200)
        if not len(xs):
            return []
        return [SimpleNamespace(box=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
                                confidence=0.6)]


def test_tile_grid_covers_the_work_area_with_overlap():
    tiles = LC.P.tile_grid((0, 60, 560, 480))
    assert len(tiles) == 4
    assert min(t[0] for t in tiles) == 0 and max(t[2] for t in tiles) == 560
    assert min(t[1] for t in tiles) == 60 and max(t[3] for t in tiles) == 480
    xs = sorted({t[0] for t in tiles})
    assert 320 - xs[1] >= 60  # neighbouring tiles overlap by more than a brick


def test_detect_tiled_maps_boxes_back_and_merges_duplicates():
    img = np.zeros((480, 640, 3), np.uint8)
    img[280:320, 260:300] = 255  # in the overlap of all four tiles
    dets = LC.P.detect_tiled(TileDetector(), img, (0, 60, 560, 480))
    assert len(dets) == 1 and dets[0].box == (260, 280, 300, 320)


def test_persistent_bricks_needs_half_the_frames():
    b = _brick(100, 100)
    near = _brick(105, 98)
    assert LC.P.persistent_bricks([[b], [near], [], []]) and len(LC.P.persistent_bricks([[b], [near], [], []])) == 1
    assert LC.P.persistent_bricks([[b], [], [], []]) == []
    assert LC.P.persistent_bricks([]) == []


def test_fixed_inspection_pose_after_grasp(build):
    c, clock, insp = build(fixed_pose=True)
    c.ready = np.array([26.3, -51.8, 30.1, 73.1, 15.9, 37.7])
    c.cmd = np.array([30.0, 20.0, -2.0, 70.0, 28.0, 0.0])
    c.loop_command("start")
    run(c, clock, 1.2)
    see_bricks(c, clock, 1)
    clock.now += 3.0
    run(c, clock, 0.1)
    run(c, clock, 1.0, grip=5.6, cmd_grip=0.0)
    assert c.L["step"] == "bring" and c.phase == "moving" and c._mover is not None
    went_up = False
    for _ in range(30 * 25):
        clock.now += 1 / 30
        c.measured[5] = 5.6
        c.tick()
        went_up = went_up or c.cmd[1] < -45  # through the raised ready pose
        if c.L["step"] != "bring":
            break
    assert went_up and np.allclose(c.cmd[:5], c.ip_pose, atol=2.5)
    assert c.cmd[5] == 0.0  # gripper command untouched: still squeezing
    assert insp.done.wait(2.0) and c.L["step"] == "inspect"


def test_pose_tuning_commands(build, tmp_path, monkeypatch):
    c, clock, _ = build(fixed_pose=True)
    saved = {}
    monkeypatch.setattr(LC.IP, "save", lambda pose, fid, **kw: saved.setdefault("pose", list(pose)) and "x.json")
    c.cmd = np.array(list(c.ip_pose) + [30.0])
    c.pose_command("goto")
    run(c, clock, 0.2)
    assert c.phase == "posed"
    c.pose_command("nudge", "shoulder_pan", "5")
    assert c.ip_pose[0] == pytest.approx(LC.IP.DEMO_POSE[0] + 5) and c.phase == "posing"
    run(c, clock, 1.0)
    assert c.phase == "posed" and c.cmd[0] == pytest.approx(c.ip_pose[0], abs=1.0)
    c.pose_command("save")
    assert saved["pose"][0] == pytest.approx(LC.IP.DEMO_POSE[0] + 5) and c.ip_src == "saved"
    c.pose_command("grip_close")
    run(c, clock, 1.5)
    assert c.cmd[5] < 1.0
    c.loop_command("start")  # tuning is refused while the loop runs
    before = c.ip_pose.copy()
    c.pose_command("nudge", "elbow_flex", "5")
    assert np.allclose(c.ip_pose, before)


def test_vla_bring_then_offset_with_live_tuning(build):
    c, clock, insp = build()
    c.ip_offset = np.zeros(5)
    c.pose_command("hold_on")
    c.loop_command("start")
    run(c, clock, 1.2)
    see_bricks(c, clock, 1)
    clock.now += 3.0
    run(c, clock, 0.1)
    run(c, clock, 1.0, grip=5.6, cmd_grip=0.0)
    assert c.L["step"] == "bring" and c.phase == "running"  # the VLA brings it (its own prompt)
    vla_end = np.array([-12.0, 34.0, -53.0, 88.0, -4.0])
    run(c, clock, 7.0, grip=5.6, cmd_grip=0.0, pose=vla_end)  # VLA settles at its endpoint
    c.cmd[:5] = vla_end
    for _ in range(60):
        clock.now += 1 / 30
        c.measured[5] = 5.6
        c.tick()
    assert c.L["step"] == "tune" and np.allclose(c._bring_base, vla_end)
    c.pose_command("off_nudge", "shoulder_pan", "-5")  # shift right in the inspection view
    for _ in range(90):
        clock.now += 1 / 30
        c.measured[5] = 5.6
        c.tick()
    assert c.cmd[0] == pytest.approx(-17.0, abs=0.6) and c.cmd[5] == 0.0
    c.pose_command("spin")
    assert insp.done.wait(2.0) and c.L["cycles"][-1]["offset"][0] == -5.0


def test_saved_offset_applied_without_hold(build):
    c, clock, insp = build()
    c.ip_offset = np.array([-6.0, 0, 0, 0, 0])
    c.loop_command("start")
    run(c, clock, 1.2)
    see_bricks(c, clock, 1)
    clock.now += 3.0
    run(c, clock, 0.1)
    run(c, clock, 1.0, grip=5.6, cmd_grip=0.0)
    vla_end = np.array([-12.0, 34.0, -53.0, 88.0, -4.0])
    run(c, clock, 7.0, grip=5.6, cmd_grip=0.0, pose=vla_end)
    c.cmd[:5] = vla_end
    for _ in range(120):
        clock.now += 1 / 30
        c.measured[5] = 5.6
        c.tick()
        if c.L["step"] != "bring":
            break
    assert insp.done.wait(2.0)
    cyc = c.L["cycles"][-1]
    assert cyc["offset"][0] == -6.0 and cyc["vla_end"][0] == pytest.approx(-12.0)


def test_vla_bring_then_straight_to_saved_pose(build):
    c, clock, insp = build(fixed_pose="vla_pose")
    c.ready = np.array([26.3, -51.8, 30.1, 73.1, 15.9, 37.7])
    c.ip_pose = np.array([-24.2, 20.0, -24.2, 72.4, -17.2])
    c.loop_command("start")
    run(c, clock, 1.2)
    see_bricks(c, clock, 1)
    clock.now += 3.0
    run(c, clock, 0.1)
    run(c, clock, 1.0, grip=5.6, cmd_grip=0.0)
    assert c.L["step"] == "bring" and c.phase == "running"  # the VLA's own bring
    vla_end = np.array([-7.9, 40.5, -45.1, 62.5, 2.2])
    path = []
    for _ in range(30 * 20):
        clock.now += 1 / 30
        c.measured[5], c.cmd[5] = 5.6, 0.0
        if c.phase == "running" and c.L["step"] == "bring":  # the VLA holds the arm at its endpoint
            c.measured[:5] = vla_end
            c.cmd[:5] = vla_end
        c.tick()
        if c.phase in ("moving", "posing"):
            path.append(c.cmd[:5].copy())
        if c.L["step"] != "bring":
            break
    path = np.stack(path)
    assert np.allclose(path[-1], c.ip_pose, atol=0.6)
    assert path[:, 1].min() > 19.0  # straight there: never up through the ready pose (lift -51.8)
    assert insp.done.wait(2.0) and c.L["cycles"][-1]["pose"] == [-24.2, 20.0, -24.2, 72.4, -17.2]



def test_spin_and_save_records_label(build, tmp_path, monkeypatch):
    import collect as C

    c, clock, insp = build()
    c.collect = C.Store(tmp_path / "ds")
    d = tmp_path / "spin1"
    d.mkdir()
    for i in range(3):
        (d / f"crop_{i:02d}.jpg").write_bytes(b"x")
    insp.r.saved = [str(d)]
    c.collect_command("spin")
    assert "label" in c.collect_msg  # refuses without a label
    c.collect_command("label", "cross")
    assert c.auto_inspector.grasp_empty_pos < -1e6  # the spin ignores the gripper reading by default
    c.L["grasp"] = {"state": "empty", "pos": 1.0}
    c.collect_command("spin")  # no gripper refusal any more
    assert insp.done.wait(2.0)
    for _ in range(200):
        if c.phase != "collecting":
            break
        threading.Event().wait(0.01)
    assert c.phase == "holding" and c.L["mode"] == "off"  # no hand-over to the VLA
    assert c.collect.counts()["cross"] == {"spins": 1, "crops": 3}


def test_bring_to_camera_with_the_vla_then_hold(build):
    c, clock, _ = build()
    c.L["grasp"] = {"state": "holding", "pos": 5.6}
    c.collect_command("bring")
    assert c.phase == "running" and c.stage == "bring"  # the VLA's own bring prompt runs
    for _ in range(30 * 8):
        clock.now += 1 / 30
        c.measured[5] = 5.6
        c.measured[:5] = [-8.7, 20.0, -17.2, 63.4, 5.8]  # the VLA settled at the camera
        c.tick()
        if c.phase == "holding":
            break
    assert c.phase == "holding" and "Spin & save" in c.collect_msg and c.L["mode"] == "off"



def test_survey_flicker_does_not_block_the_empty_table(build):
    """A phantom in 1 of 12 frames (the backdrop past the table edge did this on 14 Nov): still verified empty."""
    c, clock, _ = build()
    c.loop_command("start")
    run(c, clock, 1.2)
    t = clock.now
    with c.lock:
        for k in range(12):
            c._count_hist.append((t + k * 0.25, [_brick(500, 60, 0.35)] if k == 5 else []))
    clock.now += 3.0
    run(c, clock, 0.1)
    assert c.L["step"] == "verify" and c.L["cycle"] == 0


def test_survey_picks_the_steady_brick_not_the_flicker(build):
    c, clock, _ = build()
    c.loop_command("start")
    run(c, clock, 1.2)
    t = clock.now
    with c.lock:
        for k in range(12):
            frame = [_brick(200, 300, 0.6, "red")]
            if k in (3, 7):
                frame.append(_brick(500, 60, 0.9, "blue"))  # brighter, but only 2 of 12 frames
            c._count_hist.append((t + k * 0.25, frame))
    clock.now += 3.0
    run(c, clock, 0.1)
    assert c.L["step"] == "pick" and c.brick == "red"


def test_survey_waits_for_enough_frames_before_picking(build):
    c, clock, _ = build()
    c.loop_command("start")
    run(c, clock, 1.2)
    see_bricks(c, clock, 1, frames=2)
    clock.now += 0.6
    run(c, clock, 0.05)
    assert c.L["step"] == "survey"


# ---- plate routing: which plate each outcome goes to, changed live from the console -------------------------------------


class RoutedInspector(StubInspector):
    """Decides the bowl at inspect time from auto_inspect's BOWL / FALLBACK_BOWL, like the real AutoInspector."""

    def __init__(self, outcome):
        super().__init__(outcome=outcome, bowl=None)

    def inspect(self, arm, start_cmd):
        ai = sys.modules["auto_inspect"]
        self.r.bowl = ai.BOWL[self.r.outcome] if self.r.outcome in ai.BOWL else ai.FALLBACK_BOWL
        return super().inspect(arm, start_cmd)


def drive_to_place(c, clock, insp):
    """Start the loop and run one brick up to the decision (step 'place')."""
    c.loop_command("start")
    run(c, clock, 1.2)
    see_bricks(c, clock, 1, "blue")
    clock.now += 3.0
    run(c, clock, 0.1)
    run(c, clock, 1.0, grip=30.0, cmd_grip=30.0)
    run(c, clock, 1.0, grip=5.6, cmd_grip=0.0)
    with c.lock:
        c.perc["inspect_seen"] = {"t": clock.now + 10, "conf": 0.7, "close": True}
    run(c, clock, 5.0, grip=5.6, cmd_grip=0.0, pose=np.ones(5))
    assert insp.done.wait(2.0)
    for _ in range(100):
        if c.phase != "inspecting":
            break
        threading.Event().wait(0.01)
    run(c, clock, 0.1, grip=5.6, cmd_grip=0.0)
    assert c.L["step"] == "place"


def test_routing_defaults_are_in_the_snapshot(build):
    c, _, _ = build()
    assert c.snapshot()["routing"] == {"plain": "yellow", "circle": "orange", "cross": "purple", "marked": "purple",
                                       "uncertain": "purple"}


def test_routing_change_decides_the_next_brick_and_the_place_prompt(build):
    c, clock, insp = build(RoutedInspector("cross"))
    c.routing_command("cross", "yellow")
    assert c.snapshot()["routing"]["cross"] == "yellow"
    drive_to_place(c, clock, insp)
    assert c.bowl == "yellow" and c.inspection["bowl"] == "yellow"
    assert S.fill(S.PLACE, colour=c.bowl) == "place the brick in the yellow bowl"  # the trained prompt, new colour
    run(c, clock, 1.5, grip=30.0, cmd_grip=30.0)
    run(c, clock, 1.0, grip=30.0, cmd_grip=30.0)
    assert c.L["tallies"] == {"yellow": 1} and c.L["cycles"][-1]["bowl"] == "yellow"


def test_routing_uncertain_uses_the_fallback_plate(build):
    c, clock, insp = build(RoutedInspector("unknown"))
    c.routing_command("uncertain", "orange")
    drive_to_place(c, clock, insp)
    assert c.bowl == "orange" and S.fill(S.PLACE, colour=c.bowl) == "place the brick in the orange bowl"


def test_routing_change_after_the_decision_does_not_move_this_brick(build):
    c, clock, insp = build(RoutedInspector("cross"))
    drive_to_place(c, clock, insp)
    assert c.bowl == "purple"
    c.routing_command("cross", "yellow")  # mid-cycle: the decision is already made
    c.routing_command("uncertain", "yellow")
    assert c.bowl == "purple" and c.inspection["bowl"] == "purple"
    run(c, clock, 1.5, grip=30.0, cmd_grip=30.0)
    run(c, clock, 1.0, grip=30.0, cmd_grip=30.0)
    assert c.L["tallies"] == {"purple": 1}
    assert c.snapshot()["routing"]["cross"] == "yellow"  # ...but the next brick follows the new table


def test_routing_rejects_bad_outcomes_and_colours(build):
    c, _, _ = build()
    before = c.snapshot()["routing"]
    for outcome, colour in [("cross", "pink"), ("cross", "Yellow"), ("triangle", "yellow"), ("none", "yellow"),
                            ("cross", None), ("", "purple")]:
        with pytest.raises(ValueError):
            c.routing_command(outcome, colour)
    assert c.snapshot()["routing"] == before


def test_routing_reset_restores_the_default_table(build):
    c, _, _ = build()
    for o, col in [("plain", "purple"), ("circle", "yellow"), ("cross", "orange"), ("marked", "yellow"),
                   ("uncertain", "yellow")]:
        c.routing_command(o, col)
    assert c.snapshot()["routing"] == {"plain": "purple", "circle": "yellow", "cross": "orange", "marked": "yellow",
                                       "uncertain": "yellow"}
    c.routing_command("reset")
    assert c.snapshot()["routing"] == LC.ROUTE_DEFAULT


def test_parse_routing_flag():
    assert LC.parse_routing("plain=purple, uncertain=orange") == {"plain": "purple", "uncertain": "orange"}
    assert LC.parse_routing("") == {}
    for bad in ("plain=pink", "triangle=yellow", "plain"):
        with pytest.raises(ValueError):
            LC.parse_routing(bad)


def test_routing_endpoint_validates_and_updates(build):
    import json
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    c, _, _ = build()

    def make_handler(streams, ctl):
        class Base(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body, ctype):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                self._send(404, b"base", "text/plain")

        return Base

    srv = ThreadingHTTPServer(("127.0.0.1", 0), LC.wrap_handler(make_handler)({}, c))
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    def post(path):
        try:
            with urlopen(Request(f"http://127.0.0.1:{srv.server_port}{path}", method="POST")) as r:
                return r.status, r.read()
        except HTTPError as e:
            return e.code, b""

    try:
        status, body = post("/routing/cross/orange")
        assert status == 200 and json.loads(body)["cross"] == "orange"
        assert post("/routing/plain/purple")[0] == 200
        assert post("/routing/cross/pink")[0] == 400
        assert post("/routing/triangle/yellow")[0] == 400
        assert post("/routing/cross")[0] == 404 or post("/routing/cross")[0] == 400
        assert c.snapshot()["routing"]["cross"] == "orange" and c.snapshot()["routing"]["plain"] == "purple"
        assert post("/routing/reset")[0] == 200
        assert c.snapshot()["routing"] == LC.ROUTE_DEFAULT
    finally:
        srv.shutdown()


def test_reset_stops_and_clears_the_run(build):
    c, clock, _ = build()
    c.loop_command("start")
    run(c, clock, 1.2)
    see_bricks(c, clock, 1)
    clock.now += 3.0
    run(c, clock, 0.1)
    assert c.L["step"] == "pick" and c.L["cycle"] == 1
    c.L["tallies"]["yellow"] = 2
    c.loop_command("reset")
    assert c.L["mode"] == "off" and c.L["step"] == "idle"
    assert c.L["cycle"] == 0 and c.L["cycles"] == [] and c.L["tallies"] == {}
    assert [e.get("message") for e in c.L["events"]] == ["Reset: run cleared, arm holding where it is"]
    assert c._plates == {}
