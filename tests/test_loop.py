"""Hardware-free tests of the closed loop (dlsj.loop) with the fake workcell. NumPy only; no LeRobot."""

import json
import threading
import time
import urllib.request

import numpy as np
import pytest

from dlsj.contracts import InspectionResult, Label, MarkType, Observation
from dlsj.loop import fakes
from dlsj.loop.controller import Controller, ControllerConfig, grasp_state
from dlsj.loop.executor import ExecConfig, SkillExecutor
from dlsj.loop.planner import AnyBrickPlanner, OrderedPlanner
from dlsj.loop.reasoner import ReasonConfig, decide
from dlsj.loop.skills import (
    PICK_AND_SHOW, PLACE_BLUE, PLACE_MINOR, PLACE_PINK, PLACE_SLOT, RESHOW, GripperCondition, Outcome, Plates,
    SkillLibrary, Termination, TerminationMonitor,
)
from dlsj.loop.telemetry import RunLog

FPS = 200.0  # run the control loop fast; all speed caps scale with it so the motion per tick matches 30 Hz runs


def fast_library(**term_overrides) -> SkillLibrary:
    lib = SkillLibrary()
    for name in lib.skills:
        lib.override(name, min_duration_s=0.05, **term_overrides)
    return lib


def make_loop(bricks, tmp_path=None, inspector=None, reshow=False, latency_s=0.0, async_inf=True, max_tasks=None,
              lib=None, ccfg=None, policy=None):
    world = fakes.FakeWorld(bricks=list(bricks))
    robot = fakes.FakeRobot(world)
    cams = fakes.FakeCameras(world, robot)
    lib = lib or fast_library()
    if reshow:
        lib.enabled_optional.add(RESHOW)
    policy = policy or fakes.ScriptedPolicy(world.poses, latency_s=latency_s)
    runlog = RunLog(tmp_path)
    ecfg = ExecConfig(fps=FPS, max_speed_deg_s=3.0 * FPS, gripper_max_speed=10.0 * FPS, ramp_speed_deg_s=3.0 * FPS,
                      async_inference=async_inf)
    ex = SkillExecutor(robot, cams, policy, lib, ecfg, dict(world.poses), runlog)
    planner = AnyBrickPlanner(max_tasks=len(bricks) if max_tasks is None else max_tasks)
    ccfg = ccfg or ControllerConfig(reason=ReasonConfig(reshow_available=lib.available(RESHOW)))
    ctl = Controller(ex, cams, inspector or fakes.FakeInspector(cams.roi), planner, runlog, ccfg, world.poses["home"])
    return ctl, world, robot, policy, runlog


def events(runlog, name):
    return [e for e in runlog.events if e["event"] == name]


# ----------------------------------------------------------------------------- full loop
def test_plain_brick_goes_to_blue_plate():
    ctl, world, *_ = make_loop(["plain"])
    reports = ctl.run()
    assert [(r.status, r.label, r.place_skill, r.forced) for r in reports] == [
        ("placed", Label.PERFECT, PLACE_BLUE, False)]
    assert world.placed == [(1, "plain", "blue")]
    assert ctl.state.value == "done"


def test_crossed_brick_goes_to_pink_plate_with_region():
    ctl, world, _, _, runlog = make_loop(["cross"])
    (r,) = ctl.run()
    assert (r.label, r.place_skill) == (Label.DEFECTIVE, PLACE_PINK)
    assert world.placed == [(1, "cross", "pink")]
    insp = events(runlog, "inspection")[0]
    assert insp["label"] == "defective" and insp["score"] > insp["threshold"] and insp["region"] is not None
    x0, y0, x1, y1 = insp["region"]
    rx0, ry0, rx1, ry1 = ctl.cameras.roi
    assert rx0 <= x0 < x1 <= rx1 and ry0 <= y0 < y1 <= ry1  # the cross is localized on the brick


def test_several_bricks_in_one_run():
    ctl, world, *_ = make_loop(["plain", "cross", "plain"])
    ctl.run()
    assert [w for _, _, w in world.placed] == ["blue", "pink", "blue"]
    assert ctl.counts == {"placed_perfect": 2, "placed_defective": 1}


def test_uncertain_then_resolved_by_regrab():
    insp = fakes.ScriptedInspector([Label.UNCERTAIN, Label.PERFECT])
    ctl, world, _, policy, runlog = make_loop(["plain"], inspector=insp)
    (r,) = ctl.run()
    assert (r.label, r.forced, len(r.inspections)) == (Label.PERFECT, False, 2)
    assert world.placed == [(1, "plain", "blue")]
    assert not any("turn the brick" in c for c in policy.calls)  # RESHOW disabled -> plain re-grab
    assert [e["stage"] for e in events(runlog, "retry")] == ["detect"]


def test_uncertain_then_resolved_by_reshow_skill():
    ctl, world, _, policy, _ = make_loop(["hidden_cross"], reshow=True)
    (r,) = ctl.run()
    assert [i["label"] for i in r.inspections] == ["uncertain", "defective"]
    assert (r.label, r.forced) == (Label.DEFECTIVE, False)
    assert any("turn the brick" in c for c in policy.calls)
    assert world.placed == [(1, "hidden_cross", "pink")]


def test_glare_resolves_to_perfect_after_reshow():
    ctl, world, *_ = make_loop(["glare"], reshow=True)
    (r,) = ctl.run()
    assert [i["label"] for i in r.inspections] == ["uncertain", "perfect"]
    assert world.placed == [(1, "glare", "blue")]


def test_uncertain_twice_falls_back_to_pink_and_logs_it():
    ctl, world, _, _, runlog = make_loop(["faint"])
    (r,) = ctl.run()
    assert [i["label"] for i in r.inspections] == ["uncertain", "uncertain"]
    assert (r.label, r.forced, r.place_skill) == (Label.DEFECTIVE, True, PLACE_PINK)
    assert world.placed == [(1, "faint", "pink")]
    (fb,) = events(runlog, "uncertain_fallback")
    assert fb["attempts"] == 2 and fb["label"] == "defective"


def test_empty_table_retries_pick_then_gives_up():
    lib = fast_library()
    lib.override(PICK_AND_SHOW, max_duration_s=0.6)
    ctl, world, _, _, runlog = make_loop([], lib=lib, max_tasks=1,
                                         ccfg=ControllerConfig(max_pick_attempts=2))
    (r,) = ctl.run()
    assert r.status == "pick_failed" and r.pick_attempts == 2
    checks = events(runlog, "grasp_check")
    assert len(checks) == 2 and all(c["outcome"] == "timeout" for c in checks)
    assert world.placed == []


def test_policy_error_stops_the_run():
    class Broken(fakes.ScriptedPolicy):
        def predict_chunk(self, obs, instruction):
            raise RuntimeError("GPU fell over")

    ctl, world, _, _, runlog = make_loop(["plain", "plain"], policy=Broken())
    reports = ctl.run()
    assert [r.status for r in reports] == ["error"] and ctl.state.value == "error"
    assert "GPU fell over" in events(runlog, "error")[0]["error"]


def test_abort_mid_skill():
    lib = fast_library()
    lib.override(PICK_AND_SHOW, max_duration_s=30.0)
    ctl, world, *_ = make_loop([], lib=lib, max_tasks=1)  # empty table: the pick would never finish
    t = threading.Thread(target=ctl.run)
    t.start()
    time.sleep(0.3)
    ctl.abort()
    t.join(10)
    assert not t.is_alive()
    assert ctl.state.value == "aborted" and ctl.reports[0].status == "aborted"


def test_sync_inference_also_completes():
    ctl, world, *_ = make_loop(["cross"], async_inf=False, latency_s=0.01)
    ctl.run()
    assert world.placed == [(1, "cross", "pink")]


# ----------------------------------------------------------------------------- executor
def _executor(world, latency_s=0.0, async_inf=True, lib=None):
    robot = fakes.FakeRobot(world)
    cams = fakes.FakeCameras(world, robot)
    policy = fakes.ScriptedPolicy(world.poses, latency_s=latency_s)
    ecfg = ExecConfig(fps=FPS, max_speed_deg_s=3.0 * FPS, gripper_max_speed=10.0 * FPS, async_inference=async_inf)
    return SkillExecutor(robot, cams, policy, lib or fast_library(), ecfg, dict(world.poses)), robot, policy


def test_executor_async_overlaps_inference_and_drops_stale_actions():
    world = fakes.FakeWorld(bricks=["plain"])
    ex, robot, policy = _executor(world, latency_s=0.03)
    res = ex.run(PICK_AND_SHOW)
    assert res.outcome is Outcome.DONE and res.chunks >= 2
    assert res.dropped_actions > 0  # ~latency x fps actions skipped per chunk
    # with async inference only the very first chunk starves the queue
    assert res.starved_ticks <= int(0.03 * FPS) + 3
    assert res.stall_ms < 30.0 * res.chunks  # far less than one latency per chunk


def test_executor_sync_inference_runs_inline():
    world = fakes.FakeWorld(bricks=["plain"])
    ex, robot, policy = _executor(world, latency_s=0.01, async_inf=False)
    res = ex.run(PICK_AND_SHOW)
    assert res.outcome is Outcome.DONE and res.dropped_actions == 0 and res.starved_ticks == 0
    assert res.stall_ms >= 0.9 * 10.0 * res.chunks  # the arm waits for every chunk


def test_executor_respects_max_step_per_tick():
    world = fakes.FakeWorld(bricks=["plain"])
    ex, robot, _ = _executor(world)
    ex.run(PICK_AND_SHOW)
    sent = np.stack(robot.sent)
    steps = np.abs(np.diff(sent, axis=0))
    assert np.all(steps <= ex.max_step + 1e-4)


def test_executor_feeds_policy_full_observation():
    seen = []

    class Spy(fakes.ScriptedPolicy):
        def predict_chunk(self, obs, instruction):
            seen.append(obs)
            return super().predict_chunk(obs, instruction)

    world = fakes.FakeWorld(bricks=["plain"])
    robot = fakes.FakeRobot(world)
    cams = fakes.FakeCameras(world, robot)
    ex = SkillExecutor(robot, cams, Spy(world.poses), fast_library(),
                       ExecConfig(fps=FPS, max_speed_deg_s=3 * FPS, gripper_max_speed=10 * FPS), dict(world.poses))
    ex.run(PICK_AND_SHOW)
    o = seen[0]
    assert isinstance(o, Observation) and o.state.shape == (6,) and o.state.dtype == np.float32
    assert set(o.images) == {"wrist", "top", "inspect"} and o.images["top"].dtype == np.uint8


def test_bad_chunk_shape_is_an_error():
    class Bad(fakes.ScriptedPolicy):
        def predict_chunk(self, obs, instruction):
            return np.zeros((10, 5), dtype=np.float32)

    world = fakes.FakeWorld(bricks=["plain"])
    robot = fakes.FakeRobot(world)
    cams = fakes.FakeCameras(world, robot)
    ex = SkillExecutor(robot, cams, Bad(), fast_library(), ExecConfig(fps=FPS), dict(world.poses))
    res = ex.run(PICK_AND_SHOW)
    assert res.outcome is Outcome.ERROR and "shape" in res.error


# ----------------------------------------------------------------------------- termination
def _state(arm=0.0, grip=50.0):
    return np.array([arm] * 5 + [grip], dtype=np.float32)


def _feed(mon, states, dt=0.01):
    out = None
    for i, s in enumerate(states):
        out = mon.update(s, i * dt)
        if out is not None:
            return out, i
    return out, len(states)


def test_termination_gripper_closed_then_held():
    term = Termination(max_duration_s=100, min_duration_s=0, gripper=GripperCondition("closed", 40, hold_ticks=5))
    mon = TerminationMonitor(term, fps=100)
    out, i = _feed(mon, [_state(grip=80)] * 10 + [_state(grip=20)] * 10)
    assert out is Outcome.DONE and i == 10 + 4  # the 5th consecutive closed tick


def test_termination_closed_on_nothing_is_not_holding():
    term = Termination(max_duration_s=100, min_duration_s=0,
                       gripper=GripperCondition("closed", 40, hold_ticks=3, min_value=4.0))
    out, _ = _feed(TerminationMonitor(term, fps=100), [_state(grip=1.0)] * 20)  # jaws shut on air
    assert out is None
    out, i = _feed(TerminationMonitor(term, fps=100), [_state(grip=22.0)] * 20)  # stalled on a brick
    assert out is Outcome.DONE and i == 2


def test_termination_closed_latch_resets_when_dropped():
    term = Termination(max_duration_s=100, min_duration_s=0, gripper=GripperCondition("closed", 40, hold_ticks=3),
                       still_ticks=20, still_deg_s=1.0)
    mon = TerminationMonitor(term, fps=100)
    moving = [_state(arm=float(i), grip=20) for i in range(5)]  # grasped while moving
    out, _ = _feed(mon, moving + [_state(arm=5, grip=90)])  # then the jaws open (brick dropped)
    assert out is None and mon.grip_latched is False


def test_termination_gripper_opened_for_place():
    term = Termination(max_duration_s=100, min_duration_s=0, gripper=GripperCondition("opened", 25, hold_ticks=3))
    mon = TerminationMonitor(term, fps=100)
    out, i = _feed(mon, [_state(grip=22)] * 5 + [_state(grip=60)] * 5)
    assert out is Outcome.DONE and i == 7


def test_termination_stillness_needs_n_slow_ticks():
    term = Termination(max_duration_s=100, min_duration_s=0, still_ticks=4, still_deg_s=5.0)
    mon = TerminationMonitor(term, fps=10)  # 1 deg/tick = 10 deg/s (moving); 0.2 deg/tick = 2 deg/s (still)
    states = [_state(arm=float(i)) for i in range(5)] + [_state(arm=4 + 0.2 * k) for k in range(1, 10)]
    out, i = _feed(mon, states)
    assert out is Outcome.DONE and i == 5 + 3


def test_termination_near_saved_pose_and_missing_pose_is_ignored():
    pose = _state(arm=30.0)
    term = Termination(max_duration_s=100, min_duration_s=0, near_pose="inspect", near_pose_tol_deg=2.0)
    mon = TerminationMonitor(term, fps=30, poses={"inspect": pose})
    out, i = _feed(mon, [_state(arm=a) for a in (0, 10, 20, 27, 29, 30)])
    assert out is Outcome.DONE and i == 4
    still = Termination(max_duration_s=100, min_duration_s=0, still_ticks=2, near_pose="inspect")
    out, _ = _feed(TerminationMonitor(still, fps=30, poses={}), [_state()] * 5)  # pose not saved -> stillness only
    assert out is Outcome.DONE


def test_termination_min_duration_and_timeout():
    term = Termination(max_duration_s=0.5, min_duration_s=0.2, still_ticks=1)
    out, i = _feed(TerminationMonitor(term, fps=100), [_state()] * 100, dt=0.05)
    assert out is Outcome.DONE and i == 4  # still at once, but not before 0.2 s
    never = Termination(max_duration_s=0.5, min_duration_s=0.0, gripper=GripperCondition("opened", 25))
    out, i = _feed(TerminationMonitor(never, fps=100), [_state(grip=0)] * 100, dt=0.05)
    assert out is Outcome.TIMEOUT and i == 10


# ----------------------------------------------------------------------------- reasoner
def _res(label, score=0.5, thr=0.5, region=None):
    return InspectionResult(label, score, thr, region)


def test_reasoner_rule_table():
    assert decide(_res(Label.PERFECT, 0.1), 1).skill == PLACE_BLUE
    d = decide(_res(Label.DEFECTIVE, 0.9, region=(1, 2, 3, 4)), 1)
    assert (d.kind, d.skill, d.forced) == ("place", PLACE_PINK, False) and "(1, 2, 3, 4)" in d.reason


def test_reasoner_uncertainty_band():
    d1 = decide(_res(Label.UNCERTAIN, 0.48), 1)
    assert (d1.kind, d1.skill) == ("reinspect", None)
    d1r = decide(_res(Label.UNCERTAIN, 0.48), 1, ReasonConfig(reshow_available=True))
    assert d1r.skill == RESHOW and "turn" in d1r.reason
    d2 = decide(_res(Label.UNCERTAIN, 0.48), 2)
    assert (d2.kind, d2.label, d2.skill, d2.forced) == ("place", Label.DEFECTIVE, PLACE_PINK, True)
    assert "safe default" in d2.reason
    # extra margin turns a borderline PERFECT into UNCERTAIN
    assert decide(_res(Label.PERFECT, 0.45), 1, ReasonConfig(extra_margin=0.1)).kind == "reinspect"


def test_reasoner_phase2_slot_skill():
    d = decide(_res(Label.PERFECT, 0.1), 1, perfect_skill=PLACE_SLOT, perfect_params={"slot": 3})
    assert d.skill == PLACE_SLOT and d.params == {"slot": 3}


def test_grasp_state():
    assert grasp_state(1.0, None, 4.0, None) == "EMPTY"
    assert grasp_state(22.0, None, 4.0, None) == "HOLDING"
    assert grasp_state(1.0, 500.0, 4.0, 200.0) == "HOLDING"
    assert grasp_state(30.0, 50.0, 4.0, 200.0) == "UNKNOWN"


# ----------------------------------------------------------------------------- skills
def test_skill_library_strings_params_and_overrides(tmp_path):
    lib = SkillLibrary()
    assert lib[PICK_AND_SHOW].instruction == "pick up the brick and show it to the inspection camera"
    assert lib[PLACE_BLUE].instruction == "place the brick on the blue plate"
    assert lib[PLACE_PINK].instruction == "place the brick on the pink plate"
    assert lib[PLACE_SLOT].render(slot=2) == "place the brick in slot 2"
    with pytest.raises(ValueError):
        lib[PLACE_SLOT].render()
    assert not lib.available(RESHOW)
    cfg = tmp_path / "skills.json"
    cfg.write_text(json.dumps({"RESHOW": {"enabled": True}, "PLACE_BLUE": {"max_duration_s": 7,
                               "gripper": {"mode": "opened", "threshold": 30}}}))
    lib.load_overrides(cfg)
    assert lib.available(RESHOW)
    assert lib[PLACE_BLUE].termination.max_duration_s == 7
    assert lib[PLACE_BLUE].termination.gripper.threshold == 30


# ----------------------------------------------------------------------------- planner
def test_any_brick_planner():
    p = AnyBrickPlanner(max_tasks=3, max_consecutive_failures=2)
    t1 = p.next()
    assert t1.pick_skill == PICK_AND_SHOW and t1.perfect_skill == PLACE_BLUE and t1.defective_skill == PLACE_PINK
    p.fail(t1, "pick failed", requeue=True)
    assert p.next() is t1 and t1.attempts == 1  # requeued tasks come first
    p.fail(t1, "pick failed")
    assert p.next() is None  # two failures in a row: the table is probably empty
    q = AnyBrickPlanner(max_tasks=2)
    a = q.next()
    q.complete(a, "perfect")
    b = q.next()
    q.complete(b, "defective")
    assert q.next() is None and [t.id for t, _ in q.history] == [1, 2]


def test_ordered_planner_phase2():
    p = OrderedPlanner(min_confidence=0.5)
    dets = [{"number": 3, "confidence": 0.9, "box": (0, 0, 1, 1), "table_xy": (0.3, 0.1)},
            {"number": 1, "confidence": 0.95, "box": (0, 0, 1, 1), "table_xy": (0.1, 0.1)},
            {"number": 2, "confidence": 0.2, "box": (0, 0, 1, 1), "table_xy": (0.2, 0.1)}]
    tasks = p.replan(dets)
    assert [t.expected_number for t in tasks] == [1, 3]  # sorted, low-confidence read dropped
    assert tasks[0].pick_skill == "PICK_MARKED" and tasks[0].perfect_params == {"slot": 1}
    t = p.next()
    p.complete(t, "defective")  # brick 1 rejected: it does not take a slot
    dets[2]["confidence"] = 0.9
    tasks = p.replan(dets)
    assert [(t.expected_number, t.perfect_params["slot"]) for t in tasks] == [(2, 1), (3, 2)]


# ----------------------------------------------------------------------------- logs, status, CLI
def test_timing_and_event_logs_written(tmp_path):
    ctl, world, _, _, runlog = make_loop(["plain", "faint"], tmp_path=tmp_path)
    ctl.run()
    summary = runlog.close()
    for stage in ("perceive", "inference", "tick", "stall", "settle", "detect", "inspector", "reason", "pick", "place",
                  "brick", "skill.PICK_AND_SHOW", "skill.PLACE_BLUE", "skill.PLACE_PINK"):
        assert summary[stage]["n"] > 0, stage
        assert {"median", "p90", "p99"} <= set(summary[stage])
    lines = [json.loads(x) for x in (tmp_path / "timings.jsonl").read_text().splitlines()]
    assert lines[-1]["type"] == "summary" and all("ms" in x for x in lines[:-1])
    ev = [json.loads(x) for x in (tmp_path / "events.jsonl").read_text().splitlines()]
    kinds = {e["event"] for e in ev}
    assert {"run_start", "inspection", "decision", "retry", "uncertain_fallback", "brick_done", "run_end"} <= kinds
    insp = next(e for e in ev if e["event"] == "inspection")
    assert {"label", "score", "threshold", "region", "frame_scores", "latency_ms", "device"} <= set(insp)
    assert json.loads((tmp_path / "summary.json").read_text())["brick"]["n"] == 2


def test_status_overlay_and_server():
    from dlsj.loop.status import StatusServer, overlay

    frame = np.full((40, 60, 3), 100, dtype=np.uint8)
    heat = np.zeros((40, 60), dtype=np.float32)
    heat[10:20, 10:20] = 1.0
    out = overlay(frame, InspectionResult(Label.DEFECTIVE, 0.9, 0.5, (5, 5, 30, 30), heat))
    assert out.shape == frame.shape and out.dtype == np.uint8
    assert out[15, 15, 0] > 150 and tuple(out[5, 10]) == (255, 40, 200)

    ctl, *_ = make_loop(["cross"])
    ctl.run()
    srv = StatusServer(ctl, ctl.cameras, port=0).start()
    try:
        body = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{srv.port}/state", timeout=5).read())
        assert body["loop"]["state"] == "done" and body["loop"]["last_inspection"]["label"] == "defective"
        assert body["cameras"] == ["wrist", "top", "inspect"]
        page = urllib.request.urlopen(f"http://127.0.0.1:{srv.port}/", timeout=5).read()
        assert b"DLSJ closed loop" in page
    finally:
        srv.stop()


def test_cli_dry_run(tmp_path, capsys):
    from dlsj.loop import run

    skills = tmp_path / "skills.json"
    skills.write_text(json.dumps({n: {"min_duration_s": 0.05} for n in SkillLibrary().skills}))
    rc = run.main(["--dry-run", "--bricks", "plain,cross", "--fps", str(FPS), "--max-speed", str(3 * FPS),
                   "--ramp-speed", str(3 * FPS), "--fake-latency-ms", "5", "--status-port", "0",
                   "--skills-config", str(skills), "--log-dir", str(tmp_path / "run")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "(1, 'plain', 'blue'), (2, 'cross', 'pink')" in out
    assert (tmp_path / "run" / "summary.json").exists()


# ----------------------------------------------------------------------------- three plates (stage 2)
def _res2(mark, conf=0.95, label=Label.DEFECTIVE, score=0.9):
    return InspectionResult(label, score, 0.5, (1, 2, 3, 4), mark=mark, mark_confidence=conf,
                            stage_latency_ms={"stage1": 1.0, "stage2": 0.5})


def test_decision_tree_three_plates():
    assert decide(_res2(None, label=Label.PERFECT, score=0.1), 1).skill == PLACE_BLUE
    d = decide(_res2(MarkType.CIRCLE), 1)
    assert (d.kind, d.label, d.skill, d.forced) == ("place", Label.MINOR, PLACE_MINOR, False)
    assert "circle" in d.reason
    d = decide(_res2(MarkType.CROSS), 1)
    assert (d.kind, d.label, d.skill) == ("place", Label.DEFECTIVE, PLACE_PINK)
    d = decide(_res2(None), 1)  # stage 2 not configured: marked = defective (two-plate behaviour)
    assert (d.label, d.skill) == (Label.DEFECTIVE, PLACE_PINK)
    d = decide(_res2(MarkType.CIRCLE), 1, minor_skill="PLACE_X", minor_params={"k": 1})
    assert d.skill == "PLACE_X" and d.params == {"k": 1}


def test_decision_tree_uncertain_paths():
    # stage 2 unsure -> re-inspect, then the safest plate
    d1 = decide(_res2(MarkType.UNKNOWN, 0.55), 1)
    assert (d1.kind, d1.skill, d1.stage) == ("reinspect", None, "stage2") and "stage 2" in d1.reason
    d2 = decide(_res2(MarkType.UNKNOWN, 0.55), 2)
    assert (d2.kind, d2.label, d2.skill, d2.forced, d2.stage) == ("place", Label.DEFECTIVE, PLACE_PINK, True,
                                                                   "stage2")
    # stage 2's own threshold in the reasoner: a 0.75 circle is uncertain when 0.8 is required
    cfg = ReasonConfig(min_mark_confidence=0.8)
    assert decide(_res2(MarkType.CIRCLE, 0.75), 1, cfg).kind == "reinspect"
    assert decide(_res2(MarkType.CIRCLE, 0.85), 1, cfg).skill == PLACE_MINOR
    # stage 1 unsure -> stage 2 is irrelevant
    d = decide(_res2(MarkType.CIRCLE, label=Label.UNCERTAIN, score=0.48), 1)
    assert (d.kind, d.stage) == ("reinspect", "stage1")
    # a minor circle is never the fallback: forced decisions always go to the defective plate
    d = decide(_res2(MarkType.UNKNOWN), 3, ReasonConfig(max_uncertain=3, reshow_available=True))
    assert d.forced and d.skill == PLACE_PINK


def test_plates_config(tmp_path):
    p = Plates.parse(None)
    assert (p.good.colour, p.minor.colour, p.defective.colour) == ("blue", "yellow", "pink")
    assert p.minor.instruction == "place the brick on the yellow plate" and p.minor.skill == PLACE_MINOR
    p = Plates.parse("good=green, minor=orange,defective=red")
    assert p.good.instruction == "place the brick on the green plate" and p.good.pose == "green"
    assert p.good.skill == PLACE_BLUE and p.defective.skill == PLACE_PINK  # ids fixed per role
    cfg = tmp_path / "plates.json"
    cfg.write_text(json.dumps({"minor": {"colour": "white", "instruction": "put it on the white tray",
                                         "pose": "tray"}}))
    p = Plates.parse(str(cfg))
    assert (p.minor.instruction, p.minor.pose, p.good.colour) == ("put it on the white tray", "tray", "blue")
    lib = SkillLibrary()
    assert lib[PLACE_MINOR].instruction == "place the brick on the yellow plate"
    p.apply(lib)
    assert lib[PLACE_MINOR].instruction == "put it on the white tray"
    with pytest.raises(ValueError):
        Plates.parse("great=gold")


def test_circle_brick_goes_to_minor_plate():
    ctl, world, _, policy, runlog = make_loop(["circle"])
    (r,) = ctl.run()
    assert (r.status, r.label, r.place_skill, r.forced) == ("placed", Label.MINOR, PLACE_MINOR, False)
    assert world.placed == [(1, "circle", "yellow")]
    assert "place the brick on the yellow plate" in policy.calls
    insp = events(runlog, "inspection")[0]
    assert insp["label"] == "defective" and insp["mark"] == "circle" and insp["outcome"] == "circle"
    assert set(insp["stage_latency_ms"]) == {"stage1", "stage2"}
    assert ctl.counts == {"placed_minor": 1}


def test_three_plates_in_one_run():
    ctl, world, *_ = make_loop(["plain", "circle", "cross"])
    ctl.run()
    assert [(k, w) for _, k, w in world.placed] == [("plain", "blue"), ("circle", "yellow"), ("cross", "pink")]
    assert ctl.counts == {"placed_perfect": 1, "placed_minor": 1, "placed_defective": 1}
    summary = ctl.log.summary()
    assert summary["inspector.stage1"]["n"] == 3 and summary["inspector.stage2"]["n"] == 2


def test_stage2_uncertain_falls_back_to_defective():
    ctl, world, _, _, runlog = make_loop(["blob"])  # clearly marked, but neither a ring nor a cross
    (r,) = ctl.run()
    assert [i["mark"] for i in r.inspections] == ["unknown", "unknown"]
    assert (r.label, r.forced, r.place_skill) == (Label.DEFECTIVE, True, PLACE_PINK)
    assert world.placed == [(1, "blob", "pink")]
    (fb,) = events(runlog, "uncertain_fallback")
    assert fb["stage"] == "stage2"


def test_stage2_uncertain_then_resolved():
    insp = fakes.ScriptedInspector([(Label.DEFECTIVE, MarkType.UNKNOWN, 0.55), (Label.DEFECTIVE, MarkType.CIRCLE)])
    ctl, world, *_ = make_loop(["circle"], inspector=insp)
    (r,) = ctl.run()
    assert (r.label, r.forced, len(r.inspections)) == (Label.MINOR, False, 2)
    assert world.placed == [(1, "circle", "yellow")]


def test_cli_dry_run_three_plates_custom_colours(tmp_path, capsys):
    from dlsj.loop import run

    skills = tmp_path / "skills.json"
    skills.write_text(json.dumps({n: {"min_duration_s": 0.05} for n in SkillLibrary().skills}))
    rc = run.main(["--dry-run", "--bricks", "plain,circle,cross,blob", "--fps", str(FPS), "--max-speed",
                   str(3 * FPS), "--ramp-speed", str(3 * FPS), "--fake-latency-ms", "5", "--status-port", "0",
                   "--skills-config", str(skills), "--log-dir", str(tmp_path / "run"),
                   "--plates", "good=green,minor=orange,defective=red"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "(1, 'plain', 'green'), (2, 'circle', 'orange'), (3, 'cross', 'red'), (4, 'blob', 'red')" in out
    ev = [json.loads(x) for x in (tmp_path / "run" / "events.jsonl").read_text().splitlines()]
    cfg = next(e for e in ev if e["event"] == "config")
    assert cfg["plates"]["minor"]["instruction"] == "place the brick on the orange plate"
