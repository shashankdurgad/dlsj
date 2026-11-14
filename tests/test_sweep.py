"""Spin-and-inspect: sweep aggregation strategies, the wrist_roll sweep on the fake arm, the controller with sweeps,
loading standard Anomalib exports (tiny OpenVINO stand-in, skipped without OpenVINO) and the sweep_test pipeline."""

import json
import threading

import numpy as np
import pytest

from dlsj import arms
from dlsj.contracts import FrameInspection, InspectionResult, Label, MarkType
from dlsj.inspect import aggregate as A
from dlsj.inspect.aggregate import AggregateConfig, aggregate, aggregate_all, sweep_result
from dlsj.loop import fakes
from dlsj.loop.controller import ControllerConfig
from dlsj.loop.poses import GRIPPER
from dlsj.loop.reasoner import ReasonConfig, inspection_record
from dlsj.loop.sweep import ROLL, SweepConfig, Sweeper, plan_angles

THR, LO, HI = 0.5, 0.4, 0.6
FPS = 200.0


def frames(scores, p_cross=None, marked_p=0.9):
    """FrameInspections from scores; frames above HI get stage 2 (`p_cross[i]`, default `marked_p`)."""
    out = []
    for i, s in enumerate(scores):
        lab = A.frame_label(s, LO, HI)
        mark, conf, p = None, 0.0, None
        if lab is Label.DEFECTIVE:
            p = (p_cross[i] if p_cross is not None else marked_p)
            if p is None:
                mark = MarkType.UNKNOWN
            else:
                conf = max(p, 1 - p)
                mark = MarkType.CROSS if p >= 0.5 else MarkType.CIRCLE
        region = (10, 10, 50, 50) if lab is not Label.PERFECT else None
        out.append(FrameInspection(i, -90 + 15 * i, s, lab, region, mark, conf, p, 1.0))
    return out


def outcomes(fr, cfg=AggregateConfig()):
    return {s: a.outcome() for s, a in aggregate_all(fr, THR, LO, HI, cfg).items()}


# ----------------------------------------------------------------------------- aggregation
def test_one_face_mark_topk_and_max_find_it_majority_misses_it():
    sc = [0.1] * 8 + [0.95, 0.9, 0.85, 0.8] + [0.1] * 1  # mark visible on 4 of 13 angles
    out = outcomes(frames(sc))
    assert out["topk"] is MarkType.CROSS
    assert out["max"] is MarkType.CROSS
    assert out["majority"] is MarkType.NONE  # the naive vote calls it plain
    assert out["median"] is MarkType.UNKNOWN  # the classic rule: frames disagree


def test_single_glare_frame_does_not_flip_topk():
    sc = [0.1] * 11 + [0.97]
    fr = frames(sc, p_cross=[None] * 11 + [None])  # stage 2 finds no mark in the glint
    out = outcomes(fr)
    assert out["topk"] is MarkType.NONE and out["majority"] is MarkType.NONE
    assert out["max"] is not MarkType.NONE  # one bright frame flips the max rule
    a = aggregate(fr, THR, LO, HI)
    assert a.summary["n_marked"] == 1 and "outlier" in a.summary["reason"]


def test_all_clean_is_plain_for_every_strategy():
    out = outcomes(frames([0.1, 0.15, 0.2, 0.12, 0.18, 0.1, 0.11, 0.13, 0.14, 0.2, 0.1, 0.1]))
    assert set(out.values()) == {MarkType.NONE}


def test_conflicting_stage2_votes_are_uncertain():
    sc = [0.1] * 8 + [0.9, 0.9, 0.9, 0.9]
    fr = frames(sc, p_cross=[None] * 8 + [0.95, 0.9, 0.1, 0.05])
    a = aggregate(fr, THR, LO, HI)
    assert a.label is Label.DEFECTIVE and a.mark is MarkType.UNKNOWN and a.outcome() is MarkType.UNKNOWN
    assert a.summary["mark_votes"] == {"circle": 2, "cross": 2, "none": 0}
    assert abs(a.summary["mark_probs"]["cross"] - 0.5) < 0.05


def test_weighted_vote_trusts_the_clearest_frames():
    # two frames clearly marked and confidently cross, two barely marked frames lean circle
    sc = [0.1] * 8 + [1.4, 1.4, 0.62, 0.62]
    fr = frames(sc, p_cross=[None] * 8 + [0.9, 0.9, 0.2, 0.2])
    a = aggregate(fr, THR, LO, HI, AggregateConfig(min_mark_margin=0.2, min_mark_confidence=0.6))
    assert a.mark is MarkType.CROSS and a.summary["mark_probs"]["cross"] > 0.6
    # unweighted, the same votes are a coin flip: mean P(cross) 0.55 (median's stage 2), a 2-2 tie (majority's)
    cfg = AggregateConfig(min_mark_margin=0.2, min_mark_confidence=0.6)
    assert A._stage2("median", fr, Label.DEFECTIVE, cfg, THR, HI)[0] is MarkType.UNKNOWN
    assert A._stage2("majority", fr, Label.DEFECTIVE, cfg, THR, HI)[0] is MarkType.UNKNOWN


def test_mark_not_found_in_most_marked_frames_is_unknown():
    fr = frames([0.1] * 8 + [0.9] * 4, p_cross=[None] * 8 + [0.9, None, None, None])
    assert aggregate(fr, THR, LO, HI).outcome() is MarkType.UNKNOWN


def test_two_marked_frames_needed_and_unclear_sweeps_are_uncertain():
    fr = frames([0.1] * 6 + [0.5] * 5 + [0.9])  # 5 frames inside the band, one marked
    assert aggregate(fr, THR, LO, HI).label is Label.UNCERTAIN
    fr = frames([0.1] * 10 + [0.9, 0.9])  # exactly two marked frames: enough
    assert aggregate(fr, THR, LO, HI).label is Label.DEFECTIVE


def test_without_stage2_marked_counts_as_cross():
    fr = [FrameInspection(i, None, s, A.frame_label(s, LO, HI)) for i, s in enumerate([0.1] * 9 + [0.9] * 3)]
    a = aggregate(fr, THR, LO, HI)
    assert a.mark is None and a.outcome() is MarkType.CROSS


def test_unknown_strategy_rejected():
    with pytest.raises(ValueError):
        AggregateConfig(strategy="mean")
    with pytest.raises(ValueError):
        aggregate(frames([0.1]), THR, LO, HI, strategy="nope")


def test_sweep_result_carries_distribution_and_alternatives():
    fr = frames([0.1] * 8 + [0.95, 0.9, 0.85, 0.8])
    heats = [np.full((4, 4), i, np.float32) for i in range(len(fr))]
    r = sweep_result(fr, THR, LO, HI, heatmaps=heats, device="X", mark_device="Y")
    assert isinstance(r, InspectionResult) and r.label is Label.DEFECTIVE and r.mark is MarkType.CROSS
    assert r.aggregation == "topk" and len(r.frames) == 12 and r.frame_scores[8] == pytest.approx(0.95)
    assert r.summary["alternatives"] == {"topk": "cross", "majority": "none", "max": "cross", "median": "unknown"}
    assert r.summary["best_index"] == 8 and r.region == fr[8].region and float(r.heatmap[0, 0]) == 8
    assert r.summary["topk_mean"] == pytest.approx(0.9)
    rec = inspection_record(r)  # what the event log + status page get
    assert rec["aggregation"] == "topk" and len(rec["frames"]) == 12 and rec["frames"][8]["mark"] == "cross"
    assert A.frames_from_record(rec["frames"])[8] == fr[8]
    json.dumps(rec)


def test_classic_inspection_result_unchanged():
    r = InspectionResult(Label.PERFECT, 0.1, 0.5)
    assert r.frames == () and r.aggregation == "" and r.summary == {}
    assert "frames" not in inspection_record(r)


# ----------------------------------------------------------------------------- limits + planning
def test_wrist_roll_limits_from_calibration(tmp_path):
    assert arms.raw_range_to_deg(0, 4095) == pytest.approx((-180.0, 180.0))
    (tmp_path / "f.json").write_text(json.dumps({"wrist_roll": {"range_min": 1024, "range_max": 3071}}))
    lo, hi = arms.joint_limits_deg("f", "wrist_roll", tmp_path)
    assert lo == pytest.approx(-90.0, abs=0.1) and hi == pytest.approx(90.0, abs=0.1)
    assert arms.joint_limits_deg("missing", "wrist_roll", tmp_path) is None
    assert arms.joint_limits_deg("follower") == pytest.approx((-180.0, 180.0))  # the repo calibration: full turn


def test_plan_clips_to_limits_and_starts_at_the_nearer_end():
    p = plan_angles(0.0, SweepConfig(frames=12, range_deg=180))
    assert len(p) == 12 and p[0] == pytest.approx(-90) and p[-1] == pytest.approx(90)
    p = plan_angles(150.0, SweepConfig(frames=12, range_deg=180, limits=(-180, 180), limit_margin_deg=5))
    assert max(p) == pytest.approx(175.0) and min(p) == pytest.approx(60.0)
    assert p[0] == pytest.approx(175.0)  # nearer end first (150 -> 175 is closer than 150 -> 60)
    assert plan_angles(10.0, SweepConfig(frames=1)) == [10.0]
    assert plan_angles(0.0, SweepConfig(frames=0)) == []
    with pytest.raises(ValueError):
        plan_angles(0.0, SweepConfig(limits=(0, 5), limit_margin_deg=5))


# ----------------------------------------------------------------------------- sweep motion on the fake arm
def held_rig(kind="side_cross"):
    world = fakes.FakeWorld(bricks=[kind])
    world.held, world.on_table = world.on_table, None
    start = world.poses["inspect"].copy()
    start[GRIPPER] = fakes.HOLD_GRIP
    robot = fakes.FakeRobot(world, start=start)
    cmd = start.copy()
    cmd[GRIPPER] = 0.0  # what the pick skill commanded: closed
    return world, robot, fakes.FakeCameras(world, robot), cmd


def fast_sweep(**kw):
    d = dict(frames=12, fps=FPS, max_speed_deg_s=1500.0, settle_ms=5.0, discard_frames=0, settle_timeout_s=0.5)
    d.update(kw)
    return SweepConfig(**d)


def test_sweep_visits_every_angle_holds_the_grasp_and_returns_to_start():
    world, robot, cams, cmd = held_rig()
    seen = []
    res = Sweeper(robot, cams, fast_sweep()).run(cmd, on_frame=lambda f, a: seen.append(a))
    assert res.status == "ok" and len(res.frames) == 12 and seen == res.angles
    assert np.allclose(res.angles, res.plan, atol=2.0)
    assert res.returned and abs(res.final_roll - cmd[ROLL]) < 2.0 and np.allclose(res.final_cmd, cmd)
    sent = np.stack(robot.sent)
    assert np.all(sent[:, GRIPPER] == 0.0)  # gripper target held closed throughout
    assert np.allclose(sent[:, :4], cmd[:4])  # only wrist_roll moves
    assert max(np.abs(np.diff(sent[:, ROLL]))) <= 1500.0 / FPS + 1e-3  # per-tick step cap
    assert all(s.move_ms >= 0 and s.settle_ms > 0 and s.grab_ms >= 0 for s in res.stops) and len(res.stops) == 12
    assert world.held is not None
    rec = res.record()
    assert rec["status"] == "ok" and len(rec["stops"]) == 12
    json.dumps(rec)


def test_sweep_clipped_to_limits():
    world, robot, cams, cmd = held_rig()
    res = Sweeper(robot, cams, fast_sweep(limits=(-40.0, 40.0), limit_margin_deg=5.0)).run(cmd)
    assert min(res.angles) > -36.5 and max(res.angles) < 36.5
    assert max(np.abs(np.stack(robot.sent)[:, ROLL])) <= 35.0 + 1e-3


def test_sweep_aborts_when_the_brick_drops_and_still_returns():
    world, robot, cams, cmd = held_rig()

    def drop(frame, angle):
        if len(got) == 2:
            world.held = None  # the brick slips out: the jaws close on nothing
        got.append(angle)

    got = []
    res = Sweeper(robot, cams, fast_sweep()).run(cmd, on_frame=drop)
    assert res.status == "grasp_lost" and len(res.frames) == 3
    assert res.returned and abs(res.final_roll - cmd[ROLL]) < 2.0


def test_sweep_stops_on_abort():
    world, robot, cams, cmd = held_rig()
    ev = threading.Event()
    res = Sweeper(robot, cams, fast_sweep(), ev).run(cmd, on_frame=lambda f, a: ev.set())
    assert res.status == "aborted" and len(res.frames) == 1 and not res.returned


def test_continuous_sweep_grabs_one_frame_per_planned_angle():
    world, robot, cams, cmd = held_rig()
    res = Sweeper(robot, cams, fast_sweep(continuous=True, continuous_speed_deg_s=600.0)).run(cmd)
    assert res.status == "ok" and len(res.frames) == 12 and res.returned
    assert all(np.diff(res.angles) > 0)


@pytest.mark.parametrize("kind,expect", [("side_cross", "cross"), ("side_circle", "circle"), ("glint", "none"),
                                         ("plain", "none"), ("cross", "cross")])
def test_fake_brick_sweeps_streamed_through_the_inspector(kind, expect):
    world, robot, cams, cmd = held_rig(kind)
    insp = fakes.FakeInspector(cams.roi)
    sess = insp.start_sweep()
    res = Sweeper(robot, cams, fast_sweep()).run(cmd, on_frame=sess.add)
    r = sess.result()
    assert res.status == "ok" and r.aggregation == "topk" and len(r.frames) == 12
    assert r.summary["alternatives"]["topk"] == expect
    if kind.startswith("side_"):
        assert r.summary["alternatives"]["majority"] == "none"  # the mark shows on 4/12 angles only
    if kind == "glint":
        assert r.summary["alternatives"]["max"] != "none"
    assert "tail" in r.stage_latency_ms


# ----------------------------------------------------------------------------- controller with sweeps
def sweep_loop(bricks, aggregate_name="topk", tmp_path=None):
    from tests.test_loop import make_loop

    ccfg = ControllerConfig(reason=ReasonConfig(), sweep=fast_sweep(), aggregate=aggregate_name)
    return make_loop(bricks, tmp_path, ccfg=ccfg)


def test_controller_sweep_topk_sorts_one_face_marks_and_glints(tmp_path):
    ctl, world, robot, policy, runlog = sweep_loop(["side_cross", "side_circle", "glint", "plain"], "topk", tmp_path)
    reports = ctl.run()
    assert [r.status for r in reports] == ["placed"] * 4
    assert [w for _, _, w in world.placed] == ["purple", "orange", "yellow", "yellow"]
    sweeps = [e for e in runlog.events if e["event"] == "sweep"]
    assert len(sweeps) == 4 and all(e["status"] == "ok" and e["returned"] for e in sweeps)
    insp = [e for e in runlog.events if e["event"] == "inspection"]
    assert all(e["aggregation"] == "topk" and len(e["frames"]) == 12 for e in insp)
    snap = ctl.snapshot()["last_inspection"]
    assert len(snap["frames"]) == 12 and "alternatives" in snap["summary"]
    stages = runlog.summary()
    assert stages["sweep"]["n"] == 4 and stages["sweep.stop"]["n"] == 48


def test_controller_sweep_majority_misses_one_face_marks():
    ctl, world, *_ = sweep_loop(["side_cross"], "majority")
    ctl.run()
    assert [w for _, _, w in world.placed] == ["yellow"]  # why topk is the default


def test_controller_without_sweep_keeps_three_frames():
    from tests.test_loop import make_loop

    ctl, world, robot, policy, runlog = make_loop(["cross"], ccfg=ControllerConfig(sweep=SweepConfig(frames=0)))
    ctl.run()
    insp = [e for e in runlog.events if e["event"] == "inspection"]
    assert len(insp[0]["frame_scores"]) == 3 and "frames" not in insp[0]
    assert not [e for e in runlog.events if e["event"] == "sweep"]


def test_controller_brick_dropped_during_sweep():
    ctl, world, robot, policy, runlog = sweep_loop(["side_cross"])
    real_fresh = ctl.cameras.fresh
    calls = []

    def fresh(name, n, timeout_s=2.0):
        out = real_fresh(name, n, timeout_s)
        calls.append(1)
        if len(calls) == 3 and world.held is not None:
            world.held = None
        return out

    ctl.cameras.fresh = fresh
    reports = ctl.run()
    assert reports[0].status == "dropped"
    assert [e for e in runlog.events if e["event"] == "sweep"][0]["status"] == "grasp_lost"


def test_controller_sweep_with_classic_inspector_falls_back_to_inspect():
    from tests.test_loop import make_loop

    ccfg = ControllerConfig(sweep=fast_sweep(), aggregate="topk")
    ctl, world, *_ = make_loop(["plain"], inspector=fakes.ScriptedInspector([Label.PERFECT]), ccfg=ccfg)
    reports = ctl.run()
    assert reports[0].status == "placed" and reports[0].label is Label.PERFECT


# ----------------------------------------------------------------------------- standard Anomalib export loader
def tiny_export(root, hw=(32, 32), a=0.05, b=-0.2):
    """Stand-in for `Engine.export(..., ExportType.OPENVINO)`: f32 NCHW [?,3,?,?] input in [0, 1], outputs
    pred_score (clamp(raw*a+b)), pred_label (i8), anomaly_map (clamp(map*a+b)), pred_mask (i8). raw map = 20 x the
    per-pixel channel max, raw score = its max; written to <root>/weights/openvino/model.xml."""
    ov = pytest.importorskip("openvino")
    import openvino.opset13 as ops

    x = ops.parameter([-1, 3, -1, -1], ov.Type.f32, name="input")
    raw_map = ops.multiply(ops.reduce_max(x, np.int64(1), keep_dims=True), np.float32(20.0))  # [B,1,H,W]
    raw_score = ops.reduce_max(raw_map, np.array([1, 2, 3], np.int64), keep_dims=False)  # [B]
    s = ops.clamp(ops.add(ops.multiply(raw_score, np.float32(a)), np.float32(b)), 0.0, 1.0)
    m = ops.clamp(ops.add(ops.multiply(raw_map, np.float32(a)), np.float32(b)), 0.0, 1.0)
    lab = ops.convert(ops.greater(s, np.float32(0.5)), ov.Type.i8)
    mask = ops.convert(ops.greater(m, np.float32(0.5)), ov.Type.i8)
    s.output(0).get_tensor().set_names({"pred_score"})
    lab.output(0).get_tensor().set_names({"pred_label"})
    m.output(0).get_tensor().set_names({"anomaly_map"})
    mask.output(0).get_tensor().set_names({"pred_mask"})
    d = root / "weights" / "openvino"
    d.mkdir(parents=True)
    ov.save_model(ov.Model([s, lab, m, mask], [x], "tiny_anomalib"), str(d / "model.xml"))
    return d / "model.xml"


def test_export_folder_is_detected_reshaped_and_cut_to_raw(tmp_path):
    from dlsj.inspect import model_io
    from dlsj.inspect.detector import AnomalibInspector

    tiny_export(tmp_path)
    (tmp_path / "evaluation.json").write_text(json.dumps({"validation_selected_threshold": 0.6}))
    m, base = model_io.resolve(tmp_path, crop=(0, 0, 64, 64), input_size=(32, 32), score_space="raw")
    assert m.source == "anomalib_export" and m.input_format == "f32_nchw" and m.score_space == "raw"
    assert m.normalisation["score"] == pytest.approx([0.05, -0.2])
    assert m.threshold == pytest.approx((0.6 + 0.2) / 0.05)  # evaluation.json threshold mapped to raw units
    assert m.calibration["threshold_source"].startswith("evaluation.json")
    frame = np.zeros((64, 64, 3), np.uint8)
    frame[10:20, 10:20] = 255  # a white patch: raw = 20 there, 0 elsewhere
    kw = dict(device="CPU", batch=1, fallback=("CPU",), crop=(0, 0, 64, 64), input_size=(32, 32), stage2="none")
    raw = AnomalibInspector.from_model(tmp_path, score_space="raw", **kw)
    nrm = AnomalibInspector.from_model(tmp_path / "weights" / "openvino" / "model.xml", score_space="normalized", **kw)
    shape = raw.backend.compiled.input(0).shape
    assert list(shape) == [1, 32, 32, 3]  # static, uint8 NHWC folded in
    sr, sn = raw.score_frames([frame]), nrm.score_frames([frame])
    assert sr[0] == pytest.approx(20.0, abs=0.2) and sn[0] == pytest.approx(min(1.0, 20 * 0.05 - 0.2), abs=0.01)
    assert nrm.manifest.threshold == pytest.approx(0.6)
    rec, heat = nrm.inspect_frame(frame, 0, 10.0)
    assert rec.pred_label == 1 and rec.label is Label.DEFECTIVE and rec.region is not None and heat.shape == (64, 64)
    dark = np.zeros((64, 64, 3), np.uint8)
    assert raw.inspect_frame(dark)[0].label is Label.PERFECT
    r = raw.inspect_sweep([dark] * 6 + [frame] * 3, list(range(9)))
    assert r.label is Label.DEFECTIVE and r.summary["n_marked"] == 3


def test_export_without_threshold_metadata_uses_the_post_processor_threshold(tmp_path):
    from dlsj.inspect import model_io

    tiny_export(tmp_path)
    m, _ = model_io.resolve(tmp_path, input_size=(32, 32))
    assert m.score_space == "normalized" and m.threshold == 0.5 and m.band_low < 0.5 < m.band_high <= 0.75


def test_calibrate_writes_a_manifest_that_loads_back(tmp_path):
    from dlsj.inspect import model_io
    from dlsj.inspect.core import write_rgb
    from dlsj.inspect.detector import AnomalibInspector

    tiny_export(tmp_path / "exp")
    for kind, val in (("good", 40), ("bad", 250)):
        (tmp_path / "data" / kind).mkdir(parents=True)
        for i in range(3):
            f = np.full((64, 64, 3), 30, np.uint8)
            f[5:15, 5:15] = val + i
            write_rgb(tmp_path / "data" / kind / f"b{i}_f0.png", f)
    out = tmp_path / "cal" / "manifest.json"
    m = model_io.calibrate_model(tmp_path / "exp", tmp_path / "data", out, "CPU", crop=None, input_size=(32, 32),
                                 score_space="raw")
    assert 3.5 < m.threshold < 19.5 and m.calibration["method"] == "separable-gap"
    insp = AnomalibInspector.from_model(out, device="CPU", fallback=("CPU",), stage2="none")
    assert insp.manifest.threshold == pytest.approx(m.threshold) and insp.manifest.score_space == "raw"


# ----------------------------------------------------------------------------- sweep_test pipeline
def test_hardware_dry_run_saves_a_sweep_folder_and_collects(tmp_path):
    from dlsj.loop import sweep_test

    fast = ["--fps", "300", "--sweep-speed", "2000", "--sweep-settle-ms", "5", "--sweep-discard", "0", "--yes"]
    assert sweep_test.main(["hardware", "--dry-run", "--fake-brick", "side_cross", "--label", "cross",
                            "--out", str(tmp_path / "sw"), *fast]) == 0
    (d,) = sweep_test.find_sweeps(tmp_path / "sw")
    doc = json.loads((d / "sweep.json").read_text())
    assert doc["label"] == "cross" and len(doc["frames"]) == 12 and doc["sweep"]["status"] == "ok"
    assert doc["inspection"]["strategies"]["topk"]["outcome"] == "cross"
    assert doc["inspection"]["strategies"]["majority"]["outcome"] == "none"
    assert (d / "contact.jpg").exists() and (d / "scores.png").exists() and (d / "frame_11.png").exists()
    assert sweep_test.main(["collect", "--dry-run", "--fake-brick", "plain", "--label", "plain", "--dataset",
                            str(tmp_path / "ds"), "--out", str(tmp_path / "sw2"), *fast]) == 0
    assert len(list((tmp_path / "ds" / "good").glob("*_f*.png"))) == 12
    assert sweep_test.main(["collect", "--from-sweeps", str(tmp_path / "sw"), "--dataset", str(tmp_path / "ds")]) == 0
    assert len(list((tmp_path / "ds" / "bad").glob("*_f*.png"))) == 12
    assert len((tmp_path / "ds" / "collect.jsonl").read_text().splitlines()) == 2


class ContrastBackend:
    """Fake stage 1 for synthetic sweeps: map = per-pixel deviation from the crop's median colour (8x8 cells)."""

    def __call__(self, batch):
        b = batch.astype(np.float32)
        med = np.median(b.reshape(len(b), -1, 3), axis=1)[:, None, None, :]
        dev = np.abs(b - med).max(axis=-1)
        n, h, w = dev.shape
        maps = dev.reshape(n, h // 8, 8, w // 8, 8).mean(axis=(2, 4)) / 255.0
        return maps.max(axis=(1, 2)), maps


def test_replay_eval_on_synthetic_sweeps(tmp_path):
    from dlsj.inspect.core import Manifest
    from dlsj.inspect.detector import AnomalibInspector
    from dlsj.inspect.synthetic import generate_sweeps
    from dlsj.loop import sweep_test

    metas = generate_sweeps(tmp_path / "sw", 2, 2, 2, frames=12, seed=3, glare_prob=0.5)
    assert len(metas) == 6 and all(m["n_visible"] < 12 for m in metas if m["label"] != "plain")
    assert any(0 < m["n_visible"] <= 6 for m in metas if m["label"] != "plain")  # one-face marks
    man = Manifest(model="x.xml", model_name="Fake", input_size=(128, 128), crop=(160, 80, 480, 400), threshold=0.5,
                   band_low=0.45, band_high=0.55)
    from dlsj.inspect.classify import ShapeRuleClassifier

    insp = AnomalibInspector(man, ContrastBackend(), batch=4, device="FAKE", classifier=ShapeRuleClassifier())
    sweeps = sweep_test.find_sweeps(tmp_path / "sw")
    ev = sweep_test.evaluate_sweeps(insp, sweeps, write_back=True)
    assert set(ev["tables"]) == {"topk", "majority", "max", "median"}
    for t in ev["tables"].values():
        assert t["n"] == 6 and t["correct"] + t["wrong"] + t["unknown"] == 6
        assert sum(sum(row.values()) for row in t["table"].values()) == 6
    assert ev["latency"]["frames_per_sweep"] == 12 and ev["latency"]["sweep_median_ms"] > 0
    info = sweep_test.inspector_info(insp)
    rep = sweep_test.write_report(ev, info, tmp_path / "report.html")
    text = rep.read_text()
    assert "<svg" in text and "topk" in text and "majority" in text
    doc = json.loads((sweeps[0] / "sweep.json").read_text())
    assert len(doc["inspection"]["per_frame"]) == 12 and (sweeps[0] / "contact.jpg").exists()
    json.dumps(sweep_test.eval_json(ev, info))


def test_eval_cli_runs_a_standard_export_end_to_end(tmp_path, capsys):
    from dlsj.inspect.synthetic import generate_sweeps
    from dlsj.loop import sweep_test

    tiny_export(tmp_path / "exp")
    generate_sweeps(tmp_path / "sw", 1, 1, 1, frames=12, seed=5)
    assert sweep_test.main(["eval", "--sweeps", str(tmp_path / "sw"), "--model", str(tmp_path / "exp"), "--device",
                            "CPU", "--crop", "160,80,480,400", "--input-size", "32", "--score-space", "raw",
                            "--stage2", "none", "--report", str(tmp_path / "r.html"), "--json",
                            str(tmp_path / "r.json")]) == 0
    out = capsys.readouterr().out
    assert "[topk]" in out and "[majority]" in out and "inference per sweep" in out
    doc = json.loads((tmp_path / "r.json").read_text())
    assert doc["info"]["score_space"] == "raw" and len(doc["sweeps"]) == 3
