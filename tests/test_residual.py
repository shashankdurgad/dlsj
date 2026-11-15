"""Pick-only grasp residual (deploy/smolvla/residual.py): module units, the controller hook on the real console
controller with a fake arm / policy / detector, `--residual off` byte-for-byte against the pre-residual controller,
replay through the detector -> tracker -> residual path, and the 30 Hz timing budget. No robot, no models."""

import importlib.util
import io
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# macOS Accelerate's spurious float32 matmul FP flags (results are finite and match float64 to 1e-6). The 2-D
# np.cross deprecation is an error here: numpy 2.5 (the XPS runtime) rejects it.
pytestmark = [pytest.mark.filterwarnings("error:Arrays of 2-dimensional vectors:DeprecationWarning"),
              pytest.mark.filterwarnings("ignore:.*encountered in matmul:RuntimeWarning")]
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy" / "smolvla"))
import residual as RES  # noqa: E402
import residual_replay as RR  # noqa: E402
import smolvla_control as S  # noqa: E402

import rtc_control as R  # noqa: E402  (smolvla_control put deploy/pi05 on sys.path)

# the controller before the residual hook (main before feat/residual-deploy was merged), for the byte-for-byte check
REFERENCE_COMMIT = "f9701eb"
# pi05_control (Desmond's console) at the version with the trace log, if this checkout has no dlsj.deploy.pi05_control
PI05_CONTROL_COMMIT = "6f800bd"
PICK = "pick up the brick"


def _git_show(commit: str, path: str) -> str | None:
    try:
        return subprocess.run(["git", "show", f"{commit}:{path}"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout
    except Exception:  # noqa: BLE001
        return None


def _load_source(name: str, src: str, tmp: Path):
    f = tmp / f"{name}.py"
    f.write_text(src)
    spec = importlib.util.spec_from_file_location(name, f)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def pc(tmp_path_factory):
    """The real console controller (dlsj.deploy.pi05_control), else its committed version with a stub runtime."""
    try:
        from dlsj.deploy import pi05_control

        return pi05_control
    except ImportError:
        pass
    src = _git_show(PI05_CONTROL_COMMIT, "dlsj/deploy/pi05_control.py")
    if src is None:
        pytest.skip("no dlsj.deploy.pi05_control and no git history for it")
    sys.modules.setdefault("dlsj.deploy.pi05_runtime", types.SimpleNamespace(PICK=PICK, PLACE="place it {colour}"))
    return _load_source("pi05_control_ref", src, tmp_path_factory.mktemp("pc"))


@pytest.fixture(scope="module")
def S_ref(tmp_path_factory):
    src = _git_show(REFERENCE_COMMIT, "deploy/smolvla/smolvla_control.py")
    if src is None:
        pytest.skip(f"reference commit {REFERENCE_COMMIT} not available")
    return _load_source("smolvla_control_ref", src, tmp_path_factory.mktemp("ref"))


@pytest.fixture(autouse=True)
def fixed_rtc_delay(monkeypatch):
    # the RTC request threshold depends on measured inference latency; pin it so runs are comparable tick for tick
    monkeypatch.setattr(R, "delay_actions", lambda *a, **k: 1)


# ----------------------------------------------------------------------------- fakes
START = np.array([0.0, -40.0, 50.0, 60.0, 0.0, 30.0])
PICK_GOAL = np.array([10.0, -20.0, 30.0, 70.0, 10.0, 2.0])
BRING_GOAL = np.array([-20.0, -60.0, 70.0, 20.0, 0.0, 2.0])


class FakeArm:
    def __init__(self):
        self.q, self.sent, self.reads = START.copy(), [], 0

    def read(self):
        self.reads += 1
        return self.q.copy()

    def send(self, cmd):
        cmd = np.asarray(cmd, float).copy()
        self.sent.append(cmd)
        self.q = self.q + 0.5 * (cmd - self.q)  # first-order servo


class FakePolicy:
    """50-step chunks towards a per-prompt goal; from the measured state (state_dependent) or a fixed clock."""

    def __init__(self, state_dependent=True):
        self.state_dependent, self.calls = state_dependent, 0

    def __call__(self, state, images, task):
        self.calls += 1
        goal = PICK_GOAL if task.startswith("pick") else BRING_GOAL
        base = np.asarray(state, float) if self.state_dependent else START + (goal - START) * min(1, self.calls / 20)
        return base + np.linspace(0, 1, 50)[:, None] * (goal - base) * 0.5


class FakeDetector:
    def __init__(self, boxes=()):
        self.boxes = list(boxes)

    def detect(self, frame):
        return [SimpleNamespace(box=b[:4], confidence=b[4]) for b in self.boxes]


class Cam:
    """A camera whose frame object changes every read (the detector thread processes new frames only)."""

    @property
    def frame(self):
        return np.zeros((480, 640, 3), np.uint8)


def cameras():
    return {"wrist": Cam(), "top": Cam(), "inspect": Cam()}


def run(cls, policy, n=200, script=None, trace=None):
    arm = FakeArm()
    c = cls(arm, cameras(), policy, bowls=["yellow", "purple", "orange"], fps=30, max_step=4.0, max_step_gripper=8.0,
            sync=True, trace=trace, horizon=15)
    script = script or {0: "start", 120: "m"}
    for t in range(n):
        if t in script:
            c.command(script[t])
        c.tick()
    if hasattr(c, "res_wrist"):
        c.res_wrist.stop()
    c.arm_reads = arm.reads
    return np.stack(arm.sent), c


def base_cls(Smod, pc):
    return Smod.make_controller_class(R.make_rtc_controller_class(pc.Controller))


def write_dummy(path: Path, r_const, s_dim=25, obs_dim=25, top_target=False, hidden=256, layers=4, **meta_kw):
    """A residual_actor.npz in export_residual.py's format whose output is the constant r_const."""
    rng = np.random.default_rng(0)
    dims = [s_dim + 30] + [hidden] * (layers - 1) + [6]
    arr = {}
    for i in range(layers):
        arr[f"W{i}"] = (rng.standard_normal((dims[i], dims[i + 1])) * 0.05).astype(np.float32)
        arr[f"b{i}"] = np.zeros(dims[i + 1], np.float32)
        if i < layers - 1:
            arr[f"g{i}"] = np.ones(dims[i + 1], np.float32)
            arr[f"beta{i}"] = np.zeros(dims[i + 1], np.float32)
    arr[f"W{layers - 1}"][:] = 0.0
    arr[f"b{layers - 1}"] = np.arctanh(np.asarray(r_const, np.float32))
    meta = {**RES.ZERO_META, "layers": layers, "s_dim": s_dim, "obs_dim": obs_dim, "top_target": top_target,
            "source": "test dummy", **meta_kw}
    np.savez(path, meta=np.array(json.dumps(meta)), **arr)
    return path


# ----------------------------------------------------------------------------- off == before
def test_off_is_byte_identical_to_the_pre_residual_controller(pc, S_ref):
    ref, _ = run(base_cls(S_ref, pc), FakePolicy())
    new, _ = run(base_cls(S, pc), FakePolicy())
    assert len(ref) > 150
    assert ref.tobytes() == new.tobytes()
    ns = SimpleNamespace(residual="off")
    assert S._residual_config(ns) is None  # off never builds (or imports) the residual


def test_zero_residual_is_identity_and_logs_every_pick_tick(pc):
    off, c_off = run(base_cls(S, pc), FakePolicy())
    cfg = RES.ResidualConfig(mode="on", weights="zero")
    det = FakeDetector([(250, 200, 330, 280, 0.9)])
    trace = io.StringIO()
    on, c = run(RES.make_residual_controller_class(base_cls(S, pc), cfg, detector=det), FakePolicy(), trace=trace)
    assert off.tobytes() == on.tobytes()
    recs = [json.loads(line) for line in trace.getvalue().splitlines()]
    res = [r for r in recs if r["kind"] == "residual"]
    after = [r for r in res if r["stage"] != "pick"]  # zero offset: only the hand-off tick once the gate closes
    assert res and len(after) == 1 and after[0]["handoff"] and not after[0]["gate"]
    dec = [r for r in res if r.get("decision")]
    assert [r["k"] for r in dec][:3] == [1, 6, 11]  # one decision every 5 gated ticks
    assert len(dec[0]["state"]) == 25 and len(dec[0]["plan"]) == 30
    # one extra joint read per decision (the obs sees the arm now), none on the other ticks
    assert c.arm_reads - c_off.arm_reads == len(dec)
    starts = [r for r in recs if r["kind"] == "residual_start"]
    assert starts[0]["slew_ok"] and starts[0]["max_step"] == [4.0] * 5 + [8.0]
    assert any(r["kind"] == "residual_start" for r in recs)
    assert c.res_stats["ticks"] == len(res) and c.res_stats["rejected"] == 0


def test_dummy_residual_moves_the_pick_only_and_fades_out(pc, tmp_path):
    w = write_dummy(tmp_path / "residual_actor.npz", [0.5, -0.5, 0.0, 0.3, -0.3, 0.6])
    policy = lambda: FakePolicy(state_dependent=False)  # noqa: E731  same plan in both runs
    off, _ = run(base_cls(S, pc), policy())
    trace = io.StringIO()
    cls = RES.make_residual_controller_class(base_cls(S, pc), RES.ResidualConfig(mode="on", weights=str(w)),
                                             detector=FakeDetector())
    on, _ = run(cls, policy(), trace=trace)
    recs = [json.loads(line) for line in trace.getvalue().splitlines() if '"residual"' in line]
    gated = [r for r in recs if r["gate"]]
    assert gated and max(abs(v) for r in gated for v in r["o"]) > 0
    # rate limit: xyz <= 5 mm, wrist <= 3 deg per tick; gripper offset = +0.6 * 30 units at once
    o = np.array([r["o"] for r in gated])
    assert np.abs(np.diff(o[:, :3], axis=0)).max() <= 0.005 + 1e-9
    assert np.abs(np.diff(o[:, 3:5], axis=0)).max() <= 3.0 + 1e-9
    assert o[-1] == pytest.approx([0.0175, -0.0175, 0.0, 3.0, -3.0, 18.0], abs=1e-3)
    assert all(np.abs(np.subtract(r["goal_res"], r["goal"]))[:5].max() <= 20.0 for r in recs)
    # the commands differ while picking, the fade ends within fade_ticks of the switch to bring
    diff = np.abs(on - off).max(1)
    assert diff[:100].max() > 0.5
    fade = [r for r in recs if not r["gate"]]
    assert 0 < len(fade) <= 6 and fade[-1]["o"] == pytest.approx([0] * 6, abs=1e-9)
    assert diff[-30:].max() < 1e-6  # back on the plan once faded (rate-limited catch-up included)


def test_shadow_runs_but_never_changes_the_commands(pc, tmp_path):
    w = write_dummy(tmp_path / "residual_actor.npz", [0.9, 0.9, 0.5, 0.9, 0.9, 0.9])
    off, _ = run(base_cls(S, pc), FakePolicy())
    trace = io.StringIO()
    cls = RES.make_residual_controller_class(base_cls(S, pc), RES.ResidualConfig(mode="shadow", weights=str(w)),
                                             detector=FakeDetector())
    sh, _ = run(cls, FakePolicy(), trace=trace)
    assert off.tobytes() == sh.tobytes()
    assert '"applied": false' in trace.getvalue() and '"decision": true' in trace.getvalue()


def test_live_kill_switch_fades_out(pc, tmp_path):
    w = write_dummy(tmp_path / "residual_actor.npz", [0.9, 0.0, 0.0, 0.0, 0.0, 0.0])
    trace = io.StringIO()
    cls = RES.make_residual_controller_class(base_cls(S, pc), RES.ResidualConfig(mode="on", weights=str(w)),
                                             detector=FakeDetector())
    _, c = run(cls, FakePolicy(), n=110, script={0: "start", 60: "residual_off"}, trace=trace)
    recs = [json.loads(line) for line in trace.getvalue().splitlines() if '"residual"' in line]
    after = [r for r in recs if not r["gate"]]
    assert c.res_live == "off" and 0 < len(after) <= 6 and after[-1]["o"] == pytest.approx([0] * 6, abs=1e-9)


# ----------------------------------------------------------------------------- module units
def test_gate_on_the_deployed_prompts():
    for c in ("blue", "brown", "red", "white", "yellow"):
        assert RES.is_pick_prompt(f"pick up the {c} brick")
    assert RES.is_pick_prompt("pick up the brick and hold it in front of the inspection camera")
    for p in ("bring the brick to the inspection camera", "place the brick in the orange bowl", "", None):
        assert not RES.is_pick_prompt(p)


def test_kin_matches_the_wrist_camera_fk_and_ik_reaches_the_offset():
    kin = RES.Kin(RES.joint_offsets(RES.ZERO_META), RES.ZERO_META["tip_local"])
    cam = RES.WT.WristCamera(offsets_rad=kin.off.tolist(), tip_local=RES.ZERO_META["tip_local"])
    j = np.array([10, -30, 40, 60, 5, 20.0])
    p, Rm = kin.fk(j)
    assert p == pytest.approx((cam.gripper_pose(j)[0] @ cam.tip)[:3], abs=1e-12)
    for o in ([0.02, 0, 0], [0, -0.035, 0], [0, 0, 0.02]):
        q = kin.ik_tip(j, p + Rm @ np.array(o))
        # 2 damped Gauss-Newton steps, as in the twin (kin.py): ~1 mm left at the 3.5 cm bound
        assert np.linalg.norm(kin.fk(q)[0] - (p + Rm @ np.array(o))) < 2e-3
        assert (q[3:] == j[3:]).all()


def test_obs_layout_and_plan_feature():
    res = RES.GraspResidual(RES.ZeroPolicy({**RES.ZERO_META, "top_target": True, "obs_dim": 29, "s_dim": 29}))
    res.reset(top_xyz=np.array([0.25, -0.05, RES.WT.BRICK_CENTRE_Z]))
    res.pick, res.t_pick = True, 48
    m = np.array([10, -30, 40, 60, 5, 20.0])
    s = res.obs(m, cmd_prev=m - [0, 0, 0, 0, 0, 3.0], det=RES.Det(True, np.array([0.01, -0.02, 0.005])))
    assert s.shape == (29,)
    assert s[:6] == pytest.approx(m / 100) and s[6:9] == pytest.approx([0.1, -0.2, 0.05]) and s[9] == 1.0
    assert s[10] == pytest.approx(0.3) and s[11] == pytest.approx(48 / 240) and s[24] == 1.0 and s[28] == 1.0
    p, Rm = res.kin.fk(m)
    assert s[25:28] == pytest.approx(Rm.T @ (res.top_xyz - p) * 10)
    pf = RES.GraspResidual.plan_feat([m + 2, m + 4], m)
    assert pf.shape == (30,) and pf[:6] == pytest.approx([0.1] * 6) and pf[-6:] == pytest.approx([0.2] * 6)
    assert res.obs(m, None, RES.Det())[9] == 0.0
    res.top_xyz = None
    assert res.obs(m, None, RES.Det())[25:29] == pytest.approx([0, 0, 0, 0])
    assert RES.GraspResidual(RES.ZeroPolicy()).obs(m, None, RES.Det()).shape == (25,)  # the current exports


def test_state_path_models_get_the_plan_repeated(tmp_path):
    w = write_dummy(tmp_path / "a.npz", [0.1] * 6, s_dim=25 + 120, obs_dim=25, top_target=False)
    res = RES.GraspResidual(RES.load_policy(str(w)))
    m = START.copy()
    _, info = res.step(m, m, m, [m + 1] * 5, RES.Det(), True)
    assert info["decision"] and res.plan_reps == 4 and info["state"].shape == (25,)


@pytest.mark.parametrize("bad", [
    {"latent": True}, {"residual_scale": [0.2, 0.035, 0.02, 10, 10, 30]}, {"obs_dim": 29},
    {"residual_scale": None}, {"max_step": None}, {"fixed_noise": None}, {"finetune": None},
    {"controller_delay": None}, {"max_step": "fast"}, {"max_step": "4,8,2"}, {"tip_local": [0.0, -0.09]}])
def test_load_refuses_what_the_robot_cannot_reproduce(tmp_path, bad):
    meta = {k: v for k, v in bad.items() if v is not None}
    w = write_dummy(tmp_path / "a.npz", [0.0] * 6, **meta)
    gone = [k for k, v in bad.items() if v is None]
    if gone:  # drop the key altogether (an export from before the field existed)
        z = dict(np.load(w))
        m = json.loads(str(z.pop("meta")))
        for k in gone:
            m.pop(k)
        np.savez(w, meta=np.array(json.dumps(m)), **z)
    with pytest.raises(SystemExit):
        RES.load_policy(str(w))


def test_gate_off_with_zero_offset_returns_the_goal_object():
    res = RES.GraspResidual(RES.ZeroPolicy())
    g = START.copy()
    out, _ = res.step(g, g, g, [g], RES.Det(), True)
    assert out is g
    out, _ = res.step(g, g, g, [g], RES.Det(), False)
    assert out is g and res.idle


# ----------------------------------------------------------------------------- replay + timing
def _synthetic_episode(n=90):
    """A pick that descends onto a brick: wrist boxes rendered from the true brick position by the wrist camera."""
    cam = RES.WT.WristCamera(offsets_rad=RES.joint_offsets(RES.ZERO_META), tip_local=RES.ZERO_META["tip_local"])
    a, b = np.array([0.0, -45.0, 45.0, 75.0, 0.0, 30.0]), np.array([0.0, -20.0, 30.0, 75.0, 0.0, 30.0])
    states = np.stack([a + (b - a) * min(1, i / 60) for i in range(n)])
    states[70:, 5] = np.linspace(30, 5, n - 70)
    brick = RES.Kin(cam.off, RES.ZERO_META["tip_local"]).fk(states[60])[0].copy()
    brick[2] = RES.WT.BRICK_CENTRE_Z
    frames, boxes = [], []
    for s in states:
        uv, depth = cam.project(s, brick[None])
        half = cam.fx * 0.016 / max(depth[0], 0.05)
        u, v = uv[0]
        boxes.append([(u - half, v - half, u + half, v + half, 0.85)])
        frames.append(np.zeros((480, 640, 3), np.uint8))
    tasks = ["pick up the blue brick"] * 80 + ["bring the brick to the inspection camera"] * (n - 80)
    return {"state": states, "action": np.roll(states, -3, 0), "tasks": tasks, "wrist": frames, "top": []}, boxes, brick


class ScriptedDetector:
    def __init__(self, boxes):
        self.boxes, self.i = boxes, 0

    def detect(self, frame):
        b = self.boxes[min(self.i, len(self.boxes) - 1)]
        self.i += 1
        return [SimpleNamespace(box=x[:4], confidence=x[4]) for x in b if 0 <= x[0] and x[2] <= 640 and x[3] <= 480]


def test_replay_detector_tracker_residual_path(tmp_path):
    ep, boxes, brick = _synthetic_episode()
    w = write_dummy(tmp_path / "a.npz", [0.2, -0.2, 0.0, 0.1, -0.1, 0.0])
    log = io.StringIO()
    rep = RR.replay(ep, RES.load_policy(str(w)), ScriptedDetector(boxes), log=log)
    assert rep["pick_frames"] == 80 and rep["det_rate_pick"] > 0.5
    rows = [json.loads(line) for line in log.getvalue().splitlines()]
    # the tracker's brick estimate lands on the true brick (the box -> table-plane ray, brick standing studs-up)
    ok = [r for r in rows if r.get("det_valid")]
    assert ok
    cam = RES.WT.WristCamera(offsets_rad=RES.joint_offsets(RES.ZERO_META), tip_local=RES.ZERO_META["tip_local"])
    for r in ok[::10]:
        s = ep["state"][r["i"]]
        M = cam.gripper_pose(s)[0]
        true_rel = M[:3, :3].T @ (brick - (M @ cam.tip)[:3])
        assert np.linalg.norm(np.asarray(r["det_rel"]) - true_rel) < 0.01
    assert rep["max_change_outside_pick"] < 20.0  # the fade, nothing after it
    assert rep["residual_ms_p99"] < 5.0


@pytest.mark.skipif(not (os.environ.get("DLSJ_REPLAY_DATASET") and os.environ.get("DLSJ_WRIST_DETECTOR")),
                    reason="set DLSJ_REPLAY_DATASET (LeRobot root) and DLSJ_WRIST_DETECTOR (OpenVINO dir)")
def test_replay_recorded_episode_with_the_real_detector():
    det = RES.make_wrist_detector(os.environ["DLSJ_WRIST_DETECTOR"], os.environ.get("DLSJ_DETECTOR_DEVICE", "CPU"))
    ep = RR.load_episode(os.environ["DLSJ_REPLAY_DATASET"], int(os.environ.get("DLSJ_REPLAY_EPISODE", "2")))
    rep = RR.replay(ep, RES.ZeroPolicy(), det)
    print(json.dumps(rep))
    assert rep["det_rate_pick"] > 0.3
    assert rep["max_abs_goal_change"] == [0.0] * 6  # zero residual: the plan untouched
    assert rep["track_ms_p99"] < 5.0 and rep["residual_ms_p99"] < 5.0


def test_timing_fits_the_30hz_budget(tmp_path):
    """Per control tick (decision every 5th): full-size network + obs + IK. Per wrist frame: tracker update."""
    w = write_dummy(tmp_path / "a.npz", [0.3, -0.3, 0.1, 0.2, -0.2, 0.1], s_dim=29 + 120, obs_dim=29, top_target=True)
    res = RES.GraspResidual(RES.load_policy(str(w)))
    m = START.copy()
    ms = []
    for i in range(1500):
        t0 = time.perf_counter()
        res.step(m + 0.01 * i % 3, m, m, [m + 1] * 5, RES.Det(True, np.array([0.01, 0.0, -0.01])), True)
        ms.append((time.perf_counter() - t0) * 1e3)
    cam = RES.WT.WristCamera(offsets_rad=res.kin.off.tolist(), tip_local=res.meta["tip_local"])
    RES.warm_fin_cache(cam)
    trk = RES.SeededTracker(cam)
    tk = []
    for i in range(300):
        j = np.array([0, -30, 40, 70, 0, 30 - i % 30.0])
        t0 = time.perf_counter()
        trk.update([(250, 200, 330, 280, 0.9), (20, 300, 200, 480, 0.6), (400, 100, 470, 170, 0.7)], j)
        tk.append((time.perf_counter() - t0) * 1e3)
    p99, tp99 = np.percentile(ms, 99), np.percentile(tk, 99)
    print(f"residual step p50 {np.median(ms):.3f} p99 {p99:.3f} ms; tracker p50 {np.median(tk):.3f} p99 {tp99:.3f} ms")
    assert p99 < 5.0 and tp99 < 5.0  # 33 ms per tick; the detector itself is ~4 ms on the NPU, in its own thread


# ----------------------------------------------------------------------------- the controller it was trained behind
def _ns(*argv):
    import argparse

    ap = argparse.ArgumentParser()
    S._residual_args(ap)
    return ap.parse_args(list(argv))


def test_slew_check_against_the_meta():
    m = RES.ZERO_META
    assert RES.meta_max_step(m) == (4.0, 8.0) and RES.meta_max_step({"max_step": [4, 8]}) == (4.0, 8.0)
    assert RES.check_controller(m, 4.0, 8.0) == []
    assert RES.check_controller(m, np.array([4.0] * 5 + [8.0])) == []
    assert len(RES.check_controller(m, 8.0, 8.0)) == 1 and len(RES.check_controller(m, 5.0, 4.0)) == 2
    assert RES.check_controller(m, np.array([5.0] * 5 + [10.0]))  # the console's +/- keys scale it live


def test_residual_config_refuses_a_different_controller(tmp_path):
    w = str(write_dummy(tmp_path / "a.npz", [0.0] * 6))
    on = _ns("--residual", "on", "--residual-weights", w)
    _, cfg = S._residual_config(on, ["--max-step", "4"], np.zeros(6))  # gripper default 8 = trained
    assert "matches" in cfg.slew
    with pytest.raises(SystemExit):
        S._residual_config(on, [], np.zeros(6))  # pi05_control's defaults are 8 / 8
    with pytest.raises(SystemExit):
        S._residual_config(on, ["--max-step=4", "--max-step-gripper", "4"], np.zeros(6))
    _, cfg = S._residual_config(_ns("--residual", "shadow", "--residual-weights", w), [], np.zeros(6))
    assert "trained behind" in cfg.slew  # shadow only warns: nothing is applied
    with pytest.raises(SystemExit):
        S._residual_config(on, ["--max-step", "4"], np.array([0, 0, 0, 0, 0, 2.0]))  # --joint-offset


def test_residual_goes_in_before_the_controller_slew(pc, tmp_path):
    """The twin's order: residual -> the controller's --max-step slew of the final command -> clamp."""
    w = write_dummy(tmp_path / "a.npz", [0.9, -0.9, 0.5, 0.9, 0.9, 0.9])
    trace = io.StringIO()
    cls = RES.make_residual_controller_class(base_cls(S, pc), RES.ResidualConfig(mode="on", weights=str(w)),
                                             detector=FakeDetector())
    sent, c = run(cls, FakePolicy(), trace=trace)
    recs = [json.loads(line) for line in trace.getvalue().splitlines()]
    n = 0
    for a, b in zip(recs, recs[1:]):
        if a["kind"] == "residual" and a["applied"] and b["kind"] == "tick":
            assert b["target"] == pytest.approx(a["goal_res"], abs=2e-3)  # the slew's input is the corrected goal
            n += 1
    assert n > 50
    steps = np.abs(np.diff(sent, axis=0))
    assert steps[:, :5].max() <= 4.0 + 1e-9 and steps[:, 5].max() <= 8.0 + 1e-9


@pytest.mark.skipif(not os.environ.get("DLSJ_RESIDUAL_EXPORT"),
                    reason="set DLSJ_RESIDUAL_EXPORT to an export_residual.py deploy_* dir")
def test_real_export_meta_matches_the_spec():
    d = Path(os.environ["DLSJ_RESIDUAL_EXPORT"]).expanduser()
    pol = RES.load_policy(str(d))
    m = pol.meta
    assert m["s_dim"] == m["obs_dim"] == 25 and m["a_dim"] == 30 and not m["latent"] and not m["top_target"]
    assert m["tip_local"] == pytest.approx([0.0063, 0.001, -0.0902], abs=1e-6)  # the fin-ray grasp point
    assert RES.joint_offsets_name(m) == "W9"
    assert RES.meta_max_step(m) == (4.0, 8.0) and m["fixed_noise"] and m["finetune"] and m["controller_delay"]
    assert m["residual_scale"] == pytest.approx([0.035, 0.035, 0.02, 10, 10, 30], rel=1e-6)
    rng = np.random.default_rng(1)
    s = rng.standard_normal((16, 25)).astype(np.float32) * 0.5
    p = rng.standard_normal((16, 30)).astype(np.float32) * 0.3
    out = np.stack([pol(s[i], p[i]) for i in range(16)])
    assert np.isfinite(out).all() and np.abs(out).max() <= 1.0
    if (d / "residual_actor.onnx").exists():
        pytest.importorskip("openvino")
        ov = RES.load_policy(str(d), "CPU")
        got = np.stack([ov(s[i], p[i]) for i in range(16)])
        assert np.abs(got - out).max() < 1e-4  # numpy (the default) and OpenVINO agree


def test_ab_counts_picks_closes_and_modes(pc, tmp_path):
    import residual_ab as AB

    w = write_dummy(tmp_path / "a.npz", [0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    class ClosePolicy(FakePolicy):  # opens the jaw to 60, then closes it once (a clock, as MolmoAct2's close)
        def __call__(self, state, images, task):
            c = super().__call__(state, images, task)
            c[:, 5] = 60.0 if self.calls < 4 else 0.0
            return c

    # M at tick 120 hands over to bring
    for mode in ("off", "shadow", "on"):
        trace = io.StringIO()
        cls = base_cls(S, pc) if mode == "off" else RES.make_residual_controller_class(
            base_cls(S, pc), RES.ResidualConfig(mode=mode, weights=str(w)), detector=FakeDetector())
        run(cls, ClosePolicy(), trace=trace)
        f = tmp_path / f"{mode}.jsonl"
        f.write_text(trace.getvalue())
        (p,) = AB.picks(f)
        assert p["mode"] == mode and p["closes"] == 1 and p["grasped"] and p["first_attempt"] and p["end"] == "bring"
        assert (p["decisions"] > 0) == (mode != "off")
    assert AB.fisher(10, 0, 0, 10) < 0.001 and AB.fisher(5, 5, 5, 5) == 1.0
    assert AB.wilson(0, 10)[0] == 0.0 and AB.wilson(10, 10)[1] == 1.0
