"""Pick-only grasp residual on the robot: a small trained MLP adds a bounded correction to the VLA's plan while the
pick prompt is active (spec: `twin_vendor/sim/twin/rl/DEPLOY.md`, copied from the twin).

Per control tick, only while the controller runs the policy (phase "running"):
* gate = `is_pick_prompt(prompt)` (deterministic text filter, `twin_vendor/.../rl/gate.py`); any other prompt (bring,
  place) closes it and the applied offset fades to 0 over `fade_ticks`;
* every `decision_ticks` gated ticks: obs (joints, wrist brick target, load, time, applied offset, previous output,
  gate, top-camera target) + plan (next 5 planned goals - measured) -> r in [-1, 1]^6 -> offset target = r * scale;
* every tick: the applied offset moves toward its target at <= rate; the tick's planned goal g becomes
  IK(g with wrist += offset, tip(g) + R(g) @ offset_xyz), gripper clip(g + offset, 0, 100); then the controller's own
  max-step / sync band / LeRobot clamp, unchanged.
Bounds, rates, fingertip point and joint offsets come from the export's meta (`residual_actor.npz`), never from here.

Wrist target: `OVBrickDetector` (brick-detector-v1, plain OpenVINO on the NPU, never ultralytics) in its own thread,
only while the gate is open, feeding `TargetTracker` (`twin_vendor/.../perception/wrist_target.py`, unchanged).
`--residual-top-seed` also seeds the acquisition and the top-target obs from the loop's top-camera target
(`SeededTracker`, `TopCamera`); off by default. The top camera (twin calib/cameras_fit.json, fitted with the joe_cal2
joint offsets) and the residual's FK (W9 offsets, fitted against a different top camera) disagree at table level:
on recorded picks the grasped brick sits ~5-7 cm from the close-frame FK fingertip, a rotation of ~12-13 deg about
the base plus ~2 cm radially (about 2 cm of it is the W9 vs joe_cal2 pan offset). Image coordinates, frame, table
plane and intrinsics match the twin's scene/layout.py; the twin's own layouts show the same bias (1.7-4.6 cm, joe
offsets). Until the twin refits the top camera with the W9 FK, the top-target obs is "not available" (valid 0, as
the sim's dropouts) and the tracker uses its own gating.

Modes (`--residual`): off = nothing is built, the controller is exactly the old one; shadow = detector + residual run
and are logged, the arm gets the unmodified plan; on = applied. `--residual-weights zero` = an all-zero residual
(identity, for testing the plumbing). POST /ctl/residual_off fades it out live; /ctl/residual_on re-arms it.

Vendored twin files (`twin_vendor/sim/twin/...`, refreshed by `sync_twin_vendor.sh`): watson ~/Documents/dlsj at
b109014 + the working tree of 2026-11-15 02:39 UTC.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import math
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger("residual")
HERE = Path(__file__).resolve().parent
VENDOR = HERE / "twin_vendor" / "sim" / "twin"
MODES = ("off", "shadow", "on")
DEFAULT_DETECTOR = "~/inference/models/brick-detector-v1"
# DEPLOY.md defaults, used only by the zero (identity) residual; a real export must carry its own meta
ZERO_META = {"layers": 0, "s_dim": 29, "a_dim": 30, "obs_dim": 29, "latent": False, "top_target": True,
             "residual_scale": [0.035, 0.035, 0.02, 10.0, 10.0, 30.0], "rate_per_tick": [0.005, 0.005, 0.005, 3.0, 3.0, None],
             "tip_local": [0.0063, 0.001, -0.0902], "joint_offsets": None, "decision_ticks": 5, "fade_ticks": 6,
             "source": "zero"}
# hard sanity limits on an export's bounds (DEPLOY.md: 3.5 cm / 2 cm / 10 deg / 30 units)
MAX_SCALE = [0.05, 0.05, 0.05, 15.0, 15.0, 40.0]
TIME_NORM = 240.0  # obs[11] = ticks since the pick prompt / 240
PLAN_STEPS, PLAN_NORM = 5, 20.0


def _load_vendored(name: str, rel: str):
    """Import a vendored twin module by path (no `sim` package on sys.path: the deploy checkout may have its own)."""
    key = f"twin_vendor_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, VENDOR / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod  # dataclasses look the module up while the class body runs
    spec.loader.exec_module(mod)
    return mod


WT = _load_vendored("wrist_target", "perception/wrist_target.py")
GATE = _load_vendored("gate", "rl/gate.py")
is_pick_prompt = GATE.is_pick_prompt


# ----------------------------------------------------------------------------- policy
class ZeroPolicy:
    """Identity residual (r = 0): the whole path runs, the arm gets exactly the plan."""

    def __init__(self, meta: dict | None = None):
        self.meta = dict(meta or ZERO_META)

    def __call__(self, state, plan) -> np.ndarray:
        return np.zeros(6, np.float32)


class OVResidual:
    """The export's residual_actor.onnx on an OpenVINO device (static (1, S) / (1, 30) inputs)."""

    def __init__(self, onnx: Path, meta: dict, device: str = "CPU"):
        import openvino as ov

        self.meta = meta
        self.model = ov.Core().compile_model(str(onnx), device.upper())
        self.req = self.model.create_infer_request()

    def __call__(self, state, plan) -> np.ndarray:
        out = self.req.infer({"state": np.asarray(state, np.float32)[None], "plan": np.asarray(plan, np.float32)[None]})
        return np.asarray(next(iter(out.values())), np.float32).reshape(-1)[:6]


def load_policy(weights: str, device: str = "numpy"):
    """'zero' -> ZeroPolicy; a residual_actor.npz (or its deploy/ dir) -> numpy ResidualPolicy, or the .onnx next to
    it on an OpenVINO device. The meta is validated (refuses what the robot cannot reproduce)."""
    if weights in ("", "zero"):
        return ZeroPolicy()
    p = Path(weights).expanduser()
    if p.is_dir():
        p = p / "residual_actor.npz"
    ER = _load_vendored("export_residual", "rl/export_residual.py")
    pol = ER.ResidualPolicy(p)
    validate_meta(pol.meta, pol)
    if device.lower() != "numpy":
        onnx = p.with_suffix(".onnx")
        if not onnx.exists():
            raise SystemExit(f"--residual-device {device}: no {onnx} (export with --onnx)")
        meta = pol.meta
        pol = OVResidual(onnx, meta, device)
    return pol


def validate_meta(meta: dict, pol=None) -> None:
    need = [k for k in ("residual_scale", "rate_per_tick", "tip_local", "s_dim", "a_dim", "obs_dim") if k not in meta]
    if need:
        raise SystemExit(f"residual export meta has no {need}: re-export with the current export_residual.py")
    if meta.get("latent"):
        raise SystemExit("this residual needs the MolmoAct2 pooled latent, which the deployed export does not output")
    if meta["a_dim"] != PLAN_STEPS * 6:
        raise SystemExit(f"residual a_dim {meta['a_dim']} != {PLAN_STEPS * 6} (plan feature)")
    obs = 25 + (4 if meta.get("top_target") else 0)
    if meta["obs_dim"] != obs:
        raise SystemExit(f"residual obs_dim {meta['obs_dim']} != {obs} (top_target={meta.get('top_target')})")
    extra = meta["s_dim"] - meta["obs_dim"]
    if extra < 0 or extra % (PLAN_STEPS * 6):
        raise SystemExit(f"residual s_dim {meta['s_dim']}: obs {meta['obs_dim']} + {extra} is not k x 30 plan features")
    scale = np.asarray(meta["residual_scale"], float)
    if scale.shape != (6,) or (np.abs(scale) > MAX_SCALE).any():
        raise SystemExit(f"residual_scale {scale.tolist()} outside the safe bounds {MAX_SCALE}")
    W = getattr(pol, "W", None) if pol is not None else None
    if W and (W[0].shape[0] != meta["s_dim"] + meta["a_dim"] or W[-1].shape[1] != 6):
        raise SystemExit(f"residual weights {W[0].shape} -> {W[-1].shape} do not match the meta dims")


def joint_offsets(meta: dict) -> list[float]:
    """The FK/IK joint offsets (rad, 5) the residual was trained with; falls back to the twin default (W9) loudly."""
    if meta.get("joint_offsets") is not None:
        return [float(v) for v in meta["joint_offsets"][:5]]
    res = json.loads((VENDOR / "align/joint_fit/result.json").read_text())
    w9 = res.get("grasp_depth_candidates", {}).get("W9", {}).get("joint_offsets_rad_for_sim")
    log.warning("residual meta has no joint_offsets: using the twin default %s", "W9" if w9 else "W")
    return [float(v) for v in (w9 or res["joint_offsets_rad_for_sim"])[:5]]


# ----------------------------------------------------------------------------- kinematics
class Kin:
    """numpy port of the twin's `rl/kin.py` (FK + 2-step damped-LS fingertip IK on pan / lift / elbow), same URDF
    chain as `wrist_target.WristCamera`. Joints in LeRobot degrees (cal2 frame), gripper 0-100."""

    def __init__(self, offsets_rad, tip_local, urdf: str | Path = WT.URDF):
        self.T = WT._chain(str(urdf))
        self.off = np.asarray(offsets_rad[:5], float)
        self.tip = np.r_[np.asarray(tip_local, float), 1.0]

    def pose(self, joints) -> np.ndarray:
        q = np.deg2rad(np.asarray(joints, float)[:5]) + self.off
        M = np.eye(4)
        for i in range(5):
            c, s = math.cos(q[i]), math.sin(q[i])
            Rz = np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])
            M = M @ self.T[i] @ Rz
        return M

    def fk(self, joints):
        M = self.pose(joints)
        return (M @ self.tip)[:3], M[:3, :3]

    def jac3(self, joints, eps: float = 0.2):
        p0, _ = self.fk(joints)
        J = np.zeros((3, 3))
        for i in range(3):
            d = np.array(joints, float)
            d[i] += eps
            J[:, i] = (self.fk(d)[0] - p0) / eps
        return J, p0

    def ik_tip(self, joints, p_target, iters: int = 2, damp: float = 4e-4, max_step_deg: float = 20.0):
        q = np.array(joints, float)
        for _ in range(iters):
            J, p = self.jac3(q)
            dq = J.T @ np.linalg.solve(J @ J.T + damp ** 2 * np.eye(3), np.asarray(p_target, float) - p)
            q[:3] += np.clip(dq, -max_step_deg, max_step_deg)
        return q


class TopCamera:
    """Top-camera pixel -> table point (robot base frame) from the twin's fitted pose (calib/cameras_fit.json)."""

    def __init__(self, path: str | Path = VENDOR / "calib/cameras_fit.json"):
        c = json.loads(Path(path).read_text())["top"]
        self.R, self.C = np.array(c["world_to_cv"], float), np.array(c["position"], float)
        self.fx, self.fy, self.cx, self.cy = c["fx"], c["fy"], c["cx"], c["cy"]

    def pixel_to_table(self, u: float, v: float, plane_z: float = WT.BRICK_CENTRE_Z + 0.0096,
                       centre_z: float = WT.BRICK_CENTRE_Z) -> np.ndarray | None:
        """Ray through (u, v) meets the brick's top face (the top view sees it), returned at the brick centre."""
        d = self.R.T @ np.array([(u - self.cx) / self.fx, (v - self.cy) / self.fy, 1.0])
        if abs(d[2]) < 1e-6:
            return None
        s = (plane_z - self.C[2]) / d[2]
        if s <= 0:
            return None
        P = self.C + s * d
        P[2] = centre_z
        return P

    def project(self, X) -> np.ndarray:
        pc = self.R @ (np.asarray(X, float) - self.C)
        return np.array([self.fx * pc[0] / pc[2] + self.cx, self.fy * pc[1] / pc[2] + self.cy])


# ----------------------------------------------------------------------------- wrist target
class SeededTracker(WT.TargetTracker):
    """TargetTracker whose acquisition is seeded from the top-camera target (DEPLOY.md's `target_xy` API change):
    when not tracking, take the accepted wrist box whose table point is nearest `target_xy` (within `acquire_top`),
    not the one nearest the fingertip. Tracking afterwards is the tracker's own gating. `wrong` = the locked box is
    more than `wrong_m` from the top target (logged; the optional close-veto would use it)."""

    def __init__(self, *a, acquire_top: float = 0.08, wrong_m: float = 0.04, **kw):
        super().__init__(*a, **kw)
        self.acquire_top, self.wrong_m = acquire_top, wrong_m

    def update(self, boxes, joints, target_xy=None):
        if target_xy is None or (self.est is not None and self.misses < self.lose_after):
            t = super().update(boxes, joints)
        else:
            c = self.candidates(boxes, joints)
            pick = None
            if c:
                d = [float(np.linalg.norm(x[3][:2] - np.asarray(target_xy)[:2])) - 0.01 * x[1] for x in c]
                k = int(np.argmin(d))
                pick = c[k] if d[k] <= self.acquire_top else None
            if pick is None:
                self.misses += 1
                if self.misses >= self.lose_after:
                    self.est = None
                t = WT.Target(False, np.zeros(3), np.zeros(3) if self.est is None else self.est.copy(), 0.0, None,
                              self.misses, self.est is not None)
            else:
                self.est, self.misses = pick[3].copy(), 0
                t = WT.Target(True, pick[2], pick[3], pick[1], tuple(pick[0].tolist()), 0, True)
        t.wrong = bool(target_xy is not None and t.tracking and self.est is not None
                       and np.linalg.norm(self.est[:2] - np.asarray(target_xy)[:2]) > self.wrong_m)
        return t


class WristDetector:
    """OVBrickDetector on the newest wrist frame in its own thread, while `active` is set; tracker state published as
    `latest` (dict). Joints for the back-projection: the controller's newest measured reading at frame time."""

    def __init__(self, camera, detector, tracker: SeededTracker, joints_fn, max_hz: float = 30.0):
        self.camera, self.detector, self.tracker, self.joints_fn = camera, detector, tracker, joints_fn
        self.period = 1.0 / max_hz
        self.active = threading.Event()
        self.lock = threading.Lock()
        self.latest: dict | None = None
        self.target_xy = None
        self._reset = threading.Event()
        self._stop = threading.Event()
        self._last_frame = None
        self.n_frames, self.det_ms = 0, []
        self.thread = threading.Thread(target=self._run, daemon=True, name="wrist-detector")

    def start(self):
        self.thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def reset(self, target_xy=None) -> None:
        """New pick: forget the track, seed from this top-camera target (env xy, or None)."""
        with self.lock:
            self.target_xy, self.latest = (None if target_xy is None else np.asarray(target_xy, float)), None
        self._reset.set()

    def process(self, frame, joints) -> dict:
        """One frame -> tracker state (also used by the offline replay)."""
        if self._reset.is_set():
            self._reset.clear()
            self.tracker.reset()
        t0 = time.perf_counter()
        dets = self.detector.detect(frame)
        det_ms = (time.perf_counter() - t0) * 1e3
        boxes = [(*d.box, d.confidence) for d in dets]
        with self.lock:
            txy = self.target_xy
        tgt = self.tracker.update(boxes, joints, txy) if joints is not None else None
        rec = {"t": time.monotonic(), "det_ms": det_ms, "track_ms": (time.perf_counter() - t0) * 1e3 - det_ms,
               "n_boxes": len(boxes), "boxes": boxes, "target": tgt, "joints": None if joints is None else
               np.asarray(joints, float)}
        with self.lock:
            self.latest = rec
        self.n_frames += 1
        self.det_ms = (self.det_ms + [det_ms])[-60:]
        return rec

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self.active.wait(0.05):
                continue
            t0 = time.monotonic()
            frame = getattr(self.camera, "frame", None)
            if frame is None or frame is self._last_frame:
                time.sleep(0.004)
                continue
            self._last_frame = frame
            try:
                self.process(frame, self.joints_fn())
            except Exception:  # noqa: BLE001  the residual then sees no detection; never kill the thread
                log.exception("wrist detector failed")
                time.sleep(0.1)
            time.sleep(max(0.0, self.period - (time.monotonic() - t0)))


def warm_fin_cache(cam) -> None:
    """The tracker's fin outlines for every gripper reading (~9 ms of pure Python each): computed once here, never
    mid-pick, where they would hold the GIL against the 30 Hz servo loop."""
    t0 = time.perf_counter()
    cam._fin_poly_local("fixed", 0)
    for gb in range(101):
        cam._fin_poly_local("jaw", gb)
    log.info("residual: fin outlines cached in %.0f ms", (time.perf_counter() - t0) * 1e3)


def make_wrist_detector(path: str = DEFAULT_DETECTOR, device: str = "NPU", conf: float = 0.3):
    """brick-detector-v1 with plain OpenVINO (NPU, then CPU: the GPU runs the policy). Never ultralytics here: it
    holds the GIL long enough to starve the 30 Hz servo reads."""
    from dlsj.inspect.brick_detector import OVBrickDetector

    p = Path(path).expanduser()
    err = None
    for dev in dict.fromkeys([device.upper(), "CPU"]):
        try:
            det = OVBrickDetector(p, dev, conf)
            log.info("wrist detector: %s on %s, conf %.2f", p, dev, conf)
            return det
        except Exception as e:  # noqa: BLE001
            err = e
            log.warning("wrist detector on %s failed: %s", dev, e)
    raise SystemExit(f"no wrist detector at {p}: {err}")


# ----------------------------------------------------------------------------- the residual state machine
@dataclass
class Det:
    """What the residual sees of the wrist target this tick."""

    valid: bool = False
    rel: np.ndarray = field(default_factory=lambda: np.zeros(3))


class GraspResidual:
    """Gate / fade / decision cadence / rate limit / IK application of the residual (no threads, no hardware)."""

    def __init__(self, policy, kin: Kin | None = None, max_joint_delta: float = 20.0):
        self.policy, self.meta = policy, policy.meta
        validate_meta(self.meta)
        self.kin = kin or Kin(joint_offsets(self.meta), self.meta["tip_local"])
        self.scale = np.asarray(self.meta["residual_scale"], float)
        self.rate = np.array([1e9 if v is None else float(v) for v in self.meta["rate_per_tick"]])
        self.decision_ticks = int(self.meta.get("decision_ticks", 5))
        self.fade_ticks = max(1, int(self.meta.get("fade_ticks", 6)))
        self.top_target = bool(self.meta.get("top_target"))
        self.plan_reps = (self.meta["s_dim"] - self.meta["obs_dim"]) // (PLAN_STEPS * 6)
        self.max_joint_delta = float(max_joint_delta)
        self.reset()

    def reset(self, top_xyz=None) -> None:
        self.o, self.o_tgt, self.prev_r, self.o_hand = (np.zeros(6) for _ in range(4))
        self.pick, self.t_pick, self.k = False, 0, 0
        self.top_xyz = None if top_xyz is None else np.asarray(top_xyz, float)
        self.last_state = self.last_plan = self.last_r = None

    @property
    def idle(self) -> bool:
        return not self.pick and not np.any(np.abs(self.o) > 1e-12)

    def obs(self, measured, cmd_prev, det: Det) -> np.ndarray:
        m = np.asarray(measured, float)
        load = (m[5] - (m[5] if cmd_prev is None else float(cmd_prev[5]))) / 10.0
        rel = np.asarray(det.rel, float) if det.valid else np.zeros(3)
        parts = [m / 100.0, rel * 10.0, [float(det.valid)], [load], [self.t_pick / TIME_NORM], self.o / self.scale,
                 self.prev_r, [float(self.pick)]]
        if self.top_target:
            if self.top_xyz is not None:
                p, R = self.kin.fk(m)
                parts += [R.T @ (self.top_xyz - p) * 10.0, [1.0]]
            else:
                parts += [np.zeros(3), [0.0]]
        return np.nan_to_num(np.concatenate([np.asarray(x, float).ravel() for x in parts]), nan=0.0, posinf=0.0,
                             neginf=0.0)

    @staticmethod
    def plan_feat(plan, measured) -> np.ndarray:
        c = [np.asarray(x, float) for x in list(plan)[:PLAN_STEPS]]
        if not c:
            c = [np.asarray(measured, float)]
        while len(c) < PLAN_STEPS:
            c.append(c[-1])
        return ((np.stack(c) - np.asarray(measured, float)) / PLAN_NORM).ravel()

    def step(self, goal, measured, cmd_prev, plan, det: Det, gate: bool) -> tuple[np.ndarray, dict]:
        """One control tick. goal: this tick's planned target (6,); plan: the planned goals from this tick on.
        Returns (goal', info). goal' is `goal` itself (same object) whenever the applied offset is zero."""
        info: dict = {"gate": bool(gate)}
        if gate and not self.pick:
            self.pick = True
        elif not gate and self.pick:  # hand-off: fade the applied offset out
            self.pick, self.o_hand = False, self.o.copy()
            info["handoff"] = True
        if self.pick:
            if self.k % self.decision_ticks == 0:
                t0 = time.perf_counter()
                state = self.obs(measured, cmd_prev, det)
                pf = self.plan_feat(plan, measured)
                s_full = np.concatenate([state] + [pf] * self.plan_reps).astype(np.float32)
                r = np.clip(np.nan_to_num(np.asarray(self.policy(s_full, pf.astype(np.float32)), float)), -1.0, 1.0)
                self.o_tgt, self.prev_r = r * self.scale, r
                self.last_state, self.last_plan, self.last_r = state, pf, r
                info.update(decision=True, state=state, plan=pf, r=r, policy_ms=round((time.perf_counter() - t0) * 1e3, 3))
            self.k += 1
            self.t_pick += 1
            tgt, lim = self.o_tgt, self.rate
        else:
            tgt, lim = np.zeros(6), np.maximum(self.rate, np.abs(self.o_hand) / self.fade_ticks)
        self.o = self.o + np.clip(tgt - self.o, -lim, lim)
        self.o[np.abs(self.o) < 1e-12] = 0.0
        info["o"] = self.o.copy()
        if not np.any(self.o):
            return goal, info
        g = np.asarray(goal, float)
        g2 = g.copy()
        g2[3:5] += self.o[3:5]
        if np.abs(self.o[:5]).sum() > 1e-7:
            p, R = self.kin.fk(g)
            g2 = self.kin.ik_tip(g2, p + R @ self.o[:3])
        g2[5] = float(np.clip(g[5] + self.o[5], 0.0, 100.0))
        d = np.abs(g2 - g)
        if not np.isfinite(g2).all() or d[:5].max() > self.max_joint_delta:
            info["rejected"] = np.round(g2, 2).tolist()  # IK blew up: keep the plan for this tick
            return goal, info
        return g2, info


# ----------------------------------------------------------------------------- controller hook
@dataclass
class ResidualConfig:
    mode: str = "off"
    weights: str = "zero"
    device: str = "numpy"
    detector: str = DEFAULT_DETECTOR
    detector_device: str = "NPU"
    detector_conf: float = 0.3
    max_age_s: float = 0.15  # an older wrist result counts as "not detected"
    max_joint_delta: float = 20.0
    seed_top: bool = False  # --residual-top-seed: the top geometry is not verified yet (see residual_replay.py)


def config_from_args(ns) -> ResidualConfig:
    """From smolvla_control's --residual* flags (defined there so --residual off never imports this module)."""
    return ResidualConfig(mode=ns.residual, weights=ns.residual_weights, device=ns.residual_device,
                          detector=ns.residual_detector, detector_device=ns.residual_detector_device,
                          max_joint_delta=ns.residual_max_joint_delta, seed_top=ns.residual_top_seed)


def describe(cfg: ResidualConfig) -> str:
    pol = load_policy(cfg.weights, cfg.device)
    m = pol.meta
    return (f"residual {cfg.mode}: {cfg.weights} (source {m.get('source')}), s_dim {m['s_dim']}, top_target "
            f"{bool(m.get('top_target'))}, scale {m['residual_scale']}, rate {m['rate_per_tick']}, tip {m['tip_local']}, "
            f"decision {m.get('decision_ticks', 5)} / fade {m.get('fade_ticks', 6)} ticks")


def make_residual_controller_class(base, cfg: ResidualConfig, detector=None):
    """Outermost mixin over the final controller class (smolvla / auto / loop): corrects the goal inside `_step`
    only while phase == "running" (the policy is driving); readying, homing, inspection and scripted moves are not
    touched. `detector` overrides make_wrist_detector (tests)."""
    if cfg.mode not in ("shadow", "on"):
        raise ValueError(f"residual mode {cfg.mode!r}: the hook is only built for shadow / on")

    class ResidualController(base):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            pol = load_policy(cfg.weights, cfg.device)
            self.res = GraspResidual(pol, max_joint_delta=cfg.max_joint_delta)
            self.res_cam = WT.WristCamera(offsets_rad=self.res.kin.off.tolist(), tip_local=self.res.meta["tip_local"])
            warm_fin_cache(self.res_cam)
            self.res_top_cam = TopCamera()
            self.res_live = cfg.mode  # POST /ctl/residual_off -> "off" (fades out), residual_on -> cfg.mode
            self.res_on_prev = False
            self.res_seed: dict | None = None
            self.res_stats = {"ticks": 0, "decisions": 0, "rejected": 0, "max_ms": 0.0}
            det = detector if detector is not None else make_wrist_detector(cfg.detector, cfg.detector_device,
                                                                            cfg.detector_conf)
            self.res_wrist = WristDetector(self.cameras["wrist"], det, SeededTracker(self.res_cam),
                                           lambda: self.measured).start()
            log.info(describe(cfg))
            if getattr(self, "trace", None) is None:
                log.warning("residual: no trace file (--trace ''): the residual steps are not logged")

        # -------------------------------------------------------------- bookkeeping
        def _res_log(self, kind: str, **fields) -> None:
            if hasattr(self, "_log"):  # the console's JSON-lines trace (no-op without --trace)
                self._log(kind, **fields)

        def _res_top_target(self):
            """The loop's chosen brick (top-camera pixels) -> table point, or None (manual console, no target)."""
            L = getattr(self, "L", None)
            tgt = L.get("target") if isinstance(L, dict) else None
            if not cfg.seed_top or not tgt or not tgt.get("centre"):
                return None, None
            u, v = tgt["centre"]
            return self.res_top_cam.pixel_to_table(float(u), float(v)), [float(u), float(v)]

        def _res_new_pick(self) -> None:
            xyz, px = self._res_top_target()
            self.res.reset(xyz)
            self.res_wrist.reset(None if xyz is None else xyz[:2])
            self.res_seed = {"top_px": px, "top_xyz": None if xyz is None else np.round(xyz, 4).tolist()}
            self._res_log("residual_start", mode=self.res_live, **self.res_seed)

        def _res_det(self) -> tuple[Det, dict]:
            rec = self.res_wrist.latest
            if rec is None:
                return Det(), {"det": "none"}
            age = time.monotonic() - rec["t"]
            t = rec["target"]
            ok = t is not None and bool(t.valid) and age <= cfg.max_age_s
            info = {"det_age_ms": round(age * 1e3, 1), "det_ms": round(rec["det_ms"], 2), "n_boxes": rec["n_boxes"]}
            if t is not None:
                info.update(det_valid=bool(t.valid), det_rel=np.asarray(t.rel, float), det_conf=round(t.conf, 3),
                            det_box=t.box, det_misses=t.misses, tracking=t.tracking, wrong=getattr(t, "wrong", False),
                            det_pos=np.asarray(t.pos, float))
            return Det(ok, np.asarray(t.rel, float) if ok else np.zeros(3)), info

        # -------------------------------------------------------------- hook
        def _step(self, target):
            if self.phase != "running":
                return super()._step(target)
            try:
                target = self._res_apply(target)
            except Exception:  # noqa: BLE001  the residual must never stop the arm: plan unchanged
                self.res_stats["errors"] = self.res_stats.get("errors", 0) + 1
                if self.res_stats["errors"] <= 5 or self.res_stats["errors"] % 300 == 0:
                    log.exception("residual step failed (%d so far): plan used unchanged", self.res_stats["errors"])
            return super()._step(target)

        def _res_apply(self, target):
            t0 = time.perf_counter()
            gate = self.stage == "pick" and self.res_live != "off" and is_pick_prompt(self.prompt())
            if gate and not self.res_on_prev:
                self._res_new_pick()
            self.res_on_prev = gate
            if not gate and self.res.idle:
                return target
            measured = self.measured if self.measured is not None else np.asarray(self.arm.read(), float)
            plan = [p for p in (self._planned(self.t + j) for j in range(PLAN_STEPS)) if p is not None] or [target]
            det, dinfo = self._res_det()
            g2, info = self.res.step(np.asarray(target, float), measured, self.cmd, plan, det, gate)
            applied = self.res_live == "on" or (self.res_live == "off" and cfg.mode == "on")
            out = g2 if applied else target
            ms = (time.perf_counter() - t0) * 1e3
            st = self.res_stats
            st["ticks"] += 1
            st["decisions"] += int(bool(info.get("decision")))
            st["rejected"] += int("rejected" in info)
            st["max_ms"] = max(st["max_ms"], ms)
            info.pop("gate", None)
            self._res_log("residual", mode=self.res_live, applied=applied, gate=gate, k=self.res.k,
                      goal=np.asarray(target, float), goal_res=np.asarray(g2, float), o_tgt=self.res.o_tgt,
                      ms=round(ms, 3), **info, **dinfo)
            return out

        def tick(self) -> None:
            super().tick()
            live = self.phase == "running" and self.stage == "pick" and self.res_live != "off"
            if live:
                self.res_wrist.active.set()
            else:
                self.res_wrist.active.clear()
            if self.phase != "running" and not self.res.idle:  # stopped / held / handed to a scripted move
                self.res.reset()
            if self.phase != "running":
                self.res_on_prev = False

        def _handle(self, c: str) -> None:
            if c in ("residual_off", "residual_on"):
                self.res_live = "off" if c == "residual_off" else cfg.mode
                self.message = f"Residual: {self.res_live}"
                self._res_log("key", key=c)
                return
            super()._handle(c)

        def snapshot(self) -> dict:
            d = super().snapshot()
            with self.lock:
                rec = self.res_wrist.latest
                t = rec["target"] if rec else None
                d["residual"] = {"mode": self.res_live, "launch_mode": cfg.mode, "weights": cfg.weights,
                                 "pick": self.res.pick, "o": np.round(self.res.o, 4).tolist(),
                                 "r": None if self.res.last_r is None else np.round(self.res.last_r, 3).tolist(),
                                 "det_valid": bool(t.valid) if t is not None else None,
                                 "det_ms_median": round(float(np.median(self.res_wrist.det_ms)), 2)
                                 if self.res_wrist.det_ms else None, "seed": self.res_seed, **self.res_stats}
            return d

        def stop(self) -> None:
            self.res_wrist.stop()
            if hasattr(super(), "stop"):
                super().stop()

    return ResidualController
