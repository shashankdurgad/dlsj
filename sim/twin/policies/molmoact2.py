"""MolmoAct2 policy runner for the Isaac twin: a light client + a worker process that owns the model.

    from sim.twin.policies.molmoact2 import MolmoAct2Client
    pol = MolmoAct2Client()                       # spawns the worker, waits until the model is loaded
    chunk = pol.act({"images": {"top": top, "wrist": wrist}, "state": state, "task": "..."})   # (30, 6)
    pol.last_latency_s; pol.close()

The client imports only the standard library + numpy, so it is safe inside the Isaac process; the worker
(`python molmoact2.py --worker`, same file) imports torch + Physical AI Studio. They talk over the worker's
stdin/stdout (length-prefixed pickles). The worker exits when its stdin closes (client `close()`, `atexit`, or the
parent dying for any reason) and also has PR_SET_PDEATHSIG=SIGKILL, so it cannot outlive the client.

Model = exactly what the real robot runs (deploy/smolvla/run_molmo.sh -> deploy/molmoact2/export.py `build`):
  * `allenai/MolmoAct2-SO100_101` zero-shot through Studio's own MolmoAct2 policy (the same pre/post processors that
    are baked into the OpenVINO export), norm tag `so100_so101_molmoact2`, `adapt_to_so101=True` with the unit-scale
    calibration (our joints are already LeRobot degrees -> Studio scale exactly 1);
  * cameras `wrist,top` in that order, 480x640 each, uncropped (export.py `--cams wrist,top --hw 480,640`);
  * chunk 30 actions (norm tag action_horizon 30, export n_action_steps=30), 10 flow-matching steps (export default),
    static text length 256 (export default), flow started from Gaussian noise (`use_random_input_noise=True` in
    export.py `build`; the torch worker seeds it, `noise="zeros"` gives the deterministic start);
  * prompt "pick up the brick and lift it up" for the pick stage, "put the brick in the {colour} bowl" for place
    (run_molmo.sh `--pick-prompt/--place-prompt`, no bring stage);
  * controller side (not in this client, for the env to mirror): RTC horizon 15 (re-plan when 15 + delay planned
    actions remain), guidance 3, `--max-step 4` deg per 30 Hz tick, ready pose 18.3,-67.0,55.8,60.3,17.5,28.9.
    `act(..., prev_chunk=, inference_delay=)` runs the same RTC-guided flow loop as the deployed export
    (deploy/molmoact2/rtc_patch.py, horizon 15, guidance 3); without `prev_chunk` it is a plain chunk.

Units (runtime = the team's LeRobot units in the recorded data's calibration frame, cal2): 5 body joints in degrees
+ gripper 0-100. The checkpoint was trained on the legacy LeRobot SO-100/101 frame; Studio's transform
(physicalai/policies/molmoact2/so101.py, constants.py) is

    ckpt    = S * runtime + O          S = (1, -1, 1, 1, 1, 1),  O = (0, 90, 90, 0, 0, 0) deg, scale 1
    runtime = S * (ckpt - O)           gripper 0-100 in both frames

then q01/q99 quantile normalisation with the checkpoint's `norm_stats.json`. The worker lets Studio apply the
transform (as on the robot); `to_ckpt` / `from_ckpt` here are the same maths, used for diagnostics and the tests.

The cal2 fine-tune (Studio Lightning checkpoint, LoRA; the deployed 8k = sim/twin/data/models/molmoact2_ft_8k/
step008000.ckpt) loads with `snap="/path/step008000.ckpt"`. It was trained with the top camera cropped to x 0-560
(560x480): pass `crops={"top": (0, 0, 560, 480)}`. It knows ONLY the prompts of finetune_cal2_456.json `tasks`
("pick up the {blue|brown|red|white|yellow} brick", "bring the brick to the inspection camera", "place the brick in
the {colour} bowl"; sim/twin/rl/priors.py FT_*), not the zero-shot prompts in DEPLOY below. Its deploy loop
(run_loop.sh) uses ready pose 26.3,-51.8,30.1,73.1,15.9,37.7, --max-step 4 (gripper 8), RTC horizon 15, guidance 3,
and the OpenVINO export's FIXED flow noise (a fixed-seed op: deterministic per input; here `seed=` per call, the RL
image task passes the same seed on every request). sim/twin/rl/image_task.py requires finetune=True with a .ckpt.
"""

from __future__ import annotations

import atexit
import os
import pickle
import signal
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEFAULT_SNAP = "allenai/MolmoAct2-SO100_101"
NORM_TAG = "so100_so101_molmoact2"
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
S = np.array([1, -1, 1, 1, 1, 1], np.float32)
O = np.array([0, 90, 90, 0, 0, 0], np.float32)  # noqa: E741 (Studio name, used by tests)
CHUNK = 30
# The ZERO-SHOT deployment (deploy/smolvla/run_molmo.sh + deploy/molmoact2/export.py), see the module docstring;
# the cal2 fine-tune's prompts / ready pose / crop are in the docstring and sim/twin/rl/priors.py FT_*.
DEPLOY = {
    "cams": ("wrist", "top"),
    "pick_prompt": "pick up the brick and lift it up",
    "place_prompt": "put the brick in the {colour} bowl",
    "steps": 10,
    "chunk": CHUNK,
    "noise": "random",
    "rtc_horizon": 15,
    "rtc_guidance": 3.0,
    "max_step_deg": 4.0,
    "fps": 30,
    "ready_pose": (18.3, -67.0, 55.8, 60.3, 17.5, 28.9),
}
CAM_ORDER = ("wrist", "top", "inspect")  # deployed order is wrist, top


def to_ckpt(runtime) -> np.ndarray:
    """Team runtime units (cal2 LeRobot degrees, gripper 0-100) -> checkpoint legacy frame. Works on (..., 6)."""
    return np.asarray(runtime, np.float32) * S + O


def from_ckpt(ckpt) -> np.ndarray:
    """Checkpoint legacy frame -> team runtime units. Inverse of `to_ckpt`."""
    return S * (np.asarray(ckpt, np.float32) - O)


ROLL = JOINTS.index("wrist_roll")


def _wrap_roll(x: np.ndarray) -> np.ndarray:
    x = np.array(x, np.float32, copy=True)
    x[..., ROLL] = (x[..., ROLL] + 180.0) % 360.0 - 180.0
    return x


def apply_frame_offset(runtime, delta) -> np.ndarray:
    """Runtime (cal2) units -> the frame the model is fed: runtime + delta (wrist_roll wrapped to [-180, 180))."""
    return _wrap_roll(np.asarray(runtime, np.float32) + np.asarray(delta, np.float32))


def remove_frame_offset(model_frame, delta) -> np.ndarray:
    """Inverse of `apply_frame_offset` (actions the model predicts -> runtime units)."""
    return _wrap_roll(np.asarray(model_frame, np.float32) - np.asarray(delta, np.float32))


# Reference joint frames, as runtime(cal2) -> other-frame offsets in degrees (pan, lift, elbow, wflex, roll, gripper).
FRAME_OFFSETS = {
    "none": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    # old follower calibration: degrees_old = degrees_cal2 + d (homing/range difference of the two calibration files)
    "old_cal": (-20.35, 7.95, -0.18, -0.04, 179.0, 0.0),
    # URDF frame from Joe's cal2 fit: q_urdf_deg = degrees_cal2 + d
    "urdf": (0.0, -3.0, -1.3, -4.2, -103.4, 0.0),
}


# ----------------------------------------------------------------------------- IPC
def _send(f, obj) -> None:
    b = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    f.write(struct.pack("<Q", len(b)))
    f.write(b)
    f.flush()


def _recv(f):
    h = f.read(8)
    if len(h) < 8:
        raise EOFError("MolmoAct2 pipe closed")
    (n,) = struct.unpack("<Q", h)
    b = f.read(n)
    if len(b) < n:
        raise EOFError("MolmoAct2 pipe closed mid-message")
    return pickle.loads(b)


def _pdeathsig() -> None:  # child side of Popen: die with the parent (Linux)
    try:
        import ctypes

        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    except Exception:
        pass


# ----------------------------------------------------------------------------- client
class MolmoAct2Client:
    """Owns one MolmoAct2 worker process. `act(obs)` -> (horizon, 6) actions in team runtime units.

    Args:
        snap: HF repo id / local snapshot dir (zero-shot) or a Studio Lightning `.ckpt` (fine-tune).
            Default `allenai/MolmoAct2-SO100_101`, what the robot runs.
        cams: cameras fed to the model. They are always fed in the deployed order wrist, top, inspect
            (so the default ("top", "wrist") == the robot's "wrist,top"); `keep_order=True` uses the given order.
        instruction: overrides obs["task"]; None -> obs["task"], or the deployed pick prompt if obs has none.
        steps: flow-matching steps (10 deployed). horizon: actions returned (<= 30; the robot plans 30 and RTC
            re-plans after 15). noise: "random" (deployed, seeded with `seed`) or "zeros" (deterministic).
        crops: {cam: (x, y, w, h)} applied before the model (fine-tune: {"top": (0, 0, 560, 480)}).
        rtc_horizon / rtc_guidance: RTC settings used when act() gets `prev_chunk` (deployed 15 / 3).
        frame_offset_deg: per-joint delta (6,) or a FRAME_OFFSETS name. The model sees state + delta (and an RTC tail
            + delta); its actions are mapped back with - delta (wrist_roll wrapped). Default none = the robot.
        python: interpreter for the worker (default: this one; env MOLMOACT2_PYTHON overrides).
        hf_home: HF cache for the worker (default env HF_HOME, else sim/twin/data/hf).
    """

    def __init__(self, snap: str | None = None, cams=("top", "wrist"), instruction: str | None = None,
                 steps: int = DEPLOY["steps"], horizon: int = CHUNK, device: str = "cuda", *,
                 noise: str = DEPLOY["noise"], seed: int = 0, dtype: str = "bfloat16", keep_order: bool = False,
                 crops: dict | None = None, rtc_horizon: int = DEPLOY["rtc_horizon"],
                 rtc_guidance: float = DEPLOY["rtc_guidance"], text_len: int | None = None,
                 frame_offset_deg=None,
                 python: str | None = None, hf_home: str | None = None, offline: bool | None = None,
                 load_timeout_s: float = 1800.0, verbose: bool = True):
        if not 1 <= horizon <= CHUNK:
            raise ValueError(f"horizon must be in 1..{CHUNK}")
        if noise not in ("random", "zeros"):
            raise ValueError("noise must be 'random' or 'zeros'")
        self.cams = self._order(cams, keep_order)
        self.keep_order = keep_order
        self.instruction, self.horizon, self.crops = instruction, int(horizon), dict(crops or {})
        self.frame_offset_deg = self._delta(frame_offset_deg)
        self.last_latency_s: float | None = None
        self.last_info: dict = {}
        self._closed = False
        self._lock = threading.Lock()
        env = dict(os.environ)
        env["HF_HOME"] = hf_home or env.get("HF_HOME") or str(REPO / "sim/twin/data/hf")
        if offline or (offline is None and _cached(env["HF_HOME"], snap or DEFAULT_SNAP)):
            env["HF_HUB_OFFLINE"] = "1"
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO), env.get("PYTHONPATH", "")) if p)
        cfg = {"snap": snap or DEFAULT_SNAP, "cams": list(self.cams), "steps": int(steps), "device": device,
               "noise": noise, "seed": int(seed), "dtype": dtype, "rtc_horizon": int(rtc_horizon),
               "rtc_guidance": float(rtc_guidance), "text_len": text_len}
        py = python or os.environ.get("MOLMOACT2_PYTHON") or sys.executable
        self._proc = subprocess.Popen([py, "-u", str(Path(__file__).resolve()), "--worker"], stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, stderr=None if verbose else subprocess.DEVNULL,
                                      env=env, preexec_fn=_pdeathsig if sys.platform.startswith("linux") else None)
        atexit.register(self.close)
        _send(self._proc.stdin, cfg)
        t = time.monotonic()
        msg = self._reply(timeout=load_timeout_s)
        self.info = msg | {"spawn_to_ready_s": round(time.monotonic() - t, 1)}

    @staticmethod
    def _delta(d) -> np.ndarray:
        if d is None:
            d = "none"
        if isinstance(d, str):
            d = FRAME_OFFSETS[d]
        d = np.asarray(d, np.float32).reshape(-1)
        if d.size == 5:
            d = np.append(d, 0.0).astype(np.float32)
        if d.size != 6:
            raise ValueError("frame offset must have 5 or 6 values")
        return d

    @staticmethod
    def _order(cams, keep_order: bool) -> tuple[str, ...]:
        cams = tuple(cams)
        if keep_order:
            return cams
        return tuple(sorted(cams, key=lambda c: CAM_ORDER.index(c) if c in CAM_ORDER else len(CAM_ORDER)))

    def _reply(self, timeout: float | None = None):
        box: dict = {}

        def rd():
            try:
                box["m"] = _recv(self._proc.stdout)
            except BaseException as e:  # noqa: BLE001
                box["e"] = e

        th = threading.Thread(target=rd, daemon=True)
        th.start()
        th.join(timeout)
        if th.is_alive():
            self.close()
            raise TimeoutError(f"MolmoAct2 worker did not answer within {timeout}s")
        if "e" in box:
            rc = self._proc.poll()
            self.close()
            raise RuntimeError(f"MolmoAct2 worker died (exit {rc}); see its stderr") from box["e"]
        m = box["m"]
        if isinstance(m, dict) and m.get("error"):
            raise RuntimeError("MolmoAct2 worker error:\n" + m["error"])
        return m

    def act(self, obs: dict, *, prev_chunk: np.ndarray | None = None, inference_delay: int = 0,
            cams=None, task: str | None = None, noise: str | None = None, steps: int | None = None,
            seed: int | None = None, frame_offset_deg=None, timeout: float = 120.0) -> np.ndarray:
        """obs = {"images": {cam: uint8 (H, W, 3)}, "state": (6,) runtime units, "task": str}.

        prev_chunk: the not-yet-executed tail of the current plan (n, 6) runtime units -> RTC-guided chunk
        (the deployed controller's hand-off); inference_delay = actions that pass during this inference.
        cams / task / noise / steps / frame_offset_deg: per-call overrides (diagnostics, sweeps); seed: re-seed the
        flow's start noise for this call (common random numbers across variants).
        Returns (horizon, 6) float32 runtime units."""
        if self._closed:
            raise RuntimeError("MolmoAct2Client is closed")
        cams = self._order(cams, self.keep_order) if cams is not None else self.cams
        imgs = {}
        for c in cams:
            im = np.asarray(obs["images"][c])
            if im.dtype != np.uint8 or im.ndim != 3 or im.shape[2] != 3:
                raise ValueError(f"image {c!r} must be uint8 (H, W, 3), got {im.dtype} {im.shape}")
            if c in self.crops:
                x, y, w, h = self.crops[c]
                im = im[y:y + h, x:x + w]
            imgs[c] = np.ascontiguousarray(im)
        delta = self.frame_offset_deg if frame_offset_deg is None else self._delta(frame_offset_deg)
        state = apply_frame_offset(np.asarray(obs["state"], np.float32).reshape(6), delta)
        text = task or self.instruction or obs.get("task") or DEPLOY["pick_prompt"]
        req = {"images": imgs, "cams": list(cams), "state": state, "task": str(text), "noise": noise, "steps": steps,
               "seed": seed}
        if prev_chunk is not None and len(prev_chunk):
            req["prev_chunk"] = apply_frame_offset(np.asarray(prev_chunk, np.float32).reshape(-1, 6)[:CHUNK], delta)
            req["inference_delay"] = int(inference_delay)
        t = time.perf_counter()
        with self._lock:
            _send(self._proc.stdin, req)
            r = self._reply(timeout)
        self.last_latency_s = float(r["latency_s"])  # model time in the worker (pre + flow + post)
        self.last_info = {k: v for k, v in r.items() if k != "actions"} | {"roundtrip_s": time.perf_counter() - t}
        return remove_frame_offset(np.asarray(r["actions"], np.float32), delta)[: self.horizon]

    def act_k(self, obs: dict, k: int, seeds=None, *, prev_chunk: np.ndarray | None = None, inference_delay: int = 0,
              task: str | None = None, steps: int | None = None, latent: bool = True,
              timeout: float = 120.0) -> tuple[np.ndarray, np.ndarray | None]:
        """K candidate chunks from ONE vision-language prefix forward: the prefix / KV context is computed once and
        the action expert's flow loop runs batched over k Gaussian start noises (one per seed; seeds default
        0..k-1). Same RTC guidance as act() when prev_chunk is given (shared by all k). Returns
        (actions (k, horizon, 6) runtime units, pooled prefix latent (D,) float32 or None). With k = 1 and the same
        seed the chunk equals act(obs, seed=seed)'s up to the noise generator (torch.Generator per seed here)."""
        if self._closed:
            raise RuntimeError("MolmoAct2Client is closed")
        cams = self.cams
        imgs = {}
        for c in cams:
            im = np.asarray(obs["images"][c])
            if c in self.crops:
                x, y, w, h = self.crops[c]
                im = im[y:y + h, x:x + w]
            imgs[c] = np.ascontiguousarray(im)
        delta = self.frame_offset_deg
        state = apply_frame_offset(np.asarray(obs["state"], np.float32).reshape(6), delta)
        text = task or self.instruction or obs.get("task") or DEPLOY["pick_prompt"]
        seeds = list(range(k)) if seeds is None else [int(v) for v in seeds]
        if len(seeds) != k:
            raise ValueError("need one seed per candidate")
        req = {"op": "act_k", "k": int(k), "seeds": seeds, "images": imgs, "cams": list(cams), "state": state,
               "task": str(text), "steps": steps, "latent": bool(latent)}
        if prev_chunk is not None and len(prev_chunk):
            req["prev_chunk"] = apply_frame_offset(np.asarray(prev_chunk, np.float32).reshape(-1, 6)[:CHUNK], delta)
            req["inference_delay"] = int(inference_delay)
        t = time.perf_counter()
        with self._lock:
            _send(self._proc.stdin, req)
            r = self._reply(timeout)
        self.last_latency_s = float(r["latency_s"])
        self.last_info = {kk: v for kk, v in r.items() if kk not in ("actions", "latent")} | {
            "roundtrip_s": time.perf_counter() - t}
        a = remove_frame_offset(np.asarray(r["actions"], np.float32), delta)[:, : self.horizon]
        return a, (np.asarray(r["latent"], np.float32) if r.get("latent") is not None else None)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        p = getattr(self, "_proc", None)
        if p is None:
            return
        try:
            p.stdin.close()
        except Exception:
            pass
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait(timeout=10)
        try:
            atexit.unregister(self.close)
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _cached(hf_home: str, snap: str) -> bool:
    if Path(snap).exists():
        return True
    d = Path(hf_home) / "hub" / ("models--" + snap.replace("/", "--")) / "snapshots"
    return d.is_dir() and any(d.iterdir())


# ----------------------------------------------------------------------------- worker
def _worker() -> None:
    inp, out = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr  # library prints must not corrupt the pipe
    import traceback

    try:
        cfg = _recv(inp)
        w = _Worker(cfg)
        _send(out, w.ready)
    except BaseException:  # noqa: BLE001
        try:
            _send(out, {"error": traceback.format_exc()})
        finally:
            sys.exit(1)
    while True:
        try:
            req = _recv(inp)
        except EOFError:
            break
        try:
            _send(out, w.act(req))
        except BrokenPipeError:
            break
        except BaseException:  # noqa: BLE001
            _send(out, {"error": traceback.format_exc()})
    sys.exit(0)


class _Worker:
    def __init__(self, cfg: dict):
        import torch

        t0 = time.perf_counter()
        sys.path.insert(0, str(REPO / "deploy/molmoact2"))
        from export import build  # the deployed construction (unit-scale calibration, set_features, steps)

        self.torch = torch
        torch.manual_seed(cfg["seed"])
        snap = cfg["snap"]
        ckpt = snap if str(snap).endswith(".ckpt") else None
        policy = build(DEFAULT_SNAP if ckpt else snap, False, cfg["cams"], 480, 640, ckpt=ckpt,
                       steps=cfg["steps"], text_len=cfg.get("text_len"))
        policy.model._use_random_input_noise = cfg["noise"] == "random"
        self.dev = torch.device(cfg["device"])
        policy = policy.to(device=self.dev, dtype=getattr(torch, cfg["dtype"])).eval()
        torch.set_grad_enabled(False)
        self.policy = policy
        self.pre, self.post = policy._preprocessor, policy._postprocessor
        from rtc_patch import patch_policy

        patch_policy(policy)  # adds the RTC branch; without RTC keys it calls the original loop
        self.rtc_h, self.rtc_g = cfg["rtc_horizon"], cfg["rtc_guidance"]
        self.noise, self.steps = cfg["noise"], int(cfg["steps"])
        self.n_calls = 0
        c = policy.config
        self.ready = {"model": str(snap), "load_s": round(time.perf_counter() - t0, 1),
                      "device": torch.cuda.get_device_name(0) if self.dev.type == "cuda" else str(self.dev),
                      "dtype": cfg["dtype"], "cams": cfg["cams"], "noise": cfg["noise"],
                      "steps": int(policy.model.backbone.model.flow_matching_num_steps),
                      "chunk": int(policy.model._chunk_size), "norm_tag": getattr(c, "norm_tag", None),
                      "adapt_to_so101": getattr(c, "adapt_to_so101", None),
                      "text_len": getattr(c, "tokenizer_max_length", None), "pid": os.getpid()}
        print("MOLMOACT2_READY", self.ready, file=sys.stderr, flush=True)

    def act(self, req: dict) -> dict:
        if req.get("op") == "act_k":
            return self.act_k(req)
        torch = self.torch
        cams = req["cams"]
        self.pre._extractor.image_keys = list(cams)  # model camera order for this call
        noise, steps = req.get("noise") or self.noise, int(req.get("steps") or self.steps)
        self.policy.model._use_random_input_noise = noise == "random"
        self.policy.model.backbone.model.flow_matching_num_steps = steps
        if req.get("seed") is not None:
            torch.manual_seed(int(req["seed"]))
        batch = {"state": torch.as_tensor(req["state"])[None].to(self.dev), "task": [req["task"]],
                 "images": {c: torch.from_numpy(req["images"][c]).permute(2, 0, 1)[None].to(self.dev) for c in cams}}
        if self.dev.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        x = self.pre(batch)
        rtc = "prev_chunk" in req
        if rtc:  # same inputs the deployed controller builds (deploy/pi05/rtc_control.py rtc_inputs)
            prev = np.zeros((1, CHUNK, 6), np.float32)
            tail = req["prev_chunk"][:CHUNK]
            prev[0, :len(tail)] = tail
            x = dict(x)
            x["prev_chunk_left_over"] = torch.from_numpy(prev).to(self.dev)
            x["inference_delay"] = torch.tensor(int(min(max(req["inference_delay"], 0), self.rtc_h)))
            x["execution_horizon"] = torch.tensor(int(self.rtc_h))
            x["max_guidance_weight"] = torch.tensor(float(self.rtc_g))
        a = self.policy.model.predict_action_chunk(x)
        a = self.post({"action": a})["action"]
        a = a.float().cpu().numpy().reshape(-1, 6)
        if self.dev.type == "cuda":
            torch.cuda.synchronize()
        lat = time.perf_counter() - t
        self.n_calls += 1
        return {"actions": a, "latency_s": lat, "call": self.n_calls, "rtc": rtc, "cams": cams, "task": req["task"],
                "noise": noise, "steps": steps}


    def act_k(self, req: dict) -> dict:
        """One prefix forward, k flow-noise samples batched in the action expert (+ the pooled prefix latent)."""
        torch = self.torch
        from physicalai.policies.molmoact2.model import _MODEL_INPUT_KEYS

        cams = req["cams"]
        self.pre._extractor.image_keys = list(cams)
        k, seeds = int(req["k"]), list(req["seeds"])
        steps = int(req.get("steps") or self.steps)
        model = self.policy.model
        bb = model.backbone.model
        ae = bb._require_action_expert()
        batch = {"state": torch.as_tensor(req["state"])[None].to(self.dev), "task": [req["task"]],
                 "images": {c: torch.from_numpy(req["images"][c]).permute(2, 0, 1)[None].to(self.dev) for c in cams}}
        if self.dev.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        x = self.pre(batch)
        inputs = {kk: x[kk] for kk in _MODEL_INPUT_KEYS if kk in x}
        chunk = int(model._chunk_size)
        device, dtype = ae.action_embed.weight.device, ae.action_embed.weight.dtype
        cap = {}
        def grab(_mod, _inp, out):
            cap["h"] = out[0] if isinstance(out, (tuple, list)) else out

        hook = bb.transformer.register_forward_hook(grab)
        try:
            ctx = bb._encode_action_context(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"),
                                            token_type_ids=inputs.get("token_type_ids"), images=inputs.get("images"),
                                            token_pooling=inputs.get("token_pooling"), seq_len=chunk, device=device,
                                            dtype=dtype)
        finally:
            hook.remove()
        if self.dev.type == "cuda":
            torch.cuda.synchronize()
        t_prefix = time.perf_counter() - t
        latent = None
        if req.get("latent", True) and "h" in cap:
            m = bb._encoder_attention_mask(inputs["input_ids"], inputs.get("attention_mask"))
            h = cap["h"].float()
            m = (m.float() if m is not None else torch.ones(h.shape[:2], device=h.device))[..., None]
            latent = ((h * m).sum(1) / m.sum(1).clamp(min=1))[0].cpu().numpy()
        ctx = _expand_batch(ctx, k, torch)
        pad = inputs.get("action_dim_is_pad")
        pad = pad.expand(k, *pad.shape[1:]) if pad is not None else None
        shape = (1, chunk, bb.max_action_dim)
        if self.noise == "random":
            gens = [torch.Generator(device=device).manual_seed(int(sd)) for sd in seeds]
            traj = torch.cat([torch.randn(*shape, device=device, dtype=dtype, generator=g) for g in gens])
            traj = bb._mask_action_dims(traj, pad)
        else:
            traj = torch.zeros(k, *shape[1:], device=device, dtype=dtype)
        rtc = "prev_chunk" in req
        if rtc:
            prev_m, w, gmax = self._rtc_terms(req, chunk, bb, device, dtype)
        dt = 1.0 / steps
        for step in range(steps):
            tt = step / steps
            ts = torch.full((k,), tt, device=device, dtype=dtype)
            v = bb._mask_action_dims(ae.forward_with_context(traj, ts, context=ctx), pad)
            if rtc:
                x1 = traj + (1.0 - tt) * v
                err = (prev_m - x1) * w
                if tt == 0.0:
                    gw = gmax
                else:
                    c = (1.0 - tt) / tt
                    gw = min(c * ((1.0 - tt) ** 2 + tt ** 2) / ((1.0 - tt) ** 2), gmax)
                v = v + gw * err
            traj = bb._mask_action_dims(traj + dt * v, pad)
        a = traj[:, :chunk, : model._action_dim].float()
        a = self.post({"action": a})["action"].float().cpu().numpy().reshape(k, -1, 6)
        if self.dev.type == "cuda":
            torch.cuda.synchronize()
        lat = time.perf_counter() - t
        self.n_calls += 1
        return {"actions": a, "latent": latent, "latency_s": lat, "prefix_s": t_prefix, "k": k, "rtc": rtc,
                "call": self.n_calls, "steps": steps, "latent_dim": None if latent is None else int(latent.shape[0])}

    def _rtc_terms(self, req, chunk, bb, device, dtype):
        """deploy/molmoact2/rtc_patch.py's guidance inputs (previous chunk in model space, prefix weights)."""
        torch = self.torch
        from rtc_patch import OFFSETS, SIGNS, _action_quantiles

        q01, q99 = _action_quantiles(self.policy)
        adim = len(q01)
        prev = np.zeros((1, chunk, adim), np.float32)
        tail = np.asarray(req["prev_chunk"], np.float32)[:chunk, :adim]
        prev[0, :len(tail)] = tail
        ck = prev * np.asarray(SIGNS[:adim], np.float32) + np.asarray(OFFSETS[:adim], np.float32)
        den = np.where(q99 - q01 == 0, 1e-8, q99 - q01)
        n = np.clip(2.0 * (ck - q01) / den - 1.0, -1.0, 1.0)
        n = np.pad(n, ((0, 0), (0, 0), (0, max(0, bb.max_action_dim - adim))))
        h = float(self.rtc_h)
        start = min(float(max(int(req.get("inference_delay", 0)), 0)), h)
        idx = np.arange(chunk, dtype=np.float32)
        w = np.clip((h - idx) / (h - start + 1.0), 0.0, 1.0).reshape(1, chunk, 1)
        return (torch.from_numpy(n).to(device, dtype), torch.from_numpy(w).to(device, dtype), float(self.rtc_g))


def _expand_batch(obj, k: int, torch):
    """Repeat batch-1 tensors of an ActionExpertContext (dataclass / tuples / lists) to batch k (views, no copy)."""
    import dataclasses

    if isinstance(obj, torch.Tensor):
        return obj.expand(k, *obj.shape[1:]) if obj.dim() >= 2 and obj.shape[0] == 1 else obj
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.replace(obj, **{f.name: (_expand_batch(getattr(obj, f.name), k, torch)
                                                    if f.name != "rope_cache" else getattr(obj, f.name))
                                           for f in dataclasses.fields(obj)})
    if isinstance(obj, tuple):
        vals = [_expand_batch(v, k, torch) for v in obj]
        return type(obj)(*vals) if hasattr(obj, "_fields") else tuple(vals)
    if isinstance(obj, list):
        return [_expand_batch(v, k, torch) for v in obj]
    return obj


if __name__ == "__main__":
    if "--worker" in sys.argv:
        _worker()
    else:
        print(__doc__)
