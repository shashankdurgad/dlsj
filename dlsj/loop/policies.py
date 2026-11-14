"""pi0.5 OpenVINO back-end for the closed loop: a Physical AI Studio export run with `physicalai.InferenceModel`.

The export directory (`manifest.json` + `pi05.xml/.bin` + `tokenizer.xml/.bin`) carries the whole runtime pipeline:
quantile-normalise the state -> pi0.5 image prep (resize-with-pad to 224², [-1, 1]) + prompt with the discretised
state -> start noise -> OpenVINO tokenizer -> model -> denormalise. So this adapter only has to build the
observation dict the export was trained on, exactly as `deploy/pi05/studio_bench.py` does:

    {"images.<cam>": (1, 3, H, W) uint8 RGB, ..., "state": (1, D) float32 raw units, "task": [instruction]}

* camera names come from the manifest's `images.<name>` input features (training order; the runtime re-orders
  them itself); each maps to `Observation.images[<name>]` unless `camera_map` says otherwise;
* `state` is passed in recorder units (degrees, gripper 0-100, order `poses.MOTORS`): the export's own
  normaliser uses the dataset statistics, so no scaling happens here;
* the returned chunk is `(chunk_len, 6)` float32 in the same units (columns past the state dimension, e.g. the
  32-d padding of a raw pi0.5 head, are dropped).

Thread-safety: `predict_chunk` is called from the executor's inference thread while `reset` may come from the
controller thread; one lock serialises every call into the compiled model (an OpenVINO `CompiledModel.__call__`
shares one infer request).

RTC: if the export has a `prev_chunk_left_over` input (a Studio export made with RTC), the unexecuted tail of the
previous raw (normalised) chunk is fed back, with `inference_delay` from the measured latency, the same inputs
Studio's `RTCExecution` builds. Exports without those inputs get plain chunks. (No RTC export existed when this
was written: that branch is only unit-tested against a stub.)
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from dlsj.contracts import Observation

log = logging.getLogger(__name__)

IMAGES_PREFIX = "images."
STATE, TASK, NOISE, ACTION = "state", "task", "noise", "action"
PREV_CHUNK_LEFT_OVER, RTC_INFERENCE_DELAY = "prev_chunk_left_over", "inference_delay"
RTC_MAX_GUIDANCE_WEIGHT, RTC_EXECUTION_HORIZON = "max_guidance_weight", "execution_horizon"
DEFAULT_CACHE_DIR = Path("~/inference/ov_cache").expanduser()
WARMUP_TASK = "pick up the brick and hold it in front of the inspection camera"


class _RawActionTap:
    """Postprocessor inserted first in the export's chain: keeps the raw (normalised) action for RTC feedback."""

    def __init__(self):
        self.raw: np.ndarray | None = None

    def __call__(self, outputs: dict) -> dict:
        a = outputs.get(ACTION)
        self.raw = None if a is None else np.array(a, dtype=np.float32, copy=True)
        return outputs


def _feature_shape(feat: Any) -> tuple[int, ...]:
    return tuple(int(s) for s in (getattr(feat, "shape", None) or ()))


def _noise_shape(model: Any) -> tuple[int, ...] | None:
    """Per-sample start-noise shape from the export's noise (or pi05) preprocessor, if it declares one."""
    for p in getattr(model, "preprocessors", []):
        for attr in ("_shape", "_noise_shape"):
            shape = getattr(p, attr, None)
            if shape and NOISE in type(p).__name__.lower() + attr:
                return tuple(int(s) for s in shape)
    return None


class OpenVINOPi05Policy:
    """`dlsj.contracts.Policy` over a Studio pi0.5 OpenVINO export (any of `~/inference/exports/pi05_junction_*`).

    Args:
        model_dir: export directory with `manifest.json`.
        device: OpenVINO device of the policy (GPU on the XPS).
        cache_dir: OpenVINO `CACHE_DIR` (absolute) so the GPU compile is cached across runs.
        camera_map: manifest camera name -> `Observation.images` key, for cameras named differently.
        seed: seeds the start noise drawn here (reproducible chunks; `reset()` re-seeds). None = the export's own
            fresh noise every call.
        rtc: feed the previous chunk back when the export has RTC inputs (`None` = auto).
        fps: control rate, converts elapsed time to executed actions for RTC.
        model: an already-built `InferenceModel`-like object (tests); skips loading.
        config: extra OpenVINO compile properties.
    """

    def __init__(self, model_dir: str | Path | None, device: str = "GPU", cache_dir: str | Path | None = None,
                 camera_map: dict[str, str] | None = None, seed: int | None = None, rtc: bool | None = None,
                 fps: float = 30.0, execution_horizon: int = 10, max_guidance_weight: float = 10.0,
                 model: Any = None, config: dict[str, str] | None = None):
        self.model_dir = Path(model_dir).expanduser() if model_dir is not None else None
        self.device, self.seed, self.fps = device, seed, fps
        self._lock = threading.Lock()
        self.load_s = 0.0
        if model is None:
            if self.model_dir is None or not (self.model_dir / "manifest.json").is_file():
                raise FileNotFoundError(f"--policy-path must be a Studio export dir with manifest.json: {model_dir}")
            try:
                from physicalai.inference import InferenceModel
            except ImportError as e:
                raise RuntimeError("OpenVINOPi05Policy needs the `physicalai` runtime (see requirements-loop.txt)") from e
            cfg = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1",
                   "CACHE_DIR": str(Path(cache_dir or DEFAULT_CACHE_DIR).expanduser().resolve())}
            Path(cfg["CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
            cfg.update(config or {})
            t0 = time.perf_counter()
            model = InferenceModel(str(self.model_dir), backend="openvino", device=device, **cfg)
            self.load_s = time.perf_counter() - t0
            log.info("pi0.5 export %s compiled on %s in %.1f s", self.model_dir, device, self.load_s)
        self.model = model

        feats = list(model.input_features)
        self.cameras = [f.name.removeprefix(IMAGES_PREFIX) for f in feats if f.name.startswith(IMAGES_PREFIX)]
        if not self.cameras and any(f.name == "images" for f in feats):
            raise ValueError("single-camera export ('images' feature): not the 3-camera pi0.5 this loop expects")
        self.camera_map = {c: (camera_map or {}).get(c, c) for c in self.cameras}
        state_feats = [f for f in feats if f.name == STATE]
        if not state_feats:
            raise ValueError(f"export has no 'state' input feature: {[f.name for f in feats]}")
        self.state_dim = (_feature_shape(state_feats[0]) or (6,))[-1]
        self.chunk_len = int(getattr(model, "chunk_size", 0) or 0)
        out = [f for f in getattr(model, "output_features", []) if f.name == ACTION]
        if out and len(_feature_shape(out[0])) == 2:
            self.chunk_len = self.chunk_len if self.chunk_len > 1 else _feature_shape(out[0])[0]
        self.action_dim = self.state_dim
        self._noise_shape = _noise_shape(model)
        self._rng = np.random.default_rng(seed)

        inputs = set(getattr(getattr(model, "adapter", None), "input_names", None) or [])
        self.has_rtc = PREV_CHUNK_LEFT_OVER in inputs
        self.rtc = self.has_rtc if rtc is None else (bool(rtc) and self.has_rtc)
        if rtc and not self.has_rtc:
            log.warning("--policy-rtc requested but the export has no %s input: plain chunks", PREV_CHUNK_LEFT_OVER)
        self.execution_horizon, self.max_guidance_weight = execution_horizon, max_guidance_weight
        self._tap = _RawActionTap()
        if self.rtc:
            model.postprocessors.insert(0, self._tap)
        self._prev_raw: np.ndarray | None = None
        self._prev_t: float | None = None

        self.latency_ms = 0.0  # wall time of the last predict_chunk (input build + runtime call + post)
        self.calls = 0
        log.info("pi0.5 policy: cameras %s, state dim %d, chunk %d, rtc=%s", self.camera_map, self.state_dim,
                 self.chunk_len, self.rtc)

    # ------------------------------------------------------------------ mapping
    def build_inputs(self, obs: Observation, instruction: str, noise: np.ndarray | None = None) -> dict[str, Any]:
        """Observation + instruction -> the export's input dict (what `studio_bench.make_observation` builds)."""
        state = np.asarray(obs.state, dtype=np.float32).reshape(-1)
        if state.shape[0] != self.state_dim:
            raise ValueError(f"state has {state.shape[0]} values, the export expects {self.state_dim}")
        if not np.all(np.isfinite(state)):
            raise ValueError(f"non-finite state {state}")
        inputs: dict[str, Any] = {}
        for cam in self.cameras:  # manifest (training) order
            key = self.camera_map[cam]
            img = obs.images.get(key)
            if img is None:
                raise KeyError(f"policy camera '{cam}' needs Observation.images['{key}'], have {sorted(obs.images)}")
            img = np.asarray(img)
            if img.ndim != 3 or img.shape[2] != 3:
                raise ValueError(f"camera {key}: expected HxWx3 RGB, got {img.shape}")
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            inputs[IMAGES_PREFIX + cam] = np.ascontiguousarray(img.transpose(2, 0, 1)[None])  # (1, 3, H, W)
        inputs[STATE] = state[None]
        inputs[TASK] = [str(instruction)]
        if noise is not None:
            inputs[NOISE] = np.asarray(noise, dtype=np.float32)
        elif self.seed is not None and self._noise_shape is not None:
            inputs[NOISE] = self._rng.standard_normal((1, *self._noise_shape), dtype=np.float32)
        return inputs

    def postprocess(self, chunk: np.ndarray) -> np.ndarray:
        """Runtime output -> (n, state_dim) float32 joint targets; rejects anything non-finite or mis-shaped."""
        a = np.asarray(chunk, dtype=np.float32)
        if a.ndim == 3:
            if a.shape[0] != 1:
                raise ValueError(f"batched action output {a.shape}")
            a = a[0]
        if a.ndim != 2 or a.shape[1] < self.action_dim or len(a) == 0:
            raise ValueError(f"action chunk of shape {a.shape}, expected (n, >= {self.action_dim})")
        a = np.ascontiguousarray(a[:, : self.action_dim])
        if not np.all(np.isfinite(a)):
            raise ValueError("policy returned non-finite actions")
        return a

    def _rtc_inputs(self, inputs: dict[str, Any], t_obs: float) -> dict[str, Any]:
        n = self.chunk_len
        raw = self._prev_raw
        if raw is None or self._prev_t is None:
            left = np.zeros((1, n, raw.shape[-1] if raw is not None else 32), dtype=np.float32)
            weight, horizon, delay = 1e-8, 1, 0
        else:
            consumed = int(round((t_obs - self._prev_t) * self.fps))
            tail = raw[min(consumed, len(raw)):]
            left = np.zeros((1, n, raw.shape[-1]), dtype=np.float32)
            left[0, : len(tail)] = tail
            weight, horizon = self.max_guidance_weight, self.execution_horizon
            delay = min(max(int(np.ceil(self.latency_ms / 1000.0 * self.fps)), 0), horizon)
        return {**inputs, PREV_CHUNK_LEFT_OVER: left, RTC_INFERENCE_DELAY: np.int64(delay),
                RTC_MAX_GUIDANCE_WEIGHT: np.float32(weight), RTC_EXECUTION_HORIZON: np.int64(horizon)}

    # ------------------------------------------------------------------ Policy protocol
    def predict_chunk(self, obs: Observation, instruction: str, noise: np.ndarray | None = None) -> np.ndarray:
        with self._lock:
            t0 = time.perf_counter()
            inputs = self.build_inputs(obs, instruction, noise)
            t_obs = obs.timestamp or time.time()
            if self.rtc:
                inputs = self._rtc_inputs(inputs, t_obs)
            chunk = self.postprocess(self.model.predict_action_chunk(inputs))
            if self.rtc and self._tap.raw is not None:
                raw = self._tap.raw
                self._prev_raw, self._prev_t = (raw[0] if raw.ndim == 3 else raw), t_obs
            self.latency_ms = (time.perf_counter() - t0) * 1000.0
            self.calls += 1
            return chunk

    def reset(self) -> None:
        """New skill: clear runtime/RTC state; re-seed the noise when seeded (else the export draws fresh noise)."""
        with self._lock:
            self.model.reset()
            self._prev_raw, self._prev_t = None, None
            self._rng = np.random.default_rng(self.seed)

    def warmup(self, n: int = 2, hw: tuple[int, int] = (480, 640)) -> float:
        """Run `n` dummy chunks (first GPU call allocates/compiles kernels). Returns the last latency in ms."""
        obs = Observation(state=np.zeros(self.state_dim, np.float32),
                          images={self.camera_map[c]: np.zeros((*hw, 3), np.uint8) for c in self.cameras})
        for _ in range(n):
            self.predict_chunk(obs, WARMUP_TASK)
        self.reset()
        log.info("pi0.5 warmup: %d calls, last %.0f ms", n, self.latency_ms)
        return self.latency_ms

    def close(self) -> None:
        with self._lock:
            close = getattr(self.model, "close", None)
            if close:
                close()
