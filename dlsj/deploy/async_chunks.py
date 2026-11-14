"""Asynchronous action-chunk inference for Diffusion Policy (UMI-style receding horizon).

With blocking inference the control loop freezes for every new chunk (~250 ms on the XPS GPU), so the policy's
two-frame observation history ends up ~250 ms apart around a stationary arm, which it never saw in training
(frames 1/30 s apart, arm moving). Here a background thread predicts chunks back to back from the two most recent
30 Hz observations while the control loop keeps executing the newest chunk:

* `push(tick, obs)` every control tick (30 Hz) keeps the last `n_obs_steps` observations.
* The worker takes a snapshot, predicts `n_action_steps` actions starting at the newest snapshot tick `t0`, and
  publishes them as covering ticks `t0 .. t0 + n - 1`.
* `action(tick)` returns the newest chunk's action for that tick. Actions for ticks that passed while the chunk was
  being computed are skipped (latency compensation, as in UMI). If the newest chunk is exhausted it holds its last
  action.

A fixed initial noise sample (DDIM is deterministic given the noise) keeps consecutive chunks on the same mode
instead of flipping between e.g. two bricks.
"""

from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np
import torch


class AsyncChunker:
    def __init__(self, policy, preprocessor, postprocessor, device: torch.device, fixed_noise: bool = True,
                 seed: int = 0):
        from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

        cfg = policy.config
        if cfg.type != "diffusion":
            raise ValueError("AsyncChunker supports Diffusion Policy (it calls policy.diffusion.generate_actions)")
        self.policy, self.pre, self.post, self.device = policy, preprocessor, postprocessor, device
        self.n_obs, self.n_act = cfg.n_obs_steps, cfg.n_action_steps
        self.image_keys = list(cfg.image_features)
        self.OBS_IMAGES, self.OBS_STATE = OBS_IMAGES, OBS_STATE
        self.hist: deque[tuple[int, dict]] = deque(maxlen=self.n_obs)
        self.lock = threading.Lock()
        self.chunk: tuple[int, np.ndarray] | None = None  # (start tick, actions [n, A])
        self.latencies_ms: list[float] = []
        self._stop = threading.Event()
        self._new_obs = threading.Event()
        self.noise = None
        if fixed_noise:
            g = torch.Generator().manual_seed(seed)
            self.noise = torch.randn((1, cfg.horizon, cfg.action_feature.shape[0]), generator=g).to(device)
        self._thread = threading.Thread(target=self._run, name="dp-chunks", daemon=True)

    def start(self) -> "AsyncChunker":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._new_obs.set()

    def push(self, tick: int, obs: dict[str, np.ndarray]) -> None:
        with self.lock:
            self.hist.append((tick, obs))
        self._new_obs.set()

    def action(self, tick: int) -> np.ndarray | None:
        with self.lock:
            if self.chunk is None:
                return None
            t0, acts = self.chunk
        i = min(max(tick - t0, 0), len(acts) - 1)
        return acts[i]

    # ------------------------------------------------------------------ worker
    def _prepare(self, obs: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        from lerobot.common.control_utils import prepare_observation_for_inference

        o = prepare_observation_for_inference(dict(obs), self.device)
        return self.pre(o)

    @torch.no_grad()
    def predict(self, snapshot: list[tuple[int, dict]]) -> np.ndarray:
        frames = [self._prepare(o) for _, o in snapshot]
        batch = {self.OBS_STATE: torch.stack([f[self.OBS_STATE] for f in frames], dim=1)}
        if self.image_keys:
            per_key = [torch.stack([f[k] for f in frames], dim=1) for k in self.image_keys]  # [B, n_obs, C, H, W]
            batch[self.OBS_IMAGES] = torch.stack(per_key, dim=-4)  # [B, n_obs, n_cam, C, H, W]
        acts = self.policy.diffusion.generate_actions(batch, noise=self.noise)  # [1, n_act, A], normalised
        acts = self.post(acts)
        return acts.squeeze(0).float().cpu().numpy()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._new_obs.wait(0.1)
            self._new_obs.clear()
            with self.lock:
                if len(self.hist) < self.n_obs:
                    continue
                snapshot = list(self.hist)
            t = time.perf_counter()
            acts = self.predict(snapshot)
            self.latencies_ms.append((time.perf_counter() - t) * 1e3)
            with self.lock:
                self.chunk = (snapshot[-1][0], acts)
