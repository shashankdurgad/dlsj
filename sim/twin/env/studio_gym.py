"""Physical AI Studio `Gym` over the twin (num_envs = 1), so Studio policies / its rollout loop run in the twin.

    from sim.twin.env import TwinGym, TwinEnvCfg
    gym = TwinGym(TwinEnvCfg())                     # launches Isaac if needed
    obs, info = gym.reset(seed=0)                   # physicalai.data.Observation (batched, B = 1)
    obs, reward, terminated, truncated, info = gym.step(action)   # action (6,) or (1, 6), LeRobot units

Observation (same layout as Studio's LiberoGym): images {"top", "wrist"[, "inspect"]}: float32 (1, 3, H, W) in
[0, 1]; state: float32 (1, 6) (5 joints in cal2 degrees + gripper 0-100, the recorded observation.state); task:
[str]. `step` takes ONE action (chunking is the policy's job); info["is_success"] = brick placed in the bowl.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from physicalai.data.observation import Observation
from physicalai.gyms.base import Gym

from sim.twin.env.twin_env import TwinEnv, TwinEnvCfg

# sample_action range (LeRobot units): the follower's practical joint range in the cal2 data
ACTION_LOW = np.array([-110.0, -100.0, -100.0, -100.0, -160.0, 0.0], np.float32)
ACTION_HIGH = np.array([110.0, 100.0, 100.0, 100.0, 160.0, 100.0], np.float32)


class TwinGym(Gym):
    def __init__(self, cfg: TwinEnvCfg | None = None, env: TwinEnv | None = None, cameras: tuple | None = None,
                 device: str | torch.device = "cpu"):
        """cfg / env: the twin (an existing TwinEnv is reused); cameras: image keys to return (default the env's
        cfg.cameras); device: torch device of the returned Observation."""
        self.env = env or TwinEnv(cfg)
        self.cameras = tuple(cameras or self.env.cfg.cameras)
        self.device = torch.device(device)
        self.num_envs = 1
        self.task_description = self.env.task()
        try:
            from gymnasium import spaces

            self.action_space = spaces.Box(ACTION_LOW, ACTION_HIGH, dtype=np.float32)
        except ImportError:
            self.action_space = None

    # ------------------------------------------------------------------ Gym API
    def reset(self, *, seed: int | None = None, episode_index: int = 0, **reset_kwargs: Any):
        """reset_kwargs go to TwinEnv.reset (randomize, joint_deg, brick_pose, bowl)."""
        raw = self.env.reset(seed=seed, **reset_kwargs)
        self.task_description = raw["task"]
        info = {"is_success": False, "task": raw["task"], "episode_index": episode_index,
                "bowl": self.env.residual.bowl_names[int(self.env.residual.bowl[0])]}
        return self.to_observation(raw), info

    def step(self, action: torch.Tensor | np.ndarray):
        if isinstance(action, torch.Tensor):
            action = action.detach().to(torch.float32).cpu().numpy()
        action = np.asarray(action, np.float32)
        if action.ndim == 2 and action.shape[0] == 1:
            action = action[0]
        if action.shape != (6,):
            raise ValueError(f"TwinGym.step expects ONE action of shape (6,) or (1, 6), got {action.shape}")
        raw, reward, terminated, truncated, info = self.env.step(action)
        self.task_description = raw["task"]
        return self.to_observation(raw), float(reward), bool(terminated), bool(truncated), info

    def render(self, *args: Any, **kwargs: Any) -> np.ndarray | None:
        """Last rendered top image (H, W, 3) uint8."""
        return self.env._images.get("top")

    def get_max_episode_steps(self) -> int | None:
        return self.env.max_episode_steps

    def close(self) -> None:
        self.env.close()

    def sample_action(self) -> torch.Tensor:
        """A random goal in the follower's joint range, (6,) float32 (LeRobot units)."""
        a = np.random.default_rng().uniform(ACTION_LOW, ACTION_HIGH).astype(np.float32)
        return torch.from_numpy(a)

    def to_observation(self, raw_obs: dict) -> Observation:
        images = {}
        for cam in self.cameras:
            img = raw_obs["images"].get(cam)
            if img is None:
                continue
            t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
            images[cam] = t.to(self.device)
        state = torch.from_numpy(np.asarray(raw_obs["state"], np.float32)).unsqueeze(0).to(self.device)
        return Observation(images=images or None, state=state, task=[raw_obs["task"]])
