"""PyTorch reference for the fine-tuned π0.5: the Studio checkpoint run eagerly, same call as `Pi05Policy`.

Used to check the OpenVINO export's parity (`dlsj.deploy.pi05_parity`) and as a `--backend torch` fallback in
`dlsj.deploy.pi05_control`. Needs the Studio library env (torch, physicalai-train with the pi05 extra) and the
PaliGemma tokenizer in the HF cache. The live deployment path is OpenVINO.
"""

from pathlib import Path

import numpy as np


class TorchPi05Policy:
    def __init__(self, ckpt: str | Path, device: str = "cpu", dtype: str = "float32"):
        import torch

        from dlsj.deploy.pi05_export import load

        self.torch = torch
        policy = load(str(ckpt), steps=None)  # fp32, eval, checkpoint memory-mapped while loading
        self.policy = policy.to(getattr(torch, dtype)).to(device).eval()
        self.device = device
        self._noise = None
        inner = self.policy.inner_model
        orig = inner.sample_noise

        def sample_noise(shape, device):  # fixed start noise when given, for parity checks
            if self._noise is not None:
                return self._noise.to(device=device, dtype=torch.float32).reshape(shape)
            return orig(shape, device)

        inner.sample_noise = sample_noise

    def __call__(self, state: np.ndarray, images: dict[str, np.ndarray], task: str,
                 noise: np.ndarray | None = None) -> np.ndarray:
        """state (6,); images {camera: (H, W, 3) uint8 RGB}; noise (chunk, 32) or None -> (chunk, 6)."""
        from physicalai.data.observation import Observation

        torch = self.torch
        self._noise = None if noise is None else torch.as_tensor(np.asarray(noise, np.float32))[None]
        obs = Observation(
            state=torch.as_tensor(np.asarray(state, np.float32))[None],
            images={k: torch.as_tensor(np.ascontiguousarray(v.transpose(2, 0, 1)), dtype=torch.float32)[None] / 255.0
                    for k, v in images.items()},
            task=[task],
        )
        with torch.no_grad():
            out = self.policy.predict_action_chunk(obs)
        return out[0].float().cpu().numpy()
