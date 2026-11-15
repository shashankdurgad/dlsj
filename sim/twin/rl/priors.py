"""Prior interface for the DICE-RL residual: the frozen base policy proposes K candidate chunks per decision.

    class Prior:
        chunk_len: int        # actions per sampled chunk
        exec_len: int         # actions executed before the next request (RTC horizon)
        needs_images: bool
        latent_dim: int       # pooled prefix latent appended to the residual / critic state (0 = none)
        sample(obs_batch, k, seeds) -> np.ndarray (B, k, chunk_len, 6)   cal2 runtime units
        last_latent -> np.ndarray (B, latent_dim) | None                 from the last sample() call

* ReplayErrorPrior: the state-based sim base (recorded approach + injected VLA-like errors, GraspEnv.candidates);
  the k candidates are independent per-candidate error / noise draws around this episode's systematic error.
* MolmoAct2Prior: sim/twin/policies/molmoact2.py's worker (`MolmoAct2Client.act_k`): one vision-language prefix
  forward, k flow-noise seeds batched in the action expert, RTC via prev_chunk / inference_delay, and the
  mean-pooled prefix hidden state as the latent. `snap` = HF snapshot dir / repo id (zero-shot) or a Studio
  Lightning .ckpt (the user's fine-tune / LoRA). Outside the pick prompt (gate.is_pick_prompt) callers use k = 1.

The residual / critic inputs stay low-dim: obs (joints, detector, load, time, offsets) + [projected latent] +
candidate chunk(s) relative to the state. `LatentProjector` is the fixed random projection to `proj_dim`
(CPU-cheap; PCA from collected latents can replace it).
"""
from __future__ import annotations

import numpy as np

# The 8k fine-tune (sim/twin/data/models/molmoact2_ft_8k) knows ONLY these prompts (finetune_cal2_456.json).
FT_PICK = "pick up the {colour} brick"  # colour in FT_BRICK_COLOURS
FT_BRING = "bring the brick to the inspection camera"
FT_PLACE = "place the brick in the {bowl} bowl"  # orange | purple | yellow
FT_BRICK_COLOURS = ("blue", "brown", "red", "white", "yellow")
FT_TOP_CROP = (0, 0, 560, 480)
# twin palette name (sim/twin/visuals/twin_scene.BRICK_RGB) for each trained colour word
TWIN_PALETTE = {"blue": "light_blue", "brown": "brown", "red": "red", "white": "cream", "yellow": "yellow"}
# real top-frame brick colours (sim/twin/visuals/twin_scene.BRICK_RGB) per trained colour word
FT_COLOUR_RGB = {"blue": (5, 138, 202), "brown": (110, 60, 37), "red": (124, 24, 43), "white": (205, 200, 160),
                 "yellow": (155, 173, 86)}


def brick_colour_word(wrist_rgb: np.ndarray) -> str:
    """Trained colour word of the brick held between the jaws (median of the wrist image's lower centre)."""
    h, w = wrist_rgb.shape[:2]
    med = np.median(wrist_rgb[int(0.6 * h):int(0.95 * h), int(0.38 * w):int(0.62 * w)].reshape(-1, 3), 0)
    return min(FT_COLOUR_RGB, key=lambda k: float(np.sum((med - np.asarray(FT_COLOUR_RGB[k])) ** 2)))


class Prior:
    chunk_len = 30
    exec_len = 15
    needs_images = False
    latent_dim = 0
    last_latent = None

    def sample(self, obs_batch, k: int, seeds=None) -> np.ndarray:
        raise NotImplementedError

    def close(self):
        pass


class ReplayErrorPrior(Prior):
    """Wraps a GraspEnv: obs_batch is ignored (the env holds each episode's recorded plan and error draw)."""

    needs_images = False
    latent_dim = 0

    def __init__(self, env):
        import torch

        self.env, self.torch = env, torch
        self.chunk_len = self.exec_len = env.h

    def sample(self, obs_batch=None, k: int = 4, seeds=None) -> np.ndarray:
        gen = None
        if seeds is not None:
            gen = self.torch.Generator(device=self.env.device).manual_seed(int(np.asarray(seeds).ravel()[0]))
        return self.env.candidates(k, gen).cpu().numpy()

    def sample_t(self, k: int, gen=None):
        """GPU tensor version used by the batched trainer (n, k, h, 6)."""
        return self.env.candidates(k, gen)


class MolmoAct2Prior(Prior):
    """obs_batch: list of {"images": {"wrist", "top"}: uint8 HxWx3, "state": (6,), "task": str,
    optional "prev_chunk": (n, 6), "inference_delay": int}."""

    needs_images = True

    def __init__(self, snap: str | None = None, cams=("wrist", "top"), instruction: str | None = None,
                 steps: int = 10, exec_len: int = 15, crops: dict | None = None, **client_kw):
        from sim.twin.policies.molmoact2 import CHUNK, MolmoAct2Client

        self.client = MolmoAct2Client(snap=snap, cams=cams, instruction=instruction, steps=steps, crops=crops,
                                      **client_kw)
        self.chunk_len, self.exec_len = CHUNK, exec_len
        self.latent_dim = 0
        self.last_info: list = []

    def sample(self, obs_batch, k: int = 4, seeds=None) -> np.ndarray:
        out, lat, info = [], [], []
        for b, obs in enumerate(obs_batch):
            sd = None if seeds is None else np.asarray(seeds)[b].reshape(-1).tolist()
            a, z = self.client.act_k(obs, k, sd, prev_chunk=obs.get("prev_chunk"),
                                     inference_delay=int(obs.get("inference_delay", 0)), task=obs.get("task"))
            out.append(a)
            lat.append(z)
            info.append(dict(self.client.last_info))
        self.last_info = info
        if lat and lat[0] is not None:
            self.last_latent = np.stack(lat)
            self.latent_dim = self.last_latent.shape[1]
        return np.stack(out)

    def close(self):
        self.client.close()


class LatentProjector:
    """Fixed Gaussian random projection (seeded) of the pooled latent to proj_dim, scaled to unit variance."""

    def __init__(self, din: int, proj_dim: int = 128, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W = (rng.standard_normal((din, proj_dim)) / np.sqrt(din)).astype(np.float32)

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z, np.float32) @ self.W
