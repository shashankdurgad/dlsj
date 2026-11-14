"""WARP time-warp sampler (Yu et al., "WARP-RM", arXiv 2606.28320, Sec. 3.2 and App. D.1).

Replays a demonstration at a smoothly varying speed, with random reversals, and returns the source-frame
indices of one N-frame window plus its self-supervised progress labels y_j = (i_j - i_0) / C_norm.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WarpConfig:
    n: int = 16  # frames per window (paper: 32 on 50-120 s tasks)
    stride_s: float = 0.5  # canonical inter-token stride S (paper: 1.5 s real, 0.5 s sim)
    fps: float = 30.0
    alpha: float = 0.5  # AR(1) autocorrelation of log-velocity
    sigma: float = float(np.log(2.0))  # marginal std of log-velocity
    path_lo: float = 1 / 3  # path length l ~ U(path_lo, path_hi) * L
    path_hi: float = 5 / 3
    reversal_rate: float = 1.0  # Poisson rate of mid-window direction flips
    p_flip: float = 0.5  # probability the whole window plays backwards

    @property
    def window_s(self) -> float:
        return (self.n - 1) * self.stride_s

    @property
    def stride_frames(self) -> float:
        return self.stride_s * self.fps

    @property
    def c_norm(self) -> float:
        """Label normalisation: a window played at the canonical pace ends at y = 1."""
        return (self.n - 1) * self.stride_frames


def warp_displacements(cfg: WarpConfig, rng: np.random.Generator, path_frames: float) -> np.ndarray:
    """Signed cumulative displacements c_0..c_{N-1} (c_0 = 0), in source frames, total |path| = path_frames."""
    m = cfg.n - 1
    z = np.empty(m)
    z[0] = rng.normal(0.0, cfg.sigma)
    k = np.sqrt(1.0 - cfg.alpha**2) * cfg.sigma
    for i in range(1, m):
        z[i] = cfg.alpha * z[i - 1] + k * rng.normal()
    v = np.exp(z)
    u = path_frames * v / v.sum()  # Eq. 2: non-negative steps summing to the path length
    sign = np.ones(m)
    r = min(rng.poisson(cfg.reversal_rate), m - 1)
    if r:
        for pt in rng.choice(np.arange(1, m), size=r, replace=False):
            sign[pt:] *= -1.0  # flip every later step at each reversal point
    if rng.random() < cfg.p_flip:
        sign *= -1.0
    return np.concatenate([[0.0], np.cumsum(u * sign)])


def sample_window(cfg: WarpConfig, n_frames: int, rng: np.random.Generator, max_tries: int = 20):
    """(indices[N] int, labels[N] float) for one warped window of an episode with `n_frames` frames.

    If the sampled path does not fit the episode, it is resampled, and finally shrunk to fit, so short
    episodes still contribute (their windows just span less time).
    """
    if n_frames < 2:
        raise ValueError("episode too short")
    L = cfg.window_s * cfg.fps
    for attempt in range(max_tries + 1):
        path = rng.uniform(cfg.path_lo, cfg.path_hi) * L
        c = warp_displacements(cfg, rng, path)
        span = c.max() - c.min()
        if span > n_frames - 1:
            if attempt < max_tries:
                continue
            c = c * (n_frames - 1) / span  # last resort: shrink to fit
        lo = int(np.ceil(-c.min()))
        hi = int(np.floor(n_frames - 1 - c.max()))
        if hi < lo:
            continue
        i0 = int(rng.integers(lo, hi + 1))
        idx = np.clip(np.rint(i0 + c).astype(np.int64), 0, n_frames - 1)
        labels = (idx - idx[0]) / cfg.c_norm
        return idx, labels.astype(np.float32)
    raise RuntimeError("could not fit a window into the episode")


def canonical_window(cfg: WarpConfig, start: int) -> np.ndarray:
    """Evenly spaced inference window starting at `start` (may run past the episode; caller clips)."""
    return start + np.rint(np.arange(cfg.n) * cfg.stride_frames).astype(np.int64)
