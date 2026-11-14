"""WARP-RM: frozen per-frame features -> bidirectional transformer -> per-frame cumulative-progress bins.

Paper Sec. 3.3-3.5 / App. D.2. Tokens are [phi_j, phi_j - phi_{j-1}] (diff = 0 for j = 0), projected without
bias, plus fixed sinusoidal positions. Output: a categorical distribution over `n_bins` evenly spaced values
in [lo, hi] per frame, trained with cross-entropy against a two-hot encoding of the label.
"""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def two_hot(y: torch.Tensor, n_bins: int, lo: float, hi: float) -> torch.Tensor:
    """Encode values `y` (any shape) as two-hot distributions over evenly spaced bin centres."""
    y = y.clamp(lo, hi)
    pos = (y - lo) / (hi - lo) * (n_bins - 1)
    left = pos.floor().long().clamp(0, n_bins - 2)
    frac = (pos - left.float()).clamp(0.0, 1.0)
    out = torch.zeros(*y.shape, n_bins, device=y.device, dtype=torch.float32)
    out.scatter_(-1, left.unsqueeze(-1), (1.0 - frac).unsqueeze(-1))
    out.scatter_add_(-1, (left + 1).unsqueeze(-1), frac.unsqueeze(-1))
    return out


def sinusoidal(n: int, d: int) -> torch.Tensor:
    pos = torch.arange(n).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class WarpRM(nn.Module):
    def __init__(self, feat_dim: int, n_frames: int, d_model: int = 512, layers: int = 6, heads: int = 8,
                 dropout: float = 0.15, n_bins: int = 30, lo: float = -3.0, hi: float = 3.0):
        super().__init__()
        self.n_bins, self.lo, self.hi = n_bins, lo, hi
        self.proj = nn.Linear(2 * feat_dim, d_model, bias=False)
        self.register_buffer("pe", sinusoidal(n_frames, d_model), persistent=False)
        layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_bins)
        self.register_buffer("centres", torch.linspace(lo, hi, n_bins), persistent=False)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats [B, N, D] -> logits [B, N, n_bins]."""
        diff = torch.cat([torch.zeros_like(feats[:, :1]), feats[:, 1:] - feats[:, :-1]], dim=1)
        x = self.proj(torch.cat([feats, diff], dim=-1)) + self.pe[: feats.shape[1]]
        return self.head(self.norm(self.encoder(x)))

    def loss(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = self(feats)
        target = two_hot(labels, self.n_bins, self.lo, self.hi)
        return -(target * F.log_softmax(logits, dim=-1)).sum(-1).mean()

    @torch.no_grad()
    def expected(self, feats: torch.Tensor) -> torch.Tensor:
        """Expected cumulative progress y_hat [B, N]."""
        return (self(feats).softmax(-1) * self.centres).sum(-1)


def window_velocities(y_hat: np.ndarray, n: int) -> np.ndarray:
    """Intra-window velocities v_j = (N-1)(y_j - y_{j-1}) for j = 1..N-1 (paper Sec. 3.5)."""
    return (n - 1) * np.diff(y_hat, axis=-1)


def aggregate_velocity(n_frames: int, starts: np.ndarray, idx: np.ndarray, vel: np.ndarray) -> np.ndarray:
    """Mean velocity per source frame over every overlapping window that covers it.

    `idx[w]` are window w's frame indices (N), `vel[w]` its N-1 velocities; velocity j covers the source
    frames between idx[w, j-1] and idx[w, j]. Frames no window covers get NaN.
    """
    acc = np.zeros(n_frames)
    cnt = np.zeros(n_frames)
    for w in range(len(starts)):
        for j in range(1, idx.shape[1]):
            a, b = idx[w, j - 1], min(idx[w, j], n_frames)
            if b > a:
                acc[a:b] += vel[w, j - 1]
                cnt[a:b] += 1
    out = np.full(n_frames, np.nan)
    m = cnt > 0
    out[m] = acc[m] / cnt[m]
    return out
