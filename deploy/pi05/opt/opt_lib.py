"""Pure helpers for the π0.5 OpenVINO optimisation study (numpy only, no OpenVINO/torch).

Used by `eval_actions.py`, `bench_variants.py` and `quantize.py`, and unit-tested in
`tests/test_pi05_opt.py`: the held-out episode split, frame sampling across pick/place segments,
future-action targets with padding masks, accuracy metrics and latency statistics.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def validation_episodes(n_episodes: int = 212, frac: float = 0.1, seed: int = 42) -> list[int]:
    """Held-out episodes of the training split (same draw as training/pi05/train.py `estimate`)."""
    k = max(1, int(n_episodes * frac))
    return sorted(random.Random(seed).sample(range(n_episodes), k))


def segments(tasks: Sequence[str]) -> list[tuple[int, int, str]]:
    """Split one episode's per-frame task strings into contiguous runs `(start, stop, task)`."""
    out: list[tuple[int, int, str]] = []
    start = 0
    for i in range(1, len(tasks) + 1):
        if i == len(tasks) or tasks[i] != tasks[start]:
            out.append((start, i, tasks[start]))
            start = i
    return out


def sample_frames(episode_tasks: dict[int, Sequence[str]], n: int, seed: int = 0) -> list[tuple[int, int, str]]:
    """Pick about `n` frames `(episode, frame, task)` spread over every episode and task segment.

    Each contiguous task segment of each episode gets a share proportional to its length (at least
    one frame), taken at evenly spaced positions with a small seeded jitter. The result is sorted
    and deterministic for a given seed.
    """
    segs = [(ep, s, e, t) for ep, tasks in sorted(episode_tasks.items()) for s, e, t in segments(tasks)]
    if not segs or n <= 0:
        return []
    total = sum(e - s for _, s, e, _ in segs)
    rng = random.Random(seed)
    picks: list[tuple[int, int, str]] = []
    for ep, s, e, t in segs:
        length = e - s
        m = max(1, min(length, round(n * length / total)))
        step = length / m
        for j in range(m):
            lo = s + int(j * step)
            hi = max(lo, s + int((j + 1) * step) - 1)
            picks.append((ep, rng.randint(lo, hi), t))
    return sorted(set(picks))


def future_actions(actions: np.ndarray, t: int, chunk: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Recorded actions `[t, t+chunk)` of one episode, padded with the last action, plus a validity mask."""
    actions = np.asarray(actions)
    seg = actions[t : t + chunk]
    mask = np.zeros(chunk, dtype=bool)
    mask[: len(seg)] = True
    if len(seg) < chunk:
        seg = np.concatenate([seg, np.repeat(actions[-1:], chunk - len(seg), axis=0)], axis=0)
    return seg.astype(np.float32), mask


def chunk_errors(pred: np.ndarray, target: np.ndarray, mask: np.ndarray, ks: Iterable[int] = (10, 50),
                 joints: Sequence[str] = JOINTS) -> dict[str, Any]:
    """MSE/MAE of predicted chunks `[N,T,D]` vs recorded `[N,T,D]` over the first k steps, masked `[N,T]`.

    Returns `{"k10": {"mse", "mae", "per_joint": {joint: {"mse", "mae"}}}, ...}` in the action units.
    """
    pred, target, mask = np.asarray(pred, np.float64), np.asarray(target, np.float64), np.asarray(mask, bool)
    out: dict[str, Any] = {}
    for k in ks:
        m = mask[:, :k]
        err = pred[:, :k] - target[:, :k]
        w = m[..., None].astype(np.float64)
        cnt = max(float(m.sum()), 1.0)
        mse_j = (err**2 * w).sum(axis=(0, 1)) / cnt
        mae_j = (np.abs(err) * w).sum(axis=(0, 1)) / cnt
        out[f"k{k}"] = {
            "mse": float(mse_j.mean()), "mae": float(mae_j.mean()),
            "per_joint": {j: {"mse": float(a), "mae": float(b)} for j, a, b in zip(joints, mse_j, mae_j)},
        }
    return out


def deviation(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    """How far a variant's chunks `[N,T,D]` are from the reference's on the same inputs and noise.

    `cos_mean` is the mean per-sample cosine of the flattened chunks; `cos_centered` subtracts the
    reference's per-sample mean action first (raw joint positions are far from zero, so the plain
    cosine is close to 1 for any reasonable chunk).
    """
    p, r = np.asarray(pred, np.float64), np.asarray(ref, np.float64)
    d = p - r
    pf, rf = p.reshape(len(p), -1), r.reshape(len(r), -1)

    def cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        return (a * b).sum(axis=1) / np.where(den == 0, 1.0, den)

    mu = r.mean(axis=1, keepdims=True)
    pc, rc = (p - mu).reshape(len(p), -1), (r - mu).reshape(len(r), -1)
    return {
        "mse": float((d**2).mean()), "mae": float(np.abs(d).mean()), "max_abs": float(np.abs(d).max()),
        "cos_mean": float(cos(pf, rf).mean()), "cos_centered": float(cos(pc, rc).mean()),
        "per_joint_mae": {j: float(v) for j, v in zip(JOINTS, np.abs(d).mean(axis=(0, 1)))},
    }


def latency_stats(ms: Sequence[float]) -> dict[str, float]:
    """Median / p90 / p99 / min of a list of millisecond timings."""
    t = np.asarray(ms, dtype=np.float64)
    return {"median_ms": round(float(np.median(t)), 1), "p90_ms": round(float(np.percentile(t, 90)), 1),
            "p99_ms": round(float(np.percentile(t, 99)), 1), "min_ms": round(float(t.min()), 1), "n": int(len(t))}


def noise_for(index: int, shape: Sequence[int] = (1, 50, 32), base_seed: int = 1234) -> np.ndarray:
    """Fixed flow-matching start noise for sample `index` (identical for every variant)."""
    return np.random.default_rng(base_seed + index).standard_normal(tuple(shape), dtype=np.float32)


def accept(variant_k10_mae: float, ref_k10_mae: float, dev_mae: float, seed_floor_mae: float,
           tol: float = 0.05) -> bool:
    """Accuracy gate: k=10 MAE vs recorded at most `tol` worse than FP16, and deviation from FP16
    no larger than FP16's own seed-to-seed deviation (the policy's intrinsic sampling noise)."""
    return variant_k10_mae <= ref_k10_mae * (1 + tol) and dev_mae <= seed_floor_mae


def group_by(items: Iterable[tuple[Any, ...]], key_index: int) -> dict[Any, list[tuple[Any, ...]]]:
    """Group tuples by one field (used to batch frames by episode when decoding video)."""
    out: dict[Any, list[tuple[Any, ...]]] = defaultdict(list)
    for it in items:
        out[it[key_index]].append(it)
    return dict(out)


def classify_matmul(act_shape: Sequence[int], chunk: int = 50) -> str:
    """Subgraph of a MatMul in Studio's π0.5 IR from its activation (input 0) shape.

    The ONNX-dynamo op names are anonymous (`node_linear_123`), so shapes identify the parts:
    SigLIP runs the 3 cameras as a batch of 256-token images (`[3,256,·]`), the Gemma-2B prefix sees
    768 image + N text tokens (`[1,968,·]` at 200 tokens), and the action expert's unrolled denoising
    steps see the `chunk` action tokens (`[1,50,·]`, attention `[1,8,50,·]`).
    """
    s = list(act_shape)
    if len(s) < 2:
        return "other"
    if s[-2] == chunk:
        return "expert"
    if len(s) == 3 and s[0] == 3 and s[1] == 256:
        return "vision"
    if s[-2] >= 768 + 16:  # prefix rows: 3x256 image tokens + the prompt
        return "vlm"
    return "other"


def per_sample_mae(pred: np.ndarray, target: np.ndarray, mask: np.ndarray, k: int = 10) -> np.ndarray:
    """Masked MAE over the first k steps and all joints, one value per sample `[N]`."""
    err = np.abs(np.asarray(pred, np.float64)[:, :k] - np.asarray(target, np.float64)[:, :k]).mean(axis=2)
    m = np.asarray(mask, bool)[:, :k]
    return (err * m).sum(axis=1) / np.maximum(m.sum(axis=1), 1)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """Relative change of mean(a) over mean(b) for paired samples, with a 95% bootstrap interval."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    rel = a[idx].mean(axis=1) / b[idx].mean(axis=1) - 1
    return {"rel": float(a.mean() / b.mean() - 1), "lo": float(np.percentile(rel, 2.5)),
            "hi": float(np.percentile(rel, 97.5))}


def jerk(chunks: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Mean |second difference| of action chunks `[N,T,D]` over time (step-to-step jitter), on valid steps."""
    x = np.asarray(chunks, np.float64)
    d2 = np.abs(np.diff(x, 2, axis=1))  # [N,T-2,D]
    if mask is None:
        return float(d2.mean())
    m = np.asarray(mask, bool)
    m2 = m[:, 2:] & m[:, 1:-1] & m[:, :-2]
    return float((d2 * m2[..., None]).sum() / max(m2.sum() * x.shape[2], 1))


def range_ratio(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """Mean per-joint range of motion of predicted chunks over that of the recorded ones (full-length chunks only)."""
    full = np.asarray(mask, bool).all(axis=1)
    if not full.any():
        return float("nan")
    p, t = np.asarray(pred)[full], np.asarray(target)[full]
    return float((p.max(1) - p.min(1)).mean() / (t.max(1) - t.min(1)).mean())
