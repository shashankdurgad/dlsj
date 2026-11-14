"""Task-preserving WARP-BC chunk weights for pick -> inspect -> place.

Paper WARP-BC (Eq. 4): w = v_end * 1[v_end > tau], with v_end the progress velocity at the chunk's last frame,
and tau = 1 (expert pace). Applied blindly, that drops ~65% of chunks, including segments that are slow on
purpose and look like "no progress" on camera: holding the brick still in front of the inspection camera,
aligning a grasp, easing a brick onto the plate. Here we keep the paper's rule but add three guards:

1. Stage-relative pace: v is divided by the reference demos' median v *for that stage* (pick / place), so a
   stage that is inherently slower on camera is judged against its own normal pace, not the global one.
2. Protected frames: the inspection hold (around `inspection_frame`) and gripper close/open moments always
   keep weight >= `protect_weight`.
3. Coverage floor: each stage is cut into `phase_bins` slices of normalised stage time; if a slice keeps
   fewer than `min_keep` of its chunks across the dataset, its best chunks (highest v) are added back.

Every frame gets a `reason` so the coverage report can show exactly what was kept and why.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WeightConfig:
    tau: float = 1.0  # paper default: keep chunks ending above the stage's reference pace
    chunk: int = 100  # policy action-chunk length in frames (ACT chunk_size); v is read at its last frame
    protect_weight: float = 1.0
    hold_before_s: float = 1.0  # protected window around the inspection frame
    hold_after_s: float = 1.5
    grip_window_s: float = 0.5  # protected window around gripper open/close events
    grip_threshold: float = 15.0  # gripper action change (0-100 units) that counts as an event
    phase_bins: int = 10
    min_keep: float = 0.25  # minimum fraction of chunks kept per (stage, phase slice)
    w_max: float = 3.0  # clip very fast segments so a few chunks can't dominate


def stage_of(n_frames: int, split: int | None) -> np.ndarray:
    s = np.zeros(n_frames, dtype=np.int8)  # 0 = pick
    if split is not None and 0 <= split < n_frames:
        s[split:] = 1  # 1 = place
    return s


def phase_of(stages: np.ndarray, bins: int) -> np.ndarray:
    """Normalised time within each stage, bucketed into `bins` slices (0..bins-1)."""
    out = np.zeros(len(stages), dtype=np.int16)
    for st in np.unique(stages):
        idx = np.flatnonzero(stages == st)
        if len(idx):
            out[idx] = np.minimum((np.arange(len(idx)) * bins) // len(idx), bins - 1)
    return out


def gripper_events(grip: np.ndarray, fps: float, threshold: float, window_s: float) -> np.ndarray:
    """Boolean mask of frames within `window_s` of a large gripper command change (grasp / release)."""
    mask = np.zeros(len(grip), dtype=bool)
    if len(grip) < 2:
        return mask
    w = max(1, int(round(window_s * fps)))
    change = np.abs(grip[min(w, len(grip) - 1):] - grip[: len(grip) - min(w, len(grip) - 1)])
    for t in np.flatnonzero(change > threshold):
        mask[max(0, t - w): t + 2 * w] = True
    return mask


def episode_frame_table(ep: int, v: np.ndarray, split: int | None, grip: np.ndarray | None, fps: float,
                        cfg: WeightConfig) -> pd.DataFrame:
    n = len(v)
    stages = stage_of(n, split)
    end = np.minimum(np.arange(n) + cfg.chunk - 1, n - 1)
    protected = np.zeros(n, dtype=bool)
    if split is not None:
        a = max(0, split - int(cfg.hold_before_s * fps))
        protected[a: split + int(cfg.hold_after_s * fps)] = True
    if grip is not None:
        protected |= gripper_events(np.asarray(grip, float), fps, cfg.grip_threshold, cfg.grip_window_s)
    return pd.DataFrame({
        "episode_index": ep, "frame_index": np.arange(n), "v": v, "v_end": v[end],
        "stage": stages, "stage_end": stages[end], "phase": phase_of(stages, cfg.phase_bins), "protected": protected,
    })


def compute_weights(frames: pd.DataFrame, stage_ref: dict[int, float], cfg: WeightConfig) -> pd.DataFrame:
    """Add `v_norm`, `weight` and `reason` columns. `stage_ref[stage]` = reference median v of that stage.

    v_end is normalised by the reference pace of the stage the chunk *ends* in (a long chunk that starts in pick
    can end in place); coverage is still counted by the stage/phase of the chunk's first frame (its state).
    """
    df = frames.copy()
    ref = df["stage_end"].map(lambda s: stage_ref.get(int(s), 1.0)).astype(float).clip(lower=1e-3)
    df["v_norm"] = (df["v_end"] / ref).fillna(0.0)
    keep = df["v_norm"] > cfg.tau
    df["weight"] = np.where(keep, df["v_norm"].clip(upper=cfg.w_max), 0.0)
    df["reason"] = np.where(keep, "pace", "dropped")

    prot = df["protected"] & (df["weight"] < cfg.protect_weight)
    df.loc[prot, "weight"] = cfg.protect_weight
    df.loc[prot, "reason"] = "protected"

    for (_, _), g in df.groupby(["stage", "phase"]):
        kept = (g["weight"] > 0).sum()
        need = int(np.ceil(cfg.min_keep * len(g))) - kept
        if need > 0:
            cand = g[g["weight"] == 0].sort_values("v_norm", ascending=False).head(need)
            df.loc[cand.index, "weight"] = cand["v_norm"].clip(lower=0.25, upper=1.0)
            df.loc[cand.index, "reason"] = "coverage"
    return df


def coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    """Kept fraction per stage x phase slice, plus why frames were kept."""
    rep = df.assign(kept=df["weight"] > 0).groupby(["stage", "phase"]).agg(
        frames=("kept", "size"), kept=("kept", "mean"), mean_w=("weight", "mean"))
    rep.index = rep.index.set_levels([["pick", "place"][int(s)] for s in rep.index.levels[0]], level=0)
    return rep
