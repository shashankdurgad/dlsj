"""Score every episode with WARP-RM and write task-preserving WARP-BC weights.

    python -m dlsj.warp.score --root ~/desmond/data/duplo_teleop --feats ~/desmond/warp/feats \
        --rm ~/desmond/warp/rm/warp_rm.pt --out ~/desmond/warp/weights --chunk 100

Outputs in --out:
* warp_weights.parquet: one row per dataset frame: index, episode_index, frame_index, v, v_norm, weight,
  reason (pace / protected / coverage / dropped / failed_episode), stage, phase.
* coverage.csv: kept fraction per stage x phase slice (the check that no part of the task was cut).
* episodes.csv: per-episode length, mean v and kept fraction (mean v should fall as episodes get longer).
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dlsj.warp import sampler, weights
from dlsj.warp.data import load_episodes, load_features
from dlsj.warp.model import WarpRM, aggregate_velocity, window_velocities

log = logging.getLogger("warp.score")


@torch.no_grad()
def episode_velocity(model, cfg: sampler.WarpConfig, f: np.ndarray, device: str, bs: int = 512) -> np.ndarray:
    """Dense per-frame velocity v_t (paper Sec. 3.5 / App. D.4): stride-1 sliding canonical windows."""
    T = len(f)
    stride = cfg.stride_frames
    shrink = 1.0
    if (cfg.n - 1) * stride > T - 1:  # episode shorter than one window: squeeze the window, rescale velocity
        shrink = (T - 1) / ((cfg.n - 1) * stride)
        stride *= shrink
    offsets = np.rint(np.arange(cfg.n) * stride).astype(np.int64)
    starts = np.arange(0, T - offsets[-1])
    idx = starts[:, None] + offsets[None, :]
    y = np.concatenate([model.expected(torch.from_numpy(f[idx[i: i + bs]]).to(device)).cpu().numpy()
                        for i in range(0, len(idx), bs)])
    vel = window_velocities(y, cfg.n) / shrink
    v = aggregate_velocity(T, starts, idx, vel)
    return pd.Series(v).ffill().bfill().fillna(0.0).to_numpy()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--feats", required=True)
    p.add_argument("--rm", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--chunk", type=int, default=100, help="policy action chunk length in frames (ACT chunk_size)")
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--min-keep", type=float, default=0.25)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ck = torch.load(Path(args.rm).expanduser(), map_location="cpu", weights_only=False)
    cfg = sampler.WarpConfig(**ck["warp"])
    model = WarpRM(ck["feat_dim"], cfg.n, d_model=ck["d_model"], layers=ck["layers"])
    model.load_state_dict(ck["state_dict"])
    model.to(args.device).eval()
    eps, fps = load_episodes(args.root)
    wcfg = weights.WeightConfig(tau=args.tau, chunk=args.chunk, min_keep=args.min_keep)

    tables, ep_rows = [], []
    for e in eps:
        v = episode_velocity(model, cfg, load_features(args.feats, ck["cameras"], e.index), args.device)
        v = np.pad(v, (0, max(0, e.length - len(v))), mode="edge")[: e.length]
        t = weights.episode_frame_table(e.index, v, e.split, e.gripper, fps, wcfg)
        t["index"] = e.start + t["frame_index"]
        t["success"] = e.success
        tables.append(t)
        ep_rows.append({"episode_index": e.index, "length_s": e.length / fps, "success": e.success,
                        "split": e.split, "mean_v": float(np.mean(v))})
    frames = pd.concat(tables, ignore_index=True)

    ref = set(ck["reference"])
    rf = frames[frames.episode_index.isin(ref)]
    stage_ref = {int(s): float(g["v"].median()) for s, g in rf.groupby("stage")}
    log.info("reference median velocity per stage (0 = pick, 1 = place): %s", stage_ref)

    ok = frames[frames.success]
    scored = weights.compute_weights(ok, stage_ref, wcfg)
    failed = frames[~frames.success].assign(v_norm=0.0, weight=0.0, reason="failed_episode")
    df = pd.concat([scored, failed]).sort_values("index")

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    df[["index", "episode_index", "frame_index", "v", "v_norm", "weight", "reason", "stage", "phase"]].to_parquet(
        out / "warp_weights.parquet", index=False)
    cov = weights.coverage_report(scored)
    cov.to_csv(out / "coverage.csv")
    eps_df = pd.DataFrame(ep_rows).merge(
        df.groupby("episode_index").agg(kept=("weight", lambda w: float((w > 0).mean()))).reset_index())
    eps_df.to_csv(out / "episodes.csv", index=False)
    corr = float(np.corrcoef(eps_df.length_s, eps_df.mean_v)[0, 1]) if len(eps_df) > 2 else float("nan")
    summary = {"kept_fraction": float((df.weight > 0).mean()), "reasons": df.reason.value_counts().to_dict(),
               "stage_ref_velocity": stage_ref, "corr_length_vs_mean_v": corr,
               "min_phase_kept": float(cov["kept"].min())}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("summary: %s", json.dumps(summary))
    log.info("coverage (kept fraction per stage x phase slice):\n%s", cov.to_string())


if __name__ == "__main__":
    main()
