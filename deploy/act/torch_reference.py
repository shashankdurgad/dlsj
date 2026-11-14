"""PyTorch reference for the LeRobot ACT checkpoint on recorded frames (LeRobot 0.7 env).

Picks N frames from the recorded dataset (deterministic seed, spread over episodes, each with 100 future frames),
crops the top camera exactly as in training, runs LeRobot's own `preprocessor -> ACTPolicy.predict_action_chunk ->
postprocessor`, and saves everything the OpenVINO and Studio parity checks need into one .npz:

    state (N,6) f32 · <cam> (N,480,640,3) u8 full camera frames · ref (N,100,6) f32 predicted chunks
    future (N,100,6) f32 recorded actions · valid (N,100) bool · picks (N,2) episode/frame

Frames are fed as uint8 (dataset float pixels x255, which is exact for pyav-decoded video), the same bytes the
robot-side runtimes receive. The script also reports MAE of the predicted chunk against the recorded future actions
(first 10 and 100 steps), next to a "hold the current position" baseline, as a sanity check of policy+preprocessing.

    nice -n 5 ~/inference/.venv/bin/python deploy/act/torch_reference.py \
        --policy_path ~/inference/models/act-warp-duplo-417-20261114 \
        --dataset ~/desmond/dlsj/data/duplo_teleop_rgb --out ~/inference/exports/act_warp_417_ref.npz
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for dlsj.deploy.act_studio

from dlsj.deploy.act_studio import Crop, chunk_mae, sample_frames  # noqa: E402

CAMS = ("observation.images.wrist", "observation.images.top", "observation.images.inspect")
TOP = "observation.images.top"


def load_policy(path: str, device: str):
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    config = ACTConfig.from_pretrained(path)
    config.device = device
    policy = ACTPolicy.from_pretrained(path, config=config).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=path,
                                         preprocessor_overrides={"device_processor": {"device": device}})
    return policy, pre, post


def reference_chunk(policy, pre, post, state: np.ndarray, frames: dict[str, np.ndarray], crop: Crop,
                    device: str) -> np.ndarray:
    """LeRobot's own inference path on raw robot data: uint8 HWC frames, top cropped, state in degrees."""
    from lerobot.policies.utils import prepare_observation_for_inference

    obs = {"observation.state": state.astype(np.float32)}
    for k in CAMS:
        obs[k] = np.ascontiguousarray(crop.apply_hwc(frames[k]) if k == TOP else frames[k])
    batch = prepare_observation_for_inference(obs, torch.device(device))
    with torch.no_grad():
        actions = policy.predict_action_chunk(pre(batch))
    return post(actions).float().cpu().numpy()[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy_path", required=True)
    ap.add_argument("--dataset", required=True, help="LeRobot v3 dataset root (read-only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_episode", type=int, default=417, help="only episodes below this index (trained set)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    policy, pre, post = load_policy(args.policy_path, args.device)
    meta_json = json.loads((Path(args.policy_path) / "checkpoint-metadata.json").read_text())
    crop = Crop.from_metadata(meta_json["top_camera_crop"])
    chunk = policy.config.chunk_size

    root = Path(args.dataset)
    meta = LeRobotDatasetMetadata(root.name, root=root)
    eps = meta.episodes
    lengths = {int(e): int(n) for e, n in zip(eps["episode_index"], eps["length"], strict=True)
               if int(e) < args.max_episode}
    picks = sample_frames(lengths, args.n, seed=args.seed, horizon=chunk)
    episodes = sorted({e for e, _ in picks})
    ds = LeRobotDataset(root.name, root=root, episodes=episodes, video_backend="pyav")
    missing = [k for k in CAMS if k not in ds.meta.features]
    if missing:
        raise SystemExit(f"dataset lacks {missing}; has {list(ds.meta.features)}")

    # global index of each episode's first frame, from the (filtered) episode metadata
    starts = {int(e): int(s) for e, s in zip(ds.meta.episodes["episode_index"],
                                             ds.meta.episodes["dataset_from_index"], strict=True)}
    hf = ds.hf_dataset
    idx_col = np.asarray(hf["index"])
    row_of = {int(g): i for i, g in enumerate(idx_col)}

    n = len(picks)
    out = {k: np.zeros((n, 480, 640, 3), np.uint8) for k in CAMS}
    state = np.zeros((n, 6), np.float32)
    ref = np.zeros((n, chunk, 6), np.float32)
    future = np.zeros((n, chunk, 6), np.float32)
    valid = np.zeros((n, chunk), bool)
    times = []
    for i, (ep, f) in enumerate(picks):
        g = starts[ep] + f
        item = ds[row_of[g]]
        assert int(item["episode_index"]) == ep and int(item["frame_index"]) == f
        state[i] = item["observation.state"].numpy()
        for k in CAMS:
            x = item[k].numpy()  # (3,H,W) float [0,1]
            u8 = np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)
            assert np.abs(u8.astype(np.float32) / 255.0 - x).max() < 1e-6, "pixels are not uint8/255"
            out[k][i] = u8.transpose(1, 2, 0)
        rows = [row_of[g + t] for t in range(chunk) if g + t - starts[ep] < lengths[ep] and (g + t) in row_of]
        acts = np.stack([np.asarray(hf[r]["action"], np.float32) for r in rows])
        future[i, :len(rows)] = acts
        valid[i, :len(rows)] = True
        t0 = time.perf_counter()
        ref[i] = reference_chunk(policy, pre, post, state[i], {k: out[k][i] for k in CAMS}, crop, args.device)
        times.append(time.perf_counter() - t0)

    hold = np.repeat(state[:, None, :], chunk, axis=1)  # "stay where you are" baseline
    report = {
        "frames": n, "episodes": len(episodes), "device": args.device,
        "torch_ms_median": round(1e3 * float(np.median(times[1:])), 1),
        "mae_first10": chunk_mae(ref, future, valid, 10).round(2).tolist(),
        "mae_first100": chunk_mae(ref, future, valid, chunk).round(2).tolist(),
        "hold_baseline_mae_first10": chunk_mae(hold, future, valid, 10).round(2).tolist(),
        "hold_baseline_mae_first100": chunk_mae(hold, future, valid, chunk).round(2).tolist(),
        "action_std": [round(float(s), 2) for s in post.steps[0].stats["action"]["std"]]
        if hasattr(post.steps[0], "stats") else None,
    }
    np.savez_compressed(args.out, picks=np.asarray(picks), state=state, ref=ref, future=future, valid=valid,
                        **{k.split(".")[-1]: v for k, v in out.items()})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
