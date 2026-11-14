"""`lerobot-train` with WARP-BC weights: same CLI, plus a "warp" sample weighter and per-sample ACT loss.

    python -m dlsj.warp.train_policy \
        --dataset.repo_id=local/duplo_teleop --dataset.root=~/desmond/data/duplo_teleop \
        --policy.type=act --policy.chunk_size=100 --policy.push_to_hub=false \
        --sample_weighting.type=warp --sample_weighting.progress_path=~/desmond/warp/weights/warp_weights.parquet \
        --output_dir=~/desmond/runs/act_warp --steps=100000 --batch_size=16

Without `--sample_weighting.*` it is plain `lerobot-train` (the vanilla-BC baseline). The trainer computes
loss = sum(w_i * loss_i) / sum(w_i) per batch (LeRobot's RA-BC path). Zero-weight samples still cost compute,
so raise --batch_size by ~1/kept_fraction to keep the effective batch the paper's way (it removes them).
π0 / π0.5 / SmolVLA already support per-sample loss, so the same weights apply to them unchanged.
"""

import logging

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

import lerobot.utils.sample_weighting as sw
from lerobot.utils.sample_weighting import SampleWeighter


class WarpWeights(SampleWeighter):
    def __init__(self, path: str, device: torch.device):
        from pathlib import Path

        df = pd.read_parquet(Path(path).expanduser(), columns=["index", "weight"])
        self.w = np.zeros(int(df["index"].max()) + 1, dtype=np.float32)
        self.w[df["index"].to_numpy()] = df["weight"].to_numpy(np.float32)
        self.device = device
        self.kept = float((df["weight"] > 0).mean())
        logging.info("WARP weights: %d frames, %.1f%% kept", len(df), 100 * self.kept)

    def compute_batch_weights(self, batch: dict):
        idx = batch["index"].cpu().numpy().astype(np.int64)
        w = torch.from_numpy(self.w[np.clip(idx, 0, len(self.w) - 1)]).to(self.device)
        stats = {"mean_weight": float(w.mean()), "kept_in_batch": float((w > 0).float().mean())}
        if w.sum() > 0:
            w = w * len(w) / w.sum()
        return w, stats

    def get_stats(self) -> dict:
        return {"dataset_kept_fraction": self.kept}


_make = sw.make_sample_weighter


def make_sample_weighter(config, policy, device, dataset_root=None, dataset_repo_id=None):
    if config is not None and config.type == "warp":
        return WarpWeights(config.progress_path, device)
    return _make(config, policy, device, dataset_root=dataset_root, dataset_repo_id=dataset_repo_id)


def act_forward(self, batch, reduction: str = "mean"):
    """ACTPolicy.forward with reduction="none" -> per-sample loss [B] (L1 over valid steps + KL)."""
    if reduction == "mean":
        return _act_forward(self, batch)
    from lerobot.utils.constants import ACTION, OBS_IMAGES

    if self.config.image_features:
        batch = dict(batch)
        batch[OBS_IMAGES] = [batch[k] for k in self.config.image_features]
    actions_hat, (mu, log_sigma_x2) = self.model(batch)
    err = F.l1_loss(batch[ACTION], actions_hat, reduction="none")
    valid = (~batch["action_is_pad"]).unsqueeze(-1).float()
    l1 = (err * valid).sum((1, 2)) / (valid.sum((1, 2)) * err.shape[-1]).clamp_min(1)
    loss = l1
    out = {"l1_loss": l1.mean().item()}
    if self.config.use_vae and log_sigma_x2 is not None:
        kld = (-0.5 * (1 + log_sigma_x2 - mu.pow(2) - log_sigma_x2.exp())).sum(-1)
        out["kld_loss"] = kld.mean().item()
        loss = l1 + kld * self.config.kl_weight
    return loss, out


def install() -> None:
    global _act_forward
    from lerobot.policies.act.modeling_act import ACTPolicy

    sw.make_sample_weighter = make_sample_weighter
    if not getattr(ACTPolicy.forward, "_warp", False):
        _act_forward = ACTPolicy.forward
        act_forward._warp = True
        ACTPolicy.forward = act_forward


_act_forward = None


def main() -> None:
    install()
    from lerobot.scripts import lerobot_train

    lerobot_train.main()


if __name__ == "__main__":
    main()
