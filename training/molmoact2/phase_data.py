"""Studio data module for the MolmoAct2 run (same approach as training/pi05/phase_data.py).

1. Pick-instruction action chunks stop at the inspection frame: the remaining slots repeat the last pick target,
   so placement motion is never supervised with the pick instruction.
2. State/action normalisation comes from TRAIN-NORMALIZATION.json (train episodes only, written by
   prepare_data.py with Studio's own seed-42 split), instead of Studio computing it over all episodes.

Relies on private classes of the pinned Studio commit (checked with an assert).
"""
import json
from pathlib import Path

import numpy as np
from physicalai.data.lerobot import LeRobotDataModule
from physicalai.data.lerobot.dataset import _LeRobotDatasetAdapter


def hold_at_boundary(action, frame, boundary):
    """Repeat the final pick target past the inspection frame (action: [chunk, dims])."""
    remaining = boundary - frame
    if frame < boundary and action.ndim == 2 and 0 < remaining < action.shape[0]:
        action = action.clone()
        action[remaining:] = action[remaining - 1].clone()
    return action


class PhaseAdapter(_LeRobotDatasetAdapter):
    def __getitem__(self, idx):
        observation = super().__getitem__(idx)
        episode, frame = int(observation.episode_index), int(observation.frame_index)
        observation.action = hold_at_boundary(observation.action, frame, self.phase_boundaries[episode])
        return observation


class PhaseDataModule(LeRobotDataModule):
    @staticmethod
    def _ensure_quantile_stats(train_dataset, val_dataset, *, root, **kwargs):
        root = Path(root)
        normalization = json.loads((root / "TRAIN-NORMALIZATION.json").read_text())
        labels = [json.loads(line) for line in (root / "meta/dlsj/labels.jsonl").read_text().splitlines()
                  if line.strip()]
        boundaries = {label["episode_index"]: label["inspection_frame"] for label in labels}
        for adapter in (train_dataset, val_dataset):
            if adapter is None:
                continue
            assert type(adapter) is _LeRobotDatasetAdapter, type(adapter)  # pinned Studio internals
            adapter.__class__ = PhaseAdapter
            adapter.phase_boundaries = boundaries
            for key in ("observation.state", "action"):
                adapter._lerobot_dataset.meta.stats[key] = {
                    k: np.asarray(v) for k, v in normalization[key].items() if k != "train_episodes"}
