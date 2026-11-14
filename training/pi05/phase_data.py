"""Pinned-Studio adapter: preserve episode holdout and stop pick chunks at the mark."""
import json
from pathlib import Path
import numpy as np
from physicalai.data.lerobot import LeRobotDataModule
from physicalai.data.lerobot.dataset import _LeRobotDatasetAdapter


def hold_at_boundary(action, frame, split):
    """Repeat the final phase target instead of supervising the next stage with the current text."""
    boundaries = (split,) if isinstance(split, int) else tuple(split)
    for b in sorted(boundaries):
        if frame < b:
            remaining = b - frame
            if action.ndim == 2 and 0 < remaining < action.shape[0]:
                action = action.clone()
                action[remaining:] = action[remaining - 1].clone()
            break
    return action


class PhaseAdapter(_LeRobotDatasetAdapter):
    def __getitem__(self, idx):
        observation = super().__getitem__(idx)
        ep = int(observation.episode_index)
        frame = int(observation.frame_index)
        observation.action = hold_at_boundary(observation.action, frame, self.phase_boundaries[ep])
        return observation


class PhaseDataModule(LeRobotDataModule):
    @staticmethod
    def _ensure_quantile_stats(train_dataset, val_dataset, *, root, **kwargs):
        root = Path(root)
        normalization = json.loads((root / 'TRAIN-NORMALIZATION.json').read_text())
        labels = [json.loads(l) for l in (root / 'meta/dlsj/labels.jsonl').read_text().splitlines() if l.strip()]
        boundaries = {
            l['episode_index']: (l['grasp_frame'], l['inspection_frame']) if 'grasp_frame' in l else l['inspection_frame']
            for l in labels
        }
        for adapter in (train_dataset, val_dataset):
            if adapter is None:
                continue
            # These are the exact private adapter types at the recorded Studio commit.
            assert type(adapter) is _LeRobotDatasetAdapter
            adapter.__class__ = PhaseAdapter
            adapter.phase_boundaries = boundaries
            for key, stats in normalization.items():
                adapter._lerobot_dataset.meta.stats[key] = {k: np.asarray(v) for k, v in stats.items()}
