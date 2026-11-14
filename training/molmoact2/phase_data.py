"""Studio data module for the MolmoAct2 run (same approach as training/pi05/phase_data.py).

1. Action chunks stop at the next stage boundary (3-stage labels: pick < grasp_frame, bring-to-inspection <
   inspection_frame, place): the remaining slots repeat the last target of the current stage, so the next stage's
   motion is never supervised with the current instruction. Same rule as the team's 3-stage phase_data.py.
2. State/action normalisation comes from TRAIN-NORMALIZATION.json (train episodes only, written by
   prepare_data.py with Studio's own seed-42 split), instead of Studio computing it over all episodes.

3. An explicit validation episode list (the team's CURATION split). Studio only offers a random split
   (`sorted(random.Random(seed).sample(episodes, n_val))`), so for the duration of Studio's __init__ the datamodule
   module's `random` is replaced by a shim whose `sample()` returns exactly our list; everything else (dataset
   construction for both splits, val loader) stays Studio's own code.

Relies on private classes of the pinned Studio commit (checked with asserts).
"""
from collections.abc import Callable
import json
from pathlib import Path

import numpy as np
from physicalai.data.lerobot import LeRobotDataModule
from physicalai.data.lerobot import datamodule as studio_datamodule
from physicalai.data.lerobot.dataset import _LeRobotDatasetAdapter


class FixedSplit:
    """Stands in for the `random` module inside Studio's datamodule: sample() returns the given episodes."""

    def __init__(self, validation):
        self.validation = sorted(validation)

    def Random(self, seed):  # noqa: N802 - mirrors random.Random
        return self

    def sample(self, population, k):
        if k != len(self.validation) or not set(self.validation) <= set(population):
            raise RuntimeError(f"explicit validation split does not fit Studio's split call (k={k})")
        return list(self.validation)


def val_fraction(n_val, n_all):
    """A val_split for which Studio's max(1, int(n_all * val_split)) == n_val."""
    fraction = (n_val + 0.5) / n_all
    assert max(1, int(n_all * fraction)) == n_val and 0 < fraction < 1
    return fraction


def hold_at_boundary(action, frame, boundaries):
    """Repeat the current stage's final target past its boundary (action: [chunk, dims])."""
    boundaries = (boundaries,) if isinstance(boundaries, int) else tuple(boundaries)
    for boundary in sorted(boundaries):
        if frame < boundary:
            remaining = boundary - frame
            if action.ndim == 2 and 0 < remaining < action.shape[0]:
                action = action.clone()
                action[remaining:] = action[remaining - 1].clone()
            break
    return action


class PhaseAdapter(_LeRobotDatasetAdapter):
    def __getitem__(self, idx):
        observation = super().__getitem__(idx)
        episode, frame = int(observation.episode_index), int(observation.frame_index)
        observation.action = hold_at_boundary(observation.action, frame, self.phase_boundaries[episode])
        return observation


class PhaseDataModule(LeRobotDataModule):
    # Explicit parameters only (no **kwargs): LightningCLI/jsonargparse follows **kwargs through the parent classes
    # and then passes the grandparent's `val_eval_dataset` to LeRobotDataModule, which rejects it.
    def __init__(
        self,
        *,
        repo_id: str,
        root: str,
        episodes: list[int],
        validation_episodes: list[int] | None = None,
        train_batch_size: int = 16,
        val_batch_size: int | None = None,
        num_workers: int = 4,
        image_transforms: Callable | None = None,
        video_backend: str | None = None,
        data_format: str = "physicalai",
        val_split: float = 0.0,
        val_split_seed: int | None = None,
    ) -> None:
        kwargs = dict(repo_id=repo_id, root=root, episodes=episodes, train_batch_size=train_batch_size,
                      val_batch_size=val_batch_size, num_workers=num_workers, image_transforms=image_transforms,
                      video_backend=video_backend, data_format=data_format, val_split=val_split,
                      val_split_seed=val_split_seed)
        if validation_episodes is None:
            super().__init__(**kwargs)
            return
        assert episodes and set(validation_episodes) < set(episodes), "validation episodes must be a subset"
        kwargs["val_split"] = val_fraction(len(validation_episodes), len(episodes))
        kwargs["val_split_seed"] = 0
        original = studio_datamodule.random
        studio_datamodule.random = FixedSplit(validation_episodes)
        try:
            super().__init__(**kwargs)
        finally:
            studio_datamodule.random = original
        val = getattr(self.val_eval_dataset._lerobot_dataset, "episodes", None)
        train = getattr(self.train_dataset._lerobot_dataset, "episodes", None)
        if val is not None and train is not None:
            assert sorted(val) == sorted(validation_episodes) and not set(train) & set(validation_episodes)

    @staticmethod
    def _ensure_quantile_stats(train_dataset, val_dataset, *, root, **kwargs):
        root = Path(root)
        normalization = json.loads((root / "TRAIN-NORMALIZATION.json").read_text())
        labels = [json.loads(line) for line in (root / "meta/dlsj/labels.jsonl").read_text().splitlines()
                  if line.strip()]
        boundaries = {label["episode_index"]: (label["grasp_frame"], label["inspection_frame"]) for label in labels
                      if isinstance(label.get("grasp_frame"), int)}
        for adapter in (train_dataset, val_dataset):
            if adapter is None:
                continue
            assert type(adapter) is _LeRobotDatasetAdapter, type(adapter)  # pinned Studio internals
            adapter.__class__ = PhaseAdapter
            adapter.phase_boundaries = boundaries
            for key in ("observation.state", "action"):
                adapter._lerobot_dataset.meta.stats[key] = {
                    k: np.asarray(v) for k, v in normalization[key].items() if k != "train_episodes"}
