"""Offline checks for the SmolVLA pipeline (no GPU, no LeRobot needed except where noted).

    python3 -m unittest discover -s training/smolvla -p 'test_*.py'
"""
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prepare_data  # noqa: E402
import train  # noqa: E402

JOINT_FEATURE = {"dtype": "float32", "shape": [6], "names": train.JOINTS}


def info(cameras, top_shape=(480, 560, 3)):
    feats = {"observation.state": JOINT_FEATURE, "action": JOINT_FEATURE}
    for c in cameras:
        shape = list(top_shape) if c == "top" else [480, 640, 3]
        feats[f"observation.images.{c}"] = {"dtype": "video", "shape": shape}
    return {"codebase_version": "v3.0", "fps": 30, "total_episodes": 10, "total_frames": 1000, "features": feats}


class TrainLauncher(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "train"
        (self.root / "meta").mkdir(parents=True)
        stats = {k: {"mean": [0] * 6, "std": [1] * 6} for k in ("observation.state", "action")}
        (self.root / "meta/stats.json").write_text(json.dumps(stats))

    def tearDown(self):
        self.tmp.cleanup()

    def write_info(self, cameras):
        (self.root / "meta/info.json").write_text(json.dumps(info(cameras)))

    def test_official_command(self):
        cmd = train.build_command(dataset="/d/train", output="/o/run", python="py")
        self.assertEqual(cmd[:3], ["py", "-m", "lerobot.scripts.lerobot_train"])
        self.assertIn("--policy.path=lerobot/smolvla_base", cmd)
        self.assertIn("--batch_size=64", cmd)
        self.assertIn("--steps=20000", cmd)
        self.assertIn("--save_freq=2000", cmd)
        rename = json.loads(next(c for c in cmd if c.startswith("--rename_map=")).split("=", 1)[1])
        self.assertEqual(rename, {"observation.images.top": "observation.images.camera1",
                                  "observation.images.wrist": "observation.images.camera2"})
        # The official recipe is the checkpoint's own config: no optimizer/schedule overrides.
        self.assertFalse([c for c in cmd if c.startswith(("--policy.optimizer", "--policy.scheduler",
                                                          "--policy.train_expert_only", "--policy.freeze"))])

    def test_resume_command(self):
        cmd = train.build_command(dataset="/d", output="/o/run", resume=True, python="py")
        self.assertEqual(cmd[3:], ["--config_path=/o/run/checkpoints/last/pretrained_model/train_config.json",
                                   "--resume=true"])

    def test_dataset_must_be_two_camera(self):
        self.write_info(("wrist", "top"))
        train.check_dataset(self.root)
        self.write_info(SOURCE_CAMERAS)
        with self.assertRaisesRegex(ValueError, "prepare_data"):
            train.check_dataset(self.root)

    def test_recipe_matches_pinned_base_config(self):
        cached = list(Path.home().glob(".cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/"
                                       f"{train.versions()['SMOLVLA_REVISION']}/config.json"))
        if not cached:
            self.skipTest("pinned smolvla_base not in the local HF cache")
        train.check_base_config(json.loads(cached[0].read_text()))

    def test_recipe_drift_is_rejected(self):
        config = dict(train.RECIPE, optimizer_lr=5e-5)
        with self.assertRaisesRegex(ValueError, "optimizer_lr"):
            train.check_base_config(config)

    def test_versions_are_pinned_shas(self):
        v = train.versions()
        for key in ("SMOLVLA_REVISION", "SMOLVLM_REVISION"):
            self.assertRegex(v[key], r"^[0-9a-f]{40}$")
        self.assertIn(v["SMOLVLA_REVISION"][:8], v["BASE_CACHE_URI"])
        self.assertIn(v["SMOLVLM_REVISION"][:8], v["BASE_CACHE_URI"])


SOURCE_CAMERAS = ("wrist", "top", "inspect", "inspect_depth", "top_depth")


class PrepareData(unittest.TestCase):
    curation = {"train_episode_ids": [0, 1, 2, 3], "validation_episode_ids": [5, 6]}

    def test_accepts_three_stage_five_stream_source(self):
        train_ids, val_ids = prepare_data.check_source(info(SOURCE_CAMERAS), self.curation)
        self.assertEqual((train_ids, val_ids), ([0, 1, 2, 3], [5, 6]))

    def test_rejects_old_three_camera_dataset(self):
        with self.assertRaisesRegex(ValueError, "expected cameras"):
            prepare_data.check_source(info(("wrist", "top", "inspect")), self.curation)

    def test_rejects_uncropped_top(self):
        with self.assertRaisesRegex(ValueError, "cropped"):
            prepare_data.check_source(info(SOURCE_CAMERAS, top_shape=(480, 640, 3)), self.curation)

    def test_rejects_overlapping_splits(self):
        bad = {"train_episode_ids": [0, 1], "validation_episode_ids": [1]}
        with self.assertRaisesRegex(ValueError, "disjoint"):
            prepare_data.check_source(info(SOURCE_CAMERAS), bad)

    def test_drop_stale_episode_columns(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("pyarrow not installed")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "meta/episodes/chunk-000/file-000.parquet"
            path.parent.mkdir(parents=True)
            cols = {"episode_index": [0], "stats/inspect.prob/min": [0.0],
                    "stats/observation.images.inspect/min": [[0.0, 0.0, 0.0]],
                    "videos/observation.images.inspect/chunk_index": [0],
                    "stats/observation.images.top/min": [[0.0, 0.0, 0.0]]}
            pq.write_table(pa.table(cols), path)
            self.assertEqual(prepare_data.drop_stale_episode_columns(tmp, prepare_data.DROP[0]), 2)
            self.assertEqual(pq.read_table(path).column_names,
                             ["episode_index", "stats/inspect.prob/min", "stats/observation.images.top/min"])


if __name__ == "__main__":
    unittest.main()
