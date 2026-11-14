"""Offline checks for the MolmoAct2 pipeline (no GPU, no Studio install needed).

    python -m unittest discover -s training/molmoact2 -p 'test_*.py'
Tests needing torch/pyarrow are skipped when those aren't installed.
"""
import importlib
import json
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prepare_data  # noqa: E402
import train  # noqa: E402


def has(module):
    try:
        importlib.import_module(module)
        return True
    except ImportError:
        return False


class Recipe(unittest.TestCase):
    def config(self, **kw):
        return train.build_config(dataset="/data/snap", output="/out/run", gpus=4, **kw)

    def test_ai2_lora_recipe(self):
        m = self.config()["model"]["init_args"]
        self.assertEqual(m["pretrained_name_or_path"], "allenai/MolmoAct2-SO100_101")
        self.assertTrue(m["lora_enabled"])
        self.assertEqual(m["lora_rank"], 64)
        self.assertNotIn("lora_target_modules", m)  # Studio default targets (LLM+ViT+connector+action expert)
        for key in ("optimizer_lr", "optimizer_vit_lr", "optimizer_connector_lr", "optimizer_action_expert_lr"):
            self.assertEqual(m[key], 5e-5, key)
        self.assertEqual((m["chunk_size"], m["n_action_steps"]), (30, 30))
        self.assertEqual(m["setup_type"], "single so100/so101 robotic arm in molmoact2")
        self.assertEqual(m["control_mode"], "absolute joint pose")
        self.assertTrue(m["adapt_to_so101"])
        self.assertNotIn("calibration", m)  # data is in degrees already: Studio scale must stay 1
        self.assertIsNone(m["scheduler_decay_steps"])  # cosine over the actual run

    def test_batch_data_and_augmentation(self):
        c = self.config()
        d = c["data"]["init_args"]
        self.assertEqual(c["data"]["class_path"], "phase_data.PhaseDataModule")
        self.assertEqual(d["train_batch_size"] * c["trainer"]["devices"], 32)
        self.assertEqual((d["val_split"], d["val_split_seed"]), (0.1, 42))
        self.assertEqual(d["image_transforms"]["class_path"], "augment.Ai2Augment")
        self.assertEqual(c["trainer"]["max_steps"], 8000)
        paths = [cb["class_path"] for cb in c["trainer"]["callbacks"]]
        self.assertIn("progress.Progress", paths)

    def test_batch_must_divide(self):
        with self.assertRaises(ValueError):
            train.build_config(dataset="/d", output="/o", gpus=8, global_batch=20)

    def test_smoke_has_no_checkpoints(self):
        t = self.config(smoke=True)["trainer"]
        self.assertEqual((t["fast_dev_run"], t["callbacks"], t["enable_checkpointing"]), (1, [], False))

    def test_versions_pinned(self):
        v = train.versions()
        self.assertRegex(v["STUDIO_COMMIT"], r"^[0-9a-f]{40}$")
        self.assertRegex(v["MOLMOACT2_REVISION"], r"^[0-9a-f]{40}$")
        self.assertEqual(v["MOLMOACT2_REPO"], train.RECIPE["pretrained_name_or_path"])
        pi05 = (HERE.parent / "pi05/versions.env").read_text()
        self.assertIn(v["STUDIO_COMMIT"], pi05)  # same Studio as the pi0.5 runs / XPS export env


class Data(unittest.TestCase):
    def test_studio_split(self):
        train_ids, val_ids = prepare_data.studio_split(100)
        expected = sorted(random.Random(42).sample(list(range(100)), 10))  # Studio datamodule, pinned commit
        self.assertEqual(val_ids, expected)
        self.assertEqual(sorted(train_ids + val_ids), list(range(100)))
        self.assertEqual(len(train_ids), 90)

    def labels(self, n=3):
        return [{"episode_index": i, "success": True, "pick_task": prepare_data.PICK,
                 "place_task": "place the brick in the purple bowl", "inspection_frame": 50, "n_frames": 100}
                for i in range(n)]

    def info(self, top=(480, 640, 3)):
        joint = {"shape": [6], "names": prepare_data.JOINTS}
        return {"codebase_version": "v3.0", "fps": 30, "total_episodes": 3, "total_frames": 300,
                "features": {"observation.state": joint, "action": joint,
                             "observation.images.wrist": {"shape": [480, 640, 3]},
                             "observation.images.top": {"shape": list(top)},
                             "observation.images.inspect": {"shape": [480, 640, 3]}}}

    calibration = {j: {"range_min": 900, "range_max": 3100}
                   for j in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")}

    def test_source_ok(self):
        prepare_data.check_source(self.info(), self.labels(), self.calibration)

    def test_rejects_failed_take_and_bad_boundary(self):
        labels = self.labels()
        labels[1]["success"] = False
        labels[2]["inspection_frame"] = 100
        with self.assertRaisesRegex(ValueError, r"\[1, 2\]"):
            prepare_data.check_source(self.info(), labels, self.calibration)

    def test_rejects_already_cropped_top(self):
        with self.assertRaisesRegex(ValueError, "uncropped"):
            prepare_data.check_source(self.info(top=(480, 560, 3)), self.labels(), self.calibration)

    def test_rejects_incomplete_calibration(self):
        with self.assertRaisesRegex(ValueError, "calibration"):
            prepare_data.check_source(self.info(), self.labels(), {"shoulder_pan": {"range_min": 1}})

    @unittest.skipUnless(has("pyarrow") and has("numpy"), "pyarrow/numpy not installed")
    def test_train_normalization_uses_train_episodes_only(self):
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data/chunk-000").mkdir(parents=True)
            ep = [0] * 4 + [1] * 4
            state = [[float(e * 100 + i)] * 6 for e, i in zip(ep, range(8))]
            pq.write_table(pa.table({"episode_index": ep, "observation.state": state, "action": state}),
                           root / "data/chunk-000/file-000.parquet")
            out = prepare_data.train_normalization(root, [0])
            self.assertEqual(out["observation.state"]["max"][0], 3.0)  # episode 1 (values >= 104) excluded
            self.assertEqual(out["action"]["count"], [4])
            self.assertAlmostEqual(out["action"]["q99"][0], float(np.quantile([0, 1, 2, 3], 0.99)))


@unittest.skipUnless(has("torch"), "torch not installed")
class Phase(unittest.TestCase):
    def setUp(self):
        # phase_data imports Studio at module level; stub just enough to import the pure function.
        stub = types.ModuleType("physicalai")
        for name in ("physicalai.data", "physicalai.data.lerobot", "physicalai.data.lerobot.dataset"):
            sys.modules.setdefault(name, types.ModuleType(name))
        sys.modules.setdefault("physicalai", stub)
        sys.modules["physicalai.data.lerobot"].LeRobotDataModule = object
        sys.modules["physicalai.data.lerobot.dataset"]._LeRobotDatasetAdapter = object
        self.phase = importlib.import_module("phase_data")

    def test_pick_chunk_stops_at_inspection(self):
        import torch
        action = torch.arange(30, dtype=torch.float32)[:, None].repeat(1, 6)
        held = self.phase.hold_at_boundary(action, frame=95, boundary=100)  # 5 pick steps left
        self.assertTrue(torch.equal(held[:5], action[:5]))
        self.assertTrue(torch.all(held[5:] == action[4]))
        self.assertTrue(torch.equal(self.phase.hold_at_boundary(action, 100, 100), action))  # place phase
        self.assertTrue(torch.equal(self.phase.hold_at_boundary(action, 10, 100), action))   # far from boundary


@unittest.skipUnless(has("torchvision"), "torchvision not installed")
class Augment(unittest.TestCase):
    def test_shapes_and_range_preserved(self):
        import torch
        from augment import Ai2Augment
        aug = Ai2Augment()
        for h, w in ((480, 560), (480, 640)):
            out = aug(torch.rand(3, h, w))
            self.assertEqual(tuple(out.shape), (3, h, w))
            self.assertGreaterEqual(float(out.min()), 0.0)
            self.assertLessEqual(float(out.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
