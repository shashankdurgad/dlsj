"""Unit tests for the MolmoAct2 client conventions (no model, no GPU).

    python -m unittest sim.twin.policies.test_molmoact2
"""

from __future__ import annotations

import io
import unittest

import numpy as np

from sim.twin.policies import molmoact2 as m


class TestConversions(unittest.TestCase):
    def test_round_trip(self):
        rng = np.random.default_rng(0)
        x = rng.uniform([-110, -100, -100, -100, -160, 0], [110, 100, 100, 100, 160, 100], (1000, 6)).astype(np.float32)
        np.testing.assert_allclose(m.from_ckpt(m.to_ckpt(x)), x, atol=1e-4)
        np.testing.assert_allclose(m.to_ckpt(m.from_ckpt(x)), x, atol=1e-4)

    def test_known_values(self):
        # Studio constants: shoulder_lift flipped, lift/elbow +90, gripper untouched
        r = np.array([10, 20, 30, 40, 50, 60], np.float32)
        np.testing.assert_allclose(m.to_ckpt(r), [10, 70, 120, 40, 50, 60])
        np.testing.assert_allclose(m.from_ckpt([0, 90, 90, 0, 0, 0]), [0, 0, 0, 0, 0, 0], atol=1e-6)
        np.testing.assert_allclose(m.from_ckpt([5, 0, 0, 0, 0, 50]), [5, 90, -90, 0, 0, 50])

    def test_matches_studio_constants(self):
        try:
            from physicalai.policies.molmoact2.constants import SO101_JOINT_OFFSETS, SO101_JOINT_SIGNS
            from physicalai.policies.molmoact2.so101 import so101_joint_scales
        except ImportError:
            self.skipTest("physicalai not importable here")
        np.testing.assert_array_equal(m.S, SO101_JOINT_SIGNS)
        np.testing.assert_array_equal(m.O, SO101_JOINT_OFFSETS)
        import sys

        sys.path.insert(0, str(m.REPO / "deploy/molmoact2"))
        from export import UNIT_SCALE_CALIBRATION

        np.testing.assert_allclose(so101_joint_scales(UNIT_SCALE_CALIBRATION), 1.0, atol=1e-12)

    def test_studio_transform_equals_ours(self):
        try:
            from physicalai.policies.molmoact2.so101 import make_so101_joint_transform
        except ImportError:
            self.skipTest("physicalai not importable here")
        import sys

        import torch

        sys.path.insert(0, str(m.REPO / "deploy/molmoact2"))
        from export import UNIT_SCALE_CALIBRATION

        tr = make_so101_joint_transform(UNIT_SCALE_CALIBRATION)  # what the worker's preprocessor applies
        x = np.random.default_rng(1).uniform(-90, 90, (50, 6)).astype(np.float32)
        ck = tr.forward(torch.from_numpy(x)).numpy()
        np.testing.assert_allclose(ck, m.to_ckpt(x), atol=1e-4)
        np.testing.assert_allclose(tr.inverse(torch.from_numpy(ck)).numpy(), m.from_ckpt(ck), atol=1e-4)
        np.testing.assert_allclose(m.from_ckpt(ck), x, atol=1e-4)


class TestClientPieces(unittest.TestCase):
    def test_ipc_round_trip(self):
        buf = io.BytesIO()
        msg = {"images": {"top": np.zeros((480, 640, 3), np.uint8)}, "state": np.arange(6, dtype=np.float32)}
        m._send(buf, msg)
        buf.seek(0)
        got = m._recv(buf)
        np.testing.assert_array_equal(got["state"], msg["state"])
        with self.assertRaises(EOFError):
            m._recv(buf)

    def test_camera_order(self):
        self.assertEqual(m.MolmoAct2Client._order(("top", "wrist"), False), ("wrist", "top"))
        self.assertEqual(m.MolmoAct2Client._order(("inspect", "top"), False), ("top", "inspect"))
        self.assertEqual(m.MolmoAct2Client._order(("top", "wrist"), True), ("top", "wrist"))
        self.assertEqual(m.MolmoAct2Client._order(m.DEPLOY["cams"], False), m.DEPLOY["cams"])


if __name__ == "__main__":
    unittest.main()
