"""MolmoAct2 (Physical AI Studio) as a frozen, batched base policy on a CUDA GPU (Joe's Spark, GB10, bf16).

Two sources, same call:
* zero-shot `allenai/MolmoAct2-SO100_101` (`--pretrained`): built exactly like deploy/molmoact2/export.py
  (adapt_to_so101 with a unit-scale calibration, so runtime units are LeRobot degrees, gripper 0-100; wrist + top);
* our LoRA fine-tune (`--ckpt <lightning .ckpt | *_slim.pt>`): `MolmoAct2.load_from_checkpoint`, which restores the
  training config (tokenizer_max_length 128, its features, normalisation and so101 frame); see load_ckpt for slim files.

Inputs follow the training contract: cameras wrist + top, top cropped to the training crop (cols 0..559 of 640, the
`top=0,0,560,480` crop used by training/molmoact2/prepare_data.py), RGB uint8 (H, W, 3) per env; state (6,) in our
degrees for the CURRENT calibration (cal2; the env's joe_cal2 profile); one instruction string per env.
`predict(state (n,6), wrist (n,H,W,3), top (n,H,W,3), tasks [n]) -> (n, 30, 6)` in the same units.
"""
from __future__ import annotations

import time

import numpy as np
import torch

UNIT_SCALE_CALIBRATION = {
    j: {"id": i + 1, "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 2275}
    for i, j in enumerate(["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"])
} | {"gripper": {"id": 6, "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 4095}}
TOP_CROP_W = 560  # training crop: x 0..559 of the 640-wide top image


def load_ckpt(ckpt: str):
    """Our fine-tune: a full Lightning checkpoint, or a SLIM one (only the tensors that differ from the base: LoRA,
    action expert, vision connector; top-level keys as Lightning plus a "slim" dict). For a slim file the policy is
    rebuilt from its stored hyperparameters/config (which loads the frozen base from the HF cache), then the slim
    tensors are loaded with strict=False; every key it does not carry must be a frozen base tensor."""
    import torch
    from physicalai.policies import MolmoAct2

    head = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
    if "slim" not in head:
        return MolmoAct2.load_from_checkpoint(ckpt, map_location="cpu", weights_only=False)
    policy = MolmoAct2.load_from_checkpoint(ckpt, map_location="cpu", weights_only=False, strict=False)
    model_keys, slim_keys = set(policy.state_dict()), set(head["state_dict"])
    unexpected = sorted(slim_keys - model_keys)
    missing = model_keys - slim_keys
    trained = sorted(k for k in missing if "lora_" in k or "action_expert" in k)
    if unexpected or trained:
        raise RuntimeError(f"slim checkpoint does not match the model: unexpected {unexpected[:5]} "
                           f"(+{max(0, len(unexpected) - 5)}), trained-but-missing {trained[:5]} (+{max(0, len(trained) - 5)})")
    print(f"SLIM_LOADED {len(slim_keys)} tensors from {ckpt}; {len(missing)} frozen tensors from the base", flush=True)
    return policy


def build(pretrained: str | None = "allenai/MolmoAct2-SO100_101", ckpt: str | None = None, h: int = 480,
          w_wrist: int = 640, w_top: int = TOP_CROP_W):
    from physicalai.data.observation import Feature, FeatureType
    from physicalai.policies import MolmoAct2

    if ckpt:
        return load_ckpt(ckpt)
    policy = MolmoAct2(pretrained_name_or_path=pretrained, norm_tag="so100_so101_molmoact2", adapt_to_so101=True,
                       calibration=UNIT_SCALE_CALIBRATION, use_random_input_noise=True, n_action_steps=30)
    cfg = policy.config
    state = next(f for f in cfg.input_features if f.ftype == FeatureType.STATE)
    action = next(f for f in cfg.output_features if f.ftype == FeatureType.ACTION)
    inputs = [Feature(ftype=FeatureType.VISUAL, shape=(3, h, w_wrist), name="wrist"),
              Feature(ftype=FeatureType.VISUAL, shape=(3, h, w_top), name="top"),
              Feature(ftype=FeatureType.STATE, shape=state.shape or (6,), name=state.name)]
    outputs = [Feature(ftype=FeatureType.ACTION, shape=action.shape or (6,), name=action.name)]
    policy.set_features(inputs, outputs, copy_state_normalization=True, copy_action_normalization=True)
    return policy


class MolmoBase:
    def __init__(self, pretrained: str | None = "allenai/MolmoAct2-SO100_101", ckpt: str | None = None,
                 device: str = "cuda:0", dtype: str = "bf16"):
        torch.set_grad_enabled(False)
        self.device, self.dtype = device, {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
        t = time.perf_counter()
        self.policy = build(pretrained, ckpt).to(device=device, dtype=self.dtype).eval()
        self.load_s = time.perf_counter() - t
        self.text_len = getattr(self.policy.config, "tokenizer_max_length", None)
        self.source = ckpt or pretrained

    def _img(self, a: np.ndarray, crop_w: int | None = None) -> torch.Tensor:
        a = np.asarray(a)
        if a.ndim == 3:
            a = a[None]
        if crop_w is not None and a.shape[2] > crop_w:
            a = a[:, :, :crop_w]
        t = torch.from_numpy(np.ascontiguousarray(a[..., :3])).to(self.device)
        return (t.permute(0, 3, 1, 2).float() / 255.0).to(self.dtype)

    def predict(self, state, wrist, top, tasks) -> np.ndarray:
        from physicalai.data.observation import Observation

        obs = Observation(state=torch.as_tensor(np.asarray(state, np.float32), device=self.device),
                          images={"wrist": self._img(wrist), "top": self._img(top, TOP_CROP_W)},
                          task=list(tasks))
        a = self.policy.predict_action_chunk(obs)
        torch.cuda.synchronize()
        a = a.float().cpu().numpy()
        return a if a.ndim == 3 else a[None]


if __name__ == "__main__":  # GPU smoke test + batch timings
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", default="allenai/MolmoAct2-SO100_101")
    ap.add_argument("--ckpt")
    ap.add_argument("--bs", default="1,4,8,16")
    ap.add_argument("--iters", type=int, default=5)
    a = ap.parse_args()
    base = MolmoBase(a.pretrained, a.ckpt)
    res = {"load_s": round(base.load_s, 1), "text_len": base.text_len, "source": base.source}
    rng = np.random.default_rng(0)
    state0 = np.array([-2.0, -59.0, 56.0, 60.0, 17.5, 29.0], np.float32)
    for bs in [int(b) for b in a.bs.split(",")]:
        st = np.repeat(state0[None], bs, 0)
        wr = rng.integers(0, 255, (bs, 480, 640, 3), dtype=np.uint8)
        tp = rng.integers(0, 255, (bs, 480, 640, 3), dtype=np.uint8)
        out = base.predict(st, wr, tp, ["pick up the brick"] * bs)  # warm-up
        ts = []
        for _ in range(a.iters):
            t = time.perf_counter()
            out = base.predict(st, wr, tp, ["pick up the brick"] * bs)
            ts.append(time.perf_counter() - t)
        res[f"bs{bs}"] = {"shape": list(out.shape), "finite": bool(np.isfinite(out).all()),
                          "median_s": round(float(np.median(ts)), 3),
                          "per_env_ms": round(1e3 * float(np.median(ts)) / bs, 1),
                          "first_minus_state": np.round(out[0, 0] - state0, 1).tolist()}
        print(json.dumps({f"bs{bs}": res[f"bs{bs}"]}), flush=True)
    res["peak_gpu_mb"] = round(torch.cuda.max_memory_allocated() / 1048576)
    print("MOLMO_GPU_RESULT", json.dumps(res))
