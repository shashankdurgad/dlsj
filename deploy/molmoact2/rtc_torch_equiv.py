"""PyTorch (CPU) check that the RTC-patched MolmoAct2 flow loop equals the original when guidance ~ 0.

In OpenVINO the plain and RTC graphs each carry their own traced RandomNormal op, so their noise draws differ and the
outputs cannot be compared bit-for-bit. Here both loops get the same torch.Generator seed (and also the zero start).

    python rtc_torch_equiv.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from export import build  # noqa: E402
from rtc_patch import patch_policy  # noqa: E402


def main() -> None:
    torch.set_grad_enabled(False)
    policy = build("allenai/MolmoAct2-SO100_101", False, ["wrist", "top"], 480, 640)
    batch = policy._get_default_export_input_sample()
    model = policy.model
    orig = model.predict_action_chunk
    res = {}
    for noise in (True, False):
        g = torch.Generator().manual_seed(0)
        a = orig(batch, sample_noise=noise, generator=g)
        res[f"orig_noise{noise}"] = a
    rtc = patch_policy(policy)
    for noise in (True, False):
        g = torch.Generator().manual_seed(0)
        b = model.predict_action_chunk(batch | rtc, sample_noise=noise, generator=g)
        d = (b - res[f"orig_noise{noise}"]).abs()
        print(f"start={'noise' if noise else 'zeros'}: max|patched(g~0) - original| = {d.max().item():.3e}, "
              f"mean = {d.mean().item():.3e}")
    # guidance on, previous chunk = the original's own output (in runtime units it would differ; here we only check
    # that strong guidance pulls the prefix toward the given tail)
    g = torch.Generator().manual_seed(0)
    tail = torch.zeros_like(rtc["prev_chunk_left_over"])
    on = model.predict_action_chunk(batch | rtc | {"prev_chunk_left_over": tail, "inference_delay": torch.tensor(5),
                                                   "execution_horizon": torch.tensor(15),
                                                   "max_guidance_weight": torch.tensor(5.0)},
                                    sample_noise=True, generator=g)
    print("guided output differs from unguided (expected):", (on - res["orig_noiseTrue"]).abs().mean().item())


if __name__ == "__main__":
    main()
