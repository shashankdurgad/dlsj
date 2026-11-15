"""Cost of the batched occlusion proxy + sim detector on the GPU (no Isaac, no PyAV).

sim/twin/run.sh sim/twin/perception/bench.py [--envs 512 2048]     (the twin image's torch; or any torch python)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def main():
    import torch

    from sim.twin.perception import geometry as G
    from sim.twin.perception.detector_model import DetectorModel

    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, nargs="*", default=[512, 2048])
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    for n in a.envs:
        geo = G.Geometry(torch_device=a.device)
        j = torch.tensor(
            np.array([18.0, 4.0, 12.6, 73.3, 63.3, 24.8]) + rng.normal(0, 3, (n, 6)),
            dtype=torch.float32,
            device=a.device,
        )
        p, _ = geo.tip(j)
        pos = p.clone()
        pos[:, 2] = G.TABLE_Z + 0.0096
        pos[:, :2] += torch.randn(n, 2, device=a.device) * 0.02
        R = G.yaw_R(torch.rand(n, device=a.device) * 3.14, xp=torch)
        size = torch.tensor([0.0318, 0.0318, 0.0192], device=a.device).expand(n, 3)
        oth = (
            pos[:, None] + torch.tensor([0.05, 0.0, 0.0], device=a.device),
            R[:, None],
            size[:, None],
            torch.ones(n, 1, dtype=torch.bool, device=a.device),
        )
        m = DetectorModel(n, geo=geo)
        g = torch.Generator(device=a.device).manual_seed(0)
        m.reset(None, g)
        for k in range(13):
            if k == 3:
                if a.device.startswith("cuda"):
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
            v = geo.visibility(j, pos, R, size, oth)
            o = m.observe(j, v, g)
        if a.device.startswith("cuda"):
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 10 * 1e3
        mem = torch.cuda.max_memory_allocated() / 2**20 if a.device.startswith("cuda") else 0
        print(
            f"BENCH n={n} {a.device}: visibility + observe {ms:.1f} ms / frame, peak {mem:.0f} MiB, "
            f"detected {o['detected'].float().mean().item():.2f}, vis {v['vis'].mean().item():.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
