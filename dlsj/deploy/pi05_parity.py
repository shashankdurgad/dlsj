"""OpenVINO export vs PyTorch checkpoint parity for π0.5, on recorded frames with identical start noise.

    python -m dlsj.deploy.pi05_parity eval.npz --ckpt best.ckpt --export exports/pi05_ft_fp16 exports/pi05_ft_int8 \
        --n 12 --device CPU

Reports, per export, the mean and max absolute difference of the action chunk against PyTorch fp32 (robot units),
and the same for two PyTorch calls with different noise, which is the scale the policy itself varies by.
"""

import argparse
import json

import numpy as np

CAMS = ("wrist", "top", "inspect")


def main() -> None:
    from dlsj.deploy.pi05_runtime import Pi05Policy
    from dlsj.deploy.pi05_torch import TorchPi05Policy

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--export", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--device", default="CPU")
    args = ap.parse_args()

    d = np.load(args.npz)
    idx = np.linspace(0, len(d["episode"]) - 1, args.n).astype(int)
    rng = np.random.default_rng(0)
    noises = rng.standard_normal((len(idx), 50, 32)).astype(np.float32)
    torch_pol = TorchPi05Policy(args.ckpt)
    ref, ref2 = [], []
    for j, i in enumerate(idx):
        imgs, state, task = {c: d[c][i] for c in CAMS}, d["state"][i], str(d["task"][i])
        ref.append(torch_pol(state, imgs, task, noise=noises[j]))
        ref2.append(torch_pol(state, imgs, task, noise=noises[(j + 1) % len(idx)]))
    ref, ref2 = np.array(ref), np.array(ref2)
    out = {"samples": len(idx), "torch_noise_to_noise": {"mean_abs": round(float(np.abs(ref - ref2).mean()), 3),
                                                         "max_abs": round(float(np.abs(ref - ref2).max()), 3)}}
    del torch_pol
    for exp in args.export:
        pol = Pi05Policy(exp, args.device, cache_dir=None)
        got = []
        for j, i in enumerate(idx):
            obs = {"state": np.asarray(d["state"][i], np.float32)[None], "task": [str(d["task"][i])],
                   "noise": noises[j][None]}
            for c in CAMS:
                obs[f"images.{c}"] = np.ascontiguousarray(d[c][i].transpose(2, 0, 1)[None])
            got.append(pol.model.predict_action_chunk(obs))
        diff = np.abs(np.array(got) - ref)
        out[exp] = {"mean_abs": round(float(diff.mean()), 3), "max_abs": round(float(diff.max()), 3),
                    "per_joint_mean": np.round(diff.mean((0, 1)), 3).tolist()}
        del pol
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
