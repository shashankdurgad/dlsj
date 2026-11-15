"""Deploy artefact of a trained grasp residual: the actor MLP only (no critics, no best-of-K), as a numpy .npz with a
dependency-free forward (`ResidualPolicy`) and optionally ONNX for OpenVINO. The deploy spec is in README.md
("Deploy artefact").

    python sim/twin/rl/export_residual.py sim/twin/out/rl/<run>/residual.pt [--onnx]      # -> <run>/deploy/
    from sim.twin.rl.export_residual import ResidualPolicy
    pol = ResidualPolicy("sim/twin/out/rl/<run>/deploy/residual_actor.npz")
    r = pol(state, plan)        # (6,) in [-1, 1]; state = obs (+ projected latent), plan = next-5-goal feature (30,)
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

_ERF = np.vectorize(math.erf, otypes=[np.float32])


class ResidualPolicy:
    """numpy forward of the DICE residual actor: [state, plan] -> Linear -> LayerNorm -> GELU (x hidden) -> Linear ->
    tanh. Matches torch (exact-erf GELU) to ~1e-6."""

    def __init__(self, path: str | Path):
        z = np.load(path, allow_pickle=False)
        self.meta = json.loads(str(z["meta"]))
        n = self.meta["layers"]
        self.W = [z[f"W{i}"] for i in range(n)]
        self.b = [z[f"b{i}"] for i in range(n)]
        self.g = [z[f"g{i}"] for i in range(n - 1)]
        self.beta = [z[f"beta{i}"] for i in range(n - 1)]

    def __call__(self, state: np.ndarray, plan: np.ndarray) -> np.ndarray:
        x = np.concatenate([np.asarray(state, np.float32).ravel(), np.asarray(plan, np.float32).ravel()])
        for i, (W, b) in enumerate(zip(self.W, self.b, strict=True)):
            x = x @ W + b
            if i < len(self.W) - 1:
                mu, var = x.mean(), x.var()
                x = (x - mu) / np.sqrt(var + 1e-5) * self.g[i] + self.beta[i]
                x = 0.5 * x * (1.0 + _ERF(x / math.sqrt(2.0)))
        return np.tanh(x).astype(np.float32)


def export(ckpt: str, onnx: bool = False) -> dict:
    import torch

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    missing = [k for k in ("res_scale", "rate", "tip_local") if k not in ck]
    if missing:
        raise SystemExit(f"{ckpt}: no {missing} in the checkpoint (trained before the deploy meta was saved): the "
                         "deployed residual could not reproduce the trained one; retrain or re-save with the meta")
    a = ck["actor"]
    n = len([k for k in a if k.startswith("W.")])
    out = Path(ckpt).parent / "deploy"
    out.mkdir(exist_ok=True)
    arr = {}
    for i in range(n):
        arr[f"W{i}"] = a[f"W.{i}"][0].numpy().astype(np.float32)
        arr[f"b{i}"] = a[f"b.{i}"][0, 0].numpy().astype(np.float32)
        if i < n - 1:
            arr[f"g{i}"] = a[f"g.{i}"][0, 0].numpy().astype(np.float32)
            arr[f"beta{i}"] = a[f"beta.{i}"][0, 0].numpy().astype(np.float32)
    meta = {"layers": n, "s_dim": ck["s_dim"], "a_dim": ck["a_dim"], "obs_dim": ck.get("obs_dim", 25),
            "latent": bool(ck.get("latent", "latent_dim" in ck)),
            "latent_dim": ck.get("latent_dim"), "proj_dim": ck.get("proj_dim"), "detector": ck.get("detector"),
            "top_target": bool(ck.get("top_target", False)), "inject": bool(ck.get("inject", False)),
            "source": str(ckpt), "residual_scale": list(ck["res_scale"]), "rate_per_tick": list(ck["rate"]),
            "tip_local": list(ck["tip_local"]), "joint_offsets": ck.get("joint_offsets"),
            "decision_ticks": int(ck.get("decision_ticks", 5)), "fade_ticks": int(ck.get("fade_ticks", 6)),
            "grasp_cfg": ck.get("grasp_cfg")}
    np.savez(out / "residual_actor.npz", meta=np.array(json.dumps(meta)), **arr)
    # parity + latency against the torch actor
    from sim.twin.rl.dice import Dice

    agent, _ = Dice.load(ckpt, "cpu")
    pol = ResidualPolicy(out / "residual_actor.npz")
    rng = np.random.default_rng(0)
    s = rng.standard_normal((32, ck["s_dim"])).astype(np.float32)
    p = rng.standard_normal((32, ck["a_dim"])).astype(np.float32) * 0.3
    with torch.no_grad():
        ref = agent.pi(torch.from_numpy(s), torch.from_numpy(p)).numpy()
    got = np.stack([pol(s[i], p[i]) for i in range(32)])
    t = time.perf_counter()
    for _ in range(500):
        pol(s[0], p[0])
    lat_ms = (time.perf_counter() - t) / 500 * 1e3
    rep = {"npz": str(out / "residual_actor.npz"), "max_abs_diff_vs_torch": float(np.abs(got - ref).max()),
           "numpy_ms_per_call": round(lat_ms, 4), **meta}
    if onnx:
        class _Actor(torch.nn.Module):
            def __init__(self, ag):
                super().__init__()
                self.ag = ag

            def forward(self, state, plan):
                return self.ag.pi(state, plan)

        torch.onnx.export(_Actor(agent).eval(), (torch.from_numpy(s[:1]), torch.from_numpy(p[:1])),
                          str(out / "residual_actor.onnx"), input_names=["state", "plan"], output_names=["residual"],
                          opset_version=17)
        rep["onnx"] = str(out / "residual_actor.onnx")
    (out / "export.json").write_text(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--onnx", action="store_true")
    a = ap.parse_args()
    print("EXPORT", json.dumps(export(a.ckpt, a.onnx)), flush=True)
