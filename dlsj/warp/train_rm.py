"""Train WARP-RM on the fastest successful demos (the reference pace, v = 1).

    python -m dlsj.warp.train_rm --root ~/desmond/data/duplo_teleop --feats ~/desmond/warp/feats \
        --out ~/desmond/warp/rm --cameras top,wrist

Reference set: successful episodes no longer than the `--ref-quantile` length quantile (paper: the ~30%
shortest of the folding data). 10% of them are held out; validation reports (a) the label loss and (b) the mean
predicted velocity on held-out episodes played forwards (should be ~1) and backwards (should be ~-1).
"""

import argparse
import json
import logging
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from dlsj.warp import sampler
from dlsj.warp.data import load_episodes, load_features
from dlsj.warp.model import WarpRM

log = logging.getLogger("warp.train_rm")


def reference_split(eps, quantile: float, seed: int = 0):
    ok = [e for e in eps if e.success]
    cut = np.quantile([e.length for e in ok], quantile)
    ref = [e for e in ok if e.length <= cut]
    rng = np.random.default_rng(seed)
    rng.shuffle(ref)
    n_val = max(2, len(ref) // 10)
    return ref[n_val:], ref[:n_val], float(cut)


def batch_windows(cfg, feats: dict, eps, rng, bs: int):
    picks = rng.choice(len(eps), size=bs)
    xs, ys = [], []
    for k in picks:
        f = feats[eps[k].index]
        idx, y = sampler.sample_window(cfg, len(f), rng)
        xs.append(f[idx])
        ys.append(y)
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))


@torch.no_grad()
def direction_check(model, cfg, feats, eps, device) -> tuple[float, float]:
    """Mean end-of-window velocity on canonical forward and reversed windows of held-out episodes."""
    fwd, bwd = [], []
    span = int(round((cfg.n - 1) * cfg.stride_frames))
    for e in eps:
        f = feats[e.index]
        for s in range(0, max(1, len(f) - span), max(1, span // 2)):
            idx = np.clip(sampler.canonical_window(cfg, s), 0, len(f) - 1)
            for rev, acc in ((False, fwd), (True, bwd)):
                w = idx[::-1].copy() if rev else idx
                y = model.expected(torch.from_numpy(f[w]).unsqueeze(0).to(device))[0].cpu().numpy()
                acc.append(y[-1] - y[0])
    return float(np.mean(fwd)), float(np.mean(bwd))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--feats", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cameras", default="top,wrist")
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--stride-s", type=float, default=0.5)
    p.add_argument("--ref-quantile", type=float, default=0.4)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--feat-noise", type=float, default=0.1, help="Gaussian noise on input features, x feature std")
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    eps, fps = load_episodes(args.root)
    cams = args.cameras.split(",")
    cfg = sampler.WarpConfig(n=args.n, stride_s=args.stride_s, fps=fps)
    train, val, cut = reference_split(eps, args.ref_quantile)
    log.info("episodes %d, reference %d train + %d val (length <= %.1fs), window %.1fs",
             len(eps), len(train), len(val), cut / fps, cfg.window_s)
    feats = {e.index: load_features(args.feats, cams, e.index) for e in train + val}
    dim = next(iter(feats.values())).shape[1]

    model = WarpRM(dim, cfg.n, d_model=args.d_model, layers=args.layers).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    warm = min(1000, args.steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (args.steps - warm))))
    rng = np.random.default_rng(0)
    vx, vy = batch_windows(cfg, feats, val, np.random.default_rng(1), 4096)  # fixed held-out windows
    fstd = torch.from_numpy(np.concatenate(list(feats.values())).std(0)).float()
    best = {"mae": float("inf")}
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        x, y = batch_windows(cfg, feats, train, rng, args.bs)
        if args.feat_noise > 0:
            x = x + args.feat_noise * fstd * torch.randn_like(x)
        loss = model.loss(x.to(args.device), y.to(args.device))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                vloss = vmae = 0.0
                for i in range(0, len(vx), 1024):
                    bx, by = vx[i: i + 1024].to(args.device), vy[i: i + 1024].to(args.device)
                    vloss += model.loss(bx, by).item() * len(bx) / len(vx)
                    vmae += (model.expected(bx) - by).abs().mean().item() * len(bx) / len(vx)
            fwd, bwd = direction_check(model, cfg, feats, val, args.device)
            tag = ""
            if vmae < best["mae"]:
                best = {"mae": vmae, "loss": vloss, "step": step, "fwd": fwd, "bwd": bwd,
                        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
                tag = " *best*"
            log.info("step %d loss %.3f | val loss %.3f, progress MAE %.3f | held-out velocity fwd %.2f bwd %.2f "
                     "(want +-1) | %.0fs%s", step, loss.item(), vloss, vmae, fwd, bwd, time.time() - t0, tag)
    vloss, fwd, bwd = best["loss"], best["fwd"], best["bwd"]
    log.info("keeping step %d (val progress MAE %.3f)", best["step"], best["mae"])
    torch.save({"state_dict": best["state"], "feat_dim": dim, "cameras": cams, "warp": asdict(cfg),
                "d_model": args.d_model, "layers": args.layers,
                "reference": [e.index for e in train], "val": [e.index for e in val]}, out / "warp_rm.pt")
    (out / "train_summary.json").write_text(json.dumps(
        {"best_step": best["step"], "val_progress_mae": best["mae"], "val_loss": vloss, "val_forward_velocity": fwd, "val_reverse_velocity": bwd, "ref_len_cut_s": cut / fps,
         "n_reference": len(train), "n_val": len(val)}, indent=2))
    log.info("saved %s", out / "warp_rm.pt")


if __name__ == "__main__":
    main()
