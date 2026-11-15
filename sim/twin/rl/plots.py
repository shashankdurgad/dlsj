"""Learning curves from sim/twin/out/rl/<run>/log.jsonl (+ baseline lines from eval.json / baselines.json)."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def curves(run_dir: Path, baselines: Path | None = None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    rows = [json.loads(line) for line in (run_dir / "log.jsonl").read_text().splitlines() if line.strip()]
    if not rows:
        return
    x = [r["env_ticks"] / 1e6 for r in rows]
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    a = ax[0, 0]
    for key, lab in (("success", "success (train, last 1000 eps)"), ("centred", "centred grasp"), ("pushed", "pushed"),
                     ("tipped", "tipped")):
        a.plot(x, [r.get(key) for r in rows], label=lab)
    bl = baselines or (run_dir.parent / "baselines" / "eval.json")
    if Path(bl).exists():
        for b, ls in zip(json.loads(Path(bl).read_text()), (":", "--", "-."), strict=False):
            a.axhline(b["success"], ls=ls, c="grey", lw=1, label=f"{b['policy']} (test)")
    a.set_xlabel("env ticks (M)")
    a.set_ylabel("rate")
    a.legend(fontsize=7)
    a.set_title("training episodes (collection policy)")
    a = ax[0, 1]
    a.plot(x, [r.get("ret") for r in rows])
    a.set_title("episode return")
    a.set_xlabel("env ticks (M)")
    a = ax[1, 0]
    for key in ("critic_loss", "q_data", "target"):
        a.plot(x, [r.get(key) for r in rows], label=key)
    a.set_yscale("symlog", linthresh=0.1)
    a.legend(fontsize=7)
    a.set_title("critic")
    a.set_xlabel("env ticks (M)")
    a = ax[1, 1]
    for key in ("res_rms", "bc", "q_gain", "bc_filtered"):
        a.plot(x, [r.get(key) for r in rows], label=key)
    a.legend(fontsize=7)
    a.set_title("actor (residual RMS in [-1,1] units)")
    a.set_xlabel("env ticks (M)")
    fig.suptitle(run_dir.name)
    fig.tight_layout()
    fig.savefig(run_dir / "curves.png", dpi=110)
    print("wrote", run_dir / "curves.png", flush=True)


if __name__ == "__main__":
    curves(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else None)
