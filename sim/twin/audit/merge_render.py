"""Merge a batched render (<run>/parts/sim_<start>.npz + render_meta_<start>.json) into <run>/sim.npz and
render_meta.json.

    python sim/twin/audit/merge_render.py --run sim/twin/out/audit/runs/<stamp>      # plain python, CPU
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from sim.twin.audit.common import jdump, load_set


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    run = Path(ap.parse_args().run)
    n = len(load_set()["samples"])
    parts = sorted((run / "parts").glob("sim_*.npz"))
    metas = [json.loads((run / "parts" / f"render_meta_{p.stem[4:]}.json").read_text()) for p in parts]
    got = sum(m["n"] for m in metas)
    if got != n:
        sys.exit(f"MERGE_FAILED {got} of {n} samples rendered")
    zs = [np.load(p) for p in parts]
    np.savez_compressed(run / "sim.npz", **{k: np.concatenate([z[k] for z in zs]) for k in zs[0].files})
    meta = dict(metas[0])
    meta.update(n=got, batches=len(parts), black_retries=sum(m.get("black_retries", 0) for m in metas),
                render_s=round(sum(m["render_s"] + m["env_start_s"] for m in metas), 1),
                errors=[e for m in metas for e in m["errors"]], samples=[r for m in metas for r in m["samples"]],
                grasp={k: v for m in metas for k, v in m["grasp"].items()},
                physics_set=sorted({m.get("physics_set", "") for m in metas}))
    jdump(run / "render_meta.json", meta)
    print("MERGE_OK", got, "samples from", len(parts), "batches", flush=True)


if __name__ == "__main__":
    main()
