"""Relabel episodes of a teleop dataset as pick-only takes.

For takes recorded in full-demo mode that were really pick-only (the operator stopped after the inspection
hold): every frame gets the pick instruction, the episode's task list and task-index stats are updated, and the
labels row is marked `"mode": "pick_only"` with no plate, exactly like takes recorded with `--pick-only`.
Frames, videos and joint data are untouched. Every file it changes is first copied to --backup.

    python scripts/relabel_pick_only.py --root data/duplo_teleop --episodes 517-526 \
        --backup ~/desmond/tmp/relabel_backup_517-526

Run it only while no recorder is writing to the dataset.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PICK = "pick up the brick and hold it in front of the inspection camera"


def parse_episodes(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return sorted(set(out))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--episodes", required=True, help="e.g. 517-526 or 3,7,10-12")
    ap.add_argument("--backup", required=True, type=Path)
    ap.add_argument("--pick-task", default=PICK)
    args = ap.parse_args()
    root, eps = args.root, parse_episodes(args.episodes)

    t = pq.read_table(root / "meta/tasks.parquet").to_pandas()
    tasks = {str(s): int(i) for s, i in zip(t.index, t["task_index"], strict=True)}
    pick = tasks[args.pick_task]

    data_files = [f for f in sorted((root / "data").glob("*/*.parquet"))
                  if np.isin(pq.read_table(f, columns=["episode_index"]).column(0).to_numpy(), eps).any()]
    meta_files = [f for f in sorted((root / "meta/episodes").glob("*/*.parquet"))
                  if np.isin(pq.read_table(f, columns=["episode_index"]).column(0).to_numpy(), eps).any()]
    labels_path = root / "meta/dlsj/labels.jsonl"
    args.backup.mkdir(parents=True, exist_ok=False)
    for f in [*data_files, *meta_files, labels_path]:
        dst = args.backup / f.relative_to(root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst)

    lengths: dict[int, int] = {}
    for f in data_files:
        tab = pq.read_table(f)
        ep = tab.column("episode_index").to_numpy()
        ti = tab.column("task_index").to_numpy().copy()
        m = np.isin(ep, eps)
        ti[m] = pick
        for e in np.unique(ep[m]):
            lengths[int(e)] = lengths.get(int(e), 0) + int((ep == e).sum())
        i = tab.schema.get_field_index("task_index")
        pq.write_table(tab.set_column(i, tab.schema.field(i), pa.array(ti, type=tab.schema.field(i).type)), f)
    missing = sorted(set(eps) - set(lengths))
    if missing:
        raise SystemExit(f"episodes not found in the data files: {missing} (backup kept in {args.backup})")

    for f in meta_files:
        tab = pq.read_table(f)
        cols = {n: tab.column(n).to_pylist() for n in tab.column_names}
        for r, e in enumerate(cols["episode_index"]):
            if e not in lengths:
                continue
            cols["tasks"][r] = [args.pick_task]
            for k, v in (("min", pick), ("max", pick), ("mean", pick), ("std", 0.0), ("q01", pick), ("q10", pick),
                         ("q50", pick), ("q90", pick), ("q99", pick)):
                if f"stats/task_index/{k}" in cols:
                    cols[f"stats/task_index/{k}"][r] = [float(v)]
        pq.write_table(pa.table({n: pa.array(cols[n], type=tab.schema.field(n).type) for n in tab.column_names},
                                schema=tab.schema), f)

    rows = [json.loads(line) for line in labels_path.read_text().splitlines()]
    for row in rows:
        if row["episode_index"] in lengths:
            row.update(mode="pick_only", bowl=None, relabelled="pick_only (recorded in full-demo mode)")
    labels_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"relabelled {len(lengths)} episodes as pick-only ({sum(lengths.values())} frames); backup in {args.backup}")


if __name__ == "__main__":
    main()
