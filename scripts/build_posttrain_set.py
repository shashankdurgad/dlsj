"""Build the post-train set: every new-calibration demo, every pick-only take and the best original demos, in
ONE joint frame (the current robot calibration), uncropped, colour only.

Sources are colour-only copies of the raw recordings (no depth), e.g. made with
`lerobot_edit_dataset --operation.type remove_feature --operation.feature_names '["...top_depth", "...inspect_depth"]'`:

* `--new`: demos already in the target calibration (duplo_teleop_cal2), used whole and unchanged;
* `--pick`: pick-only takes in the OLD calibration, used whole;
* `--old`: full demos in the OLD calibration; plate corrections are applied (`--corrections`), then the best
  `--n-old` are kept: a real approach (grasp at least `--min-approach-s` after the start), one clean grasp,
  then the most direct approach (joint path length to the grasp over the straight-line distance).
  `--video-review` (verdicts from contact-sheet review) excludes visually flagged demos before ranking.

Old-calibration episodes are moved into the new frame by adding `--joint-offset` (degrees, new minus old for
the same physical pose, from configs/joint_offsets.json) to `observation.state` and `action`, and to their
per-episode and global statistics (means, extrema and quantiles shift; standard deviations don't).

    python scripts/build_posttrain_set.py --out data/posttrain/duplo_posttrain_814 \
        --new canonical/duplo_teleop_cal2_rgb_431 --pick canonical/duplo_pick_rgb_133 \
        --old canonical/duplo_teleop_rgb_517 --n-old 250 --exclude 464,230 \
        --corrections canonical/duplo_teleop_rgb_517/meta/dlsj/plate_corrections.json \
        --offsets configs/joint_offsets.json --offset-name recal_2026-11-14

Writes `meta/dlsj/labels.jsonl` (the source labels, re-indexed) and `meta/dlsj/provenance.jsonl`
(source dataset/episode, calibration, shift applied, selection scores) for every output episode.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_pick_split_set import load_table, plan  # noqa: E402
from prepare_train_set import apply_corrections  # noqa: E402

SHIFTED_STATS = ("min", "max", "mean", "q01", "q10", "q50", "q90", "q99")
KEYS = ("observation.state", "action")


def edit(args: list[str]) -> None:
    subprocess.run([sys.executable, "-W", "ignore", "-m", "lerobot.scripts.lerobot_edit_dataset", *args], check=True)


def labels_of(root: Path) -> list[dict]:
    return [json.loads(line) for line in (root / "meta/dlsj/labels.jsonl").read_text().splitlines() if line.strip()]


def shift_joints(root: Path, offset: np.ndarray) -> None:
    """Add `offset` to observation.state and action everywhere in the dataset at `root` (in place)."""
    for f in sorted((root / "data").glob("*/*.parquet")):
        t = pq.read_table(f)
        for k in KEYS:
            i = t.schema.get_field_index(k)
            v = np.stack(t.column(k).to_numpy(zero_copy_only=False)).astype(np.float64) + offset
            t = t.set_column(i, t.schema.field(i), pa.array(list(v.astype(np.float32)), type=t.schema.field(i).type))
        pq.write_table(t, f)
    for f in sorted((root / "meta/episodes").glob("*/*.parquet")):
        t = pq.read_table(f)
        cols = {n: t.column(n).to_pylist() for n in t.column_names}
        for k in KEYS:
            for s in SHIFTED_STATS:
                c = f"stats/{k}/{s}"
                if c in cols:
                    cols[c] = [(np.asarray(v, dtype=np.float64) + offset).tolist() for v in cols[c]]
        pq.write_table(pa.table({n: pa.array(cols[n], type=t.schema.field(n).type) for n in t.column_names},
                                schema=t.schema), f)
    sp = root / "meta/stats.json"
    stats = json.loads(sp.read_text())
    for k in KEYS:
        for s in SHIFTED_STATS:
            if s in stats.get(k, {}):
                stats[k][s] = (np.asarray(stats[k][s], dtype=np.float64) + offset).tolist()
    sp.write_text(json.dumps(stats, indent=4))


def main() -> None:
    from lerobot.datasets import LeRobotDatasetMetadata

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--new", required=True, type=Path)
    ap.add_argument("--pick", required=True, type=Path)
    ap.add_argument("--old", required=True, type=Path)
    ap.add_argument("--n-old", type=int, default=250)
    ap.add_argument("--exclude", default="", help="old episodes to drop, comma-separated")
    ap.add_argument("--min-approach-s", type=float, default=1.0)
    ap.add_argument("--corrections", type=Path)
    ap.add_argument("--video-review", type=Path,
                    help="json {episode: {verdict, issues, note}} for old demos; flagged ones are excluded")
    ap.add_argument("--offsets", type=Path, required=True)
    ap.add_argument("--offset-name", required=True)
    ap.add_argument("--work", type=Path, default=None, help="scratch directory (default <out>_work)")
    args = ap.parse_args()
    if args.out.exists():
        sys.exit(f"{args.out} exists")
    work = args.work or args.out.parent / (args.out.name + "_work")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    spec = json.loads(args.offsets.read_text())[args.offset_name]
    meta_new = LeRobotDatasetMetadata("local/new", root=args.new)
    joints = [n.split(".")[0] for n in meta_new.features["observation.state"]["names"]]
    offset = np.array([spec["offset_deg"][j] for j in joints], dtype=np.float64)
    print("offset (deg, new - old):", dict(zip(joints, offset.tolist())), flush=True)

    # 1. old demos: corrections on a full copy (episode numbers still match the corrections file), then select
    old_full = work / "old_full"
    shutil.copytree(args.old, old_full)
    corrected = {}
    if args.corrections:
        corr = json.loads(args.corrections.read_text())["corrections"]
        corrected = apply_corrections(old_full, corr)
        print(f"plate corrections applied to {len(corrected)} old episodes", flush=True)
    meta_old = LeRobotDatasetMetadata("local/old", root=old_full)
    table = load_table(old_full)
    old_labels = {l["episode_index"]: l for l in labels_of(args.old)}
    corr_bowl = {int(e): c["actual"] for e, c in (json.loads(args.corrections.read_text())["corrections"].items()
                                                  if args.corrections else [])}
    exclude = {int(x) for x in args.exclude.split(",") if x.strip()}
    review = {int(k): v for k, v in json.loads(args.video_review.read_text()).items()} if args.video_review else {}
    flagged = {e for e, v in review.items() if v.get("verdict") == "flag"}
    exclude |= flagged
    if review:
        print(f"video review: {len(review)} reviewed, {len(flagged)} flagged and excluded: {sorted(flagged)}", flush=True)
    min_approach = int(round(args.min_approach_s * meta_old.fps))
    plans = [p for e in range(meta_old.total_episodes) if e not in exclude
             and (p := plan(meta_old, table, old_labels, e, 9, 12)) and p["grasp"] >= min_approach]
    plans.sort(key=lambda r: (r["closings"] != 1, r["directness"]))
    keep = sorted(p["episode"] for p in plans[: args.n_old])
    scores = {p["episode"]: p for p in plans}
    print(f"old demos: {len(plans)} usable, keeping {len(keep)}", flush=True)
    drop = [e for e in range(meta_old.total_episodes) if e not in set(keep)]
    old_sel = work / "old_sel"
    edit(["--repo_id", "local/old", "--root", str(old_full), "--new_repo_id", "local/old_sel",
          "--new_root", str(old_sel), "--operation.type", "delete_episodes", "--operation.episode_indices", json.dumps(drop)])

    # 2. pick-only takes: a copy
    pick = work / "pick"
    shutil.copytree(args.pick, pick)

    # 3. old calibration -> new frame
    for root in (old_sel, pick):
        shift_joints(root, offset)
    print("joint offsets applied to the old-calibration episodes", flush=True)

    # 4. merge: new, pick, old (in that order)
    edit(["--new_repo_id", f"local/{args.out.name}", "--new_root", str(args.out), "--operation.type", "merge",
          "--operation.repo_ids", json.dumps(["local/new", "local/pick", "local/old_sel"]),
          "--operation.roots", json.dumps([str(args.new), str(pick), str(old_sel)]),
          "--operation.concatenate_videos", "false", "--operation.concatenate_data", "false"])

    # 5. labels + provenance
    rows, prov = [], []
    sources = [("duplo_teleop_cal2", args.new, list(range(meta_new.total_episodes)), labels_of(args.new), "native"),
               ("duplo_pick", args.pick, list(range(LeRobotDatasetMetadata("local/pick", root=args.pick).total_episodes)),
                labels_of(args.pick), "old->new"),
               ("duplo_teleop", args.old, keep, labels_of(args.old), "old->new")]
    i = 0
    for name, root, eps, labels, cal in sources:
        by_ep = {l["episode_index"]: l for l in labels}
        for e in eps:
            lab = dict(by_ep[e])
            if name == "duplo_teleop" and e in corr_bowl:
                lab.update(bowl=corr_bowl[e], bowl_as_recorded=by_ep[e]["bowl"])
            lab.update(episode_index=i, source_dataset=name, source_episode=e)
            rows.append(lab)
            p = {"episode_index": i, "source_dataset": name, "source_root": str(root), "source_episode": e,
                 "calibration": "new (recal_2026-11-14)" if cal == "native" else "old, shifted into new",
                 "joint_offset_added_deg": None if cal == "native" else dict(zip(joints, offset.tolist())),
                 "plate_corrected": name == "duplo_teleop" and e in corrected}
            if name == "duplo_teleop":
                p.update({k: scores[e][k] for k in ("grasp", "closings", "directness")})
                if review:
                    p["video_review"] = review.get(e, {"verdict": "not reviewed"})
            prov.append(p)
            i += 1
    d = args.out / "meta" / "dlsj"
    d.mkdir(parents=True, exist_ok=True)
    (d / "labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "provenance.jsonl").write_text("".join(json.dumps(r) + "\n" for r in prov))
    shutil.copy2(args.offsets, d / "joint_offsets.json")
    if args.corrections:
        shutil.copy2(args.corrections, d / "plate_corrections_applied_to_duplo_teleop.json")
    print(f"wrote {i} episodes to {args.out}; scratch in {work}", flush=True)


if __name__ == "__main__":
    main()
