"""Offline accuracy harness for Studio π0.5 OpenVINO exports (held-out frames, fixed noise).

Three steps, all in the Studio library env on the XPS (`.venv/bin/python`, not `uv run`):

1. `cache`: decode N frames of the LeRobot v3 dataset once (3 cameras as 640x480 uint8, the 6-d state, the
   frame's own task string, and the recorded actions [t, t+50) with a padding mask) into one `.npz`, so every
   variant sees identical inputs. `--split val` takes the 21 held-out episodes of the training split
   (`random.Random(42).sample(range(212), 21)`); `--split train` takes the others (used for calibration).
2. `run`: feed the cache through `physicalai.InferenceModel` for one export, with a fixed start noise per
   frame (`noise` passed in the observation, so the runtime keeps it), and save the denormalised chunks.
   `--capture_ir` also saves the exact tensors the IR receives (for NNCF calibration).
3. `report`: compare prediction files with the recorded actions (MSE/MAE over the first 10 and all 50
   steps, per joint) and with the FP16 10-step reference on the same inputs and noise (MSE, MAE, max abs,
   cosine). A second FP16 run with different noise seeds gives the policy's own sampling spread
   ("seed floor"), the yardstick for deviations.

    PY=~/inference/physical-ai-studio/library/.venv/bin/python
    $PY eval_actions.py cache --dataset ~/desmond/dlsj/data/duplo_teleop_rgb --split val --n 300 \
        --out ~/inference/eval/pi05_val300.npz
    $PY eval_actions.py run ~/inference/exports/pi05_junction_fp16 --cache ~/inference/eval/pi05_val300.npz \
        --out ~/inference/eval/preds/fp16.npz
    $PY eval_actions.py report --cache ~/inference/eval/pi05_val300.npz --ref ~/inference/eval/preds/fp16.npz \
        --floor ~/inference/eval/preds/fp16_seed2.npz ~/inference/eval/preds/*.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import opt_lib as L  # noqa: E402

CAMERAS = ("wrist", "top", "inspect")
N_TRAIN_EPISODES = 212  # episodes the checkpoint was trained on (the dataset is still growing)


# --------------------------------------------------------------------------- cache
def build_cache(dataset: Path, split: str, n: int, seed: int, out: Path, chunk: int) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    val = set(L.validation_episodes(N_TRAIN_EPISODES))
    episodes = sorted(val) if split == "val" else [e for e in range(N_TRAIN_EPISODES) if e not in val]
    ds = LeRobotDataset("local/duplo_teleop_rgb", root=dataset, episodes=episodes, video_backend="pyav")
    hf = ds.hf_dataset.with_format(None)
    ep_col = np.asarray(hf["episode_index"])
    fr_col = np.asarray(hf["frame_index"])
    task_col = np.asarray(hf["task_index"])
    actions = np.asarray(hf["action"], dtype=np.float32)
    tasks_df = ds.meta.tasks  # pandas DataFrame indexed by task string, column task_index
    idx_to_task = {int(row.task_index): str(name) for name, row in tasks_df.iterrows()}

    pos: dict[tuple[int, int], int] = {}
    episode_tasks: dict[int, list[str]] = {}
    episode_rows: dict[int, np.ndarray] = {}
    for ep in episodes:
        rows = np.flatnonzero(ep_col == ep)
        rows = rows[np.argsort(fr_col[rows])]
        if len(rows) == 0:
            continue
        episode_rows[ep] = rows
        episode_tasks[ep] = [idx_to_task[int(t)] for t in task_col[rows]]
        for r in rows:
            pos[(ep, int(fr_col[r]))] = int(r)

    picks = L.sample_frames(episode_tasks, n, seed=seed)
    print(f"{split}: {len(episode_rows)} episodes, {sum(map(len, episode_rows.values()))} frames, "
          f"{len(picks)} sampled", flush=True)
    imgs = {c: [] for c in CAMERAS}
    state, target, mask, meta, task = [], [], [], [], []
    t0 = time.perf_counter()
    for i, (ep, f, tsk) in enumerate(picks):
        item = ds[pos[(ep, f)]]
        assert int(item["episode_index"]) == ep and int(item["frame_index"]) == f
        assert item["task"] == tsk, (item["task"], tsk)
        for c in CAMERAS:
            x = item[f"observation.images.{c}"].numpy()  # (3,H,W) float in [0,1]
            imgs[c].append(np.clip(np.rint(x * 255), 0, 255).astype(np.uint8))
        state.append(item["observation.state"].numpy().astype(np.float32))
        tgt, m = L.future_actions(actions[episode_rows[ep]], f, chunk)
        target.append(tgt)
        mask.append(m)
        meta.append((ep, f))
        task.append(tsk)
        if i % 50 == 0:
            print(f"  {i}/{len(picks)} decoded ({time.perf_counter() - t0:.0f}s)", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **{f"img_{c}": np.stack(v) for c, v in imgs.items()}, state=np.stack(state),
             target=np.stack(target), mask=np.stack(mask), meta=np.asarray(meta, dtype=np.int64),
             task=np.asarray(task), split=split)
    counts = {t: task.count(t) for t in sorted(set(task))}
    print(json.dumps({"out": str(out), "n": len(task), "tasks": counts,
                      "episodes": sorted({e for e, _ in meta})}, indent=1))


# --------------------------------------------------------------------------- run
def observation(model: Any, cache: Any, i: int) -> dict[str, Any]:
    """One observation in the runtime's input format, keyed by the manifest's feature names."""
    obs: dict[str, Any] = {}
    for feat in model.input_features:
        name = feat.name
        if name.startswith("images"):
            cam = name.split(".")[-1]
            obs[name] = cache[f"img_{cam}"][i][None]  # (1,3,H,W) uint8
        elif name == "state":
            obs[name] = cache["state"][i][None]
        elif name == "task":
            obs[name] = [str(cache["task"][i])]
    return obs


def run_export(export_dir: Path, cache_path: Path, out: Path, device: str, seed_base: int,
               capture_ir: Path | None, limit: int | None, config: dict[str, str]) -> None:
    from physicalai.inference import InferenceModel

    cache = dict(np.load(cache_path))
    n = len(cache["task"]) if limit is None else min(limit, len(cache["task"]))
    cfg = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1",
           "CACHE_DIR": str(Path("~/inference/ov_cache").expanduser()), **config}
    t0 = time.perf_counter()
    model = InferenceModel(str(export_dir), backend="openvino", device=device, **cfg)
    load_s = time.perf_counter() - t0
    noise_shape = None
    for k, shape in zip(model.adapter.input_names, getattr(model.adapter, "input_shapes", []) or []):
        if k == "noise":
            noise_shape = shape
    captured: list[dict[str, np.ndarray]] = []
    if capture_ir:
        inner = model.adapter.predict

        def grab(inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            captured.append({k: np.array(v, copy=True) for k, v in inputs.items()})
            return inner(inputs)

        model.adapter.predict = grab
    preds, ms = [], []
    for i in range(n):
        obs = observation(model, cache, i)
        obs["noise"] = L.noise_for(i, noise_shape or (1, 50, 32), base_seed=seed_base)
        t = time.perf_counter()
        preds.append(np.asarray(model.predict_action_chunk(obs), dtype=np.float32))
        ms.append((time.perf_counter() - t) * 1e3)
        if i % 50 == 0:
            print(f"  {i}/{n} ({ms[-1]:.0f} ms)", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, pred=np.stack(preds), export=str(export_dir), device=device, seed_base=seed_base,
             config=json.dumps(cfg))
    if capture_ir:
        # one entry per call, each exactly as fed to the IR (images [3,1,3,224,224], image_masks [3,1], ...)
        np.savez(capture_ir, **{k: np.stack([c[k] for c in captured]) for k in captured[0]})
    print(json.dumps({"out": str(out), "n": n, "load_compile_s": round(load_s, 1),
                      "latency_during_eval": L.latency_stats(ms[1:] if n > 1 else ms)}))


# --------------------------------------------------------------------------- report
def report(cache_path: Path, ref_path: Path, floor_path: Path | None, preds: list[Path], out: Path | None) -> None:
    cache = np.load(cache_path)
    target, mask, task = cache["target"], cache["mask"], cache["task"]
    is_pick = np.char.startswith(task.astype(str), "pick")
    ref = np.load(ref_path)["pred"]
    rows: dict[str, Any] = {}
    floor = L.deviation(np.load(floor_path)["pred"], ref) if floor_path else None
    ref_mae = L.per_sample_mae(ref, target, mask)
    for p in preds:
        pred = np.load(p)["pred"]
        n = len(pred)
        boot = L.paired_bootstrap(L.per_sample_mae(pred, target[:n], mask[:n]), ref_mae[:n])
        r = {"n": n, "vs_recorded": L.chunk_errors(pred, target[:n], mask[:n]),
             "vs_recorded_pick": L.chunk_errors(pred[is_pick[:n]], target[:n][is_pick[:n]], mask[:n][is_pick[:n]]),
             "vs_recorded_place": L.chunk_errors(pred[~is_pick[:n]], target[:n][~is_pick[:n]], mask[:n][~is_pick[:n]]),
             "vs_fp16": L.deviation(pred, ref[:n]), "k10_mae_change_vs_fp16": boot,
             "jerk": L.jerk(pred, mask[:n]), "range_ratio": L.range_ratio(pred, target[:n], mask[:n])}
        rows[p.stem] = r
    res = {"cache": str(cache_path), "ref": str(ref_path), "seed_floor": floor, "variants": rows,
           "recorded_jerk": L.jerk(target, mask)}
    if out:
        out.write_text(json.dumps(res, indent=1))
    print("| variant | n | MAE k10 | Δ vs FP16 [95% CI] | MAE k50 | MSE k10 | gripper MAE k10 | dev MAE | dev max | "
          "cos(centred) | jerk | range ratio |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, r in rows.items():
        a, d = r["vs_recorded"], r["vs_fp16"]
        b = r["k10_mae_change_vs_fp16"]
        print(f"| {name} | {r['n']} | {a['k10']['mae']:.3f} | {b['rel']:+.1%} [{b['lo']:+.1%}, {b['hi']:+.1%}] | "
              f"{a['k50']['mae']:.3f} | {a['k10']['mse']:.3f} | "
              f"{a['k10']['per_joint']['gripper']['mae']:.3f} | {d['mae']:.3f} | {d['max_abs']:.2f} | "
              f"{d['cos_centered']:.4f} | {r['jerk']:.2f} | {r['range_ratio']:.2f} |")
    print(f"recorded jerk {res['recorded_jerk']:.2f}")
    if floor:
        print(f"seed floor (FP16 vs FP16, other noise): MAE {floor['mae']:.3f}, max {floor['max_abs']:.2f}, "
              f"cos(centred) {floor['cos_centered']:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache")
    c.add_argument("--dataset", type=Path, required=True)
    c.add_argument("--split", choices=["val", "train"], default="val")
    c.add_argument("--n", type=int, default=300)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--chunk", type=int, default=50)
    c.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("export_dir", type=Path)
    r.add_argument("--cache", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--device", default="GPU")
    r.add_argument("--seed_base", type=int, default=1234)
    r.add_argument("--capture_ir", type=Path, help="also save the IR's input tensors here (calibration)")
    r.add_argument("--limit", type=int)
    r.add_argument("--config", action="append", default=[], metavar="KEY=VALUE")
    p = sub.add_parser("report")
    p.add_argument("preds", type=Path, nargs="+")
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--ref", type=Path, required=True)
    p.add_argument("--floor", type=Path)
    p.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.cmd == "cache":
        build_cache(a.dataset.expanduser(), a.split, a.n, a.seed, a.out.expanduser(), a.chunk)
    elif a.cmd == "run":
        run_export(a.export_dir.expanduser(), a.cache.expanduser(), a.out.expanduser(), a.device, a.seed_base,
                   a.capture_ir, a.limit, dict(kv.split("=", 1) for kv in a.config))
    else:
        report(a.cache, a.ref, a.floor, a.preds, a.out)


if __name__ == "__main__":
    main()
