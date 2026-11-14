"""Offline check of a π0.5 OpenVINO package against recorded teleop episodes (no robot needed).

Two steps, because the dataset reader and the policy runtime live in different envs:

1. `extract` (any env with pyarrow + av, e.g. the recorder's): pick frames from recorded episodes and save each
   one's 3 camera images, joint state, task text and the next 50 recorded actions to an .npz. It reads only
   sealed parquet files, so it is safe while the recorder is appending to the same dataset.

       python -m dlsj.deploy.pi05_eval extract --root data/duplo_teleop --episodes 312-416:5 --out eval.npz

2. `run` (the runtime env, see `dlsj.deploy.pi05_runtime`): predict a chunk for each sample and report
   * action error against the recorded chunk (mean abs, per joint, pick vs place phase, and the first 0.5 s),
     next to a "hold still" baseline (repeat the current state for the whole chunk);
   * prompt sensitivity on place-phase samples: the same observation with each plate colour in the prompt
     (several noise draws each). If the policy follows the text, the true colour's prediction should be the
     closest to the recording, and colours should differ by more than noise does.

       python -m dlsj.deploy.pi05_eval run eval.npz --export exports/pi05_ft_int8 --device GPU
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

CAMS = ("wrist", "top", "inspect")
COLOURS = ("orange", "purple", "yellow")
CHUNK = 50
FPS = 30


def _sealed(path: Path) -> bool:
    with open(path, "rb") as f:
        f.seek(-4, 2)
        return f.read() == b"PAR1"


def _episode_index(root: Path) -> dict[int, dict]:
    import pyarrow.parquet as pq

    out = {}
    for f in sorted(root.glob("meta/episodes/*/*.parquet")):
        if not _sealed(f):
            continue
        for row in pq.read_table(f, columns=[c for c in pq.read_schema(f).names if not c.startswith("stats/")]).to_pylist():
            out[row["episode_index"]] = row
    return out


def _frames(path: Path, times: list[float]) -> list[np.ndarray]:
    """RGB frames nearest to each timestamp (seconds) in one video file."""
    import av

    want = sorted(range(len(times)), key=lambda i: times[i])
    out: list[np.ndarray | None] = [None] * len(times)
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        c.seek(int(max(0.0, times[want[0]] - 0.5) / s.time_base), stream=s, backward=True)
        k = 0
        for fr in c.decode(s):
            t = float(fr.pts * s.time_base)
            while k < len(want) and t >= times[want[k]] - 0.5 / FPS:
                out[want[k]] = fr.to_ndarray(format="rgb24")
                k += 1
            if k == len(want):
                break
    if any(o is None for o in out):
        raise RuntimeError(f"could not decode all frames from {path}")
    return out


def extract(root: Path, episodes: list[int], out: Path, pick_n: int = 4, place_n: int = 3) -> None:
    import pyarrow.parquet as pq

    labels = {r["episode_index"]: r for r in map(json.loads, open(root / "meta/dlsj/labels.jsonl"))}
    meta = _episode_index(root)
    info = json.loads((root / "meta/info.json").read_text())
    S = {k: [] for k in ("episode", "frame", "phase", "task", "bowl", "state", "future", "valid", *CAMS)}
    for ep in episodes:
        m, lab = meta.get(ep), labels.get(ep)
        if m is None or lab is None or not lab["success"] or lab["inspection_frame"] is None:
            continue
        dpath = root / info["data_path"].format(chunk_index=m["data/chunk_index"], file_index=m["data/file_index"])
        if not _sealed(dpath):
            continue
        t = pq.read_table(dpath, columns=["episode_index", "frame_index", "observation.state", "action"]).to_pandas()
        t = t[t.episode_index == ep].sort_values("frame_index")
        state, action = np.stack(t["observation.state"].values), np.stack(t["action"].values)
        n, split = len(t), lab["inspection_frame"]
        picks = [int(split * q) for q in np.linspace(0, 0.85, pick_n)]
        places = [int(split + (n - split) * q) for q in np.linspace(0, 0.6, place_n)]
        frames = picks + places
        imgs = {}
        for cam in CAMS:
            key = f"observation.images.{cam}"
            vpath = root / info["video_path"].format(video_key=key, chunk_index=m[f"videos/{key}/chunk_index"],
                                                     file_index=m[f"videos/{key}/file_index"])
            t0 = m[f"videos/{key}/from_timestamp"]
            imgs[cam] = _frames(vpath, [t0 + f / FPS for f in frames])
        for j, f in enumerate(frames):
            fut = action[f:f + CHUNK]
            valid = len(fut)
            fut = np.concatenate([fut, np.repeat(fut[-1:], CHUNK - valid, 0)]) if valid < CHUNK else fut
            phase = "pick" if f < split else "place"
            S["episode"].append(ep), S["frame"].append(f), S["phase"].append(phase), S["bowl"].append(lab["bowl"])
            S["task"].append(lab["pick_task"] if phase == "pick" else f"place the brick in the {lab['bowl']} bowl")
            S["state"].append(state[f]), S["future"].append(fut), S["valid"].append(valid)
            for cam in CAMS:
                S[cam].append(imgs[cam][j])
        print(f"episode {ep}: {len(frames)} samples", flush=True)
    np.savez_compressed(out, **{k: np.asarray(v) for k, v in S.items()})
    print(f"wrote {out}: {len(S['episode'])} samples from {len(set(S['episode']))} episodes")


def _err(pred: np.ndarray, fut: np.ndarray, valid: int, steps: int | None = None) -> np.ndarray:
    k = min(valid, steps or valid)
    return np.abs(pred[:k] - fut[:k]).mean(0)  # per joint


def run(npz: Path, export: str, device: str, draws: int = 2, json_out: Path | None = None) -> dict:
    from dlsj.deploy.pi05_runtime import Pi05Policy

    d = np.load(npz, allow_pickle=False)
    policy = Pi05Policy(export, device)
    n = len(d["episode"])
    rows, ms = [], []
    for i in range(n):
        obs = {c: d[c][i] for c in CAMS}
        state, fut, valid = d["state"][i], d["future"][i], int(d["valid"][i])
        t = time.perf_counter()
        pred = policy(state, obs, str(d["task"][i]))
        ms.append((time.perf_counter() - t) * 1e3)
        row = {"episode": int(d["episode"][i]), "phase": str(d["phase"][i]),
               "mae": _err(pred, fut, valid).tolist(), "mae_0.5s": _err(pred, fut, valid, 15).tolist(),
               "hold_mae": _err(np.repeat(state[None], CHUNK, 0), fut, valid).tolist()}
        if row["phase"] == "place":
            per = {}
            for colour in COLOURS:
                per[colour] = [policy(state, obs, f"place the brick in the {colour} bowl") for _ in range(draws)]
            true = str(d["bowl"][i])
            row["bowl"] = true
            row["prompt_mae"] = {c: float(np.mean([_err(p, fut, valid).mean() for p in ps])) for c, ps in per.items()}
            row["true_colour_best"] = min(row["prompt_mae"], key=row["prompt_mae"].get) == true
            row["noise_diff"] = float(np.mean([np.abs(ps[0] - ps[1]).mean() for ps in per.values()]))
            others = [c for c in COLOURS if c != true]
            row["prompt_diff"] = float(np.mean([np.abs(per[true][0] - per[c][0]).mean() for c in others]))
        rows.append(row)
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{n} samples, {np.median(ms):.0f} ms per chunk", flush=True)

    def summary(sel):
        if not sel:
            return None
        mae, hold = np.array([r["mae"] for r in sel]), np.array([r["hold_mae"] for r in sel])
        return {"n": len(sel), "mae": round(float(mae.mean()), 2), "mae_first_0.5s": round(float(np.mean([r["mae_0.5s"] for r in sel])), 2),
                "hold_still_mae": round(float(hold.mean()), 2), "per_joint_mae": np.round(mae.mean(0), 2).tolist()}

    place = [r for r in rows if r["phase"] == "place"]
    res = {"export": export, "device": device, "samples": n, "episodes": len({r["episode"] for r in rows}),
           "median_ms_per_chunk": round(float(np.median(ms)), 1),
           "all": summary(rows), "pick": summary([r for r in rows if r["phase"] == "pick"]), "place": summary(place),
           "prompt": None if not place else {
               "true_colour_closest_rate": round(float(np.mean([r["true_colour_best"] for r in place])), 3),
               "chance": round(1 / len(COLOURS), 3),
               "mae_true_colour_prompt": round(float(np.mean([r["prompt_mae"][r["bowl"]] for r in place])), 2),
               "mae_wrong_colour_prompts": round(float(np.mean([np.mean([v for c, v in r["prompt_mae"].items()
                                                                          if c != r["bowl"]]) for r in place])), 2),
               "diff_between_colours": round(float(np.mean([r["prompt_diff"] for r in place])), 3),
               "diff_between_noise_draws": round(float(np.mean([r["noise_diff"] for r in place])), 3)}}
    if json_out:
        json_out.write_text(json.dumps({"summary": res, "rows": rows}))
    return res


def _episodes(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        rng, _, step = part.partition(":")
        a, _, b = rng.partition("-")
        out += list(range(int(a), int(b or a) + 1, int(step or 1)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--root", type=Path, required=True)
    e.add_argument("--episodes", required=True, help="e.g. 312-416:5 or 0-200:20,312-416:5")
    e.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("npz", type=Path)
    r.add_argument("--export", required=True)
    r.add_argument("--device", default="GPU")
    r.add_argument("--draws", type=int, default=2)
    r.add_argument("--json", type=Path)
    args = ap.parse_args()
    if args.cmd == "extract":
        extract(args.root, _episodes(args.episodes), args.out)
    else:
        print(json.dumps(run(args.npz, args.export, args.device, args.draws, args.json), indent=1))


if __name__ == "__main__":
    main()
