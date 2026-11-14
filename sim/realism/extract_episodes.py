"""Stage A2 data extraction (runs on the XPS, read-only on the datasets).

For a sample of episodes from the old-calibration 517-episode set and the new-calibration cal2 set, decode top-camera
frames at: start, grasp_frame, mid phase 2, inspection_frame, end (+ wrist at start), run the one-class brick detector
on the top frames, segment the plates by colour, and dump everything (frames as JPG + one JSON) for analysis.

    PYTHONPATH=~/inference/dlsj-smolvla ~/desmond/venvs/teleop/bin/python extract_episodes.py --out ~/inference/tmp/realism
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd

SETS = {
    "old": ("/home/intel/desmond/dlsj/data/duplo_teleop", "/home/intel/inference/data/duplo_teleop_3phase"),
    "cal2": ("/home/intel/desmond/build/dlsj-main/data/duplo_teleop_cal2", "/home/intel/inference/data/duplo_teleop_cal2_3phase"),
}

# HSV ranges (OpenCV H 0-180) for the plates seen from the top camera
PLATE_HSV = {
    "yellow": ((18, 90, 120), (35, 255, 255)),
    "blue": ((92, 80, 100), (112, 255, 255)),
    "orange": ((5, 120, 120), (17, 255, 255)),
    "purple": ((120, 40, 60), (160, 255, 255)),
}
BRICK_COLOURS = {  # hue centre (OpenCV), for naming detector boxes
    "red": 0, "orange": 12, "yellow": 25, "green": 60, "blue": 105, "purple": 140,
}


def episode_table(root: str) -> pd.DataFrame:
    fs = sorted(glob.glob(root + "/meta/episodes/**/*.parquet", recursive=True))
    return pd.concat(pd.read_parquet(f) for f in fs).set_index("episode_index")


def states(root: str) -> dict[int, pd.DataFrame]:
    fs = sorted(glob.glob(root + "/data/**/*.parquet", recursive=True))
    df = pd.concat(pd.read_parquet(f, columns=["observation.state", "action", "episode_index", "frame_index"]) for f in fs)
    return {int(e): g.sort_values("frame_index") for e, g in df.groupby("episode_index")}


def decode(root: str, info: dict, row: pd.Series, cam: str, frames: list[int], fps: float) -> dict[int, np.ndarray]:
    key = f"observation.images.{cam}"
    chunk = int(row[f"videos/{key}/chunk_index"])
    fidx = int(row[f"videos/{key}/file_index"])
    t0 = float(row[f"videos/{key}/from_timestamp"])
    path = Path(root) / info["video_path"].format(video_key=key, chunk_index=chunk, file_index=fidx)
    want = {f: t0 + f / fps for f in frames}
    out: dict[int, np.ndarray] = {}
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        for f, t in sorted(want.items(), key=lambda kv: kv[1]):
            c.seek(int(max(0.0, t - 0.5) / st.time_base), stream=st, backward=True, any_frame=False)
            best = None
            for fr in c.decode(st):
                ft = float(fr.pts * st.time_base)
                if ft + 1e-3 >= t:
                    best = fr.to_ndarray(format="bgr24")
                    break
            if best is not None:
                out[f] = best
    return out


def plates(bgr: np.ndarray) -> dict:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    res = {}
    for name, (lo, hi) in PLATE_HSV.items():
        m = cv2.inRange(hsv, np.array(lo), np.array(hi))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        n, lab, st, cen = cv2.connectedComponentsWithStats(m)
        if n <= 1:
            continue
        i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        area = int(st[i, cv2.CC_STAT_AREA])
        if area < 4000:  # a plate is large; bricks are < ~2500 px
            continue
        ys, xs = np.nonzero(lab == i)
        pts = np.stack([xs, ys], 1).astype(np.float32)
        (cx, cy), r = cv2.minEnclosingCircle(pts)
        res[name] = {"center_px": [float(cx), float(cy)], "radius_px": float(r), "area_px": area,
                     "bbox": [int(st[i, 0]), int(st[i, 1]), int(st[i, 0] + st[i, 2]), int(st[i, 1] + st[i, 3])]}
    return res


def colour_name(bgr: np.ndarray, box) -> tuple[str, list[float]]:
    x0, y0, x1, y1 = box
    roi = bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return "?", [0, 0, 0]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
    sat = hsv[hsv[:, 1] > 60]
    if len(sat) < 20:
        v = hsv[:, 2].mean()
        return ("cream" if v > 150 else "brown"), hsv.mean(0).tolist()
    h = np.median(sat[:, 0])
    name = min(BRICK_COLOURS, key=lambda k: min(abs(h - BRICK_COLOURS[k]), 180 - abs(h - BRICK_COLOURS[k])))
    if name == "orange" and np.median(sat[:, 2]) < 140:
        name = "brown"
    return name, hsv.mean(0).tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-old", type=int, default=100)
    ap.add_argument("--n-cal2", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from dlsj.inspect.brick_detector import make_brick_detector

    det = make_brick_detector(str(Path("~/inference/models/brick-detector-v1").expanduser()), "CPU")
    out = a.out.expanduser()
    (out / "frames").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    records = []
    for set_name, (root, lab_root) in SETS.items():
        info = json.loads((Path(root) / "meta/info.json").read_text())
        fps = float(info["fps"])
        eps = episode_table(root)
        st = states(root)
        labels = {l["episode_index"]: l for l in map(json.loads, open(lab_root + "/meta/dlsj/labels.jsonl"))}
        usable = [e for e, l in labels.items() if not l.get("excluded") and e in eps.index and e in st]
        n = a.n_old if set_name == "old" else a.n_cal2
        pick = sorted(rng.choice(usable, size=min(n, len(usable)), replace=False).tolist())
        for e in pick:
            l = labels[e]
            g = st[e]
            S = np.stack(g["observation.state"].values)
            last = len(S) - 1
            gf, insp = int(l["grasp_frame"]), int(l["inspection_frame"])
            fr = {"start": 0, "grasp": gf, "mid2": (gf + insp) // 2, "inspect": insp, "end": last}
            try:
                top = decode(root, info, eps.loc[e], "top", sorted(set(fr.values())), fps)
                wrist = decode(root, info, eps.loc[e], "wrist", [0, gf], fps)
            except Exception as ex:  # noqa: BLE001
                print("decode fail", set_name, e, ex)
                continue
            rec = {"set": set_name, "episode": int(e), "frames": fr, "n_frames": int(len(S)), "fps": fps,
                   "bowl": l.get("bowl_final"), "state": {k: S[v].round(2).tolist() for k, v in fr.items()},
                   "dets": {}, "plates": {}}
            for k, f in fr.items():
                img = top.get(f)
                if img is None:
                    continue
                cv2.imwrite(str(out / "frames" / f"{set_name}_{e:03d}_{k}_top.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                ds = det.detect(img)
                rec["dets"][k] = [{"box": list(d.box), "conf": round(d.confidence, 3),
                                   "colour": colour_name(img, d.box)[0]} for d in ds]
                if k in ("start", "end"):
                    rec["plates"][k] = plates(img)
            for f, img in wrist.items():
                cv2.imwrite(str(out / "frames" / f"{set_name}_{e:03d}_{'start' if f == 0 else 'grasp'}_wrist.jpg"), img,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
            records.append(rec)
            print(set_name, e, {k: len(v) for k, v in rec["dets"].items()}, list(rec["plates"].get("start", {})), flush=True)
    (out / "episodes.json").write_text(json.dumps(records))
    print("DONE", len(records))


if __name__ == "__main__":
    main()
