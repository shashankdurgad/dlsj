"""Joint-offset fit, data extraction (XPS, read-only): top frames at several recorded poses per episode + joint states,
and grasp/inspection states for every usable episode. Output: <out>/frames/*.jpg + jofit_data.json

    PYTHONPATH=~/inference/dlsj-smolvla:<this dir> ~/desmond/venvs/teleop/bin/python jofit_extract.py --out ~/inference/tmp/jofit
"""
import argparse, json, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_episodes import SETS, episode_table, states, decode  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--n-old", type=int, default=70)
ap.add_argument("--n-cal2", type=int, default=50)
a = ap.parse_args()
out = Path(a.out).expanduser(); (out / "frames").mkdir(parents=True, exist_ok=True)
res = {"frames": [], "episodes": []}
rng = np.random.default_rng(1)
for set_name, (root, root3) in SETS.items():
    info = json.load(open(root + "/meta/info.json")); fps = float(info["fps"])
    labels = {l["episode_index"]: l for l in map(json.loads, open(root3 + "/meta/dlsj/labels.jsonl"))}
    S = states(root); eps = episode_table(root)
    usable = [e for e, l in labels.items() if not l.get("excluded") and l.get("grasp_frame") and l.get("inspection_frame") and e in S]
    for e in usable:  # key states for every usable episode (grasp / inspection terms)
        st = np.stack(S[e]["observation.state"].values); l = labels[e]
        res["episodes"].append({"set": set_name, "episode": int(e), "grasp": st[l["grasp_frame"]].round(3).tolist(),
                                "inspect": st[min(l["inspection_frame"], len(st) - 1)].round(3).tolist()})
    pick = rng.choice(usable, size=min(len(usable), a.n_old if set_name == "old" else a.n_cal2), replace=False)
    for e in pick:
        l = labels[int(e)]; st = np.stack(S[int(e)]["observation.state"].values); n = len(st)
        g, i = l["grasp_frame"], l["inspection_frame"]
        frs = sorted({max(0, g - 12), min(n - 1, g + 12), (g + i) // 2, min(n - 1, i), min(n - 1, i + 15), max(0, g // 2)})
        try:
            imgs = decode(root, info, eps.loc[int(e)], "top", frs, fps)
        except Exception as ex:  # noqa: BLE001
            print("decode fail", set_name, e, ex); continue
        for f in frs:
            if f not in imgs: continue
            tag = f"{set_name}_{int(e):03d}_{f:04d}"
            cv2.imwrite(str(out / "frames" / f"{tag}.jpg"), imgs[f], [cv2.IMWRITE_JPEG_QUALITY, 92])
            res["frames"].append({"tag": tag, "set": set_name, "episode": int(e), "frame": int(f), "state": st[f].round(3).tolist()})
    print(set_name, "episodes", len(usable), "frames so far", len(res["frames"]), flush=True)
(out / "jofit_data.json").write_text(json.dumps(res))
print("EXTRACT_DONE", len(res["frames"]), "frames", len(res["episodes"]), "episodes")
