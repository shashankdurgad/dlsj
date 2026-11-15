"""Held-cube YOLO26n v2 vs the live YOLO classifier v2 on our labelled rig spins (12 saved full frames each)."""
import collections, glob, json, sys, time
import cv2
sys.path.append("/home/intel/inference/pydeps/ultra")
from ultralytics import YOLO

M = "/home/intel/inference/models/held-cube-yolo26n-v2/held_cube_symbols_held_only_openvino_model"
model = YOLO(M, task="detect")
NAMES = {0: "cross", 1: "circle", 2: "plain"}
store = json.load(open("/home/intel/inference/datasets/rig_yolo/collect.json"))

def decide(frames, thr, n_base=12):
    conf = [f for f in frames if f and f[1] >= thr]
    marked = [f for f in conf if f[0] in ("circle", "cross")]
    plain = [f for f in conf if f[0] == "plain"]
    n = len(frames)
    need = max(1, round(3 * n / n_base))
    if len(marked) >= need:
        c = collections.Counter(f[0] for f in marked)
        return c.most_common(1)[0][0]
    if len(marked) <= max(1, n // 12) and len(plain) >= 0.8 * n:
        return "plain"
    return "uncertain"

rows, ms = [], []
for e in store["entries"]:
    fr = sorted(glob.glob(e["dirs"][0] + "/frame_*.png"))
    per = []
    for p in fr:
        img = cv2.imread(p)
        t = time.perf_counter()
        r = model.predict(img, imgsz=640, conf=0.05, device="npu", verbose=False)[0]
        ms.append((time.perf_counter() - t) * 1e3)
        if len(r.boxes):
            i = int(r.boxes.conf.argmax())
            per.append((NAMES[int(r.boxes.cls[i])], float(r.boxes.conf[i])))
        else:
            per.append(None)
    rows.append({"id": e["id"], "label": e["label"], "v2_live": e.get("pred"), "frames": per})

out = {"n_spins": len(rows), "ms_median": sorted(ms)[len(ms) // 2]}
for thr in (0.25, 0.5):
    fc = collections.Counter()
    spin = collections.Counter()
    conf_m = collections.Counter()
    for r in rows:
        for f in r["frames"]:
            fc["frames"] += 1
            if f and f[1] >= thr:
                fc["confident"] += 1
                fc["correct"] += f[0] == r["label"]
        d = decide(r["frames"], thr)
        r[f"held_{thr}"] = d
        spin["n"] += 1
        spin["correct"] += d == r["label"]
        spin["uncertain"] += d == "uncertain"
        spin["wrong"] += d not in (r["label"], "uncertain")
        conf_m[(r["label"], d)] += 1
    out[f"thr_{thr}"] = {"frames": dict(fc), "spins": dict(spin), "confusion": {f"{a}->{b}": n for (a, b), n in conf_m.items()}}
v2 = collections.Counter()
v2m = collections.Counter()
for r in rows:
    p = {"none": "plain", "unknown": "uncertain"}.get(r["v2_live"], r["v2_live"])
    v2["n"] += 1; v2["correct"] += p == r["label"]; v2["uncertain"] += p == "uncertain"
    v2["wrong"] += p not in (r["label"], "uncertain")
    v2m[(r["label"], p)] += 1
out["v2_live"] = {"spins": dict(v2), "confusion": {f"{a}->{b}": n for (a, b), n in v2m.items()}}
by = collections.Counter(r["label"] for r in rows)
out["spins_per_label"] = dict(by)
json.dump({"summary": out, "rows": rows}, open("/home/intel/inference/models/held-cube-yolo26n-v2/rig_eval.json", "w"), indent=1)
print(json.dumps(out, indent=1))
