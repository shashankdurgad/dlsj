"""Plain vs marked on the 38 rig spins, real decision rule: v5 (binary) vs v4 (circle/cross counted as marked)."""
import collections, glob, json, sys, time
import cv2
sys.path.append("/home/intel/inference/pydeps/ultra")
sys.path.insert(0, "/tmp/hc")
import auto_inspect as AI
from dlsj.inspect import aggregate as agg
from pathlib import Path

M = Path("/home/intel/inference/models")
insps = {"v5 plain/marked": AI.HeldCubeInspector(M / AI.HELD_CUBE_WEIGHTS["held_v5"], "NPU"),
         "v4 (circle/cross = marked)": AI.HeldCubeInspector(M / AI.HELD_CUBE_WEIGHTS["held_cube"], "NPU")}
store = json.load(open("/home/intel/inference/datasets/rig_yolo/collect.json"))
B = lambda x: {"none": "plain", "plain": "plain", "circle": "marked", "cross": "marked", "marked": "marked"}.get(x, "uncertain")
res = {}
for name, insp in insps.items():
    c, cm, ms, fr, misses = collections.Counter(), collections.Counter(), [], collections.Counter(), []
    for e in store["entries"]:
        truth = B(e["label"]); recs = []
        for i, p in enumerate(sorted(glob.glob(e["dirs"][0] + "/frame_*.png"))):
            t = time.perf_counter(); d = insp.classify(cv2.imread(p)[..., ::-1]); ms.append((time.perf_counter() - t) * 1e3)
            recs.append(insp.to_frame(i, 0.0, d)); fr["frames"] += 1
            if d["confident"]:
                fr["confident"] += 1; fr["correct"] += B(d["confident"]) == truth
        a = agg.aggregate(recs, 0.5, 1 - insp.min_conf, insp.min_conf, AI.scaled_agg(insp.agg, len(recs)), "yolo_topk")
        out = B(a.outcome().value)
        c["correct"] += out == truth; c["uncertain"] += out == "uncertain"; c["wrong"] += out not in (truth, "uncertain")
        cm[f"{truth}->{out}"] += 1
        if out != truth: misses.append((e["id"], e["label"], out))
    res[name] = {"spins": dict(c), "frames": dict(fr), "ms": round(sorted(ms)[len(ms) // 2], 1), "confusion": dict(cm), "misses": misses}
    print(name, json.dumps(res[name]))
json.dump(res, open("/home/intel/inference/models/held-brick-yolo26n-v5/rig_eval_binary.json", "w"), indent=1)
