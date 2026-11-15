"""Held-crop (v4 box -> v2 classifier) vs v4 alone vs v2 classifier (ROI/detector crop), real decision rule, 38 rig spins."""
import collections, glob, json, sys, time
import cv2
sys.path.append("/home/intel/inference/pydeps/ultra")
sys.path.insert(0, "/tmp/hc")
import auto_inspect as AI
from dlsj.inspect import aggregate as agg
from dlsj.inspect.yolo_inspector import YoloInspector
from pathlib import Path

M = Path("/home/intel/inference/models")
insps = {
    "held_crop (v4 box -> v2)": AI.HeldCropInspector(M / AI.HELD_CUBE_WEIGHTS["held_cube"], M / "brick-yolo-classifier-v2", "NPU"),
    "held_cube v4 alone": AI.HeldCubeInspector(M / AI.HELD_CUBE_WEIGHTS["held_cube"], "NPU"),
    "v2 classifier, ROI/detector crop": YoloInspector(M / "brick-yolo-classifier-v2", "NPU", padding=0.05,
                                                     detector_weights=str(M / "brick-detector-v1_openvino_model"),
                                                     detector_device="npu"),
}
store = json.load(open("/home/intel/inference/datasets/rig_yolo/collect.json"))
norm = {"none": "plain", "unknown": "uncertain"}
res = {}
for name, insp in insps.items():
    c = collections.Counter(); conf_m = collections.Counter(); ms = []; frames_ok = collections.Counter(); misses = []
    for e in store["entries"]:
        recs = []
        for i, p in enumerate(sorted(glob.glob(e["dirs"][0] + "/frame_*.png"))):
            t = time.perf_counter(); d = insp.classify(cv2.imread(p)[..., ::-1]); ms.append((time.perf_counter() - t) * 1e3)
            recs.append(insp.to_frame(i, 0.0, d))
            frames_ok["frames"] += 1
            if d["confident"]:
                frames_ok["confident"] += 1; frames_ok["correct"] += d["confident"] == e["label"]
        a = agg.aggregate(recs, 0.5, 1 - insp.min_conf, insp.min_conf, AI.scaled_agg(insp.agg, len(recs)), "yolo_topk")
        out = norm.get(a.outcome().value, a.outcome().value)
        c["correct"] += out == e["label"]; c["uncertain"] += out == "uncertain"; c["wrong"] += out not in (e["label"], "uncertain")
        conf_m[f"{e['label']}->{out}"] += 1
        if out != e["label"]:
            misses.append((e["id"], e["label"], out))
    res[name] = {"spins": dict(c), "frames": dict(frames_ok), "ms_median": round(sorted(ms)[len(ms) // 2], 1),
                 "confusion": dict(conf_m), "misses": misses}
    print(name, json.dumps(res[name]))
json.dump(res, open("/home/intel/inference/models/held_crop_rig_eval.json", "w"), indent=1)
