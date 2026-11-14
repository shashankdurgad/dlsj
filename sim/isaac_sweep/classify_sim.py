import glob, re, sys, cv2
from dataclasses import replace
from dlsj.inspect.yolo_inspector import YoloInspector
from dlsj.inspect.aggregate import AggregateConfig
base = sys.argv[1]; roi = (200, 120, 440, 360)
cfgs = {"default": AggregateConfig(strategy="yolo_topk"),
        "conf0.95": AggregateConfig(strategy="yolo_topk", yolo_min_conf=0.95, yolo_plain_frac=0.6)}
insp = YoloInspector("/home/intel/inference/models/brick-yolo-classifier-v1", "NPU", roi=roi)
for m in ["plain", "cross", "circle"]:
    fs = sorted(glob.glob(f"{base}/{m}/*.png"))
    angles = [float(re.search(r"roll([+-]\d+)", f).group(1)) for f in fs]
    frames = [cv2.imread(f)[..., ::-1].copy() for f in fs]
    for name, cfg in cfgs.items():
        insp.agg = cfg
        r = insp.inspect_sweep(frames, angles, "yolo_topk")
        s = r.summary if hasattr(r, "summary") else {}
        print(f"{m:6s} {name:9s} -> {r.label.name:10s} mark={getattr(r,'mark_type',None)} | {s.get('reason','')} | {s.get('mark_reason','')}")
