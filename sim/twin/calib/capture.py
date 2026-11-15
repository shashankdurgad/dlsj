"""Capture calibration frames on the XPS: wrist (V4L2) + top / inspect (RealSense D415), 640x480 as recorded.

    python capture.py --seconds 90 --out ~/desmond/tmp/calib_capture
Saves PNGs at --rate fps per camera to <out>/<cam>/<n>.png and prints, every 5 s, how many frames so far show
the phone / laptop board (>= 6 ChArUco corners), so the person holding the board knows it is being seen.
Stop early with Ctrl-C or SIGTERM; cameras are released on exit.
"""

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from boards import board, detect  # noqa: E402

WRIST = "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0"
RS = {"top": "346422061366", "inspect": "243722062690"}

ap = argparse.ArgumentParser()
ap.add_argument("--seconds", type=float, default=90)
ap.add_argument("--rate", type=float, default=4.0)
ap.add_argument("--cams", default="wrist,inspect,top")
ap.add_argument("--out", default=str(Path.home() / "desmond/tmp/calib_capture"))
args = ap.parse_args()
cams = args.cams.split(",")
out = Path(args.out).expanduser()
stop = False
signal.signal(signal.SIGTERM, lambda *_: globals().__setitem__("stop", True))

boards = {"phone": board("phone", 0.015), "laptop": board("laptop", 0.03)}  # sizes irrelevant for detection
grab = {}
if "wrist" in cams:
    cap = cv2.VideoCapture(WRIST, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        sys.exit("wrist camera not available")
    grab["wrist"] = lambda: (lambda ok_f: cv2.cvtColor(ok_f[1], cv2.COLOR_BGR2RGB) if ok_f[0] else None)(cap.read())
pipes = {}
if any(c in RS for c in cams):
    import pyrealsense2 as rs

    for c in cams:
        if c in RS:
            p, cfg = rs.pipeline(), rs.config()
            cfg.enable_device(RS[c])
            cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
            p.start(cfg)
            pipes[c] = p
            grab[c] = (
                lambda p: lambda: __import__("numpy")
                .asanyarray(p.wait_for_frames().get_color_frame().get_data())
                .copy()
            )(p)
for c in cams:
    (out / c).mkdir(parents=True, exist_ok=True)
log = {c: [] for c in cams}
t0 = last = time.time()
n = 0
try:
    while not stop and time.time() - t0 < args.seconds:
        tick = time.time()
        for c in cams:
            img = grab[c]()
            if img is None:
                continue
            name = f"{n:05d}.png"
            cv2.imwrite(str(out / c / name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            seen = {b: (len(d[2]) if (d := detect(img, bd)) else 0) for b, bd in boards.items()}
            log[c].append({"file": name, "t": round(tick - t0, 2), **seen})
        n += 1
        if time.time() - last >= 5:
            last = time.time()
            print(
                f"{time.time() - t0:5.0f}s "
                + " | ".join(
                    f"{c}: phone {sum(r['phone'] > 0 for r in log[c])} "
                    f"laptop {sum(r['laptop'] > 0 for r in log[c])} /{len(log[c])}"
                    for c in cams
                ),
                flush=True,
            )
        time.sleep(max(0.0, 1 / args.rate - (time.time() - tick)))
except KeyboardInterrupt:
    pass
finally:
    for p in pipes.values():
        p.stop()
    if "wrist" in grab:
        cap.release()
    (out / "detections.json").write_text(json.dumps(log))
    print("CAPTURE_DONE", {c: len(v) for c, v in log.items()}, "->", out, flush=True)
