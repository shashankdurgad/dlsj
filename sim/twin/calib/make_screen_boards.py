"""ChArUco boards shown on a screen (phone or laptop) instead of printed; specs in screen_boards.json.

phone : 4 x 6 squares, DICT_4X4_50 (portrait) - for the wrist and inspect cameras, 15-30 cm away
laptop: 7 x 5 squares, DICT_5X5_100 (same layout as the A4 PDF) - lying flat on the table for the top camera
The square size in metres depends on the screen: measure one square with a ruler and record it per device.
    uv run --no-project --with opencv-contrib-python-headless python sim/twin/calib/make_screen_boards.py
"""

import json
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
SPECS = {
    "phone": {"squares": [4, 6], "marker_ratio": 0.75, "dictionary": "DICT_4X4_50"},
    "laptop": {"squares": [7, 5], "marker_ratio": 26 / 35, "dictionary": "DICT_5X5_100"},
}
PX = 120  # pixels per square in the PNG (the page scales it up crisply)
for name, s in SPECS.items():
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, s["dictionary"]))
    b = cv2.aruco.CharucoBoard(tuple(s["squares"]), 1.0, s["marker_ratio"], d)
    img = b.generateImage((s["squares"][0] * PX, s["squares"][1] * PX), marginSize=0, borderBits=1)
    cv2.imwrite(str(HERE / f"screen_board_{name}.png"), img)
(HERE / "screen_boards.json").write_text(json.dumps(SPECS, indent=1))
print("ok")
