"""Print-ready ChArUco board for calibrating the twin's cameras (A4 landscape, 100 % scale).

7 x 5 squares of 35 mm, 26 mm DICT_5X5_100 markers: 245 x 175 mm, fits inside any printer's margins on A4.
After printing, measure one square with a ruler; if it isn't 35.0 mm, pass the measured size to the calibrator.
    uv run --with opencv-contrib-python-headless --with pillow python sim/twin/calib/make_board.py
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SQUARES, SQ_MM, MK_MM, DPI = (7, 5), 35.0, 26.0, 300
HERE = Path(__file__).resolve().parent
board = cv2.aruco.CharucoBoard(
    SQUARES, SQ_MM / 1000, MK_MM / 1000, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
)
def px(mm):
    return int(round(mm / 25.4 * DPI))
img = board.generateImage((px(SQUARES[0] * SQ_MM), px(SQUARES[1] * SQ_MM)), marginSize=0, borderBits=1)
page = np.full((px(210), px(297)), 255, np.uint8)
y0, x0 = (page.shape[0] - img.shape[0]) // 2, (page.shape[1] - img.shape[1]) // 2
page[y0 : y0 + img.shape[0], x0 : x0 + img.shape[1]] = img
cv2.putText(
    page,
    f"dlsj twin ChArUco 7x5, {SQ_MM:.0f} mm squares, {MK_MM:.0f} mm markers, DICT_5X5_100 "
    "- print at 100% (no fit-to-page)",
    (x0, y0 + img.shape[0] + px(7)),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.1,
    0,
    2,
)
Image.fromarray(page).save(HERE / "charuco_7x5_35mm_A4.pdf", resolution=DPI)
Image.fromarray(page).resize((page.shape[1] // 4, page.shape[0] // 4)).save(HERE / "charuco_preview.png")
print("wrote", HERE / "charuco_7x5_35mm_A4.pdf")
