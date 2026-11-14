"""The screen / printed ChArUco boards (specs in screen_boards.json; A4 PDF = 'laptop' layout at 35 mm)."""

import json
from pathlib import Path

import cv2
import numpy as np

SPECS = json.loads((Path(__file__).resolve().parent / "screen_boards.json").read_text())


def board(name: str, square_m: float) -> cv2.aruco.CharucoBoard:
    s = SPECS[name]
    b = cv2.aruco.CharucoBoard(
        tuple(s["squares"]),
        square_m,
        square_m * s["marker_ratio"],
        cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, s["dictionary"])),
    )
    b.setLegacyPattern(False)  # boards were generated with OpenCV >= 4.6 layout
    return b


def detect(img_rgb: np.ndarray, b: cv2.aruco.CharucoBoard, min_corners: int = 6):
    """-> (object points (N,3), image points (N,2), ids) or None."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY) if img_rgb.ndim == 3 else img_rgb
    corners, ids, _, _ = cv2.aruco.CharucoDetector(b).detectBoard(gray)
    if ids is None or len(ids) < min_corners:
        return None
    obj, imgp = b.matchImagePoints(corners, ids)
    return obj.reshape(-1, 3), imgp.reshape(-1, 2), ids.ravel()
