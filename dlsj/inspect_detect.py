"""Detect when a held Duplo brick is fully in view of the inspection camera.

The same detector splits recorded demos (pick → place) and pauses the policy at deployment, so the
training split and the runtime pause happen at the same moment.

Method: Duplo bricks are strongly saturated colours, and the gripper, table and background mostly are
not. Inside a region of interest (ROI) we keep the high-saturation pixels, clean the mask up, and take
the largest blob. "In view" means all of the following:
* the blob is large enough to be the held brick, not one lying further away;
* the blob is not touching the ROI edges, so the brick is fully in frame;
* the blob has stayed put for `stable_frames` frames.

Limitations: white or grey bricks are not saturated. Use coloured bricks, or extend this with
background differencing.
"""

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass
class DetectorConfig:
    roi: tuple[float, float, float, float] = (0.1, 0.1, 0.9, 0.9)  # x0, y0, x1, y1 as fractions of the image
    min_sat: int = 100  # HSV saturation threshold (0-255)
    min_val: int = 50  # HSV value threshold: ignore very dark pixels
    min_area: float = 0.03  # blob area as a fraction of the ROI: smaller = too far away
    max_area: float = 0.7  # bigger = probably lighting or the background, not a brick
    edge_margin: int = 3  # pixels (after downscaling): a blob this close to the ROI edge is cut off
    stable_frames: int = 5  # consecutive frames the blob must stay put
    max_jitter: float = 0.04  # max centroid movement between frames, as a fraction of the ROI diagonal
    downscale: int = 2  # process at 1/downscale resolution for speed

    @staticmethod
    def parse_roi(text: str) -> tuple[float, float, float, float]:
        vals = tuple(float(v) for v in text.split(","))
        if len(vals) != 4 or not (0 <= vals[0] < vals[2] <= 1 and 0 <= vals[1] < vals[3] <= 1):
            raise ValueError("roi must be x0,y0,x1,y1 fractions with x0<x1, y0<y1")
        return vals  # type: ignore[return-value]


@dataclass
class Detection:
    in_view: bool  # stable, fully-framed brick
    candidate: bool  # a brick-sized blob fully inside the ROI this frame (not yet stable)
    reason: str
    area: float = 0.0  # blob area / ROI area
    bbox: tuple[int, int, int, int] | None = None  # x0, y0, x1, y1 in full-image pixels
    stable: int = 0

    def as_dict(self) -> dict:
        d = asdict(self)
        d["area"] = round(self.area, 4)
        return d


class BrickInViewDetector:
    def __init__(self, cfg: DetectorConfig | None = None):
        self.cfg = cfg or DetectorConfig()
        self._stable = 0
        self._last_centroid: np.ndarray | None = None
        self.last: Detection = Detection(False, False, "no frame yet")
        self.last_mask: np.ndarray | None = None

    def reset(self) -> None:
        self._stable = 0
        self._last_centroid = None

    def _roi_px(self, h: int, w: int) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = self.cfg.roi
        return int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)

    def update(self, bgr: np.ndarray) -> Detection:
        c = self.cfg
        small = bgr[:: c.downscale, :: c.downscale] if c.downscale > 1 else bgr
        h, w = small.shape[:2]
        rx0, ry0, rx1, ry1 = self._roi_px(h, w)
        roi = small[ry0:ry1, rx0:rx1]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = ((hsv[..., 1] >= c.min_sat) & (hsv[..., 2] >= c.min_val)).astype(np.uint8)
        k = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
        self.last_mask = mask
        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n <= 1:
            return self._miss("no coloured blob in the ROI")
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, bw, bh, area = stats[i]
        roi_area = float(roi.shape[0] * roi.shape[1])
        frac = area / roi_area
        bbox = (int((x + rx0) * c.downscale), int((y + ry0) * c.downscale),
                int((x + bw + rx0) * c.downscale), int((y + bh + ry0) * c.downscale))
        if frac < c.min_area:
            return self._miss("blob too small (brick too far or not there)", frac, bbox)
        if frac > c.max_area:
            return self._miss("blob too large (lighting/background?)", frac, bbox)
        m = c.edge_margin
        if x <= m or y <= m or x + bw >= roi.shape[1] - m or y + bh >= roi.shape[0] - m:
            return self._miss("brick cut off by the ROI edge", frac, bbox)
        centroid = cents[i] / np.hypot(*roi.shape[:2])
        if self._last_centroid is not None and np.linalg.norm(centroid - self._last_centroid) > c.max_jitter:
            self._stable = 0  # still moving
        self._last_centroid = centroid
        self._stable += 1
        stable = self._stable >= c.stable_frames
        self.last = Detection(stable, True, "in view" if stable else "settling", frac, bbox, self._stable)
        return self.last

    def _miss(self, reason: str, area: float = 0.0, bbox=None) -> Detection:
        self.reset()
        self.last = Detection(False, False, reason, area, bbox, 0)
        return self.last

    def debug_image(self, bgr: np.ndarray) -> np.ndarray:
        """`bgr` with the ROI, the colour mask tinted in, and the blob box (green = in view, amber = not yet)."""
        c, out = self.cfg, bgr.copy()
        h, w = out.shape[:2]
        x0, y0, x1, y1 = self._roi_px(h, w)
        if self.last_mask is not None:
            mask = cv2.resize(self.last_mask, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST).astype(bool)
            region = out[y0:y1, x0:x1]
            region[mask] = (0.5 * region[mask] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
        cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), (255, 255, 255), 1)
        d = self.last
        if d.bbox:
            colour = (0, 220, 0) if d.in_view else ((0, 190, 255) if d.candidate else (0, 0, 255))
            cv2.rectangle(out, d.bbox[:2], d.bbox[2:], colour, 2)
        label = f"{d.reason}  area {d.area:.3f}  stable {d.stable}/{c.stable_frames}"
        cv2.putText(out, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(out, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        return out
