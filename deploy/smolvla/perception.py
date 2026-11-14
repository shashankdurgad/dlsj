"""Top-camera perception for the closed loop: plates, bricks still on the table, their colours, and overlays.

Plain functions (numpy / OpenCV) so they can be tested on saved frames. The brick boxes come from the one-class
brick detector (`dlsj.inspect.brick_detector.BrickDetector`, NPU); everything else here is cheap CPU work.

* `find_plates(bgr)`: big yellow / orange / purple blobs -> circles (plates don't move, callers may cache them).
* `split_bricks(dets, plates, bgr, workspace)`: detector boxes -> bricks on the table (to pick) and bricks already in
  a plate (sorted), after size / work-area filtering; each with a training colour (blue/brown/red/white/yellow).
* `annotate_top(...)`: the overlay the dashboard streams.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

TRAINING_COLOURS = ("blue", "brown", "red", "white", "yellow")
PLATE_BGR = {"yellow": (48, 194, 242), "orange": (36, 138, 240), "purple": (224, 127, 155)}
BRICK_BGR = {"blue": (230, 150, 40), "brown": (40, 70, 120), "red": (40, 40, 220), "white": (235, 235, 235),
             "yellow": (40, 210, 240), "?": (160, 160, 160)}

# OpenCV HSV: H 0-179, S/V 0-255
PLATE_HSV = {
    # the yellow plate reads greenish under the venue light (H ~50, S ~80); the table stays below S 40
    "yellow": [((18, 90, 120), (38, 255, 255)), ((38, 55, 140), (64, 255, 255))],
    "orange": [((4, 140, 120), (17, 255, 255))],
    # the "purple" plate reads saturated light blue on the top camera (H ~102, S 255); size keeps blue bricks out
    "purple": [((120, 50, 50), (165, 255, 255)), ((92, 120, 150), (135, 255, 255))],
}


@dataclass
class Plate:
    colour: str
    cx: float
    cy: float
    r: float


@dataclass
class Brick:
    box: tuple[int, int, int, int]
    conf: float
    colour: str  # one of TRAINING_COLOURS, or "?" when unsure
    colour_conf: float
    plate: str | None = None  # the plate it sits in, None = on the table

    @property
    def centre(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.box
        return (x0 + x1) / 2, (y0 + y1) / 2

    def to_dict(self) -> dict:
        return {**asdict(self), "centre": [round(v, 1) for v in self.centre]}


def find_plates(bgr: np.ndarray, min_area: float = 7000.0) -> list[Plate]:
    """The largest blob of each plate colour, if big enough to be a plate (a brick is < ~3500 px at 640x480)."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(bgr, (7, 7), 0), cv2.COLOR_BGR2HSV)
    out = []
    for colour, ranges in PLATE_HSV.items():
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        if area / (np.pi * r * r) < 0.45:  # long thin streaks (cables, table edges) are not plates
            continue
        out.append(Plate(colour, float(cx), float(cy), float(r)))
    return out


def classify_colour(bgr_patch: np.ndarray) -> tuple[str, float]:
    """Training colour of a brick crop (centre region): (colour, confidence 0-1)."""
    h, w = bgr_patch.shape[:2]
    if h < 4 or w < 4:
        return "?", 0.0
    core = bgr_patch[h // 4: h - h // 4, w // 4: w - w // 4]
    hsv = cv2.cvtColor(core, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
    hue, sat, val = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    votes = {
        "white": (sat < 55) & (val > 150),
        "blue": (hue >= 85) & (hue <= 130) & (sat >= 70),
        "yellow": (hue >= 18) & (hue <= 42) & (sat >= 90) & (val >= 100),
        "red": ((hue <= 8) | (hue >= 165)) & (sat >= 110) & (val >= 70),
        # dark, reddish, often low-saturation on this table (small brown brick: S 32-81, V ~104; large: H 9, S 130, V 68)
        "brown": ((hue <= 25) | (hue >= 160)) & (sat >= 25) & (val < 125)
                 & ~(((hue <= 8) | (hue >= 165)) & (sat >= 110) & (val >= 70)),
    }
    frac = {k: float(v.mean()) for k, v in votes.items()}
    best = max(frac, key=frac.get)
    return (best, frac[best]) if frac[best] >= 0.25 else ("?", frac[best])


def split_bricks(dets, plates: list[Plate], bgr: np.ndarray, workspace=(0, 60, 560, 480),
                 min_side: int = 18, max_side: int = 110, min_conf: float = 0.15,
                 ignore=((330, 0, 470, 185),)) -> tuple[list[Brick], list[Brick]]:
    """Detector boxes -> (bricks on the table, bricks in a plate). Boxes off the work area, too small or too big
    (the plate itself, clutter) are dropped, as are boxes centred in an `ignore` region (the arm's home footprint:
    a brick still in the gripper at home is not on the table). Duplicate boxes keep the smaller.
    min_conf 0.15: a brown brick on the wood table scores ~0.18 (0.3 missed it on 15 Nov)."""
    x0w, y0w, x1w, y1w = workspace
    cands = []
    for d in dets:
        box = tuple(int(v) for v in (d.box if hasattr(d, "box") else d[0]))
        conf = float(d.confidence if hasattr(d, "confidence") else d[1])
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if conf < min_conf or not (min_side <= w <= max_side and min_side <= h <= max_side):
            continue
        if not (x0w <= cx < x1w and y0w <= cy < y1w):
            continue
        if any(ix0 <= cx <= ix1 and iy0 <= cy <= iy1 for ix0, iy0, ix1, iy1 in ignore):
            continue
        cands.append((box, conf))
    cands.sort(key=lambda bc: (bc[0][2] - bc[0][0]) * (bc[0][3] - bc[0][1]))
    kept: list[tuple] = []
    for box, conf in cands:
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        if any(k[0] <= cx <= k[2] and k[1] <= cy <= k[3] for k, _ in kept):
            continue
        kept.append((box, conf))
    table, sorted_ = [], []
    for box, conf in kept:
        colour, cc = classify_colour(bgr[box[1]:box[3], box[0]:box[2]])
        b = Brick(box, conf, colour, round(cc, 2))
        cx, cy = b.centre
        inside = [p for p in plates if (cx - p.cx) ** 2 + (cy - p.cy) ** 2 <= (0.95 * p.r) ** 2]
        if inside:
            b.plate = inside[0].colour
            sorted_.append(b)
        else:
            table.append(b)
    return table, sorted_


def choose_target(table: list[Brick], allowed=TRAINING_COLOURS) -> Brick | None:
    """Next brick to pick: a confidently coloured one (the pick prompt needs its colour), most confident first."""
    ok = [b for b in table if b.colour in allowed]
    if not ok:
        return None
    return max(ok, key=lambda b: (b.colour_conf > 0.4, b.conf))


def annotate_top(bgr: np.ndarray, table: list[Brick], sorted_: list[Brick], plates: list[Plate],
                 target: Brick | None, workspace=(0, 60, 560, 480), banner: str = "") -> np.ndarray:
    img = bgr.copy()
    x0, y0, x1, y1 = workspace
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (img.shape[1], y0), (0, 0, 0), -1)
    cv2.rectangle(overlay, (x1, 0), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)  # dim what the loop ignores (outside the policy crop)
    for p in plates:
        col = PLATE_BGR.get(p.colour, (200, 200, 200))
        cv2.circle(img, (int(p.cx), int(p.cy)), int(p.r), col, 2)
        n = sum(b.plate == p.colour for b in sorted_)
        cv2.putText(img, f"{p.colour} plate: {n}", (int(p.cx - p.r * 0.7), int(p.cy + p.r + 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)
    for b in sorted_:
        bx0, by0, bx1, by1 = b.box
        cv2.rectangle(img, (bx0, by0), (bx1, by1), (150, 150, 150), 1)
    for b in table:
        bx0, by0, bx1, by1 = b.box
        is_t = target is not None and b.box == target.box
        col = (0, 255, 120) if is_t else BRICK_BGR.get(b.colour, (200, 200, 200))
        cv2.rectangle(img, (bx0, by0), (bx1, by1), col, 3 if is_t else 2)
        label = f"{'TARGET ' if is_t else ''}{b.colour} {b.conf:.2f}"
        cv2.putText(img, label, (bx0, max(12, by0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 2, cv2.LINE_AA)
    if banner:
        cv2.rectangle(img, (0, img.shape[0] - 26), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
        cv2.putText(img, banner, (8, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return img
