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


@dataclass(frozen=True)
class Det:
    box: tuple[int, int, int, int]
    confidence: float


def tile_grid(workspace, tile=(320, 240)) -> list[tuple[int, int, int, int]]:
    """Overlapping tiles covering the work area (2 x 2 for the default 560 x 420 area). Each tile is letterboxed to
    the detector's 640 px input, i.e. zoomed ~2x: the released detector (brick-detector-v1) was trained on
    inspection-camera close-ups, and the ~40 px top-camera bricks score 0.1-0.45 at 1x but 0.5-0.65 at 2x
    (live top frame, 15 Nov). Overlap >= one brick, so a brick cut by one tile is whole in its neighbour."""
    x0, y0, x1, y1 = workspace
    tw, th = min(tile[0], x1 - x0), min(tile[1], y1 - y0)
    nx, ny = max(1, -(-(x1 - x0) // tw)), max(1, -(-(y1 - y0) // th))
    xs = [x0 + round(i * (x1 - x0 - tw) / max(1, nx - 1)) for i in range(nx)] if nx > 1 else [x0]
    ys = [y0 + round(j * (y1 - y0 - th) / max(1, ny - 1)) for j in range(ny)] if ny > 1 else [y0]
    return [(x, y, x + tw, y + th) for y in ys for x in xs]


def merge_dets(dets, overlap: float = 0.5, extra=()) -> list[Det]:
    """Cross-tile NMS on overlap-over-the-smaller-box, highest confidence first. (A union of duplicates fused
    neighbouring bricks into one oversized box on live frames, 15 Nov: cut boxes are dropped in `detect_tiled`.)"""
    keep: list[Det] = []  # `extra` (lower priority) only fills in where none of `dets` overlaps
    for d in sorted(dets, key=lambda d: -d.confidence) + sorted(extra, key=lambda d: -d.confidence):
        b = d.box
        a = max(1, (b[2] - b[0]) * (b[3] - b[1]))
        if not any(max(0, min(b[2], k.box[2]) - max(b[0], k.box[0])) * max(0, min(b[3], k.box[3]) - max(b[1], k.box[1]))
                   / min(a, max(1, (k.box[2] - k.box[0]) * (k.box[3] - k.box[1]))) > overlap for k in keep):
            keep.append(d)
    return keep


def detect_tiled(detector, bgr: np.ndarray, workspace, tile=(320, 240), edge: int = 2) -> list[Det]:
    """The detector over zoomed overlapping tiles of the work area, boxes in full-frame pixels, merged. A box
    touching a tile border inside the work area may be a brick cut by the tile: the overlap shows it whole in the
    neighbouring tile, so cut boxes only survive the merge when no whole box overlaps them (the neighbour can miss
    a brick sitting right at the border: live frame 15 Nov, 0.58 in one tile, nothing in the other)."""
    wx0, wy0, wx1, wy1 = workspace
    whole, cut = [], []
    for tx0, ty0, tx1, ty1 in tile_grid(workspace, tile):
        for d in detector.detect(np.ascontiguousarray(bgr[ty0:ty1, tx0:tx1])):
            x0, y0, x1, y1 = d.box[0] + tx0, d.box[1] + ty0, d.box[2] + tx0, d.box[3] + ty0
            if ((x0 <= tx0 + edge and tx0 > wx0) or (x1 >= tx1 - edge and tx1 < wx1)
                    or (y0 <= ty0 + edge and ty0 > wy0) or (y1 >= ty1 - edge and ty1 < wy1)):
                cut.append(Det((x0, y0, x1, y1), float(d.confidence)))
            else:
                whole.append(Det((x0, y0, x1, y1), float(d.confidence)))
    return merge_dets(whole, extra=cut)


def persistent_bricks(frames: list[list[Brick]], min_frac: float = 0.5, dist: float = 20.0) -> list[dict]:
    """Bricks seen at the same place (centre within `dist` px) in >= `min_frac` of the frames: the final
    "is the table really empty" check counts these, so one-frame flickers neither block nor fake an empty table."""
    if not frames:
        return []
    clusters: list[dict] = []
    for i, bricks in enumerate(frames):
        for b in bricks:
            cx, cy = b.centre
            for c in clusters:
                if (cx - c["cx"]) ** 2 + (cy - c["cy"]) ** 2 <= dist ** 2:
                    c["frames"].add(i)
                    c["best"] = max(c["best"], b, key=lambda x: x.conf)
                    break
            else:
                clusters.append({"cx": cx, "cy": cy, "frames": {i}, "best": b})
    need = max(1, int(np.ceil(min_frac * len(frames))))
    return [{**c["best"].to_dict(), "seen": len(c["frames"]), "of": len(frames)}
            for c in clusters if len(c["frames"]) >= need]


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


def inspect_reference(bgr: np.ndarray) -> np.ndarray:
    """Blurred grey inspection frame with the arm away: the background `select_held` compares against."""
    return cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (9, 9), 0)


def select_held(bgr: np.ndarray, dets, ref: np.ndarray | None, min_frac: float = 0.35, min_side: int = 24,
                diff: int = 28):
    """(detection, foreground fraction) of the brick in the gripper on the fixed inspection camera.

    The held brick is what differs from `ref` (taken with the arm away); bricks lying on the table behind it don't,
    so they are never chosen. Without a reference: the largest detection. None if nothing qualifies."""
    dets = [d for d in dets if min(d.box[2] - d.box[0], d.box[3] - d.box[1]) >= min_side]
    if not dets:
        return None, 0.0
    if ref is None or ref.shape != bgr.shape[:2]:
        return max(dets, key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1])), 0.0
    g = inspect_reference(bgr)
    fg = cv2.morphologyEx((cv2.absdiff(g, ref) > diff).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    best, best_score, best_frac = None, 0.0, 0.0
    for d in dets:
        x0, y0, x1, y1 = (int(v) for v in d.box)
        frac = float(fg[y0:y1, x0:x1].mean()) if x1 > x0 and y1 > y0 else 0.0
        score = frac * np.sqrt((x1 - x0) * (y1 - y0))
        if frac >= min_frac and score > best_score:
            best, best_score, best_frac = d, score, frac
    return best, best_frac
