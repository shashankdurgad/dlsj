"""Fakes for dry runs and tests: a tiny simulated workcell that closes the whole loop on a laptop.

* `FakeWorld`: a queue of bricks ("plain", "cross", "circle", "blob", "faint", "hidden_cross", "glare",
  "side_cross", "side_circle", "glint"), the brick on the table, the brick in the gripper, and where each brick
  ended up. "circle" is a ring (minor defect); "blob" is a solid dot: clearly marked for stage 1 but neither ring
  nor cross for stage 2 (stage-2 uncertain). "faint" shows a low-contrast cross (stage 1 is unsure every time);
  "hidden_cross" / "glare" look faint until the brick is turned (RESHOW), then show a clear cross / a plain face.
  Sweep bricks (the wrist spins the brick, `dlsj.loop.sweep`): "side_cross" / "side_circle" carry their mark on
  one face only, visible while wrist_roll is within `SIDE_FACE_DEG` of the inspect pose's roll (4 of the 12
  default stops, so a per-frame majority vote says plain); "glint" is plain but throws a specular highlight at
  `GLINT_DEG` (exactly one stop), which flips a max-score rule.
* `FakeRobot`: joints track the command with a first-order lag; the gripper stalls on a brick when it closes
  near the grasp pose (so "closed then held" and the EMPTY grasp check behave like the real arm).
* `FakeCameras`: renders `wrist` / `top` / `inspect` frames; `inspect` shows the held brick's face when the
  arm is at the inspect pose.
* `FakeInspector`: a real image computation on the rendered `inspect` frames (colour distance to the brick
  colour -> heatmap, p99 score, threshold, uncertainty band, bounding-box region), then stage 2 on the region
  crop: the real shape-rule classifier (`dlsj.inspect.classify`, needs OpenCV) or a NumPy stand-in without it.
* `ScriptedInspector`: returns a fixed sequence of labels / (label, mark, confidence) (unit tests).
* `ScriptedPolicy`: a reactive "VLA" that maps (state, instruction) to a straight-line chunk toward the
  current sub-goal, so it behaves sensibly no matter how stale its observation is (async-safe).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from dlsj.contracts import FrameInspection, InspectionResult, Label, MarkType, Observation
from dlsj.inspect.aggregate import AggregateConfig, SweepInspectorMixin, frame_label
from dlsj.loop.poses import ARM, GRIPPER
from dlsj.loop.skills import PLACE_BLUE, PLACE_MINOR, PLACE_PINK, Plates, default_skills

# Fake joint-space poses (degrees; gripper 0 = closed .. 100 = open)
DEFAULT_POSES = {
    "home": np.array([0, -100, 95, 10, 0, 2], dtype=np.float32),
    "grasp": np.array([0, -10, 30, 70, 0, 70], dtype=np.float32),
    "inspect": np.array([60, -40, 50, 0, 0, 0], dtype=np.float32),
    "yellow": np.array([-50, -20, 40, 50, 0, 0], dtype=np.float32),  # good bowl
    "purple": np.array([-90, -20, 40, 50, 0, 0], dtype=np.float32),  # defective bowl
    "orange": np.array([-130, -20, 40, 50, 0, 0], dtype=np.float32),  # minor-defect bowl
}
NOT_PLATES = ("home", "grasp", "inspect")
BRICK_RGB = np.array([235, 200, 40], dtype=np.float32)  # yellow brick
BG_RGB = np.array([90, 90, 95], dtype=np.float32)
HOLD_GRIP = 22.0  # where the jaws stall on a brick
TURN_DEG = 30.0
SIDE_FACE_DEG = (20.0, 80.0)  # wrist_roll offset (deg) from the inspect pose where a side_* brick shows its mark
GLINT_DEG = (-62.0, -52.0)  # offset where a "glint" brick reflects the light


def _near(a: np.ndarray, b: np.ndarray, tol: float) -> bool:
    return float(np.max(np.abs(a[ARM] - b[ARM]))) <= tol


@dataclass
class Brick:
    id: int
    kind: str
    turned: bool = False

    def face(self, roll_offset: float = 0.0) -> str:
        """What the inspect camera sees, given the wrist_roll offset from the inspect pose."""
        if self.kind in ("side_cross", "side_circle"):
            lo, hi = SIDE_FACE_DEG
            return self.kind.removeprefix("side_") if lo <= roll_offset <= hi else "plain"
        if self.kind == "glint":
            lo, hi = GLINT_DEG
            return "glint" if lo <= roll_offset <= hi else "plain"
        if self.kind in ("hidden_cross", "glare"):
            return ("cross" if self.kind == "hidden_cross" else "plain") if self.turned else "faint"
        return self.kind


@dataclass
class FakeWorld:
    bricks: list[str] = field(default_factory=lambda: ["plain", "cross"])
    poses: dict[str, np.ndarray] = field(default_factory=lambda: {k: v.copy() for k, v in DEFAULT_POSES.items()})

    def __post_init__(self):
        self._queue = [Brick(i + 1, k) for i, k in enumerate(self.bricks)]
        self.on_table: Brick | None = self._queue.pop(0) if self._queue else None
        self.held: Brick | None = None
        self.placed: list[tuple[int, str, str]] = []  # (brick id, kind, where)
        self.lock = threading.Lock()

    def feed(self) -> None:
        if self.on_table is None and self._queue:
            self.on_table = self._queue.pop(0)


class FakeRobot:
    def __init__(self, world: FakeWorld, start: np.ndarray | None = None, lag: float = 0.6):
        self.world = world
        self.state = (start if start is not None else world.poses["home"]).astype(np.float32).copy()
        self.cmd = self.state.copy()
        self.lag = lag
        self.sent: list[np.ndarray] = []

    def read_state(self) -> np.ndarray:
        with self.world.lock:
            return self.state.copy()

    def read_gripper_current(self) -> float | None:
        return 300.0 if self.world.held is not None else 20.0

    def send(self, target: np.ndarray) -> None:
        w = self.world
        with w.lock:
            self.cmd = np.asarray(target, dtype=np.float32).copy()
            self.sent.append(self.cmd)
            self.state = self.state + self.lag * (self.cmd - self.state)
            g = self.state[GRIPPER]
            if w.held is None and w.on_table is not None and g <= 30.0 and _near(self.state, w.poses["grasp"], 6.0):
                w.held, w.on_table = w.on_table, None
            if w.held is not None:
                self.state[GRIPPER] = max(g, HOLD_GRIP)
                if abs(self.state[4] - w.poses["inspect"][4]) >= TURN_DEG - 5:
                    w.held.turned = True
                if g > 32.0:  # opened: release
                    plates = [p for p in w.poses if p not in NOT_PLATES]
                    where = next((p for p in plates if _near(self.state, w.poses[p], 8.0)), "table")
                    w.placed.append((w.held.id, w.held.kind, where))
                    w.held = None
                    w.feed()


class FakeCameras:
    def __init__(self, world: FakeWorld, robot: FakeRobot, size: tuple[int, int] = (480, 640), seed: int = 0):
        self.world, self.robot = world, robot
        self.h, self.w = size
        self.names = ["wrist", "top", "inspect"]
        self.rng = np.random.default_rng(seed)
        self.roi = (self.w // 2 - 90, self.h // 2 - 70, self.w // 2 + 90, self.h // 2 + 70)  # brick box in `inspect`

    def _noise(self, img: np.ndarray) -> np.ndarray:
        n = self.rng.integers(-4, 5, size=img.shape, dtype=np.int16)
        return np.clip(img.astype(np.int16) + n, 0, 255).astype(np.uint8)

    def render(self, name: str) -> np.ndarray:
        img = np.empty((self.h, self.w, 3), dtype=np.float32)
        img[:] = BG_RGB
        w = self.world
        state = self.robot.read_state()
        if name == "inspect":
            ins = w.poses["inspect"]
            if w.held is not None and float(np.max(np.abs(state[:4] - ins[:4]))) <= 10.0:  # roll may differ
                x0, y0, x1, y1 = self.roi
                img[y0:y1, x0:x1] = BRICK_RGB
                face = w.held.face(float(state[4] - ins[4]))
                if face == "glint":  # specular highlight: a bright, soft-edged ellipse
                    yy, xx = np.mgrid[y0:y1, x0:x1]
                    g = ((xx - (x0 + 50)) / 30.0) ** 2 + ((yy - (y0 + 40)) / 16.0) ** 2 <= 1.0
                    img[y0:y1, x0:x1][g] = (255, 255, 250)
                if face in ("cross", "faint"):
                    a = 1.0 if face == "cross" else 0.45  # faint = low-contrast cross
                    col = BRICK_RGB * (1 - a)
                    for t in range(-6, 7):  # two thick diagonals
                        xs = np.arange(x0 + 20, x1 - 20)
                        ys1 = (y0 + 15 + (xs - x0 - 20) * (y1 - y0 - 30) / (x1 - x0 - 40)).astype(int) + t
                        ys2 = (y1 - 15 - (xs - x0 - 20) * (y1 - y0 - 30) / (x1 - x0 - 40)).astype(int) + t
                        img[ys1, xs] = col
                        img[ys2, xs] = col
                elif face in ("circle", "blob"):
                    yy, xx = np.mgrid[y0:y1, x0:x1]
                    r = np.hypot(xx - (x0 + x1) / 2, yy - (y0 + y1) / 2)
                    m = np.abs(r - 45) <= 6 if face == "circle" else r <= 30
                    img[y0:y1, x0:x1][m] = (25, 25, 30)
        elif name == "top" and w.on_table is not None:
            img[self.h // 2 - 20:self.h // 2 + 20, self.w // 2 - 30:self.w // 2 + 30] = BRICK_RGB
        elif name == "wrist" and w.held is not None:
            img[self.h // 3:, self.w // 4:3 * self.w // 4] = BRICK_RGB
        return self._noise(img)

    def latest(self) -> dict[str, np.ndarray]:
        return {n: self.render(n) for n in self.names}

    def fresh(self, name: str, n: int, timeout_s: float = 2.0) -> list[np.ndarray]:
        return [self.render(name) for _ in range(n)]

    def jpeg(self, name: str) -> bytes | None:
        try:
            import cv2
        except ImportError:
            return None
        ok, buf = cv2.imencode(".jpg", self.render(name)[..., ::-1])
        return buf.tobytes() if ok else None


class FakeInspector(SweepInspectorMixin):
    """Colour-distance 'anomaly model' on the inspect ROI. Same output contract as the Anomalib inspector,
    including the per-frame sweep API (`inspect_frame` / `inspect_sweep` / `start_sweep`)."""

    def __init__(self, roi: tuple[int, int, int, int], threshold: float = 0.5, band: float = 0.15,
                 ref_rgb: np.ndarray = BRICK_RGB, classifier="auto", agg: AggregateConfig | None = None):
        self.roi, self.threshold, self.band, self.ref = roi, threshold, band, ref_rgb.astype(np.float32)
        self.agg = agg or AggregateConfig()
        self.norm = float(np.linalg.norm(self.ref))  # distance from the brick colour to black
        if classifier == "auto":
            try:
                from dlsj.inspect.classify import ShapeRuleClassifier

                classifier = ShapeRuleClassifier()
            except ImportError:  # no OpenCV: NumPy stand-in below
                classifier = None
        self.classifier = classifier  # stage 2; None -> `_np_mark`

    @staticmethod
    def _np_mark(heat: np.ndarray, region) -> tuple[MarkType, float]:
        """NumPy stand-in for stage 2: ring = empty centre + filled rim; cross = filled centre + empty corners."""
        x0, y0, x1, y1 = region
        m = heat[y0:y1 + 1, x0:x1 + 1] > 0.5
        h, w = m.shape
        if h < 6 or w < 6:
            return MarkType.UNKNOWN, 0.0
        centre = m[h * 2 // 5:h * 3 // 5 + 1, w * 2 // 5:w * 3 // 5 + 1].mean()
        fill = m.mean()
        if fill > 0.6:
            return MarkType.UNKNOWN, 0.5  # solid blob
        return (MarkType.CROSS, 0.9) if centre > 0.5 else (MarkType.CIRCLE, 0.9)

    def _heat(self, frame: np.ndarray) -> np.ndarray:
        x0, y0, x1, y1 = self.roi
        heat = np.zeros(frame.shape[:2], dtype=np.float32)
        crop = frame[y0:y1, x0:x1].astype(np.float32)
        heat[y0:y1, x0:x1] = np.clip(np.linalg.norm(crop - self.ref, axis=2) / self.norm, 0.0, 1.0)
        return heat

    def inspect(self, frames: list[np.ndarray]) -> InspectionResult:
        t0 = time.perf_counter()
        x0, y0, x1, y1 = self.roi
        heats = [self._heat(f) for f in frames]
        scores = [float(np.percentile(h[y0:y1, x0:x1], 99)) for h in heats]
        score = float(np.median(scores))
        disagree = len({s > self.threshold for s in scores}) > 1
        if abs(score - self.threshold) < self.band or disagree:
            label = Label.UNCERTAIN
        else:
            label = Label.DEFECTIVE if score > self.threshold else Label.PERFECT
        heat = heats[int(np.argsort(scores)[len(scores) // 2])]
        region = None
        if label is not Label.PERFECT:
            ys, xs = np.nonzero(heat > 0.5 * max(score, 1e-6))
            if len(xs):
                region = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        t1 = time.perf_counter()
        lat = {"stage1": (t1 - t0) * 1000.0}
        mark, conf, mdev = None, 0.0, ""
        if label is Label.DEFECTIVE:
            if region is None:
                mark = MarkType.UNKNOWN
            elif self.classifier is not None:
                from dlsj.inspect.classify import crop_box

                bx0, by0, bx1, by1 = crop_box(region, frames[0].shape[:2], self.classifier.pad_frac)
                pred = self.classifier.classify([f[by0:by1, bx0:bx1] for f in frames],
                                                [heat[by0:by1, bx0:bx1]] * len(frames))
                mark, conf, mdev = pred.mark, pred.confidence, self.classifier.device
            else:
                (mark, conf), mdev = self._np_mark(heat, region), "FAKE-NP"
            lat["stage2"] = (time.perf_counter() - t1) * 1000.0
        return InspectionResult(label, score, self.threshold, region, heat, tuple(scores),
                                (time.perf_counter() - t0) * 1000.0, "FAKE-CPU", mark, float(conf), mdev, lat)


    # ------------------------------------------------------------------ sweep API
    def sweep_meta(self) -> dict:
        mdev = "FAKE-NP" if self.classifier is None else self.classifier.device
        return {"threshold": self.threshold, "band_low": self.threshold - self.band,
                "band_high": self.threshold + self.band, "device": "FAKE-CPU", "mark_device": mdev}

    def inspect_frame(self, frame: np.ndarray, index: int = 0, angle: float | None = None):
        t0 = time.perf_counter()
        x0, y0, x1, y1 = self.roi
        heat = self._heat(frame)
        score = float(np.percentile(heat[y0:y1, x0:x1], 99))
        label = frame_label(score, self.threshold - self.band, self.threshold + self.band)
        region = None
        if label is not Label.PERFECT:
            ys, xs = np.nonzero(heat > 0.5 * max(score, 1e-6))
            if len(xs):
                region = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        mark, conf, p = None, 0.0, None
        if label is Label.DEFECTIVE:
            if region is None:
                mark = MarkType.UNKNOWN
            elif self.classifier is not None:
                from dlsj.inspect.classify import crop_box

                bx0, by0, bx1, by1 = crop_box(region, frame.shape[:2], self.classifier.pad_frac)
                p = self.classifier.p_cross([frame[by0:by1, bx0:bx1]], [heat[by0:by1, bx0:bx1]])[0]
            else:
                m, c = self._np_mark(heat, region)
                p = None if m is MarkType.UNKNOWN and c == 0.0 else (0.5 if m is MarkType.UNKNOWN else
                                                                    (c if m is MarkType.CROSS else 1 - c))
            if mark is None:
                if p is None:
                    mark = MarkType.UNKNOWN
                else:
                    conf = max(p, 1 - p)
                    mark = (MarkType.CROSS if p >= 0.5 else MarkType.CIRCLE) if conf >= 0.7 else MarkType.UNKNOWN
        rec = FrameInspection(index, None if angle is None else float(angle), score, label, region, mark, float(conf),
                              None if p is None else float(p), (time.perf_counter() - t0) * 1000.0)
        return rec, heat


class ScriptedInspector:
    """Returns the given results in order (then repeats the last), with plausible scores.

    Each entry is a stage-1 `Label` (mark stays None: no stage 2) or `(Label.DEFECTIVE, MarkType, confidence)`.
    """

    def __init__(self, labels: list, threshold: float = 0.5):
        self.labels, self.threshold, self.calls = list(labels), threshold, 0

    def inspect(self, frames: list[np.ndarray]) -> InspectionResult:
        entry = self.labels[min(self.calls, len(self.labels) - 1)]
        self.calls += 1
        label, mark, conf = (entry, None, 0.0) if isinstance(entry, Label) else (tuple(entry) + (0.95,))[:3]
        score = {Label.PERFECT: 0.1, Label.DEFECTIVE: 0.9, Label.UNCERTAIN: 0.48}[label]
        region = (10, 10, 50, 50) if label is not Label.PERFECT else None
        lat = {"stage1": 1.0, **({"stage2": 0.5} if mark is not None else {})}
        return InspectionResult(label, score, self.threshold, region, None, (score,) * len(frames), 1.0, "SCRIPTED",
                                mark, float(conf), "SCRIPTED" if mark is not None else "", lat)


class ScriptedPolicy:
    """Reactive stand-in for pi0.5: picks a sub-goal from the state + instruction, returns a straight-line chunk.

    PICK:   approach the grasp pose with the jaws open -> close -> carry to the inspect pose
            (back to approach if the jaws closed on nothing).
    PLACE:  holding -> carry to the plate (pose named by the plate config); at the plate -> open.
    RESHOW: roll the wrist by TURN_DEG, then back to the inspect pose.
    """

    def __init__(self, poses: dict[str, np.ndarray] | None = None, chunk: int = 50, step_deg: float = 3.0,
                 grip_step: float = 8.0, latency_s: float = 0.0, plates: Plates | None = None):
        self.poses = poses or DEFAULT_POSES
        self.n, self.step, self.grip_step, self.latency_s = chunk, step_deg, grip_step, latency_s
        lib = default_skills()
        self.kind = {lib[k].instruction: k for k in lib}
        plates = plates or Plates()
        for p in plates.all():  # configured plate strings -> place skill ids
            self.kind[p.instruction] = p.skill
        self.plate_pose = {p.skill: p.pose for p in plates.all()}
        self.calls: list[str] = []
        self.reset()

    def reset(self) -> None:
        self._turned = False
        self._phase = "approach"

    def _goal(self, s: np.ndarray, instruction: str) -> np.ndarray:
        P = self.poses
        kind = self.kind.get(instruction)
        g = float(s[GRIPPER])
        if kind in ("PICK_AND_SHOW", "PICK_MARKED"):
            # forward-only phases driven by observed state (stale observations can't move it backwards in time)
            if self._phase == "approach" and _near(s, P["grasp"], 3.0) and g > 60.0:
                self._phase = "close"
            elif self._phase == "close" and g <= 30.0:
                self._phase = "carry"
            elif self._phase == "carry" and g < 8.0:  # closed on nothing: open and try again
                self._phase = "approach"
            goal = (P["inspect"] if self._phase == "carry" else P["grasp"]).copy()
            goal[GRIPPER] = 70.0 if self._phase == "approach" else 0.0
            return goal
        if kind in (PLACE_BLUE, PLACE_PINK, PLACE_MINOR, "PLACE_SLOT"):
            plate = P[self.plate_pose.get(kind, self.plate_pose[PLACE_BLUE])]
            goal = plate.copy()
            goal[GRIPPER] = 70.0 if (g > 30.0 or _near(s, plate, 2.0)) else 0.0
            return goal
        if kind == "RESHOW":
            turned = P["inspect"].copy()
            turned[4] += TURN_DEG
            if not self._turned and _near(s, turned, 2.0):
                self._turned = True
            return P["inspect"].copy() if self._turned else turned
        return s.copy()  # unknown instruction: hold

    def predict_chunk(self, obs: Observation, instruction: str) -> np.ndarray:
        self.calls.append(instruction)
        if self.latency_s:
            time.sleep(self.latency_s)
        s = obs.state.astype(np.float32)
        goal = self._goal(s, instruction)
        step = np.full(6, self.step, dtype=np.float32)
        step[GRIPPER] = self.grip_step
        out, pos = [], s.copy()
        for _ in range(self.n):
            pos = pos + np.clip(goal - pos, -step, step)
            out.append(pos.copy())
        return np.stack(out)
