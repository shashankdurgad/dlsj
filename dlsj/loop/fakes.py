"""Fakes for dry runs and tests: a tiny simulated workcell that closes the whole loop on a laptop.

* `FakeWorld`: a queue of bricks ("plain", "cross", "faint", "hidden_cross", "glare"), the brick on the table,
  the brick in the gripper, and where each brick ended up. "faint" shows a low-contrast cross (the inspector
  is unsure every time); "hidden_cross" / "glare" look faint until the brick is turned (RESHOW), then show a
  clear cross / a plain face.
* `FakeRobot`: joints track the command with a first-order lag; the gripper stalls on a brick when it closes
  near the grasp pose (so "closed then held" and the EMPTY grasp check behave like the real arm).
* `FakeCameras`: renders `wrist` / `top` / `inspect` frames; `inspect` shows the held brick's face when the
  arm is at the inspect pose.
* `FakeInspector`: a real image computation on the rendered `inspect` frames (colour distance to the brick
  colour -> heatmap, p99 score, threshold, uncertainty band, bounding-box region).
* `ScriptedInspector`: returns a fixed sequence of labels (unit tests).
* `ScriptedPolicy`: a reactive "VLA" that maps (state, instruction) to a straight-line chunk toward the
  current sub-goal, so it behaves sensibly no matter how stale its observation is (async-safe).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from dlsj.contracts import InspectionResult, Label, Observation
from dlsj.loop.poses import ARM, GRIPPER
from dlsj.loop.skills import default_skills

# Fake joint-space poses (degrees; gripper 0 = closed .. 100 = open)
DEFAULT_POSES = {
    "home": np.array([0, -100, 95, 10, 0, 2], dtype=np.float32),
    "grasp": np.array([0, -10, 30, 70, 0, 70], dtype=np.float32),
    "inspect": np.array([60, -40, 50, 0, 0, 0], dtype=np.float32),
    "blue": np.array([-50, -20, 40, 50, 0, 0], dtype=np.float32),
    "pink": np.array([-90, -20, 40, 50, 0, 0], dtype=np.float32),
}
BRICK_RGB = np.array([235, 200, 40], dtype=np.float32)  # yellow brick
BG_RGB = np.array([90, 90, 95], dtype=np.float32)
HOLD_GRIP = 22.0  # where the jaws stall on a brick
TURN_DEG = 30.0


def _near(a: np.ndarray, b: np.ndarray, tol: float) -> bool:
    return float(np.max(np.abs(a[ARM] - b[ARM]))) <= tol


@dataclass
class Brick:
    id: int
    kind: str
    turned: bool = False

    def face(self) -> str:
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
                    where = next((p for p in ("blue", "pink") if _near(self.state, w.poses[p], 8.0)), "table")
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
            if w.held is not None and _near(state, w.poses["inspect"], 10.0):
                x0, y0, x1, y1 = self.roi
                img[y0:y1, x0:x1] = BRICK_RGB
                face = w.held.face()
                if face in ("cross", "faint"):
                    a = 1.0 if face == "cross" else 0.45  # faint = low-contrast cross
                    col = BRICK_RGB * (1 - a)
                    for t in range(-6, 7):  # two thick diagonals
                        xs = np.arange(x0 + 20, x1 - 20)
                        ys1 = (y0 + 15 + (xs - x0 - 20) * (y1 - y0 - 30) / (x1 - x0 - 40)).astype(int) + t
                        ys2 = (y1 - 15 - (xs - x0 - 20) * (y1 - y0 - 30) / (x1 - x0 - 40)).astype(int) + t
                        img[ys1, xs] = col
                        img[ys2, xs] = col
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


class FakeInspector:
    """Colour-distance 'anomaly model' on the inspect ROI. Same output contract as the Anomalib inspector."""

    def __init__(self, roi: tuple[int, int, int, int], threshold: float = 0.5, band: float = 0.15,
                 ref_rgb: np.ndarray = BRICK_RGB):
        self.roi, self.threshold, self.band, self.ref = roi, threshold, band, ref_rgb.astype(np.float32)
        self.norm = float(np.linalg.norm(self.ref))  # distance from the brick colour to black

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
        return InspectionResult(label, score, self.threshold, region, heat, tuple(scores),
                                (time.perf_counter() - t0) * 1000.0, "FAKE-CPU")


class ScriptedInspector:
    """Returns the given labels in order (then repeats the last), with plausible scores."""

    def __init__(self, labels: list[Label], threshold: float = 0.5):
        self.labels, self.threshold, self.calls = list(labels), threshold, 0

    def inspect(self, frames: list[np.ndarray]) -> InspectionResult:
        label = self.labels[min(self.calls, len(self.labels) - 1)]
        self.calls += 1
        score = {Label.PERFECT: 0.1, Label.DEFECTIVE: 0.9, Label.UNCERTAIN: 0.48}[label]
        region = (10, 10, 50, 50) if label is not Label.PERFECT else None
        return InspectionResult(label, score, self.threshold, region, None, (score,) * len(frames), 1.0, "SCRIPTED")


class ScriptedPolicy:
    """Reactive stand-in for pi0.5: picks a sub-goal from the state + instruction, returns a straight-line chunk.

    PICK:   approach the grasp pose with the jaws open -> close -> carry to the inspect pose
            (back to approach if the jaws closed on nothing).
    PLACE:  holding -> carry to the plate; at the plate -> open.
    RESHOW: roll the wrist by TURN_DEG, then back to the inspect pose.
    """

    def __init__(self, poses: dict[str, np.ndarray] | None = None, chunk: int = 50, step_deg: float = 3.0,
                 grip_step: float = 8.0, latency_s: float = 0.0):
        self.poses = poses or DEFAULT_POSES
        self.n, self.step, self.grip_step, self.latency_s = chunk, step_deg, grip_step, latency_s
        lib = default_skills()
        self.kind = {lib[k].instruction: k for k in lib}
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
        if kind in ("PLACE_BLUE", "PLACE_PINK", "PLACE_SLOT"):
            plate = P["pink"] if kind == "PLACE_PINK" else P["blue"]
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
