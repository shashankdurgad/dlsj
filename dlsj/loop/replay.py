"""Recorded-episode stand-ins for the robot and cameras: run the real models on real images without hardware.

`--robot fake --cameras replay:<dataset_root>:<episode>[:<stride>]` (run.py) loads one episode of a LeRobot dataset
recorded by `dlsj.teleop_record` (read-only, PyAV decoding) and builds:

* `ReplayCameras`: **state-matched replay**. Each frame request returns the recorded frame (wrist/top/inspect)
  whose recorded joint state is closest to the replay robot's *current* state, so the images stay consistent with
  where the policy has driven the arm (at the inspect pose you see the recorded brick in front of `inspect`).
  The scene does not react to the robot beyond that: it is an open-loop approximation, not a simulator.
* `ReplayRobot`: integrates the commanded joint targets with a first-order lag (like `fakes.FakeRobot`), and
  stalls the gripper at the recorded holding value while it is closed on the (imaginary) brick, so the pick
  skill's "closed then held" check behaves as on the real follower.
* `episode_poses`: home = first recorded state, inspect = state at the pick -> place task switch,
  `<colour>` = last recorded state (the bowl of this episode), for the skills' `near_pose` checks.

Nothing here opens a serial port or a camera.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dlsj.loop.poses import ARM, GRIPPER

log = logging.getLogger(__name__)

CAMERAS = ("wrist", "top", "inspect")


@dataclass
class Episode:
    """Decoded frames of one episode (every `stride`-th frame) plus every frame's state/action/task."""

    states: np.ndarray  # (N, 6) float32, recorder units
    actions: np.ndarray  # (N, 6)
    tasks: list[str]
    frame_idx: np.ndarray  # (M,) indices into states of the decoded frames
    images: dict[str, list[np.ndarray]]  # camera -> M frames HxWx3 uint8 RGB
    fps: float

    @property
    def split(self) -> int:
        """First frame whose task is not the first task (pick -> place switch); len-1 if none."""
        return next((i for i, t in enumerate(self.tasks) if t != self.tasks[0]), len(self.tasks) - 1)


def to_uint8_hwc(img) -> np.ndarray:
    """LeRobot frame tensor (C,H,W float [0,1]) -> HxWx3 uint8 (exact for video-decoded uint8 sources)."""
    a = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
    if a.ndim == 3 and a.shape[0] == 3:
        a = a.transpose(1, 2, 0)
    if a.dtype != np.uint8:
        a = np.clip(np.round(a * 255.0), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(a)


def open_dataset(root: str | Path, episodes: list[int] | None = None, delta_timestamps: dict | None = None):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(root).expanduser()
    return LeRobotDataset(root.name, root=root, episodes=episodes, video_backend="pyav",
                          delta_timestamps=delta_timestamps)


def task_names(ds) -> dict[int, str]:
    """task_index -> task string (LeRobot v3 `meta.tasks`: a DataFrame indexed by task with a task_index column)."""
    t = ds.meta.tasks
    if hasattr(t, "iterrows"):
        return {int(row["task_index"]): str(name) for name, row in t.iterrows()}
    return {int(k): str(v) for k, v in dict(t).items()}


def episode_range(ds, episode: int) -> tuple[int, int]:
    """Absolute [from, to) frame indices of `episode` in the dataset."""
    ep = ds.meta.episodes[episode]
    return int(ep["dataset_from_index"]), int(ep["dataset_to_index"])


def load_episode(root: str | Path, episode: int, stride: int = 2, cameras=CAMERAS) -> Episode:
    t0 = time.perf_counter()
    ds = open_dataset(root, episodes=[episode])
    lo, hi = episode_range(ds, episode)
    hf = ds.hf_dataset.with_format(None)
    rows = [r for r in hf if lo <= int(r["index"]) < hi]
    rows.sort(key=lambda r: int(r["index"]))
    states = np.array([r["observation.state"] for r in rows], dtype=np.float32)
    actions = np.array([r["action"] for r in rows], dtype=np.float32)
    names = task_names(ds)
    tasks = [names[int(r["task_index"])] for r in rows]
    frame_idx = np.arange(0, len(rows), max(1, stride))
    images: dict[str, list[np.ndarray]] = {c: [] for c in cameras}
    pos = {int(ds.hf_dataset[i]["index"]): i for i in range(len(ds.hf_dataset))}  # absolute -> local row
    for k in frame_idx:
        item = ds[pos[lo + int(k)]]
        for c in cameras:
            images[c].append(to_uint8_hwc(item[f"observation.images.{c}"]))
    log.info("replay: episode %d, %d frames (%d decoded, stride %d) in %.1f s", episode, len(rows), len(frame_idx),
             stride, time.perf_counter() - t0)
    return Episode(states, actions, tasks, frame_idx, images, float(ds.fps))


def parse_replay_spec(spec: str) -> tuple[str, int, int]:
    """'replay:<root>:<episode>[:<stride>]' -> (root, episode, stride)."""
    m = re.fullmatch(r"replay:(.+?):(\d+)(?::(\d+))?", spec)
    if not m:
        raise ValueError(f"--cameras replay spec must be replay:<dataset_root>:<episode>[:<stride>], got {spec!r}")
    return m.group(1), int(m.group(2)), int(m.group(3) or 2)


def episode_poses(ep: Episode) -> dict[str, np.ndarray]:
    poses = {"home": ep.states[0].copy(), "inspect": ep.states[max(0, ep.split - 1)].copy()}
    m = re.search(r"the (\w+) bowl", ep.tasks[-1])
    if m:
        poses[m.group(1)] = ep.states[-1].copy()
    return poses


def holding_grip(ep: Episode) -> float:
    """Follower gripper position while the brick is held (recorded state at the task switch)."""
    return float(ep.states[max(0, ep.split - 1), GRIPPER])


class ReplayRobot:
    def __init__(self, start: np.ndarray, hold_grip: float, lag: float = 0.6, release_margin: float = 12.0):
        self.state = np.asarray(start, np.float32).copy()
        self.cmd = self.state.copy()
        self.lag, self.hold_grip, self.release_margin = lag, hold_grip, release_margin
        self.holding = False
        self.lock = threading.Lock()

    def read_state(self) -> np.ndarray:
        with self.lock:
            return self.state.copy()

    def read_gripper_current(self) -> float | None:
        return 300.0 if self.holding else 20.0

    def send(self, target: np.ndarray) -> None:
        with self.lock:
            self.cmd = np.asarray(target, np.float32).copy()
            self.state = self.state + self.lag * (self.cmd - self.state)
            g = float(self.state[GRIPPER])
            if not self.holding and self.cmd[GRIPPER] < self.hold_grip and g <= self.hold_grip + 1.0:
                self.holding = True  # jaws reached the brick
            elif self.holding and self.cmd[GRIPPER] > self.hold_grip + self.release_margin:
                self.holding = False  # opened: brick released
            if self.holding:
                self.state[GRIPPER] = max(g, self.hold_grip)


class ReplayCameras:
    def __init__(self, ep: Episode, robot, fps: float | None = None):
        self.ep, self.robot = ep, robot
        self.names = [c for c in CAMERAS if ep.images.get(c)]
        self.period = 1.0 / (fps or ep.fps)
        self._states = ep.states[ep.frame_idx]  # (M, 6) recorded state of each decoded frame
        self._scale = np.ones(6, np.float32)
        self._scale[GRIPPER] = 0.3  # gripper (0-100) matters less than the arm joints for "where is the arm"
        self.last_index = 0

    def _index(self) -> int:
        s = self.robot.read_state()
        d = (self._states - s) * self._scale
        self.last_index = int(np.argmin(np.einsum("ij,ij->i", d, d)))  # weighted L2 in joint space
        return self.last_index

    def latest(self) -> dict[str, np.ndarray]:
        i = self._index()
        return {c: self.ep.images[c][i] for c in self.names}

    def fresh(self, name: str, n: int, timeout_s: float = 2.0) -> list[np.ndarray]:
        out = []
        for _ in range(n):
            time.sleep(self.period)  # a real camera delivers one new frame per period
            out.append(self.ep.images[name][self._index()])
        return out

    def jpeg(self, name: str) -> bytes | None:
        try:
            import cv2
        except ImportError:
            return None
        ok, buf = cv2.imencode(".jpg", self.ep.images[name][self.last_index][..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 70])
        return buf.tobytes() if ok else None

    def stop(self) -> None:
        pass


def arm_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a)[ARM] - np.asarray(b)[ARM])))
