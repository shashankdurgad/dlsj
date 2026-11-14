"""Adapters from the real devices/back-ends to the loop's interfaces. Everything heavy is imported lazily.

* `StreamCameras`  : `dlsj.teleop_web.CameraStream` threads (BGR) -> `Cameras` (RGB, fresh-frame waits)
* `LeRobotArm`     : LeRobot SO-101 follower with the repo calibration (`dlsj.arms`) -> `Robot`
* `load_inspector` : `dlsj.inspect.detector.AnomalibInspector` (built by another workstream)
* policy back-ends : `OpenVINOPi05Policy`, `LeRobotServerPolicy` are stubs until deploy/pi05 lands
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from dlsj.contracts import Inspector, Observation
from dlsj.loop.poses import JOINT_KEYS, MOTORS

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------- cameras
class StreamCameras:
    def __init__(self, streams: dict):
        self.streams = streams  # name -> dlsj.teleop_web.CameraStream (started)
        self.names = list(streams)

    @classmethod
    def open(cls, spec: str, width: int = 640, height: int = 480, fps: int = 30, quality: int = 80) -> "StreamCameras":
        from dlsj.teleop_web import CameraStream, parse_cameras

        cams = parse_cameras(spec, 8)
        return cls({n: CameraStream(n, i, width, height, fps, quality).start() for n, i in cams.items()})

    def wait_ready(self, timeout_s: float = 20.0) -> None:
        t_end = time.monotonic() + timeout_s
        while time.monotonic() < t_end:
            if all(s.frame is not None for s in self.streams.values()):
                return
            time.sleep(0.1)
        missing = [n for n, s in self.streams.items() if s.frame is None]
        raise RuntimeError(f"no frames from cameras {missing}: {[self.streams[n].error for n in missing]}")

    def latest(self) -> dict[str, np.ndarray]:
        out = {}
        for n, s in self.streams.items():
            with s.cond:
                f = s.frame
            if f is not None:
                out[n] = np.ascontiguousarray(f[..., ::-1])  # BGR -> RGB
        return out

    def fresh(self, name: str, n: int, timeout_s: float = 2.0) -> list[np.ndarray]:
        s = self.streams[name]
        frames = []
        with s.cond:
            seq = s.seq
        t_end = time.monotonic() + timeout_s
        while len(frames) < n:
            with s.cond:
                if not s.cond.wait_for(lambda: s.seq != seq, timeout=max(0.0, t_end - time.monotonic())):
                    raise TimeoutError(f"camera {name}: only {len(frames)}/{n} new frames in {timeout_s}s")
                seq, f = s.seq, s.frame
            frames.append(np.ascontiguousarray(f[..., ::-1]))
        return frames

    def jpeg(self, name: str) -> bytes | None:
        s = self.streams.get(name)
        return s.jpeg if s else None

    def stop(self) -> None:
        for s in self.streams.values():
            s.stop()


# ----------------------------------------------------------------------------- robot
class LeRobotArm:
    """SO-101 follower, configured exactly like teleop_web/teleop_record so units match the training data."""

    def __init__(self, port: str = "auto", follower_id: str = "follower", max_relative_target: float | None = None,
                 calibrate: bool = False):
        from lerobot.robots import make_robot_from_config
        from lerobot.robots.so_follower import SO101FollowerConfig

        from dlsj import arms

        if port == "auto":
            follower, _, volts = arms.find_ports()
            if follower is None:
                raise RuntimeError(f"no follower arm found (servo supply voltages: {volts})")
            port = follower
        self.follower_id, self.calibrate = follower_id, calibrate
        self.robot = make_robot_from_config(SO101FollowerConfig(
            port=port, id=follower_id, calibration_dir=arms.FOLLOWER_CAL_DIR, max_relative_target=max_relative_target))
        if list(self.robot.action_features) != JOINT_KEYS:
            raise RuntimeError(f"unexpected joint order {list(self.robot.action_features)}")

    def connect(self) -> None:
        from dlsj import arms

        if arms.ensure_calibrated(self.robot, "follower", force=self.calibrate):
            stale = arms.retire_home(self.follower_id)
            if stale:
                log.warning("follower recalibrated: old home pose moved to %s; save a new one", stale.name)

    def disconnect(self) -> None:
        try:
            self.robot.disconnect()
        except Exception as e:  # noqa: BLE001
            log.warning("disconnect: %s", e)

    def read_state(self) -> np.ndarray:
        obs = self.robot.get_observation()
        return np.array([float(obs[k]) for k in JOINT_KEYS], dtype=np.float32)

    def send(self, target: np.ndarray) -> None:
        self.robot.send_action({k: float(v) for k, v in zip(JOINT_KEYS, target, strict=True)})

    def read_gripper_current(self) -> float | None:
        try:
            return float(self.robot.bus.sync_read("Present_Current", ["gripper"], normalize=False)["gripper"])
        except Exception as e:  # noqa: BLE001
            log.debug("Present_Current read failed: %s", e)
            return None

    def release_for_posing(self) -> None:
        self.robot.bus.disable_torque([m for m in MOTORS])


# ----------------------------------------------------------------------------- inspector
def load_inspector(path: str | Path | None, **kwargs) -> Inspector:
    """Load the Anomalib inspector from `dlsj.inspect`: a `manifest.json` (from `dlsj.inspect.train` or
    `dlsj.inspect.model_io calibrate`), or a standard Anomalib / Studio OpenVINO export folder or `.xml`
    (`dlsj.inspect.model_io`; pass `crop=` then). Also implements the sweep API (`inspect_sweep`, `start_sweep`).

    `AnomalibInspector.from_manifest(path, device="NPU", batch=3)` applies the uncertainty band itself (label
    UNCERTAIN), returns `region` in inspect-frame pixels and a frame-aligned `heatmap`, and expects RGB frames.
    Stage 2 (mark type) runs inside it: `stage2="shape" | "none" | <CNN classifier.json>`, `stage2_device=...`.
    """
    if path is None:
        raise ValueError("--inspector-path must point at a manifest.json from dlsj.inspect.train")
    try:
        from dlsj.inspect.detector import AnomalibInspector
    except ImportError as e:
        raise RuntimeError("dlsj.inspect.detector needs openvino and opencv; use --inspector fake for a dry run") from e
    return AnomalibInspector.from_manifest(path, **{k: v for k, v in kwargs.items() if v is not None})


# ----------------------------------------------------------------------------- policy back-ends
class OpenVINOPi05Policy:
    """pi0.5 exported to OpenVINO INT8 on the Arc GPU. Stub until the export workstream lands."""

    def __init__(self, model_dir: str | Path, device: str = "GPU"):
        self.model_dir, self.device = Path(model_dir), device
        # TODO(deploy/pi05): load the OpenVINO export (3 cameras batched into one vision call, PLAYBOOK 4.3/4.5),
        # tokenizer + state/action normalisation stats; honour the same image size / camera order as training.
        raise NotImplementedError("OpenVINO pi0.5 back-end: wire deploy/pi05 here (see TODO in dlsj/loop/hardware.py)")

    def reset(self) -> None:  # pragma: no cover
        pass

    def predict_chunk(self, obs: Observation, instruction: str) -> np.ndarray:  # pragma: no cover
        # TODO(deploy/pi05): preprocess obs.images (RGB uint8, keys wrist/top/inspect) + obs.state, run the
        # compiled model, unnormalise, return (n, 6) float32 joint targets in Observation.state units.
        raise NotImplementedError


class LeRobotServerPolicy:
    """Baseline: LeRobot PyTorch-XPU policy server (~/inference/.venv on the XPS). Stub."""

    def __init__(self, address: str = "127.0.0.1:8080"):
        self.address = address
        # TODO(deploy/pi05): connect to `lerobot-policy-server` (LeRobot async inference / --inference.type=remote),
        # send Observation as its observation dict + task string, receive the action chunk.
        raise NotImplementedError("LeRobot policy-server back-end: wire deploy/pi05 here "
                                  "(see TODO in dlsj/loop/hardware.py)")

    def reset(self) -> None:  # pragma: no cover
        pass

    def predict_chunk(self, obs: Observation, instruction: str) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError
