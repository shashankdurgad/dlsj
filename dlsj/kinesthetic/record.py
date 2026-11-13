"""Kinesthetic-teaching recorder for the SO-101 follower: move the real arm by hand, record a LeRobot dataset.

The arm joints have torque off and you guide the follower itself; there is no leader arm. The gripper
stays powered and is driven by keys or a USB foot pedal, so both hands can stay on the arm. You can pass
`--gripper_mode=free` to move the gripper by hand too.

The output is a standard LeRobot v3 dataset with the same features as `lerobot-record` on an
`so101_follower`: `observation.state`, `observation.images.*` and `action`, all in the robot's
calibrated units. That means kinesthetic and teleop datasets can be trained on together. Extra data
goes in `meta/dlsj/`, outside the dataset features, so merging is not affected:

* `labels.jsonl`: one line per saved episode, holding success/failure, duration, operator and method.
* `telemetry/episode_XXXXXX.npz`: per-frame raw `Present_Current` / `Present_Load`. This is logged
  but is not a policy input.

The action is `action[t] = measured_position[t + action_lookahead]`, except for a key-driven gripper,
where it is the commanded target at `t`.

Example:

    uv run python -m dlsj.kinesthetic.record \
        --robot.type=so101_follower --robot.port=/dev/tty.usbmodemXXXX --robot.id=follower_a \
        --robot.cameras="{wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, \
                          top: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
        --dataset.repo_id=$HF_USER/duplo_kinesthetic --dataset.single_task="Sort the Duplo bricks by colour" \
        --dataset.fps=30 --dataset.episode_time_s=30 --dataset.push_to_hub=false \
        --dataset.streaming_encoding=true --operator=alice --target_s=8

Keys (a foot pedal can map to any of these):
  s / Enter        start an episode (releases arm torque, so hold the arm)
  space / g        toggle the gripper open/closed     up / down: nudge the gripper
  n / Right        save as SUCCESS                    f: save as FAILURE
  r / Left / Bksp  discard the episode
  h                (between episodes) hold or release the arm where it is
  q / Esc          quit
"""

import json
import logging
import queue
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401  (registers camera type)
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.datasets import LeRobotDataset, VideoEncodingManager
from lerobot.robots import RobotConfig, make_robot_from_config, so_follower  # noqa: F401
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features
from lerobot.utils.keyboard_input import create_key_listener
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging

from .session import ActionLookahead, Command, EpisodeStats, GripperCommand, KeyMap, Phase, SessionMachine

GRIPPER = "gripper"


@dataclass
class KinestheticRecordConfig:
    robot: RobotConfig
    dataset: DatasetRecordConfig = field(default_factory=DatasetRecordConfig)
    # "keys": gripper powered, toggled by keys/pedal (recommended). "free": gripper moved by hand too.
    gripper_mode: str = "keys"
    # Gripper targets in LeRobot's normalised 0-100 units. Check them with the idle status line first.
    gripper_open: float = 50.0
    gripper_closed: float = 0.0
    gripper_step: float = 5.0
    # action[t] = measured position at t + lookahead frames (compensates position-controller lag).
    action_lookahead: int = 1
    # Tempo target per episode in seconds: shown live to push operators toward fast demos.
    target_s: float | None = None
    # Log raw Present_Current / Present_Load per frame to meta/dlsj/telemetry (not a policy input).
    telemetry: bool = True
    operator: str = ""
    resume: bool = False

    def __post_init__(self) -> None:
        if self.gripper_mode not in ("keys", "free"):
            raise ValueError("--gripper_mode must be 'keys' or 'free'")
        if not self.dataset.repo_id or not self.dataset.single_task:
            raise ValueError("--dataset.repo_id and --dataset.single_task are required")


def _status(msg: str) -> None:
    sys.stdout.write("\r\033[K" + msg)
    sys.stdout.flush()


def _say(msg: str) -> None:
    sys.stdout.write("\r\033[K" + msg + "\n")
    sys.stdout.flush()


class KinestheticRecorder:
    def __init__(self, cfg: KinestheticRecordConfig, robot=None):
        self.cfg = cfg
        self.robot = robot if robot is not None else make_robot_from_config(cfg.robot)
        if not hasattr(self.robot, "bus"):
            raise TypeError("This recorder needs a serial-bus arm such as so101_follower.")
        self.bus = self.robot.bus
        self.joint_names = list(self.robot.action_features)  # e.g. "shoulder_pan.pos", ..., "gripper.pos"
        self.free_motors = [m for m in self.bus.motors if m != GRIPPER or cfg.gripper_mode == "free"]
        self.keymap = KeyMap()
        self.machine = SessionMachine()
        self.lookahead = ActionLookahead(cfg.action_lookahead, self.joint_names)
        self.gripper = GripperCommand(cfg.gripper_open, cfg.gripper_closed, cfg.gripper_step)
        self.stats = EpisodeStats()
        self.keys: queue.Queue[str] = queue.Queue()
        self.arm_held = True  # LeRobot's configure() leaves torque on after connect
        self.dataset: LeRobotDataset | None = None
        self._telemetry: list[tuple[np.ndarray, np.ndarray]] = []
        self._episode_start = 0.0
        self._frames_written = 0
        self._ticks = 0

    # ---------------------------------------------------------------- setup
    def _features(self) -> dict:
        video = self.cfg.dataset.video
        return {
            **hw_to_dataset_features(self.robot.action_features, ACTION, video),
            **hw_to_dataset_features(self.robot.observation_features, OBS_STR, video),
        }

    def _open_dataset(self) -> LeRobotDataset:
        d = self.cfg.dataset
        n_cams = len(getattr(self.robot, "cameras", {}))
        common = dict(
            root=d.root,
            batch_encoding_size=d.video_encoding_batch_size,
            rgb_encoder=d.rgb_encoder,
            depth_encoder=d.depth_encoder,
            encoder_threads=d.encoder_threads,
            streaming_encoding=d.streaming_encoding,
            encoder_queue_maxsize=d.encoder_queue_maxsize,
            image_writer_processes=d.num_image_writer_processes if n_cams else 0,
            image_writer_threads=d.num_image_writer_threads_per_camera * n_cams,
        )
        if self.cfg.resume:
            ds = LeRobotDataset.resume(d.repo_id, **common)
            if ds.features.keys() - {"timestamp", "frame_index", "episode_index", "index", "task_index"} != set(
                self._features()
            ):
                raise ValueError("Existing dataset features differ from this robot/camera setup.")
            return ds
        d.stamp_repo_id()
        return LeRobotDataset.create(
            d.repo_id, d.fps, robot_type=self.robot.robot_type, features=self._features(), use_videos=d.video,
            **common,
        )

    @property
    def sidecar(self) -> Path:
        p = Path(self.dataset.root) / "meta" / "dlsj"
        (p / "telemetry").mkdir(parents=True, exist_ok=True)
        return p

    # ---------------------------------------------------------------- torque
    def _hold_arm(self) -> None:
        """Torque on where the arm is now. Goal is set to the present position first so it doesn't jump."""
        present = self.bus.sync_read("Present_Position", self.free_motors)
        self.bus.sync_write("Goal_Position", present)
        self.bus.enable_torque(self.free_motors)
        self.arm_held = True

    def _release_arm(self) -> None:
        self.bus.disable_torque(self.free_motors)
        self.arm_held = False

    def _command_gripper(self) -> None:
        if self.cfg.gripper_mode == "keys":
            self.bus.write("Goal_Position", GRIPPER, self.gripper.target)

    # ---------------------------------------------------------------- episode
    def _start_episode(self) -> None:
        if self.arm_held:
            self._release_arm()
        self.lookahead.reset()
        self._telemetry = []
        self._frames_written = 0
        self._ticks = 0
        self._episode_start = time.perf_counter()
        _say(f"● REC episode {self.dataset.num_episodes} (n=success f=failure r=discard)")

    def _write(self, frames: list[dict]) -> None:
        for f in frames:
            f[ACTION] = np.asarray(f.pop("action"), dtype=np.float32)
            self.dataset.add_frame(f)
            self._frames_written += 1

    def _finish_episode(self, cmd: Command) -> None:
        duration = time.perf_counter() - self._episode_start
        if cmd is Command.DISCARD:
            self.lookahead.reset()
            self.dataset.clear_episode_buffer()
            _say(f"✗ discarded ({duration:.1f}s)")
            return
        self._write(self.lookahead.flush())
        if self._frames_written == 0:
            self.dataset.clear_episode_buffer()
            _say("✗ nothing recorded, discarded")
            return
        ep = self.dataset.num_episodes
        self.dataset.save_episode()
        success = cmd is Command.SAVE_SUCCESS
        label = {
            "episode_index": ep,
            "success": success,
            "duration_s": round(duration, 3),
            "n_frames": self._frames_written,
            "achieved_fps": round(self._ticks / duration, 2) if duration > 0 else None,
            "method": "kinesthetic",
            "gripper_mode": self.cfg.gripper_mode,
            "action_lookahead": self.cfg.action_lookahead,
            "operator": self.cfg.operator,
            "task": self.cfg.dataset.single_task,
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
        }
        with open(self.sidecar / "labels.jsonl", "a") as fh:
            fh.write(json.dumps(label) + "\n")
        if self._telemetry:
            cur, load = (np.stack(x) for x in zip(*self._telemetry, strict=True))
            np.savez_compressed(
                self.sidecar / "telemetry" / f"episode_{ep:06d}.npz",
                motors=np.array(list(self.bus.motors)), present_current=cur, present_load=load,
            )
        if success:
            self.stats.add(duration)
        _say(f"✓ saved episode {ep} as {'SUCCESS' if success else 'FAILURE'} in {duration:.1f}s "
             f"({label['achieved_fps']} fps) | {self.stats.summary()}")

    def _record_tick(self) -> None:
        obs = self.robot.get_observation()
        state = [float(obs[n]) for n in self.joint_names]
        frame = build_dataset_frame(self.dataset.features, obs, prefix=OBS_STR)
        frame["task"] = self.cfg.dataset.single_task
        cmds = {f"{GRIPPER}.pos": self.gripper.target} if self.cfg.gripper_mode == "keys" else None
        self._write(self.lookahead.push(frame, state, cmds))
        if self.cfg.telemetry:
            self._telemetry.append((
                np.array(list(self.bus.sync_read("Present_Current", normalize=False).values()), dtype=np.float32),
                np.array(list(self.bus.sync_read("Present_Load", normalize=False).values()), dtype=np.float32),
            ))
        self._ticks += 1

    # ---------------------------------------------------------------- keys
    def _handle(self, cmd: Command) -> bool:
        """Apply one accepted command. Returns False when the session should end."""
        if cmd is Command.QUIT:
            return False
        if cmd is Command.START:
            self._start_episode()
        elif cmd in (Command.SAVE_SUCCESS, Command.SAVE_FAILURE, Command.DISCARD):
            self._finish_episode(cmd)
        elif cmd is Command.GRIPPER_TOGGLE:
            self.gripper.toggle()
            self._command_gripper()
        elif cmd is Command.GRIPPER_OPEN_MORE:
            self.gripper.nudge(towards_open=True)
            self._command_gripper()
        elif cmd is Command.GRIPPER_CLOSE_MORE:
            self.gripper.nudge(towards_open=False)
            self._command_gripper()
        elif cmd is Command.TOGGLE_HOLD:
            if self.arm_held:
                self._release_arm()
                _say("arm released (torque off): hold it")
            else:
                self._hold_arm()
                _say("arm held (torque on)")
        return True

    def _drain_keys(self) -> bool:
        while True:
            try:
                key = self.keys.get_nowait()
            except queue.Empty:
                return True
            raw = self.keymap.lookup(key)
            if raw is None:
                continue
            was_recording = self.machine.phase is not Phase.IDLE
            if raw is Command.QUIT and was_recording:
                # Never lose a demo silently: quitting mid-episode discards it explicitly.
                self.machine.handle(Command.DISCARD)
                self._finish_episode(Command.DISCARD)
                return False
            cmd = self.machine.handle(raw)
            if cmd is not None and not self._handle(cmd):
                return False

    # ---------------------------------------------------------------- main loop
    def run(self) -> None:
        cfg = self.cfg
        period = 1.0 / cfg.dataset.fps
        listener = None
        self.robot.connect()  # fail on a wrong port before creating an empty dataset
        try:
            self.dataset = self._open_dataset()
            self._hold_arm()
            self._command_gripper()
            listener = create_key_listener(self.keys.put, controls_help="s start, n/f save, r discard, q quit")
            print("Keys" + __doc__.split("Keys", 1)[1])
            _say(f"dataset: {self.dataset.root}")
            _say("Arm is HELD. Grab it, then press s to start (torque is released when recording starts).")
            with VideoEncodingManager(self.dataset):
                last_status = 0.0
                while True:
                    t0 = time.perf_counter()
                    if not self._drain_keys():
                        break
                    phase = self.machine.phase
                    if phase is Phase.RECORDING:
                        self._record_tick()
                        elapsed = t0 - self._episode_start
                        if elapsed >= cfg.dataset.episode_time_s:
                            self.machine.time_limit_reached()
                            _say(f"time limit {cfg.dataset.episode_time_s}s: label it n (success), f (failure) "
                                 "or r (discard)")
                        elif t0 - last_status > 0.2:
                            tgt = ""
                            if cfg.target_s:
                                tgt = f" / target {cfg.target_s:.1f}s" + (" ⚠ OVER" if elapsed > cfg.target_s else "")
                            grip = "open" if self.gripper.is_open else "closed"
                            _status(f"● {elapsed:5.1f}s{tgt} | gripper {grip} ({self.gripper.target:.0f})")
                            last_status = t0
                    elif phase is Phase.IDLE and t0 - last_status > 0.2:
                        g = self.bus.read("Present_Position", GRIPPER)
                        _status(f"idle | arm {'HELD' if self.arm_held else 'free'} | gripper measured {g:.1f} "
                                f"target {self.gripper.target:.0f} | {self.stats.summary()} | s=start h=hold q=quit")
                        last_status = t0
                    precise_sleep(max(0.0, period - (time.perf_counter() - t0)))
        finally:
            if self.machine.phase is not Phase.IDLE and self.dataset is not None:
                self.dataset.clear_episode_buffer()
            if self.dataset is not None:
                self.dataset.finalize()
            if self.robot.is_connected:
                # disconnect() turns torque off, so the arm goes limp: give the operator time to support it.
                for s in range(3, 0, -1):
                    _status(f"Exiting: support/rest the arm, torque off in {s}s")
                    time.sleep(1.0)
                self.robot.disconnect()
            if listener is not None:
                listener.stop()
            _say(f"done | {self.stats.summary()}")


@parser.wrap()
def main(cfg: KinestheticRecordConfig) -> None:
    init_logging()
    logging.getLogger().setLevel(logging.WARNING)  # keep the live status line readable
    KinestheticRecorder(cfg).run()


if __name__ == "__main__":
    main()
