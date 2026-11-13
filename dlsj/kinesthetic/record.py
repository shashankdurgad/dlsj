"""Kinesthetic-teaching recorder for the SO-101 follower: move the real arm by hand, record a LeRobot dataset.

The arm joints have torque off and you guide the follower itself; there is no leader arm. The gripper
stays powered and is driven by keys or a foot pedal, so both hands stay on the arm. You can pass
`--gripper_mode=free` to move the gripper by hand too.

The workflow follows Flash ALOHA's hand-guided collector (MIT, see THIRD_PARTY_NOTICES.md):

1. **Take.** Press go to release the arm and record. Press go again to stop.
2. **Review.** The arm is held at its last pose. You can then:
   * accept (go / n)
   * save as a failure (f)
   * reject (h)
   * replay the unsaved take on the robot (r). The arm moves to the first pose, waits for you to
     reset the scene (go to play, h to cancel), plays every action once at the recording rate, and
     prints the tracking error. A large arm-joint error means the demo was faster than the servos can
     reproduce.
3. **Between takes** the arm stays held. Press h to release or hold it, p to save the current pose as
   the start pose. With `--go_to_start` the arm returns to that pose after every take.

The output is a standard LeRobot v3 dataset with the same features as `lerobot-record` on an
`so101_follower`: `observation.state`, `observation.images.*` and `action`. Kinesthetic and teleop
datasets can therefore be trained on together. Extra data goes in `meta/dlsj/`, outside the dataset
features:

* `labels.jsonl`: one line per saved episode, with success, duration, operator, method and replay
  tracking error.
* `telemetry/episode_XXXXXX.npz`: raw `Present_Current` / `Present_Load` per frame. This is logged but
  is not a policy input.
* `start_pose.json`

The action is `action[t] = measured_position[t + action_lookahead]`, except for a key-driven gripper,
where it is the commanded target at `t`.

Example:

    uv run python -m dlsj.kinesthetic.record \
        --robot.type=so101_follower --robot.port=/dev/tty.usbmodemXXXX --robot.id=follower_a \
        --robot.cameras="{wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, \
                          top: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
        --dataset.repo_id=local/duplo_kin --dataset.single_task="Sort the Duplo bricks by colour" \
        --dataset.fps=30 --dataset.episode_time_s=30 --dataset.push_to_hub=false \
        --dataset.streaming_encoding=true --operator=alice --target_s=8

Keys (two-pedal foot switch: left pedal = 1, right pedal = h):
  1 / s / Enter    go: start a take / stop it / accept it / start a replay
  h / x / Bksp     reject: discard the take, or cancel a replay. Between takes: hold or release the arm
  n / Right        save as SUCCESS now        f: save as FAILURE        r: (in review) replay the take
  space / g        toggle the gripper         up / down: nudge the gripper
  p                save the current pose as the start pose
  q / Esc          quit (an unsaved take is discarded)
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

from .session import (
    ActionLookahead,
    Command,
    EpisodeStats,
    GripperCommand,
    KeyMap,
    Phase,
    SessionMachine,
    approach_waypoints,
    tracking_report,
)

GRIPPER = "gripper"
_DEFAULT_FEATURES = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


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
    # Tempo target per take in seconds: shown live to push operators toward fast demos.
    target_s: float | None = None
    # Return the held arm to the saved start pose (key p) after every take.
    go_to_start: bool = False
    # Max joint speed (calibrated units/s, degrees for the arm) for moves to a start pose or replay pose.
    approach_speed: float = 45.0
    # Show the take's first frame blended over the live camera before a replay (needs a display).
    overlay: bool = True
    # Camera for the overlay; default: "top" if present, else the first camera.
    overlay_camera: str | None = None
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
        self.joint_names = list(self.robot.action_features)  # "shoulder_pan.pos", ..., "gripper.pos"
        self.motor_names = [n.removesuffix(".pos") for n in self.joint_names]
        self.free_motors = [m for m in self.bus.motors if m != GRIPPER or cfg.gripper_mode == "free"]
        self.period = 1.0 / cfg.dataset.fps
        self.keymap = KeyMap()
        self.machine = SessionMachine()
        self.lookahead = ActionLookahead(cfg.action_lookahead, self.joint_names)
        self.gripper = GripperCommand(cfg.gripper_open, cfg.gripper_closed, cfg.gripper_step)
        self.stats = EpisodeStats()
        self.keys: queue.Queue[str] = queue.Queue()
        self.arm_held = True  # LeRobot's configure() leaves torque on after connect
        self.dataset: LeRobotDataset | None = None
        self.start_pose: dict[str, float] | None = None
        cams = list(getattr(self.robot, "cameras", {}) or {})
        self.overlay_camera = cfg.overlay_camera or ("top" if "top" in cams else (cams[0] if cams else None))
        self._reset_take()

    def _reset_take(self) -> None:
        self.lookahead.reset()
        self._telemetry: list[tuple[np.ndarray, np.ndarray]] = []
        self._take_actions: list[np.ndarray] = []
        self._first_image: np.ndarray | None = None
        self._take_start = 0.0
        self._take_duration = 0.0
        self._take_closed = False
        self._frames_written = 0
        self._ticks = 0
        self._last_tracking: dict | None = None

    # ---------------------------------------------------------------- setup
    def _features(self) -> dict:
        video = self.cfg.dataset.video
        return {
            **hw_to_dataset_features(self.robot.action_features, ACTION, video),
            **hw_to_dataset_features(self.robot.observation_features, OBS_STR, video),
        }

    def _open_dataset(self) -> LeRobotDataset:
        d = self.cfg.dataset
        n_cams = len(getattr(self.robot, "cameras", {}) or {})
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
            if set(ds.features) - _DEFAULT_FEATURES != set(self._features()):
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

    # ---------------------------------------------------------------- motion
    def _present(self) -> dict[str, float]:
        return self.bus.sync_read("Present_Position")

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

    def _move_to(self, goal: dict[str, float]) -> None:
        """Speed-limited move of the held arm to `goal` (motor name -> calibrated position)."""
        if not self.arm_held:
            self._hold_arm()
        motors = [m for m in self.motor_names if m in goal]
        present = self._present()
        start = np.array([present[m] for m in motors])
        target = np.array([goal[m] for m in motors])
        for wp in approach_waypoints(start, target, self.cfg.approach_speed, self.cfg.dataset.fps):
            t0 = time.perf_counter()
            self.bus.sync_write("Goal_Position", dict(zip(motors, map(float, wp), strict=True)))
            precise_sleep(max(0.0, self.period - (time.perf_counter() - t0)))
        if GRIPPER in goal and self.cfg.gripper_mode == "keys":
            self.gripper.target = float(goal[GRIPPER])

    # ---------------------------------------------------------------- takes
    def _start_take(self) -> None:
        self._reset_take()
        if self.arm_held:
            self._release_arm()
        self._take_start = time.perf_counter()
        _say(f"● REC take for episode {self.dataset.num_episodes}  (1 stop, h reject, n/f save)")

    def _write(self, frames: list[dict]) -> None:
        for f in frames:
            action = np.asarray(f.pop("action"), dtype=np.float32)
            f[ACTION] = action
            self.dataset.add_frame(f)
            self._take_actions.append(action)
            self._frames_written += 1

    def _record_tick(self) -> None:
        obs = self.robot.get_observation()
        state = [float(obs[n]) for n in self.joint_names]
        if self._first_image is None and self.overlay_camera in obs:
            self._first_image = np.array(obs[self.overlay_camera], copy=True)
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

    def _close_take(self) -> None:
        """Stop capturing: complete the lookahead frames and hold the arm where it is. Idempotent."""
        if self._take_closed:
            return
        self._take_duration = time.perf_counter() - self._take_start
        self._write(self.lookahead.flush())
        self._take_closed = True
        self._hold_arm()

    def _enter_review(self, reason: str) -> None:
        self._close_take()
        _say(f"■ {reason}: {self._take_duration:.1f}s, {self._frames_written} frames. Review: 1/n accept, "
             "f failure, r replay on robot, h reject")

    def _finish_take(self, cmd: Command) -> None:
        if cmd is Command.DISCARD:
            self._close_take()
            self.dataset.clear_episode_buffer()
            _say(f"✗ rejected ({self._take_duration:.1f}s)")
        else:
            self._close_take()
            if self._frames_written == 0:
                self.dataset.clear_episode_buffer()
                _say("✗ nothing recorded, discarded")
            else:
                self._save_take(success=cmd is Command.SAVE_SUCCESS)
        self._reset_take()
        if self.cfg.go_to_start and self.start_pose:
            self._move_to(self.start_pose)

    def _save_take(self, success: bool) -> None:
        ep = self.dataset.num_episodes
        self.dataset.save_episode()
        label = {
            "episode_index": ep,
            "success": success,
            "duration_s": round(self._take_duration, 3),
            "n_frames": self._frames_written,
            "achieved_fps": round(self._ticks / self._take_duration, 2) if self._take_duration > 0 else None,
            "method": "kinesthetic",
            "gripper_mode": self.cfg.gripper_mode,
            "action_lookahead": self.cfg.action_lookahead,
            "operator": self.cfg.operator,
            "task": self.cfg.dataset.single_task,
            "replay_tracking": self._last_tracking,
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
            self.stats.add(self._take_duration)
        _say(f"✓ saved episode {ep} as {'SUCCESS' if success else 'FAILURE'} ({self._take_duration:.1f}s, "
             f"{label['achieved_fps']} fps) | {self.stats.summary()}")

    # ---------------------------------------------------------------- review replay
    def _wait_go_or_reject(self, prompt: str) -> bool:
        """Block until go (True) or reject/quit (False). Shows the alignment overlay if enabled."""
        _say(prompt)
        show = self._overlay_enabled()
        try:
            while True:
                if show:
                    self._draw_overlay()
                try:
                    key = self.keys.get(timeout=0.03)
                except queue.Empty:
                    continue
                k = key.lower()
                if k in self.keymap.go:
                    return True
                if k in self.keymap.reject or k in self.keymap.quit:
                    return False
        finally:
            if show:
                self._close_overlay()

    def _overlay_enabled(self) -> bool:
        if not (self.cfg.overlay and self._first_image is not None and self.overlay_camera):
            return False
        try:
            import cv2  # noqa: F401
        except ImportError:
            return False
        return True

    def _draw_overlay(self) -> None:
        import cv2

        live = self.robot.cameras[self.overlay_camera].read_latest()
        if live is None or live.shape != self._first_image.shape:
            return
        blend = cv2.addWeighted(self._first_image, 0.5, live, 0.5, 0)
        cv2.imshow("take start vs live (1 = replay, h = cancel)", cv2.cvtColor(blend, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    def _close_overlay(self) -> None:
        import cv2

        cv2.destroyAllWindows()
        cv2.waitKey(1)

    def _replay_take(self) -> None:
        if not self._take_actions:
            _say("nothing to replay")
            return
        actions = np.stack(self._take_actions)
        first = dict(zip(self.motor_names, map(float, actions[0]), strict=True))
        self._move_to(first)
        if not self._wait_go_or_reject("At the take's first pose. Reset the scene, then 1 = replay, h = cancel"):
            _say("replay cancelled. Review: 1/n accept, f failure, r replay, h reject")
            return
        measured, sent = [], []
        stopped = False
        t_start = time.perf_counter()
        for a in actions:
            t0 = time.perf_counter()
            self.bus.sync_write("Goal_Position", dict(zip(self.motor_names, map(float, a), strict=True)))
            sent.append(a)
            pos = self._present()
            measured.append([pos[m] for m in self.motor_names])
            try:
                k = self.keys.get_nowait().lower()
                if k in self.keymap.go or k in self.keymap.reject or k in self.keymap.quit:
                    stopped = True
                    break
            except queue.Empty:
                pass
            precise_sleep(max(0.0, self.period - (time.perf_counter() - t0)))
        elapsed = time.perf_counter() - t_start
        self._hold_arm()
        while not self.keys.empty():  # keys pressed during the replay must not answer the review prompt
            self.keys.get_nowait()
        rep = tracking_report(np.array(sent), np.array(measured), self.joint_names, delay=1)
        arm = [i for i, m in enumerate(self.motor_names) if m != GRIPPER]
        worst_arm = float(np.max(rep.max_abs[arm])) if arm else 0.0
        self._last_tracking = {
            "max_abs": dict(zip(self.motor_names, map(float, rep.max_abs), strict=True)),
            "median_abs": dict(zip(self.motor_names, map(float, rep.median_abs), strict=True)),
            "replay_s": round(elapsed, 3),
            "stopped_early": stopped,
        }
        _say(f"{'⏹ replay stopped' if stopped else '▶ replay done'} in {elapsed:.1f}s "
             f"(take {self._take_duration:.1f}s) | {rep.format('deg')}")
        if worst_arm > 10.0:
            _say("⚠ arm tracking error > 10 deg: this take may be too fast for the servos to reproduce")
        _say("Review: 1/n accept, f failure, r replay again, h reject")

    # ---------------------------------------------------------------- keys
    def _apply(self, cmd: Command) -> bool:
        """Execute an accepted command. Returns False when the session should end."""
        if cmd is Command.QUIT:
            return False
        if cmd is Command.START:
            self._start_take()
        elif cmd is Command.STOP:
            self._enter_review("stopped")
        elif cmd in (Command.SAVE_SUCCESS, Command.SAVE_FAILURE, Command.DISCARD):
            self._finish_take(cmd)
        elif cmd is Command.REPLAY:
            self._replay_take()
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
        elif cmd is Command.SAVE_START_POSE:
            self.start_pose = {m: float(v) for m, v in self._present().items()}
            (self.sidecar / "start_pose.json").write_text(json.dumps(self.start_pose, indent=2))
            _say("start pose saved" + ("" if self.cfg.go_to_start else " (use --go_to_start=true to return to it)"))
        return True

    def _drain_keys(self) -> bool:
        while True:
            try:
                key = self.keys.get_nowait()
            except queue.Empty:
                return True
            raw = self.keymap.lookup(key, self.machine.phase)
            if raw is None:
                continue
            if raw is Command.QUIT and self.machine.phase is not Phase.IDLE:
                # Never lose a take silently: quitting mid-take discards it explicitly.
                self.machine.handle(Command.DISCARD)
                self._finish_take(Command.DISCARD)
                return False
            cmd = self.machine.handle(raw)
            if cmd is not None and not self._apply(cmd):
                return False

    # ---------------------------------------------------------------- main loop
    def _load_start_pose(self) -> None:
        p = self.sidecar / "start_pose.json"
        if p.exists():
            self.start_pose = json.loads(p.read_text())

    def run(self) -> None:
        cfg = self.cfg
        listener = None
        self.robot.connect()  # fail on a wrong port before creating an empty dataset
        try:
            self.dataset = self._open_dataset()
            self._hold_arm()
            self._command_gripper()
            self._load_start_pose()
            if cfg.go_to_start and self.start_pose:
                self._move_to(self.start_pose)
            listener = create_key_listener(self.keys.put, controls_help="1 go, h reject, q quit")
            print("Keys" + __doc__.split("Keys", 1)[1])
            _say(f"dataset: {self.dataset.root}")
            _say("Arm is HELD. Hold it, then press 1 to start a take (torque is released when recording starts).")
            with VideoEncodingManager(self.dataset):
                last_status = 0.0
                while True:
                    t0 = time.perf_counter()
                    if not self._drain_keys():
                        break
                    phase = self.machine.phase
                    if phase is Phase.RECORDING:
                        self._record_tick()
                        elapsed = t0 - self._take_start
                        if elapsed >= cfg.dataset.episode_time_s and self.machine.time_limit_reached():
                            self._enter_review(f"time limit {cfg.dataset.episode_time_s}s")
                        elif t0 - last_status > 0.2:
                            tgt = ""
                            if cfg.target_s:
                                tgt = f" / target {cfg.target_s:.1f}s" + (" ⚠ OVER" if elapsed > cfg.target_s else "")
                            grip = "open" if self.gripper.is_open else "closed"
                            _status(f"● {elapsed:5.1f}s{tgt} | gripper {grip} ({self.gripper.target:.0f})")
                            last_status = t0
                    elif phase is Phase.IDLE and t0 - last_status > 0.25:
                        g = self.bus.read("Present_Position", GRIPPER)
                        _status(f"idle | arm {'HELD' if self.arm_held else 'free'} | gripper measured {g:.1f} "
                                f"target {self.gripper.target:.0f} | {self.stats.summary()} | 1 start, h hold, q quit")
                        last_status = t0
                    precise_sleep(max(0.0, self.period - (time.perf_counter() - t0)))
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
