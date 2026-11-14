"""Run a LeRobot ACT checkpoint on the SO-101 follower with plain PyTorch (no OpenVINO / Studio).

Observations are built exactly as in training: follower joint positions in the repo calibration's degree units,
RGB frames from the recorder's cameras (`wrist` UVC + `top`/`inspect` RealSense), the `top` frame cropped to the
training crop (default x=0,y=0,w=560,h=480), then LeRobot's own `predict_action` (saved pre/postprocessors,
`policy.select_action` with its action queue). Each 30 Hz tick the follower is moved toward the predicted target by
at most `--max-speed` deg/s per joint (Desmond's `step_toward`).

Safety:
* `--dry-run` reads the arm and cameras and prints predictions; it never sends a command.
* Without `--yes` the script waits for Enter before moving.
* `--max-seconds` bounds the run.
* Ctrl+C, or the time limit, ramps the arm back to its saved home pose and holds it there; torque stays on until
  the script exits.

Stop Desmond's recorder first (it owns the arm and cameras), and keep a hand near the arm's power switch.

    ~/desmond/venvs/teleop/bin/python -m dlsj.deploy.act_rollout \
        --policy_path ~/inference/models/act-warp-duplo-417-20261114 \
        --cameras wrist=/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0 \
        --realsense top=346422061366,inspect=243722062690 --device xpu --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time

import cv2
import numpy as np

from dlsj import arms, teleop_web

log = logging.getLogger("act_rollout")


def parse_crop(spec: str) -> tuple[int, int, int, int] | None:
    """'x,y,w,h' -> (x, y, w, h); empty or 'none' -> None."""
    if not spec or spec.lower() == "none":
        return None
    x, y, w, h = (int(v) for v in spec.split(","))
    if w <= 0 or h <= 0 or x < 0 or y < 0:
        raise ValueError(f"bad crop {spec!r}")
    return x, y, w, h


def to_policy_image(bgr: np.ndarray, crop: tuple[int, int, int, int] | None) -> np.ndarray:
    """Camera BGR frame -> contiguous HxWx3 uint8 RGB, cropped like the training data."""
    if crop is not None:
        x, y, w, h = crop
        if y + h > bgr.shape[0] or x + w > bgr.shape[1]:
            raise ValueError(f"crop {crop} outside frame {bgr.shape[:2]}")
        bgr = bgr[y:y + h, x:x + w]
    return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def check_shapes(obs: dict[str, np.ndarray], expected: dict[str, tuple[int, ...]]) -> None:
    """Fail fast if a camera frame doesn't match the policy's input feature shape (C,H,W)."""
    for key, shape in expected.items():
        if not key.startswith("observation.images."):
            continue
        if key not in obs:
            raise KeyError(f"policy needs {key}; give that camera with --cameras/--realsense")
        h, w = obs[key].shape[:2]
        if (3, h, w) != tuple(shape):
            raise ValueError(f"{key}: frame {(3, h, w)} != policy input {tuple(shape)} (check --top-crop)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy_path", required=True, help="LeRobot ACT checkpoint dir (model.safetensors + processors)")
    p.add_argument("--device", default="xpu", help="torch device: xpu, cpu")
    p.add_argument("--cameras", default="", help="UVC cameras name=index_or_path,... (e.g. wrist=/dev/v4l/by-id/...)")
    p.add_argument("--realsense", default="", help="RealSense cameras name=serial,... (e.g. top=...,inspect=...)")
    p.add_argument("--top-crop", default="0,0,560,480", help="x,y,w,h crop for the top camera ('none' to disable)")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--cam-fps", type=int, default=30)
    p.add_argument("--fps", type=int, default=30, help="control rate (training data: 30)")
    p.add_argument("--n-action-steps", type=int, help="override how many actions of each chunk are executed")
    p.add_argument("--max-speed", type=float, default=60.0, help="per-joint speed cap, deg/s (gripper units/s)")
    p.add_argument("--ramp-speed", type=float, default=30.0, help="deg/s when ramping to/from home")
    p.add_argument("--max-seconds", type=float, default=60.0)
    p.add_argument("--follower-port", default="auto")
    p.add_argument("--follower-id", default="follower")
    p.add_argument("--dry-run", action="store_true", help="never send commands; print predicted targets")
    p.add_argument("--yes", action="store_true", help="don't wait for Enter before moving")
    p.add_argument("--save-dir", help="save one frame per camera per second (plus state/target) here, for review")
    p.add_argument("--start-home", action="store_true", help="ramp to the saved home pose before the policy starts")
    p.add_argument("--release-at-end", action="store_true",
                   help="release torque when the script exits (default: keep holding the final pose)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import torch
    from lerobot.common.control_utils import predict_action
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.robots import make_robot_from_config
    from lerobot.robots.so_follower import SO101FollowerConfig

    from dlsj.realsense_stream import RealSenseStream

    top_crop = parse_crop(args.top_crop)
    device = torch.device(args.device)

    cfg = ACTConfig.from_pretrained(args.policy_path)
    cfg.device = args.device
    if args.n_action_steps:
        cfg.n_action_steps = args.n_action_steps
    policy = ACTPolicy.from_pretrained(args.policy_path, config=cfg).to(device).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=args.policy_path,
                                         preprocessor_overrides={"device_processor": {"device": args.device}})
    expected = {k: tuple(v.shape) for k, v in policy.config.input_features.items()}
    log.info("ACT loaded on %s: chunk %d, executing %d per chunk; inputs %s", device, cfg.chunk_size,
             cfg.n_action_steps, expected)

    uvc = teleop_web.parse_cameras(args.cameras, 8) if args.cameras else {}
    rs = dict(part.split("=", 1) for part in filter(None, (x.strip() for x in args.realsense.split(","))))
    streams = {n: teleop_web.CameraStream(n, i, args.width, args.height, args.cam_fps, 80).start()
               for n, i in uvc.items()}
    streams |= {n: RealSenseStream(n, sn, args.width, args.height, args.cam_fps, depth=False).start()
                for n, sn in rs.items()}

    port = args.follower_port
    if port == "auto":
        port, _, volts = arms.find_ports()
        log.info("follower by supply voltage: %s (%s)", port, volts)
        if not port:
            p.error("follower not found; check USB and the 12 V supply")
    # Never drop the arm: torque stays on (holding the last pose) after disconnect unless --release-at-end.
    robot = make_robot_from_config(SO101FollowerConfig(port=port, id=args.follower_id,
                                                       calibration_dir=arms.FOLLOWER_CAL_DIR,
                                                       disable_torque_on_disconnect=args.release_at_end
                                                       and not args.dry_run))
    robot.connect(calibrate=False)
    joints = list(robot.action_features)  # recorder order: shoulder_pan.pos ... gripper.pos
    home = arms.load_home(args.follower_id)

    def read_state() -> np.ndarray:
        o = robot.get_observation()
        return np.array([o[j] for j in joints], dtype=np.float32)

    def send(cmd: np.ndarray) -> None:
        if not args.dry_run:
            robot.send_action({j: float(v) for j, v in zip(joints, cmd)})

    def ramp_to(target: np.ndarray, cmd: np.ndarray) -> np.ndarray:
        step = args.ramp_speed / args.fps
        while np.max(np.abs(target - cmd)) > 0.5:
            cmd = teleop_web.step_toward(cmd, target, step)
            send(cmd)
            time.sleep(1.0 / args.fps)
        return cmd

    stop = False

    def on_sigint(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, on_sigint)

    try:
        t0 = time.time()
        while any(s.frame is None for s in streams.values()):  # wait for every camera's first frame
            if time.time() - t0 > 10:
                missing = [n for n, s in streams.items() if s.frame is None]
                raise RuntimeError(f"no frames from {missing}")
            time.sleep(0.05)

        cmd = read_state()
        if args.start_home and not args.dry_run:
            if home is None:
                raise RuntimeError("--start-home: no saved home pose for this follower (calibration/)")
            log.info("ramping to home at %.0f deg/s before starting", args.ramp_speed)
            cmd = ramp_to(np.array([home[j] for j in joints], dtype=np.float32), cmd)
            time.sleep(0.5)
        if not args.dry_run and not args.yes:
            input("Arm and cameras ready. Press Enter to start the policy (Ctrl+C to stop and go home)... ")
        policy.reset()
        step_cap = args.max_speed / args.fps
        lat: list[float] = []
        tick, t_start = 0, time.time()
        while not stop and time.time() - t_start < args.max_seconds:
            t_tick = time.perf_counter()
            state = read_state()
            obs = {"observation.state": state}
            for name, s in streams.items():
                obs[f"observation.images.{name}"] = to_policy_image(s.frame, top_crop if name == "top" else None)
            if tick == 0:
                check_shapes(obs, expected)
            t_inf = time.perf_counter()
            action = predict_action(obs, policy, device, pre, post, use_amp=False)
            lat.append((time.perf_counter() - t_inf) * 1e3)
            target = action.squeeze(0).detach().cpu().numpy().astype(np.float32)
            cmd = teleop_web.step_toward(cmd if not args.dry_run else state, target, step_cap)
            send(cmd)
            if tick % args.fps == 0 and args.save_dir:
                from pathlib import Path
                d = Path(args.save_dir); d.mkdir(parents=True, exist_ok=True)
                sec = tick // args.fps
                for name, s in streams.items():
                    cv2.imwrite(str(d / f"t{sec:03d}_{name}.jpg"), s.frame)
                with open(d / "trace.jsonl", "a") as fh:
                    fh.write(json.dumps({"t": sec, "state": state.round(2).tolist(), "target": target.round(2).tolist(),
                                         "cmd": cmd.round(2).tolist()}) + "\n")
            if tick % args.fps == 0:
                log.info("t=%.1fs state=%s target=%s infer %.1f ms (max so far %.0f)", time.time() - t_start,
                         np.round(state, 1), np.round(target, 1), lat[-1], max(lat))
            tick += 1
            time.sleep(max(0.0, 1.0 / args.fps - (time.perf_counter() - t_tick)))
        lat_a = np.array(lat)
        log.info("ran %d ticks; per-tick inference median %.1f ms, p90 %.1f ms, max %.1f ms (chunk refills spike)",
                 tick, np.median(lat_a), np.percentile(lat_a, 90), lat_a.max())
        if home is not None and not args.dry_run:
            log.info("ramping home")
            ramp_to(np.array([home[j] for j in joints], dtype=np.float32), read_state())
    finally:
        for s in streams.values():
            s.stop()
        if args.release_at_end and not args.dry_run and sys.stdin.isatty():
            input("Press Enter to release torque and disconnect (support the arm first)... ")
        robot.disconnect()
        log.info("disconnected; torque %s", "released" if args.release_at_end and not args.dry_run else "still on (holding)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
