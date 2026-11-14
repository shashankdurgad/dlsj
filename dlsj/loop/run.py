"""Run the closed loop: pick -> inspect -> decide -> place, brick after brick.

Laptop dry run (fake robot, cameras, inspector and scripted policy; status page on :8001):

    uv run python -m dlsj.loop.run --dry-run --bricks plain,cross,circle,blob,faint,hidden_cross --enable-reshow

Spin-and-inspect (DETECT rotates the brick on wrist_roll and decides from every angle; `--sweep-frames 0` keeps the
classic 3-frame inspection). Dry run with one-face marks and a glint that defeat the naive strategies:

    uv run python -m dlsj.loop.run --dry-run --bricks side_cross,side_circle,glint,plain \
        --sweep-frames 12 --sweep-range 180 --sweep-settle-ms 150 --aggregate topk

With the team's PatchCore (Anomalib Studio export, no manifest; GPU is idle during the sweep, NPU also works):

    uv run python -m dlsj.loop.run --cameras wrist=0,top=2,inspect=4 --policy ... \
        --inspector anomalib --inspector-path ~/inference/models/downloads/brick-patchcore-geometry-v1 \
        --inspect-crop 160,80,480,400 --score-space raw --inspector-device GPU --stage2 shape \
        --sweep-frames 12 --aggregate topk

Plates (three-class task; colours are configurable, the minor plate's yellow is a placeholder):

    --plates good=yellow,minor=orange,defective=purple   # instruction "place the brick in the <colour> bowl"
    --plates plates.json                                  # {"minor": {"colour": "green", "instruction": "...",
                                                          #   "pose": "green"}, ...}

On the XPS (once the pi0.5 export and the Anomalib inspector exist):

    uv run python -m dlsj.loop.run --cameras wrist=0,top=2,inspect=4 \
        --policy openvino --policy-path exports/pi05_ov --inspector anomalib --inspector-path exports/anomalib \
        [--stage2 shape | --stage2 exports/inspect/mark_cnn/classifier.json --stage2-device NPU]

Hardware smoke test without the VLA (scripted motion between poses saved with --save-pose):

    uv run python -m dlsj.loop.run --save-pose inspect      # pose the arm by hand, press Enter
    uv run python -m dlsj.loop.run --cameras ... --policy fake --inspector anomalib ... --step

Logs: outputs/loop/<timestamp>/{timings.jsonl, events.jsonl, summary.json}. Ctrl+C or STOP on the page
aborts at the next control tick and ramps home (like teleop_web).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from dlsj.loop import poses as P
from dlsj.loop.controller import Controller, ControllerConfig
from dlsj.loop.executor import ExecConfig, SkillExecutor
from dlsj.loop.planner import AnyBrickPlanner
from dlsj.loop.reasoner import ReasonConfig
from dlsj.loop.skills import RESHOW, Plates, SkillLibrary
from dlsj.inspect.aggregate import AggregateConfig
from dlsj.loop.sweep import SweepConfig
from dlsj.loop.telemetry import RunLog

log = logging.getLogger("loop")

SCRIPTED_POSES = ["home", "grasp", "inspect"]  # + one pose per plate (Plate.pose, default the colour)


class BrickGate:
    """--step: wait for Enter in the terminal or NEXT BRICK on the status page before each brick."""

    def __init__(self):
        self.cond = threading.Condition()
        self.answer: bool | None = None
        threading.Thread(target=self._stdin, daemon=True).start()

    def _stdin(self) -> None:
        for line in sys.stdin:
            self.set(line.strip().lower() not in ("q", "quit"))

    def set(self, go: bool) -> None:
        with self.cond:
            self.answer = go
            self.cond.notify_all()

    def __call__(self, task) -> bool:
        print(f"\nbrick #{task.id}: put a brick on the table, then Enter (q = quit) or NEXT BRICK on the page")
        with self.cond:
            self.answer = None
            self.cond.wait_for(lambda: self.answer is not None)
            return bool(self.answer)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="fake robot, cameras, inspector and policy")
    p.add_argument("--robot", choices=["so101", "fake"], default="so101")
    p.add_argument("--follower-port", default="auto")
    p.add_argument("--follower-id", default="follower")
    p.add_argument("--max-relative-target", type=float, default=None, help="LeRobot per-step cap (deg)")
    p.add_argument("--cameras", default="fake", help='"fake" or name=index,... e.g. wrist=0,top=2,inspect=4')
    p.add_argument("--policy-cameras", default=None, help="cameras fed to the policy (default: all)")
    p.add_argument("--inspector", choices=["fake", "anomalib"], default="fake")
    p.add_argument("--inspector-path", default=None, help="manifest.json, Anomalib export folder or model.xml")
    p.add_argument("--inspector-device", default="NPU",
                   help="OpenVINO device of the stage-1 model (GPU is free during the sweep: pi0.5 is idle)")
    p.add_argument("--inspector-batch", type=int, default=None, help="static batch (default 3; 1 when sweeping)")
    p.add_argument("--inspect-crop", default=None, help="x0,y0,x1,y1 ROI for an Anomalib export without a manifest")
    p.add_argument("--score-space", choices=["raw", "normalized"], default=None,
                   help="Anomalib exports: raw (cut before the normaliser) or normalized scores")
    p.add_argument("--sweep-frames", type=int, default=0, help="spin-and-inspect stops (0 = classic 3 frames)")
    p.add_argument("--sweep-range", type=float, default=180.0, help="total wrist_roll range (deg) around the start")
    p.add_argument("--sweep-settle-ms", type=float, default=150.0, help="minimum dwell at each stop")
    p.add_argument("--sweep-speed", type=float, default=90.0, help="wrist_roll speed cap during the sweep (deg/s)")
    p.add_argument("--sweep-discard", type=int, default=1, help="new frames dropped before the one kept per stop")
    p.add_argument("--sweep-continuous", action="store_true", help="roll without stopping (motion blur risk)")
    p.add_argument("--sweep-sync", action="store_true", help="infer after the sweep instead of during it")
    p.add_argument("--aggregate", default="topk", choices=["topk", "majority", "max", "median"],
                   help="sweep aggregation strategy (dlsj.inspect.aggregate)")
    p.add_argument("--topk", type=int, default=3, help="topk: frames averaged")
    p.add_argument("--stage2", default=None,
                   help='mark classifier: "shape" | "none" | CNN classifier.json (default: manifest, else shape)')
    p.add_argument("--stage2-device", default=None, help="OpenVINO device of the stage-2 CNN (default NPU)")
    p.add_argument("--plates", default=None,
                   help='bowl colours/strings: "good=yellow,minor=orange,defective=purple" (default) or a JSON file')
    p.add_argument("--policy", choices=["fake", "openvino", "lerobot-server"], default="fake")
    p.add_argument("--policy-path", default=None, help="OpenVINO export dir")
    p.add_argument("--policy-device", default="GPU")
    p.add_argument("--policy-server", default="127.0.0.1:8080")
    p.add_argument("--fake-latency-ms", type=float, default=170.0, help="simulated chunk latency of --policy fake")
    p.add_argument("--bricks", default="plain,cross,circle,faint,hidden_cross",
                   help="dry-run bricks on the fake table (plain, cross, circle, blob, faint, hidden_cross, glare, "
                        "side_cross, side_circle, glint)")
    p.add_argument("--max-bricks", type=int, default=None, help="stop after this many bricks (dry run: #bricks)")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--max-speed", type=float, default=90.0, help="arm joint cap, deg/s (step_toward per tick)")
    p.add_argument("--ramp-speed", type=float, default=30.0, help="homing speed, deg/s")
    p.add_argument("--sync-inference", action="store_true", help="baseline: no background inference thread")
    p.add_argument("--request-threshold", type=float, default=0.5)
    p.add_argument("--skills-config", default=None, help="JSON overrides for skill strings / termination")
    p.add_argument("--enable-reshow", action="store_true", help="use the RESHOW skill on uncertain inspections")
    p.add_argument("--max-uncertain", type=int, default=2)
    p.add_argument("--extra-margin", type=float, default=0.0, help="stage 1: widen the uncertainty band")
    p.add_argument("--min-mark-confidence", type=float, default=0.0,
                   help="stage 2: CIRCLE/CROSS below this confidence count as uncertain")
    p.add_argument("--grasp-current-min", type=float, default=None)
    p.add_argument("--step", action="store_true", help="wait for Enter / NEXT BRICK before every brick")
    p.add_argument("--no-home", action="store_true", help="don't ramp to the home pose at start / end")
    p.add_argument("--log-dir", default=None)
    p.add_argument("--status-port", type=int, default=8001, help="0 = no status page")
    p.add_argument("--save-pose", default=None, metavar="NAME",
                   help="release the follower, pose it by hand, Enter saves calibration/.../<id>.<NAME>.json")
    args = p.parse_args(argv)
    if args.dry_run:
        args.robot, args.cameras, args.inspector, args.policy = "fake", "fake", "fake", "fake"
    return args


def save_pose_interactive(args) -> None:
    from dlsj.loop.hardware import LeRobotArm

    arm = LeRobotArm(args.follower_port, args.follower_id, calibrate=False)
    arm.connect()
    try:
        arm.release_for_posing()
        input(f"Torque off. Pose the follower at the '{args.save_pose}' pose and press Enter...")
        path = P.save_pose(args.follower_id, args.save_pose, arm.read_state())
        print("saved", path, json.loads(path.read_text()))
    finally:
        arm.disconnect()


def build(args):
    """Wire real or fake components. Returns (controller, executor, cameras, cleanup callables)."""
    from dlsj.loop import fakes

    cleanup = []
    library = SkillLibrary()
    plates = Plates.parse(args.plates)
    plates.apply(library)
    if args.skills_config:
        library.load_overrides(args.skills_config)  # explicit per-skill overrides win over --plates
    if args.enable_reshow:
        library.enabled_optional.add(RESHOW)

    world = robot = None
    if args.robot == "fake":
        role_pose = {"good": "yellow", "minor": "orange", "defective": "purple"}  # fake plate positions per role
        wposes = {k: fakes.DEFAULT_POSES[k].copy() for k in SCRIPTED_POSES}
        for pl in plates.all():  # only the configured plates exist in the fake world
            wposes[pl.pose] = fakes.DEFAULT_POSES[role_pose[pl.role]].copy()
        world = fakes.FakeWorld(bricks=[b.strip() for b in args.bricks.split(",") if b.strip()], poses=wposes)
        robot = fakes.FakeRobot(world)
        poses = dict(world.poses)
        home = poses["home"]
    else:
        from dlsj.loop.hardware import LeRobotArm

        robot = LeRobotArm(args.follower_port, args.follower_id, args.max_relative_target)
        robot.connect()
        cleanup.append(robot.disconnect)
        poses = P.load_poses(args.follower_id, SCRIPTED_POSES + [pl.pose for pl in plates.all()])
        home = poses.get("home")
        if home is None:
            log.warning("no home pose saved for %s (teleop_web 'save home'); not homing", args.follower_id)

    if args.cameras == "fake":
        if world is None:
            raise SystemExit("--cameras fake needs --robot fake (the fake cameras render the fake world)")
        cameras = fakes.FakeCameras(world, robot)
    else:
        from dlsj.loop.hardware import StreamCameras

        cameras = StreamCameras.open(args.cameras)
        cleanup.append(cameras.stop)
        cameras.wait_ready()

    agg = AggregateConfig(strategy=args.aggregate, k=args.topk)
    if args.inspector == "fake":
        if not isinstance(cameras, fakes.FakeCameras):
            raise SystemExit("--inspector fake only understands the fake cameras' rendering; use --inspector anomalib")
        inspector = fakes.FakeInspector(cameras.roi, agg=agg)
    else:
        from dlsj.inspect.core import parse_box
        from dlsj.loop.hardware import load_inspector

        batch = args.inspector_batch or (1 if args.sweep_frames > 0 else 3)
        inspector = load_inspector(args.inspector_path, device=args.inspector_device, stage2=args.stage2,
                                   stage2_device=args.stage2_device, batch=batch,
                                   crop=parse_box(args.inspect_crop), score_space=args.score_space, agg=agg)

    if args.policy == "fake":
        missing = [n for n in SCRIPTED_POSES + [pl.pose for pl in plates.all()] if n not in poses]
        if missing:
            raise SystemExit(f"--policy fake on the real arm needs saved poses {missing} (--save-pose NAME)")
        if args.robot != "fake":
            log.warning("SCRIPTED policy on the REAL arm: it moves between your saved poses. Keep a hand on the e-stop.")
        policy = fakes.ScriptedPolicy(poses, latency_s=args.fake_latency_ms / 1000.0, plates=plates)
    elif args.policy == "openvino":
        from dlsj.loop.hardware import OpenVINOPi05Policy

        policy = OpenVINOPi05Policy(args.policy_path, args.policy_device)
    else:
        from dlsj.loop.hardware import LeRobotServerPolicy

        policy = LeRobotServerPolicy(args.policy_server)

    run_dir = Path(args.log_dir or Path("outputs") / "loop" / datetime.now().strftime("%Y%m%d-%H%M%S"))
    runlog = RunLog(run_dir)
    ecfg = ExecConfig(fps=args.fps, max_speed_deg_s=args.max_speed, ramp_speed_deg_s=args.ramp_speed,
                      async_inference=not args.sync_inference, request_threshold=args.request_threshold,
                      camera_names=tuple(args.policy_cameras.split(",")) if args.policy_cameras else None)
    executor = SkillExecutor(robot, cameras, policy, library, ecfg, poses, runlog)
    cleanup.insert(0, executor.close)
    max_bricks = args.max_bricks if args.max_bricks is not None else (len(world.bricks) if world else None)
    planner = AnyBrickPlanner(max_tasks=max_bricks)
    sweep = None
    if args.sweep_frames > 0:
        limits = None
        if args.robot != "fake":
            from dlsj import arms

            limits = arms.joint_limits_deg(args.follower_id, "wrist_roll")
            log.info("wrist_roll limits from calibration: %s", limits)
        sweep = SweepConfig(frames=args.sweep_frames, range_deg=args.sweep_range, settle_ms=args.sweep_settle_ms,
                            max_speed_deg_s=args.sweep_speed, discard_frames=args.sweep_discard, limits=limits,
                            grasp_current_min=args.grasp_current_min, continuous=args.sweep_continuous, fps=args.fps)
    ccfg = ControllerConfig(go_home=not args.no_home, grasp_current_min=args.grasp_current_min,
                            sweep=sweep, aggregate=args.aggregate, sweep_async=not args.sweep_sync,
                            reason=ReasonConfig(max_uncertain=args.max_uncertain,
                                                reshow_available=library.available(RESHOW),
                                                extra_margin=args.extra_margin,
                                                min_mark_confidence=args.min_mark_confidence))
    gate = BrickGate() if args.step else None
    controller = Controller(executor, cameras, inspector, planner, runlog, ccfg, home, wait_for_brick=gate)
    runlog.event("config", args=vars(args), run_dir=str(run_dir), poses={k: v.tolist() for k, v in poses.items()},
                 plates=plates.describe())
    return controller, executor, cameras, runlog, world, gate, cleanup


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.save_pose:
        save_pose_interactive(args)
        return 0

    controller, executor, cameras, runlog, world, gate, cleanup = build(args)
    status = None
    if args.status_port:
        from dlsj.loop.status import StatusServer

        status = StatusServer(controller, cameras, port=args.status_port,
                              on_next=gate.set if gate else None).start()

    worker = threading.Thread(target=controller.run, name="controller", daemon=True)
    worker.start()
    try:
        while worker.is_alive():
            worker.join(0.5)
    except KeyboardInterrupt:
        log.warning("Ctrl+C: aborting, then returning home")
        controller.abort()
        if gate:
            gate.set(False)
        try:
            worker.join(30)
        except KeyboardInterrupt:
            log.warning("second Ctrl+C: exiting now")
    finally:
        summary = runlog.close()
        for fn in cleanup:
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                log.warning("cleanup: %s", e)
        if status:
            status.stop()

    print("\n=== bricks ===")
    for r in controller.reports:
        print(f"  #{r.task_id}: {r.status:12s} label={r.label.value if r.label else '-':10s} forced={r.forced!s:5s} "
              f"-> {r.place_skill or '-':10s} inspections={len(r.inspections)} {r.duration_s:.1f}s")
    if world is not None:
        print("  fake world placed:", world.placed)
    print("\n=== latency (ms) ===")
    for stage, s in summary.items():
        if s.get("n"):
            print(f"  {stage:22s} n={s['n']:4d}  median={s['median']:9.2f}  p90={s['p90']:9.2f}  p99={s['p99']:9.2f}")
    print(f"\nlogs: {runlog.run_dir}")
    return 0 if controller.state.value == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
