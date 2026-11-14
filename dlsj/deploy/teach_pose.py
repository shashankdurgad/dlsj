"""Teach drop-off poses for the pick→inspect→place loop (act_rollout --loop --place NAME).

    python -m dlsj.deploy.teach_pose good defect

For each name: the follower's torque is switched off, you hold the arm with the gripper just above that bowl
(brick-holding height, gripper closed) and press Enter; the pose is saved to
calibration/robots/so_follower/<follower>.place_poses.json in the current calibration. At the end the arm holds
where it is (goal = present, torque on), so nothing drops.
"""

import argparse
import logging

from dlsj import arms

log = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("names", nargs="+", help="pose names, e.g. good defect")
    p.add_argument("--follower-port", default=None)
    p.add_argument("--follower-id", default="follower")
    args = p.parse_args()

    from lerobot.robots import make_robot_from_config
    from lerobot.robots.so_follower import SO101FollowerConfig

    port = args.follower_port or arms.find_ports()[0]
    if not port:
        p.error("follower not found; check USB and the 12 V supply")
    robot = make_robot_from_config(SO101FollowerConfig(port=port, id=args.follower_id,
                                                       calibration_dir=arms.FOLLOWER_CAL_DIR,
                                                       disable_torque_on_disconnect=False))
    robot.connect(calibrate=False)
    try:
        for name in args.names:
            input(f"\nSupport the arm. Torque goes OFF for '{name}'. Press Enter when you are holding it...")
            robot.bus.disable_torque()
            input(f"Hold the gripper just above the '{name}' bowl (closed, brick height), then press Enter...")
            obs = robot.get_observation()
            pose = {k: float(v) for k, v in obs.items() if k.endswith(".pos")}
            path = arms.save_place_pose(args.follower_id, name, pose)
            log.info("saved '%s' = %s -> %s", name, {k: round(v, 1) for k, v in pose.items()}, path)
            arms.hold_present_goal(robot.bus, "follower")  # goal = present; the write turns torque back on
            robot.bus.enable_torque()
    finally:
        robot.disconnect()
    print("Done. Run the loop with: act_rollout ... --loop --place", args.names[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
