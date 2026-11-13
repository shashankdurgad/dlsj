"""Give a replacement SO-101 servo its ID (new STS3215s ship as ID 1, which clashes with shoulder_pan).

    uv run python -m dlsj.servo_id --joint shoulder_lift            # port found by voltage (follower)
    uv run python -m dlsj.servo_id --joint gripper --port /dev/ttyACM1

Steps (it prompts for each):
1. Connect ONLY the new servo to the arm's controller board (unplug the daisy-chain cable from the others).
2. It checks exactly one servo answers, then writes the joint's ID and 1 Mbps. It refuses if it sees more than one.
3. Reconnect the chain; it checks that IDs 1..6 all answer.

The new servo's encoder zero differs from the old one, so recalibrate that arm afterwards (--calibrate follower).
"""

import argparse
import warnings

from dlsj import arms

JOINT_IDS = {m: i + 1 for i, m in enumerate(arms.MOTORS)}
BAUDS = (1_000_000, 500_000, 250_000, 128_000, 115_200, 57_600, 38_400, 19_200)


def _bus(port: str, ids):
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return FeetechMotorsBus(port, {f"id{i}": Motor(i, "sts3215", MotorNormMode.RANGE_M100_100) for i in ids})


def scan(port: str, ids=range(1, 13), bauds=BAUDS) -> dict[int, list[int]]:
    """{baud: [ids that answer]} for every baud rate where something answers."""
    bus = _bus(port, ids)
    bus.connect(handshake=False)
    found = {}
    try:
        for baud in bauds:
            bus.set_baudrate(baud)
            hits = []
            for i in ids:
                try:
                    if bus.ping(f"id{i}", num_retry=1) is not None:
                        hits.append(i)
                except Exception:  # noqa: BLE001  (no answer / garbled)
                    pass
            if hits:
                found[baud] = hits
    finally:
        bus.set_baudrate(1_000_000)
        bus.disconnect(disable_torque=False)
    return found


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--joint", required=True, choices=list(JOINT_IDS))
    p.add_argument("--port", default="auto", help="'auto' = the follower (12 V); pass the leader's port explicitly")
    args = p.parse_args()

    port = args.port
    if port == "auto":
        port, leader, volts = arms.find_ports()
        if port is None:
            # An ID clash garbles servo 1, so the follower's voltage probe fails: take the port that isn't the leader.
            others = [c for c in arms.serial_candidates() if c != leader]
            if leader is None or len(others) != 1:
                raise SystemExit(f"could not tell which port is the follower ({volts}); pass --port")
            port = others[0]
    target = JOINT_IDS[args.joint]
    print(f"Arm on {port}. Setting the new servo to ID {target} ({args.joint}).")

    input("\n1) Connect ONLY the new servo to this arm's controller board, then press ENTER...")
    found = scan(port)
    answering = sorted({i for ids in found.values() for i in ids})
    if len(answering) != 1 or sum(len(v) for v in found.values()) != 1:
        raise SystemExit(f"Expected exactly one servo, found {found or 'none'}. Nothing written.")
    (baud, (cur_id,)), = found.items()
    print(f"   found one servo: ID {cur_id} at {baud} baud")

    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bus = FeetechMotorsBus(port, {args.joint: Motor(target, "sts3215", MotorNormMode.RANGE_M100_100)})
    bus.setup_motor(args.joint, initial_baudrate=baud, initial_id=cur_id)
    bus.disconnect(disable_torque=False)
    print(f"   written: ID {target}, 1000000 baud")

    input("\n2) Reconnect the full daisy chain, then press ENTER...")
    after = scan(port, bauds=(1_000_000,)).get(1_000_000, [])
    missing = [i for i in range(1, 7) if i not in after]
    if missing:
        raise SystemExit(f"IDs answering: {after}; missing {missing}. Check the cables.")
    print(f"   all six servos answer: {after}")
    print("\nNow recalibrate this arm, e.g. scripts/xps.sh teleop --calibrate follower")


if __name__ == "__main__":
    main()
