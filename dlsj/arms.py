"""SO-101 arm helpers shared by the station tools: port discovery, calibration storage and checks, home pose.

Calibration lives in the repo under `calibration/` (not LeRobot's ~/.cache default) so it is versioned and
travels with the code to the station laptop. Point LeRobot's own tools at the same files with
`--robot.calibration_dir=calibration/robots/so_follower --teleop.calibration_dir=calibration/teleoperators/so_leader`.
"""

import glob
import json
import logging
import sys
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
CALIBRATION_ROOT = REPO_ROOT / "calibration"
FOLLOWER_CAL_DIR = CALIBRATION_ROOT / "robots" / "so_follower"
LEADER_CAL_DIR = CALIBRATION_ROOT / "teleoperators" / "so_leader"

MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
FULL_TURN = {"wrist_roll"}  # LeRobot records this one as 0..4095 by design
# The follower runs its STS3215s from a 12 V brick; the leader from 5 V. Anything above this is a follower.
FOLLOWER_MIN_VOLTS = 9.0
MIN_SPAN = 300  # raw encoder ticks a joint must have moved during calibration
EDGE = 10  # raw ticks from 0 / 4095 that indicate the range wrapped around the encoder


def serial_candidates() -> list[str]:
    pats = ["/dev/tty.usbmodem*", "/dev/ttyACM*", "/dev/ttyUSB*"] if sys.platform != "win32" else []
    return sorted({p for pat in pats for p in glob.glob(pat)})


def read_supply_volts(port: str) -> float | None:
    """Supply voltage seen by servo 1 on `port`, or None if no SO-101 servo answers there."""
    import warnings

    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bus = FeetechMotorsBus(port, {"m1": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100)})
    try:
        bus.connect(handshake=False)
        return bus.read("Present_Voltage", "m1", normalize=False) / 10.0
    except Exception as e:  # noqa: BLE001
        log.debug("no servo on %s: %s", port, e)
        return None
    finally:
        try:
            bus.disconnect(disable_torque=False)
        except Exception:  # noqa: BLE001
            pass


def classify_ports(volts: dict[str, float | None]) -> tuple[str | None, str | None]:
    """(follower_port, leader_port) from each port's servo supply voltage."""
    arms = {p: v for p, v in volts.items() if v is not None}
    followers = [p for p, v in arms.items() if v >= FOLLOWER_MIN_VOLTS]
    leaders = [p for p, v in arms.items() if v < FOLLOWER_MIN_VOLTS]
    if len(followers) > 1 or len(leaders) > 1:
        raise RuntimeError(f"ambiguous arms by voltage {arms}; pass --follower-port/--leader-port")
    return (followers[0] if followers else None, leaders[0] if leaders else None)


def find_ports() -> tuple[str | None, str | None, dict[str, float | None]]:
    volts = {p: read_supply_volts(p) for p in serial_candidates()}
    follower, leader = classify_ports(volts)
    return follower, leader, volts


def calibration_problems(cal: dict) -> list[str]:
    """Human-readable problems with a LeRobot calibration dict ({motor: {range_min, range_max, ...}})."""
    problems = []
    for motor in MOTORS:
        c = cal.get(motor)
        if c is None:
            problems.append(f"{motor}: missing")
            continue
        lo, hi = c["range_min"], c["range_max"]
        if motor in FULL_TURN:
            continue
        if hi - lo < MIN_SPAN:
            problems.append(f"{motor}: only moved {hi - lo} ticks ({lo}..{hi}), move it end to end")
        elif lo <= EDGE or hi >= 4095 - EDGE:
            problems.append(
                f"{motor}: range {lo}..{hi} touches the encoder edge, so it wrapped; "
                "start calibration with the joint in the middle of its travel"
            )
    return problems


def _as_plain(cal) -> dict:
    return {
        m: (c if isinstance(c, dict) else {"range_min": c.range_min, "range_max": c.range_max})
        for m, c in cal.items()
    }


def unwrap_step(prev_raw: int, raw: int, res: int = 4096) -> int:
    """Signed shortest move from `prev_raw` to `raw` on a `res`-tick encoder (handles 4095 -> 0)."""
    half = res // 2
    return (raw - prev_raw + half) % res - half


def centred_homing(start_raw: int, lo_u: int, hi_u: int, res: int = 4096) -> tuple[int, int, int]:
    """(raw position at the middle of the travel, range_min, range_max) after homing that middle to res//2-1.

    `lo_u`/`hi_u` are the unwrapped extremes relative to `start_raw` (so they may be < 0 or >= res).
    """
    mid_u = (lo_u + hi_u) // 2
    mid_raw = (start_raw + mid_u) % res
    centre = (res - 1) // 2
    return mid_raw, centre - (mid_u - lo_u), centre + (hi_u - mid_u)


def calibrate_from_rest(device) -> None:
    """Calibrate starting from the folded REST pose instead of LeRobot's middle pose.

    The joints are tracked across the encoder's 4095 -> 0 wrap while you move them, so no start pose can
    wrap a range. Each joint's travel is then centred on the encoder, which gives the same angle convention
    as LeRobot (0 deg = middle of the travel). Saves the file and writes it to the motors.
    """
    from lerobot.motors import MotorCalibration

    bus = device.bus
    motors = list(bus.motors)
    bus.disable_torque(motors)
    bus.set_operating_mode("position", motors)
    for m in motors:
        if bus._has_register(m, "Drive_Mode"):
            bus.write("Drive_Mode", m, 0)

    input("Put the arm in its folded REST pose (the home position) and press ENTER...")
    bus.reset_calibration(motors)  # homing offset 0, full-range limits
    res = {m: bus.resolution(bus._get_motor_model(m)) for m in motors}
    start = {m: int(v) for m, v in bus.sync_read("Present_Position", motors, normalize=False, num_retry=5).items()}
    prev, cur = dict(start), {m: 0 for m in motors}
    lo, hi = dict(cur), dict(cur)

    recorded = [m for m in motors if m not in FULL_TURN]
    print(f"Now move {recorded} one at a time through their ENTIRE range (gripper fully open and closed).")
    print("Press ENTER when done.")
    done = threading.Event()
    threading.Thread(target=lambda: (input(), done.set()), daemon=True).start()
    while not done.is_set():
        raw = bus.sync_read("Present_Position", motors, normalize=False, num_retry=5)
        for m in motors:
            cur[m] += unwrap_step(prev[m], int(raw[m]), res[m])
            prev[m] = int(raw[m])
            lo[m], hi[m] = min(lo[m], cur[m]), max(hi[m], cur[m])
        span = "  ".join(f"{m}:{hi[m] - lo[m]:4d}" for m in recorded)
        sys.stdout.write("\r\033[Kticks moved  " + span)
        sys.stdout.flush()
        time.sleep(0.02)
    print()

    cal = {}
    for m in motors:
        if m in FULL_TURN:
            mid_raw, rmin, rmax = start[m], 0, res[m] - 1  # current pose becomes the encoder middle
        else:
            mid_raw, rmin, rmax = centred_homing(start[m], lo[m], hi[m], res[m])
        d = bus._definition(m)
        off = d.homing_offset_sign * ((res[m] - 1) // 2 - mid_raw)
        lim = (1 << d.register("homing_offset").sign_bit) - 1
        off = max(-lim, min(lim, off))
        bus.write("Homing_Offset", m, off)
        cal[m] = MotorCalibration(id=bus.motors[m].id, drive_mode=0, homing_offset=int(off),
                                  range_min=int(rmin), range_max=int(rmax))
    device.calibration = cal
    bus.write_calibration(cal)
    device._save_calibration()
    print("Calibration saved to", device.calibration_fpath)


def ensure_calibrated(device, label: str, force: bool = False, after_calibration=None) -> bool:
    """Connect `device` (LeRobot robot or teleop), calibrating until the result passes `calibration_problems`.

    Unlike plain `connect()`, a stored file that is valid is used without the "press ENTER" prompt, and a bad
    one (or `force`) triggers a fresh calibration. `after_calibration(device)` runs right after a fresh
    calibration, while torque is still off (e.g. to capture the home pose by hand).
    """
    device.bus.connect()
    stored = device.calibration if device.calibration_fpath.exists() else {}
    recalibrated = False
    if stored and not force:
        if not device.bus.is_calibrated:
            log.info("%s: writing %s to the motors", label, device.calibration_fpath.name)
            device.bus.disable_torque()
            device.bus.write_calibration(device.calibration)
        log.info("%s: using %s", label, device.calibration_fpath)
    else:
        print(f"\n=== Calibrating {label.upper()} from its REST pose ===")
        calibrate_from_rest(device)
        recalibrated = True
        if after_calibration is not None:
            after_calibration(device)
    # Advisory only: with use_degrees the range just sets the centre and the clamp, so we never block.
    warn = calibration_problems(_as_plain(device.calibration))
    if warn:
        log.warning("%s calibration notes (not blocking; --calibrate %s to redo):\n  - %s",
                    label, label, "\n  - ".join(warn))
    if hasattr(device, "send_action"):  # follower only; a goal write also switches a Feetech servo's torque on
        hold_present_goal(device.bus, label)
    device.configure()  # LeRobot enables torque at the end of this for a follower
    return recalibrated


def hold_present_goal(bus, label: str = "") -> None:
    """Set every motor's Goal_Position to where it is now, with torque off.

    Feetech servos drive to the stored goal the moment torque is enabled. A stale goal (we found 0 on every
    joint after another tool had used the arm) makes the follower lunge to its limits at full torque, buzz
    against them and brown out the bus. LeRobot's configure() does not reset it, so we do. Note the write itself
    switches torque on (Feetech behaviour); with goal == present the arm just holds still.
    """
    bus.disable_torque()
    present = bus.sync_read("Present_Position", normalize=False, num_retry=5)
    goal = bus.sync_read("Goal_Position", normalize=False, num_retry=5)
    stale = {m: (int(goal[m]), int(p)) for m, p in present.items() if abs(int(goal[m]) - int(p)) > 20}
    if stale:
        log.warning("%s: stale goal positions reset to present (goal, present): %s", label, stale)
    target = {m: int(p) for m, p in present.items()}
    for _ in range(5):  # sync_write is fire-and-forget: read back until every goal took
        bus.sync_write("Goal_Position", target, normalize=False)
        goal = bus.sync_read("Goal_Position", normalize=False, num_retry=5)
        if all(abs(int(goal[m]) - target[m]) <= 2 for m in target):
            return
    raise RuntimeError(f"{label}: could not set goal positions to present; not enabling torque")


# ----------------------------------------------------------------------------- home pose
def home_pose_path(follower_id: str) -> Path:
    return FOLLOWER_CAL_DIR / f"{follower_id}.home.json"


def load_home(follower_id: str) -> dict[str, float] | None:
    p = home_pose_path(follower_id)
    return json.loads(p.read_text()) if p.exists() else None


def capture_home(device, follower_id: str, prompt=input) -> Path:
    """Ask for the folded REST pose with torque off and save it as the follower's home pose.

    Any previous home (in the old calibration's angles) is set aside first.
    """
    stale = retire_home(follower_id)
    if stale:
        log.info("old home pose moved to %s", stale.name)
    prompt("\nLast step: fold the follower back into its REST pose (this becomes HOME) and press ENTER...")
    pos = device.bus.sync_read("Present_Position", num_retry=5)  # normalised with the new calibration
    p = save_home(follower_id, {f"{m}.pos": float(v) for m, v in pos.items()})
    print("Home pose saved to", p)
    return p


def retire_home(follower_id: str) -> Path | None:
    """A new follower calibration changes what the stored angles mean, so set the old home pose aside."""
    p = home_pose_path(follower_id)
    if not p.exists():
        return None
    stale = p.with_suffix(".json.stale")
    p.replace(stale)
    return stale


def align_path(leader_id: str) -> Path:
    return LEADER_CAL_DIR / f"{leader_id}.align.json"


def load_align(leader_id: str) -> dict[str, float] | None:
    """Per-joint offsets added to the leader's reading so it matches the follower in the same physical pose."""
    p = align_path(leader_id)
    return json.loads(p.read_text()) if p.exists() else None


def save_align(leader_id: str, offsets: dict[str, float]) -> Path:
    p = align_path(leader_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({k: round(float(v), 2) for k, v in offsets.items()}, indent=2) + "\n")
    return p


def save_home(follower_id: str, pose: dict[str, float]) -> Path:
    p = home_pose_path(follower_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({k: round(float(v), 2) for k, v in pose.items()}, indent=2) + "\n")
    return p
