"""Fixed, tunable inspection pose: after a confirmed grasp the arm is driven to the SAME pose in front of the inspection
camera for every brick (instead of the VLA's bring stage), then the wrist spin runs there.

* The pose lives next to the calibration like the home pose (`calibration/robots/so_follower/<id>.inspect.json`,
  `dlsj.loop.poses`), so it follows the current calibration. Without a saved pose: `DEMO_POSE`.
* `DEMO_POSE`: median arm pose of the 130 cal2-456 demos where the held brick was presented close (>= 280 px) and
  centred on the inspection camera at the operator's "at the camera" mark (`deploy/smolvla/README` / experiment log,
  15 Nov). The demos hold the brick a few cm from the lens, 310-410 px big.
* `Mover`: joint-space easing through waypoints (grasp -> raised ready pose -> inspect pose) at a capped speed, gripper
  command untouched (it keeps squeezing the brick), using the controller's own rate-limited `_step`.
* `nudge()`: per-joint tuning steps for the dashboard buttons.
"""

from __future__ import annotations

import numpy as np

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
DEMO_POSE = (-9.4, 13.0, -6.8, 53.6, 3.8)  # median of 130 close+centred demo presentations (cal2 calibration)
DEMO_MEDOID_EP = 100  # most typical single demo of those: [-8.6, 11.1, -7.3, 56.6, 5.1]
NUDGE_LIMIT_DEG = 15.0  # one nudge never moves a joint more than this


def load(follower_id: str = "follower", cal_dir=None) -> tuple[np.ndarray, str]:
    """(5 arm joints, "saved" | "demo")."""
    try:
        from dlsj.loop.poses import load_pose

        v = load_pose(follower_id, "inspect", cal_dir)
        if v is not None:
            return np.asarray(v, float)[:5], "saved"
    except Exception:  # noqa: BLE001  no dlsj / unreadable file: fall back to the demo pose
        pass
    return np.asarray(DEMO_POSE, float), "demo"


def save(pose5, follower_id: str = "follower", gripper: float = 5.6, cal_dir=None) -> str:
    from dlsj.loop.poses import save_pose

    v = np.concatenate([np.asarray(pose5, float)[:5], [gripper]])
    return str(save_pose(follower_id, "inspect", v, cal_dir))


def nudge(pose5, joint: str, delta: float) -> np.ndarray:
    if joint not in JOINTS:
        raise ValueError(f"unknown joint {joint!r}")
    out = np.asarray(pose5, float).copy()
    out[JOINTS.index(joint)] += float(np.clip(delta, -NUDGE_LIMIT_DEG, NUDGE_LIMIT_DEG))
    return out


class Mover:
    """Ease the 5 arm joints through `waypoints` at `speed` deg/s (gripper unchanged). Call `step(ctl)` every tick."""

    def __init__(self, waypoints, speed: float = 30.0, tol: float = 2.5):
        self.waypoints = [np.asarray(w, float)[:5] for w in waypoints if w is not None]
        self.speed, self.tol, self.i = speed, tol, 0
        self._at_cmd_ticks = 0

    @property
    def done(self) -> bool:
        return self.i >= len(self.waypoints)

    def step(self, ctl) -> bool:
        """One rate-limited command towards the current waypoint; True once the last waypoint is reached."""
        if self.done:
            return True
        base = np.asarray(ctl.cmd if ctl.cmd is not None else ctl.arm.read(), float)
        target = base.copy()
        target[:5] = self.waypoints[self.i]
        step = self.speed / max(1, getattr(ctl, "fps", 30))
        cmd = base + np.clip(target - base, -step, step)
        measured = np.asarray(ctl._step(cmd), float)
        # reached = the command is exactly at the waypoint AND (the arm is within tol of it, or it has stayed as close
        # as it gets for 1 s: a loaded shoulder sags ~4 deg below its command and would otherwise never "arrive")
        at_cmd = np.abs(cmd[:5] - target[:5]).max() < 1e-6
        self._at_cmd_ticks = self._at_cmd_ticks + 1 if at_cmd else 0
        if at_cmd and (np.abs(measured[:5] - target[:5]).max() < self.tol
                       or self._at_cmd_ticks >= getattr(ctl, "fps", 30)):
            self.i += 1
            self._at_cmd_ticks = 0
        return self.done


# ---------------------------------------------------------------------- offset from the VLA's own bring endpoint
def load_offset(follower_id: str = "follower", cal_dir=None) -> tuple[np.ndarray, str]:
    """(5-joint offset added to where the VLA's bring stage stops, "saved" | "default (zero)")."""
    import json

    try:
        from dlsj.loop.poses import pose_path

        p = pose_path(follower_id, "inspect_offset", cal_dir)
        if p.exists():
            d = json.loads(p.read_text())
            return np.array([float(d.get(f"{j}.pos", 0.0)) for j in JOINTS]), "saved"
    except Exception:  # noqa: BLE001
        pass
    return np.zeros(5), "zero (not saved)"


def save_offset(offset5, follower_id: str = "follower", cal_dir=None) -> str:
    import json

    from dlsj.loop.poses import pose_path

    p = pose_path(follower_id, "inspect_offset", cal_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({f"{j}.pos": round(float(v), 2) for j, v in zip(JOINTS, offset5)}, indent=2) + "\n")
    return str(p)
