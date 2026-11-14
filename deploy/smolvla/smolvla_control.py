"""MANUAL / TESTING CONTROLLER. The loop we actually run is `auto_control.py` (wraps this file and adds the
automatic spin-and-inspect + bowl decision). Use this one only to test the VLA by hand (M + 1/2/3); keep its classes
and main() stable because auto_control.py patches `make_controller_class` and calls `main()`.

SmolVLA (3-stage, wrist + top) on the SO-101 via Desmond's console (`dlsj.deploy.pi05_control`), with RTC.

Reuses that controller (web console :8010, speed limits, Q = hold, ramp home on exit) and changes:
* prompts: the 3-stage strings the model was trained on. Space = "pick up the brick"; M (once grasped) = "bring the
  brick to the inspection camera"; M again = "place the brick in the <bowl> bowl" (choose the bowl with 1/2/3 first);
* cameras: the policy reads only `wrist` and `top` (full frame, no crop);
* `--joint-offset joint=deg,...`: current calibration minus the training-time one (as in act_rollout); the policy
  sees state - offset, the arm gets action + offset;
* RTC from `deploy/pi05/rtc_control.py` when the export has RTC inputs (`deploy/smolvla/export.py --rtc`).

    cd ~/inference/dlsj-smolvla && PYTHONPATH=$PWD ~/desmond/venvs/pi05rt/bin/python \
        ~/inference/deploy/smolvla/smolvla_control.py --export ~/inference/exports/smolvla_3stage_rtc_int8 \
        --device GPU --rtc-horizon 15 --rtc-guidance 3 --joint-offset recal_2026-11-14 --max-step 4 \
        --cameras wrist=/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0 \
        --realsense top=346422061366,inspect=243722062690
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pi05"))
sys.path.insert(0, str(Path("~/inference/deploy/pi05").expanduser()))
import rtc_control as R  # noqa: E402

log = logging.getLogger("smolvla_control")
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
PICK = "pick up the brick"
BRING = "bring the brick to the inspection camera"
PLACE = "place the brick in the {colour} bowl"


def parse_offset(spec: str) -> np.ndarray:
    """joint=deg pairs, or a name from configs/joint_offsets.json (e.g. recal_2026-11-14), like act_rollout."""
    off = np.zeros(6)
    if spec and "=" not in spec:
        import json

        from dlsj import arms

        table = json.loads((Path(arms.__file__).resolve().parents[1] / "configs" / "joint_offsets.json").read_text())
        if spec not in table or spec.startswith("_"):
            raise SystemExit(f"--joint-offset: no entry {spec!r} in configs/joint_offsets.json")
        spec = ",".join(f"{k}={v}" for k, v in table[spec]["offset_deg"].items())
    for kv in filter(None, (s.strip() for s in spec.split(","))):
        k, v = kv.split("=")
        if k not in JOINTS:
            raise SystemExit(f"--joint-offset: unknown joint {k} (joints: {JOINTS})")
        off[JOINTS.index(k)] = float(v)
    return off


def make_policy_class(base, offset: np.ndarray):
    class SmolPolicy(base):
        """Pi05Policy-compatible wrapper: only the cameras the export asks for; calibration offset applied."""

        def __call__(self, state, images, task):
            s = np.asarray(state, np.float32) - offset.astype(np.float32)
            a = super().__call__(s, images, task)
            return np.asarray(a) + offset

    return SmolPolicy


def make_controller_class(base):
    class SmolController(base):
        def prompt(self) -> str:
            if self.stage == "pick":
                return PICK
            if self.stage == "bring":
                return BRING
            return PLACE.format(colour=self.bowl or "yellow")

        def _handle(self, c: str) -> None:
            # The console only sends space / m / 1-3 / +- / q, so M advances the stage:
            # pick --M--> bring (brick grasped) --M--> place (needs a bowl chosen with 1/2/3).
            if c in ("g", "grasp", "bring", "mark", "m") and self.phase == "running" and self.stage == "pick" and BRING:
                self.stage = "bring"
                self._replan_now()
                self.message = "Stage: bring the brick to the inspection camera (M again to place)"
                return
            if c in ("mark", "m") and self.phase == "running" and self.stage in (("bring",) if BRING else ("pick",)):
                if self.bowl is None:
                    self.message = "Choose a plate (1-3), then M to place"
                    return
                self.stage = "place"
                self._replan_now()
                return
            super()._handle(c)

        def _obs(self):
            state, imgs = super()._obs()
            return state, {k: v for k, v in imgs.items() if k in ("wrist", "top")}

    return SmolController


def main() -> None:
    global PICK, BRING, PLACE
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--joint-offset", default="")
    ap.add_argument("--rtc-horizon", type=int, default=15)
    ap.add_argument("--rtc-guidance", type=float, default=3.0)
    ap.add_argument("--pick-prompt", default=PICK)
    ap.add_argument("--bring-prompt", default=BRING, help="'' = no bring stage (M goes pick -> place)")
    ap.add_argument("--place-prompt", default=PLACE, help="with {colour}")
    ours, rest = ap.parse_known_args()
    PICK, BRING, PLACE = ours.pick_prompt, ours.bring_prompt, ours.place_prompt
    offset = parse_offset(ours.joint_offset)

    from dlsj.deploy import pi05_control as pc
    from dlsj.deploy.pi05_runtime import Pi05Policy

    if "--crop" in " ".join(rest):
        raise SystemExit("SmolVLA was trained on the full top frame: drop --crop")
    pc.PICK = PICK  # used by warmup() and the console's default prompt
    base_policy = make_policy_class(Pi05Policy, offset)
    use_rtc = True
    try:
        policy_cls = R.make_rtc_policy_class(base_policy)
    except Exception:  # noqa: BLE001
        use_rtc = False
    ctl_cls = make_controller_class(R.make_rtc_controller_class(pc.Controller))

    def make_policy(args):
        # MolmoAct2: its own wrapper (RTC inputs are added after the runtime preprocessors, which drop unknown keys;
        # plain exports fall back inside the wrapper without a second 5B-model load)
        try:
            import json as _json

            name = _json.loads((Path(args.export).expanduser() / "manifest.json").read_text())["policy"]["name"]
        except Exception:  # noqa: BLE001
            name = ""
        if name == "molmoact2":
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "molmoact2"))
            sys.path.insert(0, str(Path("~/inference/deploy/molmoact2").expanduser()))
            from rtc_patch import make_molmo_rtc_policy_class

            if offset.any():
                log.warning("--joint-offset ignored for MolmoAct2 (its export maps the current calibration itself)")
            pol = make_molmo_rtc_policy_class(Pi05Policy)(args.export, args.device, cache_dir=args.cache_dir,
                                                         horizon=ours.rtc_horizon, guidance=ours.rtc_guidance)
            log.info("MolmoAct2 policy: %s", "RTC" if pol.rtc_enabled else "plain chunking (export has no RTC inputs)")
            return pol
        # RTC if the loaded model takes the RTC inputs (checked on the model's input features, not the manifest
        # text: some exports don't spell the RTC inputs out in manifest.json)
        try:
            return policy_cls(args.export, args.device, cache_dir=args.cache_dir, horizon=ours.rtc_horizon,
                              guidance=ours.rtc_guidance)
        except SystemExit as e:
            log.warning("%s -> plain chunking", e)
            return base_policy(args.export, args.device, cache_dir=args.cache_dir)

    def controller(*a, **kw):
        c = ctl_cls(*a, horizon=ours.rtc_horizon, **kw)
        if not getattr(c.policy, "rtc_enabled", hasattr(c.policy, "rtc_next")):
            c.blend = "crossfade"
        return c

    pc.make_policy = make_policy
    pc.Controller = controller
    sys.argv = [sys.argv[0], *rest]
    logging.basicConfig(level=logging.INFO)
    log.info("prompts: %r / %r / %r", PICK, BRING, PLACE)
    log.info("SmolVLA 3-stage: offset %s, RTC h=%d g=%.1f", dict(zip(JOINTS, offset)), ours.rtc_horizon,
             ours.rtc_guidance)
    pc.main()


if __name__ == "__main__":
    main()
