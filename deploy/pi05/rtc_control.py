"""Real-Time Chunking (RTC) for the π0.5 arm controller (`dlsj.deploy.pi05_control`, Desmond's console).

Reuses that controller unchanged (web console on :8010, camera crops, per-tick speed limits, Q = hold, mark /
bowl keys, ramp home on exit) and swaps only how consecutive chunks are joined:

* the policy is an **RTC export** (`deploy/pi05/studio_export.py --rtc`): its graph takes the unexecuted tail of
  the current plan (`prev_chunk_left_over`), the expected inference delay in actions, the execution horizon and
  the guidance weight, and steers the new chunk to continue that tail (Studio's gradient-free RTC correction);
* the controller asks for the next chunk when fewer than `horizon + delay` planned actions remain (the threshold
  `physicalai.runtime.RTCExecution` uses), where `delay = ceil(worst recent latency x fps)`;
* when the chunk arrives, actions whose tick passed during inference are dropped and the rest replace the plan
  (the controller's `none` blend, i.e. `RTCActionQueue.merge`): no averaging of two different plans.

A new prompt (mark / bowl change) or a restart clears the plan, so the first chunk after it is unguided.

    cd ~/desmond/build/dlsj-pi05 && PYTHONPATH=$PWD ~/desmond/venvs/pi05rt/bin/python \
        ~/inference/deploy/pi05/rtc_control.py --export ~/inference/exports/pi05_ft9060_rtc_tok64_5step_int8 \
        --device GPU --crop top=0,0,560,480 --rtc-horizon 15 --rtc-guidance 5 \
        --cameras wrist=/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0 \
        --realsense top=346422061366,inspect=243722062690

All other flags are `pi05_control`'s (`--blend` / `--replan` are ignored: RTC sets both).
"""

from __future__ import annotations

import argparse
import logging
import math
import sys

import numpy as np

log = logging.getLogger("pi05_rtc")

# physicalai.inference.constants (kept literal so this module imports without physicalai for tests)
PREV_CHUNK_LEFT_OVER = "prev_chunk_left_over"
RTC_INFERENCE_DELAY = "inference_delay"
RTC_MAX_GUIDANCE_WEIGHT = "max_guidance_weight"
RTC_EXECUTION_HORIZON = "execution_horizon"


def rtc_inputs(left_over: np.ndarray | None, delay: int, horizon: int, guidance: float, chunk: int,
               action_dim: int) -> dict:
    """The four RTC graph inputs, exactly as `RTCExecution._inject_rtc_inputs` builds them.

    `left_over`: (n, action_dim) planned actions not yet executed (robot units; the export's normaliser handles
    them). None or empty = first chunk: zero tail, guidance ~0, horizon 1, delay 0 (an unguided sample).
    """
    if left_over is None or len(left_over) == 0:
        prev = np.zeros((1, chunk, action_dim), np.float32)
        g, h, d = 1e-8, 1, 0
    else:
        tail = np.asarray(left_over, np.float32)[:chunk]
        prev = np.zeros((1, chunk, action_dim), np.float32)
        prev[0, :len(tail)] = tail
        g, h = float(guidance), int(horizon)
        d = int(min(max(delay, 0), h))
    return {PREV_CHUNK_LEFT_OVER: prev, RTC_INFERENCE_DELAY: np.int64(d),
            RTC_MAX_GUIDANCE_WEIGHT: np.float32(g), RTC_EXECUTION_HORIZON: np.int64(h)}


def delay_actions(latencies_ms, fps: float, horizon: int, window: int = 10) -> int:
    """Actions that pass during one inference: ceil(worst of the last `window` latencies x fps), <= horizon."""
    lat = list(latencies_ms)[-window:]
    if not lat:
        return 0
    return int(min(horizon, math.ceil(max(lat) / 1e3 * fps)))


def make_rtc_policy_class(base):
    """Subclass of `dlsj.deploy.pi05_runtime.Pi05Policy` that adds the RTC inputs to every call."""

    class RTCPi05Policy(base):
        def __init__(self, *a, horizon: int = 15, guidance: float = 5.0, chunk: int = 50, **kw):
            super().__init__(*a, **kw)
            names = {f.name for f in self.model.input_features}
            if PREV_CHUNK_LEFT_OVER not in names:
                raise SystemExit("this export has no RTC inputs: re-export with studio_export.py --rtc")
            self.horizon, self.guidance, self.chunk = horizon, guidance, chunk
            self.rtc_next: tuple[np.ndarray | None, int] | None = None  # set by the controller per request
            self.last_rtc: dict = {}

        def __call__(self, state, images, task):
            nxt, self.rtc_next = self.rtc_next, None
            left, delay = nxt if nxt is not None else (None, 0)
            obs = {"state": np.asarray(state, np.float32)[None], "task": [task]}
            for cam in self.cameras:
                obs[f"images.{cam}"] = np.ascontiguousarray(np.asarray(images[cam], np.uint8).transpose(2, 0, 1)[None])
            extra = rtc_inputs(left, delay, self.horizon, self.guidance, self.chunk, len(state))
            obs.update(extra)
            self.last_rtc = {"tail": 0 if left is None else len(left), "delay": int(extra[RTC_INFERENCE_DELAY]),
                             "guidance": float(extra[RTC_MAX_GUIDANCE_WEIGHT])}
            return self.model.predict_action_chunk(obs)

    return RTCPi05Policy


def make_rtc_controller_class(base):
    """Subclass of `pi05_control.Controller`: RTC request threshold and tail hand-off, plan replaced on arrival."""

    class RTCController(base):
        def __init__(self, *a, horizon: int = 15, **kw):
            super().__init__(*a, **kw)
            self.blend = "none"  # replace the plan, trimming actions that passed during inference
            self.horizon = horizon

        def delay(self) -> int:
            return delay_actions(self.lat_ms, self.fps, self.horizon)

        def tick(self) -> None:
            self.replan = self.horizon + self.delay()  # RTCExecution's queue threshold
            super().tick()

        def left_over(self) -> np.ndarray | None:
            """Planned targets from the current tick on (called under the lock, at request time)."""
            if not self.chunks:
                return None
            s, c = self.chunks[-1]
            tail = c[max(0, self.t - s):]
            return tail if len(tail) else None

        def _request(self) -> None:
            self.policy.rtc_next = (self.left_over(), self.delay())
            super()._request()

        def snapshot(self) -> dict:
            d = super().snapshot()
            rtc_on = getattr(self.policy, "rtc_enabled", hasattr(self.policy, "guidance"))
            d.update({"blend": f"rtc(h={self.horizon}, g={getattr(self.policy, 'guidance', '?')})" if rtc_on
                      else f"crossfade (no RTC inputs in this export; replan at h={self.horizon}+delay)",
                      "rtc": getattr(self.policy, "last_rtc", {})})
            return d

    return RTCController


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--rtc-horizon", type=int, default=15, help="fresh actions per chunk (<= chunk/2 = 25)")
    ap.add_argument("--rtc-guidance", type=float, default=5.0, help="RTC guidance weight (paper's beta)")
    ours, rest = ap.parse_known_args(sys.argv[1:] if argv is None else argv)
    if not 1 <= ours.rtc_horizon <= 25:
        raise SystemExit("--rtc-horizon must be in 1..25 (at most half the 50-action chunk)")

    from dlsj.deploy import pi05_control as pc
    from dlsj.deploy.pi05_runtime import Pi05Policy

    policy_cls = make_rtc_policy_class(Pi05Policy)
    ctl_cls = make_rtc_controller_class(pc.Controller)

    def make_policy(args):
        if args.backend != "openvino":
            raise SystemExit("RTC needs the OpenVINO RTC export (--backend openvino)")
        return policy_cls(args.export, args.device, cache_dir=args.cache_dir, horizon=ours.rtc_horizon,
                          guidance=ours.rtc_guidance)

    def controller(*a, **kw):
        return ctl_cls(*a, horizon=ours.rtc_horizon, **kw)

    pc.make_policy = make_policy
    pc.Controller = controller
    sys.argv = [sys.argv[0], *rest]
    log.info("RTC: horizon %d, guidance %.1f", ours.rtc_horizon, ours.rtc_guidance)
    pc.main()


if __name__ == "__main__":
    main()
