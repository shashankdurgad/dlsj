"""Serve a LeRobot Diffusion Policy through the pi0.5 web console (dlsj.deploy.pi05_control).

Same page, keys and safety as the pi0.5 console (Space start, Q stop, home, ready pose, step limits, sync band,
trace), with the policy swapped for a Diffusion Policy checkpoint:

* the console asks for a chunk from a background thread; we answer with DDIM sampling from a fixed noise
  (consistent chunks), n_action_steps = horizon - n_obs_steps + 1 actions starting at the request tick;
* Diffusion Policy needs its n_obs_steps (2) observations 1/30 s apart, as in training. The console only hands
  the policy the frame at request time, so every control tick we record (measured state, processed images) and
  the policy uses the last two ticks;
* images are prepared exactly like the training set (scripts/build_pick_set.py): crop the top camera, pad to
  640x480 with black, resize to --resize with INTER_AREA, RGB;
* no calibration offset: use models trained in the current calibration (e.g. dp_v3), or pass --joint-offset.

pi0.5_control lives in the dlsj-pi05 checkout, so run it from there (like serve_full.py):

    cd ~/desmond/build/dlsj-pi05 && PYTHONPATH=$PWD ~/desmond/venvs/teleop/bin/python \\
        ~/desmond/build/dlsj-main/scripts/serve_dp_console.py --policy-path ~/desmond/models/dp_v3_014000 \\
        --dp-device xpu --cameras wrist=/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0 \\
        --realsense top=346422061366,inspect=243722062690 --crop top=0,0,560,480 --port 8020
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import logging
import sys
import threading
from pathlib import Path

import numpy as np

log = logging.getLogger("serve_dp")
HERE = Path(__file__).resolve().parents[1]  # the dlsj-main checkout


def load_async_chunker():
    """AsyncChunker from this checkout, by file path: the `dlsj` package on sys.path is the pi05 checkout."""
    spec = importlib.util.spec_from_file_location("dp_async_chunks", HERE / "dlsj" / "deploy" / "async_chunks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.AsyncChunker


def prep(rgb: np.ndarray, pad_to: tuple[int, int], resize: tuple[int, int]) -> np.ndarray:
    """Cropped RGB frame -> padded (W, H) -> resized (H, W), as scripts/build_pick_set.py."""
    import cv2

    if rgb.shape[:2] != (pad_to[1], pad_to[0]):
        out = np.zeros((pad_to[1], pad_to[0], 3), dtype=rgb.dtype)
        out[: rgb.shape[0], : rgb.shape[1]] = rgb
        rgb = out
    if rgb.shape[:2] != tuple(resize):
        rgb = cv2.resize(rgb, (resize[1], resize[0]), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(rgb)


class DPPolicy:
    """policy(state, images, task) -> (n, 6) actions, for pi05_control.Controller."""

    def __init__(self, path: str, device: str, steps: int, pad_to, resize, offset: np.ndarray):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        cfg = PreTrainedConfig.from_pretrained(path)
        if cfg.type != "diffusion":
            raise SystemExit(f"{path} is a {cfg.type} policy; this server is for Diffusion Policy")
        cfg.device = device
        cfg.num_inference_steps = steps
        cfg.n_action_steps = cfg.horizon - cfg.n_obs_steps + 1
        self.policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg).to(torch.device(device)).eval()
        pre, post = make_pre_post_processors(self.policy.config, pretrained_path=path,
                                             preprocessor_overrides={"device_processor": {"device": device}})
        self.chunker = load_async_chunker()(self.policy, pre, post, torch.device(device))  # predict() only
        self.cams = [k.removeprefix("observation.images.") for k in cfg.image_features]
        self.n_obs, self.pad_to, self.resize = cfg.n_obs_steps, pad_to, resize
        self.offset = np.asarray(offset, np.float32)
        self.hist: collections.deque = collections.deque(maxlen=self.n_obs)  # (state, {cam: rgb}) per control tick
        self.lock = threading.Lock()
        log.info("diffusion policy %s on %s: %d DDIM steps, %d actions per chunk, cameras %s", path, device, steps,
                 cfg.n_action_steps, self.cams)

    def obs(self, state, images: dict) -> dict:
        o = {"observation.state": np.asarray(state, np.float32) - self.offset}
        for c in self.cams:
            o[f"observation.images.{c}"] = prep(images[c], self.pad_to, self.resize)
        return o

    def record(self, state, images: dict) -> None:
        o = self.obs(state, images)
        with self.lock:
            self.hist.append(o)

    def clear(self) -> None:
        with self.lock:
            self.hist.clear()

    def __call__(self, state, images, task):
        with self.lock:
            snap = list(self.hist)
        if len(snap) < self.n_obs:  # warm-up, or the first request after Space: repeat the request frame
            snap = [self.obs(state, images)] * self.n_obs
        acts = self.chunker.predict(list(enumerate(snap)))
        return np.asarray(acts, np.float32) + self.offset


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--policy-path", required=True)
    ap.add_argument("--dp-device", default="xpu")
    ap.add_argument("--dp-steps", type=int, default=8, help="DDIM steps (fewer = less latency)")
    ap.add_argument("--pad-to", default="640,480")
    ap.add_argument("--resize", default="288,384")
    ap.add_argument("--joint-offset", default="", help="name in configs/joint_offsets.json or joint=deg pairs; "
                                                       "empty for models trained in the current calibration")
    ours, rest = ap.parse_known_args()
    joints = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    offset = np.zeros(6, np.float32)
    spec = ours.joint_offset
    if spec and "=" not in spec:
        spec = ",".join(f"{k}={v}" for k, v in
                        json.loads((HERE / "configs" / "joint_offsets.json").read_text())[spec]["offset_deg"].items())
    for kv in filter(None, spec.split(",")):
        k, v = kv.split("=")
        offset[joints.index(k)] = float(v)
    pad_to = tuple(int(v) for v in ours.pad_to.split(","))
    resize = tuple(int(v) for v in ours.resize.split(","))

    from dlsj.deploy import pi05_control as pc

    holder: dict = {}

    def make_policy(args):
        holder["p"] = DPPolicy(ours.policy_path, ours.dp_device, ours.dp_steps, pad_to, resize, offset)
        return holder["p"]

    class DPController(pc.Controller):
        """Records every control tick's observation for the policy's 1/30 s history."""

        def _step(self, target):
            self._filt = None  # no command smoothing: it is for the 5-step pi0.5 export's buzz and adds ~3 ticks lag
            measured = super()._step(target)
            p = holder.get("p")
            if p is not None and self.phase == "running":
                try:
                    import cv2

                    imgs = {}
                    for name in p.cams:
                        frame = self.cameras[name].frame
                        if name in self.crops:
                            x, y, w, h = self.crops[name]
                            frame = frame[y:y + h, x:x + w]
                        imgs[name] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    p.record(measured, imgs)
                except Exception:  # noqa: BLE001  never stop the arm for the history
                    log.exception("history record failed")
            elif p is not None:
                p.clear()
            return measured

    pc.Controller = DPController
    pc.make_policy = make_policy
    sys.argv = [sys.argv[0], "--replan", "15", *rest]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    pc.main()


if __name__ == "__main__":
    main()
