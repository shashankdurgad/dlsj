"""Isaac-side check of the camera pipeline in TwinEnv's image path (hook = twin_scene.color_correct).

    NAME=dlsj-twin-sensor sim/twin/run.sh sim/twin/sensors/check_isaac.py [--ticks 30] [--dr-scale 1.0]

Reports the rendered intrinsics per camera (fx, fy, cx, cy of the sensor). Rolls the twin for a
few ticks with a hold action, nominal and with the image DR only (TwinRandomizer with every other section disabled),
and reports the per-frame cost of the pipeline per camera (raw render vs color_correct),
that PyAV never got imported into the Isaac process, and that the codec worker ran (not the emulation). Writes
out/sensors/isaac_<mode>_<cam>.png (raw | pipeline) and out/sensors/isaac_check.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
ap = argparse.ArgumentParser()
ap.add_argument("--ticks", type=int, default=30)
ap.add_argument("--dr-scale", type=float, default=1.0)
args = ap.parse_args()

from sim.twin.env import TwinEnv, TwinEnvCfg  # noqa: E402

OUT = REPO / "sim/twin/out/sensors"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    """Nominal (raw baseline, then the pipeline) and image-DR rollouts; step time per mode."""
    import cv2
    import numpy as np

    from sim.twin.env.randomize import TwinRandomizer
    from sim.twin.sensors import pipeline

    dr = TwinRandomizer({k: {"enabled": False} for k in ("camera", "light", "material", "clutter")} |
                        {"image": {"scale": args.dr_scale}})
    env = TwinEnv(TwinEnvCfg(randomizers=[dr], physics_dr=False))
    sensor = {"top": "top_cam", "wrist": "wrist_cam", "inspect": "inspect_cam"}
    rep = {"av_in_isaac_before": "av" in sys.modules, "intrinsics": {}}
    for cam, key in sensor.items():
        if key in env.scene.sensors:
            K = env.scene.sensors[key].data.intrinsic_matrices[0].detach().cpu().numpy()
            rep["intrinsics"][cam] = [round(float(K[i, j]), 2) for i, j in ((0, 0), (1, 1), (0, 2), (1, 2))]
    for mode, rand, cc in (("raw", False, False), ("nominal", False, True), ("dr", True, True)):
        env.cfg.color_correct = cc
        obs = env.reset(seed=3, randomize=rand)
        hold = obs["state"].copy()
        ts = []
        for _ in range(args.ticks):
            t0 = time.perf_counter()
            obs = env.step(hold)[0]
            ts.append(time.perf_counter() - t0)
        rep[mode] = {"step_ms_median": round(1000 * float(np.median(ts[3:])), 1),
                     "step_ms_p90": round(1000 * float(np.percentile(ts[3:], 90)), 1)}
        if rand:
            rep[mode]["draw"] = env.dr_draw.get("image")
        for cam in env.cfg.cameras:
            raw = env.scene.sensors[sensor[cam]].data.output["rgb"][0, ..., :3].detach().cpu().numpy()
            cv2.imwrite(str(OUT / f"isaac_{mode}_{cam}.png"),
                        np.concatenate([raw.astype(np.uint8), obs["images"][cam]], 1)[..., ::-1])
    rep["pipeline_ms_per_step"] = round(rep["nominal"]["step_ms_median"] - rep["raw"]["step_ms_median"], 1)
    c = pipeline._codec
    rep["codec_worker"] = None if c is None else {"failed": c.failed, "running": c.proc is not None}
    rep["av_in_isaac_after"] = "av" in sys.modules
    (OUT / "isaac_check.json").write_text(json.dumps(rep, indent=1))
    print("ISAAC_CHECK", json.dumps(rep), flush=True)
    env.close()


main()
