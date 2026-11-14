# Live spin-and-inspect page (YOLO only)

`dlsj.inspect.sweep_live` runs the released YOLO26n brick classifier
([`brick-yolo-classifier-v1`](../models/brick_yolo_classifier/README.md)) on
the `inspect` camera. It spins the held brick on `wrist_roll` and picks one bowl
from the whole sweep. Anomalib is not used. Everything streams to a browser.

## Launch (XPS)

```sh
# once: release -> ~/inference/models/brick-yolo-classifier-v1 (verify SHA256SUMS + files.json hashes)
cd ~/inference/dlsj-sweeplive          # rsync of the repo (dlsj/, calibration/)
free -g                                # need >= 10 GB available
fuser /dev/ttyACM* /dev/video*         # nothing else may hold the cameras / arm
setsid nohup ~/desmond/venvs/teleop/bin/python -m dlsj.inspect.sweep_live \
    --model-dir ~/inference/models/brick-yolo-classifier-v1 --device NPU --port 8012 \
    >> ~/inference/logs/sweep_live.log 2>&1 < /dev/null &
```

Open <http://100.77.191.42:8012> over Tailscale. The defaults are:

- `--realsense inspect=243722062690,top=346422061366`
- `--cameras wrist=/dev/v4l/by-id/usb-Sonix_…-video-index0`
- `--host 0.0.0.0`

Optional flags:

- `--roi x0,y0,x1,y1` sets a fixed crop (default: a centred square); it can also be changed on the page.
- `--no-arm` gives camera-only mode with no Connect button.
- `--arm auto` connects at start.
- `--detector-weights` enables automatic brick crops once a detector model is released.

Desmond's teleop venv already has openvino 2026.4.1, pyrealsense2 and LeRobot 0.7.

Stop the page with `pkill -f 'python -m dlsj.inspect.sweep_live'`. Run it from a
script or plain shell, because `pkill -f` inside an `ssh '…'` one-liner also matches
and kills that shell. Stopping it this way keeps torque on.

## The page

- **Cameras:** the live `inspect` view shows the crop box (coloured by the latest
  confident class) and the latest per-frame label, confidence, top-2 margin,
  probabilities, latency and device. Small `wrist` and `top` views sit beside it.
- **Classify live:** per-frame YOLO on the inspect stream, with no motion. It shows the
  model crop, the probabilities and the release's 3-frame mean rule.
- **Spin & inspect:** the sweep (`dlsj.loop.sweep.Sweeper`) works like this:
  - it makes N stops (default 12) over the range (default 180°) around the current
    roll, clipped to the calibrated wrist_roll limits;
  - at each stop it settles, then grabs a fresh frame;
  - each frame is classified on a background worker while the wrist moves on;
  - afterwards the wrist goes back to its start angle.
- **Live results:**
  - a thumbnail strip per angle (label and confidence);
  - a chart of P(plain/circle/cross) against angle, with confident points filled;
  - the final decision and bowl, with the reason and the release-rule comparison;
  - every strategy's outcome.
- **Settings:** frames, range, settle time, speed cap (hard limit 120°/s), device
  (NPU/GPU/CPU, switched live), strategy, crop ROI, grasp check and dry run.
- **Dry run** (on by default): the same sweep runs against a simulated wrist.
  Frames still come from the real camera, but no command reaches the arm.
- **Stop:** aborts the sweep. The wrist then returns to its start angle, and the
  return itself cannot be aborted.
- **Go to inspect pose:** a 20°/s ramp to `calibration/robots/so_follower/follower.inspect.json`,
  keeping the gripper command. No such pose is saved yet; save one with
  `python -m dlsj.loop.run --save-pose inspect`.
- **Connect arm:** connects the follower with the repo calibration. It refuses if
  `fuser` shows another process on a serial port. It sets goal to present, turns
  torque on and holds; nothing moves.
- **History:** each sweep is saved to `outputs/sweeps/<time>_{live,dry}/`. A folder holds:
  - `frame_XX.png` and the model crops `crop_XX.jpg`;
  - `sweep.json`, with angles, timing, per-frame probabilities, every strategy and the decision;
  - `contact.jpg` and `scores.png`.

## Decision rule

A frame is **confident** when its top probability is ≥ 0.8 and its top-2 margin is
≥ 0.2. This is the release's own gate, applied per frame.

The `yolo_topk` strategy is the default (`dlsj/inspect/aggregate.py`):

| Result | Condition |
|---|---|
| **marked** | ≥ 2 frames confidently circle or cross |
| **plain** | ≤ 1 confidently-marked frame (a glare outlier) and ≥ 80% of frames confidently plain |
| **uncertain** | anything else |

For a marked brick, circle vs cross is a confidence-weighted vote over the
confidently-marked frames: each frame votes its class with weight equal to its
confidence. The winner's share minus the loser's must be ≥ 0.4 (a 70/30 split).
Otherwise the result is **unknown**.

Bowls:

- plain → **yellow**
- circle → **orange**
- cross → **purple**
- uncertain or unknown → **re-inspect, then purple**

The release's naive rule (`yolo_mean`) is shown next to it for comparison. It
averages all frames' probability vectors and applies the same 0.8/0.2 gate. On a
one-face mark the plain angles dominate the mean, so it says plain or uncertain
where `yolo_topk` says marked.

## Measured on the XPS (13–14 Nov)

Per-frame latency, model call only, median/p90 over 300 runs (`python -m dlsj.inspect.yolo_inspector bench`):

| Device | Median | p90 |
|---|---|---|
| NPU | 1.10 ms | 1.14 ms |
| GPU (FP32) | 0.84 ms | 0.95 ms |
| CPU (FP32) | 9.1 ms | 10.6 ms |

Notes on these numbers:

- A π0.5 kernel benchmark was running at the same time, so the CPU figure is inflated.
- Crop plus JPEG round trip adds about 5 ms on the CPU.
- On the live page with real frames, NPU measured about 2.3 ms for the model and
  about 7 ms end to end.
- The NPU rejects Tan's FP32 hint ("Supported values: f16, i8"). The loader then
  recompiles at the NPU's default precision, which differed by about 0.003 from
  CPU FP32 on a test frame.

## Safety

- Nothing moves on page load or on Connect.
- The arm moves only on Spin & inspect with Dry run off (after a confirm), or on Go
  to inspect pose.
- The sweep starts from the servos' `Goal_Position`, not the measured pose, so a
  gripper squeezing a brick keeps squeezing.
- Only `wrist_roll` moves, capped per tick at speed/fps.
- Grasp loss ends the sweep.
- The wrist is always sent back to its start angle.
- Exit keeps torque on (`disable_torque_on_disconnect = False`).
- Keep the process off the XPS while π0.5 jobs hold GPU memory (`free -g` ≥ 10 GB),
  and while anyone else uses the cameras or arms.

## Caveats

- The `inspect` view is wide. The classifier expects a whole-brick crop, so set the
  ROI tightly around where the arm presents the brick before trusting predictions.
- With no brick in view, the background is classified plain at P ≈ 0.83–0.95.
  That is not a meaningful result.
- No sweep with a real brick or real arm motion has been run yet.
- The 2-frame and 80% thresholds are reasoned defaults, not calibrated values.
  Check them with `python -m dlsj.loop.sweep_test eval` on saved, labelled sweeps.
