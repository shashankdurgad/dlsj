# Wrist-camera brick detector: real behaviour, occlusion-aware sim model, robot-side target

The pick-phase residual observes the target brick's pose relative to the gripper. On the robot that comes from the
wrist camera, so the twin must be blind when and where the real detector is blind: hidden behind the TPU fins, cut
by the image edge, blurred, or simply a brick this detector does not like. Everything here is fitted to the team's
real detector run on recorded real wrist frames.

## The pipeline (robot and twin)

| step | robot | twin |
|---|---|---|
| (a) boxes | `OVBrickDetector("brick-detector-v1", "NPU")` on every wrist frame (plain OpenVINO, ~4 ms NPU / 10 ms CPU, 640 letterbox, NMS); boxes + conf, one class, no pose | `DetectorModel` (detector_model.py): the target's box statistically, from the geometric visibility (geometry.py) |
| (b) target | `wrist_target.TargetTracker`: rejects fin / plate / oversized boxes, tracks the target across frames | the tracker's measured errors as a Markov "wrong lock" mode (`wrong_target` in the json) |
| (c) pose | `WristCamera.box_to_rel`: box-centre ray -> table plane -> brick centre - fingertip in gripper_link | the same estimator (`Geometry.box_to_rel`, batched copy, parity 1e-16 m) |

`wrist_target.py` is numpy + stdlib only and is meant to be imported unchanged by the robot controller (it reads the
URDF, `robot/wrist_mount.json`, `align/joint_fit/result.json` and the fin CAD npz from the repo).

The detector is today NOT run during the pick on the robot (deploy/smolvla/loop_control.py runs it on the top and
inspect cameras at 4 Hz and pauses it while the arm moves: ultralytics starved the servo reads). Running (a) on the
NPU with plain OpenVINO in its own thread, gated by the pick prompt, is the plan this module is built for.

## Files

| file | what | run |
|---|---|---|
| real_detect.py | real detector on recorded wrist frames (cal2, 401 episodes, 2.5 s before -> 0.67 s after the close) -> out/perception/real_dets/ | plain python (PyAV) |
| geometry.py | FK, wrist camera, gripper occluders (per-pixel depth maps of the CAD fins / bases, or convex-hull ray tests), brick face samples, `visibility()`, `box_to_rel()`; numpy or torch | library |
| gt_real.py | per-episode brick pose fitted to the wrist view, per-frame visibility features + matched real boxes -> real_table.npz | plain python |
| fit_model.py | fits detector_model.json (detection probit per threshold, confidence, box noise, ray-plane height, wrong-lock) | plain python + torch cpu |
| detector_model.py | `DetectorModel` (numpy / torch), `GraspEnvDetector`, `ResidualEnvDetector` (TwinEnv) | library |
| wrist_target.py | robot-side target selection + pose (numpy) | library |
| eval_selector.py | the tracker on the real boxes -> selector.json | plain python |
| validate.py | sim detector vs real detector on the same recorded picks -> validate*.json / png | plain python |
| render_vis.py | twin wrist render with instance segmentation (visible / unoccluded target pixels) | `--prep` plain; render via run.sh |
| proxy_eval.py | occlusion proxy vs the render -> proxy_eval.json | plain python |
| held_brick.py | held brick pose in the gripper from the real wrist boxes | plain python |
| figures.py, bench.py | side-by-side real / twin figure; GPU cost | plain / run.sh |

Weights: `sim/twin/data/models/brick-detector-v1` (copied from the XPS `~/inference/models/brick-detector-v1`, YOLO26n,
trained 14 Nov on inspection / top crops; test P 0.91 R 0.90 mAP50 0.92).

## Real detector on the wrist (conf >= 0.5, 199 episodes whose brick pose fitted the wrist view)

| frames to close | detected | fin-hidden | out of frame |
|---|---|---|---|
| 45-75 | 0.66 | 0.05 | 0.06 |
| 20-45 | 0.82-0.85 | 0.02 | 0.02 |
| 10-20 | 0.68-0.78 | 0.03-0.05 | 0.03-0.05 |
| 6-10 | 0.66 | 0.07 | 0.07 |
| 3-6 | 0.51 | 0.14 | 0.08 |
| 0-3 | 0.36 | 0.26 | 0.09 |
| after close (0.67 s, held) | 0.67 | 0.16 | 0.02 |

(fin-hidden / out of frame: the occlusion proxy's fractions of the brick's projected area, with the jaw at its REAL
angle, see below.) The other 202 episodes (pose only known near the close) detect at 0.34-0.45 near the close: the per-brick difficulty
is large (probit intercept sd 1.27; brown bricks on wood, glare). By fin occlusion: 0.74 unoccluded, 0.57-0.64 at
1-20 % hidden, 0.59 at 20-40 %, 0.17 above 40 %. Mostly out of frame (> 50 %): 0.16. Motion blur: 0.72-0.77 below 10
px/frame, 0.51 at 15-20, 0.38 above 20. Misses come in runs: P(miss | miss) 0.86, P(det | det) 0.93. The confidence
saturates at ~0.78. False positives: the two TPU fins are boxed as "bricks" (0.5-0.7) in almost every frame, plus
plates and other bricks.

Pose from the box (ray -> plane 4 mm above the brick centre, brick assumed studs-up), brick-centre error:
6.6 mm median / 16.5 p90 at 20-40 frames before the close, 8.3 / 21 at 10-20, 12.3 / 30 at 5-9, 19.0 / 35 at 0-4
(bias about (-9, +9) mm in the gripper frame: the box covers the visible part and the image edge cuts it). After the
close the brick is lifted and the table-plane estimate is invalid.

Robot-side tracker on the real boxes (selector.json): target reported correctly in 54 % of frames (the detector had
it in 64 %); a wrong object in 9.6 % (runs of ~7 frames, 4-12 cm away, mostly other bricks); the fins' boxes are
rejected (IoU with the fin's image box).

Jaw angle seen by the real camera (jaw_fit.py, 496 frames, moving-fin silhouette): q_real = 0.02406 u - 0.1183, i.e.
~0.2 rad more closed than the twin's map of the time, q = 0.02308 u + 0.105, at every reading (physics now uses
bias -0.10, close to it) (at the hold it includes the TPU fin's
bend). The proxy uses q_real for the physical camera (geometry.JAW_REAL), so the jaw fin hides the brick where it
does on the robot regardless of the twin's own jaw angle. Held brick in the gripper (held_brick.py): centre
(-0.8, 2.8, -88.0) mm in gripper_link, bottom ~7.6 mm above the fixed-fin tip.

## Sim detector model (detector_model.json)

Per frame: s = x . beta + b_episode + e_t (probit; a box with conf >= threshold iff s > 0), e_t AR(1) rho 0.9,
x = [1, vis, occ_grip, oof, log area, log area^2, blur, blur^2, gripper < 12, dist, dist^2, gripper closing speed];
b_episode drawn from the fitted episodes' empirical distribution; conf | box from a linear model (sd 0.14, <= 0.79);
the box = geometric visible box with AR(1) noise whose bias / spread grow with occlusion; then `box_to_rel`;
then the tracker's wrong-lock Markov mode. Thresholds 0.5 (default, deployed), 0.3, 0.15. Latency 1 frame.

Validation (validate.py, same recorded picks, 30 seeds; target detection rate real / sim by frames to close):

| | 45-75 | 30-45 | 20-30 | 15-20 | 10-15 | 6-10 | 3-6 | 0-3 | -10-0 | -20 - -10 | all | miss run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| each episode's own difficulty | .66/.65 | .82/.79 | .85/.80 | .72/.74 | .57/.63 | .55/.59 | .46/.48 | .35/.36 | .57/.59 | .65/.63 | .643/.642 | 6.5/6.0 |
| held-out half, drawn difficulty | .67/.59 | .80/.72 | .84/.72 | .71/.69 | .55/.65 | .51/.61 | .42/.52 | .31/.41 | .55/.62 | .62/.65 | .624/.628 | 6.3/7.7 |

On held-out episodes (score model refitted on the other half, difficulty drawn) the sim detects ~0.10 MORE often than
the real detector in the last 15 frames before the close (it sees slightly more than the robot there) and 0.07-0.12
LESS often 20-75 frames out; with each episode's own difficulty the timeline agrees within 0.06 everywhere, so the
features (occlusion, edge, blur, distance, closing) carry the time dependence and the remaining gap is the per-brick
difficulty draw. Conservative variant for the residual: `det["0.5"]["b_shift"] = -0.3` (held-out: +0.03 near the
close, -0.14 far out, overall 0.565 vs real 0.624; validate_holdout_bshift-0.3.json). Pose error from the box, median
sim vs real per time bin: within -3 .. +2.4 mm before the close (validate.json); the sim's p90 is lower far out.

## Occlusion proxy (geometry.py) vs the twin's render

Default `occlusion="depth"`: per-pixel depth maps of the CAD fins / printed bases in the wrist camera (rigid in the
camera frame; the jaw parts per gripper reading, 1-unit bins; cached in out/perception/occluder_depth_<camera>.npz),
one lookup per brick sample (7 x 7 per face, projected-area weighted); other bricks as boxes (slab test); the image
edge. Against TwinEnv's instance segmentation (render_vis.py; robot visual and other bricks hidden for the unoccluded
mask; 600 states of 60 recorded picks + perturbed copies, 576 with the brick in frame; a second shard on jarvis
rendered black and was dropped), evaluated with the twin's own
camera / jaw / joint offsets: in-frame visible-fraction MAE 0.046 (0.030 on the 220 occluded states, bias +0.004),
corr 0.75, visible-box IoU median 0.91 (p10 0.80), "vis < 0.8" agreement 92 %, "vis < 0.5" 98 %. The proxy counts
slightly MORE fin occlusion than the render (0.110 vs 0.075 of the brick; it flags small occlusions more often), the
conservative direction. The printed bases never reach the image. Cost: ~10 ms per frame for 512 envs on the CPU.
The render jobs after the close keep the brick at its pre-grasp table pose (it interpenetrates the closed fins), so
they test geometry, not a realistic hold. TwinEnv note: its ResidualEnv keeps joe_cal2's joint offsets (not the
profile's W), so the twin renders FK(joe_cal2) of the recorded joints; proxy_eval.py uses the same.

Constants: the fingertip / grasp point (`geometry.TIP_LOCAL`, the fin-ray point (0.0063, 0.001, -0.0902)) and the
twin's gripper bias (`GRIP_BIAS`) come from `sim/twin/physics/apply.py`, with plain-number fallbacks; joint offsets
from `align/joint_fit/result.json` (W9). Refit / re-validated with them (15 Nov): detection timeline and pose errors
unchanged within 0.02 / 2 mm.

## Integration

GraspEnv (sim/twin/rl/grasp_env.py), after construction:

    from sim.twin.perception.detector_model import GraspEnvDetector
    det = GraspEnvDetector(env)                 # threshold=0.5, others=True (pool bricks occlude), samples=5
    env.detector_model = det                    # GraspEnv hook: f(env, rel) -> (det (n, 3), valid (n,))
    prev = env.tick_hook                        # 30 Hz camera + 1-frame latency (optional; without it: one frame per obs)
    env.tick_hook = (lambda e: (prev(e) if prev else None, det.tick(e)))

Per-env episode resets are detected from `env.t` going back. `det.last` has conf / box / vis / occ_grip / oof /
wrong for logging or an extended observation.

TwinEnv / ResidualEnv (replaces `ResidualEnv.detector()` on the instance, no edit of env.py):

    from sim.twin.perception.detector_model import ResidualEnvDetector
    env.residual.detector = ResidualEnvDetector(env.residual, dt_frames=1)

Robot:

    from dlsj.inspect.brick_detector import OVBrickDetector
    from sim.twin.perception.wrist_target import TargetTracker, WristCamera
    det, trk = OVBrickDetector(path, "NPU", confidence=0.3), TargetTracker(WristCamera(plane_dz=0.004))
    trk.reset()                                                    # each pick
    t = trk.update([(*d.box, d.confidence) for d in det.detect(bgr)], joints)   # t.valid, t.rel, t.conf
