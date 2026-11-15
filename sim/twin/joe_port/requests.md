# Requests from the joe_port survey

Ranked by expected effect on the replay lift gap, then on realism. Evidence paths are relative to
`sim/twin/joe_port/`. Background and the full comparison: `README.md`.

## 1. grasp_gate + scene: place the brick from the image, and re-fit the top camera per episode

**Ask.** Make the image-based brick pose (rung 0) the default for every replay number we quote, and add a
per-episode top-camera refinement before the brick is unprojected: keep joint offsets fixed, fit the top camera
pose on that episode's arm pixels (Joe used 34 frames, 25-200), then unproject the brick.

**Why.** In Joe's own sim the outcome flips on about 1 cm of brick placement, with nothing else changed:
* cal2 ep120, camera fixed from ep101 / 110: lift 3.8 cm, dropped; the jaw pushed the brick 12 mm before closing
  (`upstream/cal2-calibration/replay-ep120-heldout/result.json`).
* same episode after re-fitting the top camera on ep120 (camera moved 4.7 cm / 5.2 deg, brick moved 10.7 mm):
  lift 16.4 cm, held 10.2 cm (`.../replay-ep120-camera-adapted/result.json`,
  `reference-ep120-camera-adapted.json` -> `episode_camera_adaptation`).
* old ep8: brick at the FK grasp site 0.5 cm lift; brick unprojected from the image 16.2 cm
  (`upstream/calibration-evaluation/calibrated-demo-replay/summary.json`).
Our rung 2 (FK placement) lifts 38 % (27-51 %) of 60 held-out demos, with 28 % missed and 18 % pushed. The top camera, or the fit to it, is not stable to 1 cm between
episodes, so a single camera pose leaves about 1 cm of placement error, which is the size of the miss.

## 2. sensors + grasp_gate: model the image-to-state lag

**Ask.** Use a camera lag of about 1.3 frames (44 ms at 30 Hz) relative to joint states when a brick or arm pose
is read from a frame (rung 0, camera fits), and add it to the observation model the policies see.

**Why.** Joe's per-episode camera fit estimated `image_state_lag_frames: 1.33`
(`upstream/cal2-calibration/reference-ep120-camera-adapted.json`). At demo arm speeds one frame is several
millimetres at the gripper. It is one estimate from one episode, so check it on our data before fixing the value.

## 3. physics + gripper: do not port Joe's finger colliders, friction or gains

**Ask.** Keep the fin-ray model. Do not spend time on convex-decomposition stock jaws or on 200 / 10 drive gains
as a fix for the lift gap.

**Why.** His fingers are the stock rigid jaws, which the real robot does not have, with friction 1.0 / 0.8 on jaw
and brick and a box brick. Gate rung 2, same 60 held-out demos:

| Variant (rung 2: arm on the recorded states, brick at the FK grasp point, 60 held-out 2x2 demos) | Lifts | 95 % CI | Failures | Held reading sim / real |
|---|---|---|---|---|
| our fin-ray fingers (`r2_base`, run by the gate owner) | 38.3 % | 27-51 % | missed 28 %, pushed 18 %, tipped 10 %, slipped 5 % | 5.8 / 5.6 |
| fin-ray + Joe's friction (fin 1.0, brick 1.0 / 0.8) and effort 0.5 N m | 43.3 % | 32-56 % | missed 30 %, tipped 15 %, pushed 12 % | 5.5 / 5.6 |
| stock jaws, convex decomposition, our stock-jaw friction and effort | 8.3 % | 4-18 % | pushed 48 %, missed 38 % | 4.7 / 5.6 |
| stock jaws, convex decomposition + Joe's physics (jaw and brick friction 1.0 / 0.8, effort 0.5 N m, gripper drive 50 / 2, contact offset 1 mm) | 3.3 % | 1-11 % | missed 48 %, pushed 32 %, tipped 12 % | 4.3 / 5.6 |

Rung 2 writes the arm joints, so this does not test his arm gains; see `README.md` for the other limits.

His arm tracking with 200 / 10 gains is 0.6-2.0 deg mean per joint, the same as our servo model, so the drive
gains are not where his extra lifts came from. The earlier explanation in `sim/realism/joe_cal2_merge.md`
(colliders and gains) should be replaced by "brick placement, friction 1.0 and four episodes".

## 4. sensors / gripper (`fit_wrist.py`): re-fit the wrist camera pose with the measured focal

**Ask.** Re-fit the wrist mount pose with the focal fixed at the measured 342 px (`gripper/fit_wrist.py` already
loads the measured intrinsics; make its result the profile's wrist pose). Use Joe's pose only as the starting point.

**Why.** His pose was fitted to 5 hand-marked points (residuals 5-15 px) with focal 394.3 px
(`upstream/cal2-calibration/reference-cal2.json` -> `calibration_provenance.camera_fit.wrist`). `calib/profile.py`
renders that pose with 342 px. A 15 % focal error is absorbed as a camera pulled back along its axis and a
rotation of a few degrees. He made no better wrist pose after cal2; `fix_wrist*.py` are older and obsolete.
For reference, his axis is 14.3 deg from the URDF camera-mount axis and the position is 57 mm from the URDF mount.

## 5. grasp_gate + audit: add Joe's acceptance checks

**Ask.** Add to the gate metrics and the approval gate:
* release-drop: after the hold, open the gripper and require the brick to fall >= 3 cm (a held brick that does
  not fall is wedged or interpenetrating, not gripped);
* stall: the gripper joint stops inside a band that corresponds to the brick width, compared with the real held
  reading (5.6-6.0 units on cal2 for a 31.7 mm grasp);
* zero physics steps during a camera capture, and distractor bricks displaced < 2 mm.

**Why.** These are cheap and catch false lifts (`upstream/calibration-evaluation/replay_real_demo.py.txt`,
`upstream/synthetic/README.md`). The held reading is a direct check on the fin compliance: his rigid jaws read
5.4-5.6 against 5.6-6.0 real only because the 0.105 rad bias was tuned for it.

## 6. audit / rl: the approval gate needs a positive control

**Ask.** Before a policy's zero in the twin is read as "the policy fails", show that a policy known to work on
the real robot scores above zero in the twin. Report demo-replay success and policy success separately.

**Why.** Joe's sim passed its replay gate and then returned 0.0 cm lift for every policy rollout (pi0.5 with
three prompts and three control variants, ACT step 7000 in four scenes; `upstream/*/outputs/evaluation.json`,
`upstream/prompt-comparison-v2/RESULTS.json`, `upstream/smooth-control-comparison/comparison-summary.json`).
No real-robot outcome exists for those checkpoints, so his evals carry no sim-vs-real correlation.

## 7. visuals: keep foreground objects out of backdrop photos

**Ask.** Check that `visuals/textures/backdrop_inspect.jpg` has no bricks, plates or robot in it as seen from the
wrist and top cameras.

**Why.** Joe had to remove his photo backdrop for that reason (`upstream/cal2-calibration/clean-background.py.txt`:
"it contained foreground brick imagery visible from the wrist camera"). Ours is a median image, so this is
probably already fine. Nothing else in his appearance (flat colours, uncalibrated dome + distant light, a small
mirrored wood patch) is better than ours.

## Not requested

* His synthetic dataset (442 scripted episodes, round-1 calibration, stock jaws): not real-matched.
* His solver settings (60 Hz, 32 / 8 iterations, 1 mm contact offset): adequate for rigid jaws, no evidence they
  are better than our 120 Hz, 64 / 4, 2 mm.
