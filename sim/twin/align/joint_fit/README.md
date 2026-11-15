# Joint-offset check from camera views (SO-101 twin)

Question: do the joe_cal2 joint offsets (rad `[0, -0.053, -0.022, -0.073, -1.805, 0]`, i.e. lift -3.0 deg, elbow
-1.3 deg, wrist_flex -4.2 deg, wrist_roll -103.4 deg) map the recorded cal2 joint readings onto the real arm, now that
the cameras are measured? Machine-readable best estimate: **`result.json`** (offsets, spread, wrist mount, inspect pose,
grasp-depth candidates). Plain python in the twin image (numpy / scipy / cv2 / trimesh), no Isaac.

## Answers (short)

* **FK = sim.** The numpy FK reproduces Isaac's link points to 0.45 mm. It differs only where Isaac stops a joint at a
  URDF limit. **wrist_flex is above the sim's +95 deg limit in 8.5 % of all cal2 frames (max +3.5 deg) and
  shoulder_lift below -100 deg in 1.2 % (max 6.9 deg)**, so the twin cannot reach those recorded poses. Widen the
  USD limits by ~8 deg.
* **wrist_roll -103 deg is consistent only with joe_cal2's wrist mount.** Without finger geometry, the
  top and inspect silhouettes are flat in roll (+-8 deg changes the top loss by < 0.05 px). Roll is pinned only by the
  wrist camera's view of the table, i.e. roll + wrist mount together:
  * with the joe_cal2 mount refitted (C): -106.6 deg;
  * with the mount fixed to `robot/wrist_mount.json` (W): -96.3 deg. That mount differs from joe_cal2 by 19 deg, mostly
    in roll, and the joint absorbs most of it.

  With the measured mount, **-96 deg (+-3 deg) is the value to use**. Use roll and mount as a pair: W with
  wrist_mount.json, or C with its own mount. With joe_cal2 offsets, wrist_mount.json fits the table view much worse
  (22.7 vs 14.9 px). An earlier pass that matched the
  stock URDF jaws gave -89 deg, an artefact of the wrong finger geometry (the real fingers are TPU fin-ray).
* **shoulder_lift: joe_cal2's -3.0 deg is about 3 deg too negative, but lift is weakly identified.** Every camera fit
  moves lift up by +2.6 to +3.4 deg (total -0.4 to +0.4 deg) and elbow down by about -3.7 deg (total about -4.9 deg).
  Lift and elbow trade against each other and against the top camera's depth, so each alone is good to about
  +-2-4 deg. Pan moves -2.8 to -4.6 deg, which may partly be top-camera yaw.
* **Inspect residual: the camera moved; the joints don't explain it.** With the inspect camera fixed at its ChArUco
  pose, refitting the joints (B2) improves the held-out inspect loss only from 15.0 to 13.5 px. Freeing the camera
  pose alone (B0) brings it to 12.2 px, and joints + pose (B3) to 11.9 px. The fitted pose is 4.5 deg / 12 mm from the
  ChArUco one, and it is stable over the recording: fitted separately on the first, middle and last third of the
  episodes, it varies by < 1.2 deg / 5 mm. So the camera most likely moved between the recording and the ChArUco
  session (or the ChArUco chain through the top camera is off by that much).
* **Grasp depth (8-10 mm replay gap).** At the gripper-close frames the fingertip height changes by:

  | change | fingertip height |
  |---|---|
  | lift +1 deg | -3.9 mm |
  | elbow +1 deg | -2.8 mm |
  | wrist_flex +1 deg | -0.6 mm |
  | lift sag +1 deg/(N m) (static lift torque ~0.64 N m) | -2.5 mm |

  With the wrist mount fixed to `wrist_mount.json`:
  * camera-only fit W: 5.4 mm lower than joe_cal2 (held-out mean);
  * W9, with a 9 mm grasp-depth term: 8.3 mm lower, at no camera cost (held-out top 5.89 vs 5.90, inspect 13.8 vs 13.5,
    table 14.9 vs 14.9 px);
  * WS9, the sag model (lift / elbow / wflex offsets left at joe_cal2; sag lift 2.5, elbow 1.3 deg/(N m); pan / roll
    free): 7.4 mm lower, at the same camera cost.

  **So constant offsets (W9) and gravity sag (WS9) both reproduce a ~8 mm lower fingertip while keeping the
  silhouettes.** A camera-only fit with offsets and sag both free (WE) leaves sag at 0, so the silhouettes across poses
  with different lift torques give no evidence for torque-proportional link sag beyond constant offsets. Their
  sensitivity to a 2-3 deg lift change is ~1 px, so that is weak evidence either way. The dynamics agent finds that
  encoder-visible lift sag at the grasp is only 0.26 deg. **Recommendation for the grasp gate: W9 offsets with
  `wrist_mount.json`** (`result.json` -> `grasp_depth_candidates.W9`). WS9 is the sag alternative.

## Numbers

Held-out = 22 of 66 episodes (episodes[2::3]). Loss is in px at 320x240 (see Method); IoU is inside a band around the
robot. Offsets are totals in deg (sim q = deg2rad(state) + offset).

| fit | pan | lift | elbow | wrist_flex | wrist_roll | sag lift / elbow (deg/(N m)) | top loss / IoU | inspect loss / IoU | table loss | fingertip dz (mm) |
|---|---|---|---|---|---|---|---|---|---|---|
| joe_cal2 (current twin) (joe_cal2 mount) | 0.0 | -3.0 | -1.2 | -4.2 | -103.4 | - | 6.93 / 0.290 | 15.01 / 0.166 | 15.78 | - |
| B0 (joe_cal2 mount) | 0.0 | -3.0 | -1.2 | -4.2 | -103.4 | - | 6.93 / 0.290 | 12.22 / 0.199 | 15.78 | - |
| B1 (joe_cal2 mount) | -3.0 | -0.3 | -1.0 | -0.9 | -97.8 | - | 5.79 / 0.335 | 13.74 / 0.165 | 18.15 | - |
| B2 (joe_cal2 mount) | -1.2 | -0.3 | -1.3 | -0.9 | -97.3 | - | 6.14 / 0.318 | 13.45 / 0.181 | 18.92 | - |
| B3 (joe_cal2 mount) | -2.9 | -0.2 | -3.6 | 0.1 | -103.5 | - | 5.85 / 0.331 | 11.92 / 0.198 | 16.65 | - |
| C0 (joe_cal2 mount) | -2.8 | -0.4 | -5.0 | -0.8 | -106.8 | - | 5.97 / 0.326 | 12.41 / 0.184 | 15.24 | - |
| C (joe_cal2 mount refit) | -2.8 | -0.4 | -4.9 | -0.8 | -106.6 | - | 5.96 / 0.326 | 12.42 / 0.184 | 12.78 | - |
| joe_cal2 offsets + wrist_mount (wrist_mount.json) | 0.0 | -3.0 | -1.2 | -4.2 | -103.4 | - | 6.93 / 0.290 | 15.01 / 0.166 | 22.66 | 0.0 |
| C offsets + wrist_mount (wrist_mount.json) | -2.8 | -0.4 | -4.9 | -0.8 | -106.6 | - | 5.96 / 0.326 | 12.42 / 0.184 | 17.88 | -2.6 |
| W (wrist_mount.json) | -4.6 | 0.4 | -4.9 | -1.0 | -96.3 | - | 5.90 / 0.321 | 13.46 / 0.158 | 14.90 | -5.4 |
| W9 (wrist_mount.json) | -4.7 | 1.0 | -4.6 | -1.7 | -96.3 | - | 5.89 / 0.321 | 13.76 / 0.157 | 14.92 | -8.3 |
| WS9 (wrist_mount.json) | -5.2 | -3.0 | -1.2 | -4.2 | -96.3 | 2.5 / 1.3 | 6.00 / 0.315 | 13.67 / 0.158 | 14.39 | -7.4 |
| WE (wrist_mount.json) | -4.6 | 0.4 | -4.9 | -1.0 | -96.3 | -0.0 / -0.0 | 5.90 / 0.321 | 13.46 / 0.158 | 14.90 | -5.4 |

Fingertip dz = mean change (held-out close frames) of the env grasp point vs joe_cal2. Medians over all 456 episodes
(`tip_height.json`): joe_cal2 18.2 mm above the table, C 16.2, W 13.5, W9 10.5, WS9 11.3. Sag uses the corrected
torque (the camera mass is added to gripper_link): about 0.64 N m on lift and 0.37 N m on elbow at the grasp frames.

Fits (Powell; each fit's cameras weighted equally):

| fit | free | data |
|---|---|---|
| B0 | inspect pose | inspect (joints joe_cal2) |
| B1 | 5 offsets | top only |
| B2 | 5 offsets | top + inspect, inspect fixed at ChArUco |
| B3 | 5 offsets + inspect pose | top + inspect |
| C0 | B3 + table plane h | top + inspect + wrist table view, joe_cal2 wrist mount |
| C | C0 + wrist mount (6) | as C0 |
| W | 5 offsets + inspect + h | as C0, wrist mount fixed to `robot/wrist_mount.json` |
| W9 | as W | W + grasp-depth term (fingertip 9 mm below joe_cal2 at every close frame) |
| WS9 | sag (3) + pan / roll + inspect + h | as W9, lift / elbow / wflex offsets at joe_cal2 |
| WE | W + sag (3) | as W (cameras only) |

1-D profiles around C (`stage2.json`, loss vs joint delta -8 to +8 deg, all frames):

| joint | top | inspect | table view |
|---|---|---|---|
| pan | sharp (6.9 -> 7.2 at +-2 deg, 11 at +-8) | weak | sharp |
| lift | flat, keeps falling to +8 deg (6.90 -> 6.67) | min at 0 (+0.6 at +2) | min at 0 (+1.0 at +2) |
| elbow | flat, falls to +8 deg | min at 0 | min at 0 |
| wrist_flex | flat | min at 0 | min at 0 |
| wrist_roll | flat (+-0.05 over +-8 deg) | weak | min at 0 (+0.2-0.5 at +-2, +1.5 at +-8) |

Uncertainty: the bootstrap over episodes did not finish in the time available. Treat the spread across fit variants
and the profile widths as the uncertainty: pan +-1 deg; lift, elbow +-2-4 deg each (the sum is better constrained);
wrist_flex +-2 deg; wrist_roll +-3 deg for a given wrist mount.

## Overlays (held-out frames, red = joe_cal2, green = refit C)
* `out/align/joint_fit/overlay_top.jpg`, `overlay_inspect.jpg`: real arm mask tinted blue, red fingers / orange
  things don't-care (tinted yellow). The silhouettes leave out the URDF jaws.
* `overlay_table.jpg`: wrist frames with the top camera's blue objects cast through the table plane (red joe_cal2
  offsets + mount, green C). Held blue bricks are not on the table, so they cast wrongly; that is noise in this term.

## What depends on the finger model
* Excluded on purpose: the URDF fixed / moving jaws and the real red fingers (don't-care in both images). Offsets come
  from the arm links + gripper body only, so they do not depend on the finger model.
* Gripper visual mapping (scale / bias) and the jaw-based wrist-mount fit: dropped (they needed the fingers).
* Fingertip heights quoted here use the env's TIP_LOCAL point (gripper_link (8, 0, -90) mm). Per the gripper agent, the
  fin-ray tip is at the same height as the stock jaw tip, so the dz numbers carry over.

## Method
1. **Numpy FK** (`fk.py`, `verify_fk.py`): parses `assets/so101/urdf/so101_new_calib.urdf` (joint origins / axes /
   limits, inertials). It decimates the STL meshes per visual by vertex clustering (2 mm grid; 0.7 mm for close-up
   jaws). It also computes the static gravity torque per joint (`gravity_torque`, URDF masses + a 20 g wrist
   camera added to gripper_link; an earlier version replaced gripper_link's mass instead, fixed). Checked against `points.npz` (Isaac, 594 samples); log in `out/align/joint_fit/verify_fk.log`.
2. **Data** (`prep.py`): 66 episodes (every 7th labelled one).
   * Frames: the slowest frame (minimum joint speed) in each sixth of the episode, plus the inspection mark +-8 frames.
   * Top / inspect masks: the pixel differs from the episode median AND is dark. Red pixels are a separate
     don't-care mask (the fingers).
3. **Silhouettes** (`model.py`):
   * Triangle-rasterised arm links + gripper body (gripper_link geometry above z = -45 mm). Base, shoulder and fingers
     are don't-care.
   * Loss as `fit_cameras.py`: mean capped distance of the projected surface points to the real mask, plus the mean
     distance of real-mask pixels in a band around the sim to the sim silhouette.
   * Top camera: joe_cal2 pose + factory intrinsics, fixed. Inspect: ChArUco start, free.
4. **Wrist table view** (`table.py`):
   * Blue / lavender objects (blue plate, blue bricks, marker discs) are segmented in the wrist frame and in the top
     frame taken at the same instant.
   * Each set is cast through the table plane z = h into the other camera; the loss is a symmetric chamfer distance.
   * This ties the wrist camera's world pose (chain + mount) to the fixed top camera. It is the only roll-sensitive
     evidence.
5. **Grasp frames** (`tip_height.py`): per episode, the first frame after the widest opening at which the gripper
   reading reaches the held value, before the inspection mark (456 episodes).
6. `fit.py` (B0-C, scoreboard), `stage2.py` (profiles, inspect drift, bootstrap), `stage3.py` (offsets vs sag with the
   joe_cal2 mount, superseded by stage4), `stage4.py` (wrist mount fixed: W, W9, WS9, WE), `write_result.py` ->
   `result.json`, `overlays.py`, `limits_check.py`.
   Outputs: `sim/twin/out/align/joint_fit/` (`fit.json`, `stage2.json`, `stage4.json`, `tip_height.json`, `*.log`,
   `overlay_*.jpg`, `data.npz`; `stock_jaws_pass/` = the superseded first pass with the URDF jaws).
