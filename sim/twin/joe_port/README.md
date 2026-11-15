# joe_port — what Joe's Isaac Sim setup can teach our twin

Read-only survey of `/home/joe-spark-2/pi05-isaac-so101` on Joe's GB10 (15 Nov 2026; his tree last changed
14 Nov 14:40). Copied references are in `upstream/` (his files, see `upstream/README.md`). Requests for the other
twin agents are in `requests.md`. Gate comparison logs: `sim/twin/out/joe_port/`, results
`sim/twin/out/grasp_gate/joe_r2_*/result.json` (jarvis).

## Short answer

Joe's sim is not better matched to the real robot than ours; it is easier to grasp in.

* His gripper is the **stock rigid SO-101 jaw** (red CAD jaws, PhysX convex decomposition with default settings),
  not the fin-ray TPU fingers on the real robot. Jaw and brick friction are 1.0 / 0.8, the brick is a plain box.
* "Lifts more replays than ours" rests on **four distinct episodes** (old dataset 0 and 8, cal2 101 and 120). Two of
  them only lift after the brick is re-placed by about 1 cm (ep120: 3.8 cm -> 16.4 cm after a per-episode
  top-camera re-fit; ep8: fails with the brick at the FK grasp site, lifts with the brick unprojected from the
  image).
* His scripted-expert generator accepted 442 of 442 attempts, so that contact model rejects almost nothing.
* Every policy rollout in his sim (pi0.5 three prompts, pi0.5 GCP step 6115 with three control variants, ACT
  step 7000 in four scene variants) lifted the brick 0.0 cm. There is no real-robot result in the tree, so his
  data says nothing about sim-vs-real correlation.
* The cal2 reference we already merged (`sim/residual_env/assets/joe_cal2/reference-cal2.json`) is still his
  latest. He did not improve the wrist camera after it.

In our gate (rung 2, same 60 held-out cal2 demos and seed) his colliders and physics do not help: they lower the
lift rate from 38 % to 3-8 %, and his friction on our fins changes nothing measurable.

| Variant (rung 2: arm on the recorded states, brick at the FK grasp point, 60 held-out 2x2 demos) | Lifts | 95 % CI | Failures | Held reading sim / real |
|---|---|---|---|---|
| our fin-ray fingers (`r2_base`, run by the gate owner) | 38.3 % | 27-51 % | missed 28 %, pushed 18 %, tipped 10 %, slipped 5 % | 5.8 / 5.6 |
| fin-ray + Joe's friction (fin 1.0, brick 1.0 / 0.8) and effort 0.5 N m | 43.3 % | 32-56 % | missed 30 %, tipped 15 %, pushed 12 % | 5.5 / 5.6 |
| stock jaws, convex decomposition, our stock-jaw friction and effort | 8.3 % | 4-18 % | pushed 48 %, missed 38 % | 4.7 / 5.6 |
| stock jaws, convex decomposition + Joe's physics (jaw and brick friction 1.0 / 0.8, effort 0.5 N m, gripper drive 50 / 2, contact offset 1 mm) | 3.3 % | 1-11 % | missed 48 %, pushed 32 %, tipped 12 % | 4.3 / 5.6 |

Limits of this test: rung 2 writes the arm joints, so his 200 / 10 arm gains and 60 Hz step are not exercised;
our stock-jaw mode cooks 32 hulls / 64 vertices / 300k voxels where he used PhysX defaults; all 60 demos are 2x2
bricks (his ep101 was a 2x4); the brick is at the FK grasp point, not image-placed as in his replays. A fifth run
with his grasp point (`gripper.tip_local=[0.014,0,-0.098]`) gave the identical result, so that override does not
reach rung 2's placement and is not a separate data point. The fin-ray friction change is inside the noise.

What is worth taking is his **protocol**, not his physics: image-based brick placement, a per-episode top-camera
re-fit, the 1.3-frame image-to-state lag he measured, and his acceptance checks (release-drop, stall, zero physics
steps during capture).

## 1. How Joe matched sim to real

Two rounds, both visual estimates (his own README: "a visual approximation, not a measured digital twin").

1. **Round 1, old 212-episode dataset** (`assets/camera-fit.json`, `configure_reference.py`). Top camera (pinhole)
   and three pitch-joint offsets fitted jointly to one hand-marked wrist-housing landmark over 12 frames of
   episode 0 (6 px RMS). Wrist camera set by hand in six tries (`fix_wrist.py` ... `fix_wrist6.py`, 02:45-02:48):
   each picks a camera position (the URDF camera mount, or a guess) and solves the rotation from two rays (the
   target brick and the jaw tip) by SVD. Focal assumed 350 px. Gripper = action / 100 x URDF upper limit.
   A later "multi-pose" offset set (`calibration-evaluation/calibrated-replay-config.json`,
   [9.7, -4.7, -8.5, 6.6, -15.7] deg) made episode 0 replay.
2. **Round 2, cal2** (`cal2-calibration/reference-cal2.json`, 14:26-14:38). Multi-pose fit on episodes 101 and 110,
   episode 120 held out: top camera + joint offsets + housing landmark, from hand-marked housing and jaw pixels
   (fit median 5.8 px housing / 20.7 px jaws; held-out 9.0 / 32.4 px, p90 59 px). Inspect camera from 7 points
   (1-19 px), wrist camera from 5 points (5-15 px, focal 394 px). The fitting code is not in his tree; only the
   result is. Gripper: q = 0.02308 rad/unit x action + 0.105 rad, the scale from the follower's encoder span and
   the bias chosen so the CAD jaws touch a 32 mm brick at the recorded held state (~6 units).
   Target brick: centre and long axis marked in the first top frame, unprojected onto the table with the fitted
   camera ("not placed at FK grasp").

Replay protocol (`cal2-calibration/replay.py`, `calibration-evaluation/replay_real_demo.py`): reset to the first
recorded state, send each recorded **action** as the joint position target, two physics steps (60 Hz) per 30 Hz
frame, no attachment, 180 frames. Then open the gripper for 90 steps and measure how far the brick drops.

| Replay | Brick placement | Max lift | Held last 1 s | Release drop | Note |
|---|---|---|---|---|---|
| old ep0, round-1 offsets | image | 0.0 cm | no | - | grasp point 6 cm off in x |
| old ep0, multi-pose offsets | image | 16.7 cm | 11.6 cm | 11.1 cm | fit episode |
| old ep8 held-out | image (unprojected) | 16.2 cm | 9.9 cm | 8.7 cm | |
| old ep8 held-out | FK grasp site | 0.5 cm | no | 0 | same episode; the two placements are 2.5 cm apart |
| cal2 ep101 | image | 15.6 cm | 8.8 cm | 7.9 cm | fit episode; grasp point within 2.2 mm of the brick |
| cal2 ep120 held-out | image, fixed camera | 3.8 cm | dropped | 0 | brick pushed 12 mm before the jaw closed |
| cal2 ep120 "camera-adapted" | image, top camera re-fitted on ep120 | 16.4 cm | 10.2 cm | 9.6 cm | camera moved 4.7 cm / 5.2 deg, brick moved 10.7 mm |

Arm tracking in his replays (sim joint vs commanded): 0.6-2.0 deg mean per joint. Sim state vs the recorded real
state: 0.6-2.1 deg. That is the same range as our servo model (shoulder lift 1.55 deg).
Held gripper reading: his sim 5.4-5.6 units vs real 5.6-6.0. It matches because the 0.105 rad bias was chosen
for exactly that with rigid CAD jaws.

## 2. Robot and gripper model, side by side

| Item | Joe | Our twin |
|---|---|---|
| Stack | Isaac Sim 6.2, one persistent Kit session driven through a localhost Python server; one env; PhysX on CPU | Isaac Lab 3.0 on Isaac Sim 6.2 in Docker; vectorised envs; GPU PhysX |
| Robot asset | stock `so101_new_calib.urdf` through `URDFImporter` (`merge_fixed_joints`, `collision_from_visuals`, force drive, position target) | same URDF converted to USD, plus our fin-ray fingers |
| Finger geometry | **stock rigid CAD jaws** (fixed finger + moving jaw) | **XLeRobot fin-ray TPU fins** from the open CAD (`gripper/`) |
| Finger colliders | `convexDecomposition` on the two finger meshes, PhysX default parameters (no hull / voxel settings authored) | per-fin chain of convex pieces; stock modes `pads_cad`, `convex_decomposition` (32 hulls, 64 vertices, 300k voxels), `sdf` kept for comparison |
| Finger compliance | none | compliant contact 10 000 N/m, damping 60 on the fin material |
| Finger / brick friction | 1.0 / 0.8 on all `gripper_link` shapes and the target brick (default combine = average) | fin 0.9 / 0.8 (combine max), brick 0.45 / 0.38, stock jaw 0.5 / 0.4 |
| Contact / rest offset | 1 mm / 0 on every collider | 2 mm / 0 |
| Solver iterations | 32 position / 8 velocity, set per rigid body | 64 / 4 on robot and bricks |
| Physics step | 1/60 s, 2 steps per 30 Hz action | 1/120 s, 4 steps per action |
| Arm drive | implicit PD 200 / 10, max effort 10 N m; the raw action is the target (no servo model) | servo model (reference filter + PD, steady-state bias) over drive 60 / 2, effort 2.9 N m |
| Gripper drive | 50 / 2, max effort 0.5 N m (first trials 40 / 1 / 0.3) | effort 0.4 N m with fins, 1.2 N m stock jaw |
| Gripper mapping | q = 0.02308 x units + 0.105 rad | same scale; 0.105 rad for the stock jaw, fin-ray bias set by the gripper model |
| Gripper limits | URDF: -0.1745 ... 1.7453 rad | same |
| Joint offsets (rad) | [0, -0.0530, -0.0218, -0.0725, -1.8049, 0] | same (joe_cal2); `align/joint_fit` in progress |
| Grasp point (gripper_link) | (0.014, 0, -0.098) | `tip_local` (0.008, 0, -0.090) stock; fins: 15 mm above the fin tip (fin tip z = -0.105) |
| Table | cube, top at z = -0.002; friction default | same height; 0.5 / 0.42 |
| Brick | box 64 x 32 x 19.2 mm (2x4) or 32 x 32 x 19.2 mm, 12 g, box collider; studs are visual only | measured Duplo shapes, convex hull, 6.75 g (2x2) / 12 g (2x4) |
| Plates | 78 mm floor cylinder + 24 rim boxes (static) | measured profile as a mesh collider |

Lowest grasp-point height in his cal2 replays: 8.2 mm (ep101) and 11.4 mm (ep120) above the table, with his
grasp point at z = -0.098 in `gripper_link`. Our fin tip is 7 mm further out (z = -0.105), so the same joint
trajectory puts our fin tips about 1-4 mm above the table at the grasp. That is consistent with the gate's
"fin tip -0.5 mm above the table" in successful replays, and gives no sign that his offsets hold the arm lower
than ours (they are the same offsets).

## 3. Wrist camera

* Final pose (cal2): position (0.0030, 0.0601, -0.0491) m in `gripper_link`, optical axis (0.081, -0.621, -0.779),
  focal 394.3 px. It is 57 mm from the URDF camera-mount position (0.0025, 0.0734, 0.0066) and its axis is 14.3 deg
  from the URDF mount's axis. Fitted to 5 hand-marked points, residuals 5-15 px.
* `fix_wrist*.py` are round-1 tries from before cal2 (focal 350). The cal2 pose is 133 mm / 171 deg away from the
  round-1 pose, so round 1 is obsolete.
* Nothing newer than the file we already use. `cal2-calibration/replay-ep101-corrected-cameras/` re-ran ep101 with
  the same reference (identical `result.json`).
* One inconsistency on our side: `calib/profile.py` takes his pose, fitted under focal 394 px, and renders it with
  the measured focal 342 px. A pose fitted with a 15 % longer focal sits further back and is rotated by a few
  degrees for off-centre landmarks (about 3 deg at 200 px from the centre). It does not explain all of our
  10-14 deg error, but the pose should be re-fitted with the focal fixed at the measured value.

## 4. Scene and appearance

* `scene-reference.usda` is the stage after `configure_reference.py` (props, materials, cameras).
  `synthetic-scene.usda` is the same stage after `synthetic/setup_physics.py` (friction material, offsets, solver
  counts, convex decomposition, gains). Both reference asset paths on his machine.
* Table: UsdPreviewSurface with a small wood patch cropped from a real frame (`wood-patch.png`), mirrored 3 x 3
  with a colour scale. Ours (top-camera median texture) is closer to the real table.
* Plates: lathed ring mesh, flat colours (blue, and yellow-green in cal2). Bricks: box + stud rings, flat colours.
  Robot: flat black, red fingers. No real fin-ray appearance.
* Background: a textured quad 1.4 m behind the scene in the inspect camera's frustum. In round 1 it carried a crop
  of a real inspect frame. `clean-background.py` replaced it in cal2 with a grey quad plus a dark "monitor" quad
  at 0.6 m, because the photo contained bricks and the robot, which then showed up in the wrist view.
* Lighting: dome light 700 (1500 at setup) and one distant light 600 at rotateXYZ (25, -30, 20). Not calibrated.
* Rendering: Replicator RGB annotators, `rt_subframes=1`, capture with `delta_time=0` and a check that the
  capture added zero physics steps.
* `synthetic/` (3.5 GB): 442 scripted-expert episodes (342 train / 100 validation of a 400 / 100 target, paused),
  top / wrist / inspect at 640 x 480, 30 fps, LeRobot v3. Privileged IK controller on a mint 2x2 brick in a
  1.6 x 2 cm region, yaw 48-68 deg, with the round-1 calibration. Not real-matched data; of no use to the twin.

## 5. Policy-in-sim evaluations

| Eval | Policy | Scene | Result |
|---|---|---|---|
| reference rollout | pi0.5 `w3joe/pi05-junction-so101-20261114` | round 1 | 500 actions, brick did not move |
| `prompt-comparison-v2` | pi0.5, 3 prompts | round 1 | 0.0 cm lift in all three |
| `smooth-control-comparison`, `eval-latest-gcp-6115` | pi0.5 GCP step 6115; execute 50, re-plan 10, re-plan 10 + rate limit | round 1, multi-pose offsets | 0.0 cm lift in all three |
| `act-step7000`, `-single-brick` | ACT step 7000 (cal2 pickup) | round 1 | 0.0 cm |
| `cal2-act-step7000`, `cal2-act-clean-stage` | ACT step 7000 | cal2, photo and clean backdrop | 0.0 cm (max 0.08 mm) |
| `cal2-act-real-frame-check` | ACT step 7000 on real frames | offline | inference check only |

No hardware result is recorded for any of these checkpoints (`act-step7000/model/TESTING.md`: "No hardware rollout
has been performed"). So sim success did not correlate with anything: the sim returned zero for every policy while
its replay gate passed. For our approval gate this means a passing demo replay is necessary but not sufficient,
and the gate needs a policy that is known to work on the real robot as a positive control before a zero is read
as a verdict on a policy.

## What we did with it

* `upstream/`: his scripts, reference JSONs, replay results, arrays and a few renders.
* `requests.md`: changes for the gate, sensors, gripper, visuals and audit owners.
* Gate comparison above. No twin code was changed.
