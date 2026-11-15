# Twin gripper: the real TPU fin-ray fingers

The station's gripper is not the stock SO-101 jaw. It is the XLeRobot soft gripper: two printed bases and two red TPU
fin-ray fins. This folder models it for the twin (colliders, compliance, visuals, fingertip point). It is switched on
through the physics layer (`sim/twin/physics/params.json`, `gripper.collider = "finray"`, the default), so TwinEnv and
the residual env both use it.

```
python sim/twin/gripper/build_assets.py            # CAD -> assets/finray_so101.{npz,json} (plain python)
python sim/twin/gripper/evidence.py                # real wrist / top frames over the gripper reading
python sim/twin/gripper/fit_wrist.py frames|fit    # fin silhouettes vs real wrist frames (camera pose, joint angle)
python sim/twin/gripper/fin_shape.py frames|fit --hinges N|score ...   # real fin bending: hinged-chain fit / sim score
sim/twin/run.sh sim/twin/gripper/render_compare.py [--wrist fitted]     # twin wrist render vs real, contact sheet
sim/twin/run.sh sim/twin/gripper/bench.py --tag x [--efforts 0.4 0.7] [--set key=json]   # stall / sink / buzz bench
sim/twin/run.sh sim/twin/gripper/jitter.py --ep 18 --tag x --video 1    # brick jitter in a replayed grasp
python sim/twin/gripper/jitter.py --encode x                             # hold video from the saved frames
sim/twin/run.sh sim/twin/gripper/validate_gripper.py [--episodes 60] [--configs a,b]     # realism gate table
```
"plain python" = the twin image's venv without Isaac (see the top-level twin README). Outputs: `sim/twin/out/gripper/`.

## The finger design (evidence)

* **Design:** `hardware/SO101_soft_fin.stl` of [Vector-Wangel/XLeRobot](https://github.com/Vector-Wangel/XLeRobot)
  (Apache-2.0), the "Compliant Gripper (new)" linked from
  [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) (TPU 95A finger, PLA base, two extra M3
  screws). The file is in `assets/` with its licence and `SOURCE.md`. One print plate holds three bodies: the
  fixed-finger base (replaces the stock wrist-roll follower), the moving-jaw base (servo-horn hub), and one fin.
* **Why this one:** `data/photos/table_topdown.jpg` shows two different black bases, each with a clevis and two
  screws, and a red wedge fin with a flat inner face, a curved back and about 20 slanted ribs. Scaled by the M3 pair
  (10 mm), the photo gives 62.4 mm from the screw line to the tip (CAD 63.5) and 21.3 mm root thickness (CAD 21.7).
  The user's "6 cm from the mount to the tip" agrees.
* **Fin:** 67.2 mm long, 21.7 mm thick at the root, 24 mm wide at the root and 11 mm at the tip, flat contact face.
* **Kinematics:** the stock rotary jaw. One fin is bolted to `gripper_link`, the other turns with
  `moving_jaw_so101_v1_link` about the gripper joint. It is not a parallel gripper.
* **Placement (derived, `build_assets.py`):** each base is registered onto the stock URDF mesh it replaces (same CAD
  axes; translation from the shared servo body / horn hub), and the fin sits where its two M3 holes meet the clevis
  holes. Fixed fin tip in `gripper_link`: (-8.2, 1.0, -105.2) mm. The stock fingertip is at z = -104.4 mm, so the fins
  do not move the fingertip along the gripper. The contact faces toe in (fixed fin 5 deg).
* **Rigid geometry** (`assets/finray_so101.json`): a 31.7 mm brick fits at q = 0.203 rad at the tips, 0.217 rad 15 mm
  up, 0.240 rad 30 mm up. The fin tips meet at q = -0.178 (the URDF's closed limit is -0.175).
* **Closed and empty, the tips touch** (user; `out/gripper/closed_empty_evidence.jpg`): the real reading is 2.66-2.86
  at the start of episodes 126, 361, 285, 137, 207, with the fin tips touching in the wrist frames. A bias of -0.242
  makes the rigid tips meet there (used for the first bending runs below), but the open wrist frames fit a linear map
  q = 0.0229 u - 0.099 (see "Gripper mapping across the range"), so the squeezed-shut real fins touch by bending.
  The default is bias -0.10. joe_cal2's 0.105 is about 0.2 rad too open (`out/gripper/fin_diag.jpg`).
* **The fins bend on every pick** (user, and the arithmetic): with the tips touching at 2.8, the real hold of 5.72 is
  q = -0.11, about 0.33 rad of jaw rotation past where a rigid fin would meet a 31.7 mm brick, about 12 mm of give
  per fin at the contact. Rigid fins with compliant contact cannot give that.

## Model

* **Colliders** (`model.add_colliders`): the stock jaw hulls are deactivated. Each base is one convex hull (PLA
  material); each fin is 4 convex pieces (`TwinFin0..3`).
* **Bending fins** (`gripper.finray.bend`, `model.segments` / `add_hinges`): each fin is a chain of rigid segments,
  the root piece fixed to its link and each later segment a light rigid body (3 g) on a revolute spring-damper joint
  (axis = the gripper joint axis, so the fin bows in the closing plane; +-45 deg limit). `slices: true` (default, 4
  hinges, k 2.5 N m/rad) cuts the fin mesh into a 12 mm clamped root and N equal slices (1-5 hinges); `slices: false`
  cuts at the CAD convex pieces (1-3 hinges). The hinges are articulation joints `fin_<fixed|jaw>_<a..e>`,
  driven by an implicit actuator group `fins` (target 0, stiffness `stiffness_nm_per_rad`, optional
  `stiffness_jaw_nm_per_rad` for the moving fin, damping 0.01 N m s/rad). Code that indexes joints by name (`jidx`)
  is unaffected; `default_joint_pos` has the extra zeros. The visual fin is cut at the same hinges and rides on the
  segment bodies. `curl_gearing` couples the distal hinges to the root one (PhysX mimic joints, fin-ray curl); see
  Results: it is not usable.
* **Compliant contact** on the fin material as well (10 kN/m per contact point, 60 N s/m): it removed the 60 Hz
  hard-contact limit cycle (see Jitter).
* **Gripper effort:** 0.74 N m (`gripper.finray.effort_nm`). The station runs the gripper STS3215 on 12 V (stall
  ~30 kg cm = 2.94 N m) with LeRobot 0.7's `so_follower` gripper settings: `Max_Torque_Limit` 500 (50 %, 1.47 N m),
  `Protection_Current` 250, `Overload_Torque` 25 (25 % = 0.74 N m once the overload protection trips, which a
  sustained hold does). 0.74 and 1.47 give the same lift rate.
* **Mapping bias:** -0.10 (`gripper.finray.bias_rad`), the linear map of the open wrist frames; see above.
* **Friction:** TPU on ABS 0.9 static / 0.8 dynamic (no measured value; printed TPU is quoted at 0.6-1.0). The fin
  material combines `average`, the bricks `max`, the table `min` at 0.15 / 0.12, which gives fin-brick 0.9 / 0.8,
  fin-table 0.15 (the TPU tip slides when the jaw sweeps over the varnished table instead of stalling the jaw; grasp
  gate: the stalls were the moving fin dragging on the table) and brick-table 0.45 / 0.38 (was 0.475 / 0.40).
* **Fingertip point:** `apply.patch_profile` sets the residual env's `TIP_LOCAL` to the centre of a 31.7 mm brick
  held against the fixed fin 15 mm above its tip: (0.0063, 0.001, -0.0902). `tip_w()` / `tip_local` follow.
* **Visuals** (`model.replace_visuals`, called from `visuals/paint.py`): the stock jaw visuals of `RobotVisual` are
  deactivated and replaced by the two bases (body material) and the two fins (the measured red). The ribs are in the
  mesh.

### Switches (`params.json`, `--set key=json`, or `TWIN_PHYSICS_SET="key=json;key=json"` with `docker run -e`)

| key | meaning |
|---|---|
| `gripper.collider` | `finray` (on) or `pads_cad` (stock jaw pads, as before) |
| `gripper.finray.material.compliant_contact_stiffness` / `_damping` | compliance; 0 = rigid contact |
| `gripper.finray.material.static_friction` / `dynamic_friction` | fin friction |
| `gripper.finray.effort_nm` | gripper servo effort limit |
| `gripper.finray.bias_rad` | gripper mapping bias (null = `gripper.bias_rad`, joe_cal2's 0.105) |
| `gripper.finray.grasp_depth` / `tip_local` | fingertip point: height above the fin tip, or an explicit [x, y, z] |
| `gripper.finray.extend` | experiment: fins mounted this far (m) further out along their axis |
| `gripper.finray.bend.hinges` | hinges per fin; 0 = rigid fins |
| `gripper.finray.bend.slices` / `root_len` | false: cut at the CAD pieces (1-3 hinges); true: clamped root + N equal slices (1-5) |
| `gripper.finray.bend.stiffness_nm_per_rad` / `stiffness_jaw_nm_per_rad` | hinge stiffness, scalar or list root -> tip; the second (optional) for the moving fin |
| `gripper.finray.bend.damping_nms_per_rad` / `segment_mass_kg` / `limit_deg` | hinge damping, segment mass, joint limit |
| `gripper.finray.bend.curl_gearing` | mimic gearing of the distal half of the hinges to the root hinge (0 = off) |
| `gripper.finray.bend.rest_curl_deg` | unloaded hinge angles, + = curled toward the other fin (scalar or list root -> tip) |
| `table.compliant_contact_stiffness` / `_damping` | spring table contact (default off) |
| `robot.limit_margin_deg` / `limit_margin_joints` | open the arm joint limits at runtime (default 0) |
| `robot.joint_offsets_rad` | replace the profile's joint offsets (e.g. W9) |
| `robot.arm_stiffness` | arm actuator stiffness (scalar or {joint: N m/rad}) |

## Default (gate pick "W9 + B25C")

4 sliced hinges per fin, k 2.5 N m/rad, bias -0.10, effort 0.74 N m, `robot.limit_margin_deg` 8, table-only low
friction, with the profile on the joint fit's W9 offsets. Gate agent: lift 81.7 % (85.0 % with real brick poses),
held reading 6.0, held x +1.2 mm, wrist_flex RMSE 0.6 deg. This folder's n = 60 run of the same (`finray_b010_n4_w9`,
jarvis): lift 78 % (66-87), held reading 6.04, held centre (+1.0, 3.1, -95.6) mm vs real (-0.8, 2.8, -88.0).
With `rest_curl_deg` [0, 0, 8, 8]: 72 %, held z -93.7 (1.9 mm deeper); [0, 0, 15, 15]: 47 %. Not used.

**What is fitted vs assumed.** Measured / fitted: the fin placement (CAD registered onto the URDF meshes), the jaw
mapping q = 0.0229 u - 0.099 (rigid-fin silhouette fits on 157 open wrist frames; perception's 496-frame fit gives
0.02406 u - 0.118), the effort (LeRobot 0.7 `so_follower` gripper limits on a 12 V STS3215), the hinge stiffness
(held reading 5.72 on a 31.7 mm brick; sim 6.0), and the check of the held pose against perception's held-brick fit.
Assumed: 4 rigid slices on independent spring hinges below a 12 mm clamped root (uniform stiffness, 3 g segments,
damping 0.01), compliant fin contact 10 kN/m / 60 N s/m, TPU-ABS friction 0.9 / 0.8 (literature), fin-table friction
0.15. Not reproduced: the fin-ray wrap that draws the brick in (coupled curl unstable, a fixed pre-curl costs lift),
so the brick sits about 7.6 mm shallower than real (the arm reaches the real depth to 0.6 mm; the gap is set at the
close); and closed-empty touching, which with the linear map is the squeezed fins bending against each other (fin-fin
contact is off), so the twin's empty closed tips stay about 12 mm apart.

## Results: bending fins

Realism gate, 60 cal2 episodes, joint-fit W offsets (`calib/profile.py` default since 1123cfa), fin-table friction
0.15, bias -0.242, effort 0.74 N m unless marked; `out/gripper/n60/`, `validate_gripper.py` configs. Held pose = the
median held brick centre in `gripper_link` from the gate traces; real = (-0.8, +2.8, -88.0) mm (perception agent,
599 held frames of 2x2 bricks), i.e. 17 mm above the fixed fin tip.

| config | lift >= 3 cm | held reading (real 5.72) | inspect view | held brick x / z (mm) | dx / dz to real |
|---|---|---|---|---|---|
| rigid fins (joe_cal2 bias, 0.4 N m) | 70.0 % | 5.86 | 87 % | 7.7 / -99.3 | 8.5 / 11.3 |
| bend2 (CAD pieces) k0.7 | 76.7 % | 5.87 | 80 % | -4.0 / -98.8 | 3.2 / 10.8 |
| bend2 k1.0 | 80.0 % | 6.48 | 80 % | -3.6 / -99.5 | 2.8 / 11.5 |
| bend3 k0.7 (params.json now) | 78.3 % | 5.69 | 75 % | -3.7 / -96.9 | 2.9 / 8.9 |
| sliced 2 hinges k1.05 | 80.0 % | 7.45 | 85 % | -1.5 / -96.6 | 0.7 / 8.6 |
| sliced 3 hinges k1.25 | 80.0 % | 6.29 | 77 % | -3.2 / -96.6 | 2.4 / 8.6 |
| **sliced 4 hinges k1.45** | **80.0 %** | **5.97** | 73 % | -2.8 / -95.6 | **2.0 / 7.6** |
| sliced 5 hinges k1.7 | 80.0 % | 5.39 | 77 % | -3.4 / -95.5 | 2.6 / 7.5 |
| bend3, moving fin root k3 | 76.7 % | 6.2 | 72 % | -8.3 / -96.7 | 7.5 / 8.7 |

Lift intervals at n = 60 are about +-11 points (Wilson), so the configs do not differ in lift; they differ in the
held reading and the held pose. With the same W offsets the rigid fins lift 70-72 %, the bending fins 77-80 %.

* **Coupling ("fin-ray curl", PhysX mimic joints, distal half of the hinges geared -1 to the root):** not usable.
  Bench (31.7 mm brick): 2 hinges nearly rigid (stall at reading 18); 3 hinges throw 2 of 5 bricks; 4 hinges blow up.
  The independent spring hinges are stable.
* **Hinge count:** 2 -> 4 hinges seats the brick 1-3 mm deeper; 4 and 5 are the same (returns stop at 4).
* **Centring:** the rigid fins hold the brick 8.5 mm towards the moving jaw. Every bending config holds it within
  0.7-3.2 mm of the real x; the fixed fin gives as much as the moving one, as the forces on the brick are equal.
* **Depth:** every config still holds the brick 7.5-11 mm shallower than real. Hinges move it 2-3 mm; fin-brick
  friction (rigid mu 0.6 / 1.2), effort and stiffness do not move it. The rest is not a finger lever (arm / tip
  height; the W9 offsets lower the tip about 3 mm).
* **Why the brick sits shallow: the real arm presses the fins into the table.** FK of the recorded closes (450
  episodes, the frames around the close, lowest CAD fin tip) puts the undeflected tips a median 5.3 mm (W) / 8.4 mm
  (W9) below the table top (76 % / 86 % of episodes below it; joe_cal2: 0.5 mm). The real TPU tips deflect on the
  table; the twin's arm is stopped by it. Tried on the 4-hinge fins (n = 60, joint limits +8 deg; lift, held centre z):

  | variant | lift | held z (mm, real -88.0) |
  |---|---|---|
  | W | 82 % | -95.5 |
  | W + compliant table 2 / 5 kN/m | 82 % / 82 % | -96.6 / -96.9 |
  | W9 | 78 % | -95.0 |
  | W9 + compliant table 0.5 / 2 / 5 kN/m | 82 / 78 / 78 % | -95.1 / -95.0 / -94.7 |
  | W9 + soft arm (lift / elbow / wflex drive 20 instead of 60 N m/rad) | 72 % | -95.9 (lift / elbow RMSE 3.5 deg) |

  None moves the held brick more than about 1 mm (the soft table raises the grasp-detected rate to 90-97 % but not
  the lift rate). The wrist frames at the close do not show the tips (the brick hides them), so tip bowing on the
  table cannot be checked there (`out/gripper/close_frames.jpg`).
* **The arm is not the limit; the squeeze pushes the brick out.** On the 60 W9 closes (gate traces, FK of the sim's
  achieved vs the real joint states) the undeflected CAD tip is 3.4 mm below the table in sim vs 5.0 mm in the real
  FK (shortfall 0.6 mm; lift -0.6 deg, elbow +0.3, wrist_flex 0.0 deg): the bending fins already deflect on the
  table. The brick's height in the gripper is set at the close (-94.6 mm at the close, -95.0 while held), not by
  slipping during the lift. Outward-bending hinges make a V that opens downward, so the squeeze pushes the brick
  toward the tips (bench: placed 6 / 12 mm above the tip, it ends 2 / 8 mm above); a real fin-ray wraps and draws the
  object in. `bend.rest_curl_deg` pre-curls the distal hinges toward the other fin so the V closes at the bottom
  (bench, bias -0.10, 4 hinges, k 2.5, brick placed 12 / 20 mm up: no curl 7.9 / 20.0 mm at the end, curl 8 deg
  15.1 / 21.5, curl 15 deg 9.8 / 29.7).
* **Bias -0.10 (the linear servo map from the open wrist frames)** needs stiffer hinges: with the jaw target that
  close (u 2.7 -> q -0.038) the fins must hold the jaw off. Bench, 4 sliced hinges, brick 20 mm up: k 0.8 / 2.5 / 6
  read 3.9 / 5.9 / 7.3 (real 5.72), so k 2.5.
* **Inspect view** drops from 87 % to 73-85 % with bending, while the held x moves towards the real one: worth
  checking the inspect-view test / camera against the real held pose.
* **Shape vs the real wrist frames** (`fin_shape.py`, 30 hold frames, median boundary distance): rigid fins at
  joe_cal2's / the touch bias 20.6 / 18.2 px, bend2 at its bench angles 15.2 px; a free per-frame fit of the chain
  reaches 13.1-13.4 px for 2-5 hinges (floor set by how well the CAD silhouette registers; closed frames are also at
  12-15 px). The fitted real angles agree across hinge counts: fixed fin root ~14-15 deg away from the brick with the
  distal hinges +1 to +3 deg toward it (a slight curl); moving fin 2-5 deg per hinge spread along it. The fit is not
  precise enough to set a stiffness profile, so the hinges are uniform.
* **TwinEnv replay** (bend3 k0.7, episodes 7 18 33 122 140 230 285 379): 8 / 8 lifted, held reading 5.0-6.1 (real
  4.65-6.05), gripper RMSE 5.2 deg (11 and 8 deg on episodes 7 and 18).
* **Stability** (`jitter.py`, brick in the gripper frame, 0.1 s high-pass): bend2 k1.5 on episode 17: 0.0001 mm RMS
  static, 0.058 mm while moving; sliced 4 hinges k1.45 on episode 18: 0.12 mm static, 0.17 mm while moving (held
  reading 5.7, brick centre z -87.2 mm, x -3.9 mm). Videos (wrist | inspect): `out/gripper/jitter_bend2_k1.5_ep17.mp4`,
  `out/gripper/jitter_grid_n4_ep18.mp4`.
* **Joint limits:** `robot.limit_margin_deg = 8` (arm joints, see physics README) takes the gate's wrist_flex RMSE
  from 1.67 to 0.66 deg with the same lift (78 %).
* **Throughput:** 90-400 ticks/s at 60 envs for both rigid and bending fins on the shared GPUs; no clean
  measurement (one same-stream pair: rigid 197, bend2 106-117).
* **Gripper mapping across the range** (`fin_shape.py mapping`, rigid-fin fits on 157 open frames before the close,
  9.8 px): q = 0.0229 u - 0.099 fits the open readings (u 8.7: 0.118, 13.5: 0.201, 20: 0.412, 25.8: 0.549,
  32.4: 0.802 rad; the touch mapping gives 0.15-0.30 rad less, joe_cal2's 0.105 bias about 0.13 more). The closed
  frames (tips touching at u 2.8) are not on that line (it gives q -0.035): either the servo map is linear with bias
  about -0.10 and squeezed-shut fins bend tip to tip (the twin cannot show this: fin-fin contact is off with the
  articulation's self-collision), or the map is not linear (quadratic through both: -0.000521 u^2 + 0.0469 u - 0.327).
  Not changed; worth a gate run with `bias_rad = -0.10`.

Wrist-camera red silhouettes, contour fits and overlays: `out/gripper/fin_shape_fit_<N>.{json,jpg}`,
`fin_shape_score_*.{json,jpg}`, `fin_mapping.json`, `fin_diag.jpg`, `closed_empty_evidence.jpg`.

## Results: rigid fins (before the bending model)

Realism gate, 60 cal2 episodes (validate.py's seed-0 list, plates parked), jarvis; `out/gripper/n60/`.
Re-run on watson after the last shared changes (`plates.rim_h` 0.0238, the `curved_2x3` brick mass of 9.3 g, the wrist
pose from `robot/wrist_mount.json`), config `finray_final` = params.json: **67 % (54-77), inspect view 90 %, held
reading 5.79 / real 5.75, grip RMSE 2.18 deg**, the same as before them.

| config | lift >= 3 cm (Wilson 95 %) | inspect view | held reading sim / real | grip RMSE (deg) |
|---|---|---|---|---|
| before: stock jaw pads (`pads_cad`) | 37 % (26-49) | 80 % | 5.83 / 5.72 | 1.75 |
| **fin-ray, params.json (k 10 kN/m, 0.4 N m)** | **67 % (54-77)** | 90 % | 5.87 / 5.72 | 2.18 |
| fin-ray, k 20 kN/m, 0.7 N m | 68 % (56-79) | 90 % | 5.20 | 2.03 |
| fin-ray, k 20 kN/m, 0.5 / 1.2 N m | 67 % / 68 % | 90 % / 90 % | 5.45 / 4.59 | 2.13 / 2.09 |
| fin-ray, rigid contact, 0.7 N m | 58 % (46-70) | 85 % | 7.46 | 2.29 |
| fin-ray, friction 0.6 / 1.2 (k 20 kN/m, 0.7 N m) | 65 % / 67 % | 90 % / 88 % | 5.24 / 5.93 | 1.89 / 2.27 |
| fin-ray, fins 5 / 10 mm further out | 67 % / 72 % | 90 % / 93 % | 6.13 / 6.82 | 2.33 / 2.70 |
| fin-ray, bias 0.075 | 70 % (57-80) | 90 % | 7.21 | 2.46 |

Arm tracking RMSE is the same with pads and fins (pan 0.74 / 0.75, lift 1.49 / 1.49, elbow 1.23 / 1.23 deg).
Throughput on a shared GPU was 136-186 ticks/s at 60 envs for every config, with no consistent difference between
pads and fins.

* The step from pads to fins is outside the intervals. Stiffness, friction, effort and bias change the held reading,
  not the lift rate.
* Mounting the fins 10 mm further out gives 72 %, inside the interval of the default, and the arm then hits the table
  (wrist-flex RMSE 1.13 -> 1.65 deg). Finger length is not the lever.
* The gripper RMSE is higher with the fins (1.75 -> 2.2). The extra error is all at real readings 4.5-15, the
  close and the hold (8-15: 2.9 -> 4.5 RMS, sim reads 2.6 above real while closing); with the jaw open it is
  identical (1.46 above 30). More episodes now have a brick between the fins, and the sim fins meet it earlier than
  the real ones do.
* `finray_bias_0.135` crashed at start-up on jarvis ("Failed to get DOF velocities from backend") and was not rerun.

TwinEnv replay (`validate_twin.py --only replay --tag _finray`, episodes 7 18 33 122 140 230 285 379, watson):

| | lift | held reading sim (real 4.65-6.05) | gripper RMSE (deg) | arm RMSE pan / lift / elbow / wflex / wroll |
|---|---|---|---|---|
| pads (physics agent's run) | 4 / 8 | 5.5-6.4 on the 4 lifted | 1.53 | 0.69 / 1.62 / 1.14 / 0.92 / 0.44 |
| fins | 8 / 8 | 5.62-6.04 | 2.41 | 0.70 / 1.58 / 1.16 / 1.02 / 0.45 |

**Videos** (`sim/twin/env/replay_video.py`, TwinEnv with the fins and its physics DR, top | wrist | inspect, real
above sim): `out/env/replay_ep007.mp4` (brick lifted) and `out/env/replay_ep033.mp4` (not lifted in this DR draw;
it lifted in the validate_twin replay). Jitter clips (episode 18 hold, pads vs fins): `out/gripper/jitter_before.mp4`,
`out/gripper/jitter_finray.mp4`.

Wrist view (`render_compare.py`, 12 recorded states): the rendered fins have the right shape, ribs and colour family,
but they do not land on the real ones. Red-mask IoU is 0.08 with the profile's wrist camera and 0.16 with the pose
fitted on the fixed fin. In the real frames the fins are larger and lower in the image, so the wrist-camera pose is
off in position as well as rotation (see below). The rendered red is also darker than the real one.

**Jitter** (`jitter.py`, replay of episode 18, brick pose in the gripper frame, 0.1 s high-pass):

| | hold while moving, 120 Hz | static hold, 120 Hz | jaw angle buzz (static) | held reading |
|---|---|---|---|---|
| pads | 0.095 mm / 0.22 deg RMS | 0.096 mm / 0.22 deg | 3.5 mrad RMS | 6.22 |
| fins | 0.018 mm / 0.011 deg | 0.0003 mm / 0.0003 deg | 0.002 mrad | 5.82 (real 5.98) |

The jitter was physics: with hard contact the jaw ran a 60 Hz limit cycle (q alternating 0.2447 / 0.2523 rad every
substep), pushed out by depenetration and pulled back by the drive. The rendered copy was never the cause: the visual
proxy pose equals the physics pose at every rendered frame (0.0 mm). A 30 Hz render samples one phase of the cycle, so
a static hold looked still, while the grasp transient jumped 1-2 mm / 4 deg between frames.

## Estimated, not measured

* **Contact stiffness, effort and bias** are fixed by one number, the held reading. 10 kN/m with 0.4 N m and
  joe_cal2's bias reproduces it; 20 kN/m with 0.5 N m does too. The user describes the TPU as quite soft, which is
  why the softer pair is the default. The real servo torque limit is not known.
* **Friction** is a literature range, not a station measurement.
* **Fin bending** is a chain of rigid segments on spring hinges, bending away from the brick. The real fin-ray also
  curls its tip in (wrap); the coupled hinges that would give that are unstable in PhysX here, and FEM was not tried.
  Hinge stiffness is calibrated on the held reading only (uniform along the fin).
* **Grasp depth:** the bending fins hold the brick centre 7.5-11 mm shallower than the real 17 mm above the fin tip;
  see Results.
* **Wrist camera:** `fit_wrist.py` now starts from the fitted mount (`sim/twin/robot/wrist_mount.json`, the
  physical rotation `local_to_cv`, since it projects with the measured intrinsics and distortion; joe_cal2 only if the
  file is missing). With that pose the CAD fixed fin's outline is 3.7 px from the real one without any correction
  (joe_cal2's pose: 17-20 px); a further 10 deg / 4 mm correction only brings it to 3.1 px, so the mount agrees with
  the fingers. The `render_compare.py` IoU numbers above were taken with the older joe_cal2 profile pose. The per-frame joint-angle fit of the moving fin is too noisy to check the
  gripper mapping.
* **Bases:** registration residuals to the stock meshes are a few tenths of a mm; the fixed base's z translation is
  good to about 0.15 mm.
