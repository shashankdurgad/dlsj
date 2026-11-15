# Twin physics: grasp and contact realism

`apply.py` + `params.json` = the physics layer the core env (`sim/twin/env/twin_env.py`) loads:
`patch_profile(profile)` -> `patch_scene_cfg(scene_cfg, profile)` -> `post_spawn(env)`; runtime DR with
`sample_physics_dr(rng)` / `apply_physics_dr(scene, draw)` / `nominal_physics()`.

```
sim/twin/run.sh sim/twin/physics/validate_physics.py                    # realism gate with vs without + drop test
sim/twin/run.sh sim/twin/physics/validate_physics.py --configs baseline,physics,sdf --no_drop
sim/twin/run.sh sim/twin/physics/realism_gate.py --physics on --out sim/twin/out/physics/x [--set key=json] [--trace 1]
sim/twin/run.sh sim/twin/physics/drop_test.py --physics on|off ; ... drop_test.py --plot_only
sim/twin/run.sh sim/twin/physics/gap_vs_q.py                             # jaw gap vs gripper angle (pads)
```
Outputs in `sim/twin/out/physics/` (`validate_physics.{json,md}`, `replay_compare.png`, `drop_test.{png,json}`).

The gate (`realism_gate.py`) runs `sim/residual_env/validate.py` unchanged with the twin profile (twin robot USD,
measured cameras), A1 servo / A2 scene params from `sim/realism/`, 20 cal2 episodes (seed 0), replay clamp off, and
the plates parked out of reach (the replay does not know each episode's real plate layout; with 23 mm rims the
replayed arm hits randomly placed plates, which inflated tracking RMSE to 4-5 deg). Extra metrics from per-tick traces:
held reading on lifted episodes only (validate.py's own number mixes in jaws that closed on nothing), brick tilt in
the grip, and the jaw gap at the held angle (< 31.7 mm = a pad is inside the brick).

## Results (20 cal2 episodes, plates parked; `validate_physics.json`)

| | grasp | lift >= 3 cm | inspect view | held reading, lifted eps (real same eps) | jaw gap while held (brick 31.7 mm) | RMSE pan/lift/elbow/wflex/wroll/grip (deg) | ticks/s (20 envs) |
|---|---|---|---|---|---|---|---|
| baseline (residual env as left, twin USD) | 25 % | 25 % | 70 % | 4.31 (6.05) | n/a (hull colliders still on) | 0.89 / 1.41 / 1.28 / 0.97 / 0.65 / 2.25 | 563 |
| **physics (params.json)** | 30 % | 30 % | 80 % | **5.81 (5.88)** | 34.3 mm, 0 pads inside a brick | 0.83 / 1.56 / 1.17 / 0.90 / 0.48 / **1.73** | 354 |

TwinEnv's own replay (`sim/twin/env/validate_twin.py --only replay --n_episodes 8`, episodes 7 18 33 122 140 230 285 379):
lift 4/8 with the hooks = 4/8 without; held reading 5.5-6.4 with (real 5.5-6.05) vs 3.7-4.4 without; gripper RMSE
1.53 vs 2.23 deg.

### What each change did (ablations, same 20 episodes)
* **Hull colliders still active on the twin USD (bug).** The URDF converter instances the collision meshes, so
  `app.finger_pads` never disabled them (and put a CollisionAPI on the visual Xforms, which PhysX parsed as extra
  hulls: the "triangle mesh ... falling back to convexHull" errors). Fixed by deactivating the collision instances.
* **Pads fitted to the CAD fingertips** (inner faces at x = -/+7.9 mm at q = 0, 16 x 36 x 8 mm): `gap_vs_q.py`
  measures the jaw gap in the PhysX joint frame: 31.7 mm at q = 0.192 (fingertip) / 0.205 (5 mm up) / 0.219 rad
  (10 mm up). Convex decomposition now cooks on Isaac Sim 6.2 but grasps worst (15 %); SDF ~ pads.
* **Rigid jaw-brick contact (the main physics finding).** With the env's defaults (brick max_depenetration_velocity
  0.5, 16 solver iterations) a light brick squeezed between the two jaws does not stop the jaw: in `jaw_stall.py`
  (gravity off) the jaw reaches its commanded angle through the brick (6 mm overlap) at 0.6 and 2.9 N m alike; a
  kinematic brick stops it at the geometric angle. So in every earlier config the held reading was just
  command + servo bias (~4.2), and the "lifts" came from pads sunk 2-7 mm into the brick (the teammate's 80 % on
  Joe's USD has the same signature: hold 4.26 = command + bias). bricks max_depenetration_velocity 20 + 64 position
  iterations (bricks and robot) make the jaw stop at the kinematic-brick angle (0.6 N m exactly; 2.9 N m sinks
  ~0.02 rad). Cost: ~1.6x slower physics (563 -> 354 ticks/s at 20 envs); 32 iterations is not enough.
* **Gripper bias**: with real contact, Joe's cal2 mapping (q = 0.02308 u + 0.105, "estimated against 32 mm CAD-jaw
  contact") reproduces the real held reading (5.81 vs 5.88) without fitting. The residual env's re-solved bias
  (-0.0263) only compensated its penetrating pads; with rigid contact it gives 7.4 (too high).
* **Gripper effort** 1.2 N m (LeRobot limits the gripper servo to ~50 % torque/current): no effect on the gate
  (soft-contact runs identical at 0.6 / 1.2 / 2.9); with rigid contact 2.9 N m reads 5.5 (slight sinking).
* **Jaw friction** 0.5 vs 0.9: 35 -> 40 % lift in the soft-contact setup; kept the literature value (PLA on ABS).
* **Bricks**: 2x2 6.75 g (BrickLink 3437), 2x4 12 g, curved 2x2 5.5 g (estimate); ABS mu 0.45/0.38, restitution
  0.1; stacked / touching pairs on the table and in the plate drift < 0.01 mm over 2 s, bounce on a 10 cm table drop
  1.5 mm (before: same; the env had no jitter problem).
* **Plates** (`drop_test.png`): the flat 6 mm disc kept every brick released inside its radius exactly where it
  landed (10 cm drop at r = 0.09 m: stays on the disc edge). The measured profile (triangle mesh on the kinematic
  plate, floor + sloped wall + 23 mm rim) makes bricks released over the wall slide / tip inward and lean against the
  wall, a brick on the lip can balance there, bricks just outside the rim end up leaning against the plate's outer
  wall. Plate contact offset 12 mm: with 2 mm a brick tunnelled through the thin shell at r = 0.085 m.
* **Table height** (experiment only, `--table_dz`): lowering the table 5-10 mm makes grasping worse, raising it
  5-12 mm makes the replayed arm hit the table (RMSE 6-39 deg). The table plane is consistent with the arm.

### Why the replay lift rate stays ~30 %
In the failed episodes the jaw closes to its commanded angle without touching the brick: the brick centre sits
15-25 mm below the fingertip point (TIP_LOCAL) when the jaw closes, i.e. the pads (bottom at z = -104 mm) overlap
the brick's upper 0-5 mm. The real held reading (5.75 -> q 0.238 with Joe's mapping) corresponds in the stall bench
to a brick centre ~10 mm below TIP_LOCAL, so the real fingers went ~0.5-1.5 cm deeper around the brick than the
open-loop sim replay puts them. That is a kinematics / calibration (or arm compliance) gap of the open-loop replay,
not contact physics; a closed-loop policy would lower the gripper. Configs that "lift" more than this (75 % with
the re-solved bias, 55 % rigid + bias 0.072) do it by sinking the pads into the brick or by over-reading.

### Physics DR (`params.json["dr"]`, `apply_physics_dr`)
brick friction 0.3-0.7, jaw friction 0.3-0.9, brick mass x0.85-1.15, gripper effort 0.6-1.8 N m. Isaac Lab 3.0's
PhysX views return warp arrays (fixed: `wp.to_torch` / `wp.from_torch`, CPU material buffers);
`drop_test.py` runs a draw + `nominal_physics()` as a smoke test.

### Geometry (visuals/assets.py, also the colliders)
Plate: R 0.092 m, rim 23 mm above the table, 2 mm shell, floor top 3 mm above the table (assumed foot), flat floor to
0.75 R (photo). Bricks: `brick_mesh('2x2' | '2x4' | 'curved_2x2')`, `brick_hull_points` (exact convex collider; the
curved slope is convex), `apply.brick_usd(kind)` writes a spawnable USD with a convex-hull collider.

### Later keys (fin-ray fingers; details and gate numbers in `sim/twin/gripper/README.md`)
* `gripper.finray.bend.*`: bending fins (hinged segments, articulation joints `fin_<fixed|jaw>_<a..e>`, actuator group
  `fins` added by `patch_scene_cfg`). `gripper.finray.bias_rad` -0.242 (fin tips meet at the real closed reading
  2.8), `gripper.finray.effort_nm` 0.74 (LeRobot `so_follower` gripper limits on a 12 V STS3215).
* Friction pair: table combine `min` 0.15 / 0.12, fins `average`, bricks `max` -> fin-table 0.15, fin-brick 0.9,
  brick-table 0.45.
* DR (small shifts around the nominal): `dr.gripper_effort_nm` 0.6-0.9 (nominal 0.74), `dr.fin_stiffness_scale`
  0.8-1.2 (`apply_physics_dr` writes the fin hinge stiffness x scale). A gripper-bias DR (+-0.02 rad) is not wired:
  the bias is the profile's scalar used by the env's unit conversions, so it needs a per-episode hook in the env.
* `table.compliant_contact_stiffness` / `_damping` (default off): spring table contact (`_material`), a stand-in for
  the soft fin tips pressed into the table at the real close (FK of the recorded closes puts the CAD fin tips a
  median 5 mm (W) / 8 mm (W9) below the table top).
* `robot.limit_margin_deg` (default 0) / `robot.limit_margin_joints`: `post_spawn` opens the arm joints' USD limits
  by that many degrees before `sim.reset()` (the demos pass wrist_flex +95 and shoulder_lift -100 deg). 8 deg: n=60
  wrist_flex RMSE 1.67 -> 0.66 deg, lift unchanged (config `finray_limits8`).
* `robot.joint_offsets_rad` (default unset): `patch_profile` replaces the profile's joint offsets, e.g. the joint
  fit's W9 (`validate_gripper.py` configs `*_w9`).
* `robot.arm_stiffness` (default unset): the arm actuator stiffness (scalar or {joint: N m/rad}); 20 on lift / elbow /
  wrist_flex costs lift (72 %) and tracking (lift / elbow RMSE 3.5 deg) without seating the brick deeper.

### Remaining gaps
* open-loop grasp depth (above); arm compliance / gravity sag beyond the encoder is not modelled;
* a 3 cm release onto the rim lip can launch the brick ~2 cm (thin trimesh + 12 mm contact offset);
* box pads, no ribbed jaw texture; brick inertia is a solid box (real Duplo is a hollow shell, COM ~2 mm higher);
* 64 solver iterations cost ~37 % throughput; no real drop / friction measurements on the station (literature values).

