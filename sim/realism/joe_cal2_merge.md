# Joe's cal2 sim calibration merged into the residual env (14 Nov 2026)

Source (read-only): `pi05-isaac-so101/cal2-calibration/` on Joe's Spark — `reference-cal2.json` (his
"replay-ep101-corrected-cameras": multi-pose camera/joint fit on cal2 episodes 101 and 110, held-out 120), the
ep120 held-out and camera-adapted references, `setup_scene.py` / `configure_reference.py` (camera convention:
OpenCV rows -> USD rows x, -y, -z; focal 20 mm, apertures 20*W/f), `replay.py` (his protocol). Copied JSONs and
his two recordings into `sim/residual_env/assets/joe_cal2/`. Robot USD `so101_new_calib_11` is referenced by path;
it is content-identical to `_2` (only the generator's temp path in the doc string differs).

What the profile takes from Joe: joint offsets (incl. the reseated wrist_roll, -1.80 rad), top / wrist / inspect
cameras, his gripper encoder scale (0.0231 rad/unit from the follower's tick range). What stays ours: the A1
servo model (reference filter + PD), the LeRobot clamp, measurement noise, the box finger pads, the A2 scene
distributions. Gripper bias re-solved for our pads (his 0.105 rad assumes his CAD-jaw contact).

Result: `joe_cal2` is the default for cal2. On 20 cal2 replays it lifts 80 % (vs 20 % for our fit_cal2) with
better shoulder-lift tracking (1.55 vs 2.71 deg); in Joe's own protocol our fingertip reaches his image-placed
target within ~11 mm (fit_cal2: ~100 mm). Our earlier data-fitted offsets were limited by Joe's old camera (the
fit could not move the camera); his cal2 camera fixes most of the camera gap noted in joint_offset_fit.md.
Open gaps: held-gripper reading 4.3 vs real 5.8, and fewer lifts than Joe's own sim on his episodes (colliders /
drive gains). Numbers: `sim/residual_env/README.md`, `sim/residual_env/results/joe_cal2_merge/`.
