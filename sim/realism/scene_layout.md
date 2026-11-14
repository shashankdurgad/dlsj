# Stage A2: scene layout and initial conditions from recorded data

**Sources:**
- Key joint states and phase timing for all 740 usable episodes: 516 from `duplo_teleop` (old calibration) and 224 from `duplo_teleop_cal2` (new calibration, converted to the old frame with `recal_2026-11-14`). Phases come from the 3-phase labels.
- Top and wrist frames from 65 sampled episodes (40 old + 25 cal2).
- Forward kinematics and renders in Joe's Isaac scene, read-only, on the Spark.
- Numbers are in `scene_params.json` (frame: Joe's scene, metres, robot base at the origin).

## Top camera and table plane
- **Camera model:** Joe's `camera-fit.json` pinhole (f = 611.6 px, principal point (320, 240)). Back-projection to z = 0 is in `table_plane.py`.
- **Robot base:** projects to (336, 65) px, where the base is in the real frames.
- **Scale:** 1.13 mm/px at the image centre.
- **Plates:** they come out at a consistent 9.5–10.6 cm radius (yellow 10.6, blue 9.9, orange 9.5).
- **Brick boxes:** about 50 mm, larger than a 31.8 mm Duplo 2×2, because the boxes include padding and side faces. They're unusable as a scale check.

## What the sim should randomise
| Item | From data | Sim recommendation |
|---|---|---|
| Target brick xy (FK fingertip at `grasp_frame`) | x 0.18–0.42 m, y −0.19–0.17 (old) / −0.24–0.09 (cal2); mean (0.30, −0.02) | Sample uniformly inside the p5–p95 box; the brick lies flat, with yaw uniform. Bricks are also picked from inside plates, not only the open table. |
| Grasp fingertip height | median 2.5 cm (p10 0.7, p90 4.8) | Brick 2×2, 31.8 × 31.8 × 19.2 mm (plus studs) |
| Other bricks on the table | median 2 detected (the detector under-counts) | 2–6 distractor bricks, mostly 2×2 (2×4 rare) |
| Plates | 3 plates (yellow, orange, blue = "purple" bowl), radius about 0.10 m. **The layout changes every few dozen episodes**, with release positions drifting 5–14 cm per bowl. | Random non-overlapping plate centres in x 0.12–0.39, y −0.23–0.16, at least 0.2 m apart, per episode |
| Start joint state (deg, old cal) | median [−2, −59, 56, 60, 18, 29], IQR [26, 65, 53, 32, 20, 10] | Sample from the empirical start states (`joints.json`) |
| Inspection pose | fingertip (0.32, 0.19, 0.15) m, IQR about 3 cm; joints median [−34, 16, 2, 46, 12, 6] | Fixed target pose with ±2 cm noise |
| Release (place) | fingertip about 0.10–0.12 m above the plate, xy at the current plate | Release over the sampled plate centre |
| Gripper holding value | 5.9 (p10 4.7, p90 6.2), in 0–100 units | Grasp success means the gripper stalls at about 4.5–6.5 |
| Phase durations (s) | grasp 2.5, present 1.5 (old) / 2.4 (cal2), place 1.2 / 1.6, episode 6.5 / 7.6 | Episode horizon about 12 s with margin |

## Camera realism check
**Method:**
- Render Joe's top camera at the recorded joint states (mid-present and inspection frames, 129 usable frames).
- Compare the red-finger mask against the real frame. The real mask is restricted to a window around the sim fingers, so red bricks don't count.

**Model consistency:** projecting the FK finger points with `table_plane.py` lands within 8.6 px of Isaac's own render of the same points, so the camera model and the USD camera agree.

**Sim vs real:**
- **Systematic offset:** small, (−1.4, −10.8) px mean.
- **Per frame:** median 38 px, p90 69 px, finger-mask IoU about 0.1. This includes segmentation noise, since the real fingers are small and shadowed from above.
- **Visual check:** the overlay (`overlay_sim_fingers_on_real.jpg`) shows the sim fingers on or near the real gripper in most frames.
- **Projected grasp fingertips** (`grasp_tip_overlay.jpg`) land on or next to the brick in the gripper.

**Refinement:** a 6-DoF + focal fit on 70% of the frames improves the fit set from 35 to 23 px median, but the held-out set only from 50 to 43 px. It needs large changes (rotation −5°/8.5°/0.5°, translation 64/90/40 mm, focal ×0.93), so it's likely absorbing joint-offset and segmentation error. **Recommendation:** keep Joe's camera; treat the refined parameters (`camera_check.json` → `refined_camera`) as tentative until a tag-based calibration (stage A1).

## Caveats
- **Detector-based counts are biased:** the detector boxes plates and blue cable clips and misses low-contrast bricks. The colour counts (blue-heavy) are unreliable; the target-brick distribution from FK is the trustworthy one.
- **Plate segmentation:** some false plate centres come from background objects at the top of the frame. The p5–p95 box is still reasonable.
- **Joint conversion:** this uses Joe's estimated `joint_offsets_rad` (from one landmark). Grasp heights look plausible, but an offset error there shows up as camera error above.
- **Wrist camera:** renders at the start pose are in `renders/` on the Spark (`~/realism/renders`). No quantitative wrist comparison was done.

## Files
- **Scripts:** `extract_episodes.py`, `joint_stats.py`, `fk_keyposes.py`, `render_poses.py`, `table_plane.py`, `camera_check.py`
- **Outputs:** `scene_params.json`, `camera_check.json`
- **Figures:** `layout_scatter.jpg`, `grasp_tip_overlay.jpg`, `overlay_sim_fingers_on_real.jpg`
- **Raw:** on the XPS at `~/inference/tmp/realism/run`; on the Spark at `~/realism/{fk.json,renders}`
