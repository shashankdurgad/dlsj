# Sim joint-offset fit from recorded data

The fit corrects the 5 body-joint offsets that map our LeRobot degrees onto Joe's SO-101 USD joints:
`sim_rad = deg2rad(state + delta) + joe_offset`. Joe's top camera stays fixed: Stage A2 showed that fitting the joints and the camera together is ill-posed.

## Method
- **FK.** Numpy forward kinematics (`so101_fk.py`) uses the joint chain dumped from Joe's USD (`dump_kinematics.py` → `kin.json`). It matches Isaac's physics link poses to 0.4–1.3 mm per link.
- **Data.** `jofit_extract.py` runs on the XPS, read-only:
  - 720 top frames at recorded poses across all phases: 420 old calibration (70 episodes), 299 `cal2` (50 episodes);
  - grasp and inspection states for 735 episodes, from the 3-phase labels.
- **Loss** (`joint_offset_fit.py`). Robust terms, summed:
  1. **Top-camera finger keypoints.** The real red-finger mask, restricted to a window around the prior projection, gives a centroid and a far end. These are compared with the projected FK finger points, using a Huber loss with a 40 px knee.
  2. **Grasp height.** The fingertip z at the grasp frame should sit in [0.8, 3.5] cm, with a strong penalty below the table.
  3. **Inspection.** A weak term (×0.05): the fingertip at the inspection mark should be near the inspect camera's axis. It is weak because Joe's inspect camera pose is itself estimated.
  4. **Prior.** A pull toward Joe's offsets, (delta / 5°)².
- **Validation.** A 70/30 split by episode, plus 8 bootstrap refits for the confidence intervals.

## Result (`joint_offsets_fit.json`)

Fitted delta in degrees, ± bootstrap std:

| set | pan | lift | elbow | wrist flex | wrist roll |
|---|---|---|---|---|---|
| old (duplo_teleop) | **+2.64 ± 0.30** | **−1.90 ± 0.21** | **+1.28 ± 0.20** | **+5.38 ± 0.49** | −0.12 ± 0.29 |
| cal2 (raw cal2 states) | −18.78 ± 0.52 | +3.54 ± 0.11 | +4.06 ± 0.22 | +4.69 ± 0.57 | +0.51 ± 0.13 |

- **Consistency.** cal2 − old comes out as pan −21.4 (expected −20.35 from `recal_2026-11-14`), lift +5.4 (expected +8.0), elbow +2.8, wrist flex −0.7 and roll +0.6. The fit recovers the known recalibration from images and heights alone.
- **Prior bound.** No joint was pulled more than 10° from Joe's offsets in the old set.

Held-out errors, before → after:

| | keypoint median / p90 (px) | grasp z in band | below table | tip in inspect FOV |
|---|---|---|---|---|
| old | 55.4 / 91.3 → **47.4 / 87.6** | 0.50 → 0.52 | 0.04 → 0.06 | 0.02 → 0.10 |
| cal2 | 71.3 / 122.2 → 70.2 / 116.1 | 0.37 → 0.45 | 0.08 → 0.08 | 0.05 → 0.25 |

- **What's left.** The remaining ~47 px keypoint error, and the inspect-FOV rate, are beyond what joint offsets can explain. The likely causes are the camera poses (Joe's estimates), finger visibility from above, and segmentation. That needs a tag-based rig calibration.
- **Overlay.** `joint_offset_overlay.jpg` shows red = Joe's offsets and green = fitted, on real top frames.

## Replay validation in the residual env (Stage B)

20 recorded old-calibration demos replayed open-loop, same episodes in both runs:

| | Joe | **fit_old** |
|---|---|---|
| grasp detected | 15% | **30%** |
| lifted ≥ 3 cm | 15% | **30%** |
| brick in inspect view at the mark | 20% | 15% (1 episode) |
| tracking RMSE pan/lift/elbow/wflex/wroll/grip (deg) | 1.42/1.92/1.38/0.92/0.84/1.34 | 1.42/2.12/1.39/1.02/0.86/1.33 |

Changes in the env:
- `fit_old` is now the env default: `EnvCfg.joint_offsets`, `resolve_joint_offsets` in `residual_env/env.py`.
- Use `"joe"` for the original offsets, and `"fit_cal2"` for raw cal2 states.
- `validate.py --joint_offsets` selects the source.
