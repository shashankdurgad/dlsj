# Grasp residual on the robot: deploy spec (draft, for the deploy agent)

The residual is a small deterministic MLP that adds bounded corrections to the MolmoAct2 8k plan **only while the pick
prompt is active**. One MolmoAct2 forward per chunk (K = 1, as today); no critic, no best-of-K at test time.

## Artefact
`python sim/twin/rl/export_residual.py sim/twin/out/rl/<run>/residual.pt [--onnx]` writes `<run>/deploy/`:
* `residual_actor.npz`: weights + `meta` (JSON: dims, flags `latent`, `detector`, `top_target`, `inject`,
  `residual_scale`, `rate_per_tick`, `tip_local`, `joint_offsets`, decision / fade ticks, the training `grasp_cfg`,
  source checkpoint). The export REFUSES checkpoints without `res_scale` / `rate` / `tip_local`: the robot must use
  exactly the meta's values, never hard-coded ones;
* `residual_actor.onnx` (inputs `state` (1, S), `plan` (1, 30); output `residual` (1, 6)) for OpenVINO;
* `export.json`: numpy-vs-torch parity (2e-7) and latency (0.09 ms per call in numpy on a laptop CPU).
Runtime without torch: `from sim.twin.rl.export_residual import ResidualPolicy; r = ResidualPolicy(npz)(state, plan)`.
The "observation normaliser" is the fixed scaling below (no learned statistics).

## When it runs
* Gate: `sim/twin/rl/gate.py` `is_pick_prompt(prompt)`. The 8k fine-tune knows ONLY its training prompts
  (`sim/twin/data/models/molmoact2_ft_8k/finetune_cal2_456.json`), sent to MolmoAct2 verbatim:
  - pick (gate open): `pick up the blue brick`, `pick up the brown brick`, `pick up the red brick`,
    `pick up the white brick`, `pick up the yellow brick`;
  - bring (gate closed, fade): `bring the brick to the inspection camera`;
  - place (gate closed): `place the brick in the orange bowl`, `place the brick in the purple bowl`,
    `place the brick in the yellow bowl`.
  Tests: `test_gate.py`. Any non-pick prompt: residual exactly 0.
* Fade: `Fader(6)`: when the prompt switches away (bring / place, or the operator's mark) the applied offset falls
  linearly to 0 over 6 control ticks (0.2 s); the residual output is ignored from then on.
* Rate: one residual decision every 5 control ticks (30 Hz control -> 6 Hz), the offset is rate-limited every tick.

## Inputs (all from the robot), float32, in this order
`state` (S = 25, + 4 with `top_target`, + 128 with `latent`):
| idx | field | units / frame | scale |
|---|---|---|---|
| 0-5 | measured joints (pan, lift, elbow, wrist_flex, wrist_roll, gripper) | LeRobot deg of the cal2 frame, gripper 0-100 | / 100 |
| 6-8 | wrist detector: target brick centre minus the fingertip point `tip_local` (meta) | gripper_link frame, m | x 10 |
| 9 | detector valid (target locked and detected this frame) | 0 / 1 | |
| 10 | gripper load proxy = measured gripper - servo/command reference | units | / 10 |
| 11 | time since the pick prompt started | control ticks | / 240 |
| 12-17 | the residual's currently applied offset (xyz m, wrist_flex deg, wrist_roll deg, gripper units) | | / bounds below |
| 18-23 | previous residual output r | [-1, 1] | |
| 24 | pick gate (1 while the pick prompt is active) | 0 / 1 | |
| 25-27 | (`top_target`) top-camera target position minus the fingertip point `tip_local` | gripper_link frame, m | x 10 |
| 28 | (`top_target`) top target valid | 0 / 1 | |
| +128 | (`latent`) MolmoAct2 mean-pooled prefix hidden state (2560) @ fixed projection `LatentProjector(2560, 128, seed 0)` | | |

`plan` (30): the controller's next 5 planned goals (the current RTC plan at ticks t..t+4) minus the measured joints,
flattened (5 x 6), / 20.

**Fingertip point** (`tip_local`, gripper_link frame, m): ONE constant for the obs (6-8, 25-27), the residual's
fingertip IK and the sim's success checks: the physics layer's grasp point (`sim/twin/physics/apply.tip_local()`,
written into the residual env's `TIP_LOCAL` by `patch_profile`). With the current twin defaults (B25C fin-ray
fingers) it is (0.0063, 0.001, -0.0902); the older box-pad twin used (0.008, 0, -0.090). Always take it from the
export meta, and give the same point to `wrist_target.box_to_rel` on the robot.

Detector rel on the robot: `sim/twin/perception/wrist_target.py` `TargetTracker` + `box_to_rel` on the
`OVBrickDetector("brick-detector-v1", "NPU")` wrist boxes (~4 ms NPU). Needed API change: the tracker should acquire
the wrist box whose back-projected table point is nearest the PROJECTED TOP-CAMERA TARGET (not the fingertip), i.e.
`TargetTracker.update(boxes, joints, target_xy=...)`; until then use its current gating.

## Output and how it is applied
`r` (6) in [-1, 1] -> offset TARGETS = r * `residual_scale` (meta; the v4 / image-path runs train at fingertip
x, y in the gripper frame +-3.5 cm, z +-2 cm along the fingers, wrist_flex / wrist_roll +-10 deg, gripper +-30
units; older v1-v3 runs at +-2.5 cm on all three). +30 gripper units keep the jaw open: the residual can delay / veto
a close for as long as it outputs it.
Per control tick: applied offset moves toward the target at <= `rate_per_tick` (meta: 5 mm xyz, 3 deg wrist; gripper
immediate); the planned goal of this tick `g` becomes
`g' = IK(g with wrist += offset_wrist, tip(g) + R(g) @ offset_xyz)` (2-step damped least squares on pan / lift /
elbow, `sim/twin/rl/kin.py` `Kin(meta joint_offsets, tip_local=meta tip_local)`, URDF `so101_new_calib`), gripper
`clip(g_grip + offset_grip, 0, 100)`. Then the controller's own max-step (4 deg per tick) and the follower's
`max_relative_target`, unchanged. Outside the gate: offset fades to 0 (above), then `g' = g`.

## Optional safety rule (evaluated in sim with `--veto 1`)
Close-veto: if the wrist tracker's locked box is not the target at the moment the plan closes the jaw, hold the jaw
at its measured opening until it is (deterministic, independent of the residual).

## Versions
Each export records its source checkpoint; the training run's `config.json` records the sim snapshot (git head +
dirty files of the twin) and flags.
