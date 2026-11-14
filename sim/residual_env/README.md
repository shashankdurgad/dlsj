# Residual-RL environment (Isaac Lab 3.0, SO-101 brick task)

Batched sim for training a small **residual correction policy** (gripper + wrist) on top of a frozen VLA,
parameterised from our recorded data. Runs on Joe's Spark (GB10) as user `yan` in `~/isaaclab-venv`
(Isaac Lab 3.0.0 over Isaac Sim 6.2; a copy of `~/isaacsim-venv`, so the benchmark env is untouched).

```
cd ~/residual_env
export OMNI_KIT_ACCEPT_EULA=YES LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1
~/isaaclab-venv/bin/python validate.py --episodes 20 --num_envs 20        # realism gate (cal2, joe_cal2 profile)
~/isaaclab-venv/bin/python validate.py --calib fit_old --grasp_rule labels \
    --data ~/residual_env/data/duplo_teleop_3phase                          # old-calibration data
~/isaaclab-venv/bin/python crosscheck_joe.py                                 # Joe's ep101/ep120 replays, his protocol
~/isaaclab-venv/bin/python train.py --base replay --num_envs 32 --minutes 5 # SAC smoke run
# frozen SmolVLA in the loop (renders wrist + top; start the worker first, in its own env):
HF_HOME=~/smolbench/hf ~/isaacsim-venv/bin/python vla_batch_worker.py \
    --ckpt ~/smolbench/ckpt/checkpoints/snapshots/step020000.ckpt --ipc ~/residual_env/ipc &
~/isaaclab-venv/bin/python train.py --base smolvla --num_envs 8 --latency 0.074 --minutes 60
```

## Files
| file | what |
|---|---|
| `calib.py` | calibration profiles: joint offsets, cameras (top/wrist/inspect), gripper mapping, robot USD |
| `crosscheck_joe.py` | Joe's cal2 replay protocol (his episodes, 2x4 target from his top image) in our env |
| `assets/joe_cal2/` | Joe's cal2 calibration JSONs, his ep101/ep120 recordings and his own replay results |
| `scene.py` | N cloned envs: SO-101 (Joe's USD), table, target 2x2 brick + 1-3 distractors, 3 kinematic plates, wrist + top cameras; `load_layout` reads stage A2's `scene_params.json` |
| `env.py` | `ResidualEnv`: servo model, clamp, noise, base-policy latency/hand-off, residual, simulated detector, phases, rewards, resets |
| `app.py` | launcher (GB10 render fix), finger pads, robot paint, `make(cfg, base)` |
| `bases.py` | `ReplayBase` (recorded teleop as the base), `HoldBase`, `IPCVLABase` (frozen VLA in its own process, batched) |
| `vla_batch_worker.py` | batched SmolVLA worker (Studio checkpoint, wrist + top) |
| `sac.py`, `train.py` | compact GPU SAC + training loop |
| `validate.py`, `debug_replay.py` | realism gate; single-episode replay with frames + trace |
| `smoke_scene.py`, `jawstall2.py`, ... | diagnostics used while building (kept for reference) |

## Calibration profiles (`EnvCfg.calib`, default `joe_cal2`)
| profile | joint offsets | cameras | gripper q (rad) | pairs with |
|---|---|---|---|---|
| **`joe_cal2`** (default) | Joe's cal2 multi-pose fit (`assets/joe_cal2/reference-cal2.json`), incl. wrist_roll -1.80 rad | Joe's cal2 top + wrist + inspect | `units * 0.02308 - 0.0263` (Joe's encoder scale; bias re-solved for our pads) | cal2 data, the current robot |
| `fit_cal2` | our data fit (sim/realism) | Joe's original (old) | `(units + 2.4) / 100 * 1.745` | cal2 data (superseded) |
| `fit_old` | our data fit | Joe's original | same | old-calibration data (517 set) |
| `joe` | Joe's original camera-fit.json | Joe's original | same | old-calibration data |

20 cal2 episodes replayed open loop (Desmond's 456 3-stage build, brick at FK of the gripper-close frame):
| | `joe_cal2` | `fit_cal2` |
|---|---|---|
| grasp detected / lifted >= 3 cm | **80 % / 80 %** | 20 % / 20 % |
| brick in the inspect view at the operator's mark | **75 %** | 45 % |
| tracking RMSE pan / lift / elbow / wrist flex / roll / gripper (deg) | 1.06 / **1.55** / 1.31 / 1.02 / 0.74 / 2.30 | 1.00 / 2.71 / 1.41 / 0.73 / 0.49 / 1.48 |
| gripper reading while holding (real 5.75-5.9) | 4.26 | 4.15 |

Joe's own protocol (`crosscheck_joe.py`: his recorded commands, 2x4 target placed from the top image): our fingertip
comes within 10.5-11.8 mm of his target with `joe_cal2` vs 99-114 mm with `fit_cal2`; ep101 lifts 8.1 cm (not held;
Joe's sim: 15.6 cm, held), ep120 does not lift (Joe: 3.7 cm held-out, 16.4 cm with his ep120-adapted camera).
Known gaps: the held gripper reads ~4.3 instead of the real ~5.8 (the pads let the jaw close ~1.5 units further
than the real current-limited servo; Joe's 0.5 N m gripper effort did not change it), and Joe's sim lifts more of
his own episodes than ours (different finger colliders: his convex decomposition vs our box pads, and his drive
gains 200/10 vs our servo model). Results: `results/joe_cal2_merge/`.

## Realism layers (all from data)
- **Servo** (A1, `servo_params.json`): per-joint first-order response K at 30 Hz, velocity limits, gravity bias,
  measurement noise. Implemented as a reference filter tracked by a stiff PD drive, so contacts stay physical.
  Step test: sim K within 3-11 % of the fitted K, 0.57-0.89 deg RMSE vs the model.
- **LeRobot `max_relative_target` clamp** (default 10 units/tick).
- **Base policy latency**: the chunk arrives `base_latency_s` of SIM time after its observation (SmolVLA OV 0.074 s,
  pi0.5 INT8 ~0.15 s, MolmoAct2 ~0.34 s); the plan is replaced from the current tick (the deployed controller's
  `none` blend; `crossfade_ticks` optional). New chunk requested when <= horizon + latency ticks remain (RTC rule).
- **Scene distributions** (A2): target brick where the gripper actually grasped in 740 episodes (clipped Gaussian),
  1-3 distractors, plates re-laid out per episode (the real layout changed every few dozen episodes), start pose
  distribution, inspection fingertip pose. Cameras: from the calibration profile (default `joe_cal2`); for the old
  profiles Joe's `camera-fit.json`; A2's refined camera is opt-in only (`camera_source="a2_refined"`, rejected by A2).
- **Gripper**: box finger pads at the measured inner faces (the USD's convex-hull finger colliders fill the gap and
  push bricks out; convex decomposition does not cook on this machine); units calibrated so the jaw stalls on a
  31.8 mm brick at the real hold value (~5.8): `GRIP_OFFSET_UNITS = 2.4` (old profiles); `joe_cal2` uses Joe's
  encoder scale with the bias re-solved from the measured pad stall (`calib.GRIP_STALL_RAD_CAL2`).
- **Perception for the residual**: simulated wrist-camera brick detector (true pose + 4 mm noise, 5 % dropouts,
  FOV/range gating); gripper load proxy (measured minus commanded gripper).

## Residual interface
Action (3,) in [-1, 1]: gripper +-15 units, wrist_flex +-5 deg, wrist_roll +-5 deg, added to the base target before
the clamp. Observation (45,): joint state, next-5-tick base targets (relative), load proxy, detector xyz + valid,
phase one-hot, time. Rewards: reach shaping, grasp, carried-lift, present (inspect FOV or recorded inspection pose),
place in the sampled bowl; penalties for time, residual size, jerk, clamp hits, drops.

## GB10 / Isaac Lab gotchas (fixed here)
1. Fabric scene delegate -> cameras render from the world origin (black RGB, 8 mm depth): launch with
   `--/app/useFabricSceneDelegate=false` (see `app.KIT_ARGS`).
2. The SO-101 USD's fixed `root_joint` + `replicate_physics=True` pins every clone to env_0 (blow-up on the first
   tick): `replicate_physics=False`.
3. `fix_root_link` fails on the referenced USD (Newton attributes); not needed (the USD root joint fixes the base).
