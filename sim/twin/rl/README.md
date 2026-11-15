# Grasp residual RL in the twin (DICE-RL, pick phase only)

A small learned residual that corrects a frozen base policy while it **picks up** a Duplo brick, trained with the
DICE-RL recipe (arXiv 2603.10263; the user's port in `~/Documents/dice-rl-wam*`, read-only) in the batched Isaac Lab
3.0 twin of the SO-101 station. Headline metric: **pick success** = brick lifted >= 3 cm above its rest height and
held >= 0.5 s (15 ticks) near the fingertip; centred grasp is secondary. Outside the pick prompt the base runs
unmodified (residual exactly 0, single base sample).

```
sim/twin/run.sh sim/twin/rl/check_kin.py                       # FK/IK vs PhysX, recorded lift timing, base batch
sim/twin/run.sh sim/twin/rl/evaluate.py --out baselines        # baselines over 512 held-out episodes (fixed seeds)
sim/twin/run.sh sim/twin/rl/expert.py                          # RLPD expert rows from posttrain_v3_1 (no Isaac)
NAME=dlsj-twin-rl setsid nohup sim/twin/run.sh sim/twin/rl/train.py --run dice_b2_rlpd --minutes 30 \
    --d_bc_weight 2 --d_batch 1024 --expert sim/twin/out/rl/expert_v31.pt > sim/twin/out/rl/dice_b2_rlpd.log 2>&1 &
sim/twin/rl/launch_wandb.sh RUN [train.py args] <<< "$WANDB_KEY"   # same, W&B on (key via stdin -> FIFO, never argv)
sim/twin/run.sh sim/twin/rl/evaluate.py --out X_eval --ckpt sim/twin/out/rl/X/residual.pt
sim/twin/run.sh sim/twin/rl/render_grasps.py --stage record --ckpt .../residual.pt --out X   # then --stage render
sim/twin/run.sh sim/twin/rl/molmo_eval.py --episodes 200 --K 1 --out molmo8k_base \
    --snap sim/twin/data/models/molmoact2_ft_8k/step008000.ckpt                     # visual prior alone (TwinEnv)
sim/twin/run.sh sim/twin/rl/train_image.py --run img8k --snap .../step008000.ckpt   # DICE-RL on the visual prior
sim/twin/run.sh sim/twin/rl/bench_prior.py [--snap ...]        # best-of-K cost (one prefix, K flow seeds), latent
sim/twin/run.sh sim/twin/rl/dist.py --role launcher --run d --env_gpus 0,1,2 --envs_per_proc 512   # actor/learner
sim/twin/run.sh -m pytest sim/twin/rl/test_gate.py sim/twin/rl/test_dice.py ; sim/twin/lint.sh sim/twin/rl
```
Outputs (git-ignored): `sim/twin/out/rl/<run>/{config.json, log.jsonl, curves.png, residual.pt, eval.json}`.

## Files
| file | what |
|---|---|
| `grasp_env.py` | `GraspEnv(ResidualEnv)`: batched grasp-only episodes (256-1024 envs, no cameras), base candidates with VLA-like errors, residual application, prompt gate + fade, rewards, outcomes; `make(GraspCfg)` = twin profile + twin physics hooks |
| `kin.py` | batched URDF FK + 2-step damped-least-squares fingertip IK (torch); matches PhysX to 0.4 mm / 0.03 deg |
| `gate.py`, `test_gate.py` | `is_pick_prompt(prompt)`: the residual gate (deterministic text filter), `Fader` |
| `dice.py`, `test_dice.py` | DICE-RL learner + n-step replay with MC returns (tests: n-step targets, MC returns, zero-init) |
| `rollout.py`, `evaluate.py`, `plots.py` | policies (base / scripted / DICE), fixed-seed evaluation, Wilson CIs, failure breakdown, residual size |
| `train.py`, `launch_wandb.sh` | single-process state-based training (resume, checkpoints, W&B hook) |
| `expert.py` | RLPD expert rows from the reviewed post-train set (pick segments) |
| `priors.py` | `Prior` interface, `ReplayErrorPrior`, `MolmoAct2Prior`, `LatentProjector`, the 8k fine-tune's prompts |
| `image_task.py`, `molmo_eval.py`, `train_image.py` | image path: TwinEnv episodes with the visual prior, its evaluation and DICE-RL |
| `visual_dr.py` | small visual DR (`dr_scale`, default 0.4) + separate spotlight knob (`spot_scale`), off / fixed / ramp |
| `prior_server.py`, `dist.py` | shared prior server (AF_UNIX) + actor / learner split for several Isaac processes / GPUs |
| `bench_prior.py`, `check_kin.py`, `diag_base.py`, `render_grasps.py` | measurements |
| `../policies/molmoact2.py` | + `MolmoAct2Client.act_k(obs, k, seeds)`: one prefix forward, k flow seeds batched, pooled latent (tests pass) |

## Design
**Episodes** (`GraspEnv`, current = "v3"): a random cal2 episode (`sim/twin/data/canonical/duplo_teleop_cal2_rgb_456`,
434 with a gripper-close frame by validate.py's rule; split 2:1 train/test, seed 0). The target brick sits at the FK
fingertip of the recorded close frame, perturbed by xy +-2.5 cm / yaw +-25 deg. Sampled per episode:
* **start type** (`start_mix`): near (0.7-1.2 s before the close) 30 %, approach (0.5-4 s before) 25 %, episode
  frame 0 15 %, perturbed (an approach state + +-7.5 deg per body joint, fingertip kept >= 3 cm above the table) 20 %,
  recovery (the jaw has just closed where the brick was, the brick 1.5-3 cm away, 30 % knocked on its side) 10 %;
  horizon = time to the recorded close + 1.5 s (<= 8 s), recovery 2.5 s;
* **target kind** 2x2 / 2x4 / curved 2x3 (LDraw 2302b) from `scene/scene_stats.json` target_kind (0.80 / 0.18 / 0.02)
  and **resting pose** (`pose_mix`, the grasp gate's 60 hand-labelled demos): up 35 %, upside-down 45 %, side 20 %
  (rolled about the body x axis: a 2x2 is squeezed across its 31.7 mm side), end 0 %; tilt metrics are relative to
  the resting pose; long kinds lie across the jaws;
* **clutter**: 0-7 distractors (2x2 / 2x4, count from scene_stats distractor_count, 0-6 in the data), near /
  touching the target (25 %), on a plate (41 %) or elsewhere on the table; three plates on the table (the residual
  env's layout sampler, clear of the target);
* **dynamics**: the fitted play model (`sim/twin/dynamics/play.py`: hold band / backlash, load sag, quantised
  encoders, per-env DR) replaces the Gaussian measurement noise; LeRobot clamp 10; twin physics hooks (fin jaws,
  rigid contact); twin profile = the fitted joint offsets W (`calib/profile.py`).
The calibration profile is a parameter (`--profile`, `--joint_offsets_json`). Integration hooks:
`GraspEnv.detector_model(env, rel) -> (det, valid)` for the perception agent's occlusion-aware detector and
`GraspEnv.goal_hook(env, goal) -> goal`. Earlier result blocks below used "v1" (near starts only, 3 s, no clutter,
flat 2x2 80 % / 2x4 / curved 2x2, joe_cal2 offsets, Gaussian noise) or "v2" (v1 + W offsets + poses).

**Base = `ReplayErrorPrior`**: the recorded actions from that start with per-episode VLA-like errors: task-space
xyz offset (sd 1 cm xy / 6 mm z, through the IK), wrist-roll error +-8 deg, close-timing shift +-5 ticks, 15 %
"hover" (stops 2-4 cm above), and per-candidate spread (xyz 4 mm, close timing +-2 ticks, action noise 0.3 deg).
The K candidates of a decision are K such draws (the MolmoAct2 analogue: K flow-noise seeds).

**Residual** (decision every 5 ticks = 6 Hz, applied before the clamp + servo model): offset targets of the fingertip
in the gripper frame (+-2.5 cm), wrist flex / roll (+-10 deg) and gripper (+-30 units); the applied offset moves
toward them at <= 5 mm / 3 deg per tick and is turned into pan / lift / elbow by the IK (wrist corrections pivot
about the fingertip). Gate: `pick_phase = is_pick_prompt(prompt)`; at the hand-off (held 3 cm lift, or the
controller's prompt switching) the offset fades linearly to 0 over 6 ticks and best-of-K stops.

**Observation** (25, all available on the robot): measured joints, wrist-detector brick centre in the gripper frame
(FK of the measured joints; true pose + 4 mm noise, 5 % dropouts, FOV / range gate), valid flag, load proxy (measured
- commanded gripper), time, applied offset, previous residual, pick flag. The critic sees the same (no privileged
critic: best-of-K must run the critic on the robot). + the first 4 candidates' next-5 targets (pi0.5 template: state
= latent + all K chunks); the residual is conditioned on the candidate it corrects.

**Reward**: +1 at success, +0.3 if the hold was centred (brick centre |y| < 5 mm across the finger width, at pad
height, tilt < 15 deg for >= half the hold), potential shaping on lift height (0.2) and grasp (0.1), -0.3 push
(> 1.5 cm before any grasp), -0.5 tip (> 35 deg on the table, terminal), -0.01 per tick of pad-table contact, -0.1 per
re-open after the close started, -0.01 |r|^2 and -0.02 |r - r_prev|^2 per decision.

**Learner (DICE-RL, `dice.py`)**, following the pi0.5 LIBERO config: zero-initialised residual MLP (GELU +
LayerNorm, tanh-bounded) conditioned on the base action; 10-critic ensemble (min); n-step 3, gamma 0.99,
Polyak 0.01; actor and target every 2nd critic step; Adam 1e-4, wd 1e-5, cosine restarts (1000 / 10 / 1e-6), grad
clip 1; actor loss -Q / mean|Q| over online rows + bc_weight * ||r||^2 with the soft Q-filter (threshold -0.55 on
Q(s, a_stored) - MC return, after 3000 steps); best-of-K with max_q_min over 16 candidates at collection (uniform
for the first 100k rows); RLPD expert mixing 0.5 -> 0.1 over 10k steps, Q loss off on expert rows. Deviations:
networks [256]x3 (low-dim inputs; [1024]x3 in the reference), the 4 stored candidates replace 16 fresh noise samples
in the actor loss (the base cannot be re-sampled offline), Gaussian exploration noise 0.05 on the residual, 20 grad
steps of batch 1024 per vector step (512 envs x 1 decision: UTD 0.04 steps/row, every row replayed ~40 times),
**truncation is terminal** (the observation carries t/T, so the state at t = T never starts a row; bootstrapping
there made Q drift to -1.6 against MC returns of -0.08 in the smoke run; `--trunc_bootstrap 1` switches it back).

**Expert rows** (`expert.py`): the post-train set v3.1, pick segments of the 441 cal2 demos (only those whose source
episode is in the TRAIN split: 280) and the 130 pick-only takes, 4 starts each (21/26/31/36 ticks before the close):
20,302 rows from 410 episodes, mean return 1.17. Same features as online rows, computed without Isaac from the
recorded joints with the twin FK (brick at the close-frame fingertip, carried by the gripper after the close;
detector noise model on top; load proxy from the recorded gripper state vs command), a_base = the demo chunk, r = 0.
For the visual prior the same rows additionally need the prior's latent and K candidates on the REAL frames
(`MolmoAct2Client.act_k` on the dataset's wrist/top frames); that adds a real-image vs sim-image feature gap.

## Results (state-based, `ReplayErrorPrior`, held-out test episodes, 512 per row, fixed seeds)
Wilson 95 % intervals. All rows of one block share episodes, brick poses and base error draws; GPU PhysX is not
bit-deterministic, so the same "base" row varies by ~1-2 points between blocks.

Baselines (`baselines/eval.json`):
| policy | pick success | centred | missed / pushed / slipped / tipped |
|---|---|---|---|
| base, no errors, brick exactly at the recorded fingertip | 19.0 % [15.8, 22.6] | 6.1 % | 57 / 20 / 0.4 / 3.9 % |
| base, no errors (brick +-2.5 cm / +-25 deg) | 9.6 % [7.3, 12.4] | 1.6 % | 62 / 22 / 0.2 / 6.3 % |
| base, with the VLA-like errors | 5.5 % [3.8, 7.8] | 0.4 % | 66 / 20 / 0.6 / 8.0 % |
| scripted align-on-detector-then-close (+ errors) | 3.5 % [2.2, 5.5] | 0.2 % | 43 / 40 / 0.6 / 13 % |

Trained residuals (each block: its own base row, then the residual; `<run>/eval.json`):
| run (budget) | base | residual, best-of-1 | sel. only, best-of-4 | residual + best-of-4 | residual + best-of-16 | best-of-1, detector ablated | mean abs residual (tip xyz mm / wrist deg / grip units) |
|---|---|---|---|---|---|---|---|
| `dice_b10` (bc 10, 25 min) | 5.1 % | 10.2 % [7.8, 13.1] | 6.3 % | 12.7 % | 12.5 % | 5.1 % | 0.8 0.6 0.9 / 0.3 0.3 / 1.2 |
| `dice_b2` (bc 2, 25 min) | 5.9 % | 15.6 % [12.7, 19.0] | 6.6 % | 17.0 % | 18.4 % | 5.1 % | 1.9 1.8 2.0 / 0.6 0.7 / 3.2 |
| `dice_b2_rlpd` (bc 2 + experts, 30 min) | 7.8 % | **20.1 % [16.9, 23.8]** | 6.6 % | 19.0 % | **24.6 % [21.1, 28.5]** | 6.8 % | 1.7 3.1 2.1 / 1.1 1.1 / 4.1 |
| `dice_b05_rlpd` (bc 0.5 + experts, 40 min, GPU shared: 0.49 M rows) | 6.8 % | 18.4 % [15.3, 21.9] | 7.2 % | 20.3 % | 20.7 % | 11.1 % | 2.8 3.7 4.1 / 1.4 1.8 / 6.3 |

* The residual roughly **triples** pick success over the base with errors (5.5-7.8 % -> 15.6-20.1 % best-of-1, 24.6 %
  with best-of-16) while staying small (mean |offset| 2-3 mm, ~1 deg, ~4 gripper units). It also exceeds the
  error-free replay (9.6 %) and gets close to the replay with the brick exactly at the recorded fingertip (19 %).
* It **uses the detector**: zeroing it drops success back to the base (5.1-6.8 %).
* Best-of-K selection alone (residual 0) does not help (6.6 %): with the replay prior, the candidates differ by
  4 mm / 2 ticks, much less than the episode's systematic error; selection only pays together with the residual.
* BC weight sets the size / gain trade-off: bc 10 -> 10 %, bc 2 -> 16 % (20 % with RLPD); bc 0.5 doubles the
  corrections (3-4 mm, 6 gripper units) without more success in its (contended, 0.49 M-row) budget and leans less on
  the detector (ablated: 11 %). RLPD expert rows add ~4 points at bc 2. Recommended: bc 2 + RLPD.
* Failures stay dominated by **missed** (jaw closes without the brick: 54-66 %): the twin's grasp-depth gap below.
* Throughput (GB10): 512 envs, 64 solver iterations: ~7,900 env ticks/s physics alone on an idle GPU (jarvis),
  ~2,200-2,600 on a shared one (watson); with the learner (20 x batch 1024 per vector step) 3,100-3,600 wall ticks/s
  idle / 1,300 shared. 25 min = 0.4-0.8 M decision rows (2-4 M env ticks, 10-25 k episodes).
* Curves: `dice_*/curves.png` (training success, return, critic, residual RMS / Q-filter).

**v2 (W offsets + brick poses up 55 / upside-down 40 / side 3 / end 2 %, still near starts, no clutter)**, 512
held-out episodes (`baselines_W/eval.json`, `dice_W_b2_rlpd/eval.json`; the W twin replays the demos better, so the
base is stronger and there is less systematic error left to fix):
| policy | pick success | per pose (up / upside-down / side / end) |
|---|---|---|
| base, no errors, brick exactly at the recorded fingertip | 32.8 % [28.9, 37.0] | 34 / 30 / 42 / 33 % |
| base, no errors | 18.2 % [15.1, 21.7] | 18 / 18 / 8 / 22 % |
| base, with errors | 13.7 % [11.0, 16.9] | 12-13 / 15-16 / 8 / 22-33 % |
| scripted align-then-close | 8.4 % [6.3, 11.1] | |
| `dice_W_b2_rlpd` (bc 2 + RLPD, 30 min on a shared GPU: 0.20 M rows), best-of-1 | 19.0 % [15.8, 22.6] | 17 / 20 / 25 / 33 % |
| same, best-of-4 / best-of-16 | 20.9 % / 21.3 % [18.0, 25.0] | |
| same, best-of-1 detector ablated / selection only (best-of-4, r = 0) | 16.2 % / 15.2 % | |
Mean |residual| 1.1-1.7 mm tip, 0.4 deg wrist, 1.8 gripper units. +5 to +8 points from a quarter of the v1 run's data.

**v3** (`--preset v3`: start types, 0-7 distractors + plates, play model, W offsets, fin jaws; 512 held-out
episodes, `baselines_v3/eval.json`):
| policy | pick success | missed / pushed / slipped / tipped | near / approach / frame 0 / perturbed / recovery | 1 / 2-3 / 4-5 / 6-8 bricks |
|---|---|---|---|---|
| base, no errors | 23.1 % [19.6, 26.9] | 29 / 28 / 4 / 16 % | 30 / 24 / 17 / 24 / 7 % | 23 / 24 / 20 / 25 % |
| base, with errors | 15.4 % [12.6, 18.8] | 36 / 29 / 4 / 15 % | 17 / 20 / 6 / 21 / 4 % | 19 / 16 / 13 / 0 % |
| scripted align-then-close | 10.2 % [7.8, 13.1] | 32 / 40 / 3 / 16 % | 12 / 15 / 9 / 9 / 0 % | 9 / 10 / 12 / 0 % |
(start-type n = 156 / 117 / 91 / 103 / 45; brick-count n = 47 / 355 / 102 / 8.) Frame-0 and recovery starts are the
hard ones for the replay base (it cannot react). "Tipped" rises to 15-16 %: side / upside-down bricks and clutter.
Clutter / play on vs off with the same base: v2 (off, near starts only) 13.7 % vs v3 near starts 17.3 % (with errors);
the per-factor split is being run by the grasp gate (it pins `--play 0 --distractors 0 --plates 0`).

Image: `out/rl/dice_b2/grasps_wrist.jpg` (base vs residual on the same episodes, played back in TwinEnv; the plates /
distractors in those frames are the twin's visual layout, not part of the recorded physics).

## Results with the visual prior (MolmoAct2 8k fine-tune, TwinEnv, cameras)
`image_task.py` protocol (same held-out starts, brick at the close-frame fingertip +-2.5 cm / +-25 deg, 2x2 recoloured
to the trained colour word, 3 s, RTC horizon 15 with a 10-tick inference delay, clamp 10, DR off), `molmo_eval.py`:

| prior (twin version) | episodes | pick success | centred | missed / pushed / slipped / tipped | prior latency (GB10 torch bf16, K=1) |
|---|---|---|---|---|---|
| 8k fine-tune alone (v1 twin: joe_cal2 offsets, legacy bricks, near starts, 3 s) | 170 | **47.6 % [40.3, 55.1]** | 16.5 % (69 % of successes not centred) | 25.3 / 11.8 / 11.8 / 3.5 % | 0.90 s median per chunk |
| 8k fine-tune alone (v3 twin: W offsets incl. the arm drive, brick pool, start types, Modal) | 80 | 22.5 % [14.7, 32.8] | 11.3 % | 30 / 22.5 / 22.5 / 2.5 % | |
| 8k + DICE residual (35 min on 1 GPU), best-of-1, same seeds | - | not measured (see above) | | | |

Offsets used by each image-path run: the 170-episode baseline ran before the W profile landed (joe_cal2 for the drive
and for the rl FK: consistent). Runs between 1123cfa and 58fea2d (TwinEnv still drove the arm with joe_cal2 while the
twin profile, used by `image_task`'s FK for the brick placement / detector / residual IK, was already W) are
inconsistent and are NOT reported: the jarvis `img8k_b2` training (60 episodes, success 27 % with exploration;
stopped) and the first Modal smoke (`rl-smoke`, plumbing only). The v3 rows above use 58fea2d+ (W everywhere).
**Do we need the wrist detector?** Same 8k prior, v3 twin, start / brick mix and eval seeds (4321, 100 episodes,
best-of-1) for every arm; each residual trained 35 min on one RTX-PRO-6000 with the 710 image expert rows:
| arm | state of the residual | pick success | added cost per decision on the XPS |
|---|---|---|---|
| (d) prior alone | - | 22.5 % [14.7, 32.8] (n = 80, sampled starts, 3-8 s; run `rl-arms2`) | 0 (MolmoAct2 INT8 0.34 s per chunk, as today) |
| (a) prior features only | obs without detector + projected prior latent | not measured: the training jobs were stopped (other runs' cleanup / deadline) | residual 0.04 ms + projection 0.05 ms; the export must also output the pooled prefix latent |
| (b) fitted detector | obs + detector (nominal; conservative in training mix) | not measured: the training jobs were stopped (other runs' cleanup / deadline) | wrist YOLO ~4 ms NPU / 10 ms CPU per frame (own thread) + residual 0.04 ms |
| (c) features + detector | obs + detector + latent | not measured: the training jobs were stopped (other runs' cleanup / deadline) | both of the above |

The visual prior is far stronger than the replay base (47.6 % vs 13.7 %), so DICE-RL has to raise it from ~50 %.
Wall time per image episode ~20-25 s (prior inference dominates; the GB10 is shared) -> ~150 episodes / hour /
station. The 8k model loads in 340-370 s (22 GB fp32 Lightning ckpt with unmerged LoRA; 40 s when page-cached).
Black renders appeared once under heavy GPU-memory pressure from other jobs; `image_task.request` re-renders and
`molmo_eval` excludes any episode whose prior saw a black frame (none in the numbers above).

**Failure mix vs the real arm (8k prior, a realism criterion).** Real 8k loop: 64 % of the closes were empty
("closes on nothing", too early / high / off-centre). Twin, 8k prior alone (v1, 170 episodes): of the 89 failures
48 % missed (empty close), 23 % pushed, 23 % slipped, 7 % tipped; i.e. the twin under-produces empty closes and
over-produces pushes / slips relative to the real arm. The arms run (below, current defaults) reports the mix again.

## Twin gaps that limit these numbers
* **Grasp depth**: in the replay the sim fingertips are 1-1.5 cm higher than the real fingers were (physics
  README); most failures are "missed" = the jaw closes above / beside the brick. The residual learns to lower the
  tip and re-centre, which partly compensates a calibration / sag gap rather than a policy error. The joint_fit refit
  (`--joint_offsets_json`) should be re-run through `evaluate.py` and the residual retrained when it lands.
* Bricks held by their top 5-13 mm swing in the jaw (mean tilt 18-32 deg while lifted), which is why "centred"
  stays low (1-5 %); `ResidualEnv.grasped()` is true on only ~7 % of held ticks (its gripper-velocity / squeeze
  thresholds), so the centred metric and the shaping use geometry instead.
* LeRobot clamp 10: the error-free replay succeeds 27 % with the clamp off vs 12 % with it on (diag_base.py); the
  real controller clamps, so it stays on.
* No distractors / plates in the grasp episodes; box pads; solid-box brick inertia.

## Deployment on the robot
**What the wrist detector must output** (per frame, the residual's only perception): the target brick centre in the
gripper_link frame (m, relative to the fin-ray grasp point: the checkpoint / export meta `tip_local`, today
physics.apply.tip_local() = (0.0063, 0.001, -0.0902); the old pad tip (0.008, 0, -0.090) is NOT the deploy spec) + a
valid flag. The team's
one-class YOLO (`dlsj/inspect/brick_detector.py`, OpenVINO) gives axis-aligned boxes; convert with the wrist camera
intrinsics + mount (calib profile) by intersecting the box-centre ray with the table plane (z = table + 9.6 mm) at the
FK pose of the measured joints, then express in the gripper frame. The residual was trained with 4 mm noise and 5 %
dropouts; the detector needs wrist-camera training data (it was built for the top / inspect views), a latency under
one decision (<= 33 ms at 30 Hz, it runs every 5 ticks), and target selection when several bricks are visible
(`select_target` with the commanded colour). No orientation is needed.

**Composition with the controllers** (`dlsj/deploy/pi05_control.py`; the MolmoAct2 RTC controller in
`deploy/smolvla/smolvla_control.py` + `deploy/molmoact2/rtc_patch.py`):
1. Gate: `from sim.twin.rl.gate import is_pick_prompt, Fader`; `scale = fader.update(controller.prompt())` every
   tick. With the 8k fine-tune the pick prompt is `pick up the <colour> brick` (exact trained strings only).
2. Residual (every 5 ticks while scale > 0): obs as above from the measured joints, detector, load
   (measured - commanded gripper), time since the pick prompt started, the residual's own applied offset / last
   output, and the next 5 planned targets of the current plan (`_planned(t..t+4)`); output r -> offset targets ->
   rate limit -> IK (`kin.py`, profile offsets) -> `target = planned + scale * offset` inside `tick()` BEFORE
   `_step()` (max-step 4 deg clamp, sync band) and the follower's max_relative_target, i.e. where the sim applies it.
3. Best-of-K (optional, only while the pick prompt is active): in `_request()`, sample K chunks from ONE prefix
   forward (`act_k`), score each with min over 2-3 critic heads of Q(s, chunk[d:d+5], pi(s, chunk[d:d+5])) and add
   the best; otherwise the normal single sample.
4. Hand-off: when the controller switches to `bring the brick to the inspection camera` (operator mark or held
   lift), `Fader` takes the residual to 0 over 6 ticks; nothing else changes.

**Cost on the XPS (budget <= 50 ms per chunk in the pick phase)**: residual MLP [256]x3 ~0.04 ms, residual +
best-of-4 with all 10 critic heads ~0.35 ms, with 2 heads ~0.15 ms (numpy, 4 laptop threads; the GB10 torch
numbers in `bench_cpu.json` are dominated by framework overhead on a shared CPU), the latent projection 2048->128
~0.05 ms. So **residual-only, best-of-1** adds < 1 ms per decision + the detector. Best-of-K adds the K-1 extra flow
loops of the action expert. Measured with the 8k fine-tune in torch bf16 on the (shared) GB10
(`bench_prior_8k.json`, one real cal2 frame): one call = 1.05 s (prefix 0.58 s + 10 flow steps); K = 2 / 4 / 8 from
ONE prefix: 0.96 / 1.04 / 1.10 s, i.e. best-of-4 costs ~0-5 % more on a GPU (the batched flow is latency-bound). On
the XPS (OpenVINO INT8, 0.34 s per chunk, CPU / iGPU) the K-1 extra flow loops are closer to linear: with the flow
at ~40-45 % of the chunk (the GB10 split) best-of-4 would add ~0.15-0.45 s per request, well over the 50 ms budget,
so **residual-only, best-of-1 is the XPS mode**; best-of-K only if the action expert runs on a GPU / NPU at batch K.
Candidate spread of the 8k prior (first 15 actions): 0.2-0.9 deg per joint, 0.2-0.7 deg with RTC; RTC keeps the
first 10 actions within 1.0 deg of the previous tail. The pooled prefix latent is 2560-d. The current OpenVINO export (`deploy/molmoact2/export.py`) is ONE graph with
the 10 flow steps unrolled and the flow started from zeros ("deterministic per input", README), so on the robot
today there is no noise to sample and no way to reuse the prefix: best-of-K needs (a) a noise input on the graph
and (b) a split export: prefix graph (images + text + state -> per-layer KV context + pooled latent) and an action
expert graph (context, noise (K, 30, D), [RTC inputs] -> K chunks), so K costs one prefix + K batched flow loops.

**Real-robot fine-tune (off-policy, plausible later)**: the learner is off-policy and the critic uses only robot
observations, so a short on-robot phase can reuse the sim-trained actor + critics (`--init`) and the replay format:
log (obs, plan feature, residual, reward) at 6 Hz in the pick phase; reward from a lift check (gripper load +
FK height, or the operator's mark) and a centred check from the wrist frame; a few hundred pick attempts at
UTD ~1 are within reach of DICE-RL's sample efficiency, with the BC weight raised for safety and the residual bounds
kept (2.5 cm / 10 deg / 30 units, 5 mm per tick).

## Deploy artefact and spec (XPS: one MolmoAct2 forward per chunk, residual = actor mean, no critic)
`python sim/twin/rl/export_residual.py <run>/residual.pt [--onnx]` -> `<run>/deploy/residual_actor.npz` (+ `.onnx`,
`export.json`): the actor MLP only, with the checkpoint's own residual scale / rates / fingertip point / GraspCfg
(export refuses checkpoints without them; the full robot spec is `DEPLOY.md`); `ResidualPolicy(npz)(state, plan)` is numpy-only (exact-erf GELU), matches torch
to 2e-7 and takes 0.09 ms per call (`dice_W_b2_rlpd`: [145 + 30] -> 256 x 3 -> 6). Every reported number in the
"deploy" rows is K = 1 prior sample + this deterministic output; best-of-K rows are labelled not deployable.

Inputs, every 5 control ticks (6 Hz) while the gate is open:
* `state` = obs (25), all from the robot: measured joints / 100 (LeRobot degrees of the cal2 frame, gripper 0-100);
  wrist-detector brick centre minus the fin-ray grasp point (meta `tip_local`, today (0.0063, 0.001, -0.0902) m) in
  the gripper_link frame
  x 10 (m) and its valid flag (`sim/twin/perception/wrist_target.py` `TargetTracker` + `box_to_rel` on the
  `OVBrickDetector` boxes; zeros when not detected); load = (measured - servo reference of the gripper) / 10; time
  since the pick prompt / 240 ticks; the residual's applied offset / scale (6); the previous residual output (6); 1.
  State-path models append the first 4 base candidates' plan features (best K = 1 deploy: the same plan 4 times);
  image-path models append the prior's pooled prefix latent x the fixed projection (`LatentProjector(2560, 128,
  seed 0)`).
* `plan` (30) = the next 5 planned goals of the controller's current plan minus the measured joints, / 20.
Output r in [-1, 1]^6 -> offset targets: fingertip xyz in the gripper frame +-2.5 cm, wrist_flex / wrist_roll
+-10 deg, gripper +-30 units. Applied offset moves toward the target at <= 5 mm / 3 deg per tick; xyz + wrist go
through the 2-step damped-LS IK (`kin.py`, the profile's joint offsets) onto pan / lift / elbow, the gripper offset
is added and clipped to 0-100; then the controller's own max-step (4 deg) and the follower's max_relative_target.
Gate: `is_pick_prompt(prompt)` (exact trained pick prompts pass); when it closes (bring prompt / held lift) `Fader`
takes the offset to 0 over 6 ticks. Runtime cost < 1 ms per decision + the wrist YOLO (~4 ms NPU) if the detector arm
is used.

## Scale-out (Modal / several GPUs)
`dist.py`: launcher -> one learner (owns the replay, updates, checkpoints to `--out_root`, `--resume`, W&B with
`--pylibs`) + N actor processes (`--env_gpus 1,...,7 --learner_gpu 0`, `--envs_per_proc`), each an Isaac process
with a batched GraspEnv streaming decision rows over AF_UNIX and reloading the published weights. `prior_server.py`
hosts one MolmoAct2 for several image-path actors (`RemotePrior`); the image path's actor / learner split is not
wired yet: `train_image.py` is one station + learner per process (shared-learner image actors are the next step).

Modal runs go through the fan-out tool (`sim/twin/modal/fanout.py`: snapshot of watson's tree, data Volumes
dlsj-twin-data / dlsj-twin-hf, outputs back to `watson:sim/twin/out/modal_fanout/<run>/`, app stopped at the end),
always `MODAL_PROFILE=desmond-zee uv run --no-project --with modal --with pyyaml python sim/twin/modal/fanout.py ...`.
wandb is not in the image: each job pip-installs it into /tmp/pylibs (`--pylibs`); the key comes from the
`dlsj-wandb` secret.

Smokes (1x RTX-PRO-6000):
* `rl-smoke`: `train_image.py` with the 8k prior (K = 4) + residual + image expert rows, 6 min: 53 episodes,
  894 decision rows (~4.5 k env ticks), 862 gradient steps, **~12 env ticks/s per GPU including the prior** (6.8 s per
  3 s episode; 20-75 s on the shared GB10), W&B run logged, checkpoint written
  (`watson:sim/twin/out/modal_fanout/rl-smoke/`). Container 470 s, **~$0.51** (+ image pull). Nothing broke.
* `rl-dist-smoke`: `dist.py` launcher, learner + 1 actor (256 envs) on one GPU, 4 min: actor 1,650 env ticks/s,
  learner busy 10 %, ~$0.30. The plumbing worked but the critic losses were NaN: the v3 clutter placed distractors
  inside the target (PhysX launched them). Fixed after the smoke (footprint-aware placement, slots dropped instead of
  overlapping, non-finite observations / rows sanitised, blown-up envs end as "lost"); needs one more smoke.

Full runs (launch only after the sim is approved; the fan-out refuses > $15 without `--yes-cost`):
```
U="uv run --no-project --with modal --with pyyaml python sim/twin/modal/fanout.py"   # MODAL_PROFILE=desmond-zee
PIP='/opt/isaaclab-venv/bin/python -m pip install -q --target /tmp/pylibs wandb==0.31.0 >/dev/null 2>&1'
WB='--wandb_project dlsj-twin-dicerl --wandb_entity sfb5s44zhz-university-of-cambridge --pylibs /tmp/pylibs'
# image path (8k prior): 8 independent stations / learners, one per GPU (seeds), 50 min each
$U run --name img8k-x8 --grid seed=0,1,2,3,4,5,6,7 --max-containers 8 --wide --secret dlsj-wandb --timeout 3600 \
  --memory 65536 --est-min 55 --yes-cost --cmd "bash -c \"$PIP; /workspace/isaaclab/isaaclab.sh -p \
  sim/twin/rl/train_image.py --run img8k_s{seed} --seed {seed} --minutes 50 --K 4 --bc_weight 2 \
  --snap sim/twin/data/models/molmoact2_ft_8k/step008000.ckpt --expert sim/twin/out/rl/expert_v31_img8k.pt $WB\""
# state path: learner on GPU 0, 7 actors x 512 envs in one 8-GPU container
$U run --name dist-x8 --gpu RTX-PRO-6000:8 --cpu 64 --memory 262144 --secret dlsj-wandb --timeout 3600 --est-min 40 \
  --yes-cost --cmd "bash -c \"$PIP; /workspace/isaaclab/isaaclab.sh -p sim/twin/rl/dist.py --role launcher \
  --run dist_x8 --learner_gpu 0 --env_gpus 1,2,3,4,5,6,7 --envs_per_proc 512 --minutes 30 \
  --expert sim/twin/out/rl/expert_v31_W.pt $WB\""
```
Cost: 300 k env ticks on the image path = 300 k / (8 x 12 ticks/s) ~ 52 min on 8 GPUs ~ 8 x 0.9 h x $3.03 +
CPU / memory ~ **$22-28**. The state path is cheap per tick (7 actors x 1.6-7 k ticks/s: 300 k ticks in ~1 min;
a useful 5 M-tick run ~10-30 min ~ $8-15 on 8 GPUs). Camera pipeline: train without the sensor agent's H.264 codec
round-trip (75-330 ms per 3-camera step) and evaluate with it; the flag is the sensor agent's (`sensors/pipeline.py`).
