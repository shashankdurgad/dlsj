# DICE-RL grasp residual: how to run it on 8× RTX PRO 6000

The goal is a small residual that raises **first-attempt pick success** of the frozen MolmoAct2 8k fine-tune.
The task is pick-up only: the target brick is lifted ≥3 cm and held 0.5 s. Training runs in the Isaac Lab 3.0
digital twin, with the real 8k MolmoAct2 in the loop. The code is in `sim/twin/` (start with `sim/twin/README.md`
and `sim/twin/rl/README.md`).

## What the first ablation, (c), trains
- **Prior:** MolmoAct2 8k (`step008000.ckpt`). It runs once per chunk, K=1, with the deployed fixed flow noise.
  Prompts come verbatim from `finetune_cal2_456.json` ("pick up the <colour> brick"). The top camera is cropped to
  560×480. The controller is emulated as on the robot (max-step 4°/tick, gripper 8, one-tick delay).
- **Residual:** an MLP with an obs-only state (25-d): joints, the prior's plan, the gripper reading, and the brick
  from the wrist-camera YOLO through the fitted detector model and tracker. It does **not** use the prior's latent,
  which the XPS export can't provide.
  - Bounds: ±3.5 cm lateral, ±2 cm z, ±10° wrist, gripper.
  - Rate-limited, with a 6-tick fade. Active only on the pick prompt.
- **Learner:** DICE-RL (critic ensemble, Q-filtered BC weight 2) plus RLPD expert rows. The expert rows are built
  from the real post-train v3.1 demos: residual label = clip(demo − 8k prior chunk on the real frames).
- **Domain randomisation:**
  - Always on: physics (effort, fin stiffness, frictions, brick mass), the play / backlash model, and
    **deployment drift** (per-episode joint calibration error, N(0, 3°) on pan / lift / elbow and 1° on wrist_flex).
    The drift reproduces the real robot's first-close stats (~43 % first attempt, empty closes ~28 mm lateral).
  - Visual DR (sensor, lighting, spotlights) ramps from 0 to 0.5× over the first 500 episodes.
- **Scenes:** ready-pose start, bricks at the real deploy positions (`sim/twin/out/failure_match/real_brick_xy.json`),
  2–7 distractors with some near the target, and an 11 s horizon. A wrong-brick lift counts as a failure.
- **Eval:** the held-out pool is the 8k checkpoint's own `plan.json` validation episodes (44).
  - Each round runs K=1, deterministic, nominal, every 20 min: 40 nominal + 8 full-DR episodes.
  - Round 0 is the prior alone. Nominal baseline: first attempt 58 %, in-episode 69 %.
- **Sim defaults:**
  - `sim/twin/physics/params.json`: fin-ray B25C fingers at k 4.5, LDraw Duplo shell colliders.
  - `calib/profile.py`: W9 joint offsets; the top camera is the joe_cal2 pose.
  - Servo fit: `sim/twin/servo_fit/servo_params_fit.json`.

## Option A: a Linux box with 8× RTX PRO 6000 (Docker)
Isaac needs RT cores. RTX PRO 6000 / L40S / RTX work; H100 / A100 / B200 / H200 cannot render the cameras.

1. **Image.** Run `sim/twin/docker/build.sh`. It builds Isaac Lab 3.0 on isaac-sim 6.2.0, then the twin image
   `dlsj/twin:latest`, and takes about an hour. Docker needs the NVIDIA container toolkit with CDI
   (`--device nvidia.com/gpu=all`). The Dockerfile comments mention arm64 because it was first built on a GB10; the
   base is multi-arch.
2. **Robot USD.** `sim/twin/scripts/fetch_so101.sh`.
3. **Data** under `sim/twin/data/` (git-ignored). The private Hugging Face repos need access from Desmond:
   - `canonical/duplo_teleop_cal2_rgb_456` ← `akoniti/dlsj-canonical` (dataset)
   - `posttrain_v3_1` ← `akoniti/dlsj-posttrain-v3-1` (dataset)
   - `models/molmoact2_ft_8k/step008000.ckpt` ← `akoniti/dlsj-molmoact2-ft-8k` (model)
   - `hf/` (HF cache; `HF_HOME` defaults here): `allenai/MolmoAct2-SO100_101` at `152569fe`, plus `allenai/MolmoAct2`
   ```bash
   huggingface-cli login        # token with access to the akoniti repos; don't commit it
   huggingface-cli download akoniti/dlsj-canonical --repo-type dataset --local-dir sim/twin/data/canonical
   huggingface-cli download akoniti/dlsj-posttrain-v3-1 --repo-type dataset --local-dir sim/twin/data/posttrain_v3_1
   huggingface-cli download akoniti/dlsj-molmoact2-ft-8k --local-dir sim/twin/data/models/molmoact2_ft_8k
   HF_HOME=sim/twin/data/hf huggingface-cli download allenai/MolmoAct2-SO100_101 --revision 152569fe57914d97be91055800035f54e250d009
   ```
4. **Expert rows.** About 5 min on 8 GPUs; writes `sim/twin/out/rl/expert_img8k_prior.pt`:
   `sim/twin/rl/jobs/build_expert_local.sh`
5. **Train.** W&B is optional. Set `WANDB_API_KEY` in your shell and pass your own entity / project:
   ```bash
   WANDB_ENTITY=<entity> WANDB_PROJECT=dlsj-twin-dicerl sim/twin/rl/jobs/run_abl_c_local.sh   # 8 GPUs x 2 procs, 90 min cap
   ```
   To stop early, `touch /tmp/share_abl_c/STOP` inside the container, or tag the W&B run `stop`.
   Each eval round saves `sim/twin/out/rl/abl_c_r0/rounds/residual_roundNN.pt`.
6. **Export for the robot.** `sim/twin/run.sh sim/twin/rl/export_residual.py --ckpt <rounds/residual_roundNN.pt>`
   produces `deploy_*/residual_actor.npz` plus meta (numpy, 0.09 ms per call). The robot side is in `rl/DEPLOY.md`.

GPU sizing: each process uses ~40 GB (Isaac rendering + MolmoAct2 bf16). Two processes per 96 GB GPU run at
87–99 % utilisation; a 4-GPU run made ~9.6 episodes/min.

## Option B: Modal (Desmond's workspace)
`sim/twin/modal/fanout.py` snapshots the code from a box over SSH and bills Desmond's Modal workspace, so it only
works from his machine. The job file is `sim/twin/rl/jobs/abl_c_8g.yaml`; see `sim/twin/modal/README.md`.
Add `--include sim/twin/out/rl/expert_img8k_prior.pt`, because files over 2 MB under `out/` are not snapshotted
otherwise. 8-GPU RTX-PRO-6000 containers can queue for 10+ minutes.

## Gotchas
- Keep at most ~4 Isaac containers per GB10 box; more runs out of GPU graphics-context memory and Isaac hangs.
  On 96 GB cards, 2 processes per GPU is the tested setting.
- Never import PyAV inside an Isaac process: it blacks out the cameras.
- Eval rounds with MolmoAct2 in the loop are expensive; keep them ≤ 20 % of wall time.
- The deploy residual must be K=1, obs-only, `top_target false`. The robot loader refuses anything else.
