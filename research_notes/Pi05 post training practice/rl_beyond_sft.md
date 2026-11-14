# Post-training π0.5-class VLAs beyond SFT (as of Nov 2026), and what a hackathon team can actually do

Scope: methods that improve a fine-tuned flow-matching VLA (π0 / π0.5 / SmolVLA-class) after supervised fine-tuning. Practicality is judged for a 1–2 day hackathon with one real SO-101 arm and cloud GPUs.

Caveat on sourcing: pi.website (the blog and the π*0.6 PDF) returned HTTP 429 during this session. π*0.6 facts therefore come from the arXiv abstract page and third-party implementations. Per-task numbers from the PI paper are listed under Gaps rather than quoted.

## 1. π0.6 / π*0.6 and RECAP

### Takeaway
RECAP is an offline-RL-style recipe, not online policy-gradient RL. It works in four steps:
1. Train a value function that predicts progress to success.
2. Use it to compute per-action advantages.
3. Binarize the advantages into an "improvement indicator" token.
4. Train the policy conditioned on that token, then condition on "good" at inference, optionally with CFG-style guidance.

PI reports that it more than doubles throughput and roughly halves failure rate on its hardest tasks. PI has not released official code. RLinf has a working community pipeline for π0.5 in LeRobot format, and this is the most transferable "beyond SFT" idea for a hackathon because it needs no online RL infrastructure.

### Cited Findings
- **Paper:** "π*0.6: a VLA That Learns From Experience", arXiv 2511.14759, Physical Intelligence et al., Nov 18, 2025. — [arXiv](https://arxiv.org/abs/2511.14759)
- **Data sources:** RECAP ("RL with Experience and Corrections via Advantage-conditioned Policies") combines three kinds of data:
  - demonstrations
  - on-policy autonomous rollouts
  - expert teleoperated corrections made during autonomous runs

  The model is pre-trained with offline RL, then specialized per task through on-robot data collection. — [arXiv abstract](https://arxiv.org/abs/2511.14759)
- **Results:** Tasks were laundry folding in real homes, box assembly, and espresso preparation on a professional machine. On some of the hardest tasks RECAP "more than doubles task throughput and roughly halves the task failure rate." — [arXiv abstract](https://arxiv.org/abs/2511.14759)
- **Mechanism (from the PI PDF, via search snippet):** Train a value function that evaluates progress toward successful completion. Use it to estimate the advantage of each action in the dataset. Condition the policy on an improvement indicator derived from that advantage. — [PI PDF](https://www.pi.website/download/pistar06.pdf)
- **RLinf's RECAP pipeline (offline, "no online environment interaction"), four steps:**
  1. Compute discounted returns per trajectory. Reward is -1 per step, failed rollouts get a terminal penalty (default -300), and γ = 1.0.
  2. Train a value model by SFT. It uses a SigLIP2-so400m vision encoder, Gemma3-270M, and a critic expert, and outputs a 201-bin categorical over normalized return in [-1, 0].
  3. Compute N-step advantages and label the top 30% as positive.
  4. Train π0.5 (OpenPI, PyTorch) with CFG: positive samples are the conditional input and negative samples the unconditional one. A guidance scale is applied at inference.

  Data is in LeRobot format. It mixes SFT datasets (successes) with rollout datasets that may include failures. — [RLinf RECAP docs](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/recap.html)
- **RLinf RECAP result:** On LIBERO-10 Task 0 with a few-shot π0.5 baseline, one RECAP iteration raised success from 48.8% to 66.5% (+17.7 points). The docs warn that this is a narrow setup. — [RLinf RECAP docs](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/recap.html)
- **Community re-implementations (all unofficial and experimental):**
  - hzm8341/pi0.6 injects positive and negative advantage tokens into the prefix and computes both conditioned and unconditioned losses. — [GitHub](https://github.com/hzm8341/pi0.6)
  - liudi159/pi05-recap is deployment-only. It explicitly does not implement the full model, the KI objective, or online HITL, and does not validate the paper's results. — [GitHub](https://github.com/liudi159/pi05-recap)
  - Others: [qrafty-ai/pi-openpi](https://github.com/qrafty-ai/pi-openpi), [GordonLuQwQ/PI0.5-RECAP](https://github.com/GordonLuQwQ/PI0.5-RECAP), [giaoleng/openpi-recap-sim](https://github.com/giaoleng/openpi-recap-sim), [Knight1112D/CBC_Pi0.7_Openpi](https://github.com/Knight1112D/CBC_Pi0.7_Openpi)
- **Official release status:** Search found no official RECAP code or π*0.6 weights in PI's openpi. One fork's claim that a π0.6 checkpoint is "coming soon" is unverified. — [search over openpi forks](https://github.com/Knight1112D/CBC_Pi0.7_Openpi)

### Inferences
- **Why it suits a small team:** RECAP is essentially SFT with an extra conditioning token plus a small value model. Training cost is close to an ordinary π0.5 fine-tune plus value-model training.
- **Hackathon recipe:**
  1. SFT π0.5 on demos.
  2. Run 30–100 autonomous rollouts on the SO-101, with a human taking over via the leader arm when it fails.
  3. Label each episode success or fail.
  4. Train the value model and compute advantages.
  5. Retrain π0.5 with advantage conditioning (marking human-correction segments positive, as in the paper's spirit).
  6. Deploy with the indicator set to "good".

  One iteration is plausible in a day if the infrastructure is pre-built. Two or more iterations is unlikely.
- **Main risk:** The RLinf pipeline targets OpenPI π0.5 (LIBERO/Franka configs). SO-101 6-DoF joint actions need a new data config and normalization stats.

### Gaps
- PI's exact per-task throughput and success numbers, data volumes per iteration, how the value model was initialized, the threshold choice, and comparisons with AWR/PPO baselines could not be retrieved (pi.website returned HTTP 429). Fetch https://www.pi.website/blog/pistar06 or the PDF directly to fill these in.
- Found no report of RECAP applied to SO-100/101 or LeRobot's own π0.5 port.

## 2. Online RL fine-tuning of flow-matching VLAs (πRL/RLinf, ReinFlow, FPO, DSRL, PA-RL, SimpleVLA-RL, VLA-RL, GRPO, residual RL)

### Takeaway
- **Sim-centric methods:** πRL (in RLinf) is the mature, open-source route to online PPO/GRPO on π0/π0.5. It reports large simulation gains (LIBERO π0.5 77.1% → 97.9%), but it needs GPU-parallel simulators and multi-GPU nodes.
- **Real-robot methods:** DSRL is the most practical for real-robot, black-box improvement of π0. It runs SAC over the input noise with a tiny actor and leaves the VLA frozen. PA-RL and ConRFT are the other real-robot-proven methods.
- **Token-based methods:** SimpleVLA-RL and VLA-RL target OpenVLA/OpenVLA-OFT, not flow-matching π0.5.

### Cited Findings
**πRL / RLinf**
- πRL (arXiv 2510.25889; Tsinghua, PKU, CAS, CMU, Infinigence) bills itself as the first open-source framework for parallel online RL fine-tuning of flow-based VLAs π0 and π0.5. Code is in RLinf (github.com/RLinf/RLinf) and checkpoints are on HF under RLinf. — [arXiv](https://arxiv.org/html/2510.25889v3); [HF papers](https://huggingface.co/papers/2510.25889)
- **The log-likelihood fix:** Exact log-likelihoods for flow policies come from two techniques:
  - **Flow-Noise:** a learned noise network, `noise_logvar_range` [0.08, 0.16], entropy bonus 0.005, joint logprob.
  - **Flow-SDE:** an ODE-to-SDE conversion, `noise_level` 0.5.

  — [RLinf π0 docs](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/pi0.html)
- **Reported results (baseline → RL):**

  | Benchmark | Model | Baseline | After RL |
  |---|---|---|---|
  | LIBERO avg | π0 | 57.6% (few-shot) | 96.0% (PPO), 90.0% (GRPO) |
  | LIBERO avg | π0.5 | 77.1% (few-shot) | 97.9% (PPO), 91.5% (GRPO) |
  | ManiSkill3 | π0.5 | 40.1% | 90.9% |
  | MetaWorld MT50 | π0.5 | 43.8% | 70.7% |
  | CALVIN ABC-D | π0.5 | 61.3% | 87.0% |

  PPO beats GRPO on LIBERO. — [RLinf π0 docs](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/pi0.html)
- **Compute:** No minimum GPU count is documented. The reference comparison ran on 4×H100. A low-resource quickstart config exists. Supported environments are LIBERO, ManiSkill3, MetaWorld, and CALVIN. — [RLinf π0 docs](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/pi0.html)
- **RLinf-VLA (arXiv 2510.06710):**
  - 98.11% across 130 LIBERO tasks and 97.66% across 25 ManiSkill tasks.
  - A preliminary real Franka test found RL-trained policies generalized better than SFT. — [arXiv](https://arxiv.org/html/2510.06710v1)
- **RLinf-USER (real-world online learning):**
  - Supports online IL and RL for CNN/MLP, flow, and VLA policies.
  - π0 with HG-DAgger: the async pipeline cut time per update from 45.01 s to 7.90 s on Franka.
  - v0.3 added SAC-Flow for real-world RL plus dual-arm Franka, GimArm, and DOS-W1 support.
  - Real-world robots listed are Franka, GimArm, XSquare Turtle2, and DOS-W1. SO-100/101 is not listed. — [RLinf-USER](https://rlinf.readthedocs.io/en/latest/rst_source/resources/publications/rlinf_user.html); [v0.3 release notes](https://rlinf.readthedocs.io/en/latest/rst_source/resources/release_v0.3.html)

**DSRL (Diffusion Steering via RL)**
- **Method:** Wagenmaker, Nakamoto et al., CoRL 2025, arXiv 2506.15799. RL runs in the latent-noise space of a diffusion/flow policy, so the base weights never change and only black-box access is needed. The authors describe it as "highly sample efficient" and say it enables real-world autonomous improvement. — [PMLR](https://proceedings.mlr.press/v305/wagenmaker25a.html); [arXiv](https://arxiv.org/abs/2506.15799)
- **Official code:** github.com/nakamotoo/dsrl_pi0, JAX, steers π0. Simulation: LIBERO and ALOHA. Real robot: Franka/DROID, with π0-DROID served remotely and DSRL running on the robot client. — [GitHub](https://github.com/nakamotoo/dsrl_pi0)
- **RLinf reimplementation:** A small SAC agent (~500K trainable parameters) acts in noise space while π0 stays frozen. — [RLinf DSRL docs](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/dsrl.html)

**ReinFlow / FPO**
- **ReinFlow** (NeurIPS 2025, arXiv 2505.22094) adds learnable noise to the flow path, which makes the likelihood tractable for online RL. Code and checkpoints are at reinflow.github.io. — [arXiv](https://arxiv.org/pdf/2505.22094); [project](https://reinflow.github.io/)
- **FPO** (arXiv 2510.09976) is an on-policy policy-gradient method for flow matching, applied to VLA fine-tuning on LIBERO and ALOHA sim. No code link was found. ReFPO adds a Reflow regularizer. — [emergentmind](https://www.emergentmind.com/papers/2510.09976); [deeppaper summary](https://arxiv.deeppaper.ai/papers/2510.09976v1?locale=en)

**PA-RL (Policy-Agnostic RL)**
- **Method and results:** Mark, Finn, Kumar et al., arXiv 2412.06685, ICLR 2025 workshop. It replaces the policy-improvement step with supervised imitation of critic-maximizing actions near the data support. Reported results:
  - Diffusion policies improved 80–100% in 1–2 hours of online RL on a real WidowX.
  - OpenVLA (7B) improved 75% after 1 hour.
  - Code is on the project site. — [arXiv](https://arxiv.org/html/2412.06685v1); [project](https://PolicyAgnosticRL.github.io/)

**ConRFT**
- ConRFT (arXiv 2502.05450) uses an offline stage plus an online human-in-the-loop stage built on a consistency policy. Reported results:
  - 96.3% average success after 45–90 minutes of real online fine-tuning, a 144% gain over SFT.
  - Episodes 1.9× shorter.
  - Baselines: HG-DAgger 65%, PA-RL 71.3%.
  - Eight real tasks, 20 trials each. No independent replication was found. — [arXiv](https://arxiv.org/html/2502.05450v2)

**SimpleVLA-RL**
- **Method:** ICLR 2026, arXiv 2509.09674, code PRIME-RL/SimpleVLA-RL. Built on veRL; GRPO with binary outcome rewards, dynamic sampling, higher rollout temperature, clip [0.8, 1.28], no KL. — [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/cbfbcb4da14235bd69b134070898ae9d-Abstract-Conference.html); [emergentmind](https://www.emergentmind.com/papers/2509.09674)
- **Reported results:**
  - LIBERO ~91% → ~99%.
  - With one demo per task, ~49% → ~97%.
  - RoboTwin: 80% relative improvement, outperforming π0.
  - Real AgileX Piper after sim-only RL: 17.5% → 38.5% (from a secondary Korean review).
  - The backbone is OpenVLA-OFT (autoregressive/parallel decoding), not flow matching. — [ICLR](https://iclr.cc/virtual/2026/poster/10009314); [wikidocs review](https://wikidocs.net/343469)

**GRPO vs PPO for π0.5:** Both are implemented in RLinf, and PPO is stronger on LIBERO (97.9% vs 91.5% for π0.5). — [RLinf π0 docs](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/pi0.html)

### Inferences
- **What applies to π0.5 directly:**
  - πRL/RLinf (sim, PPO/GRPO): open source.
  - DSRL: open source for π0. Porting to π0.5 should be conceptually trivial because it only replaces the initial noise.
  - RECAP: see section 1.
  - ReinFlow/FPO are algorithmic building blocks, and RLinf already absorbs the ideas.
  - SimpleVLA-RL / VLA-RL / GRAPE target token-based OpenVLA and would need adaptation.
- **Best fit for one SO-101 in 1–2 days:**
  - Keep π0.5 frozen and served on a cloud GPU.
  - Train a DSRL noise-space SAC agent, or a small residual actor, on the robot, with success labeled by keyboard or a reward classifier.
  - Rewards: sparse success labels.
  - Budget: an hour or two of on-robot interaction for a single short task. This is an extrapolation from the PA-RL and ConRFT hour-scale real-robot results and is not verified on SO-101.
- **Not feasible in a hackathon:** full-weight online PPO on π0.5 on a single real arm. The documented runs rely on massively parallel simulation.

### Gaps
- Could not retrieve details on VLA-RL (Lu et al. 2025), the Policy Decorator / residual-RL-on-π0 work, or RoboMonkey. These searches returned nothing usable within the tool budget.
- No primary-source numbers for DSRL's real-world sample counts.
- No source shows DSRL or πRL running on SO-101 hardware.

## 3. Human-in-the-loop corrections / DAgger-style data, and preference methods

### Takeaway
LeRobot ships a full HIL-SERL pipeline with explicit SO-100/SO-101 support. It covers the leader arm as the intervention device, a reward classifier, and an actor–learner SAC setup, and claims training in "a few hours". However, it trains a small Gaussian-actor SAC policy on cropped 128×128 images in end-effector space, not a VLA.

For a VLA, the practical HIL options are:
- **HG-DAgger style:** record interventions and add them to the SFT data. RLinf-USER supports this for π0.
- **RECAP:** corrections plus advantage labels.

Preference methods (GRAPE, DPO) exist for OpenVLA but are not established for flow VLAs.

### Cited Findings
- **LeRobot HIL-SERL pipeline:**
  - Combines a few teleop demos, a vision reward classifier (e.g. ResNet10), and a distributed actor/learner SAC loop over gRPC with human interventions.
  - Needs a gamepad or keyboard or a leader arm, an NVIDIA GPU, and a URDF. The SO-101 URDF comes from TheRobotStudio/SO-ARM100.
  - Recommended tasks take 5–10 s (push cube, pick-and-lift).
  - Claims you can learn "in just a few hours". — [LeRobot HIL-SERL docs](https://huggingface.co/docs/lerobot/hilserl)
- **SO-101 specifics:**
  - The geared SO-101 leader tracks the follower, so takeover is smoother than with the SO-100.
  - Space toggles intervention; `s` marks success and `esc` marks failure.
  - Learning runs in end-effector space with EE bounds from `lerobot-find-joint-limits`, at 10 fps and 128×128 cropped images.
  - The policy is `type="gaussian_actor"` with `algorithm.type="sac"`. — [LeRobot HIL-SERL docs](https://huggingface.co/docs/lerobot/hilserl)
- **Effect of interventions:** The docs show that interventions cut steps-to-max-reward by about a quarter compared with no interventions. The intervention rate should decline over training. Tuning tips: `temperature_init` ≈ 1e-2 and a weight push every 1–2 s. — [LeRobot HIL-SERL docs](https://huggingface.co/docs/lerobot/hilserl)
- **Original paper:** HIL-SERL, Luo, Xu, Wu, Levine, arXiv 2410.21845. — [LeRobot docs citation](https://huggingface.co/docs/lerobot/hilserl)
- **RLinf-USER** supports online imitation (HG-DAgger) for π0 on real Franka with an async pipeline (7.90 s per update). — [RLinf-USER](https://rlinf.readthedocs.io/en/latest/rst_source/resources/publications/rlinf_user.html)
- **ConRFT** uses HIL in its online stage and reports beating HG-DAgger (65%) and PA-RL (71.3%) with 96.3%. — [arXiv](https://arxiv.org/pdf/2502.05450)
- **GRAPE (arXiv 2411.19309):**
  - Trajectory-level preference optimization on OpenVLA with LoRA, iterating on online-collected trajectories. Preferences come from VLM-proposed stage costs (GCPG).
  - Reported gains vary by version (e.g. +51.79% in-domain, +58.20% or +60.36% unseen); one review reports +33.14% over OpenVLA-DPO in sim.
  - The authors argue PPO-style online RL is impractical for VLAs because of rollout and reward costs. — [arXiv](https://arxiv.org/pdf/2411.19309); [HF](https://huggingface.co/papers/2411.19309)

### Inferences
- **Fastest real gain on SO-101:**
  1. Run the SFT'd π0.5.
  2. Record leader-arm takeovers on failure.
  3. Fine-tune on demos plus intervention segments (HG-DAgger).
  4. Optionally add RECAP labels.

  This reuses existing LeRobot record and train tooling and needs no RL code.
- **Using HIL-SERL itself:** It could serve as a residual or specialist for a narrow sub-skill (e.g. the final grasp), or as a reward-classifier source for other methods. It does not fine-tune π0.5.
- **DPO for flow VLAs:** Needs likelihoods. Flow-Noise/Flow-SDE make likelihoods available, but no mature open DPO-for-π0.5 code was found.

### Gaps
- No evidence found of LeRobot officially supporting HIL-SERL with π0.5 or SmolVLA as the actor.
- No primary source for DPO applied to π0/π0.5.

## 4. Sim-based post-training and sim-to-real for SO-100/101

### Takeaway
Sim RL for π0.5 is mature in RLinf (LIBERO, ManiSkill3, MetaWorld, CALVIN), but none of these environments model the SO-101 out of the box. SO-101 simulation assets exist:
- ManiSkill3 has SO-100 upstream; SO-101 is in the Squint authors' repo.
- Isaac Lab hosts SO-101 Bench and NVIDIA's SO-101 sim-to-real course (built around GR00T).

Building a matching sim task, running RL, and transferring to the real arm is a multi-week project, not a hackathon task.

### Cited Findings
- **Squint (arXiv 2602.21203):**
  - Off-policy visual RL with the SO-101 in ManiSkill3 across 8 domain-randomized tasks.
  - Zero-shot real result: 73/80 (91.3%) against 96.1% in sim, using a 16×16 wrist camera.
  - A community review disputes the ManiSkill claim: upstream has only SO-100, and SO-101 is in the authors' repo. The policy is single-task RL, not a VLA. — [arXiv](https://arxiv.org/pdf/2602.21203); [review](https://mcobzarenco-fontaine-blog.static.hf.space/papers/squint.html)
- **SO-101 Bench (arXiv 2610.36420):**
  - An Isaac Lab digital twin. Sim rankings matched real rankings (Spearman ρ = 1.00), but sim overestimated success.
  - Best learned policy ~61%; an LLM agent with a grasp policy reached 83.2%.
  - A secondary source puts the sim overestimate at 9–31 pp. — [opentrain summary](https://www.opentrain.ai/papers/so-101-bench-accessible-spatial-manipulation--arxiv-2610.36420/); [aiweekly](https://aiweekly.co/alerts/so-101-bench-shows-sim-policy-rankings-match-real-hardware)
- **NVIDIA's "Train an SO-101 Robot From Sim-to-Real With NVIDIA Isaac" course:**
  - GR00T VLA for vial pick-and-place.
  - Covers domain randomization, sim-real co-training, Cosmos augmentation, and actuator-gap estimation. — [NVIDIA docs](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/index.html)
- **Isaac Teleop:** Docs cover SO-101 demo collection in Isaac Lab and on the real arm, then fine-tuning GR00T N1.7. Some pages are marked in progress. — [Isaac Teleop](https://nvidia.github.io/IsaacTeleop/preview/pr-831/getting_started/lerobot/index.html)
- **RLinf sim–real co-training:** The RECAP docs reference a `pi05_maniskill_sim_real_co_training` config (Franka). — [RLinf RECAP docs](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/recap.html)

### Inferences
- **Hackathon use of sim:** Use it only for evaluation or ranking (SO-101 Bench shows rank correlation) or for debugging an RL pipeline on RLinf's ManiSkill/LIBERO π0.5 before touching hardware.
- **Sim-to-real RL for π0.5 on SO-101:** Not realistic in 1–2 days.

### Gaps
- No source found showing π0.5 RL-fine-tuned in sim and transferred to SO-100/101.

## 5. Cheap inference-time improvements (RTC, value-guided selection, ensembling)

### Takeaway
RTC is production-ready in LeRobot for π0, π0.5, and SmolVLA. It is a one-flag change and the highest-value, lowest-effort improvement for smooth real-robot execution. Intel ships a π0.5 + RTC OpenVINO pipeline, which matters for the Intel hardware track.

Value-guided action selection (V-GPS) gives sizable reported gains (+82% real-world) without touching policy weights. It needs a value function, and that same value model can be shared with RECAP. Temporal ensembling is beaten by RTC under latency.

### Cited Findings
- **RTC in LeRobot:**
  - An inference-time method that keeps Pi0, Pi0.5, and SmolVLA motion smooth and reactive under high latency.
  - Command: `lerobot-rollout --strategy.type=base --inference.type=rtc`, with guidance weight 10.0 recommended for 10-step flow. — [LeRobot RTC docs](https://huggingface.co/docs/lerobot/en/rtc)
- **RTC paper:** "Real-Time Execution of Action Chunking Flow Policies", arXiv 2506.07339. Real-world tests used π0.5 on bimanual tasks, and RTC was more robust than temporal ensembling under added latency. — [review](https://www.themoonlight.io/en/review/real-time-execution-of-action-chunking-flow-policies)
- **Unverified caveat:** A third-party post claims that in LeRobot v0.6.1 the async gRPC server calls the policy without RTC guidance, and that the code default schedule (LINEAR) differs from the documented EXP. — [particula.tech](https://particula.tech/blog/real-time-chunking-vs-async-inference-lerobot)
- **Intel π0.5 + RTC sample pipeline:** Runs on a LeRobot fork with OpenVINO. The exported model takes extra inputs, `prev_chunk_left_over` and `prefix_weights`. — [Intel Open Edge docs](https://docs.openedgeplatform.intel.com/2026.1/edge-ai-suites/robotics-ai-suite/embodied/sample_pipelines/pi05_with_rtc.html)
- **V-GPS** (Nakamoto, Mees, Kumar, Levine, CoRL 2024):
  - Re-ranks actions sampled from a generalist policy using an offline-RL value function. Needs no fine-tuning or weight access.
  - +82% real-world improvement; consistently improved five generalist policies across embodiments.
  - Code at nakamotoo.github.io/V-GPS. — [arXiv](https://arxiv.org/html/2410.13816); [PMLR](https://proceedings.mlr.press/v270/nakamoto25a.html)
- **Related generate-and-verify methods:** V-GPS, RoboMonkey, and FOREWARN sample candidate actions and score them with an external verifier. UF-OPS trains verifiers on the policy's own rollouts for self-improvement with small datasets. — [arXiv 2602.19372](https://arxiv.org/html/2602.19372)

### Inferences
- **What to do first in a hackathon:**
  1. Turn on RTC.
  2. Sample N chunks per step (π0.5 flow sampling is stochastic in the noise).
  3. Pick the best chunk with a lightweight success/progress classifier trained on the team's own rollouts. This is V-GPS-like and doubles as the RECAP value model.
- **Cost:** N× inference cost per chunk, which is affordable on a cloud GPU at the chunk rate.
- **DSRL as an upgrade:** DSRL can be read as an "amortized" version of best-of-N noise selection.

### Gaps
- No quantitative RTC gains on SO-101 were found.
- No published V-GPS results with π0.5 specifically.
- RoboMonkey's details were not retrieved.
- The particula.tech claim about async-server RTC was not checked against LeRobot source.
