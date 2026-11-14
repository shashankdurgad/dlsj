# π0.5 action-expert cost and execution-side latency hiding (state as of 14 Nov 2026)

Reader notes:
- Scope is the flow-matching action expert and the execution loop only. The VLM prefix, quantisation and OS/runtime tuning are excluded.
- "Our numbers" means the team's measurements given in the brief (53 ms of 169 ms at 10 Euler steps, about 5.3 ms per step, 30 Hz control, 50-action chunks). Arithmetic on those numbers is marked as inference.
- Repository facts were read directly from GitHub on 14 Nov 2026 (Physical AI Studio `main`, LeRobot `main`, latest LeRobot release v0.7.0 dated 5 Nov 2026). Paper facts come from arXiv HTML pages read through a summarising fetch tool, so table values should be re-checked against the PDFs before being quoted externally.
- Items marked UNVERIFIED come from search snippets, abstracts or secondary blogs only.

## Step distillation for π0.5 (SnapFlow, Drift, MeanFlow, shortcut/consistency, reflow)

### Takeaway
SnapFlow is the only one-step method with π0.5 results, a frozen-backbone expert-only recipe, and shipping code in Physical AI Studio (including an SO-101 config and OpenVINO export); it matches its 10-step teacher on LIBERO in two independent runs, but all quantitative evidence is simulation and nobody has reported its effect on motion amplitude. For the team it fits in a day on one cloud GPU and would remove about 48 ms of the 169 ms (inference from our numbers).

### Cited Findings

SnapFlow, the paper:
- SnapFlow self-distils a pretrained flow-matching VLA to one step by mixing ordinary flow-matching samples with "shortcut" samples whose target is the model's own two-step Euler endpoint under stop-gradient; a zero-initialised destination-time projection keeps the model identical to the teacher at the start. No separate teacher network is needed. v1 was posted 7 Apr 2026 and v2 on 14 Oct 2026. — [arXiv 2604.05656 abstract](https://arxiv.org/abs/2604.05656)
- Expert-only distillation with a frozen backbone is the method as published: PaliGemma is frozen, and the roughly 300M-parameter action expert plus the new projection are trained. — [arXiv 2604.05656 v2 HTML](https://arxiv.org/html/2604.05656v2)
- Training cost for the π0.5 run: about 12 hours on one A800 for 30,000 updates at batch size 4, bfloat16, AdamW at 2.5e-5, gradient clip 1.0, 500 warm-up steps, mix α = 0.5, shortcut weight λ = 0.1, shortcut target clipped to [−20, 20]. The first 5k updates took about 25 minutes. Dataset size is not stated. — [arXiv 2604.05656 v2 HTML](https://arxiv.org/html/2604.05656v2)
- LIBERO success for π0.5 (Spatial / Object / Goal / Long-10 / average): 10-step teacher 98.0 / 100.0 / 96.0 / 97.0 / 97.75; naive one-step Euler with no training 96.0 / 99.0 / 98.0 / 94.0 / 96.75; SnapFlow one-step 99.0 / 100.0 / 99.0 / 97.0 / 98.75. — [arXiv 2604.05656 v2 HTML](https://arxiv.org/html/2604.05656v2)
- For π0 the paper's own teacher scored 70.5% against a published 96.6%, and the authors say they could not isolate the cause, so the "+10.5 points over teacher" π0 result is weak evidence. — [arXiv 2604.05656 v2 HTML](https://arxiv.org/html/2604.05656v2)
- Latency on an A800 at batch one: teacher 274 ms, naive one-step 81 ms, SnapFlow 83 ms (3.3×); teacher at 2 / 3 / 5 steps 103 / 124 / 167 ms. The prefix costs about 60 ms and each expert step about 23 ms. — [arXiv 2604.05656 v2 HTML](https://arxiv.org/html/2604.05656v2)
- Held-out action error (500 samples, π0.5 LIBERO): mean MSE 0.01169 → 0.00773 (−33.9%), standard deviation 0.0541 → 0.0296, with gains concentrated in the P90/P95 tail rather than the median. — [arXiv 2604.05656 v2 HTML](https://arxiv.org/html/2604.05656v2)
- Table 9 of the paper shows the untrained teacher's offline MSE falling as steps are removed: 0.00892 / 0.00918 / 0.00957 / 0.01116 / 0.01238 at 1 / 2 / 3 / 5 / 10 steps. The distilled student is best at 2–3 steps (0.00723 / 0.00716) and 0.00826 at one step. — [arXiv 2604.05656 v2 HTML](https://arxiv.org/html/2604.05656v2)
- The paper does not discuss mode collapse, action magnitude, smoothness or variance. Real-robot evidence is nine qualitative episodes on a dual-arm AgiBot humanoid with no success rates reported; the authors list "real-robot evidence is qualitative" as a limitation. — [arXiv 2604.05656 v2 HTML](https://arxiv.org/html/2604.05656v2)
- Conflict to flag: older summaries and the Physical AI Studio how-to quote a "9.6× denoising speed-up, 274 → 83 ms end to end" and "about 80% of latency is the loop"; the v2 abstract now states only about 3.3× batch-one model latency. — [Physical AI Studio SnapFlow how-to](https://github.com/open-edge-platform/physical-ai-studio/blob/main/library/docs/how-to/training/snapflow_distillation.md); [arXiv 2604.05656 abstract](https://arxiv.org/abs/2604.05656)

SnapFlow in Physical AI Studio:
- The repo ships `library/configs/physicalai/pi05/so101/snapflow.yaml`, `pi05/aloha/snapflow.yaml`, `smolvla/pusht/snapflow.yaml`, a `SnapFlowPhaseCallback`, and mixins in `library/src/physicalai/policies/mixins/snapflow.py`. — [Physical AI Studio repo](https://github.com/open-edge-platform/physical-ai-studio)
- The SO-101 config is a single run: epochs 0–4 are a full-model flow-matching fine-tune from `lerobot/pi05_base`, epochs 5–7 are SnapFlow with the VLM frozen (about 10% of parameters, 311.9M of 3138.4M, trainable). Batch 16, lr 2.5e-5, bf16, chunk 50, example dataset `Daankrol/pick-and-place-multi-obj`. — [so101/snapflow.yaml](https://github.com/open-edge-platform/physical-ai-studio/blob/main/library/configs/physicalai/pi05/so101/snapflow.yaml)
- An existing checkpoint can be distilled without repeating phase 1: `physicalai fit --config configs/physicalai/pi05/aloha/snapflow.yaml --weights_from <ckpt>`, or `policy.enable_snapflow(alpha=0.5, lambda_=0.1, num_inference_steps=1)` in Python. — [Physical AI Studio SnapFlow how-to](https://github.com/open-edge-platform/physical-ai-studio/blob/main/library/docs/how-to/training/snapflow_distillation.md)
- Phase-2 steps are about 2–3× slower than ordinary training steps because the consistency branch runs three velocity passes per sample. The guide recommends about 2–5 epochs for phase 2 and warns that "distilling an undertrained phase-1 model just distills noise". — [Physical AI Studio SnapFlow how-to](https://github.com/open-edge-platform/physical-ai-studio/blob/main/library/docs/how-to/training/snapflow_distillation.md)
- `num_inference_steps` can be set above 1 for "intermediate modes", and the checkpoint exports with `physicalai export --backend openvino`, carrying the one-step sampling path. — [Physical AI Studio SnapFlow how-to](https://github.com/open-edge-platform/physical-ai-studio/blob/main/library/docs/how-to/training/snapflow_distillation.md)
- Independent replication by the Studio team: a π0.5 LIBERO student distilled for 20,000 steps (batch 8, VLM frozen, SnapFlow on from step 0) reached 97.55% (1,951/2,000 episodes) against 97.05% for the 10-step teacher; the 10k-step student reached 96.80%. On an H100 the median chunk latency was 58.9 ms against 175.4 ms (2.98×). The report says the 0.5-point gap is one seed and not statistically significant. — [Studio SnapFlow LIBERO report](https://github.com/open-edge-platform/physical-ai-studio/blob/main/library/docs/explanation/benchmark/snapflow-libero.md)

Nota AI "Drift" one-step π0.5:
- The Nota AI model is not on the official `nota-ai` organisation; it is a gated repo, `geonmin-kim/nota-pi05-libero-vtp-drift-openvino`, described as optimised by Nota AI. Drift trains the action expert to produce the chunk in one step; the graph is unchanged with `num_inference_steps` = 1. It was fine-tuned from `lerobot/pi05_libero_finetuned_v044` for 10k steps. Whether only the expert was trained is not stated. — [Nota Drift model card](https://huggingface.co/geonmin-kim/nota-pi05-libero-vtp-drift-openvino)
- LIBERO-10 on an Arc B390 iGPU (Core Ultra X7 358H, OpenVINO 2026.4.0, 200 episodes): base 95.5%, Drift 94.0%, Drift plus visual-token pruning 92.0%. Latency per chunk: base 258 ms, Drift 175 ms, Drift plus pruning 117 ms. Simulation only, single-device single-episode latency. — [Nota Drift model card](https://huggingface.co/geonmin-kim/nota-pi05-libero-vtp-drift-openvino)
- The card reports that OpenVINO's ONNX-frontend `RandomNormalLike` produced biased noise, and that moving noise to an external N(0,1) input raised iGPU LIBERO-10 success from 90.5% to 94.0% for Drift. It also says zero or non-Gaussian noise lowers success. — [Nota Drift model card](https://huggingface.co/geonmin-kim/nota-pi05-libero-vtp-drift-openvino)
- A separate community pair, `zuoxingdong/pi05_drift_libero` and `zuoxingdong/pi05_fm_libero`, offers a one-step Drifting-objective π0.5 and its matched flow-matching baseline. UNVERIFIED beyond the search summary. — [pi05_drift_libero](https://huggingface.co/zuoxingdong/pi05_drift_libero)

Other one-step families:
- A MeanFlow-based one-step VLA built on the π0 action expert (arXiv 2603.01469, March 2026) trains from scratch with a MeanFlow objective rather than distilling, and claims generation 8.7× faster than SmolVLA on real robots. A secondary blog says this came with about 78% average success against 84.5% for SmolVLA. UNVERIFIED against the paper. — [arXiv 2603.01469](https://arxiv.org/html/2603.01469); [one-step notes blog](https://mcobzarenco-fontaine-blog.static.hf.space/papers/one-step-menu.html)
- One-Step Flow Policy (arXiv 2603.12480) is a from-scratch self-distillation method without a pretrained teacher; MP1 (arXiv 2507.10543) applies MeanFlow to point-cloud policies and is not a VLA. UNVERIFIED beyond the search summary. — [arXiv 2603.12480](https://arxiv.org/pdf/2603.12480); [arXiv 2507.10543](https://arxiv.org/pdf/2507.10543v4)
- A February 2026 Tommoro Robotics paper distils a multi-step flow action expert into a rectified model for edge control on a different VLA. UNVERIFIED beyond the search summary. — [arXiv 2602.18813](https://arxiv.org/pdf/2602.18813)
- A third-party research log reports that a SnapFlow student on its own flow model matched its teacher at one step, but a 4k-step fine-tune on its real rig data made hold-out results worse and was not shipped. This is one unnamed author's log, not a paper. — [research log, idea 12](https://mcobzarenco-fontaine-blog.static.hf.space/ideas/12-solver-heun-gap.html)

### Inferences
- From our numbers, one step instead of ten removes about 9 × 5.3 ≈ 48 ms, taking 169 ms to about 121 ms (1.4× end to end). At 30 Hz that moves the inference delay from 6 control steps to 4.
- The Studio SO-101 config assumes a full-model phase 1. The team's checkpoint is already expert-only fine-tuned, so the `--weights_from` phase-2-only path is the fit; the VLM stays frozen either way, which matches how the checkpoint was trained.
- Compute fits a day: the paper's 30k updates at batch 4 took 12 h on one A800, and the Studio run reached teacher parity at 20k steps of batch 8. A larger cloud GPU and 10–20k steps should finish well inside 24 h, leaving time for export.
- Mean-seeking is the main risk. SnapFlow's shortcut target is the model's own two-step Euler endpoint, and the team already sees motion range fall to 0.82–0.86 at 3–5 steps. A student trained toward few-step targets may inherit that shrinkage. No source measures it, so motion range against recorded demos should be a release gate alongside offline error.
- Lower offline MSE after distillation is not evidence of better behaviour. The paper's own Table 9 shows the untrained teacher's MSE improving as steps are removed, the same pattern the team measured.
- The static-graph constraint favours distillation: the exported graph is the existing one with a single unrolled step and one extra time input, and roughly 330 of the 3,300 per-chunk GPU kernels remain.
- Exporting at 2 steps is a middle option worth one extra export, since the student's offline error is lowest at 2–3 steps and the cost is about 5 ms.

### Gaps
- No source reports SnapFlow or Drift success rates on a real robot, or on any SO-100/SO-101 arm.
- No source measures action amplitude, jerk or multimodality of one-step students. Mode collapse is not discussed in the SnapFlow paper.
- No published result on distilling a teacher that was itself only expert-fine-tuned for a few thousand steps on about 500 demos. The Studio guide's warning about undertrained teachers applies and is untested at this scale.
- Drift's training recipe, loss and compute are not public on the card; the repo is gated.
- I found no consistency-distillation or reflow result specifically on π0.5 other than SnapFlow and Drift.

## Training-free fewer-step sampling (solvers, time grids, amplitude shrinkage)

### Takeaway
Fewer Euler steps lowering offline error is a documented effect, but I found no source that names or fixes the amplitude shrinkage for π0/π0.5, and no measured Heun, midpoint, DPM-Solver or Adams-Bashforth result on a π-family model. The only sourced training-free schedule change is "dense-jump" (steps packed into the first half, then one jump), tested in simulation on small policies only.

### Cited Findings
- The untrained π0.5 teacher's held-out MSE falls monotonically as steps are removed (0.01238 at 10 steps, 0.01116 at 5, 0.00957 at 3, 0.00892 at 1), and naive one-step Euler still scored 96.75% on LIBERO against 97.75% at 10 steps. — [arXiv 2604.05656 v2 HTML](https://arxiv.org/html/2604.05656v2)
- Dense-Jump Flow Matching reports that adding uniform Euler steps can lower policy performance (Walker2D success 84.6% at 2 steps, 77.6% at 4, 71.2% at 16, 66.6% at 64) and gives two causes: uniform grids spend many evaluations late in the trajectory, where the policy drifts toward memorised training actions, and the ideal velocity field's Lipschitz constant grows as 1/(1−t) near the data end. — [arXiv 2509.13574 v2](https://arxiv.org/html/2509.13574v2)
- Its inference schedule puts N−1 uniform Euler steps on the first half of the time interval and makes one terminal jump from the midpoint. An ablation applies it to a model trained with ordinary uniform time sampling, so it can be used without retraining; the best results also use a Beta(0.2, 0.2) training-time schedule. — [arXiv 2509.13574 v2](https://arxiv.org/html/2509.13574v2)
- That paper reports no action magnitude, variance or smoothness metrics, and tests only Walker2D, HumanoidStandup and Adroit Pen in simulation; VLA and real-robot evaluation are listed as future work. — [arXiv 2509.13574 v2](https://arxiv.org/html/2509.13574v2)
- The RTC paper ran π0.5 on real bimanual robots with 5 denoising steps rather than 10 (76 ms on an RTX 4090), so 5-step π0.5 has first-party real-world use. — [arXiv 2506.07339 v2](https://arxiv.org/html/2506.07339v2)
- LeRobot's π0.5 defaults are `num_inference_steps = 10` with training time sampled from a Beta(1.5, 1.0) distribution scaled by 0.999 with offset 0.001, which weights training toward the noise end. — [LeRobot configuration_pi05.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/pi05/configuration_pi05.py)
- A third-party research log found that a Heun-versus-Euler gap seen on one flow model did not transfer to another, and expects a training-free RK4 check to be a null result. Single unnamed author, not π0.5. — [research log, idea 12](https://mcobzarenco-fontaine-blog.static.hf.space/ideas/12-solver-heun-gap.html)
- An image-domain paper (PRECISE) argues that some few-step update rules suffer covariance contraction that biases the output distribution. It is not a robot-policy result. UNVERIFIED beyond the search summary. — [arXiv 2605.23522](https://arxiv.org/html/2605.23522)

### Inferences
- The likely mechanism for the team's shrinkage: the first Euler step from pure noise predicts roughly the conditional mean action, and with few steps the sample never moves far from it. That lowers squared error and narrows motion range at once. This matches the team's −5.8% / −10.3% error and 0.86 / 0.82 range, and SnapFlow's Table 9, but no source states it for π0.5.
- Higher-order solvers are a poor fit here. Heun and midpoint cost two expert evaluations per step, so 5 Heun steps cost the same as 10 Euler steps. They reduce discretisation error, which would push the sample back toward the 10-step result, restoring both the amplitude and the higher offline error.
- Multistep solvers (Adams-Bashforth 2, DPM-Solver++ 2M) reuse the previous velocity and add no expert evaluations, and they unroll cleanly into a static graph. They are the cheapest thing to try for "5-step cost, closer to 10-step amplitude", but there is no π0.5 evidence either way.
- A non-uniform grid is a one-line change in an unrolled export. Dense-jump concentrates steps at the noise end; given the team's shrinkage, the opposite allocation (more steps near the data end) is the one more likely to restore amplitude. Both should be measured against motion range, not offline error.
- Rescaling output amplitude by a constant (about 1/0.86) or raising initial noise scale are untested heuristics with no source. Classifier-free guidance does not apply, because π0.5 has no unconditional branch.
- Since any step count must be fixed at export, the practical experiment is a small matrix of exported graphs (10 Euler, 5 Euler, 5 AB2, 5 non-uniform, 1–2 step distilled) scored on motion range, jerk and on-robot success.

### Gaps
- No published measurement of motion amplitude against step count for π0 or π0.5.
- No published Heun, midpoint, DPM-Solver or Adams-Bashforth comparison on π0/π0.5 or SmolVLA.
- No evidence on whether reduced amplitude at 3–5 steps hurts or helps real-robot success; the RTC paper's 5-step use suggests it is tolerable on Physical Intelligence's own checkpoints but does not isolate it.

## Caching inside the denoising loop, expert pruning and width reduction

### Takeaway
Caching methods report 3–4× on Diffusion Policy without retraining, but none has been tested on a π-family action expert, and their ceiling is below one-step distillation. Layer distillation (Shallow-π) is the published route to a smaller expert and needs teacher-student training.

### Cited Findings
- Block-wise Adaptive Caching (BAC, ICLR 2026) caches action features per block, chooses update steps by dynamic programming on feature similarity, and forces upstream blocks to refresh when a downstream feed-forward block refreshes. It reports about 3.4× on Robomimic and 3.6× on Block Pushing and Kitchen with average success unchanged or slightly higher. — [arXiv 2506.13456](https://arxiv.org/html/2506.13456v1); [ICLR 2026 poster](https://iclr.cc/virtual/2026/poster/10008539)
- BAC's schedule is computed once offline per task and is then fixed per block. It was tested only on transformer Diffusion Policy; the authors claim it is integrable with VLAs but report no VLA experiment. Some settings collapse (average success 0.07 with too few upstream blocks and updates). — [arXiv 2506.13456](https://arxiv.org/html/2506.13456v1)
- Sparse ActionGen (arXiv 2601.12894) prunes computation adaptively during the rollout and claims up to 4× without updating weights. UNVERIFIED beyond the abstract. — [Sparse ActionGen](https://www.alphaxiv.org/abs/2601.12894.md)
- C3ache (arXiv 2606.08962) caches action-expert residuals across consecutive chunks for World Action Models and says it composes with within-chunk caching. UNVERIFIED beyond the search summary. — [arXiv 2606.08962](https://arxiv.org/html/2606.08962)
- Shallow-π (Samsung Research, January 2026) distils flow-based VLAs from 18 to 6 transformer layers in both the backbone and the action head, with the student initialised by uniformly sub-sampling teacher layers. The abstract claims over 2× faster inference with under one point absolute success drop, and close to 10 Hz on Jetson Orin. Numbers are from the abstract and secondary coverage. — [arXiv 2601.20262](https://arxiv.org/abs/2601.20262v1)
- The SnapFlow authors name Shallow-π and EfficientVLA-style compression as complementary to step distillation but did not evaluate the combination. — [arXiv 2604.05656 v2 HTML](https://arxiv.org/html/2604.05656v2)
- Nota AI's visual-token pruning shortens the prefix from 968 to 584 tokens, which the card says also shrinks the action expert's KV cache; with Drift it took iGPU latency from 175 ms to 117 ms at a cost of 2 points on LIBERO-10. — [Nota Drift model card](https://huggingface.co/geonmin-kim/nota-pi05-libero-vtp-drift-openvino)

### Inferences
- A fixed, offline-computed cache schedule like BAC's is compatible with a static unrolled OpenVINO graph: skipped blocks are simply absent from the later steps. A runtime-adaptive scheme like Sparse ActionGen is not, without dynamic control flow.
- Caching and one-step distillation do not stack: at one step there is nothing to reuse. Caching only matters if the team stays at 5–10 steps, where a 3× claim on a 53 ms loop would save about 35 ms, less than distillation's 48 ms and with no π0.5 evidence.
- The team's observation that expert MLP layers already read weights at about 80% of memory bandwidth means per-step cost scales with weight bytes read. Skipping whole blocks (caching, layer pruning) cuts that directly, while fusing kernels would not.
- A shorter prefix reduces every expert step's cross-attention cost, so prefix-side token pruning (another researcher's scope) multiplies with step count.
- Layer distillation of the expert needs both teacher and student in memory and a longer training run than SnapFlow; it is beyond a one-day budget and only worth it after going to one or two steps.

### Gaps
- No caching result on π0, π0.5 or SmolVLA action experts. π0.5's expert attends to a fixed prefix KV cache, so feature drift across steps may differ from Diffusion Policy.
- No public BAC code link found in the paper text read.
- No source on width reduction of the π0.5 expert, or on Shallow-π applied to π0.5 specifically with expert-only compression.

## Execution-side methods that hide latency (RTC, async queues, ensembling, horizon choice)

### Takeaway
Asynchronous execution with Real-Time Chunking is the best-evidenced way to make 140–170 ms stop mattering: first-party π0.5 results show no degradation up to about 300 ms of delay, both LeRobot and Physical AI Studio implement it, and LeRobot v0.7.0 adds training-time RTC for π0.5 that removes the inference overhead. Temporal ensembling is contraindicated for this model class.

### Cited Findings

Inference-time RTC:
- RTC generates the next chunk while the current one executes and treats the overlap as inpainting: the first d actions (the inference delay) are frozen to the previous chunk, later overlapping actions get exponentially decaying weight, and the constraint is d ≤ s ≤ H − d, where s is the execution horizon and H the chunk length. The delay is forecast conservatively as the maximum of recent observed delays, and the actual horizon is max(d, s_min). — [arXiv 2506.07339 v2](https://arxiv.org/html/2506.07339v2)
- Real-world π0.5 setup: 5 denoising steps, H = 50, 50 Hz control, six bimanual tasks, 480 episodes. Baseline delay was about 6 control steps; +100 ms and +200 ms injected latency gave about 11 and 16 steps. The paper states RTC "is completely robust to injected delay, showing no degradation", while synchronous inference degraded linearly. RTC also finished tasks faster than synchronous inference with pauses removed. — [arXiv 2506.07339 v2](https://arxiv.org/html/2506.07339v2)
- Overhead on an RTX 4090 at 5 steps: 76 ms plain, 97 ms with RTC. Guidance weight is clipped at β = 5 because high weights made chunks diverge with few denoising steps. — [arXiv 2506.07339 v2](https://arxiv.org/html/2506.07339v2)
- Temporal ensembling "performs poorly across the board, even with an inference delay of d=0" in simulation, and on the real robot both ensembling variants oscillated enough to trigger protective stops at +100 and +200 ms. The paper notes "averages of valid actions are not necessarily valid". — [arXiv 2506.07339 v2](https://arxiv.org/html/2506.07339v2)
- Bidirectional decoding was beaten by RTC in simulation with the gap widening with delay, and it costs 115–223 ms against 97 ms because it samples many chunks per call; it was not run on the real robot. — [arXiv 2506.07339 v2](https://arxiv.org/html/2506.07339v2)
- In simulation RTC did better with a shorter execution horizon; the paper gives no other tuning rule. — [arXiv 2506.07339 v2](https://arxiv.org/html/2506.07339v2)
- An independent ALOHA write-up with π0 at 10 denoising steps found β ≈ 10 (rule of thumb β = number of steps) and an assumed prior standard deviation of 0.2 gave much smaller chunk-transition gaps than the paper's defaults, while β = 14 caused erratic motion on the real robot. One task, few episodes. — [Smooth-As-Butter Robot Policies](https://alexander-soare.github.io/robotics/2025/08/05/smooth-as-butter-robot-policies.html)

Implementations:
- LeRobot supports RTC for π0, π0.5 and SmolVLA through `lerobot-rollout --inference.type=rtc`, with a background inference thread and an action queue. Documented settings: `execution_horizon` typically 8–12, `max_guidance_weight` 10 for 10-step models, exponential prefix schedule recommended, and `inference_delay` passed per call from measured latency. The documented real-robot example uses `so100_follower` at 30 fps. — [LeRobot RTC docs](https://huggingface.co/docs/lerobot/en/rtc)
- In LeRobot, `execution_horizon` means how many steps of the previous chunk to stay consistent with, which is not the same quantity as the paper's "actions executed per chunk". — [LeRobot RTC docs](https://huggingface.co/docs/lerobot/en/rtc)
- Physical AI Studio has `RTCPolicyMixin` and `RTCModelMixin` for π0.5 and SmolVLA. Its correction uses the direct weighted error between the previous chunk and the predicted clean actions "not autograd.grad" for OpenVINO traceability; `inference_delay`, `execution_horizon` and `max_guidance_weight` are graph inputs, and RTC requires `n_action_steps` to equal the chunk size. — [Studio rtc.py](https://github.com/open-edge-platform/physical-ai-studio/blob/main/library/src/physicalai/policies/mixins/rtc.py)
- Intel publishes a "Pi0.5 with Real-Time Chunking" sample pipeline in its robotics AI suite docs for 2026.0, 2026.1 and dev. The page redirected and could not be read, so its contents are UNVERIFIED. — [Intel Pi0.5 with RTC](https://docs.openedgeplatform.intel.com/2026.1/edge-ai-suites/robotics-ai-suite/embodied/sample_pipelines/pi05_with_rtc.html)
- Intel's π0.5 optimisation article for Core Ultra Series 3 uses 10 denoising steps, runs the action expert on the NPU, and shows a runtime config with `SyncExecution`, chunk mode and 30 fps; it does not discuss step reduction, RTC or success rates. — [Intel π0.5 optimisation article](https://docs.openedgeplatform.intel.com/dev/OEP-articles/publications/optimizing-pi0.5-lva-model.html)

Training-time RTC:
- Training-time action conditioning (Physical Intelligence, December 2025) feeds the first d ground-truth actions clean, with their flow timestep set to the clean end, noises the rest, and computes loss only on the rest. It needs per-token timesteps and adds no parameters. Delays were sampled uniformly from 0–10 in the real-world runs, covering 200 ms at 50 Hz. — [arXiv 2512.05964 v2](https://arxiv.org/html/2512.05964v2)
- Results: in simulation it beats inference-time RTC at delays of 2 steps and above and is marginally worse at 0–1; on real box-building and espresso tasks with π0.6 the two perform similarly. End-to-end latency at 5 steps on a remote H100 was about 108 ms against 135 ms for inference-time RTC. Real-world models were fine-tuned for 8,000 steps at batch 512. — [arXiv 2512.05964 v2](https://arxiv.org/html/2512.05964v2)
- LeRobot added optional training-time RTC for π0.5 on 21 Aug 2026 (PR #4056): train with `--policy.rtc_training_max_delay=10`, roll out with `--inference.rtc.mode=trained`. The documented example is 30,000 steps at batch 8 on an `so100_follower`. Trained mode "avoids the guidance backward pass". — [LeRobot pi05 docs](https://github.com/huggingface/lerobot/blob/main/docs/source/pi05.mdx); [LeRobot RTC docs](https://huggingface.co/docs/lerobot/en/rtc)
- LeRobot's admission rule for trained mode is `rtc_training_max_delay <= execution_horizon <= chunk_size - rtc_training_max_delay`, with `inference.queue_threshold` at least the maximum delay; a latency spike beyond the trained range discards that chunk and retries. — [LeRobot pi05 docs](https://github.com/huggingface/lerobot/blob/main/docs/source/pi05.mdx)
- Conflict: a third-party FluxVLA page says π0.5 cannot support training-time RTC without model changes; LeRobot's shipped implementation contradicts that. — [FluxVLA RTC tutorial](https://fluxvla.limxdynamics.com/md_source/tutorials/inference/rtc.html); [LeRobot RTC docs](https://huggingface.co/docs/lerobot/en/rtc)
- "Action-Prior Denoising for Smooth Real-Time Chunking" (May 2026) proposes Soft RTC, arguing the binary prefix mask in training-time RTC leaves non-prefix tokens fully unconstrained. UNVERIFIED beyond the search summary. — [arXiv 2605.25537](https://arxiv.org/abs/2605.25537)

Other async methods with small-arm evidence:
- VLASH (MIT, arXiv 2512.01031) fine-tunes the policy to condition on the robot state expected when the chunk starts executing, by pairing each image with a state and action chunk offset by a random delay. It needs no inference-time overhead. — [arXiv 2512.01031](https://arxiv.org/html/2512.01031v1)
- VLASH real-world results with π0.5 on Galaxea R1 Lite and LeRobot SO-101 (pick-and-place, stacking, sorting; 16 rollouts per method; chunk 24 at 30 Hz; delay 4 steps; laptop RTX 4090): score 83% synchronous, 89.7% naive asynchronous, 94% VLASH; completion 18.8 s against 21 s. Training was 30,000 steps at batch 32. — [arXiv 2512.01031](https://arxiv.org/html/2512.01031v1)
- VLASH's "action quantisation" sums q consecutive delta actions into one macro-action to move faster: q = 2 kept accuracy with up to 2.03× speed-up, q = 3 cost 4.7% of score. — [arXiv 2512.01031](https://arxiv.org/html/2512.01031v1)
- Legato, a training-time continuation method for flow VLAs, reports about 10% better trajectory smoothness and task completion time than RTC across five real manipulation tasks. UNVERIFIED beyond the search summary. — [arXiv 2602.12978](https://papers.cool/arxiv/2602.12978)

### Inferences
- Delay at 30 Hz: one control step is 33.3 ms, so 140 ms is 4.2 steps and 170 ms is 5.1 steps. Rounding up and adding one step of jitter margin gives d = 6 today, and d = 4–5 after one-step distillation (about 121 ms).
- With H = 50 and d = 6 the paper's constraint allows an execution horizon from 6 to 44. A reasonable start is to request a new chunk every 8–12 executed actions (about 270–400 ms of motion), which keeps the policy reactive while leaving a few steps of slack beyond the delay. In LeRobot terms: `execution_horizon` 10, `queue_threshold` at least d, and for trained mode `rtc_training_max_delay` 8–10.
- Even the naive asynchronous baseline beat synchronous execution on SO-101 in VLASH (89.7% against 83%), so switching the runtime from `SyncExecution` to an action queue is worth doing before any retraining.
- Physical AI Studio's gradient-free RTC correction is an approximation of the paper's guidance. It should cost close to nothing per step (the paper's method cost about 21 ms on a 4090), but its fidelity to the paper's results is untested in any source I found.
- Guided RTC and one-step distillation probably conflict. The guidance is spread across denoising steps, and the RTC paper reports divergence with high weights at few steps; with one step the clipped weight is applied once to the whole chunk. Training-time RTC is the variant that fits one-step sampling, because the prefix is simply supplied clean. No source tests SnapFlow with either RTC variant.
- Training-time RTC touches only the action tokens and their time conditioning, which live in the action expert, so it should work with an expert-only fine-tune and a frozen VLM. This is not confirmed by any source.
- The prefix must be the commanded actions from the previous chunk, in the same normalised action space the model emits. With INT8-induced jitter in the previous chunk, RTC will faithfully continue the jitter over the frozen prefix, so RTC is a continuity fix and not a de-noising fix.
- Chunk length 50 at 30 Hz is 1.67 s. With replanning every 10 steps, 40 of the 50 predicted actions are never executed, so a shorter trained chunk (VLASH used 24–25) loses nothing at execution time, but the expert's per-step cost falls only modestly because the prefix tokens dominate attention.

### Gaps
- No published RTC success-rate numbers for π0.5 on SO-100/SO-101; LeRobot's docs give commands for `so100_follower` but no results.
- No source on choosing execution horizon at 30 Hz specifically; the first-party numbers are 50 Hz.
- Whether Physical AI Studio's runtime ships an asynchronous executor and action queue for exported OpenVINO models is unclear. The repo has an action-queue unit test and RTC graph inputs, and Intel's article shows only `SyncExecution`; the Intel RTC sample page could not be read.
- Physical AI Studio has no training-time RTC (a code search for it returned nothing), so that path means training in LeRobot and exporting through Studio or a custom script, with per-token timesteps in the exported graph.
- No evidence on combining SnapFlow, training-time RTC and VLASH-style state offsets in one fine-tune.

## Smoothing and filtering for jerky chunks

### Takeaway
I found no study showing that low-pass or spline post-filtering of quantisation-jittered chunks improves real-robot success. Published work addresses jerk at chunk boundaries, where training-free and training-time methods cut it by roughly 10–30% with success unchanged.

### Cited Findings
- SEAM (July 2026) is a training-free, gradient-free correction applied after each Euler step using the previous chunk's unexecuted tail. On LIBERO-10 with π0.5 it cut boundary jerk by 28% and chunk-transition discontinuity by 27% at about 1.01× denoising cost, with success at baseline level (95.7% against 94.8%, per a search summary). RTC reached lower jerk at 1.22× cost and 95.1% success in the same comparison. Simulation only; no code link found. — [arXiv 2607.04609](https://arxiv.org/abs/2607.04609)
- ABPolicy (February 2026) has an asynchronous flow policy predict B-spline control points instead of raw actions, reporting a 29.2% lower velocity zero-crossing rate than raw actions and smoother chunk boundaries after refitting. UNVERIFIED beyond the search summary; real-robot success numbers not seen. — [arXiv 2602.23901](https://arxiv.org/pdf/2602.23901)
- Averaging overlapping chunks (temporal ensembling) caused oscillation and protective stops on real π0.5 robots under delay. — [arXiv 2506.07339 v2](https://arxiv.org/html/2506.07339v2)
- LeRobot's remote-inference backend offers optional chunk alignment and blending for plain chunks, and the rollout loop supports an `--interpolation_multiplier` that sends interpolated targets between policy actions. — [LeRobot RTC docs](https://huggingface.co/docs/lerobot/en/rtc); [LeRobot inference docs](https://github.com/huggingface/lerobot/blob/main/docs/source/inference.mdx)
- On real SO-101 arms, grasp instability and repetition loops were the dominant π0.5 failure types (each tagged in 91.43% of failed trials), ahead of precision misalignment (14.29%). — [arXiv 2606.08881](https://arxiv.org/html/2606.08881)

### Inferences
- The team's 2.4× jerk increase is within-chunk noise from INT8 weights, which is a different problem from the boundary discontinuity these papers treat. A zero-phase low-pass or smoothing spline can be applied to the whole 50-action chunk offline before execution, so it adds no lag, unlike a causal filter in the control loop.
- Filter the arm joints only. Gripper commands are near-binary, and smoothing them delays grasp closure, which matters given grasp instability is the leading SO-101 failure.
- Within-chunk smoothing and RTC are complementary: smooth the chunk first, then pass the smoothed tail as the RTC prefix, so the next chunk continues a clean trajectory.
- The cut-off should sit above the demonstrations' motion bandwidth. Comparing the power spectrum of recorded demo actions with INT8 output is a cheap way to set it and to confirm the added energy is high-frequency.
- SO-101 servos and their position loop already low-pass commands to some degree, so measured chunk jerk may overstate motion jerk. Comparing commanded and observed joint traces is the direct check.
- Smoothing compounds amplitude loss slightly if the cut-off is too low; motion range should be re-measured after filtering.

### Gaps
- No source measures real-robot success with and without within-chunk filtering, for any VLA.
- No source connects weight quantisation to action jerk or proposes a fix on the execution side; the team's 2.4× figure appears to be new.
- SEAM and ABPolicy have no verified real-robot success numbers.

## What limits π0.5 on SO-101-class arms after an expert-only fine-tune, and cheap training-side changes that also help inference

### Takeaway
The one quantitative SO-101 benchmark puts π0.5 at 56% average success with execution failures (grasp instability, repetition loops) dominating over semantic ones, but it does not say how the model was fine-tuned, and I found no controlled study of expert-only fine-tuning on SO-101. Cheap training-side changes with evidence are delay-aware fine-tuning (training-time RTC, VLASH) and one-step distillation; shorter chunks have only indirect support.

### Cited Findings
- A June 2026 benchmark fine-tuned π0.5, SmolVLA, Wall-X and ACT on real SO-101 demonstrations, 100 per task. Success for π0.5: pen transfer 95%, colour sorting 10%, object packing 55%, pen placement 65%, average 56.25% (SmolVLA 32.5%, ACT 33.75%, Wall-X 51.25%). — [arXiv 2606.08881](https://arxiv.org/html/2606.08881)
- Its π0.5 failure tags: grasp instability 91.43% and repetition loop 91.43% of failed trials, state mismatch 45.71%, precision misalignment 14.29%. The percentages sum to more than 100% although the paper describes one primary mode per failure, so read them as indicative. π0.5 had the highest recovery rate, 30.77%. — [arXiv 2606.08881](https://arxiv.org/html/2606.08881)
- The authors conclude π0.5 has the best semantic grounding of the four but "maintains relatively high execution failure", and that execution instability is the dominant failure source across models. The paper does not state whether fine-tuning was expert-only, nor steps, chunk size, denoising steps, latency or execution mode. — [arXiv 2606.08881](https://arxiv.org/html/2606.08881)
- A secondary blog criticises the same benchmark for small per-task samples and for failure labels that are not per-episode. — [SO-101 benchmark notes blog](https://mcobzarenco-fontaine-blog.static.hf.space/papers/so101-vla-benchmark.html)
- Expert-only SO-101 checkpoints exist on the Hub (for example a one-task cup-placement model at 50 and 70 episodes) but state that on-robot success was not measured. UNVERIFIED beyond the search summary. — [pi05_so101_1task_expert_50ep](https://huggingface.co/nevertmr/pi05_so101_1task_expert_50ep_260716)
- One expert-only π0.5 checkpoint on the simulated LeIsaac PickOrange task reported 1.7% success, which its author attributes to a vision bottleneck as a hypothesis. UNVERIFIED beyond the search summary. — [Pi0.5-PickOrange](https://huggingface.co/wsagi/Pi0.5-PickOrange)
- LeRobot's π0.5 config exposes `train_expert_only` (freeze the whole VLM, train the action expert and projections) and `freeze_vision_encoder` as separate switches; Physical AI Studio's SO-101 recipe uses a full-model fine-tune for phase 1. — [LeRobot configuration_pi05.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/pi05/configuration_pi05.py); [so101/snapflow.yaml](https://github.com/open-edge-platform/physical-ai-studio/blob/main/library/configs/physicalai/pi05/so101/snapflow.yaml)
- VLASH's SO-101 and R1 Lite runs used a 24-action chunk at 30 Hz, and its offset-augmented fine-tune converged more slowly early (87.1% against 94.1% on LIBERO at 10k steps) before matching at 30k (96.6% against 96.8%). — [arXiv 2512.01031](https://arxiv.org/html/2512.01031v1)
- Published fine-tune budgets for these add-ons are far above "a few thousand steps": 30,000 steps at batch 8 in LeRobot's training-time RTC example, 30,000 at batch 32 for VLASH, 8,000 at batch 512 for Physical Intelligence's own training-time RTC, 20,000–30,000 at batch 4–8 for SnapFlow. — [LeRobot pi05 docs](https://github.com/huggingface/lerobot/blob/main/docs/source/pi05.mdx); [arXiv 2512.01031](https://arxiv.org/html/2512.01031v1); [arXiv 2512.05964 v2](https://arxiv.org/html/2512.05964v2); [arXiv 2604.05656 v2 HTML](https://arxiv.org/html/2604.05656v2)

### Inferences
- If the dominant SO-101 failures are grasp instability and repetition loops, the levers most likely to move success are execution-side (stale observations, chunk seams, gripper timing) and training length, not the 48 ms the expert costs. That argues for doing async plus RTC before or alongside distillation.
- Repetition loops are consistent with open-loop execution of long stale chunks and with mean-seeking samples that under-reach. Both the team's amplitude shrinkage at few steps and long synchronous chunks could contribute; neither link is established by a source.
- A few thousand expert-only steps is short relative to every published recipe above. Extending the fine-tune (with `rtc_training_max_delay` switched on at no extra cost per step) is the cheapest training-side change that also helps inference.
- Shorter chunks (about 24–30 actions) are supported only indirectly, by VLASH's setup. They reduce expert tokens per step, but the prefix dominates attention cost, so the latency gain is likely small; the main benefit is not learning a long tail that is never executed. Changing chunk size means retraining and re-exporting.
- Lowering action dimensionality offers little: π0.5 pads actions to 32 dimensions inside one linear projection, so the SO-101's 6 dimensions already cost the same as 32.
- Ordering for one day of cloud GPU: (1) continue the expert-only fine-tune with training-time RTC enabled and a held-out split, (2) run SnapFlow phase 2 from the best checkpoint, (3) export 10-step, 5-step and 1–2-step graphs and compare motion range, jerk and on-robot success under async execution.

### Gaps
- No controlled comparison of expert-only against full or LoRA fine-tuning for π0.5 on SO-100/SO-101 with real-robot success rates.
- No source on how many steps an expert-only π0.5 fine-tune needs at about 500 demos.
- No evidence on chunk-length ablations for π0.5 on small arms.
- Physical Intelligence's knowledge-insulation work was not retrieved in this pass, so its bearing on frozen-backbone fine-tuning is not covered here.

## Ranked shortlist for a team with a fine-tuned π0.5 checkpoint

### Takeaway
Ranked by expected benefit per unit of effort: asynchronous execution with RTC first, SnapFlow distillation second, training-time RTC third, then training-free step and solver experiments, within-chunk smoothing, and lastly caching or layer distillation.

### Cited Findings
- 1. Asynchronous execution with inference-time RTC. Training-free. Mechanism: overlap inference with motion and inpaint the overlap. Evidence: no degradation on real π0.5 up to +200 ms; naive async alone beat sync on SO-101 (89.7% against 83%). Cost: about 21 ms on a 4090 with true guidance; Studio's gradient-free variant is OpenVINO-traceable. Code: LeRobot and Physical AI Studio. Effort: runtime integration, hours to a day. — [arXiv 2506.07339 v2](https://arxiv.org/html/2506.07339v2); [arXiv 2512.01031](https://arxiv.org/html/2512.01031v1); [Studio rtc.py](https://github.com/open-edge-platform/physical-ai-studio/blob/main/library/src/physicalai/policies/mixins/rtc.py)
- 2. SnapFlow one-step distillation. Needs retraining of the expert only, VLM frozen: 20–30k steps at batch 4–8, about 12 h on one GPU, using the existing demos. Evidence: 98.75% against 97.75% (paper) and 97.55% against 97.05% (Studio) on LIBERO; about 3× model latency on GPUs. Code: Physical AI Studio config, callback and OpenVINO export. Risks: simulation-only evidence, amplitude not measured, weak teacher. — [arXiv 2604.05656 v2 HTML](https://arxiv.org/html/2604.05656v2); [Studio SnapFlow LIBERO report](https://github.com/open-edge-platform/physical-ai-studio/blob/main/library/docs/explanation/benchmark/snapflow-libero.md)
- 3. Training-time RTC. Needs fine-tuning with random clean prefixes; no inference overhead and compatible in principle with few-step sampling. Evidence: matches inference-time RTC on real π0.6 tasks and beats it at higher delays in simulation. Code: LeRobot v0.7.0 for π0.5; not in Physical AI Studio. — [arXiv 2512.05964 v2](https://arxiv.org/html/2512.05964v2); [LeRobot pi05 docs](https://github.com/huggingface/lerobot/blob/main/docs/source/pi05.mdx)
- 4. Training-free fewer steps. Five steps has first-party real-robot use on π0.5; naive one-step lost one point on LIBERO. No sourced fix for amplitude shrinkage. Effort: re-export only. — [arXiv 2506.07339 v2](https://arxiv.org/html/2506.07339v2); [arXiv 2604.05656 v2 HTML](https://arxiv.org/html/2604.05656v2)
- 5. Non-uniform time grids. Dense-jump is usable without retraining but has simulation-only evidence on small policies. Effort: re-export only. — [arXiv 2509.13574 v2](https://arxiv.org/html/2509.13574v2)
- 6. Boundary and chunk smoothing. SEAM cut boundary jerk 28% at 1.01× cost on π0.5 in simulation with success unchanged; temporal ensembling should be avoided. — [arXiv 2607.04609](https://arxiv.org/abs/2607.04609); [arXiv 2506.07339 v2](https://arxiv.org/html/2506.07339v2)
- 7. VLASH-style future-state conditioning. Needs a 30k-step fine-tune; has the only SO-101 π0.5 execution-method numbers (94% score). Code: `mit-han-lab/vlash`. — [arXiv 2512.01031](https://arxiv.org/html/2512.01031v1); [VLASH project page](https://hanlab.mit.edu/projects/vlash)
- 8. Block caching. Training-free with offline calibration, about 3.4× on Diffusion Policy, no VLA evidence. — [arXiv 2506.13456](https://arxiv.org/html/2506.13456v1)
- 9. Layer distillation of the expert. Over 2× claimed with under one point loss, but needs teacher-student training beyond a one-day budget. — [arXiv 2601.20262](https://arxiv.org/abs/2601.20262v1)

### Inferences
- Items 1 and 2 address different things and both are worth doing: RTC makes the remaining latency mostly invisible, distillation shortens the delay from about 6 control steps to about 4 and frees iGPU time.
- If both are wanted together, item 3 is the compatible RTC variant, so the one-day training plan should enable `rtc_training_max_delay` during the fine-tune that precedes SnapFlow, accepting that the combination is untested.
- Items 4 and 5 cost only exports and evaluation time, so they can run in parallel with training as a fallback if the distilled student under-reaches.
- Every acceptance test should include motion range against recorded demos and on-robot success, because offline action error has been shown to improve while behaviour narrows.

### Gaps
- No single source evaluates these techniques together, and none reports results on an Intel iGPU other than Nota's simulation-only card.
- Real-robot success evidence exists only for items 1, 3 and 7; everything on the expert-cost side (items 2, 4, 5, 8, 9) is simulation or offline.
