# Fine-tuning flow-matching VLAs (π0 / π0.5 / SmolVLA / GR00T) on real robots: data quality, failure diagnosis, evaluation (as of Nov 2026)

Evidence labels used below: **[Paper]** peer-reviewed or arXiv result; **[Docs]** official repo or library documentation; **[Maintainer]** reply from the model authors' team on GitHub; **[Anecdote]** practitioner issue, blog, or forum report (n=1, uncontrolled).

## 1. Demonstration quality: consistency, pauses, recovery, diversity, episode count, multi-task balance, trimming

### Takeaway
The strongest primary guidance (π0 paper) is that post-training data should show a *consistent and fluent strategy*, while recovery behaviour comes mostly from the diverse pretraining mix. Idle frames should be filtered: openpi does this by default for DROID. About 50 episodes per task, structured as roughly 10 repeats for each of about 5 placements, is the documented SmolVLA baseline, and 25 was too few. Practitioner reports for π0/π0.5 show simple pick-and-place often needs a few hundred episodes and correct inference plumbing. Reducing spatial variance and making trajectories more consistent helps more than adding variety at small data sizes.

### Cited Findings
- [Paper] π0: pretraining data should cover many tasks with "a diversity of behaviors"; post-training data should show "a consistent and fluent strategy." Training only on high-quality data "yields a brittle model that does not reliably recover from mistakes"; the lower-quality pretraining data supplies recovery behaviour — [π0 paper, arXiv 2410.24164](https://arxiv.org/html/2410.24164)
- [Paper] π0 task-specific fine-tuning sets range from about 5 hours for simple tasks to 100+ hours for the most complex. Sizes are reported in hours, not episodes — [π0 paper](https://arxiv.org/html/2410.24164)
- [Docs] openpi DROID training: DROID has many idle timesteps (partly caused by the VR teleop interface), and filtering them "can improve policy performance." The default recipe drops timesteps whose *next action chunk* would be largely idle (`compute_droid_nonidle_ranges.py`, `filter_dict_path`). Data filtering in general "significantly improve[s] policy performance" — [openpi examples/droid/README_train.md](https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/README_train.md)
- [Paper] ACT/ALOHA: pauses in demonstrations are a "temporally correlated confounder" that is hard to model. Action chunking is partly motivated as a mitigation — [ACT paper, arXiv 2304.13705](https://arxiv.org/pdf/2304.13705.pdf)
- [Anecdote/vendor] A checklist claims pauses longer than 0.5 s teach "do nothing" chunks and that 50 clean demos beat 500 noisy ones. No primary evidence was found for the 0.5 s threshold. It also warns about episodes "bleeding together" (operator breaks, two attempts joined because nobody pressed end) — [roboticscenter.ai teleop data checklist](https://es.roboticscenter.ai/blog/teleoperation-data-quality-checklist)
- [Paper/observational] The Penn PAL π0-in-the-wild study attributes mid-task freezing (holding an object in the air, after grasping a handle) partly to the memoryless policy: frames of "holding" in training data are mostly idle, so the most common learned action there is to do nothing — [Penn PAL, Evaluating π0 in the Wild](https://penn-pal-lab.github.io/Pi0-Experiment-in-the-Wild/)
- [Docs] SmolVLA: record about 50 episodes as a start, with "enough demonstrations per each variation." The reference dataset is 50 episodes = 5 cube positions × 10 episodes. "We tried similar dataset with 25 episodes, and it was not enough" — [LeRobot SmolVLA docs](https://huggingface.co/docs/lerobot/smolvla); the same 5×10 structure is in the [SmolVLA paper, arXiv 2506.01844](https://arxiv.org/html/2506.01844)
- [Paper] OpenVLA-OFT ALOHA fine-tuning used 20 demos (fold shorts), 30 (fold shirt), 45 (scoop, 15 per ingredient) and 300 ("put X into pot", 100 per object) — [OpenVLA-OFT, arXiv 2502.19645](https://arxiv.org/html/2502.19645v2)
- [Anecdote] SO-101 apple-in-bowl: 20 varied episodes gave about 0% (SmolVLA). 50 episodes with *tighter* object/bowl regions and more consistent trajectories gave about 30% (SmolVLA, 3/10) and about 30% (π0.5). ACT reached about 40% on the same 50 episodes. Episode count and consistency changed together, so the effects are confounded — [Tiensmith blog, Apr 2026](https://justintiensmith.github.io/vla-research/2026/04/10/are-vlas-overhyped-first-results-on-a-real-robot.html)
- [Anecdote] openpi #427: π0 pick-and-place with 200 demos (10 Hz) trained 300k steps "performs terrible"; it "can work when >400 demos … we simply optimized our inference codes" — [openpi #427](https://github.com/Physical-Intelligence/openpi/issues/427)
- [Anecdote] openpi #912: π0.5 with about 200 teleop episodes and EEF-pose actions reaches the object but never closes the gripper. The author suspects grasp-data coverage among other causes — [openpi #912](https://github.com/Physical-Intelligence/openpi/issues/912)
- [Anecdote/guide] Remove failed grasps or label them separately, because mixing them makes the action target ambiguous. Check camera keys, shapes and joint traces for consistency across episodes before training — [OpenELAB SO-101 SmolVLA guide](https://openelab.io/blogs/learn/how-to-fine-tune-smolvla-on-so-101-with-lerobot)
- [Paper] DemInf (RSS 2025) scores demos by mutual information between states and actions, which rewards demos where actions are predictable from state, i.e. consistent ones. Subsampling with it improves policies on RoboMimic and real ALOHA (RoboCrowd) data. It assumes all demos are successful — [Hejna et al., arXiv 2502.08623](https://arxiv.org/html/2502.08623v3)
- [Paper] Belkhale et al. (NeurIPS 2023) define data quality via *action divergence* (expert vs. learned policy at the same states) and *transition diversity*, and find that more state diversity does not always help — [Data Quality in Imitation Learning, arXiv 2306.02437](https://arxiv.org/abs/2306.02437v1)
- [Paper] Follow-ups: a training-free power-spectral-density demo-quality metric compared against DemInf/CUPID ([arXiv 2605.01544](https://arxiv.org/pdf/2605.01544)), and segment-level filtering "GiB" ([arXiv 2605.01529](https://www.emergentmind.com/papers/2605.01529)). Only the abstracts were seen.
- [Paper] SmolVLA cleaned noisy task strings ("task desc", "Hold", "Up") with Qwen2.5-VL-3B during pretraining. No ablation numbers are reported — [SmolVLA paper](https://arxiv.org/html/2506.01844)
- [Anecdote] Multi-task prompts: each episode must carry its own task string in `episodes.jsonl` / `tasks.jsonl` (LeRobot format). In openpi #864 (3 tasks × 30 demos, 30k steps) the prompt had no effect. The team attributed this to overfitting on a small dataset, and earlier checkpoints *did* respond to language — [openpi #864](https://github.com/Physical-Intelligence/openpi/issues/864)

### Inferences
- Practical recipe: one operator (or a scripted protocol) using one strategy; no idle starts or ends (trim until motion begins and after the task finishes); no mid-episode pauses; drop or relabel failures; structured placement grid with about 10 repeats per cell; expand variance gradually. Recovery demos are worth adding sparingly (e.g. re-grasp after a deliberate small miss), because the π0 paper says recovery comes from diverse data. This is a design inference, not a tested ratio.
- For about 200 episodes on one task, data quantity is probably not the bottleneck. Inconsistent strategies, idle frames, and inference plumbing (section 4) are more likely culprits.
- Episode boundaries matter for chunked policies: chunks that straddle the end of an episode are padded. openpi's idle filter works at the chunk level, which suggests trimming idle *tails* as well as idle heads.

### Gaps
- No controlled study was found that gives an optimal recovery-demo ratio, or a per-task episode count for π0.5 specifically.
- No primary-source evidence on demonstration *speed* (fast vs. slow teleop) for flow VLAs. Anecdotally, slow teleop yields slow policies.
- Multi-task balancing ratios for π0.5 fine-tuning: no published guidance was found beyond giving each episode its own prompt.
- The π0 paper's fine-tuning data-size curves (e.g. 1 h vs 5 h per task) were seen only through a summarizer and are not reported here as verified.

## 2. Camera setup

### Takeaway
The standard setup for these models is one fixed external (third-person) camera plus a wrist camera per arm, at 224×224 after resize-with-pad (π0/π0.5) or up to 3 named slots (SmolVLA). Wrist cameras carry most of the grasp signal. More cameras can *hurt* language grounding, because the model can latch onto spurious visual correlations (OpenVLA-OFT on ALOHA). Camera names, resolution and placement must match exactly between training and deployment.

### Cited Findings
- [Paper] OpenVLA-OFT: on LIBERO, adding the wrist image and proprio state raised average success from 95.3% to 97.1%. On ALOHA (1 top + 2 wrist cams), policies "latch onto spurious correlations" in the visual input and ignore language. Without FiLM, language following dropped to chance (33%) — [OpenVLA-OFT](https://arxiv.org/html/2502.19645v2)
- [Paper/observational] π0 in the wild: blocking the wrist camera led to freezing (0%); blocking the side camera mid-trial dropped success to 50%. The policy tolerated changes in side-camera *position* — [Penn PAL](https://penn-pal-lab.github.io/Pi0-Experiment-in-the-Wild/)
- [Paper] SmolVLA used top+wrist (SO100) or top+side (SO101), and standardized camera viewpoints/names into 3 slots during pretraining (benefit claimed, not ablated) — [SmolVLA paper](https://arxiv.org/html/2506.01844)
- [Docs] SmolVLA requires `--rename_map` to map your camera keys onto `camera1..3`, and inference must use "the same camera names and resolution as in your dataset" — [LeRobot SmolVLA docs](https://huggingface.co/docs/lerobot/smolvla)
- [Docs] π0.5 in LeRobot: images are resized to 224×224. `empty_cameras` pads missing slots. LIBERO uses 2 views + `empty_cameras=1` — [LeRobot π0.5 docs](https://huggingface.co/docs/lerobot/pi05)
- [Docs] openpi slots are `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`. Images are resized-with-pad to 224×224 — [openpi model.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/model.py)
- [Paper, abstract only] MaskVLA reports "severe trajectory overfitting" when fine-tuning on limited data, and randomly masks part of the main camera to force use of the wrist camera — [chatpaper listing](https://chatpaper.com/paper/347690)
- [Paper, abstract only] Wrist fisheye cameras overfit in simple scenes but generalize better with enough scene diversity — [arXiv 2603.02139](https://arxiv.org/pdf/2603.02139)
- [Thesis, snippet only] For π0.5 fine-tuned on a UR5e, shifting the viewpoint hurt more for the wrist camera than for the shoulder camera — [UT Tartu MSc thesis 2026](https://ims.ut.ee/www-public2/at/2026/msc/june/atprog-magistrit55-loti-05.036-lenards-patriks-skrodelis-text-20260520.pdf)
- [Anecdote] Hidden camera latency (a Python pipe buffering 64 KB of frames, plus about 4 frames of in-camera latency) caused biased first approaches in π0.5 pick-and-place. Fixing it "markedly improved performance" — [openpi #906](https://github.com/Physical-Intelligence/openpi/issues/906)
- [Anecdote] An arm occluding the object in a top-down camera was suspected as a cause of grasp misses. An angled camera was suggested for depth cues — [Tiensmith blog](https://justintiensmith.github.io/vla-research/2026/04/10/are-vlas-overhyped-first-results-on-a-real-robot.html)

### Inferences
- Use 1 external + 1 wrist camera per arm. Add a third (side) camera only if there is real occlusion, and expect it to need more data and possibly to weaken language grounding.
- Lock exposure and white balance. Native capture above 224 px is wasted, apart from crop headroom. Keep the external camera rigidly mounted, because the wrist view is sensitive to mount changes.
- Measure end-to-end camera latency on the real rig.

### Gaps
- No controlled ablation was found of 2 vs 3 cameras for π0/π0.5 fine-tuning.
- No primary evidence was found on auto-exposure or white-balance drift for VLAs, although color jitter in training (section 3) partly covers it.

## 3. Augmentation in VLA fine-tuning (openpi, LeRobot)

### Takeaway
openpi applies augmentation by default during training: color jitter on all cameras, plus a 95% random crop and ±5° rotation on non-wrist cameras only. LeRobot image transforms are **off by default** (`enable=False`), so LeRobot π0.5/SmolVLA fine-tunes run without augmentation unless you turn it on.

### Cited Findings
- [Docs/code] openpi `preprocess_observation(train=True)`: for non-wrist keys, `RandomCrop(0.95·W, 0.95·H)` → `Resize` → `Rotate(-5, 5)`. For all cameras, `ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)`, no hue. Each sample gets its own RNG. Wrist cameras are deliberately excluded from geometric augmentation — [openpi model.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/model.py)
- [Docs] LeRobot `ImageTransformsConfig(enable=True)` offers ColorJitter (brightness/contrast/saturation/hue), SharpnessJitter, and others, each with a sampling weight. Any `torchvision.transforms.v2` pipeline can be passed instead. `lerobot-imgtransform-viz` previews them — [LeRobot dataset docs](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)
- [Paper] OpenVLA-OFT's fine-tuning description in the main text does not mention image augmentation — [OpenVLA-OFT](https://arxiv.org/html/2502.19645v2)
- [Paper, snippet] A VLA scaling study found that sensory dropout does not consistently improve performance at scale — [arXiv 2602.09722](https://arxiv.org/pdf/2602.09722v1)

### Inferences
- When fine-tuning π0.5 through LeRobot rather than openpi, enable at least color jitter to match openpi's training. Avoid geometric augmentation on wrist cameras, as openpi does.
- Augmentation addresses lighting and exposure variation, not spatial placement. Placement diversity still has to come from the data.

### Gaps
- No ablation was found that quantifies augmentation's effect on real-robot π0.5 fine-tuning success.

## 4. Common failure modes: causes and fixes

### Takeaway
Most "low loss, bad robot" reports trace back to deployment plumbing rather than the model. The usual causes are normalization-stats mismatch, action-space convention (delta vs absolute, gripper sign or encoding), executing too many open-loop steps per chunk, and latency. Gripper failures, whether never closing or closing early, are the most reported symptom. Over-training on small datasets causes language to be ignored and trajectories to be memorized.

### Cited Findings
**Gripper never closes / pushes object / closes early**
- [Anecdote] π0.5: the arm reaches the object, outputs tiny actions and pushes it, and never grasps, even though open-loop replay looked fine. Suggested checks: log the raw gripper output before de-normalization, and confirm the norm stats exactly match training. A small mismatch on the gripper dimension can cause this. No maintainer reply — [openpi #912](https://github.com/Physical-Intelligence/openpi/issues/912)
- [Anecdote] Gripper closes before reaching the cube. A commenter attributes this to the model generating an "optimistic" long chunk (approach → grasp → move) that is executed open-loop, so an early error in the grasp point propagates. Reducing the horizon from 50 to 5–10 did not fix it for the poster. Another user got their best real-robot results with horizon 50 and π0.5 — [openpi #766](https://github.com/Physical-Intelligence/openpi/issues/766)
- [Anecdote] LoRA π0.5: joint predictions are accurate but gripper predictions are poor (0/1 gripper). A commenter suggests matching the gripper encoding used in pretraining — [openpi #850](https://github.com/Physical-Intelligence/openpi/issues/850)
- [Anecdote] π0-FAST-DROID in Isaac Sim: the gripper stays mostly open — [openpi #585](https://github.com/Physical-Intelligence/openpi/issues/585)
- [Docs/code comment] openpi convention: π0 models train on delta actions *relative to the first state of each chunk*. Use the `DeltaActions` transform if your data is absolute, and keep **gripper actions absolute**. LIBERO actions are already deltas, so `extra_delta_transform` should arguably be False there — [openpi #750](https://github.com/Physical-Intelligence/openpi/issues/750)
- [Docs] LeRobot π0.5 relative actions: recompute stats in relative space with `relative_exclude_joints=['gripper']` and `chunk_size` matching the policy. π0.5 uses **quantile (q01/q99) normalization** by default. Stats aggregated during recording or merging are a conservative envelope, not true quantiles, and recomputing them changes the loss scale — [LeRobot π0.5 docs](https://huggingface.co/docs/lerobot/pi05)
- [Docs] π0.5 model card warning: sampled gripper predictions can slightly exceed the expected range, and gripper direction and controller timing must match deployment. Clip on the robot side — [HF model card pi05-rm65-lora](https://huggingface.co/ireash/pi05-rm65-lora)
- [Paper/observational] π0 in the wild: grasps were often not secure (too little force on heavy objects), attributed to image-only control without force feedback — [Penn PAL](https://penn-pal-lab.github.io/Pi0-Experiment-in-the-Wild/)

**Normalization and state mismatch**
- [Maintainer] Asked about poor generalization, a π0 maintainer first asked: "are you using the right norm stats?" ([openpi norm_stats.md](https://github.com/Physical-Intelligence/openpi/blob/36dc3c037eb8a3921be9ecb94369d60cbf56f58f/docs/norm_stats.md#normalization-statistics)) — [openpi #427](https://github.com/Physical-Intelligence/openpi/issues/427)
- [Anecdote] A π0.5 deployment sent raw (un-normalized) state and used un-denormalized actions, giving "noticeably abnormal" trajectories — [openpi #817](https://github.com/Physical-Intelligence/openpi/issues/817)
- [Anecdote] "Low loss but real-robot failure almost always comes down to … obs normalization mismatch or action chunk execution timing" — [openpi #906](https://github.com/Physical-Intelligence/openpi/issues/906)

**Jerky / oscillating motion and chunk execution**
- [Paper] π0 executes chunks open-loop (H=50; re-infers after 16 steps at 20 Hz or after 25 steps at 50 Hz). **Temporal ensembling was tried and hurt performance** — [π0 paper](https://arxiv.org/html/2410.24164)
- [Paper] RTC (Black, Galliker, Levine; NeurIPS 2025): naive async chunking causes pauses or jerky out-of-distribution motion at chunk boundaries. RTC treats the next chunk as inpainting while freezing the actions already committed, needs no retraining, and works for any diffusion or flow VLA. Tested with π0.5 on 6 real bimanual tasks, robust to more than 300 ms latency — [arXiv 2506.07339](https://arxiv.org/html/2506.07339v2)
- [Docs] LeRobot supports inference-time RTC (`--inference.type=rtc --inference.rtc.execution_horizon=10 --inference.rtc.max_guidance_weight=10.0`) and *training-time* RTC for π0.5 (`policy.rtc_training_max_delay`, [arXiv 2512.05964](https://arxiv.org/abs/2512.05964)) — [LeRobot π0.5 docs](https://huggingface.co/docs/lerobot/pi05), [SmolVLA docs](https://huggingface.co/docs/lerobot/smolvla)
- [Paper] SmolVLA LIBERO: by chunk size, n=1 gives 50.0%, 10 gives 84.0%, 50 gives 80.3%, and 100 gives 74.5%. By steps executed before re-observing, 1 gives 80.3%, 10 gives 82.8%, 30 gives 70.8%, and **50 gives 51.8%**. Async inference was about 30% faster with similar success (avg 73.3% vs 78.3% sync; sorting dropped from 70% to 50%) — [SmolVLA paper](https://arxiv.org/html/2506.01844)
- [Anecdote, paired stats] LeRobot `smolvla_libero` ships `n_action_steps=50` (= chunk_size). Switching to 10 gives +20 to 23 pp (McNemar p=0.031 and p=2e-7). LeRobot defaults set `n_action_steps == chunk_size` for ACT, π0 and SmolVLA — [lerobot #4614](https://github.com/huggingface/lerobot/issues/4614)
- [Docs] LeRobot π0.5 LIBERO recipe explicitly sets `n_action_steps=10`. `--policy.pretrained_path` loads weights only and resets `n_action_steps` to the default 50 — [LeRobot π0.5 docs](https://huggingface.co/docs/lerobot/pi05)
- [Anecdote] SmolVLA was "slow, jittery, hesitant" while π0.5 was "more confident and decisive" on the same 50 episodes. Failed grasps led to a retry loop of re-approach, miss, jitter, repeat — [Tiensmith blog](https://justintiensmith.github.io/vla-research/2026/04/10/are-vlas-overhyped-first-results-on-a-real-robot.html)

**Hovering / hesitation / freezing**
- [Paper/observational] Freezing mid-task is attributed to the memoryless policy plus mostly-idle frames in similar training states, and to unfamiliar language — [Penn PAL](https://penn-pal-lab.github.io/Pi0-Experiment-in-the-Wild/)
- [Docs] LeRobot π0.5 now offers optional short-horizon memory (MEM: `use_visual_memory`, `use_proprioceptive_memory`). Adding it only at post-training corresponds to the paper's *weaker* "Posttrain-Only" ablation — [LeRobot π0.5 docs](https://huggingface.co/docs/lerobot/pi05)

**Overfitting, memorization, ignoring language**
- [Maintainer] "100,000 steps is a lot for such a small dataset. It's very likely that the model has overfit … we are using 20K or 30K in our fine-tuning examples with much larger datasets." Context: 700 demos of one task, after which the model executed only that task whatever the prompt — [openpi #427](https://github.com/Physical-Intelligence/openpi/issues/427)
- [Anecdote] Multi-task language ignored at 30k steps on 3×30 real demos. Earlier checkpoints restored language sensitivity — [openpi #864](https://github.com/Physical-Intelligence/openpi/issues/864)
- [Paper] LangGap: state-of-the-art VLAs largely ignore language and rely on visual shortcuts; fewer than 2% of instructions in common datasets are linguistically unique — [arXiv 2603.00592](https://arxiv.org/html/2603.00592)
- [Paper] OpenVLA-OFT needed FiLM to restore language following with multiple cameras (above) — [OpenVLA-OFT](https://arxiv.org/html/2502.19645v2)
- [Paper] Knowledge Insulation (π0.5+KI): stop-gradient from the action expert to the VLM backbone preserves VLM knowledge and generalizes better. This is relevant to the choice between full fine-tuning and expert-only training — [PI KI page](https://pi.website/research/knowledge_insulation)

**State over-reliance (copycat)**
- [Paper] The "copycat" problem: BC policies with observation history learn to copy the previous action instead of reacting to the scene — [Wen et al., arXiv 2010.14876](https://arxiv.org/pdf/2010.14876)
- [Paper, snippets] Where state enters the model matters. VLANeXt finds proprio fed into the VLM beats omitting it or injecting it at the action head ([arXiv 2602.18532](https://arxiv.org/pdf/2602.18532)). ThinkProprio finds text tokenization of state beats learned projectors ([arXiv 2602.06575](https://arxiv.org/html/2602.06575)). SmolVLA: state as prefix vs suffix gives 80.3% vs 73.3% with cross-attention, but 53.3% vs 74.8% with self-attention ([SmolVLA paper](https://arxiv.org/html/2506.01844)). π0.5 discretizes state into 256 bins inside the prompt ([LeRobot π0.5 docs](https://huggingface.co/docs/lerobot/pi05)).

### Inferences
- Diagnostic order for "low loss, fails on robot":
  1. Replay a training episode's observations through the deployed server and compare actions with the ground truth. This catches normalization, key-mapping, delta/absolute and gripper-sign bugs.
  2. Log raw normalized gripper output around contact.
  3. Verify the norm stats file used at inference is byte-identical to training.
  4. Measure camera and inference latency.
  5. Set `n_action_steps` to about 10–25 (π0: 16 at 20 Hz, 25 at 50 Hz) or use RTC.
  6. Only then suspect the data.
- A gripper that never closes, with tiny actions near the object, combined with good open-loop replay, points to a closed-loop distribution shift or a gripper normalization problem. A gripper that closes early points to long open-loop chunks and grasp-point error. A sparse, binary gripper dimension makes timing errors more likely. A gripper-specific loss weight is a plausible fix, but no primary source was found for it in π0/π0.5.
- Treat language-ignoring in multi-task fine-tunes as an early sign of overfitting: test earlier checkpoints with swapped prompts.

### Gaps
- No maintainer-confirmed root cause exists for the openpi gripper-never-closes reports (#912, #766).
- No primary evidence was found on gripper-dimension loss reweighting or class balancing for flow-matching VLAs.
- No direct study of copycat or state over-reliance in π0/π0.5 fine-tuning was found. State-dropout ablations specific to π0.5 are absent.

## 5. Offline metrics vs real success; evaluation protocols

### Takeaway
Held-out action MSE or MAE and validation loss are weak, sometimes inversely related, predictors of real success for imitation policies, including flow policies. Real-robot evaluation with a fixed, randomized-position protocol of about 10–24 trials per condition, scored with partial-credit rubrics, is the standard. Restricting MSE to task-critical segments (CI-MSE) correlates better than raw MSE.

### Cited Findings
- [Blog/experiment] Diffusion Policy and ACT in simulation: lower validation loss often meant *lower* success. PushT validation loss rose about 135% while success rose about 55%. Action MSE was within seed noise. Robomimic found early stopping on validation loss gave 50–100% worse policies than the best checkpoint — [HF blog, validation loss in robotics](https://huggingface.co/blog/m1b/validation-loss-robotics)
- [Paper] Flow policies: more integration steps improved success without lowering validation loss. The appendix states validation loss is not a good proxy — [Much Ado About Noising, arXiv 2512.01809](https://arxiv.org/pdf/2512.01809)
- [Paper] CI-MSE (MSE restricted to task-critical intervals) reaches Spearman -0.87 vs -0.61 for raw MSE across 27 checkpoints — [arXiv 2606.29898](https://arxiv.org/pdf/2606.29898)
- [Paper] Validation MSE is only weakly correlated with success on AutoEval ([arXiv 2503.24278](https://arxiv.org/pdf/2503.24278)). In the data-scaling-laws study MSE sometimes tracked success and sometimes did not ([arXiv 2410.18647](https://arxiv.org/pdf/2410.18647)). **Counterexample**: GM-100 found a clear inverse correlation between action MSE and success across 3 models ([arXiv 2601.11421](https://arxiv.org/pdf/2601.11421))
- [Anecdote] Open-loop results "look good" yet the grasp fails in #912; π0.5 full fine-tune loss about 0.025 still fails to pick — [openpi #912](https://github.com/Physical-Intelligence/openpi/issues/912). A community card notes that training loss of 0.0013 "does not establish task success" — [HF renbin1/pi05](https://huggingface.co/renbin1/pi05)
- [Paper] Protocols:
  - π0: 10 episodes per task, normalized score with partial credit (bussing scored per object out of 7 or 12, shirt folding with time caps) — [π0 paper](https://arxiv.org/html/2410.24164)
  - OpenVLA-OFT ALOHA: 10–24 trials per task; object placement randomized within 45×20 cm; predefined partial-completion rubric; language following scored separately — [OpenVLA-OFT](https://arxiv.org/html/2502.19645v2)
  - SmolVLA: subtask scoring (0.5 for grasp + 0.5 for place); 5 start positions — [SmolVLA paper](https://arxiv.org/html/2506.01844)
- [Anecdote] Success criterion of "apple in bowl within 60 s" over 10 trials — [Tiensmith blog](https://justintiensmith.github.io/vla-research/2026/04/10/are-vlas-overhyped-first-results-on-a-real-robot.html)
- [Anecdote] Paired evaluation (same initial states, one variable changed) with McNemar's test to compare settings. The thread also asks that eval reports include `n_action_steps`, `chunk_size` and fps so numbers are comparable — [lerobot #4614](https://github.com/huggingface/lerobot/issues/4614)

### Inferences
- Use offline action error only as a bug detector (a large gap means a plumbing error), not to rank checkpoints. If an offline metric is needed, compute it on critical windows (approach, grasp, release) and on gripper timing.
- A suggested protocol: a fixed grid of placements (for example the training grid plus held-out positions), at least 10–20 trials per checkpoint, identical scene resets, subtask partial credit, a time limit, and the inference settings logged.

### Gaps
- No study was found that validates CI-MSE specifically on π0/π0.5.

## 6. Checkpoint selection and step counts (~200 episodes)

### Takeaway
Official π0/π0.5 fine-tuning examples use about 20–30k steps, and the openpi maintainer warned that 100k steps on a small dataset very likely overfits. Save every 1–5k steps and select by real-robot rollouts, not loss. Flow-matching training loss keeps falling slowly (to about 0.02–0.03 on real data) and is not a stopping signal.

### Cited Findings
- [Maintainer] openpi uses 20k–30k steps in its fine-tuning examples "with much larger datasets"; 100k on a small dataset is likely overfit — [openpi #427](https://github.com/Physical-Intelligence/openpi/issues/427)
- [Docs] LeRobot π0.5 LIBERO: 30k steps, batch size 64, `save_freq=5000`. SmolVLA: 20k steps (about 4 h on an A100), "tune the number of steps based on performance" — [LeRobot π0.5 docs](https://huggingface.co/docs/lerobot/pi05), [SmolVLA docs](https://huggingface.co/docs/lerobot/smolvla)
- [Paper] SmolVLA real-world fine-tuning used 200k steps, noting "far fewer steps can work without major performance loss" — [SmolVLA paper](https://arxiv.org/html/2506.01844)
- [Docs] openpi DROID full fine-tune: about 100k iterations at batch size 256, about 1 epoch. LoRA "didn't perform well" for DROID — [openpi DROID README_train](https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/README_train.md)
- [Anecdote] Community π0.5 LoRA fine-tunes on small real datasets commonly run 20k steps with checkpoints every 1k — [HF pi05-droid-lora-real-table-sft-66](https://huggingface.co/IDEAS-Lab-Northwestern/pi05-droid-lora-real-table-sft-66)
- [Anecdote] 105 trials × 40 s, π0.5 LoRA: loss plateaued around 0.022 by 60k steps — [openpi #850](https://github.com/Physical-Intelligence/openpi/issues/850). 200 demos × 300k steps gave "terrible" results — [openpi #427](https://github.com/Physical-Intelligence/openpi/issues/427)
- [Paper] OpenVLA-OFT selected LIBERO checkpoints every 50k steps by test success, which is an optimistic number. For RDT-1B, an 18k checkpoint beat a 40k one on a real task — [OpenVLA-OFT](https://arxiv.org/html/2502.19645v2)
- [Paper] In offline robot fine-tuning, real performance can collapse below the base policy after too many updates, and the onset is not easily predicted — [arXiv 2004.10190](https://arxiv.org/pdf/2004.10190)
- [Anecdote] Earlier checkpoints restored language responsiveness in a multi-task fine-tune — [openpi #864](https://github.com/Physical-Intelligence/openpi/issues/864)
- [Anecdote] `train_expert_only` and `freeze_vision_encoder` settings were suspected to be suboptimal in a π0.5 run. LeRobot docs say freezing the VLM costs "some … success rate" — [Tiensmith blog](https://justintiensmith.github.io/vla-research/2026/04/10/are-vlas-overhyped-first-results-on-a-real-robot.html), [LeRobot π0.5 docs](https://huggingface.co/docs/lerobot/pi05)

### Inferences
- For about 200 episodes (roughly 40–80k frames at 30 Hz) with batch size 32–64, 20–30k steps is about 10–40 epochs. Evaluate checkpoints at roughly 5k, 10k, 15k, 20k and 30k on the robot with a short fixed protocol (about 10 trials), then run a full evaluation on the best one or two. Longer runs are mainly a risk to language following and generalization.

### Gaps
- No published learning curve was found of π0.5 real-robot success vs steps for about 200 episodes. Validation-loss behaviour specific to flow matching (noise-sampled loss variance) is not characterized in the sources found.
