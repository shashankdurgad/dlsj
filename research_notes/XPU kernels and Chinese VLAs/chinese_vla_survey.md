# Open VLA Models from Chinese Labs (2025–2026) vs pi0.5 / SmolVLA / GR00T N1.x

Research date: 2026-11-13. About 16 search/fetch calls. Most benchmark numbers are **self-reported by the model authors** unless marked as third-party. Comparison table first, then the per-question sections.

### Comparison table (verified fields only; "?" = not verified in this pass)

| Model (org, date) | Params | Backbone VLM | Action head | Pretraining data | Licence (code / weights) | Headline self-reported results | Third-party results |
|---|---|---|---|---|---|---|---|
| **X-VLA** (Tsinghua AIR, Oct 2025, ICLR 2026) | ~0.9B (?) | Soft-prompted Transformer (cross-embodiment soft prompts) | Flow matching | Cross-embodiment mix (?) | Open on GitHub; in LeRobot (licence ?) | LIBERO 98.1% avg (full FT); Simpler-WidowX 95.8% (full FT); LoRA (~1% params) 93% LIBERO / 54% WidowX | RoboDojo: sim 6.52% (4th), real 3.3% |
| **Xiaomi-Robotics-0** (Xiaomi, Feb 2026) | 4.7B | VLM + multi-layer DiT (Mixture-of-Transformers) | Diffusion/flow DiT | ? | Open code + checkpoints (licence ?) | LIBERO 98.7%; SimplerEnv Google VM 85.5 / VA 74.7, WidowX 79.2; CALVIN ABC-D 4.75, ABCD-D 4.80; 80 ms latency on RTX 4090 | RoboDojo: sim 4.18%, real 3.9% |
| **LingBot-VLA** (Ant Group / Robbyant, Jan 2026); **2.0** Jul 2026 | ? | ? (depth variant) | ? | ~20,000 h real data, 9 platforms | Open source: model, post-training code, GM-100 benchmark | GM-100 real (3 platforms): 15.7% / 17.3% (depth) vs pi0.5 13.0%; RoboTwin 2.0 +5.82 (clean) / +9.92 (randomized) pts over pi0.5. 2.0 claims to beat pi0.5 and GR00T N1.7 on GM-100 dual-arm | RoboDojo sim 2.96% (real not evaluated) |
| **InternVLA-A1-3B** (Shanghai AI Lab, Jan 2026); **A1.5** Jul 2026 | 3B | ? | Unified understanding + generation + action | ? | Weights on HF (licence ?) | RoboTwin 2.0: 89.40 Easy / 89.64 Hard vs pi0.5 86.76 / 86.96, pi0 79.98 / 79.50. A1.5: 93.2 vs Being-H0.7 89.9 | RoboDojo: sim 1.08%, **real 7.2% (2nd behind pi0.5)** |
| **WALL-OSS** (X Square Robot, Sep 2025); **WALL-OSS-0.5** May 2026 | ~4B | Qwen2.5-VL-3B | Flow matching (FLOW) or FAST tokens (FAST) | ? | **Apache-2.0** (per HF card / LeRobot docs) | Passed pi0 on RoboChallenge (Jan 2026) but well behind pi0.5 | SO-101 benchmark (arXiv 2606.08881): Wall-X 51.25% vs pi0.5 56.25%, SmolVLA 32.5% (figures from a blog, not checked against the paper) |
| **Spirit v1.5** (Spirit AI, Jan 2026) | ~4B+ | Qwen3-VL-4B-Instruct | DiT | ? | Weights + eval code open (licence not confirmed) | RoboChallenge Table30 #1 at release: 50.33% SR, score 66.09, ahead of pi0.5 | Results come from RoboChallenge, run by a third party |
| **DM0** (Dexmal + StepFun, Feb 2026); **DM0.5** | ~2B (DM0; 2.4B per a weak source); DM0.5 ~4B (weak source) | VLM | Flow-matching expert | ? | Open (Dexbotic toolbox + HF weights; licence ?) | Table30 specialist 62.0%, generalist 37.3% SR / 49.08 vs pi0.5 42.67 specialist / 17.7 generalist; DM0.5 Table30 v2 43% / 54.42 (secondary source); DM0.5 "topped RoboDojo" (government press) | RoboDojo paper: DM0 sim 0.05%, real 0.0% (conflicts with the DM0.5 claim) |
| **GigaBrain-0.1** (GigaAI) | 3B | ? | ? | ? | ? | Table30 specialist 51.67% | — |
| **RDT2** (Tsinghua TSAIL, Feb 2026) | 7B | Qwen2.5-VL-7B-Instruct | RDT2-VQ: residual-VQ autoregressive tokens; RDT2-FM: flow-matching expert | >10,000 h UMI human demos | Apache-2.0 (aggregator only) | Zero-shot cross-embodiment (needs specified gripper and camera hardware) | — |
| **Galaxea G0.5** (Galaxea, Jun 2026) | ? | Single decoder (autoregressive) | Autoregressive reasoning + action tokens | ? | **Non-commercial** G0.5 Community Licence; gated HF | LIBERO 98.9%; DROID zero-shot 82.5%; real R1 FT 76.7% (secondary) | RoboDojo (G0): sim 2.96%, real 4.4% |
| **VLA-Adapter** (Sep 2025) | 0.5B | Qwen2.5-0.5B | Bridge-attention policy | No robot pretraining | ? | LIBERO ~97.3% (vs OpenVLA-OFT 7B 97.1%); trains in ~8 h on 1 GPU | — |
| **Evo-1** (Nov 2025) | 0.77B | InternViT-300M + Qwen2.5-0.5B | Diffusion transformer | No robot pretraining | ? | LIBERO 94.8%, Meta-World 80.6%; 2.3 GB VRAM, 16.4 Hz on RTX 4090d | — |
| *Reference:* pi0.5 (PI) | 3B | PaliGemma | Flow matching | — | Open (openpi) | — | **Leads RoboDojo real (12.8%)**, 3rd in sim; top of SO-101 benchmark |
| *Reference:* SmolVLA (HF) | ~0.45B | SmolVLM2 | Flow matching | — | Apache-2.0 | — | RoboDojo sim 0.85%; SO-101 32.5% |
| *Reference:* GR00T N1.7 (NVIDIA) | ? | — | Diffusion/flow | — | Open | — | RoboDojo sim 1.31%, real 1.7% |

---

## Which models have open weights and permissive licences usable at a hackathon?

### Takeaway
Almost every major Chinese VLA from 2025–2026 has open weights, but the licences differ. WALL-OSS / WALL-OSS-0.5 are confirmed Apache-2.0 and already integrated in LeRobot. X-VLA is also integrated in LeRobot. RDT2 is reported as Apache-2.0. Galaxea G0.5 is non-commercial and gated. That is still fine for a non-commercial hackathon, but it adds friction. For Spirit v1.5, Xiaomi-Robotics-0, DM0, LingBot-VLA and InternVLA-A1, I could not confirm the licence name in this pass.

### Cited Findings
- WALL-OSS (released 15 Sep 2025) is listed as Apache-2.0 with open weights. It is a 4B model on a Qwen2.5-VL-3B backbone and comes in FLOW and FAST variants — [roboticscenter.ai X Square](https://www.roboticscenter.ai/institutions/models/x-square-robot); LeRobot docs tie it to the WallX repo — [LeRobot WALL-OSS docs](https://huggingface.co/docs/lerobot/walloss)
- WALL-OSS-0.5 (29 May 2026): 4B VLA on a 3B VLM, Apache-2.0. It uses flow matching for continuous actions and FAST for discrete actions — [roboticscenter.ai](https://www.roboticscenter.ai/institutions/models/x-square-robot); [arXiv 2605.30877](https://arxiv.org/pdf/2605.30877)
- WALL-OSS FAST variant: injects discrete action priors through FAST tokenization — [arXiv 2509.11766](https://arxiv.org/html/2509.11766v1)
- X-VLA has public checkpoints on GitHub and a LeRobot integration. The LeRobot docs quote 93% on LIBERO for the LIBERO checkpoint — [GitHub 2toinf/X-VLA](https://github.com/2toinf/X-VLA); [LeRobot X-VLA docs](https://huggingface.co/docs/lerobot/xvla)
- Xiaomi-Robotics-0: code and checkpoints are on GitHub and HF, with a base model plus LIBERO, CALVIN and SimplerEnv fine-tunes. The full post-training pipeline was released on 27 Apr 2026 — [Xiaomi project page](https://robotics.xiaomi.com/xiaomi-robotics-0.html); [Gizmochina](https://www.gizmochina.com/2026/02/12/xiaomi-robotics-0-open-source-vla-model/)
- Spirit v1.5: weights are at HF Spirit-AI-robotics/Spirit-v1.5 and code at GitHub Spirit-AI-Team/spirit-v1.5. The licence name was not visible — [PR Newswire](https://www.prnewswire.com/news-releases/robochallenges-top-ranked-embodied-ai-model-goes-open-source-challenging-clean-data-collection-paradigm-302658247.html); [CGTN](https://news.cgtn.com/news/2026-01-14/Chinese-AI-startup-tops-global-embodied-intelligence-benchmark-1JV32RBb7QQ/p.html)
- Galaxea G0.5: HF repo gated behind an agreement. Content from 2026-06-16 onward is under the "G0.5 Community License (Non-Commercial + Limited Patent License)". Checkpoints: g05-base, g05-libero, g05-droid — [HF OpenGalaxea/G05](https://huggingface.co/OpenGalaxea/G05); [sourcepulse GalaxeaVLA](https://www.sourcepulse.org/projects/31538735)
- RDT2 licence is listed as Apache-2.0 by an aggregator only. Inference needs about 16 GB VRAM (RTX 4090); full fine-tuning of RDT2-VQ needs more than 80 GB — [roboticscenter.ai TSAIL](https://www.roboticscenter.ai/institutions/models/tsinghua-tsail); [GitHub RDT2](https://github.com/eesoymilk/RDT2)
- LingBot-VLA: model, post-training code, technical report and GM-100 benchmark are all open-sourced — [QbitAI](https://www.qbitai.com/2026/01/373396.html); [Aliyun dev article](https://developer.aliyun.com/article/1709507)
- DM0: code is in Dexmal's Dexbotic toolbox and weights are on HF. DM0.5 is described as fully open (weights, training framework, workflow) — [arXiv 2602.14974](https://arxiv.org/html/2602.14974v1); [Liangjiang gov](https://english.liangjiang.gov.cn/2026-08/28/c_1209436.htm)
- InternVLA-A1-3B and a RoboTwin fine-tune are on HF — [HF InternVLA-A1-3B](https://huggingface.co/InternRobotics/InternVLA-A1-3B)

### Inferences
- Safest hackathon picks on licence and tooling: WALL-OSS(-0.5) (Apache-2.0, in LeRobot, and already benchmarked on SO-101 by a third party) and X-VLA (in LeRobot). SmolVLA and pi0.5 remain the easiest baselines.
- Check the LICENSE files for Spirit v1.5, Xiaomi-Robotics-0, DM0, LingBot and InternVLA-A1 before relying on them commercially. For hackathon or research use they appear to be openly downloadable.

### Gaps
- Exact licence names (Apache, MIT or custom) for Spirit v1.5, Xiaomi-Robotics-0, DM0/DM0.5, LingBot-VLA, InternVLA-A1/A1.5 and X-VLA weights were not confirmed.
- **Not researched in this pass (no verified sources):** AgiBot GO-1 / Genie Operator, ByteDance Seed GR-3, RoboBrain 2.x (BAAI), EO-1, UniVLA, Being-H0/H0.5/H0.7, GraspVLA (Galbot), DexVLA / ChatVLA / DiVLA (Midea), Hume, villa-X, F1-VLA, FlowVLA, Motus, Hy-Embodied-0.5-VLA (Tencent?), Unitree/Huawei/Alibaba DAMO releases. From training knowledge only (unverified): GR-3 weights were not released, and GO-1 was open-sourced by AgiBot. Verify both before citing.
- RDT2's Apache-2.0 licence is aggregator-only.

## Which report results that clearly beat pi0.5 / SmolVLA, on which benchmarks, and how credible are the comparisons?

### Takeaway
Many Chinese VLAs claim to beat pi0.5 on saturated simulation benchmarks (LIBERO ~98–99%, SimplerEnv, RoboTwin 2.0). Those numbers come from the authors' own fine-tuning setups and are weak evidence. The most credible wins are on RoboChallenge Table30, a real-robot benchmark run by a third party. There, Spirit v1.5 (Jan 2026) and then DM0 (Feb 2026) went past pi0.5, and GigaBrain-0.1 also scored higher. However, the independent RoboDojo benchmark (HKU MMLab, Jul 2026) found that **pi0.5 was still clearly best on real robots**. Most Chinese models scored far below their self-reported numbers there.

### Cited Findings
- **RoboChallenge Table30** (30 real tasks on UR5, Franka, ARX5 and Cobot Magic ALOHA) — [RoboChallenge paper arXiv 2510.17950](https://arxiv.org/html/2510.17950v1); [Emergent Mind Table30](https://www.emergentmind.com/topics/table30)
  - Specialist SR: DM0 (2B) 62.00%, GigaBrain-0.1 (3B) 51.67%, Spirit-v1.5 (4B) 51.00%, pi0.5 (3B) 42.67%. Generalist SR / score: DM0 37.3 / 49.08, pi0.5 17.7 / 31.3, pi0 9.0 / 20.2 — [Emergent Mind Table30](https://www.emergentmind.com/topics/table30) (citing the DM0 paper, 16 Feb 2026, and the RoboChallenge paper)
  - Spirit v1.5: 50.33% SR, total score 66.09. It was #1 at release and the only model above 50% — [CGTN](https://news.cgtn.com/news/2026-01-14/Chinese-AI-startup-tops-global-embodied-intelligence-benchmark-1JV32RBb7QQ/p.html); [PR Newswire](https://www.prnewswire.com/news-releases/robochallenges-top-ranked-embodied-ai-model-goes-open-source-challenging-clean-data-collection-paradigm-302658247.html)
  - WALL-OSS-Flow passed pi0 in Jan 2026 but trailed pi0.5 by a wide margin — [Leiphone](https://m.leiphone.com/category/industrynews/2XqXrduDStzpWS53.html)
  - DM0.5 is reported at 43% overall and 54.42 score on Table30 v2 (18 new bimanual tasks). This comes from a republished secondary summary — [besthub](https://www.besthub.dev/articles/how-six-robots-built-a-3-5-meter-great-wall-in-15-hours-using-vla-world-model-cefd414c17e6)
- **RoboDojo** (independent sim+real benchmark led by HKU MMLab with Berkeley, Stanford, MIT, CMU and Tsinghua collaborators; arXiv 2607.04434, Jul 2026) — [arXiv 2607.04434](https://arxiv.org/html/2607.04434v1):
  - Real-world overall SR / score: pi0.5 12.8 / 22.9 (clear leader); InternVLA-A1 7.2 / 12.0; Galaxea G0 4.4 / 9.0; Xiaomi-Robotics-0 3.9 / 7.9; X-VLA 3.3 / 7.6; pi0 1.7 / 5.8; GR00T-N1.7 1.7 / 5.9; DM0 0.0 / 0.0.
  - Sim overall SR: Hy-Embodied-0.5-VLA 8.80% (top), Spatial Forcing 8.04%, pi0.5 6.91%, X-VLA 6.52%, Xiaomi-R0 4.18%, G0 2.96%, LingBot-VLA 2.96%, pi0 1.53%, GR00T-N1.7 1.31%, InternVLA-A1 1.08%, SmolVLA (single task) 0.85%, DM0 0.05%. Human experts reach 76.03%.
  - The authors conclude: "current policies remain far from reliable task completion across diverse manipulation scenarios".
  - Conflict: a Chongqing government press release says DM0.5 "topped the RoboDojo rankings" — [Liangjiang gov](https://english.liangjiang.gov.cn/2026-08/28/c_1209436.htm). This is probably a later leaderboard entry than the paper, and it is unverified.
- **SO-101 benchmark** (Hiroshima Univ., arXiv 2606.08881): fine-tuned pi0.5, SmolVLA, Wall-X and ACT on 4 real SO-101 tasks. A blog reports average SR of pi0.5 56.25%, Wall-X 51.25%, ACT 33.75%, SmolVLA 32.5%; I did not check these against the paper. The paper finds execution instability is the dominant failure source — [HF papers 2606.08881](https://huggingface.co/papers/2606.08881); [blog summary](https://mcobzarenco-fontaine-blog.static.hf.space/papers/so101-vla-benchmark.html)
- **Self-reported sim claims vs pi0/pi0.5:**
  - InternVLA-A1-3B on RoboTwin 2.0: 89.40 / 89.64 (Easy / Hard) vs pi0.5 86.76 / 86.96 and pi0 79.98 / 79.50 — [HF card](https://huggingface.co/InternRobotics/InternVLA-A1-3B); [arXiv 2601.02456](https://arxiv.org/abs/2601.02456)
  - InternVLA-A1.5: RoboTwin 93.2 vs Being-H0.7 89.9 and InternVLA-A1 89.5 — [arXiv 2607.04988](https://arxiv.org/html/2607.04988v1)
  - LingBot-VLA on RoboTwin 2.0: beats pi0.5 by 5.82 (clean) and 9.92 (randomized) points with depth. On GM-100 real (3 platforms): 15.7% / 17.3% vs pi0.5 13.0% — [arXiv 2601.18692](https://arxiv.org/html/2601.18692)
  - LingBot-VLA 2.0 claims higher progress and SR than pi0.5 and GR00T N1.7 on GM-100 dual-arm (AgileX Cobot Magic, Galaxea R1 Pro) — [IT Brief](https://itbrief.co.uk/story/robbyant-open-sources-robot-model-for-multiple-types)
  - Xiaomi-Robotics-0: LIBERO 98.7%, SimplerEnv Google VM 85.5 / VA 74.7, WidowX 79.2, CALVIN 4.75 / 4.80 — [Gasgoo](https://autonews.gasgoo.com/articles/ev/xiaomi-robotics-open-sources-vla-model-xiaomi-robotics-0-2021913256057122817); [arXiv 2602.12684](https://arxiv.org/pdf/2602.12684)
  - X-VLA: LIBERO 98.1%, Simpler WidowX 95.8% (full FT); LoRA 93% / 54% — [arXiv 2510.10274](https://arxiv.org/html/2510.10274v1)
  - Galaxea G0.5: LIBERO 98.9%, DROID zero-shot 82.5% (secondary) — [sourcepulse](https://www.sourcepulse.org/projects/31538735)

### Inferences
- LIBERO is saturated: every recent model reports about 97–99%. It no longer separates SOTA models, and the gains over pi0.5 there are within protocol noise.
- On RoboTwin 2.0, the margins over pi0.5 are 2–10 points and come only from the authors' own runs.
- The independent RoboDojo results contradict most "beats pi0.5" claims. For example, DM0 led Table30 but scored 0% on RoboDojo real. The likely reason is that self-reports and Table30 rely on per-task fine-tuning and familiar embodiments, while RoboDojo tests harder long-horizon generalization.
- The most credible Chinese results are Spirit v1.5 and DM0 on Table30 (third-party run, but they are specialist fine-tunes) and InternVLA-A1 as the runner-up in RoboDojo real. Among Chinese entries, X-VLA has the strongest sim robustness.
- **"Genuinely SOTA as of late 2026":** I found no independent evidence that any open Chinese VLA beats pi0.5 across benchmarks. The claims are benchmark-specific: Spirit v1.5, DM0/DM0.5 and GigaBrain-0.1 on RoboChallenge; InternVLA-A1.5 and LingBot-VLA on RoboTwin 2.0; Xiaomi-R0 on CALVIN / SimplerEnv. Each lab beats pi0.5 on its own chosen benchmark.

### Gaps
- I could not fetch the live robochallenge.ai leaderboard (the page rendered empty), so the late-2026 Table30 / Table30 v2 rankings are unknown.
- RoboDojo's later leaderboard (DM0.5, Hy-Embodied, Being-H0.7, WALL-OSS-0.5) is not verified.
- RoboArena and LeRobot community evaluations of Chinese models were not found in this pass.
- No head-to-head CALVIN results against pi0.5 or SmolVLA were found apart from Xiaomi's self-reported numbers.

## Parameter counts and which are small enough (<~3B) for real-time inference on a single consumer GPU

### Takeaway
Under ~1B: VLA-Adapter (0.5B), Evo-1 (0.77B), SmolVLA (~0.45B), and X-VLA (sub-1B, unverified). These are the clear fits for consumer or integrated GPUs. Around 2–3B: DM0 (~2B), InternVLA-A1-3B, GigaBrain-0.1 (3B), and pi0.5 (3B). At 4B and above: WALL-OSS (4B), Spirit v1.5 (4B VLM plus a DiT head), Xiaomi-R0 (4.7B; 80 ms on an RTX 4090) and RDT2 (7B; about 16 GB to run). These need a discrete 16–24 GB GPU.

### Cited Findings
- VLA-Adapter: 0.5B (Qwen2.5-0.5B), LIBERO ~97.3%, trains in ~8 h on one GPU, no robot pretraining — [arXiv 2509.09372](https://arxiv.org/pdf/2509.09372v2)
- Evo-1: 0.77B (InternViT-300M + Qwen2.5-0.5B + DiT). Uses 2.3 GB VRAM and runs at 16.4 Hz on an RTX 4090d. LIBERO 94.8%, Meta-World 80.6% — [arXiv 2511.04555](https://arxiv.org/html/2511.04555v1)
- DM0: 2B per the Table30 summary — [Emergent Mind](https://www.emergentmind.com/topics/table30); DM0.5: 4B per a weak source — [we0.ai](https://we0.ai/zh-HK/articles/dexmal-dm05-embodied-foundation-model)
- InternVLA-A1: 3B — [HF](https://huggingface.co/InternRobotics/InternVLA-A1-3B)
- Spirit v1.5: Qwen3-VL-4B backbone plus DiT head — [CGTN](https://news.cgtn.com/news/2026-01-14/Chinese-AI-startup-tops-global-embodied-intelligence-benchmark-1JV32RBb7QQ/p.html); [Emergent Mind (4B)](https://www.emergentmind.com/topics/table30)
- Xiaomi-Robotics-0: 4.7B, 80 ms latency and 30 Hz control on an RTX 4090 — [lavx news](https://news.lavx.hu/article/xiaomi-open-sources-4-7b-parameter-embodied-ai-model-with-80ms-latency)
- RDT2: 7B (Qwen2.5-VL-7B); inference ~16 GB VRAM — [GitHub RDT2](https://github.com/eesoymilk/RDT2)
- WALL-OSS: 4B (3B VLM) — [roboticscenter.ai](https://www.roboticscenter.ai/institutions/models/x-square-robot)

### Inferences
- For a single consumer or integrated GPU, the best options are VLA-Adapter, Evo-1, SmolVLA and X-VLA. The ~2–3B tier (DM0, InternVLA-A1-3B, pi0.5) is likely workable on 12–16 GB GPUs with bf16 and action chunking. This is inferred, not measured. Intel compatibility is covered by the other researcher.

### Gaps
- X-VLA's exact parameter count (I believe ~0.9B) was not confirmed this session.
- I found no latency measurements on integrated or Intel GPUs for any Chinese VLA.

## Any independent third-party benchmarks comparing these

### Takeaway
There are three useful third-party sources. RoboChallenge Table30 (Dexmal-affiliated organisers, real robots, remote submission). RoboDojo (HKU MMLab and others, sim+real, Jul 2026), which is the most comprehensive cross-lab comparison and favours pi0.5. The SO-101 benchmark (arXiv 2606.08881), which compares pi0.5, SmolVLA, Wall-X and ACT on the cheap arm most relevant to a LeRobot hackathon.

### Cited Findings
- RoboChallenge: the online real-robot evaluation system; Table30 is its first benchmark. A leaderboard snapshot dated 2026-01-23 was cited, and a Table30 v2 with 18 bimanual tasks was planned for 2026 — [arXiv 2510.17950](https://arxiv.org/html/2510.17950v1); [speakerdeck annual report](https://speakerdeck.com/takasumasakazu/robochallenge-en)
- RoboDojo: maintained by the AI MMLab Club. Results are in the previous section — [arXiv 2607.04434](https://arxiv.org/html/2607.04434v1)
- SO-101: results are in the previous section — [HF papers 2606.08881](https://huggingface.co/papers/2606.08881)
- Other 2026 evaluation papers turned up but were not read: RoboSPA (long-horizon, arXiv 2609.05324) and "From Foundation to Application: Improving VLA Models in Practice" (arXiv 2607.06403) — [arXiv 2609.05324](https://arxiv.org/pdf/2609.05324); [arXiv 2607.06403](https://arxiv.org/pdf/2607.06403)

### Inferences
- RoboChallenge's link to Dexmal is relevant to DM0's top ranking there. The RoboChallenge paper's author list (Yakefu et al.) should be checked for overlap with Dexmal before treating DM0's lead as fully independent. I inferred this link and did not verify it.

### Gaps
- RoboArena results for Chinese models were not found.
- RoboSPA and arXiv 2607.06403 were not read.
