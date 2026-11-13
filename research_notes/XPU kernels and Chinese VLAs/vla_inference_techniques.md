# Model-Structure-Aware Inference Optimisations for VLA Policies (VLM backbone + flow/diffusion action expert) and their mapping to Intel GPUs

Scope: batch size 1, ~30 Hz control with action chunks. Models: pi0, pi0.5, SmolVLA, GR00T N1.x, plus OpenVLA/CogACT where they are the only test bed. Compiler and OpenVINO plugin internals are out of scope. Research budget: about 15 tool calls. Many numbers come from search snippets or secondary summaries, and they are flagged where that applies.

## Overall ranking: which techniques cut end-to-end action-chunk latency at batch size 1

### Takeaway
Ranked by expected latency gain per unit of effort for a pi0/pi0.5-class model on an Intel GPU:
1. Prefix KV cache, applied once per chunk. This is already standard.
2. Whole-pipeline graph capture with static shapes, plus fusion. This gave about 4x versus naive PyTorch on a 4090.
3. Fewer denoising steps or 1-step distillation. This cuts the action-expert term by up to 10x.
4. Weight-only or W4A8 quantisation. This works for pi0.5 but is fragile for pi0.
5. Visual token reduction and smaller or truncated VLMs.
6. Layer skipping or early exit.
7. Asynchronous inference or RTC. This hides latency rather than reducing it, and is needed on top of everything else.

### Cited Findings
- Reference cost breakdown, pi0 on an RTX 4090 with 2 views and 10 denoising steps, after optimisation: vision encoder 4.06 ms, LLM prefill 12.50 ms, action expert (10 steps) 11.00 ms, total about 27.3 ms. The roofline values are 2.49, 10.73 and 6.49 ms, about 19.7 ms total. — [Running VLAs at Real-time Speed, arXiv 2510.26742](https://arxiv.org/html/2510.26742v1)
- In the same paper, end-to-end latency for naive PyTorch was 105.0, 106.5 and 113.9 ms for 1, 2 and 3 views. openpi/JAX measured 43.8, 53.7 and 67.6 ms. The optimised version measured 20.0, 27.3 and 36.8 ms. — [arXiv 2510.26742](https://arxiv.org/html/2510.26742v1)
- A survey baseline table lists OpenVLA (7B) at 166 ms (about 6 Hz) and pi0 (3.3B) at 73 ms on an RTX 4090. — [A Survey on Efficient VLA Models, arXiv 2510.24795](https://arxiv.org/html/2510.24795v3)
- Intel reference point: stock PyTorch pi0.5 (3 cameras at 224x224, 64 language tokens, 10 denoising steps, BF16/FP16) takes 294 ms on a Core Ultra X7 358H (Panther Lake) at 40 W, versus 287 ms on Jetson Thor. — [Intel OEP article: Optimizing pi0.5 on Core Ultra Series 3](https://docs.openedgeplatform.intel.com/dev/OEP-articles/publications/optimizing-pi0.5-lva-model.html)

Summary table. Each row gives the mechanism, the reported speedup and accuracy impact, the models it applies to, the effort, and whether it is hardware-agnostic (HW-agn) or needs kernel work (Kernel).

| Rank | Technique | Reported speedup | Accuracy impact | Applies to | Effort | HW-agn / Kernel | Code |
|---|---|---|---|---|---|---|---|
| 1 | Prefix KV cache: compute the VLM prefix once per chunk and reuse it across N denoise steps | Avoids re-running about 12.5 ms of prefill N times. Baseline in all serious implementations | None (exact) | pi0, pi0.5, SmolVLA, GR00T (cross-attn variant) | Low (present in openpi/LeRobot) | HW-agn, but needs a static-shape KV buffer for graph capture | openpi, LeRobot |
| 2 | Graph capture of the whole pipeline, static shapes, op folding/fusion (RMSNorm folding, fused QKV, fused gated MLP, time-embedding folding) | 105 to 27 ms (about 3.9x) on a 4090. CUDA graphs alone about 2x. Folding about 7-8 ms. Gated-MLP fusion 1.7 ms | None (equivalent rewrites) | Any; demonstrated on pi0 | Medium to high | Graph capture is a runtime feature (XPU graph or OpenVINO single model). Fusions are partly algebraic (HW-agn), partly kernels | github.com/Dexmal/realtime-vla |
| 3 | Fewer steps / 1-step distillation (SnapFlow, OFP; 10 to 1 NFE) | Action-expert term cut about 10x. SnapFlow reports pi0.5 end-to-end 274 to 83 ms (secondary source) | pi0.5 LIBERO 98.75% (1-step) vs 97.75% (10-step teacher). Naive 1-step 96.75% | pi0, pi0.5, SmolVLA, GR00T (already 4 steps) | Medium (about 12 h of GPU fine-tuning) | HW-agn | SnapFlow paper |
| 4 | Quantisation (W4A8, W4A4, INT8, FP8/NVFP4) | GR00T N1.7 on Thor: 125 to 39.9 ms with mixed NVFP4 + TensorRT. Memory 4.27 to 1.28 GB for pi0.5 W4A8 | pi0.5 robust (97.6% W4A8 QuantVLA; 98.0% W4A4 HoloQ). pi0 fragile under RTN W4A8 (Spatial 78 to 64.5) | Model-dependent | Medium | Needs low-bit GEMM kernels to realise speedups | QuantVLA, HoloQ-VLA, FoldQuantVLA |
| 5 | Visual token reduction (VLA-Pruner, APS-VLA, VLA-Cache, EfficientVLA, SP-VLA, SmolVLA pixel shuffle to 64 tokens) | 1.4-1.9x on OpenVLA/CogACT. APS-VLA 1.37x on pi0.5 | Usually <1 pt drop | Mainly AR VLAs; pi0/pi0.5 for VLA-Pruner and APS-VLA | Low to medium (training-free) | HW-agn, but dynamic token counts break static graphs | Varies |
| 6 | Truncated VLM / layer skip / early exit (SmolVLA L/2, EfficientVLA layer pruning, DeeR-VLA, MoLe-VLA) | SmolVLA: "halves" VLM compute | Small with training | SmolVLA natively; others need fine-tuning | Medium | HW-agn (static truncation). Dynamic exit hurts graph capture | LeRobot (SmolVLA) |
| 7 | Async inference / RTC / training-time RTC | Does not reduce latency. SmolVLA: about 30% faster task completion, 2x throughput | RTC keeps smoothness under delay | Any chunked flow policy | Low to medium | HW-agn | LeRobot async, openpi RTC, Intel pi05-rtc-ov |

### Inferences
- On a pi0-class model with about 3 views, the time splits roughly 15% vision, 45% prefill and 40% denoise at 10 steps, using the 4090 numbers. Once denoise steps drop to 1-2, prefill dominates. After that point, token reduction, a truncated VLM and low-bit weights for the LLM are the levers that remain.
- Intel's stock-PyTorch pi0.5 figure of 294 ms is about 10x off a 30 Hz budget. Kernel work alone is unlikely to close that gap, judging by the 4090 roofline of about 20 ms on far higher bandwidth (153 GB/s vs about 1 TB/s). Model-level cuts (fewer steps, quantisation, fewer tokens) plus RTC to hide the remainder are needed on Intel iGPUs.
- At batch size 1 the workload is memory-bandwidth-bound. Weight-only INT4/INT8 should give near-proportional speedups on Intel iGPUs for the prefill and action-expert GEMMs. This is an inference; I found no Intel-specific VLA quantisation latency numbers.

### Gaps
- I found no independent, apples-to-apples latency table that applies all of these techniques to the same model and hardware.

## Prefix KV caching: how pi0/openpi and LeRobot implement it, and the cost breakdown

### Takeaway
The pi0-family structure splits tokens into a prefix of images, language and (for pi0.5, discretised) state, and a suffix of noisy actions (plus continuous state in pi0). The prefix K/V is computed once per observation. The action expert then runs N (default 10) Euler steps that attend to the cached prefix, so only the suffix is recomputed per step. The prefix is the dominant cost once steps are few.

### Cited Findings
- Image, language and state tokens form a prefix whose K/V is computed once. The noisy action tokens form a suffix, and each denoising pass attends to the cached prefix. The action expert runs 10 times against the same cache. — [mlrf pi0 explainer](https://mlrf.pages.dev/pi_zero/); [fastcrest "Decomposed pi0.5"](https://fastcrest-docs.pages.dev/concepts/decomposed/) (secondary/third-party docs)
- `num_inference_steps` / `num_steps` defaults to 10 in the HF transformers pi0 port. `sample_actions` takes images, image masks, language tokens and masks, state, and optional noise. — [HF transformers pi0 docs](https://huggingface.co/docs/transformers/v5.13.1/model_doc/pi0); [OpenTau pi0 docs](https://opentau.readthedocs.io/en/latest/_autosummary/opentau.policies.pi0.modeling_pi0.html)
- Pitfall: reusing prefix K/V across environment steps (stale observations) while re-running the suffix broke suffix attention, and the pi0.5 policy failed the task. — [CrossVLA, arXiv 2605.21854](https://arxiv.org/pdf/2605.21854) (search snippet)
- Cost breakdown, pi0 on a 4090 with 2 views after optimisation: vision 4.06 ms, prefill 12.50 ms, 10-step action expert 11.00 ms. Dropping the unused last-layer LLM outputs saved about 0.7 ms. — [arXiv 2510.26742](https://arxiv.org/html/2510.26742v1)
- Intel splits pi0.5 so that the vision encoder and language model run on the iGPU and the action expert runs on the NPU. The per-layer KV cache is the only handoff between devices. It is stored in shared DDR as zero-copy USM-host tensors. — [Intel OEP pi0.5 article](https://docs.openedgeplatform.intel.com/dev/OEP-articles/publications/optimizing-pi0.5-lva-model.html)

### Inferences
- The prefix KV cache is the natural seam for splitting into two graphs: a "prefix model" that outputs per-layer KV, and a "denoise-step model" or "full N-step loop model". Intel's iGPU+NPU split uses exactly this seam.
- The KV cache must live in a fixed-shape buffer, with fixed token counts and padded language, for graph capture and OpenVINO static-shape compilation.

### Gaps
- I could not fetch the openpi and LeRobot source within budget to quote exact function names and lines (e.g. `embed_prefix`, `denoise_step`, `past_key_values` handling). The description above comes from third-party docs.
- I have no per-stage breakdown on Intel hardware. The Intel article gives only end-to-end numbers and ratios.

## Reducing denoising steps (consistency/shortcut distillation, SnapFlow, 1-step flow, fewer Euler steps)

### Takeaway
One-step self-distillation is now shown on pi0.5 with no loss on LIBERO. Even naive 1-step drops only about 1 point on LIBERO average, but drops more on matched hard initial states and long-horizon tasks. GR00T N1.x already ships with 4 steps.

### Cited Findings
- SnapFlow (arXiv 2604.05656, April 2026) does progressive self-distillation to 1 NFE. It mixes flow-matching samples with consistency samples whose targets are two-step Euler shortcut velocities from the model itself. It adds a zero-initialised target-time embedding, needs no teacher or architecture change, and trains in about 12 h on 1 GPU. — [SnapFlow arXiv](https://arxiv.org/pdf/2604.05656)
- SnapFlow results on pi0.5 LIBERO: 98.75% (1-step) vs 97.75% (10-step teacher) vs 96.75% (naive 1-step). On matched initial states the figures are 81.00% (SnapFlow) vs 77.75% (naive 1-step), with the largest gain on the long-horizon suite. The authors caution that this is their own evaluation. — [SnapFlow arXiv](https://arxiv.org/pdf/2604.05656)
- One secondary summary reports a 9.6x speedup and 274 to 83 ms end to end on pi0.5. This is unverified against the paper text. — [emergentmind summary](https://www.emergentmind.com/papers/2604.05656)
- One-Step Flow Policy (OFP) uses self-distillation, and on RoboTwin 2.0 its 1-step pi0.5 is reported to surpass the original 10-step policy. — [Moonlight review of OFP](https://www.themoonlight.io/review/one-step-flow-policy-self-distillation-for-fast-visuomotor-policies) (secondary)
- GR00T N1.6-3B uses 4 denoising steps. The Thor torch.compile row in the README reads 5 / 39 / 61 / 105 ms, 9.5 Hz, for a single view. Column labels were not visible, but 105 ms is likely the total. — [Isaac-GR00T README](https://github.com/NVIDIA/Isaac-GR00T) (search snippet)
- The survey lists consistency-distillation work for VLAs (CEED-VLA, LightDP) without numbers. — [arXiv 2510.24795](https://arxiv.org/html/2510.24795v3)
- WarmPrior (arXiv 2605.13959) straightens flow-matching policies with temporal priors, which is relevant to fewer steps. I saw only the title. — [WarmPrior](https://arxiv.org/pdf/2605.13959)

### Inferences
- On the 4090 breakdown, going from 10 to 1 step removes about 10 ms of 27 ms (about 37%). On Intel, where the action expert is a larger share at low bandwidth, the relative gain is likely similar or larger.
- Step reduction is hardware-agnostic and the cheapest large win after the KV cache. Simply running 3-5 Euler steps without retraining is a zero-cost experiment, but I found no published pi0 ablation table for 3 or 5 steps in this pass.

### Gaps
- I found no published accuracy-vs-steps curve (10/5/3/1) for pi0 or SmolVLA without distillation.

## Fusing the denoising loop (graph capture, static shapes)

### Takeaway
On NVIDIA, capturing the full pipeline (vision, prefill and all 10 denoise steps) as one CUDA graph with static buffers removed most CPU launch overhead, about 2x. Algebraic folding and fusions gave a further 1.5x or so. The pi0 graph has no dynamic branches, so it is capture-friendly. The Intel equivalent is exporting a static-shape single model (OpenVINO) or using XPU graphs.

### Cited Findings
- CUDA graphs: roughly 2x speedup. Launch/sync overhead fell from 12.92 ms (PyTorch) to 1.72 ms (graph) in a microbenchmark. Capture requires constant kernel sequences and buffer pointers, which holds for pi0 because it has no dynamic branches. — [arXiv 2510.26742](https://arxiv.org/html/2510.26742v1)
- Graph simplification gave 7-8 ms: folding RMSNorm affine into the next linear, folding the action-time encoder, fused QKV, and RoPE fusion. Fused gated linear layers gave 1.7 ms, Triton GEMM tile tuning about 1.5 ms, scalar-op fusion about 4 ms, and dropping last-layer LLM outputs about 0.7 ms. A software-barrier persistent-kernel attempt gave negative end-to-end gains and was not used. The remaining headroom to roofline is about 30%. — [arXiv 2510.26742](https://arxiv.org/html/2510.26742v1) (code: github.com/Dexmal/realtime-vla)
- Intel's pi05-rtc-ov pipeline extends LeRobot with OpenVINO plus RTC. It has separate conversion scripts, `convert_ov.py` and `convert_ov_rtc.py`, the latter producing RTC-enabled IR. Core Ultra Series 3 support arrived in the 2026.1 release. — [Intel pi05 with RTC docs](https://docs.openedgeplatform.intel.com/2026.1/edge-ai-suites/robotics-ai-suite/embodied/sample_pipelines/pi05_with_rtc.html); [release notes](https://docs.openedgeplatform.intel.com/2026.1/edge-ai-suites/robotics-ai-suite/embodied/release-notes.html)
- Export uses `physicalai export --backend openvino`, and the runtime config uses "chunk" synchronous mode. OpenVINO at 40 W on Core Ultra X7 358H is up to 2.6x faster than AGX Orin and 1.1x faster than Thor at iso-TDP, and up to 1.5x faster than AGX Orin at 65 W. Absolute ms were not stated. — [Intel OEP pi0.5 article](https://docs.openedgeplatform.intel.com/dev/OEP-articles/publications/optimizing-pi0.5-lva-model.html)
- Torch-TensorRT wraps each GR00T N1.5 component (VLM backbone, state/action encoders, DiT, action decoder) separately, with results "roughly on par" with ONNX-TensorRT. — [pytorch/TensorRT PR 3887](https://github.com/pytorch/TensorRT/pull/3887/files) (snippet)

### Inferences
- Algebraic folds (RMSNorm-into-linear, fused QKV, time-embedding folding, dropping unused outputs) are hardware-agnostic weight-level transforms. They can be applied before OpenVINO or XPU export. Fused gated MLP and scalar epilogue fusion need kernel or compiler support.
- Unrolling the N-step Euler loop into one static graph needs a fixed step count and fixed noise or chunk shape. This combines well with step distillation: a 1-4 step unrolled graph is small.

### Gaps
- I found no published XPU-graph or OpenVINO single-graph latency numbers for pi0 or pi0.5 in absolute ms.

## Visual token reduction, pruning and merging, image resolution, and camera batching

### Takeaway
Training-free token pruning and caching gives about 1.4-1.9x on autoregressive VLAs (OpenVLA, CogACT), usually with <1 point success loss. On flow VLAs (pi0, pi0.5) the reported gains are smaller, about 1.37x for APS-VLA on pi0.5, because the action expert and fixed overheads remain. The most robust lever is architectural: fewer tokens per image by design (SmolVLA: 64 tokens per frame via pixel shuffle), lower resolution, and fewer views (4090: 20.0 / 27.3 / 36.8 ms for 1/2/3 views).

### Cited Findings
- VLA-Cache reuses static visual tokens' KV across timesteps. On OpenVLA it gives a 27.31% FLOPs cut and 1.63x latency improvement with negligible success loss. — [VLA-Cache arXiv 2502.02175](https://arxiv.org/pdf/2502.02175)
- ActDistill's comparison table lists VLA-Cache at about 1.46x on OpenVLA LIBERO and about 1.47x on CogACT. — [ActDistill arXiv 2511.18082](https://arxiv.org/pdf/2511.18082)
- EfficientVLA is training-free and combines layer pruning, visual token selection and diffusion-head feature caching. On CogACT in SIMPLER it gives 1.93x speedup, FLOPs at 28.9%, and a 0.6% success drop. The ActDistill table reports only about 1.53-1.54x for it, so the sources conflict. — [EfficientVLA, HF papers 2506.10100](https://huggingface.co/papers/2506.10100); contradicted by [ActDistill](https://arxiv.org/pdf/2511.18082)
- SP-VLA (ICLR 2026) does joint model scheduling and token pruning. It reports 1.5x lossless on LIBERO (OpenVLA) and 2.4x on SimplerEnv (CogACT). — [SP-VLA arXiv 2506.12723](https://arxiv.org/pdf/2506.12723)
- VLA-Pruner (arXiv 2511.16449) does temporal-aware dual-level pruning. On pi0 it is reported to keep comparable success at 12.5% token retention. — [VLA-Pruner](https://arxiv.org/html/2511.16449v4) (search snippet)
- APS-VLA (arXiv 2611.13030) does action-guided, future-aware pruning: 1.37x on pi0.5 and 1.52x on XR-1. — [APS-VLA](https://arxiv.org/pdf/2611.13030) (search snippet)
- SmolVLA uses 64 visual tokens per frame via pixel shuffle, e.g. a 512x512 image becomes 64 tokens instead of 1024. — [HF SmolVLA blog](https://huggingface.co/blog/smolvla)
- View count matters: pi0 optimised on a 4090 takes 20.0 / 27.3 / 36.8 ms for 1 / 2 / 3 views. — [arXiv 2510.26742](https://arxiv.org/html/2510.26742v1)

### Inferences
- Batching camera views through the vision encoder in one call (stacking views on the batch dimension) is cheap and graph-friendly. Per-view cost on the 4090 is about 7-9 ms per added view across the whole pipeline.
- Dynamic pruning produces variable token counts, which conflicts with static-shape graphs and OpenVINO static compilation. Fixed-ratio pruning (a fixed top-k) or architectural reduction (pixel shuffle, lower resolution) are the Intel-friendly variants.

### Gaps
- I found no TinyVLA or PD-VLA numbers in this pass. PD-VLA (parallel fixed-point decoding) targets autoregressive action tokens, so it is not directly relevant to flow heads.

## Layer skipping, early exit, and truncated VLMs

### Takeaway
A static truncation (SmolVLA feeds the action expert from layer L/2) is the cleanest and most deployable option: it halves VLM compute and is graph-friendly. Dynamic early exit and routing (DeeR-VLA, MoLe-VLA) save compute on average, but complicate static graphs. I found no flow-VLA latency numbers for them in this pass.

### Cited Findings
- SmolVLA's action expert uses features from around layer N = L/2, which "effectively halves computational costs while maintaining performance". — [SmolVLA blog](https://huggingface.co/blog/smolvla); [arXiv 2506.01844](https://www.emergentmind.com/papers/2506.01844)
- The survey describes DeeR-VLA as action-consistency early exit and MoLe-VLA as dynamic layer routing via STAR, without numbers. — [arXiv 2510.24795](https://arxiv.org/html/2510.24795v3)
- EfficientVLA prunes redundant LLM layers training-free, as part of its 1.93x result on CogACT. — [EfficientVLA](https://huggingface.co/papers/2506.10100)

### Inferences
- For pi0 and pi0.5, a SmolVLA-style truncation would require fine-tuning, because the action expert attends to per-layer KV of all prefix layers. Truncating the backbone also removes the matching expert layers, unless the expert is redesigned.

### Gaps
- I have no quantitative DeeR-VLA or MoLe-VLA speedups from primary sources in this pass.

## Asynchronous and real-time chunking (hiding latency)

### Takeaway
Real-time chunking (RTC) and async inference overlap inference with execution, so a policy with latency longer than one control period can still run smoothly. They are complementary to latency reduction. Intel ships a pi0.5+RTC OpenVINO pipeline.

### Cited Findings
- RTC treats asynchronous chunking as inpainting. Actions that will execute during the inference delay are frozen, and the rest of the next chunk is inpainted. It applies to any diffusion or flow VLA without retraining. — [PI RTC blog](https://www.pi.website/research/real_time_chunking); [arXiv 2506.07339](https://arxiv.org/pdf/2506.07339)
- Training-time RTC (arXiv 2512.05964) conditions on action prefixes during training to simulate delay. It removes inference-time inpainting overhead, which relies on Jacobian linearisation and degrades with long prefixes, and it improves robustness to variable latency. — [arXiv 2512.05964](https://arxiv.org/pdf/2512.05964)
- SmolVLA async inference gives about 30% faster task completion (e.g. 9.7 s vs 13.75 s) and 2x pick-place cycles in a fixed window, with similar success rates. — [HF SmolVLA blog](https://huggingface.co/blog/smolvla)
- Intel's pi05-rtc-ov pipeline extends LeRobot with OpenVINO plus RTC on Intel platforms. — [Intel pi05 with RTC](https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/embodied/sample_pipelines/pi05_with_rtc.html)

### Inferences
- Inference-time RTC guidance adds per-step cost (a vector-Jacobian product through the action expert at each denoise step). Training-time RTC avoids this, which favours it for compute-limited Intel iGPUs and NPUs.

### Gaps
- The PI RTC page returned HTTP 429, so I could not extract its quantitative results (injected latency, throughput).
- Bidirectional decoding was not researched in this pass.

## Quantisation results specifically on VLAs

### Takeaway
pi0.5 is robust to W4A8 and even W4A4 with VLA-aware PTQ (about 97-98% LIBERO vs 97.1% FP16). pi0 and OpenVLA-OFT are much more fragile under naive RTN. Generic LLM PTQ (GPTQ, AWQ, OmniQuant) applied to the full stack, action expert included, can collapse pi0.5. Keep sensitive parts (action expert, projections) at higher precision or use VLA-specific calibration.

### Cited Findings
- QuantVLA (CVPR 2026) on pi0.5 W4A8 reaches 97.6% average LIBERO vs 97.1% FP16, and memory drops from 4.27 GB to 1.28 GB. DuQuant at W4A8 gets 76.3%. — [QuantVLA arXiv 2602.20309](https://arxiv.org/pdf/2602.20309)
- VLAQuantBench (RTN, closed-loop LIBERO, 200 episodes per setting), pi0.5 W4A8: 99.0 / 100.0 / 96.5 / 96.5 vs baseline 99.0 / 99.0 / 94.0 / 95.0 on Spatial / Object / Goal / Long. pi0 W4A8: Spatial 64.5 vs 78.0 and Long 41.0 vs 48.5. OpenVLA-OFT W4A8 collapses on several suites. — [VLAQuantBench arXiv 2609.25376](https://arxiv.org/html/2609.25376)
- HoloQ-VLA reaches W4A4 at 98.0% on pi0.5. Under full-stack W4A16, GPTQ, AWQ and OmniQuant fall to 16.0%, 11.5% and 10.3% on pi0.5. — [HoloQ-VLA arXiv 2605.28803](https://arxiv.org/pdf/2605.28803)
- FoldQuantVLA's selective-INT8 variant gets 97.8% on pi0.5. Its re-run of HoloQ's long suite gave 94.0% vs the reported 96.0%, so there is reproducibility variance. — [FoldQuantVLA arXiv 2609.24433](https://arxiv.org/html/2609.24433)
- GR00T N1.7 on Jetson Thor with TensorRT and mixed NVFP4 goes from 125 ms to 39.9 ms end to end (3.1x), about 25 Hz. — [Jetson AI Lab GR00T N1.7 on Thor](https://www.jetson-ai-lab.com/tutorials/groot_n17_on_thor/)
- The survey says 4-bit PTQ on OpenVLA halves GPU memory with negligible performance loss. BitVLA (ternary) and QVLA (channel-wise bit allocation) are listed without numbers. — [arXiv 2510.24795](https://arxiv.org/html/2510.24795v3)

### Inferences
- For Intel, weight-only INT8 or INT4 on the VLM prefill (bandwidth-bound at batch 1) is the safest first step. Activation quantisation and action-expert quantisation need per-model validation, and pi0 needs more care than pi0.5.

### Gaps
- No Intel-hardware VLA quantisation latency results were found. The Intel pi0.5 article uses BF16/FP16 only.

## Published work running VLAs on Intel hardware or edge accelerators (reference latencies)

### Takeaway
The main Intel data point is pi0.5 on Panther Lake (Core Ultra X7 358H): 294 ms with stock PyTorch at 40 W. OpenVINO with an iGPU+NPU split beats AGX Orin by up to 2.6x and Thor by 1.1x at iso-TDP, but no absolute ms were published. On NVIDIA edge, an optimised GR00T reaches about 25 Hz on Thor with NVFP4. A forum post claims pi0.5 or GR00T at 23 Hz on Thor.

### Cited Findings
- Core Ultra X7 358H: 16 cores, 15-65 W, 180 platform TOPS and 122 GPU TOPS (INT8), about 153 GB/s memory bandwidth. Stock PyTorch pi0.5 takes 294 ms (40 W) vs Thor 287 ms. AGX Orin is about 3.5x slower at 60 W and about 5.2x slower at 40 W (about 1.0 s and 1.5 s, derived from the ratios). OpenVINO at 40 W is up to 2.6x better than AGX Orin and 1.1x better than Thor (which used TensorRT). Setup: 3 cameras at 224x224, 10 steps, FP16/BF16. — [Intel OEP pi0.5 article](https://docs.openedgeplatform.intel.com/dev/OEP-articles/publications/optimizing-pi0.5-lva-model.html)
- GR00T N1.6-3B on Thor with torch.compile, 4 steps: about 105 ms, 9.5 Hz (column labels inferred). — [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) (snippet)
- GR00T N1.7 on Thor with TensorRT and NVFP4: 39.9 ms, 25 Hz. — [Jetson AI Lab](https://www.jetson-ai-lab.com/tutorials/groot_n17_on_thor/)
- An NVIDIA forum post titled "Real-Time Inference on Thor & RTX: Pi0.5/GR00T N1.6/1.7 — Thor 23 Hz, RTX 5090 50-80 Hz". I saw only the title, not the content. — [NVIDIA forum](https://forums.developer.nvidia.com/t/real-time-inference-on-thor-rtx-pi0-5-gr00t-n1-6-1-7-thor-23-hz-rtx-5090-50-80hz/368788)
- pi0 on an RTX 4090 reaches 20-37 ms (1-3 views) with CUDA graphs plus fusions. — [arXiv 2510.26742](https://arxiv.org/html/2510.26742v1)

### Inferences
- If OpenVINO at 40 W is about 1.1x Thor and Thor's stock PyTorch is 287 ms, the Intel OpenVINO number is likely well below the stock 294 ms. The exact value cannot be derived, because Thor's TensorRT figure is not given.
- Reaching 30 Hz (33 ms) on a Panther Lake iGPU for pi0.5 with 3 views likely requires stacking 1-step distillation, INT4/INT8 weights, fewer or smaller views and RTC. The stack follows the inference above; reaching 30 Hz is an estimate, not a reported result.

### Gaps
- No absolute OpenVINO latency for pi0.5 on Intel was found. No Arc discrete GPU VLA benchmarks were found. No Intel numbers for SmolVLA, GR00T or X-VLA were found.
- I found no X-VLA-specific inference-optimisation results in this pass.
