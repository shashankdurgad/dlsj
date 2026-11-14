# Cutting the vision + VLM-prefix cost of π0.5 for on-device (OpenVINO, Arc B390 iGPU) inference — state as of 14 Nov 2026

Scope notes for the report writer:
- Our baseline: 169 ms per chunk at 10 steps; prefix 82 ms, vision 22 ms, expert 53 ms. Prefix length is 3 x 256 + 64 = 832 tokens.
- Everything under "Inferences" is my reasoning, not a sourced result. All millisecond estimates for our machine are linear-scaling estimates, not measurements.
- Out of scope and not covered: denoising-step reduction, quantisation, OS/runtime tuning. Where a source bundles those with prefix work (Nota "Drift", Guild SnapFlow/FP8) I report only enough to separate out the prefix part.
- Several 2026 arXiv items were read through an HTML-to-text summariser; numbers quoted from tables should be re-checked against the PDF before they go in a slide. I flag the ones I could not fully verify.

## 1. Fixed-ratio visual token pruning / merging / pooling on π0/π0.5 and PaliGemma-style VLAs

### Takeaway
The best-supported option is a fixed-K-per-camera selector placed after the SigLIP projector and before Gemma: it keeps static shapes, and three independent 2026 sources report it on π0.5 (Nota AI K=128: about 40% fewer prefix tokens, 175 -> 117 ms on Panther Lake iGPU at about 1-2 points LIBERO-10 cost; VLA-ACL K=64: 2.2x prefill speed-up at -0.8 points with the VLA frozen and public code). Training-free in-LLM pruners (FastV, VLA-Pruner, VLA-Cache, SAFE-Pruner) give smaller savings on π0.5 and lose more accuracy, and the temporal ones carry closed-loop risk.

### Cited Findings

Nota AI π0.5 OpenVINO card (the "about 105 ms" number we had heard)
- The card ships three π0.5 LIBERO variants as static OpenVINO IR for Core Ultra X7 358H / Arc B390, OpenVINO 2026.4.0, FP16 weights, batch 1, three camera slots at 224x224 (third slot empty and masked), 200-token prompt — [Nota AI HF card](https://huggingface.co/geonmin-kim/nota-pi05-libero-vtp-drift-openvino)
- Visual token pruning there is "a small learned sampler after the SigLIP projector, one per camera", selecting 128 of 256 tokens per camera by bilinear grid sampling; the prefix shrinks from 968 to 584 tokens and the expert's KV cache shrinks with it — [Nota AI HF card](https://huggingface.co/geonmin-kim/nota-pi05-libero-vtp-drift-openvino)
- iGPU-only latency per chunk (mean over one real-time LIBERO-10 episode, includes pre/post-processing): Base 10-step 258 ms; Drift 1-step 175 ms; Drift + VTP K=128 117 ms. Median of 10 runs on the bundled example observation: 105 ms for Drift + VTP, 250 ms for Base. So the "105 ms" figure is real but is the single-observation median; the episode mean is 117 ms — [Nota AI HF card](https://huggingface.co/geonmin-kim/nota-pi05-libero-vtp-drift-openvino)
- LIBERO-10 success on the iGPU (200 episodes, seed 7): Base 95.5%, Drift 94.0%, Drift + VTP 92.0% (95% CI 87.4-95.0). PyTorch reference from Nota's training report: 95.8 / 94.8 / 93.6 — [Nota AI HF card](https://huggingface.co/geonmin-kim/nota-pi05-libero-vtp-drift-openvino)
- Both compressed checkpoints were fine-tuned by Nota from `lerobot/pi05_libero_finetuned_v044` for 10k steps on LIBERO; the card does not say which components were frozen, the batch size, or the learning rate. Checkpoints: `geonmin-kim/pi05-Mdrift_vtp_K128_cft-Dlibero-step10000` — [Nota AI HF card](https://huggingface.co/geonmin-kim/nota-pi05-libero-vtp-drift-openvino)
- Caveats stated on the card: LIBERO simulation only, no real robot; single-episode, single-device latency; repo files are gated behind manual access approval; about 70% of energy per inference goes to the prefix — [Nota AI HF card](https://huggingface.co/geonmin-kim/nota-pi05-libero-vtp-drift-openvino)
- Correctness note relevant to us: with default settings the iGPU "gave wrong prefix outputs on OpenVINO 2026.4" and Nota sets `GPU_ENABLE_SDPA_OPTIMIZATION=NO` for the prefix — [Nota AI HF card](https://huggingface.co/geonmin-kim/nota-pi05-libero-vtp-drift-openvino)

Learned fixed-K selectors with the VLA frozen
- VLA-ACL (arXiv 2610.08133): a 5-layer bidirectional-transformer patch scorer is trained while the VLA stays frozen; it keeps a fixed K tokens per camera view (K = 32 or 64 of 256), identical at every control step, and pruning happens before the LLM. Training is 5,000 steps, AdamW, lr 1e-4, batch 8, described as "a few hours of training on a single GPU"; real-world tasks used 50-100 demos each. Code: github.com/du-owen/VLA-ACL — [VLA-ACL](https://arxiv.org/html/2610.08133v2)
- VLA-ACL on π0.5 LIBERO (K=64, i.e. 25% of visual tokens): 96.1% average vs 96.9% unpruned; TFLOPs 2.115 -> 0.701; prefill latency 25.9 -> 11.6 ms (2.2x) on one A100 — [VLA-ACL](https://arxiv.org/html/2610.08133v2)
- Same table, other methods on π0.5 LIBERO (numbers copied by VLA-ACL from the SAFE-Pruner paper, not re-run): VLA-Cache 93.9% at 1.632 TFLOPs; VLA-Pruner 94.2% at 1.583; SAFE-Pruner 95.5% at 1.482 — [VLA-ACL](https://arxiv.org/html/2610.08133v2)
- APS-VLA (arXiv 2611.13030, posted 9 Nov 2026): learned selector with the VLA frozen, fixed per-view budget, input-level pruning, no dependence on previous-step state at deployment, 5,000 training updates. On π0.5 / RoboCasa365 (A800): unpruned 70.3 ms; FastV 56.2 ms (1.25x); SAFE-Pruner 55.4 ms (1.27x); VLA-ACL 52.5 ms (1.34x); APS-VLA 51.5 ms (1.37x), with seen-atomic success 39.60 / 36.47 / 37.50 / 38.97 / 39.15. Real robot (π0.5, three cameras, 5 tasks x 40 trials): 60.0% -> 61.0%, 137.25 -> 93.25 ms (1.47x). No code yet ("released upon acceptance") — [APS-VLA](https://arxiv.org/html/2611.13030)

Supervised multi-view pruning
- BFA++ (arXiv 2602.20566) prunes per view then globally with fixed ratios [0.3, 0.2, 0.2, 0.5] on π0 and RDT-1B; the authors tried an adaptive-ratio variant and dropped it for inconsistent speed and lower success. It needs full-parameter fine-tuning (5,000 steps, batch 256 for π0) plus offline region/view annotation. π0 on RoboTwin: 6.5 -> 10.3 Hz, success 0.496 -> 0.583; real world 5.8 -> 9.4 Hz, 0.27 -> 0.48. The abstract's "1.8x" does not match the roughly 1.6x in its own tables. No code link — [BFA++](https://arxiv.org/html/2602.20566)
- BFA++'s stated motivation: the head view suffices for approach and post-manipulation phases, i.e. view importance changes by phase — [BFA++](https://arxiv.org/html/2602.20566)

Training-free pruners
- VLA-Pruner (arXiv 2511.16449): plug-and-play, combines semantic prefill attention with temporally smoothed action attention, "up to 1.99x speedup with comparable manipulation quality"; code at github.com/MINT-SJTU/VLA-Pruner — [VLA-Pruner abstract](https://arxiv.org/abs/2511.16449)
- SAFE-Pruner (arXiv 2605.29662): training-free, forecasts token saliency from earlier keyframes with a consistency-triggered refresh; evaluated on OpenVLA and π0.5; claims up to 1.89x with under 1.7% success loss. A later paper says no official implementation was available — [SAFE-Pruner](https://arxiv.org/pdf/2605.29662)
- "The Gate, Not the Cache" (arXiv 2608.00391, OpenVLA-OFT, not π0.5): when the skip/prune gate is computed from the model's own accelerated forwards, LIBERO-Object success at 0.9 skip ratio falls to 0.68 (caching) and 0.31 (deletion) vs 1.00 dense; with a gate from a dense pass both reach about 0.98. Failures were "silent" (action-level detectors AUROC about 0.45-0.65). Their fix, a dense refresh pass in actuation slack, costs about 1.6x dense FLOPs and about 18% more energy — [The Gate, Not the Cache](https://www.alphaxiv.org/abs/2608.00391)
- FastVLA (arXiv 2610.22146): training-free but the kept-token count varies per step (dynamic rate), and it has no π0.5 results — [FastVLA](https://arxiv.org/html/2610.22146)
- LightVLA (arXiv 2509.12594): differentiable (Gumbel-softmax) token selection, needs training, LIBERO only; FLOPs -59.1%, latency -38.2%, +2.6% success. I found no π0/π0.5 result — [LightVLA](https://arxiv.org/abs/2509.12594v2)
- EcoVLA (arXiv 2602.00780) is channel pruning, not token pruning, and its sparsity pattern is updated online (dynamic). π0.5 LIBERO: 96.9% / 81.94 ms -> 96.7% / 62.66 ms (1.31x) at 25% sparsity; 95.0% / 55.98 ms (1.46x) at 37.5%; RTX 3090 — [EcoVLA](https://arxiv.org/html/2602.00780)
- OpenVINO GenAI ships a training-free visual token pruner (CDPruner) with an integer `pruning_ratio` for its VLM pipeline; nothing there is π0.5-specific — [OpenVINO GenAI docs](https://openvinotoolkit.github.io/openvino.genai/docs/concepts/optimization-techniques/visual-token-pruning)

Counter-evidence on how much token reduction buys
- Shallow-π reports that on an H100 "token reduction provides minimal benefit", listing LightVLA at 22.0 ms and CogVLA at 31.0 ms against π0 at 22.6 ms — [Shallow-π](https://arxiv.org/html/2601.20262v1)
- OneWM-VLA: static average pooling of PaliGemma's 256 visual tokens (down to one token per view, in a world-model VLA, not a π0.5 fine-tune) was 20.5 points worse than learned pooling — [One Token Per Frame](https://arxiv.org/html/2605.07931v1)

### Inferences
- Static-shape classification. Friendly: any fixed-K-per-camera selection before the LLM (Nota grid sampler, VLA-ACL, APS-VLA, BFA++ fixed ratios, plain fixed pooling) — the graph is a TopK/Gather or GridSample with constant output length, so one static IR works. Workable but awkward: fixed-ratio in-LLM pruning at a fixed layer (FastV style) — shapes are static but differ before and after the prune layer, so the expert's per-layer KV inputs have two lengths, and the method needs attention maps exposed. Unfriendly: dynamic-rate methods (FastVLA), online-sparsity methods (EcoVLA), and anything that carries state between calls (VLA-Cache, VLA-Pruner's temporal term, SAFE-Pruner keyframes) — these need extra state inputs and are the ones with documented silent closed-loop failures.
- The H100 "minimal benefit" finding should not transfer to our machine. On an H100 a 1k-token prefix is latency-bound by overheads; on the Arc iGPU our prefix is compute-bound (FullyConnected is 69% of GPU time). Nota's own numbers on the same iGPU family back this: a 39.7% token cut (968 -> 584) removed 58 ms of 175 ms with the vision encoder and the single expert step unchanged.
- Linear-scaling estimates for our 82 ms prefix at 832 tokens (estimates only): K=128 on all three cameras -> 448 tokens, about 44 ms (saves about 38 ms); K=64 on all three -> 256 tokens, about 25 ms (saves about 57 ms). The expert also attends over a shorter prefix each step, so its per-step cost should fall a little too. The vision encoder's 22 ms is untouched because selection sits after SigLIP.
- For our constraints (expert-only fine-tune, frozen backbone, about 500 demos, one day) VLA-ACL is the closest fit on paper: VLA frozen, fixed K, public code, a few GPU-hours, π0.5 results. Open risk: it was shown on LeRobot/openpi LIBERO checkpoints, not on an expert-only fine-tune, and its real-robot numbers are for OpenVLA-OFT, not π0.5.
- Nota's recipe is proven on our exact runtime, but their 10k-step fine-tune probably touched more than a small sampler (unspecified), and VTP was trained jointly with one-step "Drift", so the 1.5-2 point VTP-only cost (94.0 -> 92.0 on iGPU; 94.8 -> 93.6 in PyTorch) is the cleanest prefix-only number available.

### Gaps
- No source reports plain 2x2 average pooling (256 -> 64) or ToMe-style merging on π0/π0.5. I found no ToMe, SP-VLA, EfficientVLA or LightVLA result on a flow-matching π model; EfficientVLA and SP-VLA numbers in the literature are for OpenVLA/CogACT-class models.
- VLA-Pruner's π0 table (retention ratios and success) could not be read; only the abstract's "up to 1.99x" is confirmed. The VLA-Cache/VLA-Pruner/SAFE-Pruner π0.5 numbers above are second-hand (via VLA-ACL citing SAFE-Pruner).
- Nota did not publish which parameters were trained for VTP, nor a real-robot result, nor any INT8 result.
- No pruning result exists for a π0.5 whose backbone was frozen and only the expert fine-tuned.

## 2. Lower image resolution, fewer cameras, per-camera token budgets

### Takeaway
There is no published accuracy evidence for running π0.5 below 224x224, and the only camera-removal evidence shows that masking a camera without retraining is destructive for π0.5 (wrist masked: 4.6%; agent view masked: 55.4% on LIBERO-Spatial). Removing a view is a large, static-shape-friendly saving (256 tokens plus one SigLIP pass), but it must be done by retraining the expert on the reduced input, not by masking at inference.

### Cited Findings
- Camera ablation on `lerobot/pi05_libero_finetuned_v044`, LIBERO-Spatial, no retraining: normal 98.4% (492/500); wrist masked 4.6% (23/500); agent view masked 55.4% (277/500); wrist frame frozen to the first frame 0/500; all cameras black 0% — [libero-camera-ablation repo](https://github.com/chaoshengsc/libero-camera-ablation)
- The same repo's π0 LeRobot checkpoint was far less camera-sensitive (72.0% normal, 71.2% wrist masked, 74.2% agent masked, 53-56% all black), but its baseline varies by up to 18 points between repeats, so treat it as weak — [libero-camera-ablation repo](https://github.com/chaoshengsc/libero-camera-ablation)
- Latency vs view count for a π0-class model on an RTX 4090 (optimised stack): 1 view 20.0 ms, 2 views 27.3 ms, 3 views 36.8 ms, i.e. about 8-9 ms per extra view out of 37 ms — [Running VLAs at Real-time Speed](https://arxiv.org/html/2510.26742)
- NVIDIA's TensorRT Edge-LLM π0.5 export drops masked camera slots entirely ("a dropped slot removes 256 prefix tokens"); its profiles use 2 views for LIBERO and DROID and 3 for ALOHA, and camera count is fixed per export — [TensorRT Edge-LLM π0.5 design](https://nvidia.github.io/TensorRT-Edge-LLM/0.11.0/developer_guide/models/pi05.html)
- Nota's OpenVINO export, by contrast, keeps the empty third camera in the prefix (968 = 3 x 256 + 200 tokens with the third slot "empty and masked") — [Nota AI HF card](https://huggingface.co/geonmin-kim/nota-pi05-libero-vtp-drift-openvino)
- Intel's paper notes that token count grows with resolution and scales linearly with the number of camera streams (search-snippet level; the article text I could read gives no per-camera numbers) — [Intel OEP article](https://docs.openedgeplatform.intel.com/dev/OEP-articles/publications/optimizing-pi0.5-lva-model.html)
- Per-camera budgets: BFA++ uses different prune ratios per view (0.3 / 0.2 / 0.2) then a global 0.5; APS-VLA and VLA-ACL use a fixed budget per view — [BFA++](https://arxiv.org/html/2602.20566); [APS-VLA](https://arxiv.org/html/2611.13030); [VLA-ACL](https://arxiv.org/html/2610.08133v2)
- SmolVLA caps visual tokens at 64 per frame via pixel shuffle with 512x512 inputs and no tiling, with a frozen VLM and only the expert trained; it reports no ablation over token count or view count — [SmolVLA](https://arxiv.org/html/2506.01844)
- Latency sensitivity for context: injecting delay into base π0.5 lowered success from 90% to 85% (66 -> 266 ms) on one task and 44% to 35% on another — [Guild Robotics](https://www.guildrobotics.com/blog/pi05-on-the-edge)

### Inferences
- Dropping the inspect view from the policy input (it mainly serves the separate classifier) takes the prefix from 832 to 576 tokens (-31%) and removes one of three SigLIP passes. Linear estimate for our machine: prefix 82 -> about 57 ms, vision 22 -> about 15 ms, roughly 30 ms per chunk in total, before any pruning. Stacked with K=128 on the two remaining cameras the prefix is 320 tokens (about 32 ms by the same estimate).
- This fits our training setup better than any other item: the backbone stays frozen and untouched, and the expert is re-fine-tuned on two views with the same recipe already used. The camera-ablation result says the expert, as trained, leans heavily on every view it saw, so a retrain is mandatory and a go/no-go evaluation is needed; if the task actually needs the inspect view at some phase, this fails.
- A softer variant is a fixed small budget for the inspect view (for example 64 or 32 tokens through a fixed pool or learned selector) with 256 or 128 for wrist and top. It keeps static shapes and some inspect information; it has no direct π0.5 evidence.
- Lowering input resolution (for example 160x160 -> about 121 tokens, or 112x112 -> 64 tokens) cuts both SigLIP and prefix cost, but SigLIP-So400m/14 at 224 has position embeddings for a 16x16 grid and the frozen backbone never saw other grids. I would rank it below post-projector token selection, which keeps the encoder in distribution.

### Gaps
- No published π0/π0.5 result for sub-224 resolution, for retraining with fewer cameras, or for mixed per-camera resolutions. No π0.5 latency-vs-camera-count table on Intel hardware.
- How openpi/LeRobot mask a missing camera for π0.5 (whether masked tokens are still computed) should be checked in our own export; I did not verify it from source in this pass.

## 3. Using fewer VLM layers (truncation, layer drop, early exit) and whether a frozen-backbone fine-tune can adopt it

### Takeaway
Gemma's language layers are the most redundant part of π0.5: dropping 9 of 18 blocks (about 58% of FLOPs left) or even 16 of 18 recovers baseline LIBERO success, while dropping SigLIP layers hurts badly. But every π0.5 result uses full-model recovery fine-tuning or distillation (about 30k steps); nobody has shown it with a frozen backbone and expert-only retraining, so this is a multi-day item for us, not a one-day one.

### Cited Findings
- Drop-Then-Recovery / VLADrop (arXiv 2606.27755), π0.5 on LIBERO, average success: baseline 91.7; language drop-half 93.3 (60.6% size, 57.9% FLOPs); language keep-2 91.0 (25.1% FLOPs); vision drop-half 80.8; vision keep-2 62.4; action drop-half 93.2; action keep-2 26.2 — [DTR paper](https://arxiv.org/html/2606.27755v1)
- With their GateProbe block ranking: Drop-9 94.0, Drop-12 92.1, Drop-16 92.2, Drop-17 88.7; with compute-matched longer recovery Drop-12 reaches 93.7 and Drop-16 92.6 — [DTR paper](https://arxiv.org/html/2606.27755v1)
- Block choice matters at Drop-9: Taylor/IGIA 94.2, CosSim 90.2, Magnitude 88.0; the Taylor checkpoint keeps blocks [0,1,3,4,5,6,7,8,9] — [VLADrop HF card](https://huggingface.co/s1ghhh/VLADrop_pi05_LIBERO_Drop9Block_Taylor-IGIA)
- Recovery is full fine-tuning (not LoRA; nothing stated as frozen): AdamW, batch 32, 30k steps, lr 5e-5, mixed LIBERO data. Zero-shot (no recovery) results are not reported for π0.5. No π0.5 latency numbers are given, only FLOPs. Code and checkpoints: github.com/s1ghhh/VLADrop — [DTR paper](https://arxiv.org/html/2606.27755v1)
- How DTR handles the layer-wise expert coupling: for a dropped language block the Q, O and MLP weights are removed but its K and V projections and input norm are kept, and the hidden state passes through as identity, so the expert still gets a K/V for every layer — [DTR paper](https://arxiv.org/html/2606.27755v1)
- DTR real robot (π0.5, xArm 850, Jetson Thor, about 600 grasps / 110k frames): Env 1 full 63.3%, Drop-9 65.0%, Drop-16 55.0%; Env 2 full 75.0%, Drop-9 71.7%, Drop-16 66.7%; Drop-16 degrades more out of distribution (green light 35% vs 50%; container removed 40% vs 60%) — [DTR paper](https://arxiv.org/html/2606.27755v1)
- Shallow-π (arXiv 2601.20262) distils π0/π0.5 from 18 to 6 layers in both backbone and expert (uniform layer sub-sampling init; task loss + teacher-velocity loss + mid-layer attention distillation). LIBERO: π0.5 96 -> π0.5-L6 95. H100 latency/FLOPs: π0.5 25.5 ms / 3.39 T; L9 14.8 ms / 1.82 T; L6 11.3 ms / 1.30 T. Jetson Thor end-to-end: π0.5 130 ms -> 78 ms. Training: 30k steps batch 64 for LIBERO; 100k steps batch 128 for real robots, on 92-2,600 episodes per task. No code release stated — [Shallow-π](https://arxiv.org/html/2601.20262v1)
- Shallow-π's own training-free skipping experiment: success collapses once more than three layers are removed in sensitivity order — [Shallow-π](https://arxiv.org/html/2601.20262v1)
- Guild Robotics (2 Jul 2026) reproduced Shallow-π on π0.5 for Jetson Thor at 9 layers (6 layers "showed slight quality regression"): p95 260 -> 127 ms from layer reduction alone; a drop on one task was recovered with a short behaviour-cloning step on under 100 teacher rollouts. No code release — [Guild Robotics](https://www.guildrobotics.com/blog/pi05-on-the-edge)
- CLP (arXiv 2606.20246): CKA-guided layer selection, then task fine-tuning of the pruned model; π0 LIBERO 94.6 -> 93.9 at 1.39x; flat up to about 50% pruning; 10 real-robot tasks 73.5% -> 75.9% average with individual tasks moving both ways. π0, GR00T-N1.5 and SmolVLA, not π0.5. Code: github.com/jibby2803/CLP_VLA — [CLP project page](https://clpvla.github.io/)
- SmolVLA precedent for frozen VLM + expert-only training on truncated layers: LIBERO average by number of VLM layers used N=8 75.0, N=16 78.5, N=24 79.5, N=32 80.3; skipping every second layer 75.5; smaller VLM 75.8. The VLM is frozen and the expert is trained from scratch — [SmolVLA](https://arxiv.org/html/2506.01844)
- A training-free, exact saving: only the KV cache of the last LLM layer is consumed by the expert, so the last layer's output computation can be skipped (about 0.7 ms of about 27 ms on an RTX 4090 for π0) — [Running VLAs at Real-time Speed](https://arxiv.org/html/2510.26742)

### Inferences
- Because expert layer i reads the K/V of backbone layer i, "use the first half of the VLM" is not a pure inference-time switch for π0.5: either the expert is shortened too (Shallow-π), or dropped blocks keep their K/V projections as in DTR. With the DTR layout a dropped block still costs its norm and two small K/V projections, so the saving per dropped block is close to, but not exactly, 1/18 of prefix compute.
- Linear estimate for us if Drop-9 held: prefix 82 -> about 45-50 ms; Drop-12: about 35-40 ms. This stacks multiplicatively with token reduction.
- Can a frozen-backbone, expert-only retrain adopt it? Untested on π0.5. SmolVLA shows the pattern can work in principle (frozen VLM, expert reading early layers), and DTR notes the retained K/V get gradients from the expert, which suggests an intermediate recipe: delete blocks, train only the expert plus the retained K/V projections. That is my extrapolation; Shallow-π's finding that un-recovered skipping collapses past three layers says recovery training is doing real work. With about 500 demos, the Drop-16 OOD degradation is the warning sign; Drop-9 looks like the safe target.
- One-day feasibility: DTR's 30k steps at batch 32 of full π0.5 fine-tuning is realistic on rented multi-GPU within about a day of wall-clock, but it replaces our frozen backbone with a task-tuned one and needs a fresh export and closed-loop validation. I would schedule it as the second-round item after camera/token reduction.
- The last-layer skip is worth checking in our IR today: if the exported prefix graph has only K/V outputs, OpenVINO should already have dead-code-eliminated the last block's attention output and MLP; if the IR still exposes final hidden states, about 1/18 of the prefix (roughly 4 ms by linear estimate) is being wasted.
- Do not truncate SigLIP: vision drop-half costs about 11 points even after recovery.

### Gaps
- No π0.5 result for early exit / dynamic depth (DeeR-VLA, MoLe-VLA are OpenVLA/Flamingo-class and dynamic-depth, hence also not static-graph friendly); Shallow-π cites them but gives no numbers.
- No π0.5 result for layer drop with a frozen backbone or with LoRA-only recovery; no zero-shot drop numbers for π0.5; no wall-clock latency for DTR π0.5 configs.
- DTR HF cards for the action keep-2 config disagree on per-suite labels (Spatial and Long appear swapped between two cards); the averages agree.

## 4. Reusing / caching the prefix or vision features across chunks or cameras

### Takeaway
The only π0.5-specific evidence on cross-step prefix reuse is negative, and independent work on OpenVLA-OFT shows cache/prune gates derived from the model's own accelerated passes can fail silently in closed loop. I found no published, validated scheme for static-camera or keyframe feature reuse on π0.5; treat it as unproven and low priority.

### Cited Findings
- CrossVLA (arXiv 2605.21854, single author) tested two caches on π0.5 LIBERO-Spatial. Chunk-level reuse (cosine >= 0.95): success 50/50 -> 40/50 and wall time 1258 s -> 1796 s despite 82% reuse. Token-level VLA-Cache-style prefix K/V reuse across env steps with the expert re-run: 0/1 success at threshold 0.92 (86% hit rate) and 0/2 at 0.98 (64% hit rate); "stale prefix K/V breaks suffix attention sufficiently that the policy plateaus and never solves the task" — [CrossVLA PDF](https://arxiv.org/pdf/2605.21854)
- Its latency anatomy (H20 GPU, single-image observation, PyTorch): prefix about 60 ms (21.4%), 10-step denoise loop about 220 ms (78.6%), hence its "21% ceiling" for prefix caching. Code: github.com/lz-googlefycy/vla-lab — [CrossVLA PDF](https://arxiv.org/pdf/2605.21854)
- VLA-Cache (NeurIPS 2025) reuses KV for static visual tokens between adjacent frames, training-free; up to 1.7x CUDA latency speed-up on OpenVLA, OpenVLA-OFT and CogACT; not evaluated on π0/π0.5 by its authors — [VLA-Cache](https://arxiv.org/abs/2502.02175)
- Second-hand π0.5 number for VLA-Cache: 93.9% LIBERO average vs 96.9% unpruned at 1.632 vs 2.115 TFLOPs — [VLA-ACL](https://arxiv.org/html/2610.08133v2)
- "The Gate, Not the Cache": self-harvested gates drop success to 0.68 (caching) at a 0.9 skip ratio vs about 0.98 with a clean gate; failures are not caught by action-level detectors; the remedy is an unconditional dense refresh during actuation slack at about 1.6x dense FLOPs — [The Gate, Not the Cache](https://www.alphaxiv.org/abs/2608.00391)
- π0.5 fails completely when a camera's frame is frozen to a stale image (wrist frozen: 0/500 on LIBERO-Spatial) — [libero-camera-ablation repo](https://github.com/chaoshengsc/libero-camera-ablation)
- In the standard π0.5 split, visual and prefix engines already run once per request and the prefix K/V is reused across all denoising steps — [TensorRT Edge-LLM π0.5 design](https://nvidia.github.io/TensorRT-Edge-LLM/0.11.0/developer_guide/models/pi05.html)

### Inferences
- CrossVLA's "21% ceiling" does not apply to us: our prefix plus vision is 66% of latency at 10 steps and larger at 5 steps, so caching would matter if it were safe. Its correctness finding is the relevant part, and it is thin (one to two rollouts per setting, naive whole-prefix reuse, one author). It shows naive reuse is unsafe, not that all reuse is impossible.
- Why whole-prefix reuse is structurally wrong here: Gemma's prefix is fully self-attended, so every token's K/V at every layer depends on all cameras; a changed wrist image invalidates the cached K/V of the "static" top camera too. Only SigLIP outputs are per-camera and can be reused exactly, and that saves at most about 7 ms per camera of our 22 ms vision time, while the frozen-frame result above shows what happens if a stale view is wrong.
- Per-token partial KV reuse (true VLA-Cache) needs per-step dynamic token sets and cross-call state, which conflicts with one static IR; the dense-refresh fix costs more total compute, which is the wrong trade on a thermally limited laptop iGPU.
- Lower-risk alternative that gets a similar effect: give the mostly-static camera a small fixed token budget (section 2) rather than caching it.

### Gaps
- No published π0/π0.5 result for SigLIP-feature reuse on static cameras, keyframe prefix reuse with asynchronous refresh, or VLA-Cache run by its own authors on a flow-matching VLA.
- CrossVLA's negative result has very small sample sizes; I found no replication.

## 5. Smaller drop-in backbones or distillation of π0.5 to a smaller VLM

### Takeaway
There is no open drop-in smaller backbone for π0.5 as of November 2026; the published routes are depth distillation of the same Gemma backbone (Shallow-π, reproduced by Guild) and layer dropping (DTR), both requiring full retraining. Switching to a different small VLA (SmolVLA class) is a model change, not a prefix optimisation, and its LIBERO numbers are well below π0.5.

### Cited Findings
- Shallow-π: 18 -> 6 layers for backbone and expert together, over 2x faster with under 1 point average drop on LIBERO; authors argue shrinking only the backbone leaves the expert as the bottleneck because the expert mirrors backbone depth — [Shallow-π](https://arxiv.org/html/2601.20262v1)
- Shallow-π real-robot results favour the student in their setups (for example RB-Y1 Recycle 12/20 teacher vs 17/20 student on Jetson Thor), which the reduced latency helps explain — [Shallow-π](https://arxiv.org/html/2601.20262v1)
- Guild Robotics pipeline on Jetson Thor: Shallow-π 9-layer (260 -> 127 ms), then one-step distillation, TensorRT and FP8 to 24 ms; matched base performance on one task and recovered another with a short BC step; DROID distillation used only the KD loss on a quality-filtered subset — [Guild Robotics](https://www.guildrobotics.com/blog/pi05-on-the-edge)
- DTR keep-2 language blocks (30% size, 25% FLOPs) still reaches 91.0 on LIBERO vs 91.7 baseline, which the authors read as benchmarks putting little pressure on language grounding — [DTR paper](https://arxiv.org/html/2606.27755v1)
- SmolVLA (0.45B) reports LIBERO 87.3% vs its π0 baseline, and in Shallow-π's table SmolVLA scores 87 at 26.0 ms on H100, slower than π0.5-L6 at 11.3 ms — [SmolVLA blog](https://huggingface.co/blog/smolvla); [Shallow-π](https://arxiv.org/html/2601.20262v1)
- A Haptic Labs blog (April 2026) is cited in search results as planning backbone shrinking for Jetson Orin Nano; the page returned 404 when fetched, so it is unverified — [Haptic Labs (404)](https://www.hapticlabs.ai/blog/2026/04/23/my-first-week-at-haptic-making-a-3b-robot-policy-faster)

### Inferences
- For us this collapses into section 3: "smaller backbone" in practice means the same Gemma with 6-9 layers. Shallow-π needs teacher and student training of 30k-100k steps and has no released code, so DTR (code and π0.5 checkpoints public) is the more practical starting point.
- With only about 500 demos, pure-KD distillation on teacher outputs (Guild's DROID recipe) is attractive because the fine-tuned checkpoint can label extra rollouts cheaply, but that is a multi-day project.

### Gaps
- No open π0.5 student with a different, smaller VLM (for example a sub-1B Gemma or SmolVLM backbone) was found. No Shallow-π code or weights.

## 6. What Nota AI, Intel, NVIDIA, Quadric and others published in 2026 on prefix-side optimisation of π0.5

### Takeaway
Only Nota AI has published a prefix-side reduction of π0.5 on Intel hardware (learned K=128 per-camera sampler). Intel's and NVIDIA's public material is about partitioning, placement and KV plumbing, with one useful structural trick each (masked-camera slots dropped; last-layer skip in third-party work), and I found nothing from Quadric.

### Cited Findings
- Nota AI: see section 1; in addition the card documents a split export with `prefix.xml` on the iGPU and `denoise.xml` on the NPU, K/V shaped [18, 1, 1, P, 256], and shows the NPU path is slower end to end (154 vs 117 ms; 402 vs 258 ms for base) — [Nota AI HF card](https://huggingface.co/geonmin-kim/nota-pi05-libero-vtp-drift-openvino)
- Intel Open Edge Platform article: vision encoder and language model on the iGPU, action expert on the NPU, per-layer KV cache as the only cross-device hand-off via zero-copy USM-host tensors; benchmarked π0.5 DROID with 3 cameras at 224x224, 64-token context, 10 steps; 294 ms on Core Ultra X7 358H at 40 W vs 287 ms on Jetson Thor (PyTorch). The readable text contains no token pruning, layer reduction or caching — [Intel OEP article source](https://docs.openedgeplatform.intel.com/dev/_sources/OEP-articles/publications/optimizing-pi0.5-lva-model.md.txt)
- Intel Robotics AI Suite documents a π0.5 OpenVINO pipeline with NNCF INT8 weight compression on the iGPU (search-result level) — [Intel Embodied Intelligence SDK docs](https://docs.openedgeplatform.intel.com/2026.1/edge-ai-suites/robotics-ai-suite/embodied/openvino_optimization.html)
- An open Physical AI Studio issue reports 0% LIBERO accuracy for OpenVINO π0.5 vs PyTorch XPU (search-result level; not read in full) — [physical-ai-studio issue 932](https://github.com/open-edge-platform/physical-ai-studio/issues/932)
- NVIDIA TensorRT Edge-LLM π0.5: four engines (visual, prefix, action, optional cond); prefix is "compact" (masked views and padded tokens not emitted); camera count and horizon fixed per export; FP16 only; no token pruning or layer reduction; no Jetson latency on that page — [TensorRT Edge-LLM π0.5 design](https://nvidia.github.io/TensorRT-Edge-LLM/0.11.0/developer_guide/models/pi05.html)
- NVIDIA GR00T N1.7 on Jetson Thor: 125 -> 39.9 ms with TensorRT and mixed NVFP4 quantisation, benchmarked with 1 camera and 4 denoising steps; this is quantisation/runtime work, not prefix reduction (search-result level) — [Jetson AI Lab tutorial](https://www.jetson-ai-lab.com/tutorials/groot_n17_on_thor/)
- Dexmal "Running VLAs at Real-time Speed" (π0-class, RTX 4090): graph-level rewrites (fold RMS affine, fuse QKV/RoPE, skip last LLM layer output) reaching 27.3 ms for 2 views; token reduction only named as future work. Code: github.com/Dexmal/realtime-vla — [Running VLAs at Real-time Speed](https://arxiv.org/html/2510.26742)

### Inferences
- Nota's configuration differs from ours in ways that matter when comparing numbers: X7 358H not X9 388H, FP16 not INT8, 200-token prompt not 64, two live cameras plus one masked slot, a 10-action output slice, and mean-over-episode latency including pre/post-processing. Our 140 ms (5 steps, INT8, three live cameras) is not directly comparable to their 117 ms (1 step, K=128).
- Their base 10-step number (258 ms) against our 169 ms suggests our INT8 + 64-token setup already banks most non-structural gains; the remaining prefix lever on this hardware is token count and depth.

### Gaps
- No Quadric publication on π0.5 found. No Intel publication applying token pruning or layer reduction to π0.5. No NVIDIA prefix-side reduction for π0.5 or GR00T found (token counts per camera for GR00T N1.6/N1.7 not confirmed in this pass).
- The Intel PDF version of the paper and its figures were not read; per-component numbers may be there.

## 7. Ranked shortlist for our situation (expert-only fine-tune, frozen backbone, about 500 demos, cloud GPUs, about one day, one static OpenVINO IR)

### Takeaway
Rank order: (1) remove or shrink the inspect view and retrain the expert; (2) fixed-K per-camera token selection before Gemma (VLA-ACL-style frozen-VLA selector, or a Nota-style sampler with fine-tuning); (3) verify the free last-layer skip; (4) drop Gemma blocks with recovery training as a second-round item; everything cache-based or dynamically shaped ranks last.

### Cited Findings
- Evidence for each row is in sections 1-6; the key anchors are Nota K=128 on Arc B390 (175 -> 117 ms, 94.0 -> 92.0%) — [Nota AI HF card](https://huggingface.co/geonmin-kim/nota-pi05-libero-vtp-drift-openvino); VLA-ACL K=64 on π0.5 (2.2x prefill, 96.9 -> 96.1%, VLA frozen, code public) — [VLA-ACL](https://arxiv.org/html/2610.08133v2); DTR language Drop-9 on π0.5 (57.9% FLOPs, 91.7 -> 93.3/94.0%, full fine-tune) — [DTR paper](https://arxiv.org/html/2606.27755v1); camera masking without retraining is destructive — [libero-camera-ablation repo](https://github.com/chaoshengsc/libero-camera-ablation); cross-step prefix K/V reuse failed on π0.5 — [CrossVLA PDF](https://arxiv.org/pdf/2605.21854)

### Inferences

Ranked table (speed-ups for our machine are linear estimates from 82 ms prefix / 22 ms vision; "published" columns are the closest sourced result):

| Rank | Technique | Mechanism | Published result (closest model) | Estimate for us | Training | Static shapes | Code | Effort |
|---|---|---|---|---|---|---|---|---|
| 1 | Drop the inspect view from the policy (or give it a 32-64 token budget) | 832 -> 576 prefix tokens, one fewer SigLIP pass | No retrained-π0.5 result; masking without retraining gives 4.6-55% on LIBERO-Spatial, so retraining is required; π0-class latency 36.8 -> 27.3 ms going 3 -> 2 views on a 4090 | Prefix about 57 ms, vision about 15 ms; about 30 ms saved | Expert-only retrain with our existing recipe | Yes | None needed | Low; under a day including retrain; needs closed-loop go/no-go |
| 2 | Fixed-K per-camera selector before Gemma, VLA frozen (VLA-ACL; APS-VLA when released) | Small scorer picks K of 256 tokens per view | π0.5 LIBERO: K=64, 96.9 -> 96.1%, prefill 2.2x; APS-VLA real robot 1.47x at equal success | K=128: prefix about 44 ms; K=64: about 25 ms (on three cameras) | Train scorer only, about 5k steps, hours on one GPU; unproven on an expert-only fine-tune | Yes (fixed TopK/Gather) | VLA-ACL public; APS-VLA not yet | Medium; about a day if the repo ports cleanly to our checkpoint; export work for TopK/Gather |
| 2b | Nota-style learned grid sampler K=128 | Bilinear grid sampling after projector | π0.5 on Arc B390: 175 -> 117 ms, about -1.5 to -2 points LIBERO-10 | Same as K=128 above | 10k-step fine-tune, trained parts unspecified | Yes, proven in OpenVINO IR | Checkpoints gated; no training code | Medium-high; we would re-implement the sampler |
| 3 | Last-layer skip / confirm dead-code elimination | Expert only needs the last layer's K/V | About 0.7 of 27 ms on a 4090 (π0) | Up to about 4 ms if not already eliminated | None; exact | Yes | Dexmal repo | Minutes to check the IR |
| 4 | Drop Gemma blocks with recovery (DTR Drop-9) | Remove 9 of 18 language blocks, keep their K/V projections | π0.5 LIBERO 91.7 -> 93.3/94.0 at 57.9% FLOPs; real robot roughly equal at Drop-9, worse at Drop-16 | Prefix about 45-50 ms; stacks with rows 1-2 | Full fine-tune 30k steps; frozen-backbone variant untested | Yes | Public code and π0.5 checkpoints | High; more than a day end to end; small-data risk |
| 5 | Depth distillation (Shallow-π 9 or 6 layers) | Student with fewer backbone and expert layers | π0.5 96 -> 95 LIBERO; Thor 130 -> 78 ms; Guild 260 -> 127 ms at 9 layers | Roughly halves prefix and expert | Teacher-student, 30k-100k steps | Yes | None | High; multi-day |
| 6 | Training-free in-LLM pruning (FastV, VLA-Pruner, SAFE-Pruner) | Drop low-attention tokens at a middle layer | π0.5: 1.25-1.27x with 1.4-3 point loss (second-hand) | Under about 20 ms | None | Fixed-ratio FastV yes; temporal variants need state | VLA-Pruner public; SAFE-Pruner not | Medium; smaller gain than row 2 at higher accuracy cost |
| 7 | Cross-step prefix / KV / feature caching (VLA-Cache style, keyframes) | Reuse stale K/V or features | π0.5: failed (0/1, 0/2); OpenVLA-OFT: silent failures unless a dense refresh is added | Not recommended | None | No (dynamic sets, cross-call state) | VLA-Cache public | High, with correctness risk |
| 8 | Sub-224 input resolution | Fewer SigLIP patches | No π0/π0.5 evidence | Unknown | Likely needs backbone adaptation | Yes | n/a | Not recommended before rows 1-2 |
| - | Online channel pruning (EcoVLA) | Environment-adaptive sparsity | π0.5 1.31x at -0.2 points | n/a | None | No (dynamic pattern) | None | Not applicable to a static IR |

- Rows 1 and 2 compose: two cameras at K=128 gives a 320-token prefix (about 32 ms by linear estimate) plus about 15 ms vision, which would put a 5-step chunk near 95-100 ms on our machine if the expert time is unchanged. This is an estimate to be measured, not a claim.
- Order of work for one day: check row 3 in the IR; start the two-camera expert retrain (row 1); in parallel train a VLA-ACL scorer on the frozen checkpoint (row 2) and evaluate both separately before stacking, since their accuracy costs have never been measured together or on an expert-only fine-tune.

### Gaps
- None of the ranked items has been measured on INT8 weights or on an expert-only fine-tuned π0.5; interaction between token reduction and INT8 accuracy is unknown.
- All sim evidence is LIBERO/RoboCasa/RoboTwin; real-robot evidence on π0.5 exists only for APS-VLA (5 tasks), DTR (one grasping setup), Shallow-π and Guild.
