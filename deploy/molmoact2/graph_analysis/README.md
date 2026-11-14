# MolmoAct2 graph-level analysis: loop-invariant work in the 10 flow steps

14 Nov 2026, XPS Arc B390, zero-shot INT8 export (`molmoact2_so101_rtc_int8`).

**Verdict: nothing to hoist.** Studio and OpenVINO already remove all step-invariant work, and the expert's attention
runs on OpenVINO's best fused kernel. The remaining expert levers are model-level: fewer flow steps and 128-token text.

1. **Source** (Studio 8f13241). `MolmoAct2Backbone._encode_action_context` runs once before the Euler loop. It
   precomputes the per-layer KV context (expert `context_k_proj`/`context_v_proj` + RMS norms + `k_norm`), the
   cross/self masks and the RoPE cache (`ActionExpertContext`, "shared across denoising steps").
   Inside each step, the only work that doesn't depend on the noisy actions is the time conditioning: `time_embed`
   (sinusoid -> MLP) and every block's adaLN `modulation` Linear, plus the final layer's. All are functions of the
   constant `t = step/steps`.
2. **IR** (`graph_invariants.py`): 391 MatMuls depend only on constants and static shapes:
   - 360 block modulation GEMVs [1x768]x[6912x768] (36 blocks x 10 steps)
   - 20 time_embed
   - 10 final-layer modulation
   - 1 RoPE

   Executed, they would stream about 1.9 GB of weights per chunk.
3. **Compiled GPU graph** (`profile_invariant.py`, `profile_status.py`): none of them exist in the runtime model. All 810
   profiling entries are `OPTIMIZED_OUT` (0.0 ms): the GPU plugin constant-folds them at compile time.
   Results: `result_invariant_profile_int8.json`, `result_status_kernels_int8.json`.
4. **Expert attention** (`profile_status.py`): all 720 SDPA calls (36 x 10 x self+cross) use `ocl::sdpa::opt` f16.
   They total 84 ms with per-op profiling on, which serialises kernels and inflates totals.
   Micro-benchmarks at the exact shapes (`sdpa_micro.py`, `sdpa_layout.py`):
   - **Fused vs decomposed:** fused SDPA beats decomposed MatMul-Softmax-MatMul by 2-4x.
     - cross 30x650: 31-78 us fused vs 73-124 us decomposed, max |diff| 5e-4
     - self 30x30: 6-7 us fused vs 10-17 us decomposed, max |diff| 2e-3
   - **K/V layout:** transposed-input vs contiguous makes no difference (68 vs 77.5 us). Run-to-run clock variance on
     these tiny kernels is about 2x.

   So no graph rewrite of the attention helps.

**What does reduce expert time:**
- **Fewer flow steps.** Each step is 72 SDPA calls plus 36 blocks of GEMMs, about 11 ms in INT8.
- **A shorter prefix.** Cross-attention reads all 650 K/V rows every step. 128-token text trims the 158 pad rows; it is
  trained in via `training/molmoact2/policy_text.py`.

Both levers are in `deploy/molmoact2/optimize.py` for the fine-tuned checkpoint.
