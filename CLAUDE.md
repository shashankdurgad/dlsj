# CLAUDE.md

Hackathon repo for an SO-101 robot arm (leader/follower) on an Intel GPU (`xpu` device). Two cameras: a fixed **overhead RealSense** RGB-D (likely D405, <50 cm from the workspace) and a plain **RGB webcam on the wrist**.
Tasks: pick-and-place, colour sorting, anomaly detection. Judged (assumed) on success, robustness to lighting/texture/colour, and speed.

**Read `docs/PLAYBOOK.md` before planning or writing code.** It is the source of truth for decisions, commands and code snippets.

## Decisions already made (don't re-litigate without a reason)
- Motor policy: **ACT** trained in **bf16 on XPU**. Avoid fp16 `GradScaler`, since Arc A-series lacks FP64.
- Deploy: OpenVINO export + `physicalai` runtime with `AsyncExecution`; **no temporal ensembling**.
- Colour sorting: perception (HSV or a tiny classifier) **outside** the policy, then a per-bin policy. Keep hue jitter ≈ 0 if the policy itself must see colour.
- Anomaly detection: Anomalib PaDiM/PatchCore on the fixed overhead RealSense view (arm parked out of frame), separate from the policy; add a depth height-map check against the table plane.
- Force: gripper `Present_Current` in `observation.state` (never in actions), plus a rule-based grasp check.
- VLA (parallel track): **π0.5** (SnapFlow 1-step → OpenVINO INT8 + RTC) if a ≥24 GB card, else **SmolVLA**. No Chinese VLA beats π0.5 in independent tests; X-VLA/EVO1 only as smoke tests. See PLAYBOOK §2.1 and `reports/XPU kernels and Chinese VLAs.md`.
- XPU inference: follow the PLAYBOOK §4.5 checklist. `torch.compile(mode="reduce-overhead")` is a no-op on XPU; use manual `torch.xpu.graph`. No IPEX. Don't quantise ACT.

## Conventions
- Record and run at 30 fps on both cameras; RealSense at 640×480@30. Auto-exposure and auto-WB **locked** on both (webcam via `v4l2-ctl`). Never move the overhead camera.
- VLA export uses 2 cameras (`-c 2`), batched into one vision call.
- Apply any sensor filtering (median/EMA) identically at record and inference time.
- Items marked ⚠️ Unverified in the playbook must be smoke-tested on hardware before relying on them.
