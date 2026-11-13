# CLAUDE.md

Hackathon repo for an SO-101 robot arm (leader/follower, wrist RealSense, likely a D405) on an Intel GPU (`xpu` device).
Tasks: pick-and-place, colour sorting, anomaly detection. Judged (assumed) on success, robustness to lighting/texture/colour, and speed.

**Read `docs/PLAYBOOK.md` before planning or writing code.** It is the source of truth for decisions, commands and code snippets.

## Decisions already made (don't re-litigate without a reason)
- Motor policy: **ACT** trained in **bf16 on XPU**. Avoid fp16 `GradScaler`, since Arc A-series lacks FP64.
- Deploy: OpenVINO export + `physicalai` runtime with `AsyncExecution`; **no temporal ensembling**.
- Colour sorting: perception (HSV or a tiny classifier) **outside** the policy, then a per-bin policy. Keep hue jitter ≈ 0 if the policy itself must see colour.
- Anomaly detection: Anomalib PaDiM/PatchCore at a fixed inspection pose, separate from the policy; add a depth height-map check.
- Force: gripper `Present_Current` in `observation.state` (never in actions), plus a rule-based grasp check.

## Conventions
- Record and run at 30 fps; RealSense at 640×480@30 with auto-exposure and auto-WB **locked**.
- Apply any sensor filtering (median/EMA) identically at record and inference time.
- Items marked ⚠️ Unverified in the playbook must be smoke-tested on hardware before relying on them.
