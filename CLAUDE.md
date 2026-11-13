# CLAUDE.md

Hackathon repo for **Intel's challenge at Junction**: end-to-end robotic defect detection and manipulation with an SO-101 arm, using **Intel Physical AI Studio + Anomalib + OpenVINO**, deployed on a **Core Ultra Series 3** laptop using **CPU, GPU and NPU**. Closed loop: **PERCEIVE → DETECT → REASON → ACT → OPTIMIZE**. Primary task: tell "perfect" from "defective" objects and sort them per the kickoff rules.

**Read `docs/PLAYBOOK.md` before planning or writing code.** It is the source of truth for decisions, commands and code snippets. The Challenge brief and Section 8 describe the system.

## Hardware
- SO-101 leader/follower arms.
- Cameras: a fixed **overhead RealSense** RGB-D (likely D405, <50 cm from the workspace) and a plain **RGB webcam on the wrist**.
- **Deploy machine:** Dell XPS 16 DA16260: Core Ultra X9 388H, Arc B390 iGPU, NPU, 64 GB, Ubuntu 26.04. Reach it with `ssh xps` (user `intel`, over a teammate's Tailscale network). Setup and environments: PLAYBOOK §1.0.
- **Training:** cloud/team NVIDIA RTX GPUs.

## Decisions already made (don't re-litigate without a reason)
- **VLA:** **π0.5 trained via Physical AI Studio** on cloud RTX, exported to OpenVINO INT8, run with the `physicalai` runtime on the XPS GPU. SmolVLA is the fallback; **ACT is the baseline**. No Chinese VLA beats π0.5 in independent tests (PLAYBOOK §2.1, `reports/XPU kernels and Chinese VLAs.md`).
- **Device split:** Anomalib on the **NPU** (static shapes); π0.5 on the **GPU**; cameras, rules and control on the **CPU**. Stretch: π0.5 action expert on the NPU.
- **REASON:** a rule table maps (defect flag, colour, kickoff rules) to a fixed instruction string, which conditions π0.5. Use identical strings in training demos and at run time.
- **Defect detection:** Anomalib PaDiM/PatchCore on the fixed overhead view (arm parked out of frame), plus a depth height-map check against the table plane.
- **Force:** gripper `Present_Current` in `observation.state` (never in actions), plus a rule-based grasp check.
- **OPTIMIZE:** always report baseline vs optimised median/p99 latency (PLAYBOOK §8.1). The LeRobot PyTorch-XPU policy server in `~/inference/.venv` is the baseline.
- **XPU inference:** follow the PLAYBOOK §4.5 checklist. `torch.compile(mode="reduce-overhead")` is a no-op on XPU; no IPEX; don't quantise ACT.

## Conventions
- Record and run at 30 fps on both cameras; RealSense at 640×480@30. Auto-exposure and auto-WB **locked** on both (webcam via `v4l2-ctl`). Never move the overhead camera.
- VLA export uses 2 cameras (`-c 2`), batched into one vision call.
- Apply any sensor filtering (median/EMA) identically at record and inference time.
- Items marked ⚠️ Unverified in the playbook must be smoke-tested on hardware before relying on them.
