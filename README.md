# dlsj: Intel Physical AI challenge (SO-101, Junction)

End-to-end robotic **defect detection and manipulation**: an SO-101 arm tells "perfect" from "defective" objects and sorts them by the kickoff rules. Built on **Intel Physical AI Studio + Anomalib + OpenVINO**, deployed on a **Core Ultra Series 3** laptop (Dell XPS 16, X9 388H) using **CPU, GPU and NPU**.

The full plan, commands, code snippets and sources are in **[docs/PLAYBOOK.md](docs/PLAYBOOK.md)**.

## The loop

| Step | What | Where |
|---|---|---|
| **PERCEIVE** | Overhead RealSense (RGB + depth) + wrist webcam | CPU |
| **DETECT** | Anomalib PaDiM/PatchCore (OpenVINO) + depth height check → perfect / defective | **NPU** |
| **REASON** | Kickoff rules + defect flag + colour → language instruction | CPU |
| **ACT** | π0.5 fine-tuned via Physical AI Studio, OpenVINO INT8, `physicalai` runtime with RTC | **GPU** |
| **OPTIMIZE** | Baseline (PyTorch-XPU) vs OpenVINO FP16 / INT8 / NPU latency, tasks/min | all |

Training runs on cloud RTX GPUs; ACT is the baseline policy and SmolVLA the fallback VLA.

## Timeline

1. **0–1 h:** XPS is ready (`ssh xps`); verify the arm and both cameras; write down the kickoff rules.
2. **1–3 h:** collect demos with language instructions; capture good/defective images for Anomalib.
3. **3–5 h:** train π0.5 + ACT on cloud RTX via Studio; train Anomalib on the XPS and export to the NPU.
4. **5–7 h:** export π0.5 to OpenVINO INT8, deploy, wire REASON and the grasp check.
5. **7 h+:** test on held-out lighting; fix failures with more demos; fill in the OPTIMIZE table.
6. **Final:** full-loop demo with the anomaly heatmap and live per-stage latency.

## Key tools

- [Physical AI Studio](https://github.com/open-edge-platform/physical-ai-studio): train and export policies (ACT, π0.5, SmolVLA, …)
- [physicalai runtime](https://github.com/openvinotoolkit/physicalai): deploy on the robot with OpenVINO
- [Anomalib](https://github.com/open-edge-platform/anomalib): anomaly detection
- [OpenVINO](https://github.com/openvinotoolkit/openvino): CPU/GPU/NPU inference
- [LeRobot](https://github.com/huggingface/lerobot): recording, dataset tools, PyTorch baseline
