# dlsj: SO-101 Robot Learning Hackathon (Intel XPU)

Pick-and-place, colour sorting and anomaly detection on an SO-101 leader/follower arm with a fixed overhead Intel RealSense depth camera and a wrist RGB webcam, trained and deployed on an Intel GPU (XPU).

The full plan, commands, code snippets and sources are in **[docs/PLAYBOOK.md](docs/PLAYBOOK.md)**.

## Plan at a glance

| Component | Choice |
|---|---|
| Motor policy | ACT, bf16 on XPU, 20–40k steps |
| Deployment | OpenVINO export + `physicalai` runtime, AsyncExecution, chunk 50–100, no temporal ensembling |
| Data | ~150 brisk demos across 6–10 lighting/background conditions, idle frames trimmed |
| Robustness | LeRobot image augmentation (brightness, contrast, shadow, white balance), hue jitter ≈ 0 |
| Sorting | HSV colour selector → per-bin ACT policy (the decision stays outside the policy) |
| Anomaly detection | Anomalib PaDiM/PatchCore on the overhead RealSense view, OpenVINO export, plus a depth height check |
| Force | Gripper `Present_Current` in `observation.state` + rule-based grasp verification |

## Timeline

1. **0–1 h:** environment setup; verify the arm and both cameras.
2. **1–3 h:** collect diverse demos; capture good/bad images for anomaly detection.
3. **3–5 h:** trim data, train ACT on XPU, train Anomalib.
4. **5–7 h:** export to OpenVINO, deploy async, add the HSV selector and grasp check.
5. **7 h+:** test on held-out lighting/background and fix failures with more demos of that condition.
6. **Final:** demo video with the anomaly heatmap, live scores and a lighting-change test.

## Key tools

- [physicalai runtime](https://github.com/openvinotoolkit/physicalai): deployment (`pip install "physicalai[so101,realsense]"`)
- [Physical AI Studio](https://github.com/open-edge-platform/physical-ai-studio): training and OpenVINO export
- [LeRobot](https://github.com/huggingface/lerobot): alternative training stack with image augmentation
- [Anomalib](https://github.com/open-edge-platform/anomalib): anomaly detection (`pip install "anomalib[xpu,openvino]"`)
- [Robotics AI Suite](https://github.com/open-edge-platform/robotics-ai-suite): reference OpenVINO pipelines
