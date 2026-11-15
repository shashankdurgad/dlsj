# dlsj: Intel Physical AI challenge (SO-101, Junction)

End-to-end robotic **defect detection and manipulation**: an SO-101 arm picks a brick and shows it to an inspection camera. **Anomalib** decides whether the brick is plain or marked and finds the mark; a **classifier** tells a **circle** (minor defect) from a **cross** (defective); a decision tree then has the VLA place it on the good, minor-defect or defective plate.

For the trained **PatchCore brick anomaly model and live camera tester**, see
[download and setup](models/brick_patchcore_geometry/README.md). This standalone
tester reports marked versus unmarked; it does not run the robot or stage-2 classifier.

The optional [trained YOLO classifier](models/brick_yolo_classifier/README.md)
adds plain/cross/circle labels and an uncertain/reinspect result to that camera
tester while retaining Anomalib's anomaly score and heatmap.

The full plan, commands, code snippets and sources are in **[docs/PLAYBOOK.md](docs/PLAYBOOK.md)**.

## Open data and models

Everything on Hugging Face: https://huggingface.co/collections/akoniti/so-101-duplo-inspection-data-vla-digital-twin-6af9610802014765d70d3423

Write-up of the hackathon entry and our approach: **[docs/OPEN_SOURCE.md](docs/OPEN_SOURCE.md)**.

- Datasets (CC BY 4.0, LeRobot format): [akoniti/dlsj-canonical](https://huggingface.co/datasets/akoniti/dlsj-canonical),
  [akoniti/dlsj-teleop-campaign1](https://huggingface.co/datasets/akoniti/dlsj-teleop-campaign1),
  [akoniti/dlsj-posttrain-v3-1](https://huggingface.co/datasets/akoniti/dlsj-posttrain-v3-1)
- Model (Apache-2.0, as its base MolmoAct2): [akoniti/dlsj-molmoact2-ft-8k](https://huggingface.co/akoniti/dlsj-molmoact2-ft-8k)
- Digital twin assets and residual RL checkpoints: [akoniti/dlsj-so101-twin](https://huggingface.co/datasets/akoniti/dlsj-so101-twin)
- Licence: code under MIT ([LICENSE](LICENSE)); third-party parts under their own licences
  ([THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)).

## The loop

| Step | What | Where |
|---|---|---|
| **PERCEIVE** | Cameras `wrist`, `top`, `inspect`; the arm picks a brick and presents it to `inspect` | CPU |
| **DETECT** | Stage 1: Anomalib PaDiM (OpenVINO) → plain vs marked + region; stage 2: cross vs circle classifier on the region | **NPU** / CPU |
| **REASON** | Decision tree: plain → yellow bowl, circle → orange bowl, cross → purple bowl; uncertain → re-inspect → purple | CPU |
| **ACT** | π0.5 does every movement: pick and show to `inspect`, then place in the yellow (good), orange (minor) or purple (defective) bowl (OpenVINO INT8, `physicalai` runtime) | **GPU** |
| **OPTIMIZE** | Baseline (PyTorch-XPU) vs OpenVINO FP16 / INT8 / NPU latency, tasks/min | all |

No ACT: one π0.5 model handles pick and place by instruction (Intel's reference uses ACT + SmolVLA; SmolVLA is our fallback). Training runs on the Modal GPU instance.

**Phase 2 (extension):** bricks are numbered; a number reader on the `top` camera (NPU) and a planner sort them, π0.5 picks each brick in order, and `inspect` re-checks the number and defects before placing. See PLAYBOOK §8.2.

## Physical AI Studio

The Studio project **dlsj brick sorting** on the XPS holds the robot, the three cameras, the environment, the
training dataset and the deployed VLA (MolmoAct2 fine-tuned with Studio's training library, exported with Studio's
OpenVINO exporter). The run-time loop (`deploy/smolvla/run_loop.sh`) picks its policy by **Studio model ID** and
runs it with Studio's runtime (`physicalai`, OpenVINO on the GPU). Details, IDs and commands:
[deploy/studio/README.md](deploy/studio/README.md).

## Timeline

1. **0–1 h:** XPS is ready (`ssh xps`); verify the arm and the three cameras; write down the kickoff rules.
2. **1–3 h:** collect demos with language instructions; capture good/defective images for Anomalib.
3. **3–5 h:** fine-tune π0.5 on the Modal GPU; train Anomalib on the XPS and export to the NPU.
4. **5–7 h:** export π0.5 to OpenVINO INT8, deploy, wire REASON and the grasp check.
5. **7 h+:** test on held-out lighting; fix failures with more demos; fill in the OPTIMIZE table.
6. **Final:** full-loop demo with the anomaly heatmap and live per-stage latency.

## Team tools (`dlsj/`)

- `dlsj.teleop_web` / `dlsj.teleop_record`: leader→follower teleop and LeRobot v3 recording in the browser
- `dlsj.kinesthetic_web`: hand-guided (kinesthetic) recording
- `dlsj.arms` + `calibration/`: port discovery and calibration stored in the repo

## Key tools

- [Physical AI Studio](https://github.com/open-edge-platform/physical-ai-studio): train and export policies (ACT, π0.5, SmolVLA, …)
- [physicalai runtime](https://github.com/openvinotoolkit/physicalai): deploy on the robot with OpenVINO
- [Anomalib](https://github.com/open-edge-platform/anomalib): anomaly detection
- [OpenVINO](https://github.com/openvinotoolkit/openvino): CPU/GPU/NPU inference
- [LeRobot](https://github.com/huggingface/lerobot): recording, dataset tools, PyTorch baseline
