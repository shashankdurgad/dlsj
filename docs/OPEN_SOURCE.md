# Open source release

Everything on Hugging Face: https://huggingface.co/collections/akoniti/so-101-duplo-inspection-data-vla-digital-twin-6af9610802014765d70d3423

Our entry to the **Intel Physical AI challenge at Junction 2026**: an SO-101 arm that inspects and sorts Duplo
bricks, built with Intel Physical AI Studio, Anomalib and OpenVINO and run on an Intel Core Ultra laptop (CPU, GPU
and NPU).

## Overview

An end-to-end manufacturing-inspection cell built for reliability, robustness, adaptability and consistent
throughput. The arm picks a brick from anywhere in its workspace, shows it to an inspection camera, decides with
anomaly detection whether it is good, slightly defective or defective, and places it on the pile of the colour
assigned to that class (the colour assignment is configurable). It runs autonomously, with online reasoning for
adaptive behaviour, and we demonstrated it out of distribution: changing lighting, moving objects and moved plates.

## Approach

- **Data:** 1,106 teleoperated episodes (~2.2 h at 30 fps), with a mix of teleop speeds and tasks (full pick,
  inspect and place demos, plus pick-only takes), focused on gripping and failure modes.
- **Policy training:** LoRA supervised fine-tuning (pre-post-training) of the MolmoAct2 VLA with a specialised
  expert-data schedule (faster demonstrations towards the end of the schedule); "Astra WSP" for fine-tuned failure
  modes, as specialised post-post-training; the policy is conditioned on a text prompt for adaptability in deployment.
- **Simulation and RL:** a digital twin of the station in Isaac Sim / Isaac Lab, integrated with the Physical AI
  Studio pipeline (fitted cameras, servo and gripper physics, a camera sensor model, full domain randomisation), used
  for RL as mid-post-training: a residual RL policy (DICE-RL) on top of the frozen VLA prior for the pick, trained
  under perturbations.
- **Vision:** an ensemble of computer-vision models for anomaly detection. PatchCore (Anomalib) is the core anomaly
  model and detects any marking on a brick (92 % overall accuracy on a 129-image test set); YOLO detects, segments
  and identifies bricks (77.2 % box mAP). Trained on our own collected and labelled data (~200 images per class) with
  augmentation (rotation, brightness, zoom, ...).
- **Deployment on Intel hardware:** the vision models run on the NPU, the VLA runs as OpenVINO INT8 on the Arc GPU,
  and cameras, rules and control run on the CPU.

## Data, models and code

| What | Where | Licence |
|---|---|---|
| Code (recording, training, deployment, digital twin) | this repository; the twin is in [`sim/twin/`](../sim/twin/) | MIT ([`LICENSE`](../LICENSE)) |
| Teleop demos, current calibration (456) plus RGB copies of the first campaign | [akoniti/dlsj-canonical](https://huggingface.co/datasets/akoniti/dlsj-canonical) | CC BY 4.0 |
| First campaign in its native recording format, with depth (517 pick-and-place + 133 pick-only) | [akoniti/dlsj-teleop-campaign1](https://huggingface.co/datasets/akoniti/dlsj-teleop-campaign1) | CC BY 4.0 |
| Curated post-training set (651 episodes, one calibration frame; derived from the above) | [akoniti/dlsj-posttrain-v3-1](https://huggingface.co/datasets/akoniti/dlsj-posttrain-v3-1) | CC BY 4.0 |
| MolmoAct2 fine-tune (step 8000) | [akoniti/dlsj-molmoact2-ft-8k](https://huggingface.co/akoniti/dlsj-molmoact2-ft-8k) | Apache-2.0 (the base model's licence) |
| Digital twin assets and residual RL checkpoints | [akoniti/dlsj-so101-twin](https://huggingface.co/datasets/akoniti/dlsj-so101-twin) | CC BY 4.0 (data / assets), MIT (code) |

Third-party parts (SO-101 URDF and meshes, the fin-ray finger CAD, LDraw Duplo parts, MolmoAct2, Isaac Lab / Isaac
Sim, Ultralytics YOLO, ...) keep their own licences: see [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).
The inspection camera in the datasets may show people at the venue in the background.
