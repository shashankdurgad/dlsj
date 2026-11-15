# Pickup-only ACT, step 7000

Held-out loss 0.3034 on 12 chronological held-out episodes. This is not a physical pickup success rate. No hardware rollout has been performed for this checkpoint.

Complete native LeRobot checkpoint: model.safetensors, config.json, saved pre/postprocessors and normalization tensors. Keep these files together.

On the Intel XPS, use the existing dlsj.deploy.act_rollout runner with this directory as --policy_path. Retain the actual wrist/top/inspect camera mappings from the recorder and the cal2 robot calibration. This model uses full 640x480 RGB views: pass --top-crop none and do not add resizing, padding, or the old recalibration joint offset. Run --dry-run first to check prediction/latency on current observations. No robot movement was started during checkpoint preparation.

Suggested initial options: --device xpu --top-crop none --n-action-steps 10 --max-seconds 10 --dry-run. For a subsequent physical test, keep the runner's explicit movement confirmation and bounded speed/duration controls. A completed rollout still needs observation of an actual grasp and lift to count as success.

Private source: gs://onflow-30k-pi05-artifacts/act/runs/act-cal2-grasp120-20261114/test_checkpoints/step7000
