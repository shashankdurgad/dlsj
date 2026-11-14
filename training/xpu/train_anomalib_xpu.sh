#!/usr/bin/env bash
# Train the two defect-detection models on the Intel XPU (Arc iGPU) of the XPS, with our existing trainers:
#   stage 1: Anomalib PaDiM (plain vs marked)       -> python -m dlsj.inspect.train --accelerator xpu
#   stage 2: circle-vs-cross CNN (MobileNetV3-small) -> python -m dlsj.inspect.train_classifier --accelerator xpu
# Both export a static-shape OpenVINO IR and calibrate/evaluate it on $CALIB_DEVICE.
#
#   training/xpu/train_anomalib_xpu.sh padim      <inspect frames dir: good/ [bad/ minor/]> <out dir>
#   training/xpu/train_anomalib_xpu.sh classifier <crops dir: circle/ cross/> <out dir> [test crops dir]
#
# Defaults: niced, OpenVINO calibration on the NPU. If someone else is using the NPU (e.g. a recording session
# running the live detector), use CALIB_DEVICE=CPU. Run from the repo root on the XPS with the Anomalib env.
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=${PY:-$HOME/inference/.venv-anomalib/bin/python}   # Anomalib 2.7, OpenVINO 2026.4.1, torch 2.15+xpu
CALIB_DEVICE=${CALIB_DEVICE:-NPU}
NICE=${NICE:-19}

mode=${1:-}; data=${2:-}; out=${3:-}
if [ -z "$mode" ] || [ -z "$data" ] || [ -z "$out" ]; then
  sed -n '2,12p' "$0" >&2; exit 2
fi
"$PY" -c 'import torch, sys; sys.exit(0 if torch.xpu.is_available() else "torch.xpu is not available in this env")'

case "$mode" in
  padim)
    exec nice -n "$NICE" "$PY" -m dlsj.inspect.train --data "$data" --out "$out" --model padim \
      --accelerator xpu --calib-device "$CALIB_DEVICE" "${@:4}" ;;
  classifier)
    test_args=(); if [ -n "${4:-}" ]; then test_args=(--test "$4"); fi
    exec nice -n "$NICE" "$PY" -m dlsj.inspect.train_classifier --data "$data" --out "$out" \
      --arch mobilenet_v3_small --epochs "${EPOCHS:-15}" --accelerator xpu --device "$CALIB_DEVICE" "${test_args[@]}" ;;
  *) echo "mode must be padim or classifier" >&2; exit 2 ;;
esac
