#!/usr/bin/env bash
# Run the full π0.5 OpenVINO latency matrix on the XPS and append JSON lines to $OUT.
#
#   bash ~/inference/deploy/pi05/run_benchmarks.sh [iters]      # default 30 iterations per row
#   BASELINE=1 bash ~/inference/deploy/pi05/run_benchmarks.sh   # also rerun the PyTorch-XPU baseline
#
# Expects the exports from README.md under ~/inference/exports. Check the machine is quiet first
# (`pgrep -fa python`, `uptime`): other GPU/CPU jobs inflate every number here.
set -uo pipefail
ITERS=${1:-30}
E=~/inference/exports
D=~/inference/deploy/pi05
OUT=~/inference/logs/pi05/final_$(date +%Y%m%d_%H%M).jsonl
STUDIO=~/inference/physical-ai-studio/library/.venv/bin/python   # OpenVINO 2026.4.1
INTEL=~/inference/.venv-pi05ov/bin/python                          # OpenVINO 2026.0 (Intel's pin)

echo "# $(date)  $(uptime)" | tee -a "$OUT.log"
pgrep -fa python | grep -v "networkd\|unattended\|resource_tracker" | cut -c1-120 | tee -a "$OUT.log"

# 1. Studio runtime (InferenceModel: tokenizer + pre/post + model), the deployment path.
# (The tok64 export cannot run through the runtime with pi05_base: its 32-d state makes a 135-token prompt.)
for x in pi05_studio_fp16 pi05_studio_int8 pi05_studio_int8_5step pi05_studio_int8_1step; do
  [ -d "$E/$x" ] && $STUDIO $D/studio_bench.py "$E/$x" --device GPU --iters "$ITERS" --json "$OUT" 2>/dev/null | grep '^{'
done

# 2. Raw IR, same harness, OpenVINO 2026.4.1: Studio vs Intel graphs.
for x in pi05_studio_int8/pi05.xml pi05_studio_int8_tok64/pi05.xml pi05_studio_int8_5step/pi05.xml \
         pi05_studio_int8_1step/pi05.xml intel_pi05/pi05_rtc_3c_INT8/model.xml intel_pi05/pi05_rtc_3c_FP16/model.xml; do
  $STUDIO $D/ov_bench.py "$E/$x" --device GPU --iters "$ITERS" --json "$OUT" 2>/dev/null | grep '^{'
done

# 3. Intel IR on Intel's pinned OpenVINO 2026.0.
$INTEL $D/ov_bench.py $E/intel_pi05/pi05_rtc_3c_INT8/model.xml --device GPU --iters "$ITERS" --json "$OUT" 2>/dev/null | grep '^{'
# 3b. Intel's own benchmark (their LeRobot pre/post + OV; prints means only).
RAS=~/inference/robotics-ai-suite/src/pipelines/pi05-rtc-ov/lerobot/examples/pi05_with_openvino/scripts
(cd ~/inference/exports/intel_pi05 && $INTEL $RAS/benchmark_pi05_ov_rtc.py --model_dir pi05_rtc_3c_INT8 \
  --device GPU -c 3 -n "$ITERS" 2>/dev/null | grep "Average\|Total") | sed 's/^/intel_benchmark INT8 GPU: /' | tee -a "$OUT.log"

# 4. Compile-option A/B on the Studio INT8 IR.
$STUDIO $D/ov_bench.py $E/pi05_studio_int8/pi05.xml --device GPU --iters "$ITERS" --precision_hint f16 --json "$OUT" 2>/dev/null | grep '^{'
$STUDIO $D/ov_bench.py $E/pi05_studio_int8/pi05.xml --device GPU --iters "$ITERS" \
  --config GPU_ENABLE_SDPA_OPTIMIZATION=NO --json "$OUT" 2>/dev/null | grep '^{'
# 5. Optional: PyTorch-XPU baseline (bench/pi05_torch_xpu.py) under the same conditions.
if [ "${BASELINE:-0}" = 1 ]; then
  (cd ~/inference && .venv/bin/python bench/pi05_torch_xpu.py bfloat16 10 2>/dev/null | tail -3) | tee -a "$OUT.log"
fi
echo "# done $(date)  $(uptime)" | tee -a "$OUT.log"
echo "results: $OUT"
