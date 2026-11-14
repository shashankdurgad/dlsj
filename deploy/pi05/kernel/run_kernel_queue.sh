#!/usr/bin/env bash
# Job queue for the π0.5 kernel study on the XPS. Each stage waits until the machine is free of the other agents'
# heavy jobs (NNCF quantisation, GPU benchmarks/evals) with >= 25 GB available, then runs under
# `flock ~/inference/tmp/heavy.lock` at nice 10. Logs: ~/inference/logs/pi05/kernel/.
#
#   nohup bash run_kernel_queue.sh A B C > ~/inference/logs/pi05/kernel/queue.log 2>&1 &
#   A = release 2026.4.1: kernel profiles, benchmark_app -pc, config sweep, host overhead, dyn-quant accuracy
#   B = build OpenVINO 2026.4.1 with debug caps (+ GeGLU patch)
#   C = debug build: GatedMLP off / stock on / GeGLU on, profiles, latency, accuracy
set -uo pipefail
K=$(cd "$(dirname "$0")" && pwd)
I=$HOME/inference
LOG=$I/logs/pi05/kernel
PY=$I/physical-ai-studio/library/.venv/bin/python
DPY=$I/.venv-ovdbg/bin/python
EVAL=$I/deploy/pi05/opt/eval_actions.py
CACHE=$I/eval/pi05_val300.npz
INT8=$I/exports/pi05_junction_int8
FP16=$I/exports/pi05_junction_fp16
mkdir -p "$LOG" "$I/eval/preds/kernel"

wait_idle() {
  while true; do
    busy=$(pgrep -f "python[0-9.]* .*(quantize\.py|bench_variants|eval_actions|studio_bench)" | grep -v -w $$ || true)
    avail=$(free -g | awk '/^Mem:/{print $7}')
    if [[ -z $busy && $avail -ge 25 ]]; then return; fi
    echo "$(date +%T) waiting: busy=[$(echo $busy | tr '\n' ' ')] avail=${avail}G"
    sleep 180
  done
}

locked() { flock "$I/tmp/heavy.lock" nice -n 10 "$@"; }

stage_A() {
  cd "$I/physical-ai-studio/library"
  B=$I/physical-ai-studio/library/.venv/bin/benchmark_app
  locked bash -c "
    set -x
    [ -s $LOG/prof_int8_release.json ] || $PY $K/profile_kernels.py $INT8/pi05.xml --out $LOG/prof_int8_release.json > $LOG/prof_int8_release.log 2>&1
    mkdir -p $LOG/bapp_int8
    [ -s $LOG/bapp_int8/benchmark_detailed_counters_report.csv ] || $B -m $INT8/pi05.xml -d GPU -hint latency -niter 20 -pc -pcsort sort -report_type detailed_counters \
       -report_folder $LOG/bapp_int8 > $LOG/bapp_int8.log 2>&1
    [ -s $LOG/prof_fp16_release.json ] || $PY $K/profile_kernels.py $FP16/pi05.xml --out $LOG/prof_fp16_release.json > $LOG/prof_fp16_release.log 2>&1
    [ -s $LOG/sweep_int8.jsonl ] || $PY $K/sweep.py $INT8 --rounds 3 --iters 40 --out $LOG/sweep_int8.jsonl > $LOG/sweep_int8.log 2>&1
    [ -s $LOG/host_calls.json ] || $PY $K/host_overhead.py calls $INT8/pi05.xml --out $LOG/host_calls.json > $LOG/host_calls.log 2>&1
    [ -s $LOG/host_compile.json ] || $PY $K/host_overhead.py compile $INT8/pi05.xml --out $LOG/host_compile.json > $LOG/host_compile.log 2>&1
    for v in 32 64 128; do
      $PY $EVAL run $INT8 --cache $CACHE --limit 100 --out $I/eval/preds/kernel/int8_dq\$v.npz \
         --config DYNAMIC_QUANTIZATION_GROUP_SIZE=\$v --config CACHE_DIR=$I/ovcache_kernel/dq\$v >> $LOG/eval_dq.log 2>&1
    done
    OV_GPU_DYNAMIC_QUANTIZATION_THRESHOLD=16 $PY $EVAL run $INT8 --cache $CACHE --limit 100 \
       --out $I/eval/preds/kernel/int8_dqthr16.npz --config CACHE_DIR=$I/ovcache_kernel/dq_thr16 >> $LOG/eval_dq.log 2>&1
  "
}

stage_B() {
  locked bash "$K/build_ov_debugcaps.sh" > "$LOG/build.log" 2>&1
  tail -5 "$LOG/build.log"
}

stage_C() {
  # shellcheck disable=SC1091
  source "$I/ovbuild/env.sh"
  cd "$I/physical-ai-studio/library"
  locked bash -c "
    set -x
    $DPY -c 'import openvino as ov; print(ov.__file__, ov.get_version())'
    $DPY $K/profile_kernels.py $INT8/pi05.xml --out $LOG/prof_int8_dbg_off.json > $LOG/prof_int8_dbg_off.log 2>&1
    OV_GPU_DISABLE_GATED_MLP_FUSION=0 $DPY $K/profile_kernels.py $INT8/pi05.xml --out $LOG/prof_int8_dbg_stock_on.json > $LOG/prof_int8_dbg_stock_on.log 2>&1
    OV_GPU_DISABLE_GATED_MLP_FUSION=0 OV_GPU_GATED_MLP_GELU=1 $DPY $K/profile_kernels.py $INT8/pi05.xml --out $LOG/prof_int8_dbg_gelu.json > $LOG/prof_int8_dbg_gelu.log 2>&1
    OV_GPU_DISABLE_GATED_MLP_FUSION=0 OV_GPU_GATED_MLP_GELU=1 $DPY $K/profile_kernels.py $FP16/pi05.xml --out $LOG/prof_fp16_dbg_gelu.json > $LOG/prof_fp16_dbg_gelu.log 2>&1
    $DPY $K/profile_kernels.py $FP16/pi05.xml --out $LOG/prof_fp16_dbg_off.json > $LOG/prof_fp16_dbg_off.log 2>&1
    SWEEP_CACHE_ROOT=$I/ovcache_kernel_dbg $DPY $K/sweep.py $INT8 --python $DPY --rounds 3 --iters 40 \
       --only default,gm_gelu,gm_stock,release --out $LOG/sweep_dbg_int8.jsonl > $LOG/sweep_dbg_int8.log 2>&1
    SWEEP_CACHE_ROOT=$I/ovcache_kernel_dbg_fp16 $DPY $K/sweep.py $FP16 --python $DPY --rounds 2 --iters 30 \
       --only default,gm_gelu,release --out $LOG/sweep_dbg_fp16.jsonl > $LOG/sweep_dbg_fp16.log 2>&1
    $DPY $EVAL run $INT8 --cache $CACHE --out $I/eval/preds/kernel/int8_dbg_off.npz \
       --config CACHE_DIR=$I/ovcache_kernel_dbg/eval_off >> $LOG/eval_dbg.log 2>&1
    OV_GPU_DISABLE_GATED_MLP_FUSION=0 OV_GPU_GATED_MLP_GELU=1 $DPY $EVAL run $INT8 --cache $CACHE \
       --out $I/eval/preds/kernel/int8_dbg_gelu.npz --config CACHE_DIR=$I/ovcache_kernel_dbg/eval_gelu >> $LOG/eval_dbg.log 2>&1
  "
}

for s in "$@"; do
  echo "$(date +%T) stage $s: waiting for an idle machine"
  wait_idle
  echo "$(date +%T) stage $s: start"
  "stage_$s"
  echo "$(date +%T) stage $s: done"
done
