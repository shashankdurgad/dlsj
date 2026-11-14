#!/usr/bin/env bash
# Sequential job queue for the π0.5 optimisation study on the shared XPS.
# Every step takes ~/inference/tmp/heavy.lock (one heavy job on the machine at a time) and exports also
# wait for >= 45 GB available RAM. Run detached:  nohup bash run_queue.sh <step>... > log 2>&1 &
set -uo pipefail
PY=~/inference/physical-ai-studio/library/.venv/bin/python
D=~/inference/deploy/pi05
E=~/inference/exports
V=~/inference/eval
CKPT=~/inference/models/pi05-junction-20261114/best.ckpt
LOCK=~/inference/tmp/heavy.lock
export TMPDIR=~/inference/tmp
cd ~/inference/physical-ai-studio/library

locked() { echo "== $(date +%T) $*"; flock "$LOCK" "$@"; local rc=$?; echo "== $(date +%T) exit $rc"; return $rc; }
wait_mem() { until [ "$(free -g | awk '/Mem:/{print $7}')" -ge 45 ]; do echo "waiting for RAM ($(free -g | awk '/Mem:/{print $7}') GB)"; sleep 60; done; }
export_ckpt() {  # name, extra args
  local name=$1; shift
  [ -f "$E/$name/pi05.xml" ] && { echo "skip $name (exists)"; return; }
  wait_mem
  locked nice -n 19 /usr/bin/time -v $PY $D/studio_export.py --ckpt "$CKPT" --out "$E/$name" "$@"
}
int8_from() { [ -f "$E/$2/pi05.xml" ] || locked nice -n 19 $PY $D/studio_export.py --compress_from "$E/$1" --out "$E/$2"; }
quant() {  # mode, out name, extra args (source: the FP16 10-step export)
  local mode=$1 name=$2; shift 2
  [ -f "$E/$name/pi05.xml" ] && { echo "skip $name (exists)"; return; }
  wait_mem
  locked nice -n 19 /usr/bin/time -v $PY $D/opt/quantize.py $mode $E/pi05_junction_fp16 --out "$E/$name" "$@"
}
quant_from() {  # mode, source name, out name, extra args
  local mode=$1 src=$2 name=$3; shift 3
  [ -f "$E/$name/pi05.xml" ] && { echo "skip $name (exists)"; return; }
  wait_mem
  locked nice -n 19 /usr/bin/time -v $PY $D/opt/quantize.py $mode "$E/$src" --out "$E/$name" "$@"
}
evalv() {  # export name, pred name, extra args
  locked nice -n 5 $PY $D/opt/eval_actions.py run "$E/$1" --cache $V/pi05_val300.npz --out "$V/preds/$2.npz" "${@:3}"
}

for step in "$@"; do
  case $step in
    ref)    evalv pi05_junction_fp16 fp16
            evalv pi05_junction_fp16 fp16_seed2 --seed_base 99991
            evalv pi05_junction_int8 int8 ;;
    calib)  locked nice -n 5 $PY $D/opt/eval_actions.py run $E/pi05_junction_fp16 --cache $V/pi05_train300.npz \
              --out $V/preds/train_fp16.npz --capture_ir $V/calib_ir_train.npz ;;
    tok64)  export_ckpt pi05_junction_fp16_tok64 --tokens 64
            int8_from pi05_junction_fp16_tok64 pi05_junction_int8_tok64
            evalv pi05_junction_int8_tok64 int8_tok64
            evalv pi05_junction_fp16_tok64 fp16_tok64 ;;
    s5)     export_ckpt pi05_junction_fp16_tok64_5step --tokens 64 --steps 5
            int8_from pi05_junction_fp16_tok64_5step pi05_junction_int8_tok64_5step
            evalv pi05_junction_int8_tok64_5step int8_tok64_5step ;;
    s3)     export_ckpt pi05_junction_fp16_tok64_3step --tokens 64 --steps 3
            int8_from pi05_junction_fp16_tok64_3step pi05_junction_int8_tok64_3step
            evalv pi05_junction_int8_tok64_3step int8_tok64_3step ;;
    w8a8)   # static W8A8 over the whole graph (statistics on CPU take ~15 s/sample, so default 64 samples)
            quant w8a8 pi05_junction_w8a8 --calib $V/calib_ir_train.npz --subset ${W8A8_SUBSET:-64}
            evalv pi05_junction_w8a8 w8a8 ;;
    w8a8_noexp)  # W8A8 on vision + VLM prefix only; the action expert keeps INT8 weight-only
            quant w8a8 pi05_junction_w8a8_noexp_tmp --calib $V/calib_ir_train.npz --subset ${W8A8_SUBSET:-256} --ignore expert
            ;&
    w8a8_noexp_finish)  # (falls through from w8a8_noexp)
            quant_from int8 pi05_junction_w8a8_noexp_tmp pi05_junction_w8a8_noexp --ignore vision --ignore vlm --ignore other
            evalv pi05_junction_w8a8_noexp w8a8_noexp ;;
    int4)   # INT4_SYM g128 on the Gemma VLM weights; SigLIP + action expert INT8_SYM (SigLIP's 4304-wide MLP
            # is not divisible by 128: nncf InvalidGroupSizeError when vision is included)
            quant int4 pi05_junction_int4vlm --ignore expert --ignore vision --int4_mode INT4_SYM --group 128
            evalv pi05_junction_int4vlm int4vlm ;;
    int4a)  quant int4 pi05_junction_int4vlm_asym64 --ignore expert --int4_mode INT4_ASYM --group 64
            evalv pi05_junction_int4vlm_asym64 int4vlm_asym64 ;;
    bench:*)  # bench:<export>,<export>,...  interleaved latency, 2 rounds x 40 iters
            IFS=, read -ra xs <<< "${step#bench:}"
            locked nice -n 5 $PY $D/opt/bench_variants.py "${xs[@]/#/$E/}" --rounds 2 --iters 40 \
              --json ~/inference/logs/pi05/opt/bench.jsonl ;;
    int8vlm)  # INT8_SYM weights on vision + VLM only, action expert stays FP16
            quant int8 pi05_junction_int8vlm --ignore expert
            evalv pi05_junction_int8vlm int8vlm ;;
    evalc:*) IFS=: read -r _ ex nm cfg <<< "$step"; evalv "$ex" "$nm" --config "$cfg" ;;  # evalc:<export>:<pred>:KEY=VALUE
    eval:*) IFS=: read -r _ ex nm <<< "$step"; evalv "$ex" "$nm" ;;   # eval:<export dir name>:<pred name>
    *)      echo "unknown step $step" ;;
  esac
done
echo "== queue done $(date +%T)"
