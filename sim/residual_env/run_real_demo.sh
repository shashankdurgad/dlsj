#!/bin/bash
# Residual-RL demo on top of the REAL MolmoAct2 (frozen), on Joe's Spark: policy worker (CUDA venv) + Isaac Lab trainer
# + video. One command; logs and outputs under ~/residual_env/out/<name>.
#
#   ./run_real_demo.sh --ckpt /path/to/lora.ckpt [--prompts "grasp|present|place"] [--envs 8] [--hours 2] [--name demo_molmo]
#   ./run_real_demo.sh --pretrained [--minutes 5] [--no_video]   # zero-shot allenai/MolmoAct2-SO100_101 (smoke test)
#
# --prompts must match the strings the checkpoint was trained on (e.g. "pick up the yellow brick|...|...").
set -euo pipefail
CKPT="" PRETRAINED=0 VIDEO=1 PROMPTS="" ENVS=8 MINUTES=120 NAME="" LAT=0.31 EVAL_MIN=""
while [ $# -gt 0 ]; do
  case "$1" in
    --ckpt) CKPT="$2"; shift 2;;
    --pretrained) PRETRAINED=1; shift;;
    --prompts) PROMPTS="$2"; shift 2;;
    --envs) ENVS="$2"; shift 2;;
    --hours) MINUTES=$(python3 -c "print(float('$2')*60)"); shift 2;;
    --minutes) MINUTES="$2"; shift 2;;
    --latency) LAT="$2"; shift 2;;
    --eval_every_min) EVAL_MIN="$2"; shift 2;;
    --name) NAME="$2"; shift 2;;
    --no_video) VIDEO=0; shift;;
    *) echo "unknown arg $1"; exit 2;;
  esac
done
[ -n "$CKPT" ] || [ "$PRETRAINED" = 1 ] || { echo "need --ckpt PATH or --pretrained"; exit 2; }
NAME=${NAME:-$([ -n "$CKPT" ] && echo demo_molmo_lora || echo demo_molmo_zeroshot)}
EVAL_MIN=${EVAL_MIN:-$(python3 -c "print(max(1.0, min(15.0, $MINUTES/8)))")}
# Paths default to the Spark layout; modal_demo.py overrides them for the Modal container.
HERE=$(cd "$(dirname "$0")" && pwd)
ISAAC_PY=${ISAAC_PY:-~/isaaclab-venv/bin/python}
MOLMO_PY=${MOLMO_PY:-~/molmo-cuda-venv/bin/python}
OUT_ROOT=${OUT_ROOT:-~/residual_env/out}
export HF_HUB_OFFLINE=1
[ -n "${HF_HOME:-}" ] || export HF_HUB_CACHE=${HF_HUB_CACHE:-/home/yan/molmobench/hf}
PRELOAD=""
[ -f /lib/aarch64-linux-gnu/libgomp.so.1 ] && PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1
cd "$HERE"
OUT=$OUT_ROOT/$NAME IPC=${IPC_ROOT:-$OUT_ROOT}/ipc_$NAME
mkdir -p "$OUT" "$IPC"
export OMNI_KIT_ACCEPT_EULA=YES

SRC=$([ -n "$CKPT" ] && echo "--ckpt $CKPT" || echo "--pretrained allenai/MolmoAct2-SO100_101")
nohup "$MOLMO_PY" molmo_batch_worker.py \
  --ipc "$IPC" $SRC > "$OUT/worker.log" 2>&1 &
WPID=$!
echo "$WPID" > "$OUT/worker.pid"
trap 'touch "$IPC/done"; sleep 2; kill $WPID 2>/dev/null || true' EXIT
echo "worker pid $WPID ($SRC); waiting for ready"
for _ in $(seq 600); do [ -f "$IPC/ready" ] && break; kill -0 $WPID 2>/dev/null || { tail -20 "$OUT/worker.log"; exit 1; }; sleep 1; done
cat "$IPC/ready"; echo

PARGS=()
[ -n "$PROMPTS" ] && PARGS=(--prompts "$PROMPTS")
LD_PRELOAD=$PRELOAD "$ISAAC_PY" demo_train.py --base molmo --num_envs "$ENVS" \
  --minutes "$MINUTES" --eval_every_min "$EVAL_MIN" --warmup_steps 1000 --latency "$LAT" --ipc "$IPC" --out "$OUT" \
  "${PARGS[@]}" > "$OUT/train.log" 2>&1
grep -E "^EVAL|^DEMO_DONE" "$OUT/train.log" || true

if [ "$VIDEO" = 0 ]; then echo "REAL_DEMO_DONE $OUT (no video)"; exit 0; fi
# the trainer wrote $IPC/done; restart a fresh worker session for the video
kill $WPID 2>/dev/null || true; wait $WPID 2>/dev/null || true
rm -f "$IPC/done" "$IPC/ready"
nohup "$MOLMO_PY" molmo_batch_worker.py \
  --ipc "$IPC" $SRC > "$OUT/worker_video.log" 2>&1 &
WPID=$!
for _ in $(seq 600); do [ -f "$IPC/ready" ] && break; sleep 1; done
LABEL=$([ -n "$CKPT" ] && echo "MolmoAct2 + our LoRA fine-tune, frozen" || echo "MolmoAct2-SO100_101 zero-shot, frozen")
LD_PRELOAD=$PRELOAD "$ISAAC_PY" demo_video.py --run "$OUT" --base molmo \
  --ipc "$IPC" --base_label "$LABEL" --out "$OUT/$NAME.mp4" > "$OUT/video.log" 2>&1
grep -E "^VIDEO_DONE" "$OUT/video.log" || { tail -30 "$OUT/video.log"; exit 1; }
echo "REAL_DEMO_DONE $OUT"
