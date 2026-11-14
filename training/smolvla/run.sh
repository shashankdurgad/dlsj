#!/usr/bin/env bash
# SmolVLA fine-tune on a 1-GPU GCP VM (official LeRobot recipe, wrist + top cameras). Idempotent: safe to rerun
# after a Spot preemption; finished stages are skipped and training resumes from the newest checkpoint.
#
#   RUN_NAME=smolvla-517-2cam-20261114 STAGE=gate bash run.sh   # setup, data, smoke, 200-step timing + gate ckpt
#   RUN_NAME=smolvla-517-2cam-20261114 STAGE=full bash run.sh   # the 20k-step run (resumes if interrupted)
#
# Optional env: RUN_ARTIFACTS (default gs://onflow-30k-pi05-artifacts/runs/$RUN_NAME), WORK_DIR (/opt/$RUN_NAME),
# MAX_UPTIME_S (VM uptime budget, default 21600 = 6 h), AUTO_SHUTDOWN=1 (power off when done; set by launch_vm.sh).
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$here/versions.env"
: "${RUN_NAME:?set RUN_NAME}"
STAGE="${STAGE:-gate}"
[[ "$STAGE" == gate || "$STAGE" == full ]] || { echo "STAGE must be gate or full" >&2; exit 2; }
export RUN_ARTIFACTS="${RUN_ARTIFACTS:-gs://onflow-30k-pi05-artifacts/runs/$RUN_NAME}"
work="${WORK_DIR:-/opt/$RUN_NAME}"
MAX_UPTIME_S="${MAX_UPTIME_S:-21600}"
deadline=$(python3 -c "import time; print(int(time.time() + $MAX_UPTIME_S - float(open('/proc/uptime').read().split()[0])))")
mkdir -p "$work" && cd "$work"
exec > >(tee -a "$work/run-$STAGE.log") 2>&1

sync_checkpoints() {  # upload completed checkpoint dirs (<= the `last` symlink target) once each
  local run="$1" dest="$2" last step d
  [[ -L "$run/checkpoints/last" ]] || return 0
  last=$(basename "$(readlink "$run/checkpoints/last")")
  for d in "$run"/checkpoints/[0-9]*; do
    step=$(basename "$d")
    [[ -e "$d/.uploaded" || 10#$step -gt 10#$last ]] && continue
    gcloud storage rsync --recursive "$d" "$dest/checkpoints/$step" && touch "$d/.uploaded"
  done
}
finish() {
  local code=$?
  trap - EXIT
  [[ "$STAGE" == full && -d runs/full ]] && sync_checkpoints runs/full "$RUN_ARTIFACTS/full" || true
  printf '{"stage":"%s","exit_code":%s,"finished_at":"%s"}\n' "$STAGE" "$code" "$(date -u +%FT%TZ)" > "status-$STAGE.json"
  gcloud storage cp "run-$STAGE.log" "status-$STAGE.json" "$RUN_ARTIFACTS/" || true
  sync
  [[ "${AUTO_SHUTDOWN:-0}" == 1 ]] && shutdown -h now
  exit "$code"
}
trap finish EXIT
echo "STAGE_STARTED $STAGE $(date -u +%FT%TZ) artifacts=$RUN_ARTIFACTS deadline=$(date -u -d @"$deadline" +%FT%TZ)"

# --- environment (LeRobot, pinned) ------------------------------------------------------------------------
for _ in $(seq 1 36); do nvidia-smi -L && break; sleep 5; done
command -v ffmpeg >/dev/null || { apt-get update -q && apt-get install -yq ffmpeg libgl1; }
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; }
export PATH="$HOME/.local/bin:$PATH"
if [[ ! -x venv/bin/python ]]; then
  uv venv --python 3.12 venv
  VIRTUAL_ENV="$work/venv" uv pip install "lerobot[smolvla,training]==$LEROBOT_VERSION"
fi
py="$work/venv/bin/python"
"$py" -c "import torch; assert torch.cuda.is_available(); print('CUDA', torch.__version__, torch.cuda.get_device_name(0))"

# --- pinned base model (hub cache mirrored by cache_base.sh) ----------------------------------------------
export HF_HOME="$work/hf-home" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
if [[ ! -d "$HF_HOME/hub" ]]; then
  mkdir -p "$HF_HOME" base
  gcloud storage cp "$BASE_CACHE_URI/hf-hub-cache.tar" "$BASE_CACHE_URI/hf-hub-cache.tar.sha256" base/
  (cd base && sha256sum -c hf-hub-cache.tar.sha256)
  tar -C "$HF_HOME" -xf base/hf-hub-cache.tar && rm base/hf-hub-cache.tar
fi

# --- data: curated snapshot -> wrist+top train/validation -------------------------------------------------
if [[ ! -f data/prepared/manifest.json ]]; then
  rm -rf data/prepared && mkdir -p data/src
  gcloud storage rsync --recursive "$DATASET_URI/" data/src/
  (cd data/src && sha256sum --quiet -c SHA256SUMS.txt)
  "$py" "$here/prepare_data.py" data/src data/prepared
  gcloud storage cp data/prepared/manifest.json "$RUN_ARTIFACTS/data-manifest.json"
fi
train_data="$work/data/prepared/train"

if [[ "$STAGE" == gate ]]; then
  [[ -d runs/smoke ]] || "$py" "$here/train.py" "$train_data" --output runs/smoke --smoke
  if [[ ! -f runs/timing.json ]]; then
    rm -rf runs/timing
    "$py" "$here/train.py" "$train_data" --output runs/timing --steps 200 --save-freq 200 2>&1 | tee runs/timing.log
    "$py" - <<'PY'
import json, re, statistics
# LeRobot logs "step:50 ... step_s:0.512 ..." every log_freq (50) steps: 4 lines over 200 steps.
steps = [float(x) for x in re.findall(r"\bstep_s:([0-9.]+)", open("runs/timing.log").read())]
if len(steps) < 2:
    raise SystemExit("too few step_s lines in runs/timing.log")
s = statistics.median(steps[1:])  # drop the first (warm-up) interval
report = {"median_step_s": s, "log_lines": len(steps),
          "projected_20k_hours": round(20000 * s / 3600, 2), "note": "training only; add ~10% for checkpoints"}
json.dump(report, open("runs/timing.json", "w"), indent=2)
print("TIMING", json.dumps(report))
PY
  fi
  gcloud storage rsync --recursive runs/timing/checkpoints/000200/pretrained_model "$RUN_ARTIFACTS/gate/000200/pretrained_model"
  gcloud storage cp runs/timing.json "$RUN_ARTIFACTS/gate/timing.json"
  echo "GATE_READY $RUN_ARTIFACTS/gate/000200/pretrained_model : check Studio import / OpenVINO export before STAGE=full"
  exit 0
fi

# --- STAGE=full: 20k steps, resume-safe ---------------------------------------------------------------------
run=runs/full
if [[ ! -L "$run/checkpoints/last" ]] && gcloud storage ls "$RUN_ARTIFACTS/full/checkpoints/" >/dev/null 2>&1; then
  newest=$(gcloud storage ls "$RUN_ARTIFACTS/full/checkpoints/" | sed -E 's#.*/([0-9]+)/$#\1#' | grep -E '^[0-9]+$' | sort -n | tail -1)
  if [[ -n "$newest" ]]; then
    mkdir -p "$run/checkpoints"
    gcloud storage rsync --recursive "$RUN_ARTIFACTS/full/checkpoints/$newest" "$run/checkpoints/$newest"
    touch "$run/checkpoints/$newest/.uploaded"
    ln -sfn "$newest" "$run/checkpoints/last"
    echo "RESTORED checkpoint $newest from GCS"
  fi
fi
resume=(); [[ -L "$run/checkpoints/last" ]] && resume=(--resume)
( while sleep 600; do sync_checkpoints "$run" "$RUN_ARTIFACTS/full" || true; done ) &
syncer=$!
remaining=$(( deadline - $(date +%s) - 600 ))
(( remaining > 600 )) || { echo "not enough VM time left ($remaining s)"; exit 1; }
set +e
timeout --signal=TERM --kill-after=60s "${remaining}s" "$py" "$here/train.py" "$train_data" --output "$run" "${resume[@]}"
code=$?
set -e
kill "$syncer" 2>/dev/null || true
sync_checkpoints "$run" "$RUN_ARTIFACTS/full"
echo "TRAINING_FINISHED exit=$code $(date -u +%FT%TZ)"
exit "$code"
