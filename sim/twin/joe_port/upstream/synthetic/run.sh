#!/usr/bin/env bash
# Reproduce the pilot on the existing Spark installation; no model training/upload.
set -euo pipefail
SYN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$(dirname -- "$SYN_DIR")"
SEND="$HOME/isaacsim/skills/isaac-sim-remote/scripts/isaacsim_send.py"
cd "$SYN_DIR"
exec 9> .run.lock
flock -n 9 || { echo 'Another synthetic run is active.' >&2; exit 1; }
if [[ -d raw || -d dataset ]]; then
    if [[ "${1:-}" != --new-run ]]; then
        echo 'Existing data is preserved. Use --new-run to archive it and generate a new pilot.' >&2
        exit 1
    fi
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    archive="runs/$stamp"
    mkdir -p "$archive"
    for item in raw dataset generation-progress.json generation-results.json summary.json dataset-sha256.json; do
        if [[ -e "$item" ]]; then mv "$item" "$archive/"; fi
    done
    for item in plans.json plans-slow.json batch-config.json provenance.json repeatability.json repeatability-slow.json repeatability-progress.json; do
        if [[ -e "$item" ]]; then cp -a "$item" "$archive/"; fi
    done
fi
mkdir -p "$TASK_DIR/logs"
PY=(docker exec -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e HF_DATASETS_CACHE=/work/hf-cache/datasets pi05-isaac-runtime /work/venv/bin/python)
read -r train_count validation_count candidates < <(python3 -c 'import json,math; x=json.load(open("batch-config.json")); print(x["train"],x["validation"],math.ceil(sum(x.values())*1.8))')
if [[ "${SYN_REUSE_PLANS:-0}" != 1 ]]; then
    plan_args=(--candidates "$candidates")
    if [[ "${SYN_WIDE:-0}" == 1 ]]; then plan_args+=(--wide); fi
    "${PY[@]}" /work/synthetic/make_trial.py
    "${PY[@]}" /work/synthetic/make_plans.py "${plan_args[@]}"
    "${PY[@]}" /work/synthetic/shorten_plans.py
fi
if [[ "${SYN_REUSE_STAGE:-0}" != 1 ]]; then
    for script in setup_scene.py configure_reference.py setup_physics.py repeatability.py; do
        python3 "$SEND" --context pi05 --timeout 600 --execution-timeout 0 --file "$SYN_DIR/$script"
    done
fi
python3 "$SEND" --context pi05 --timeout 120 --execution-timeout 0 --file "$SYN_DIR/provenance.py"
"${PY[@]}" /work/synthetic/export_dataset.py --wait --train "$train_count" --validation "$validation_count" > "$TASK_DIR/logs/export-batch.log" 2>&1 &
export_pid=$!
if ! python3 "$SEND" --context pi05 --timeout 43200 --execution-timeout 0 --file "$SYN_DIR/record_batch.py" > "$TASK_DIR/logs/generation-batch.log" 2>&1; then
    kill "$export_pid" 2>/dev/null || true
    wait "$export_pid" || true
    exit 1
fi
wait "$export_pid"
bash "$SYN_DIR/finish.sh" > "$TASK_DIR/logs/finalize-batch.log" 2>&1
