#!/usr/bin/env bash
set -euo pipefail
SYN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$(dirname -- "$SYN_DIR")"
deadline=$((SECONDS + 7200))
until [[ -s "$SYN_DIR/dataset/manifest.json" ]]; do
    if (( SECONDS >= deadline )); then echo 'Dataset export did not finish before the deadline.' >&2; exit 1; fi
    sleep 5
done
PY=(docker exec -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e HF_DATASETS_CACHE=/work/hf-cache/datasets pi05-isaac-runtime /work/venv/bin/python)
docker exec pi05-isaac-runtime sh -c "chown -R $(id -u):$(id -g) /work/synthetic/dataset && chmod -R a+rX /work/synthetic/dataset"
"${PY[@]}" /work/synthetic/validate_dataset.py /work/synthetic/dataset
"${PY[@]}" /work/synthetic/training_smoke.py /work/synthetic/dataset
docker exec pi05-isaac-runtime sh -c "chown -R $(id -u):$(id -g) /work/synthetic/dataset && chmod -R a+rX /work/synthetic/dataset"
python3 - "$SYN_DIR" <<'PY'
import json,hashlib,sys
from pathlib import Path
root=Path(sys.argv[1]);manifest=json.loads((root/'dataset/manifest.json').read_text());generation=json.loads((root/'generation-results.json').read_text())
accepted=[r for r in generation['episodes'] if r['success']]
frames=sum(r['frames'] for r in accepted)
summary={'status':'validated','episodes':len(accepted),'camera_clips':3*len(accepted),'frames_per_camera':frames,'fps':30,'unique_task_seconds':frames/30,'all_camera_seconds':3*frames/30,'split_counts':manifest['counts'],'attempts':generation['attempts'],'minimum_hold_lift_m':min(r['minimum_hold_lift_m'] for r in accepted),'maximum_hold_lift_m':max(r['minimum_hold_lift_m'] for r in accepted),'minimum_release_drop_m':min(r['release_drop_m'] for r in accepted),'native_validation':json.loads((root/'dataset/validation.json').read_text())['passed'],'training_adapter_validation':json.loads((root/'dataset/training-smoke.json').read_text())['passed']}
(root/'summary.json').write_text(json.dumps(summary,indent=2))
files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'dataset').rglob('*') if p.is_file()}
(root/'dataset-sha256.json').write_text(json.dumps(files,indent=2));print('PILOT_VALIDATED',json.dumps(summary),flush=True)
PY
