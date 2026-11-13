#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/dlsj"
python3 - <<'PY'
import json
from pathlib import Path
p=Path('/tmp/pi05-smoke-run/training-config.json')
c=json.loads(p.read_text())
c['trainer'].update(fast_dev_run=False, max_steps=30, max_epochs=100,
    num_sanity_val_steps=1, limit_val_batches=1, check_val_every_n_epoch=1000,
    enable_checkpointing=False, logger=False, default_root_dir='/tmp/pi05-benchmark-run',
    callbacks=[{'class_path':'pi05_benchmark_callback.StepTiming'}])
Path('/tmp/pi05-benchmark-config.json').write_text(json.dumps(c,indent=2)+'\n')
PY
export PYTHONPATH="/tmp${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8
training/pi05/.studio/library/.venv/bin/physicalai fit --config /tmp/pi05-benchmark-config.json > /tmp/pi05-benchmark.log 2>&1
cat /tmp/pi05-benchmark-result.json
