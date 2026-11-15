"""Re-score gate runs from their saved traces with the current metrics.py (no Isaac; numpy only).

    python sim/twin/grasp_gate/reanalyse.py <tag> [<tag> ...]     -> rewrites out/grasp_gate/<tag>/result.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from sim.twin.grasp_gate.common import OUT, dump  # noqa: E402
from sim.twin.grasp_gate.metrics import analyse, summarise  # noqa: E402

for tag in sys.argv[1:]:
    old = json.loads((OUT / tag / "result.json").read_text())["summary"]
    tr = dict(np.load(OUT / tag / "trace.npz"))
    per = analyse(tr)
    res = summarise(per)
    res.update({k: v for k, v in old.items() if k not in res})
    dump(OUT / tag / "result.json", {"summary": res, "episodes": per})
    print(tag, res["success"], res["failures"], flush=True)
