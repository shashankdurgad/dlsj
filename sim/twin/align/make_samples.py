"""Pick the frames used to fit/validate the cameras: 65 episodes spread over the dataset; 6 frames across each
episode (top / wrist) and the operator's inspection mark +-8 frames (inspect camera)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from frames import labels, states  # noqa: E402

L = labels()
eps = sorted(L)[::7]
top, insp = [], []
for e in eps:
    n = len(states(e))
    top += [[e, int(n * f)] for f in (0.08, 0.25, 0.4, 0.55, 0.7, 0.88)]
    m = L[e].get("inspection_frame")
    if m is not None:
        insp += [[e, int(min(max(m + d, 0), n - 1))] for d in (-8, 0, 8)]
out = Path(__file__).resolve().parents[1] / "out/align"
out.mkdir(parents=True, exist_ok=True)
(out / "samples.json").write_text(json.dumps(top + insp))
(out / "samples_split.json").write_text(json.dumps({"episodes": eps, "n_top": len(top), "n_inspect": len(insp)}))
print(len(eps), "episodes |", len(top), "top/wrist samples |", len(insp), "inspect samples")
