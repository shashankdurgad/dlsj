"""Real wrist / inspect frames during the hold (how deep the brick sits between the jaws)."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from sim.twin.align import frames as F  # noqa: E402

out = REPO / "sim/twin/out/physics/real_frames"
out.mkdir(parents=True, exist_ok=True)
lab = F.labels()
for ep in [int(x) for x in sys.argv[1:]] or [17, 76, 224]:
    s = F.states(ep)
    ins = lab[ep]["inspection_frame"]
    g = s[:, 5]
    close = next(f for f in range(1, ins) if g[f - 1] > 12 >= g[f])
    idx = [close, close + 8, (close + ins) // 2, ins]
    tiles = []
    for cam in ("wrist", "inspect", "top"):
        fr = F.video_frames(ep, cam, idx)
        tiles.append(np.concatenate([fr[i] for i in idx], 1))
    Image.fromarray(np.concatenate(tiles, 0)).resize((1280, 720)).save(out / f"ep{ep}.jpg", quality=85)
    print(ep, "close", close, "ins", ins, "grip", np.round(g[idx], 1).tolist())
