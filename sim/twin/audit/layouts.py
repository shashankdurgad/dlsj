"""Scene layouts of the audit episodes from their first top frame, with the scene agent's CURRENT detector.

    python sim/twin/audit/layouts.py --run sim/twin/out/audit/runs/<stamp>     # plain python (PyAV)

sim/twin/scene/layout.layout_from_episode(ep, cache=False) per episode (re-detected every audit, so a detector change
shows up); if it fails for an episode the layout of the first successful audit (set/layouts_first.json) is used and
flagged. Writes <run>/layouts.json = {"layouts": {ep: layout}, "source": {ep: "detected" | "frozen" | "none"}}.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from sim.twin.audit.common import DATA, SET, jdump, load_set


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    run = Path(ap.parse_args().run)
    eps = load_set()["episodes"]
    frozen_p = SET / "layouts_first.json"
    frozen = json.loads(frozen_p.read_text()) if frozen_p.exists() else {}
    lays, src, errs = {}, {}, {}
    try:
        from sim.twin.scene.layout import layout_from_episode
    except Exception:  # noqa: BLE001
        layout_from_episode = None
        errs["import"] = traceback.format_exc(limit=3)
    for ep in eps:
        lay = None
        if layout_from_episode is not None:
            try:
                lay = layout_from_episode(ep, DATA, cache=False)
                src[str(ep)] = "detected"
            except Exception:  # noqa: BLE001
                errs[str(ep)] = traceback.format_exc(limit=3)
        if lay is None and str(ep) in frozen:
            lay, src[str(ep)] = frozen[str(ep)], "frozen"
        if lay is None:
            src[str(ep)] = "none"
            continue
        lays[str(ep)] = lay
        t = lay.get("target") or {}
        print(f"ep {ep:3d} {src[str(ep)]}: plates {sorted(lay.get('plates', {}))} target {t.get('color')} "
              f"{t.get('kind')} distractors {len(lay.get('distractors', []))}", flush=True)
    if not frozen and len(lays) == len(eps):
        jdump(frozen_p, lays)
    jdump(run / "layouts.json", dict(layouts=lays, source=src, errors=errs))
    print("LAYOUTS_OK", len(lays), "of", len(eps), "| errors", len(errs))


if __name__ == "__main__":
    main()
