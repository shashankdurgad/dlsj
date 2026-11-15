"""Extra routes for loop_control.py: laptop utilisation, the demo slide and the 3D-arm assets.

Copied next to loop_control.py with sys_stats.py (dlsj/sys_stats.py) and brief_page.py (python -m dlsj.brief_slide).
loop_control imports this inside a try, so a problem here never stops the loop.

  GET /sys/stats                     SysStats snapshot of this process and the laptop (CPU, GPU, NPU, memory)
  GET /brief                         the fixed demo slide
  GET /demo_arm.js, /api/arm, /vendor/*, /so101/*   assets for the slide's 3D arm (from the dlsj board checkout)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from sys_stats import SysStats

STATS = SysStats(period=0.5)
STATS.start()
BOARD = Path(os.environ.get("DLSJ_BOARD_DIR", str(Path.home() / "desmond/dlsj/dlsj")))
MIME = {".js": "text/javascript; charset=utf-8", ".json": "application/json", ".stl": "model/stl"}


def _brief() -> bytes | None:
    try:
        from brief_page import BRIEF
    except Exception:  # noqa: BLE001
        return None
    return BRIEF.encode()


def _inside(root: Path, name: str) -> Path | None:
    if not name or "\\" in name or name.startswith("/"):
        return None
    path = (root / name).resolve()
    return path if root.resolve() in path.parents and path.is_file() else None


def _asset(path: str) -> Path | None:
    if path == "/demo_arm.js":
        return _inside(BOARD, "demo_arm.js")
    if path == "/api/arm":
        return _inside(BOARD, "demo_traces.json")
    for prefix, sub in (("/vendor/", "vendor"), ("/so101/", "so101_assets")):
        if path.startswith(prefix):
            return _inside(BOARD / sub, path.removeprefix(prefix))
    return None


def handle(h) -> bool:
    """Serve the request on handler `h` if it is one of ours. True when handled."""
    try:
        path = h.path.split("?", 1)[0]
        if path == "/sys/stats":
            h._send(200, json.dumps(STATS.snapshot()).encode(), "application/json")
            return True
        if path == "/brief":
            body = _brief()
            if body is None:
                h._send(404, b"brief_page.py missing", "text/plain")
            else:
                h._send(200, body, "text/html; charset=utf-8")
            return True
        asset = _asset(path)
        if asset is not None:
            h._send(200, asset.read_bytes(), MIME.get(asset.suffix, "application/octet-stream"))
            return True
    except (BrokenPipeError, ConnectionResetError):
        return True
    return False
