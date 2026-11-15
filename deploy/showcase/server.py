"""Showcase dashboard server (port 8030): the live 3D twin + mission control + the presentation, beside the robot.

It only talks to the loop controller over HTTP (default http://localhost:8020), so a crash here never touches the robot:
  GET  /api/state           -> controller /ctl/state (or {"offline": true, "error": ...})
  GET  /live/<cam>?w=640    -> controller /live/<cam> MJPEG, relayed chunk by chunk (latest-frame streams)
  POST /api/loop/<cmd>      -> controller /loop/<cmd>  (start | pause | resume | stop | home)
  GET  /api/story           -> story.json (the presentation: every number measured, caveats included)
  GET  /api/replay          -> replay.json (a recorded run for the offline demo), 404 if absent
  GET  /<file>              -> static/

    python3 deploy/showcase/server.py [--robot http://localhost:8020] [--port 8030]
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
ROBOT = "http://localhost:8020"
LOOP_CMDS = {"start", "pause", "resume", "stop", "home"}
CAMS = {"top", "inspect", "wrist"}
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("model/stl", ".stl")
mimetypes.add_type("application/xml", ".urdf")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code: int, body: bytes, ctype: str, cache: bool = False) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/state":
            try:
                with urllib.request.urlopen(ROBOT + "/ctl/state", timeout=1.5) as r:
                    return self._send(200, r.read(), "application/json")
            except (urllib.error.URLError, OSError, ValueError) as e:
                return self._json({"offline": True, "error": f"robot controller not reachable at {ROBOT}: {e}"})
        if path in ("/api/story", "/api/replay"):
            f = HERE / (path.removeprefix("/api/") + ".json")
            return self._send(200, f.read_bytes(), "application/json") if f.exists() else self._send(404, b"", "text/plain")
        if path.startswith("/live/") and path.removeprefix("/live/") in CAMS:
            return self._relay_stream()
        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        f = (STATIC / rel).resolve()
        if STATIC in f.parents or f == STATIC / "index.html":
            if f.is_file():
                ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
                return self._send(200, f.read_bytes(), ctype, cache="/vendor/" in path or "/robot/" in path)
        return self._send(404, b"not found", "text/plain")

    def _relay_stream(self):
        try:
            up = urllib.request.urlopen(ROBOT + self.path, timeout=5)
        except (urllib.error.URLError, OSError) as e:
            return self._send(503, f"camera not available: {e}".encode(), "text/plain")
        self.send_response(200)
        self.send_header("Content-Type", up.headers.get("Content-Type", "multipart/x-mixed-replace; boundary=frame"))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            while True:
                chunk = up.read1(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            up.close()
            self.close_connection = True

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path.startswith("/api/loop/") and path.removeprefix("/api/loop/") in LOOP_CMDS:
            req = urllib.request.Request(ROBOT + "/loop/" + path.removeprefix("/api/loop/"), data=b"", method="POST")
            try:
                with urllib.request.urlopen(req, timeout=3) as r:
                    return self._send(r.status, r.read() or b"{}", "application/json")
            except (urllib.error.URLError, OSError) as e:
                return self._json({"ok": False, "error": str(e)}, 502)
        return self._send(404, b"", "text/plain")


def main() -> None:
    global ROBOT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", default=ROBOT, help="loop controller base URL")
    ap.add_argument("--port", type=int, default=8030)
    ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args()
    ROBOT = a.robot.rstrip("/")
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    srv.daemon_threads = True
    print(f"showcase: http://localhost:{a.port}  (robot controller {ROBOT})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
