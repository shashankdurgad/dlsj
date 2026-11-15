"""Station display: a wordless, black-and-white status page around the live Isaac twin.

Runs on the XPS next to the robot controller. It polls the running console (smolvla_control / auto_control /
loop_control, all expose GET /ctl/state) at ~30 Hz and serves:

  /             the page (index.html)
  /api/state    the latest console snapshot, trimmed to what the page draws, plus link health
  /api/pose     {"t", "q": [pan, lift, elbow, wrist_flex, wrist_roll, gripper], "frame": "cal2", "age"} for the
                Isaac live-mirror renderer on watson (LeRobot degrees, current follower calibration, gripper 0-100)
  /api/config   where the page finds the twin stream and the camera streams

    python deploy/frontend/server.py                      # console on http://localhost:8020, twin on watson
    python deploy/frontend/server.py --mock               # synthetic robot, no hardware (design / demo check)

Stdlib only, so it runs in any venv on the XPS.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
KEEP_LOOP = ("mode", "step", "cycle", "attempts", "tallies", "cycles", "step_s", "target", "grasp")


def trim(snap: dict) -> dict:
    """The fields the page draws (the console snapshot also carries config, events and collection state)."""
    out = {k: snap.get(k) for k in ("phase", "stage", "bowl", "bowls", "brick", "latency_ms", "latency_median_ms",
                                    "chunk_age_s", "chunks", "inflight", "measured", "cmd", "blend")}
    loop = snap.get("loop") or {}
    out["loop"] = {k: loop.get(k) for k in KEEP_LOOP} if loop else None
    perc = snap.get("perception") or {}
    out["perception"] = {k: perc.get(k) for k in ("table", "sorted", "plates", "target", "inspect", "workspace",
                                                   "age_s")} if perc else None
    studio = snap.get("studio") or {}
    out["policy"] = {k: studio.get(k) for k in ("name", "policy", "model_id")} if studio else None
    out["inspection"] = snap.get("inspection")
    return out


class Source:
    """Background poller of the console's /ctl/state (or a synthetic robot with --mock)."""

    def __init__(self, url: str, hz: float, mock: bool):
        self.url, self.period, self.mock = url, 1.0 / hz, mock
        self.lock = threading.Lock()
        self.snap: dict = {}
        self.t_ok = 0.0
        self.policy_hint = None
        threading.Thread(target=self._run, daemon=True, name="console-poll").start()

    def _run(self) -> None:
        nxt = time.perf_counter()
        while True:
            try:
                snap = MockRobot.step() if self.mock else json.loads(
                    urllib.request.urlopen(self.url, timeout=0.5).read())
                with self.lock:
                    self.snap, self.t_ok = snap, time.time()
            except Exception:  # noqa: BLE001  console not running: keep the last snapshot, the page shows "offline"
                pass
            nxt += self.period
            time.sleep(max(0.0, nxt - time.perf_counter()))
            nxt = max(nxt, time.perf_counter() - self.period)

    def state(self) -> dict:
        with self.lock:
            snap, t_ok = dict(self.snap), self.t_ok
        out = trim(snap) if snap else {}
        out["link"] = {"age_s": None if not t_ok else round(time.time() - t_ok, 2)}
        if out.get("policy") is None and self.policy_hint:
            out["policy"] = self.policy_hint
        return out

    def pose(self) -> dict:
        with self.lock:
            q, t_ok = self.snap.get("measured"), self.t_ok
        return {"t": t_ok, "q": q, "frame": "cal2", "age": None if not t_ok else round(time.time() - t_ok, 3)}


class MockRobot:
    """A synthetic pick -> bring -> place loop with detections, for checking the page without hardware."""

    t0 = time.time()
    colours = ("blue", "brown", "red", "white", "yellow")
    bowls = ("yellow", "purple", "orange")
    cycles: list = []
    tallies: dict = {}
    bricks = [{"box": [80 + 70 * i, 140 + (i % 2) * 90, 120 + 70 * i, 180 + (i % 2) * 90], "conf": 0.9,
               "colour": c, "colour_conf": 0.9, "plate": None} for i, c in enumerate(("blue", "red", "white", "yellow",
                                                                                         "brown", "red"))]

    @classmethod
    def step(cls) -> dict:
        t = time.time() - cls.t0
        period = 12.0
        n, ph = int(t // period), (t % period) / period
        stage = "pick" if ph < 0.45 else ("bring" if ph < 0.7 else "place")
        step = {"pick": "pick", "bring": "bring" if ph < 0.6 else "inspect", "place": "place"}[stage]
        brick = cls.colours[n % 5]
        bowl = cls.bowls[(n * 7) % 3]
        while len(cls.cycles) < n:
            k = len(cls.cycles)
            ok = k % 6 != 4
            cls.cycles.append({"n": k + 1, "brick": cls.colours[k % 5], "result": "sorted" if ok else "pick timeout",
                               "seconds": round(period + random.uniform(-1, 1), 1)})
            if ok:
                b = cls.bowls[(k * 7) % 3]
                cls.tallies[b] = cls.tallies.get(b, 0) + 1
        cur = cls.cycles[-11:] + [{"n": n + 1, "brick": brick, "result": "running"}]
        w = 2 * math.pi * ph
        q = [25 * math.sin(w), -50 + 60 * math.sin(w / 2) ** 2, 30 - 50 * math.sin(w / 2) ** 2,
             70 + 10 * math.sin(2 * w), 15 * math.sin(3 * w), 35 if 0.2 < ph < 0.35 or ph > 0.85 else 6]
        cmd = [v + random.uniform(-1.5, 1.5) for v in q]
        table = [b for i, b in enumerate(cls.bricks) if i != n % len(cls.bricks)]
        return {"phase": "running", "stage": stage, "bowl": bowl, "bowls": list(cls.bowls), "brick": brick,
                "latency_ms": round(300 + 25 * math.sin(t * 1.7) + random.uniform(-15, 15), 1),
                "latency_median_ms": 312.0, "chunk_age_s": round((t % 0.33), 2), "chunks": 2, "inflight": True,
                "measured": [round(v, 2) for v in q], "cmd": [round(v, 2) for v in cmd], "blend": "rtc(h=15, g=3.0)",
                "loop": {"mode": "running", "step": step, "cycle": n + 1, "attempts": 0, "tallies": dict(cls.tallies),
                         "cycles": cur, "step_s": round(ph * period, 1), "target": table[0] if table else None,
                         "grasp": {"state": "holding" if stage != "pick" else "open"}},
                "perception": {"table": table, "sorted": [], "workspace": [0, 60, 560, 480],
                               "plates": [{"colour": "yellow", "cx": 480, "cy": 120, "r": 55},
                                          {"colour": "purple", "cx": 500, "cy": 300, "r": 55},
                                          {"colour": "orange", "cx": 470, "cy": 420, "r": 50}],
                               "target": table[0] if table else None,
                               "inspect": {"held": [250, 170, 390, 300], "close": step == "inspect"} if stage == "bring"
                               else None, "age_s": 0.2},
                "studio": {"name": "molmoact2-cal2-456-8k", "policy": "molmoact2", "model_id": "mock"},
                "inspection": {"state": "ok", "outcome": "good" if n % 4 else "defect", "confidence": 0.93}
                if step == "inspect" else None}


def make_handler(source: Source, config: dict):

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                return self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
            if path == "/api/state":
                return self._send(200, json.dumps(source.state()).encode(), "application/json")
            if path == "/api/pose":
                return self._send(200, json.dumps(source.pose()).encode(), "application/json")
            if path == "/api/config":
                return self._send(200, json.dumps(config).encode(), "application/json")
            return self._send(404, b"", "text/plain")

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--console", default="http://localhost:8020", help="the robot console (serves /ctl/state)")
    ap.add_argument("--twin", default="http://watson:8030/stream.mjpg", help="Isaac live-mirror MJPEG stream")
    ap.add_argument("--cams", default="", help="base URL of the camera streams (/stream/<cam>); default the console")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8040)
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("--mock", action="store_true", help="synthetic robot, no hardware")
    a = ap.parse_args()
    source = Source(a.console.rstrip("/") + "/ctl/state", a.hz, a.mock)
    config = {"twin": a.twin, "cams": "" if a.mock else (a.cams or a.console).rstrip("/"), "mock": a.mock}
    server = ThreadingHTTPServer((a.host, a.port), make_handler(source, config))
    print(f"station display on http://{a.host}:{a.port}  (console {a.console}{', MOCK' if a.mock else ''}; "
          f"twin {a.twin})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
