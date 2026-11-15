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
KEEP_LOOP = ("mode", "step", "cycle", "attempts", "tallies", "cycles", "step_s", "target", "grasp", "started")


def trim(snap: dict) -> dict:
    """The fields the page draws (the console snapshot also carries config, events and collection state)."""
    out = {k: snap.get(k) for k in ("phase", "stage", "prompt", "bowl", "bowls", "brick", "latency_ms", "latency_median_ms",
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


class Silicon:
    """Live CPU / GPU / NPU utilisation (0-1) on the XPS from counters readable without root:
    /proc/stat, the xe driver's GT idle residency, and the intel_vpu NPU busy time. Missing counters read None."""

    GPU_IDLE = "/sys/class/drm/card*/device/tile0/gt0/gtidle/idle_residency_ms"
    NPU_BUSY = "/sys/class/accel/accel*/device/npu_busy_time_us"

    def __init__(self, hz: float = 4.0, keep: int = 60, mock: bool = False):
        import glob

        self.mock, self.period = mock, 1.0 / hz
        self.gpu_f = next(iter(sorted(glob.glob(self.GPU_IDLE))), None)
        self.npu_f = next(iter(sorted(glob.glob(self.NPU_BUSY))), None)
        self.hist = {k: [] for k in ("cpu", "gpu", "npu")}
        self.keep, self.lock = keep, threading.Lock()
        threading.Thread(target=self._run, daemon=True, name="silicon").start()

    @staticmethod
    def _read(path):
        try:
            return float(Path(path).read_text().split()[0])
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _cpu():
        try:
            v = [float(x) for x in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
            return sum(v), v[3] + v[4]  # total, idle + iowait
        except Exception:  # noqa: BLE001
            return None

    def _run(self) -> None:
        prev = (time.monotonic(), self._cpu(), self._read(self.gpu_f) if self.gpu_f else None,
                self._read(self.npu_f) if self.npu_f else None)
        while True:
            time.sleep(self.period)
            now = (time.monotonic(), self._cpu(), self._read(self.gpu_f) if self.gpu_f else None,
                   self._read(self.npu_f) if self.npu_f else None)
            dt = now[0] - prev[0]
            if self.mock:
                t = time.time()
                u = {"cpu": 0.25 + 0.1 * math.sin(t), "gpu": 0.55 + 0.4 * math.sin(t * 2.1) ** 2,
                     "npu": 0.1 + 0.6 * max(0.0, math.sin(t * 0.7)) ** 4}
            else:
                u = {"cpu": None, "gpu": None, "npu": None}
                if now[1] and prev[1] and now[1][0] > prev[1][0]:
                    u["cpu"] = 1 - (now[1][1] - prev[1][1]) / (now[1][0] - prev[1][0])
                if now[2] is not None and prev[2] is not None:
                    u["gpu"] = 1 - (now[2] - prev[2]) / (dt * 1000)
                if now[3] is not None and prev[3] is not None:
                    u["npu"] = (now[3] - prev[3]) / (dt * 1e6)
            with self.lock:
                for k, v in u.items():
                    self.hist[k] = (self.hist[k] + [None if v is None else round(max(0.0, min(1.0, v)), 3)])[-self.keep:]
            prev = now

    def state(self) -> dict:
        with self.lock:
            return {k: list(v) for k, v in self.hist.items()}


class Ledger:
    """One entry per sorted brick: when a bowl's tally goes up, record the brick colour of the cycle and the bowl."""

    def __init__(self, keep: int = 40):
        self.keep, self.items, self.prev, self.started = keep, [], {}, None

    def update(self, snap: dict) -> None:
        loop = snap.get("loop") or {}
        if not loop:
            return
        if loop.get("started") != self.started:  # a new run: start a new ledger
            self.started, self.items, self.prev = loop.get("started"), [], dict(loop.get("tallies") or {})
            return
        tallies = loop.get("tallies") or {}
        cycles = [c for c in (loop.get("cycles") or []) if c.get("result") == "sorted"]
        brick = cycles[-1].get("brick") if cycles else snap.get("brick")
        for bowl, n in tallies.items():
            for _ in range(max(0, n - self.prev.get(bowl, 0))):
                self.items.append({"brick": brick, "bowl": bowl, "t": time.time()})
        self.prev = dict(tallies)
        self.items = self.items[-self.keep:]


class Source:
    """Background poller of the console's /ctl/state (or a synthetic robot with --mock)."""

    def __init__(self, url: str, hz: float, mock: bool):
        self.url, self.period, self.mock = url, 1.0 / hz, mock
        self.lock = threading.Lock()
        self.snap: dict = {}
        self.t_ok = 0.0
        self.policy_hint = None
        self.silicon, self.ledger = Silicon(mock=mock), Ledger()
        threading.Thread(target=self._run, daemon=True, name="console-poll").start()

    def _run(self) -> None:
        nxt = time.perf_counter()
        while True:
            try:
                snap = MockRobot.step() if self.mock else json.loads(
                    urllib.request.urlopen(self.url, timeout=0.5).read())
                with self.lock:
                    self.snap, self.t_ok = snap, time.time()
                    self.ledger.update(snap)
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
        out["now"] = time.time()
        out["silicon"] = self.silicon.state()
        with self.lock:
            out["ledger"] = list(self.ledger.items)
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
        prompt = {"pick": f"pick up the {brick} brick", "bring": "bring the brick to the inspection camera",
                  "place": f"place the brick in the {bowl} bowl"}[stage]
        return {"phase": "running", "stage": stage, "prompt": prompt, "bowl": bowl, "bowls": list(cls.bowls), "brick": brick,
                "latency_ms": round(300 + 25 * math.sin(t * 1.7) + random.uniform(-15, 15), 1),
                "latency_median_ms": 312.0, "chunk_age_s": round((t % 0.33), 2), "chunks": 2, "inflight": True,
                "measured": [round(v, 2) for v in q], "cmd": [round(v, 2) for v in cmd], "blend": "rtc(h=15, g=3.0)",
                "loop": {"mode": "running", "step": step, "cycle": n + 1, "attempts": 0, "tallies": dict(cls.tallies),
                         "cycles": cur, "step_s": round(ph * period, 1), "target": table[0] if table else None,
                         "started": cls.t0,
                         "grasp": {"state": "holding" if stage != "pick" else "open", "pos": q[5]}},
                "perception": {"table": table, "sorted": [], "workspace": [0, 60, 560, 480],
                               "plates": [{"colour": "yellow", "cx": 480, "cy": 120, "r": 55},
                                          {"colour": "purple", "cx": 500, "cy": 300, "r": 55},
                                          {"colour": "orange", "cx": 470, "cy": 420, "r": 50}],
                               "target": table[0] if table else None,
                               "inspect": {"held": [250, 170, 390, 300], "close": step == "inspect"} if stage == "bring"
                               else None, "age_s": 0.2},
                "studio": {"name": "molmoact2-cal2-456-8k", "policy": "molmoact2", "model_id": "mock"},
                "inspection": {"state": "ok", "outcome": ("none", "circle", "cross")[n % 3], "frames": 61}
                if step == "inspect" else None}


LOOP_COMMANDS = ("start", "pause", "resume", "stop", "home")


def make_handler(source: Source, config: dict, twin_upstream: str = "", console: str = ""):

    class Handler(BaseHTTPRequestHandler):
        # keep-alive: the twin renderer polls /api/pose ~30 times a second through an SSH tunnel, where a new
        # connection costs 0.2-0.4 s. Every reply sets Content-Length; the streaming relay closes its connection.
        protocol_version = "HTTP/1.1"

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

        def _relay(self, url: str) -> None:
            """Pass the twin's MJPEG stream through, so browsers that are not on the XPS can show it too."""
            try:
                up = urllib.request.urlopen(url, timeout=5)
            except Exception:  # noqa: BLE001  renderer not running: the page shows the top camera instead
                return self._send(502, b"", "text/plain")
            self.send_response(200)
            self.send_header("Content-Type", up.headers.get("Content-Type", "multipart/x-mixed-replace"))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            try:
                while chunk := up.read1(65536):
                    self.wfile.write(chunk)
            except Exception:  # noqa: BLE001  viewer closed the page or the stream dropped
                pass
            finally:
                up.close()

        def do_POST(self):  # noqa: N802
            """Loop controls (Start / Pause / Resume / Stop / Home), passed to the console's /loop/<cmd>."""
            name = self.path.split("?", 1)[0].removeprefix("/loop/")
            if not self.path.startswith("/loop/") or name not in LOOP_COMMANDS:
                return self._send(404, b"", "text/plain")
            if not console:  # mock
                return self._send(200, b"{}", "application/json")
            try:
                req = urllib.request.Request(f"{console}/loop/{name}", data=b"", method="POST")
                urllib.request.urlopen(req, timeout=3).read()
                return self._send(200, b"{}", "application/json")
            except Exception:  # noqa: BLE001  console down or not a loop controller
                return self._send(502, b"{}", "application/json")

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
            if path == "/twin.mjpg" and twin_upstream:
                return self._relay(twin_upstream)
            if path == "/twin/health" and twin_upstream:
                try:
                    body = urllib.request.urlopen(twin_upstream.rsplit("/", 1)[0] + "/health", timeout=2).read()
                    return self._send(200, body, "application/json")
                except Exception:  # noqa: BLE001
                    return self._send(502, b"{}", "application/json")
            return self._send(404, b"", "text/plain")

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--console", default="http://localhost:8020", help="the robot console (serves /ctl/state)")
    ap.add_argument("--twin", default="http://localhost:8030/stream.mjpg",
                    help="Isaac live-mirror MJPEG stream (relayed to the page as /twin.mjpg)")
    ap.add_argument("--cams", default="", help="base URL of the camera streams (/stream/<cam>); default the console")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8040)
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("--mock", action="store_true", help="synthetic robot, no hardware")
    a = ap.parse_args()
    source = Source(a.console.rstrip("/") + "/ctl/state", a.hz, a.mock)
    config = {"twin": "/twin.mjpg" if a.twin else "", "cams": "" if a.mock else (a.cams or a.console).rstrip("/"),
              "mock": a.mock}
    server = ThreadingHTTPServer((a.host, a.port),
                                 make_handler(source, config, a.twin, "" if a.mock else a.console.rstrip("/")))
    print(f"station display on http://{a.host}:{a.port}  (console {a.console}{', MOCK' if a.mock else ''}; "
          f"twin {a.twin})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
