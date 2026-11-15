"""HTTP side of the live twin mirror: MJPEG stream, health JSON and the pose poller, in a child process.

The Isaac process (live.py) creates a `Shared` block and starts `serve()` in a forked child BEFORE Isaac boots, so this
process never shares state with Kit (and never imports PyAV). Data flow through shared memory:
* pose: this process polls `pose_url` at ~30 Hz and writes the latest valid pose; Isaac reads it every frame;
* frames: Isaac writes RGB frames into two slots (double buffer) and bumps `seq`; this process JPEG-encodes each new
  frame once and fans it out to every connected `<img src=".../stream.mjpg">` client;
* stats: Isaac writes render_ms / frame / black-frame counters; /health reports them with the measured stream fps.
Stdlib + numpy + cv2 (falls back to PIL) only.
"""
from __future__ import annotations

import fcntl
import http.client
import json
import math
import multiprocessing as mp
import os
import socket
import struct
import subprocess
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

# stats slots (float64)
S_RENDER_MS, S_FRAMES, S_BLACK, S_READY, S_LOOP_MS, S_HEARTBEAT, S_DIED = range(7)
# pose slots: [seq, t_recv (local), t_src, q0..q5]
P_SEQ, P_RECV, P_SRC, P_Q = 0, 1, 2, 3


class Shared:
    """Shared-memory block created before the fork (lock-free; a torn frame read is at worst a mixed frame)."""

    def __init__(self, width: int, height: int):
        self.w, self.h = width, height
        n = width * height * 3
        self.frames = [mp.RawArray("B", n), mp.RawArray("B", n)]
        self.seq = mp.RawValue("q", 0)  # frame counter; the latest frame is in slot seq % 2
        self.stats = mp.RawArray("d", 8)
        self.pose = mp.RawArray("d", 3 + 6)

    def frame_view(self, slot: int) -> np.ndarray:
        return np.frombuffer(self.frames[slot], np.uint8).reshape(self.h, self.w, 3)

    # Isaac side
    def publish(self, rgb: np.ndarray):
        nxt = int(self.seq.value) + 1
        self.frame_view(nxt % 2)[...] = rgb
        self.seq.value = nxt

    def read_pose(self):
        """(seq, t_recv, t_src, q (6,)) of the latest polled pose; seq 0 = none yet."""
        p = np.frombuffer(self.pose, np.float64).copy()
        return int(p[P_SEQ]), float(p[P_RECV]), float(p[P_SRC]), p[P_Q:P_Q + 6]


def iface_ipv4(name: str) -> str | None:
    """IPv4 address of a network interface (SIOCGIFADDR), None if absent."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        packed = fcntl.ioctl(s.fileno(), 0x8915, struct.pack("256s", name[:15].encode()))
        return socket.inet_ntoa(packed[20:24])
    except OSError:
        return None
    finally:
        s.close()


def tailscale_ip() -> str | None:
    try:
        out = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=3).stdout.split()
        if out:
            return out[0]
    except (OSError, subprocess.SubprocessError):
        pass
    return iface_ipv4("tailscale0")


def resolve_bind(bind: str) -> str:
    """'auto' -> the Tailscale IPv4 if this host has one, else 127.0.0.1. Never 0.0.0.0 unless given explicitly."""
    if bind != "auto":
        return bind
    return tailscale_ip() or "127.0.0.1"


def _parse_pose(body: bytes):
    """-> (t_src, q or None, frame, age or None). q None (no console seen yet) counts as stale."""
    d = json.loads(body)
    q = d.get("q")
    age = d.get("age")
    age = float(age) if isinstance(age, (int, float)) and math.isfinite(age) else None
    if q is not None:
        if len(q) < 6:
            raise ValueError(f"q has {len(q)} values")
        q = [float(v) for v in q[:6]]
        if not all(math.isfinite(v) for v in q):
            raise ValueError("non-finite q")
        q[5] = min(100.0, max(0.0, q[5]))
    t = float(d.get("t", 0.0) or 0.0)  # extra fields (e.g. "plan") are ignored
    return t, q, d.get("frame"), age


class PosePoller:
    """GET pose_url at up to `hz` over `workers` PERSISTENT HTTP/1.1 connections (keep-alive; reconnect on error),
    staggered. Answers are accepted in source-time order and written into shared memory. Isaac holds the last pose
    (nothing new is written) when the server is unreachable, q is null, or the server's own `age` (s since its last
    good console read) is null or exceeds `stale_s`. pose_age_s = time since the last fresh answer + its age."""

    def __init__(self, sh: Shared, url: str, hz: float = 30.0, stale_s: float = 1.0, workers: int = 5):
        self.sh, self.url, self.hz, self.stale_s, self.workers = sh, url, hz, stale_s, workers
        self.lock = threading.Lock()
        self.ok = self.errors = 0
        self.last_error = ""
        self.last_new = 0.0  # local time the source timestamp last advanced
        self.last_src = None
        self.frame_name = None
        self.clock_offset = None  # local - source clock (s), informational
        self.src_age = None  # the server's own "age" field
        self.last_good = 0.0  # local time of the last pose handed to Isaac
        self.held = 0  # polls answered but held (q null / stale)
        self.good_age = 0.0  # the server's age of the pose last handed to Isaac
        self.rtt_ms = 0.0
        self.accepted: list[float] = []  # local times of accepted poses (rate)
        self._last_log = 0.0

    def start(self):
        for k in range(self.workers):
            threading.Thread(target=self._worker, args=(k,), daemon=True).start()

    def _accept(self, t_src, q, fr, age, rtt):
        now = time.time()
        with self.lock:
            self.ok += 1
            self.rtt_ms = 0.9 * self.rtt_ms + 0.1 * rtt * 1e3 if self.rtt_ms else rtt * 1e3
            if t_src and self.last_src and t_src < self.last_src:
                return  # an older answer overtaken by a newer one
            self.frame_name, self.src_age = fr, age
            self.clock_offset = (now - t_src) if t_src else None
            if t_src != self.last_src or not t_src:
                self.last_src, self.last_new = t_src, now
            # the server's own age decides (its t may not advance while the arm is still); age null = stale
            fresh = q is not None and age is not None and age <= self.stale_s
            if not fresh:
                self.held += 1
                return
            self.last_good, self.good_age = now, age or 0.0
            self.accepted = [t for t in self.accepted if now - t < 5.0] + [now]
            p = self.sh.pose
            p[P_RECV], p[P_SRC] = now, t_src
            for i, v in enumerate(q):
                p[P_Q + i] = v
            p[P_SEQ] = p[P_SEQ] + 1

    def _worker(self, k: int):
        u = urllib.parse.urlsplit(self.url)
        path = (u.path or "/") + (f"?{u.query}" if u.query else "")
        period = self.workers / self.hz  # each connection's own period: all together poll at <= hz
        time.sleep(k * period / self.workers)
        conn = None
        nxt = time.monotonic()
        while True:
            t0 = time.monotonic()
            try:
                if conn is None:
                    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=2.0)
                conn.request("GET", path, headers={"Connection": "keep-alive"})
                r = conn.getresponse()
                body = r.read()
                if r.status != 200:
                    raise OSError(f"HTTP {r.status}")
                if r.getheader("Connection", "").lower() == "close" or r.version == 10:
                    conn.close()
                    conn = None
                self._accept(*_parse_pose(body), time.monotonic() - t0)
                delay = period
            except Exception as e:  # noqa: BLE001 (any failure: hold the last pose, reconnect, retry slower)
                if conn is not None:
                    conn.close()
                    conn = None
                with self.lock:
                    self.errors += 1
                    self.last_error = f"{type(e).__name__}: {e}"[:200]
                    if time.time() - self._last_log > 30:
                        print(f"[live/pose] {self.url}: {self.last_error} (holding the last pose)", flush=True)
                        self._last_log = time.time()
                delay = 0.5
            nxt = max(nxt + delay, time.monotonic())
            time.sleep(max(0.0, nxt - time.monotonic()))

    def rate(self) -> float:
        now = time.time()
        ts = [t for t in self.accepted if now - t < 5.0]
        return round((len(ts) - 1) / max(1e-6, ts[-1] - ts[0]), 1) if len(ts) > 1 and now - ts[-1] < 1.0 else 0.0

    def age(self) -> float | None:
        """Age of the pose the mirror is showing: time since it was received + the server's own age at that time."""
        if not self.last_good:
            return None
        return round(time.time() - self.last_good + self.good_age, 3)


class Encoder(threading.Thread):
    """JPEG-encodes each new shared frame once; clients wait on `cond` for `jpeg_seq` to change."""

    def __init__(self, sh: Shared, quality: int):
        super().__init__(daemon=True)
        self.sh, self.quality = sh, quality
        self.cond = threading.Condition()
        self.jpeg, self.jpeg_seq = b"", 0
        self.times: list[float] = []  # publish times of the last frames (fps)
        self.enc_ms = 0.0
        try:
            import cv2

            self._cv2 = cv2
        except ImportError:
            self._cv2 = None

    def encode(self, rgb: np.ndarray) -> bytes:
        if self._cv2 is not None:
            ok, buf = self._cv2.imencode(".jpg", rgb[..., ::-1], [self._cv2.IMWRITE_JPEG_QUALITY, self.quality])
            return buf.tobytes()
        import io

        from PIL import Image

        b = io.BytesIO()
        Image.fromarray(rgb).save(b, "JPEG", quality=self.quality)
        return b.getvalue()

    def run(self):
        seen = 0
        while True:
            s = int(self.sh.seq.value)
            if s == seen:
                time.sleep(0.002)
                continue
            seen = s
            t0 = time.perf_counter()
            rgb = self.sh.frame_view(s % 2).copy()
            jpg = self.encode(rgb)
            self.enc_ms = (time.perf_counter() - t0) * 1e3
            now = time.time()
            with self.cond:
                self.jpeg, self.jpeg_seq = jpg, s
                self.times = [t for t in self.times if now - t < 5.0] + [now]
                self.cond.notify_all()

    def fps(self) -> float:
        now = time.time()
        ts = [t for t in self.times if now - t < 5.0]
        if len(ts) < 2:
            return 0.0
        return round((len(ts) - 1) / max(1e-6, ts[-1] - ts[0]), 2) if now - ts[-1] < 1.0 else 0.0


def make_handler(sh: Shared, enc: Encoder, poller: PosePoller, info: dict, clients: list):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # quiet
            pass

        def _headers(self, code: int, ctype: str, length: int | None = None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate, private, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            if length is not None:
                self.send_header("Content-Length", str(length))
            self.end_headers()

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/health":
                body = json.dumps(health(sh, enc, poller, info, clients)).encode()
                self._headers(200, "application/json", len(body))
                self.wfile.write(body)
            elif path in ("/frame.jpg", "/snapshot.jpg"):
                with enc.cond:
                    if not enc.jpeg:
                        enc.cond.wait(5.0)
                    jpg = enc.jpeg
                if not jpg:
                    self._headers(503, "text/plain", 8)
                    self.wfile.write(b"booting\n")
                    return
                self._headers(200, "image/jpeg", len(jpg))
                self.wfile.write(jpg)
            elif path == "/stream.mjpg":
                self._stream()
            elif path == "/":
                body = (b"<!doctype html><title>twin live</title><body style='margin:0;background:#000'>"
                        b"<img src='stream.mjpg' style='width:100%'></body>")
                self._headers(200, "text/html", len(body))
                self.wfile.write(body)
            else:
                self._headers(404, "text/plain", 10)
                self.wfile.write(b"not found\n")

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate, private, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            me = object()
            clients.append(me)
            sent = -1
            min_dt = 1.0 / max(1.0, info["max_fps"])
            last = 0.0
            try:
                while True:
                    with enc.cond:
                        if enc.jpeg_seq == sent:
                            enc.cond.wait(1.0)  # no new frame for 1 s: resend the last one (keeps proxies alive)
                        jpg, sent = enc.jpeg, enc.jpeg_seq
                    if not jpg:
                        continue
                    wait = min_dt - (time.time() - last)
                    if wait > 0:
                        time.sleep(wait)
                    last = time.time()
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                                     str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
                pass
            finally:
                clients.remove(me)
            self.close_connection = True

    return Handler


def health(sh: Shared, enc: Encoder, poller: PosePoller, info: dict, clients: list) -> dict:
    st = list(sh.stats)
    ready = st[S_READY] > 0
    hb = time.time() - st[S_HEARTBEAT] if st[S_HEARTBEAT] else None
    return {
        "fps": enc.fps(),
        "render_ms": round(st[S_RENDER_MS], 2),
        "loop_ms": round(st[S_LOOP_MS], 2),
        "encode_ms": round(enc.enc_ms, 2),
        "pose_age_s": poller.age(),
        "frames": int(st[S_FRAMES]),
        "black_frames_skipped": int(st[S_BLACK]),
        "pose_url": poller.url,
        "pose_ok": poller.ok,
        "pose_held": poller.held,
        "pose_hz": poller.rate(),  # fresh answers per second
        "pose_rtt_ms": round(poller.rtt_ms, 1),
        "pose_src_age_s": poller.src_age,
        "pose_errors": poller.errors,
        "pose_last_error": poller.last_error,
        "pose_frame": poller.frame_name,
        "pose_clock_offset_s": round(poller.clock_offset, 3) if poller.clock_offset is not None else None,
        "look": info["look"],
        "resolution": [sh.w, sh.h],
        "status": "live" if ready and hb is not None and hb < 5 else ("booting" if not ready else "stalled"),
        "isaac_heartbeat_age_s": round(hb, 2) if hb is not None else None,
        "clients": len(clients),
        "bind": info["binds"],
        "started": info["started"],
    }


def serve(sh: Shared, pose_url: str, bind: str, port: int, look: str, quality: int, max_fps: float,
          pose_conns: int = 5):
    """Child-process entry point: pose poller + encoder + HTTP server(s). With bind='auto' it starts on 127.0.0.1
    (or the Tailscale IP) and adds a listener on the Tailscale IP as soon as one appears."""
    parent = os.getppid()
    info = {"look": look, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "binds": [], "max_fps": max_fps}
    clients: list = []
    poller = PosePoller(sh, pose_url, workers=pose_conns)
    poller.start()
    enc = Encoder(sh, quality)
    enc.start()
    handler = make_handler(sh, enc, poller, info, clients)

    def listen(host: str):
        srv = ThreadingHTTPServer((host, port), handler)
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        info["binds"].append(f"{host}:{port}")
        print(f"[live/http] serving http://{host}:{port}/stream.mjpg  /health", flush=True)

    host = resolve_bind(bind)
    if host == "0.0.0.0":
        print("[live/http] WARNING: bound to 0.0.0.0 on request (all interfaces)", flush=True)
    listen(host)
    while True:  # exit with the Isaac process; with --bind auto, pick up Tailscale when it comes up
        time.sleep(5)
        if os.getppid() != parent:
            os._exit(0)
        if bind == "auto" and host == "127.0.0.1":
            ts = tailscale_ip()
            if ts:
                try:
                    listen(ts)
                    host = ts
                except OSError as e:
                    print(f"[live/http] could not bind {ts}: {e}", flush=True)
