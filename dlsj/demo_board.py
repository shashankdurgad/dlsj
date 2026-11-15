"""Pitch dashboard for the closed loop.

On the XPS: python -m dlsj.demo_board --host 0.0.0.0
Then open http://100.77.191.42:8090 from the tailnet, or http://127.0.0.1:8090 on the XPS.

Shows the held-out Anomalib numbers and the dataset already in Physical AI
Studio. The optimize frame stays empty until those measurements exist. The
brief page does not drive the arm. /live is the brick-sorting console: it
forwards the camera streams and the start, pause, home and stop controls to
the loop already running on port 8020, and it does not load a model.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import struct
import xml.etree.ElementTree as ET
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from string import Template
from urllib.error import URLError
from urllib.request import Request, urlopen

from dlsj import live_console

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
PATCHCORE = ROOT / "models/brick_patchcore_geometry/evaluation.json"
CLASSIFIER = ROOT / "models/brick_yolo_classifier/joint_evaluation.json"
TRACES = HERE / "demo_traces.json"
ARM_JS = HERE / "demo_arm.js"
VENDOR = HERE / "vendor"
SO101 = HERE / "so101_assets"
QINDEX = {
    "shoulder_pan": 0,
    "shoulder_lift": 1,
    "elbow_flex": 2,
    "wrist_flex": 3,
    "wrist_roll": 4,
    "gripper": 5,
}

# duplo_posttrain_814_v2, the set imported into Studio as "Duplo teleoperation".
# 431 full (cal2) + 133 pick-only + 250 full (original) = 814.
DATASET = {
    "episodes": 814,
    "frames": 168_119,
    "full": 681,
    "pick_only": 133,
    # 431 full runs on the 14 Nov calibration, 250 earlier full runs kept after video review.
    "cal2": 431,
    "kept": 250,
    "yellow": 229,
    "orange": 201,
    "purple": 251,
}

def _pct(correct: int, total: int) -> str:
    return f"{100 * correct / total:.1f}"


def load_context() -> dict:
    patch = json.loads(PATCHCORE.read_text())
    stage2 = json.loads(CLASSIFIER.read_text())["test"]
    test = patch["test"]
    classes = [
        ("Plain", "plain", test["good"], "good bricks kept"),
        ("Circle", "minor", test["circle"], "minor mark caught"),
        ("Cross", "bad", test["bad"], "defect caught"),
    ]
    anomaly = []
    for label, kind, row, caption in classes:
        anomaly.append({
            "label": label,
            "kind": kind,
            "correct": row["correct"],
            "captures": row["captures"],
            "pct": _pct(row["correct"], row["captures"]),
            "caption": caption,
        })
    return {
        "balanced": f"{100 * patch['test_balanced_accuracy']:.1f}",
        "train_images": f"{patch['training_images']:,}",
        "anomaly": anomaly,
        "stage2_pct": f"{100 * stage2['accuracy_among_decided']:.1f}",
        "stage2_decided": stage2["decided"],
        "stage2_correct": stage2["correct"],
        "stage2_uncertain": stage2["uncertain"],
        "stage2_captures": stage2["captures"],
        **DATASET,
    }


def _bar(kind: str, label: str, value: str, pct: str) -> str:
    return (
        f'<div class="row {kind}"><div class="row-top"><span>{label}</span><b>{value}</b></div>'
        f'<div class="track"><i style="width:{pct}%"></i></div></div>'
    )


def render_page(ctx: dict) -> str:
    detect = "".join(
        _bar(r["kind"], r["label"], f'{r["correct"]}/{r["captures"]}', r["pct"])
        for r in ctx["anomaly"]
    )
    total = ctx["episodes"]
    data = "".join(
        _bar("flat", label, str(n), f"{100 * n / total:.1f}")
        for label, n in (
            ("After 14 Nov", ctx["cal2"]),
            ("Kept", ctx["kept"]),
            ("Pick only", ctx["pick_only"]),
        )
    )
    return Template(PAGE).safe_substitute(
        balanced=ctx["balanced"],
        train_images=ctx["train_images"],
        stage2_correct=ctx["stage2_correct"],
        stage2_decided=ctx["stage2_decided"],
        episodes=ctx["episodes"],
        frames=f"{ctx['frames']:,}",
        detect=detect,
        data=data,
    )


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DLSJ</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
:root{--bg:#0a0b0d;--surface:#121418;--surface-2:#1a1d22;--line:rgba(255,255,255,.07);--fg:#ececee;--mute:#8b9099;--faint:#545a63;--yellow:#f2c230;--orange:#f08a24;--purple:#9b7fe0;--ok:#3ecf8e;--red:#e5484d}
*{box-sizing:border-box;margin:0}
html,body{height:100%}
body{background:var(--bg);color:var(--fg);font:400 14px/1.4 Inter,ui-sans-serif,system-ui,sans-serif;-webkit-font-smoothing:antialiased;font-variant-numeric:tabular-nums}
header{height:56px;display:flex;align-items:center;gap:20px;padding:0 20px;border-bottom:1px solid var(--line)}
.brand{font-weight:600;font-size:13px;letter-spacing:.22em}
.seg{display:flex;background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:3px}
.seg a{padding:5px 14px;border-radius:7px;color:var(--mute);text-decoration:none;font-size:13px;font-weight:500}
.seg a.on{background:var(--surface-2);color:var(--fg)}
.status{display:flex;align-items:center;gap:8px;font-size:13px;color:var(--mute)}
.status b{font-weight:500}
.status i{width:7px;height:7px;border-radius:50%;background:var(--faint)}
.status.on i{background:var(--ok);box-shadow:0 0 0 4px rgba(62,207,142,.15)}
.status.on{color:var(--fg)}
.page{height:calc(100vh - 56px);display:grid;grid-template-rows:1fr auto;gap:16px;padding:20px}
main{min-height:0;display:grid;grid-template-columns:1fr 1.5fr 1fr;gap:16px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:18px;display:flex;flex-direction:column;min-height:0}
.label{font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:.08em;color:var(--mute)}
.big{font-weight:300;font-size:64px;line-height:1;letter-spacing:-.03em;margin-top:22px}
.big small{font-size:28px;color:var(--mute);margin-left:4px;letter-spacing:0}
.cap{color:var(--mute);font-size:13px;margin-top:8px}
.rows{margin-top:auto;display:grid;gap:18px;padding-top:28px}
.row-top{display:flex;justify-content:space-between;font-size:13px;color:var(--mute)}
.row-top b{font-weight:400;color:var(--fg)}
.track{height:3px;background:var(--surface-2);border-radius:99px;margin-top:8px;overflow:hidden}
.track i{display:block;height:100%;border-radius:99px;background:var(--faint)}
.plain .track i{background:var(--yellow)}.minor .track i{background:var(--orange)}.bad .track i{background:var(--purple)}
.flat .track i{background:var(--mute)}
.stage{position:relative;flex:1;min-height:220px;margin-top:14px;border-radius:10px;overflow:hidden}
#arm{position:absolute;inset:0;width:100%;height:100%;display:block}
.cap.mid{text-align:center;margin-top:12px;color:var(--fg);font-weight:300;font-size:15px}
footer{display:flex;flex-wrap:wrap;gap:6px 28px;align-items:center;color:var(--faint);font-size:12px}
footer span b{color:var(--mute);font-weight:500}
footer .opt{margin-left:auto}
@media(max-width:980px){.page{height:auto}main{grid-template-columns:1fr}.big{font-size:56px}.stage{min-height:320px}footer .opt{margin-left:0}}
</style></head>
<body>
<header><span class="brand">DLSJ</span><nav class="seg"><a class="on" href="/">Brief</a><a href="/live">Live</a></nav><span style="flex:1"></span><span id="live" class="status"><i></i><b>Idle</b></span></header>
<div class="page">
<main>
<section class="card">
  <div class="label">Detect</div>
  <div class="big">$balanced<small>%</small></div>
  <p class="cap">balanced accuracy &middot; $train_images plain images</p>
  <div class="rows">$detect</div>
</section>
<section class="card">
  <div class="label">Teleop</div>
  <div class="stage"><canvas id="arm"></canvas></div>
  <div class="cap mid" id="decision">All 814 runs, together</div>
</section>
<section class="card">
  <div class="label">Data</div>
  <div class="big">$episodes</div>
  <p class="cap">episodes &middot; $frames frames</p>
  <div class="rows">$data</div>
</section>
</main>
<footer>
  <span><b>Check</b> 15 of 16 plain kept, was 9 of 16</span>
  <span><b>Unsure</b> twice, then defective</span>
  <span><b>Motion</b> the bowl is the sentence</span>
  <span><b>Pick</b> 133 stop at the camera</span>
  <span class="opt"><b>Optimize</b> not measured yet</span>
</footer>
</div>
<script type="importmap">{"imports":{"three":"/vendor/three.module.js"}}</script>
<script type="module" src="/demo_arm.js?v=4"></script>
<script>
async function tick(){
  try{
    const s=await (await fetch("/api/live")).json();
    const el=document.getElementById("live"), t=el.querySelector("b"), d=document.getElementById("decision");
    if(!s.live){t.textContent="Idle"; el.className="status"; return;}
    t.textContent="Live \\u00b7 "+s.state; el.className="status on";
    if(s.label) d.textContent=s.label;
  }catch(e){}
}
setInterval(tick,700); tick();
</script>
</body></html>"""



def compile_traces(root: Path, steps: int = 16) -> dict:
    """Downsample every episode to `steps` joint frames. Bowl comes from the place task."""
    import pyarrow.parquet as pq

    traces = []
    for path in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(path, columns=["episode_index", "observation.state", "task_index"])
        episodes = table.column("episode_index").to_pylist()
        states = table.column("observation.state").to_pylist()
        tasks = table.column("task_index").to_pylist()
        start = 0
        while start < len(episodes):
            end = start
            while end < len(episodes) and episodes[end] == episodes[start]:
                end += 1
            span = end - start - 1
            frames = []
            for step in range(steps):
                row = states[start + span * step // (steps - 1)]
                frames.append([round(float(v), 1) for v in row])
            seen = set(tasks[start:end])
            bowl = next((code for code in (2, 3, 0) if code in seen), 1)
            traces.append({"b": bowl, "q": frames})
            start = end
    return {"steps": steps, "episodes": len(traces), "traces": traces}


def live_snapshot(port: int = 8001) -> dict:
    idle = {"live": False, "policy": "π0.5", "cameras": ["wrist", "top", "inspect"]}
    try:
        with urlopen(f"http://127.0.0.1:{port}/state", timeout=0.4) as resp:
            state = json.load(resp)
    except (URLError, TimeoutError, OSError, json.JSONDecodeError):
        return idle
    loop = state.get("loop") or {}
    inspection = loop.get("last_inspection") or {}
    decision = loop.get("last_decision") or {}
    skill = loop.get("skill") or {}
    label = (inspection.get("mark") or inspection.get("label") or "").replace("_", " ")
    return {
        "live": True,
        "state": loop.get("state") or "running",
        "cameras": state.get("cameras") or idle["cameras"],
        "policy": "π0.5",
        "label": label.upper() if label else "WATCHING",
        "detail": decision.get("reason") or decision.get("skill") or "",
        "inspection": {
            "label": inspection.get("label"),
            "mark": inspection.get("mark"),
            "score": inspection.get("score"),
            "threshold": inspection.get("threshold"),
            "outcome": inspection.get("outcome"),
        } if inspection else None,
        "decision": {
            "skill": decision.get("skill"),
            "reason": decision.get("reason"),
            "forced": bool(decision.get("forced")),
        } if decision else None,
        "policy_view": {
            "instruction": skill.get("instruction") or None,
            "queue": skill.get("queue") if isinstance(skill.get("queue"), int) else None,
            "plan": _plan(skill.get("plan")),
        },
    }


def _plan(raw) -> list | None:
    """A policy chunk is (n, 6) joint targets. Reject anything else before it reaches the page."""
    if not isinstance(raw, list) or not 2 <= len(raw) <= 64:
        return None
    out = []
    for row in raw:
        if not isinstance(row, (list, tuple)) or len(row) != 6:
            return None
        try:
            out.append([round(float(v), 2) for v in row])
        except (TypeError, ValueError):
            return None
    return out


def _jpeg_from(url: str, timeout: float = 0.8) -> bytes | None:
    """One JPEG from a still or from the first frame of a multipart stream."""
    try:
        with urlopen(url, timeout=timeout) as resp:
            kind = resp.headers.get("Content-Type", "")
            if "image/jpeg" in kind:
                return resp.read()
            data = b""
            while len(data) < 1_500_000:
                chunk = resp.read(16384)
                if not chunk:
                    break
                data += chunk
                start = data.find(b"\xff\xd8")
                end = data.find(b"\xff\xd9", start + 2) if start >= 0 else -1
                if start >= 0 and end >= 0:
                    return data[start:end + 2]
    except (URLError, TimeoutError, OSError):
        return None
    return None


_CAMERAS = {"wrist", "top", "inspect"}


def live_frame(name: str, port: int = 8001) -> bytes | None:
    if name not in _CAMERAS:
        return None
    return _jpeg_from(f"http://127.0.0.1:{port}/stream/{name}")


def inspection_frame(port: int = 8001) -> bytes | None:
    return _jpeg_from(f"http://127.0.0.1:{port}/inspection.jpg")


LOOP = os.environ.get("DLSJ_LOOP_URL", "http://127.0.0.1:8020")
_LOOP_CMDS = {"start", "pause", "resume", "home", "stop", "reset"}
_LOOP_STREAMS = {"top", "inspect", "wrist", "perc_inspect"}
# Pose, collect and plate-routing commands of the 8020 console, e.g. pose/off_nudge/shoulder_pan/-5, collect/set/plain,
# routing/cross/orange, routing/reset.
_LOOP_TOOL = re.compile(r"(pose|collect|routing)/[a-z_]+(/[a-z_]+(/-?\d+)?)?")


def _open_loop(path: str, method: str = "GET"):
    return urlopen(Request(LOOP + path, method=method), timeout=4)


# The console page lives in dlsj/live_console.py, shared with the loop on 8020.
LIVE_PAGE = live_console.page(proxy=True)


class Board:
    def __init__(self, host: str = "127.0.0.1", port: int = 8090):
        self.page = render_page(load_context()).encode()
        self.live_page = LIVE_PAGE.encode()
        self.traces = TRACES.read_text() if TRACES.exists() else '{"traces":[]}'
        self.server = ThreadingHTTPServer((host, port), self._handler())
        self.port = self.server.server_address[1]

    def _handler(self):
        page, live_page, traces = self.page, self.live_page, self.traces.encode()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                return

            def _send(self, code: int, body: bytes, content: str, cache: bool = False) -> None:
                self.send_response(code)
                self.send_header("Content-Type", content)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "public, max-age=3600" if cache else "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _pipe(self, resp) -> None:
                ctype = resp.headers.get("Content-Type", "application/octet-stream")
                self.send_response(resp.status)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                length = resp.headers.get("Content-Length")
                if length and "multipart" not in ctype:
                    self.send_header("Content-Length", length)
                self.end_headers()
                try:
                    while True:
                        chunk = resp.read(16384)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return

            def _loop(self, path: str, method: str = "GET"):
                try:
                    with _open_loop(path, method) as resp:
                        self._pipe(resp)
                except (URLError, TimeoutError, OSError):
                    self._send(502, b"offline", "text/plain")

            def do_POST(self):
                path = self.path.split("?", 1)[0]
                cmd = path.removeprefix("/live/loop/")
                if path.startswith("/live/loop/") and cmd in _LOOP_CMDS:
                    return self._loop("/loop/" + cmd, "POST")
                tool = path.removeprefix("/live/")
                if path.startswith("/live/") and _LOOP_TOOL.fullmatch(tool):
                    return self._loop("/" + tool, "POST")
                self._send(404, b"not found", "text/plain")

            def do_GET(self):
                path = self.path.split("?", 1)[0]
                if path in ("/", "/index.html"):
                    return self._send(200, page, "text/html; charset=utf-8")
                if path == "/live":
                    return self._send(200, live_page, "text/html; charset=utf-8")
                if path == "/live/state":
                    return self._loop("/ctl/state")
                if path.startswith("/live/stream/"):
                    name = path.removeprefix("/live/stream/")
                    if name in _LOOP_STREAMS:
                        return self._loop("/stream/" + name)
                    return self._send(404, b"", "text/plain")
                if path.startswith("/live/cam/"):
                    name = path.removeprefix("/live/cam/")
                    width = self.path.partition("?w=")[2]
                    if name in _CAMERAS:
                        return self._loop(f"/live/{name}?w={width if width.isdigit() else 640}")
                    return self._send(404, b"", "text/plain")
                if path == "/live/strip.jpg":
                    return self._loop("/loop/strip.jpg")
                if path.startswith("/live/frame/"):
                    jpeg = live_frame(path.removeprefix("/live/frame/"))
                    if jpeg:
                        return self._send(200, jpeg, "image/jpeg")
                    return self._send(404, b"", "text/plain")
                if path == "/live/inspection.jpg":
                    jpeg = inspection_frame()
                    if jpeg:
                        return self._send(200, jpeg, "image/jpeg")
                    return self._send(404, b"", "text/plain")
                if path == "/api/live":
                    body = json.dumps(live_snapshot()).encode()
                    return self._send(200, body, "application/json")
                if path == "/api/arm":
                    return self._send(200, traces, "application/json")
                static = _static(path)
                if static is not None:
                    body, mime = static
                    return self._send(200, body, mime, cache=path != "/demo_arm.js")
                self._send(404, b"not found", "text/plain")

        return Handler

    def serve(self) -> None:
        print(f"demo board: http://{self.server.server_address[0]}:{self.port}", flush=True)
        self.server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--compile", type=Path, help="LeRobot dataset to compress into demo_traces.json")
    parser.add_argument("--bake-so101", metavar="STUDIO", help="Download the SO-101 URDF and meshes from this Studio URL")
    args = parser.parse_args()
    if args.compile:
        payload = compile_traces(args.compile)
        TRACES.write_text(json.dumps(payload, separators=(",", ":")))
        print(f"wrote {payload['episodes']} episodes to {TRACES}", flush=True)
        return
    if args.bake_so101:
        bake_so101(args.bake_so101)
        return
    Board(args.host, args.port).serve()


_MIME = {".js": "text/javascript; charset=utf-8", ".json": "application/json", ".stl": "model/stl"}


def _static(url_path: str) -> tuple[bytes, str] | None:
    if url_path == "/demo_arm.js":
        path = ARM_JS
    elif url_path.startswith("/vendor/"):
        path = _inside(VENDOR, url_path.removeprefix("/vendor/"))
    elif url_path.startswith("/so101/"):
        path = _inside(SO101, url_path.removeprefix("/so101/"))
    else:
        return None
    if path is None or not path.is_file():
        return None
    return path.read_bytes(), _MIME.get(path.suffix, "application/octet-stream")


def _inside(root: Path, name: str) -> Path | None:
    if not name or "\\" in name or name.startswith("/"):
        return None
    path = (root / name).resolve()
    if root.resolve() not in path.parents:
        return None
    return path


def _rpy(roll: float, pitch: float, yaw: float) -> list[list[float]]:
    """URDF fixed-axis rpy: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz],
        [cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz],
        [-sy, sx * cy, cx * cy],
    ]


def _mat(rotation: list[list[float]], xyz: list[float]) -> list[float]:
    """Column-major 4x4, matching THREE.Matrix4."""
    r = rotation
    return [
        r[0][0], r[1][0], r[2][0], 0.0,
        r[0][1], r[1][1], r[2][1], 0.0,
        r[0][2], r[1][2], r[2][2], 0.0,
        xyz[0], xyz[1], xyz[2], 1.0,
    ]


def _mul(a: list[float], b: list[float]) -> list[float]:
    out = [0.0] * 16
    for col in range(4):
        for row in range(4):
            out[col * 4 + row] = (
                a[row] * b[col * 4]
                + a[4 + row] * b[col * 4 + 1]
                + a[8 + row] * b[col * 4 + 2]
                + a[12 + row] * b[col * 4 + 3]
            )
    return out


def _rot_axis(axis: list[float], angle: float) -> list[float]:
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    c1 = 1.0 - c
    return _mat(
        [
            [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
            [y * x * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
            [z * x * c1 - y * s, z * y * c1 + x * s, c + z * z * c1],
        ],
        [0.0, 0.0, 0.0],
    )


def _apply(m: list[float], v: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = v
    return (
        m[0] * x + m[4] * y + m[8] * z + m[12],
        m[1] * x + m[5] * y + m[9] * z + m[13],
        m[2] * x + m[6] * y + m[10] * z + m[14],
    )


def link_frames(joints: list[dict], q_deg: list[float]) -> dict[str, list[float]]:
    """World matrix of every link. Joint angles are dataset degrees."""
    frames = {"base_link": _mat([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0, 0, 0])}
    for joint in joints:
        q = math.radians(q_deg[QINDEX[joint["name"]]])
        origin = _mat(_rpy(*joint["rpy"]), joint["xyz"])
        local = _mul(origin, _rot_axis(joint["axis"], q))
        frames[joint["child"]] = _mul(frames[joint["parent"]], local)
    return frames


def _origin(node: ET.Element | None) -> tuple[list[float], list[float]]:
    if node is None:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    xyz = [float(v) for v in (node.get("xyz") or "0 0 0").split()]
    rpy = [float(v) for v in (node.get("rpy") or "0 0 0").split()]
    return xyz, rpy


def _chain(joints: list[dict]) -> list[dict]:
    by_parent: dict[str, list[dict]] = {}
    for joint in joints:
        by_parent.setdefault(joint["parent"], []).append(joint)
    ordered: list[dict] = []

    def walk(link: str) -> None:
        for joint in by_parent.get(link, []):
            ordered.append(joint)
            walk(joint["child"])

    walk("base_link")
    return ordered


def _read_stl(data: bytes) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    count = struct.unpack_from("<I", data, 80)[0]
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    offset = 84
    for _ in range(count):
        offset += 12
        base = len(verts)
        for _v in range(3):
            verts.append(struct.unpack_from("<fff", data, offset))
            offset += 12
        offset += 2
        faces.append((base, base + 1, base + 2))
    return verts, faces


def _write_stl(path: Path, verts: list[tuple[float, float, float]], faces: list[tuple[int, int, int]]) -> None:
    out = bytearray(b"so101" + b"\0" * 75)
    out += struct.pack("<I", len(faces))
    for a, b, c in faces:
        va, vb, vc = verts[a], verts[b], verts[c]
        nx = (vb[1] - va[1]) * (vc[2] - va[2]) - (vb[2] - va[2]) * (vc[1] - va[1])
        ny = (vb[2] - va[2]) * (vc[0] - va[0]) - (vb[0] - va[0]) * (vc[2] - va[2])
        nz = (vb[0] - va[0]) * (vc[1] - va[1]) - (vb[1] - va[1]) * (vc[0] - va[0])
        norm = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
        out += struct.pack("<fff", nx / norm, ny / norm, nz / norm)
        out += struct.pack("<fff", *va)
        out += struct.pack("<fff", *vb)
        out += struct.pack("<fff", *vc)
        out += b"\0\0"
    path.write_bytes(out)


def _weld(verts, faces, grid: float):
    index: dict[tuple[int, int, int], int] = {}
    fresh: list[tuple[float, float, float]] = []
    remap = []
    for x, y, z in verts:
        key = (round(x / grid), round(y / grid), round(z / grid))
        slot = index.get(key)
        if slot is None:
            slot = len(fresh)
            index[key] = slot
            fresh.append([x, y, z, 1])
        else:
            acc = fresh[slot]
            acc[0] += x
            acc[1] += y
            acc[2] += z
            acc[3] += 1
        remap.append(slot)
    fresh = [(v[0] / v[3], v[1] / v[3], v[2] / v[3]) for v in fresh]
    welded = []
    seen = set()
    for a, b, c in faces:
        tri = (remap[a], remap[b], remap[c])
        if len(set(tri)) < 3 or tri in seen:
            continue
        seen.add(tri)
        welded.append(tri)
    return fresh, welded


def _simplify(verts, faces, limit: int):
    grid = 0.0012
    best = (verts, faces)
    while len(best[1]) > limit and grid < 0.02:
        best = _weld(verts, faces, grid)
        grid *= 1.4
    return best


def _fetch_asset(source: str, name: str) -> bytes:
    if source.startswith("http"):
        prefix = source.rstrip("/") + "/api/robots/catalog/SO101_Follower"
        url = f"{prefix}/urdf" if name == "so101.urdf" else f"{prefix}/assets/{name}"
        with urlopen(url, timeout=60) as resp:
            return resp.read()
    return (Path(source) / name).read_bytes()


def bake_so101(studio: str) -> None:
    """Bake the Studio SO-101 URDF. `studio` is a Studio base URL or a directory of the URDF and STLs."""
    root = ET.fromstring(_fetch_asset(studio, "so101.urdf"))
    joints = []
    for node in root.findall("joint"):
        if node.get("type") != "revolute":
            continue
        xyz, rpy = _origin(node.find("origin"))
        axis = [float(v) for v in (node.find("axis").get("xyz") if node.find("axis") is not None else "0 0 1").split()]
        joints.append({
            "name": node.get("name"),
            "parent": node.find("parent").get("link"),
            "child": node.find("child").get("link"),
            "xyz": xyz,
            "rpy": rpy,
            "axis": axis,
        })
    joints = [j for j in _chain(joints) if j["name"] in QINDEX]
    SO101.mkdir(parents=True, exist_ok=True)
    cache: dict[str, tuple] = {}
    visuals = []
    moving_tris = 0
    for link in root.findall("link"):
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                continue
            filename = Path(mesh.get("filename") or "").name
            if filename not in cache:
                cache[filename] = _read_stl(_fetch_asset(studio, filename))
            xyz, rpy = _origin(visual.find("origin"))
            placed = [_apply(_mat(_rpy(*rpy), xyz), v) for v in cache[filename][0]]
            material = (visual.find("material").get("name") if visual.find("material") is not None else "") or ""
            kind = "motor" if "sts" in material else "plastic"
            limit = 500 if kind == "motor" else 1400
            verts, faces = _simplify(placed, cache[filename][1], limit)
            out_name = f"{link.get('name')}__{len(visuals)}.stl"
            _write_stl(SO101 / out_name, verts, faces)
            visuals.append({"link": link.get("name"), "file": out_name, "material": kind, "tris": len(faces)})
            if link.get("name") != "base_link":
                moving_tris += len(faces)
            print(f"{out_name} {kind} {len(faces)} tris", flush=True)
    flock = max(32, min(814, 1_400_000 // max(moving_tris, 1)))
    payload = {"joints": joints, "visuals": visuals, "flock": flock, "tcp": [-0.0079, -0.000218121, -0.0981274]}
    (SO101 / "kin.json").write_text(json.dumps(payload))
    print(f"moving tris {moving_tris}, flock {flock}", flush=True)
    VENDOR.mkdir(parents=True, exist_ok=True)
    three = VENDOR / "three.module.js"
    if not three.exists():
        with urlopen("https://unpkg.com/three@0.160.1/build/three.module.js", timeout=60) as resp:
            three.write_bytes(resp.read())
        print(f"wrote {three}", flush=True)


if __name__ == "__main__":
    main()
