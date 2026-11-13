"""Capture tagged stills from the inspection camera for a brick-in-view detector, from a browser.

    uv run python -m dlsj.inspect_capture --camera 4 --sync gb10:desmond/inspect_frames/

Open http://localhost:8002.

Frames are saved as full-resolution JPEGs to <out>/<session>/<tag>/<00000>.jpg and listed in
<session>/index.csv (file, tag, unix time, width, height). The same camera frame is never saved twice.

Keys while the page is focused:
  1   in view (brick fully in view)       2   not in view (no brick, partial brick, hands, bowls)
  space  start or stop burst capture      s   save one frame

With --sync, the output folder is copied with rsync every ~10 s and once at exit. ssh auth comes from the
environment (SSH_ASKPASS, SSH_ASKPASS_REQUIRE=force, DISPLAY). Only paths under ~/desmond on the remote
host are allowed, and rsync never deletes anything there.
"""

import argparse
import csv
import json
import logging
import shlex
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

from dlsj import teleop_web

log = logging.getLogger("inspect_capture")

TAGS = ("in_view", "not_in_view")
DEFAULT_SSH_CONFIG = (
    "/private/tmp/claude-501/-Users-desmondzee-Hardware-dlsj/4698d3ab-97f9-4aa3-a532-a5fb532f6b27"
    "/scratchpad/gb10/ssh_config"
)


class Capture:
    """Saves frames from one camera stream into tag folders, singly or in bursts."""

    def __init__(self, stream, out: str | Path, burst_fps: float = 5.0, session: str | None = None, quality: int = 95):
        self.stream = stream
        self.out = Path(out)
        self.session = session or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = self.out / self.session
        self.burst_fps = burst_fps
        self.quality = quality
        self.tag = TAGS[0]
        self.counts = {t: 0 for t in TAGS}
        self.bursting = False
        self.last_seq: int | None = None
        self.syncer: "Syncer | None" = None
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def command(self, name: str) -> None:
        if name in TAGS:
            self.tag = name
        elif name == "burst":
            self.bursting = not self.bursting
        elif name == "save":
            self.save()
        else:
            raise ValueError(f"unknown command {name!r}")

    def save(self) -> Path | None:
        """Write the latest frame under the current tag. Returns None if there is no new frame."""
        with self._lock:
            frame, seq = self.stream.frame, self.stream.seq
            if frame is None or seq == self.last_seq:
                return None
            tag = self.tag
            folder = self.dir / tag
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{self.counts[tag]:05d}.jpg"
            if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality]):
                log.error("could not write %s", path)
                return None
            index = self.dir / "index.csv"
            new = not index.exists()
            with index.open("a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["file", "tag", "time", "width", "height"])
                w.writerow([path.relative_to(self.dir).as_posix(), tag, f"{time.time():.3f}", frame.shape[1], frame.shape[0]])
            self.last_seq = seq
            self.counts[tag] += 1
            return path

    def run(self) -> None:
        period = 1.0 / max(self.burst_fps, 0.1)
        while not self._stop.is_set():
            if self.bursting:
                self.save()
            self._stop.wait(period if self.bursting else 0.05)

    def stop(self) -> None:
        self.bursting = False
        self._stop.set()

    def snapshot(self) -> dict:
        return {
            "tag": self.tag,
            "counts": dict(self.counts),
            "bursting": self.bursting,
            "folder": str(self.dir),
            "camera": self.stream.info(),
            "sync": self.syncer.snapshot() if self.syncer else None,
        }


class Syncer:
    """Copies the output folder to a remote host with rsync, in a background thread."""

    def __init__(self, out: str | Path, dest: str, ssh_config: str = DEFAULT_SSH_CONFIG, interval: float = 10.0):
        host, sep, path = dest.partition(":")
        if not sep or not host:
            raise ValueError(f"sync destination must look like host:path, got {dest!r}")
        rel = path.removeprefix("~/")
        if rel.split("/")[0] != "desmond" or ".." in rel.split("/"):
            raise ValueError(f"sync destination must be under ~/desmond, got {path!r}")
        self.out, self.dest, self.host, self.remote_dir = Path(out), dest, host, rel.rstrip("/")
        self.ssh_config, self.interval = ssh_config, interval
        self.last_ok: float | None = None
        self.error: str | None = None
        self.busy = False
        self._made_dir = False
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def _ssh(self) -> list[str]:
        return ["ssh", "-F", self.ssh_config]

    def commands(self) -> list[list[str]]:
        cmds = []
        if not self._made_dir:
            cmds.append(self._ssh() + [self.host, "mkdir", "-p", self.remote_dir])
        cmds.append(["rsync", "-az", "-e", shlex.join(self._ssh()), f"{self.out}/", self.dest])
        return cmds

    def sync_once(self) -> bool:
        with self._lock:
            self.busy = True
            try:
                self.out.mkdir(parents=True, exist_ok=True)
                for cmd in self.commands():
                    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)
                    if r.returncode != 0:
                        msg = (r.stderr or r.stdout or "").strip().splitlines()
                        self.error = f"{cmd[0]} exit {r.returncode}" + (f": {msg[-1]}" if msg else "")
                        log.warning("sync failed: %s", self.error)
                        return False
                    if cmd[0] == "ssh":
                        self._made_dir = True
                self.last_ok, self.error = time.time(), None
                return True
            except (OSError, subprocess.SubprocessError) as e:
                self.error = str(e)
                log.warning("sync failed: %s", e)
                return False
            finally:
                self.busy = False

    def run(self) -> None:
        while not self._stop.is_set():
            self.sync_once()
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()

    def snapshot(self) -> dict:
        return {"dest": self.dest, "last_ok": self.last_ok, "error": self.error, "busy": self.busy}


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Inspect capture</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Barlow:wght@200;300;400;500;600&display=swap" rel="stylesheet">
<style>
:root{--bg:#15171a;--panel:#1d2024;--line:#2b2f35;--fg:#f1f2f4;--mute:#80868f;--dim:#4a4f57;--red:#e31937}
*{box-sizing:border-box;margin:0}
html,body{height:100%}
body{background:var(--bg);color:var(--fg);font:400 15px/1.35 Barlow,ui-sans-serif,system-ui,sans-serif;-webkit-font-smoothing:antialiased;overflow:hidden}
.app{display:grid;grid-template-columns:minmax(0,1fr) 340px;grid-template-rows:minmax(0,1fr) auto;height:100vh}
.view{padding:20px 0 12px 20px;min-height:0;display:flex}
.main{position:relative;flex:1;min-height:0;border-radius:18px;overflow:hidden;background:#000;outline:3px solid transparent;outline-offset:-3px;transition:outline-color .2s}
.main img{width:100%;height:100%;object-fit:contain;display:block}
.rec .main{outline-color:var(--red)}
.main.flash{outline-color:var(--fg);transition:none}
.side{padding:20px 20px 12px;display:flex;flex-direction:column;gap:14px;min-height:0}
.tag{display:flex;flex-direction:column;align-items:flex-start;gap:6px;padding:18px 20px;border-radius:16px;border:1px solid var(--line);background:var(--panel);color:var(--mute);font:400 15px Barlow,ui-sans-serif,system-ui,sans-serif;cursor:pointer;text-align:left;transition:border-color .2s,color .2s,background .2s}
.tag .n{font-weight:200;font-size:72px;line-height:1;letter-spacing:-.03em;color:var(--dim);font-variant-numeric:tabular-nums;transition:color .2s}
.tag kbd{font:500 12px Barlow,sans-serif;border:1px solid var(--line);border-radius:6px;padding:1px 7px;margin-right:8px;color:var(--fg)}
.tag.sel{border-color:var(--fg);color:var(--fg);background:#24282d}
.tag.sel .n{color:var(--fg)}
.tag:focus-visible,.burst:focus-visible{outline:2px solid var(--fg);outline-offset:2px}
.burst{margin-top:auto;padding:16px;border-radius:14px;border:1px solid var(--line);background:transparent;color:var(--fg);font:500 16px Barlow,ui-sans-serif,system-ui,sans-serif;cursor:pointer;transition:background .2s,border-color .2s}
.rec .burst{background:var(--red);border-color:var(--red)}
.burst kbd{font:500 12px Barlow,sans-serif;border:1px solid currentColor;border-radius:6px;padding:1px 7px;margin-left:10px;opacity:.7}
.status{grid-column:1/-1;padding:10px 20px 14px;color:var(--mute);font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.status .err{color:var(--red)}
@media (max-width:820px){body{overflow:auto}.app{grid-template-columns:minmax(0,1fr);height:auto}.view{padding:16px 16px 0}.main{flex:none;width:100%;aspect-ratio:4/3}.side{padding:14px 16px;display:grid;grid-template-columns:1fr 1fr;gap:12px}.tag{padding:14px 16px}.tag .n{font-size:52px}.burst{grid-column:1/-1;margin-top:0}.status{padding:4px 16px 16px;white-space:normal}}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style></head>
<body><div class="app" id="app">
<section class="view"><div class="main" id="main"><img id="cam" alt="Inspection camera" src="/stream/inspect"></div></section>
<aside class="side">
 <button class="tag" id="in_view"><span class="n" id="n_in_view">0</span><span><kbd>1</kbd>In view</span></button>
 <button class="tag" id="not_in_view"><span class="n" id="n_not_in_view">0</span><span><kbd>2</kbd>Not in view</span></button>
 <button class="burst" id="burst"><span id="blabel">Start burst</span><kbd>Space</kbd></button>
</aside>
<div class="status" id="status">Connecting</div>
</div>
<script>
const $=id=>document.getElementById(id),app=$('app');
const cap=c=>fetch('/cap/'+c,{method:'POST'}).then(tick);
let last=null;
$('in_view').onclick=()=>cap('in_view');$('not_in_view').onclick=()=>cap('not_in_view');$('burst').onclick=()=>cap('burst');
function flash(){const m=$('main');m.classList.add('flash');setTimeout(()=>m.classList.remove('flash'),120)}
document.addEventListener('keydown',e=>{if(e.repeat||e.metaKey||e.ctrlKey||e.altKey)return;const k=e.key.toLowerCase();
 if(k==='1')cap('in_view');else if(k==='2')cap('not_in_view');
 else if(k===' '){e.preventDefault();cap('burst')}else if(k==='s'){flash();cap('save')}});
function ago(t){const s=Math.max(0,Math.round(Date.now()/1000-t));return s<60?s+'s ago':Math.round(s/60)+'m ago'}
function esc(s){return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
async function tick(){try{
 const r=await fetch('/cap/state').then(x=>x.json());
 app.classList.toggle('rec',r.bursting);$('blabel').textContent=r.bursting?'Stop burst':'Start burst';
 for(const t of ['in_view','not_in_view']){$(t).classList.toggle('sel',r.tag===t);$('n_'+t).textContent=r.counts[t]}
 const c=r.camera,parts=[];
 parts.push(c.error?'<span class="err">Camera '+esc(c.error)+'</span>':'Camera '+(c.size||'waiting')+(c.fps?' at '+c.fps+' fps':''));
 const s=r.sync;
 if(s)parts.push(s.error?'<span class="err">Sync failed: '+esc(s.error)+'</span>':s.last_ok?'Synced '+ago(s.last_ok):'Syncing');
 else parts.push('Sync off');
 parts.push(esc(r.folder));
 $('status').innerHTML=parts.join(' · ');last=r;
}catch(e){$('status').textContent='Disconnected'}}
setInterval(tick,150);tick();
</script></body></html>"""


def make_handler(streams: dict, capture: Capture):
    base = teleop_web.make_handler(streams, None)
    page = PAGE.encode()
    commands = set(TAGS) | {"burst", "save"}

    class Handler(base):
        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(200, page, "text/html; charset=utf-8")
            if self.path == "/cap/state":
                return self._send(200, json.dumps(capture.snapshot()).encode(), "application/json")
            return super().do_GET()

        def do_POST(self):
            name = self.path.removeprefix("/cap/")
            if self.path.startswith("/cap/") and name in commands:
                capture.command(name)
                return self._send(200, b"{}", "application/json")
            return super().do_POST()

    return Handler


def main() -> None:
    from http.server import ThreadingHTTPServer

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", type=int, required=True, help="OpenCV index of the inspection camera")
    p.add_argument("--out", default="data/inspect_frames")
    p.add_argument("--burst-fps", type=float, default=5.0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8002)
    p.add_argument("--sync", help="rsync destination, e.g. gb10:desmond/inspect_frames/ (must be under ~/desmond)")
    p.add_argument("--ssh-config", default=DEFAULT_SSH_CONFIG)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    syncer = None
    if args.sync:
        try:
            syncer = Syncer(args.out, args.sync, args.ssh_config)
        except ValueError as e:
            p.error(str(e))

    stream = teleop_web.CameraStream("inspect", args.camera, args.width, args.height, 30, 80).start()
    streams = {"inspect": stream}
    capture = Capture(stream, args.out, args.burst_fps)
    capture.syncer = syncer
    threading.Thread(target=capture.run, name="capture", daemon=True).start()
    if syncer:
        threading.Thread(target=syncer.run, name="sync", daemon=True).start()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(streams, capture))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, name="http", daemon=True).start()
    log.info("capture UI: http://localhost:%d  saving to %s", args.port, capture.dir)
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        capture.stop()
        server.shutdown()
        stream.stop()
        log.info("saved %s", capture.counts)
        if syncer:
            syncer.stop()
            log.info("final sync to %s", syncer.dest)
            log.info("final sync %s", "ok" if syncer.sync_once() else f"failed: {syncer.error}")


if __name__ == "__main__":
    main()
