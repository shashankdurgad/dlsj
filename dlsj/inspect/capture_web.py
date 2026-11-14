"""Browser anomaly-data capture from a selectable laptop or USB camera.

python -m dlsj.inspect.capture_web --out ~/inference/inspect_data/real
The browser requests camera permission; frames are saved by the local server.
"""

import argparse
import base64
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
import urllib.request
import uuid

import cv2
import numpy as np

CATEGORIES = {"good": "plain", "bad": "cross-marked", "circle": "circle-marked"}

PAGE = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Brick anomaly data</title><style>
*{box-sizing:border-box}body{margin:0;background:#111827;color:#f9fafb;font:17px system-ui}
main{max-width:1050px;margin:auto;padding:24px}h1{font-size:28px;margin:0 0 8px}
p{color:#cbd5e1;line-height:1.5}.camera{background:#030712;border-radius:16px;overflow:hidden}
video{width:100%;max-height:65vh;object-fit:contain;display:block}.buttons{display:flex;gap:16px;margin-top:20px}
label{display:block;margin:16px 0 8px}.selector{display:flex;gap:12px;margin-bottom:16px}
select{flex:1;min-width:0;padding:12px;font:17px system-ui;border-radius:10px}
#connect{flex:0 0 auto;padding:12px;font-size:16px;background:#334155;color:white}
button{flex:1;padding:20px;border:0;border-radius:12px;font:600 20px system-ui;cursor:pointer}
#circle{background:#93c5fd;color:#172554}
#good{background:#86efac;color:#052e16}#bad{background:#fda4af;color:#4c0519}
button:disabled{opacity:.45;cursor:wait}#deletePhoto,#undo{flex:0 0 auto;background:#334155;color:white;font-size:16px}
#message{min-height:28px;margin-top:16px}small{display:block;color:#94a3b8;overflow-wrap:anywhere}
@media(max-width:650px){.buttons{flex-wrap:wrap}button{min-width:40%}}
</style><main><h1>Brick anomaly data</h1>
<p>Hold the brick in front of the selected camera, with its marked face visible. Each capture saves three views.
Use different bricks and orientations. Plain images train the detector; crossed images are for evaluation. Circle-marked images are saved separately.</p>
<label for="camera">Camera on this laptop</label><div class="selector">
<select id="camera" aria-label="Camera"><option value="">Default laptop camera</option></select>
<button id="connect">Enable camera</button></div>
<div class="camera"><video id="preview" autoplay playsinline muted aria-label="Live camera preview"></video></div>
<div class="buttons"><button id="good">1 · Plain brick <span id="goodCount">0</span></button>
<button id="bad">2 · Cross-marked brick <span id="badCount">0</span></button>
<button id="circle">3 · Circle-marked brick <span id="circleCount">0</span></button>
<button id="deletePhoto">Delete last photo</button>
<button id="undo">Delete last brick (3 views)</button></div><div id="message" role="status">Connecting…</div>
<small id="folder"></small><small>Full inspection frames are saved. Select the crop when training.</small>
<script>
const $=id=>document.getElementById(id);let busy=false,switching=false,stream=null,ready=false;
function controls(){for(const k of ['good','bad','circle'])$(k).disabled=busy||!ready;
$('camera').disabled=busy||switching;$('connect').disabled=busy||switching;}
async function state(){const r=await fetch('/api/state');const s=await r.json();
for(const k of ['good','bad','circle'])$(k+'Count').textContent='('+s.counts[k]+')';
$('folder').textContent='Saving to: '+s.folder;$('undo').disabled=busy||!s.can_undo;$('deletePhoto').disabled=busy||!s.can_delete_photo;}
async function nextFrame(){const v=$('preview');if(v.requestVideoFrameCallback)
await Promise.race([new Promise(resolve=>v.requestVideoFrameCallback(resolve)),
new Promise((_,reject)=>setTimeout(()=>reject(Error('Camera stopped producing frames')),3000))]);
else await new Promise(resolve=>setTimeout(resolve,180));}
async function frames(){const v=$('preview'),canvas=document.createElement('canvas');
if(!ready||v.readyState<2||!v.videoWidth)throw Error('Enable a camera before capturing');
canvas.width=v.videoWidth;canvas.height=v.videoHeight;const ctx=canvas.getContext('2d'),images=[];
for(let i=0;i<3;i++){await nextFrame();if(!ready)throw Error('Camera disconnected');
ctx.drawImage(v,0,0);images.push(canvas.toDataURL('image/jpeg',0.95).split(',')[1]);}
return {frames:images,camera:stream.getVideoTracks()[0].label||'Laptop camera'};}
async function command(path){if(busy)return;busy=true;controls();$('undo').disabled=true;$('deletePhoto').disabled=true;
$('message').textContent='Saving…';try{const options={method:'POST'};
if(path.startsWith('/capture/')){options.headers={'Content-Type':'application/json'};options.body=JSON.stringify(await frames());}
const r=await fetch(path,options);const s=await r.json();
if(!r.ok)throw Error(s.error);$('message').textContent=s.message;}catch(e){$('message').textContent=e.message;}
finally{busy=false;controls();await state();}}
$('good').onclick=()=>command('/capture/good');$('bad').onclick=()=>command('/capture/bad');
$('circle').onclick=()=>command('/capture/circle');
$('deletePhoto').onclick=()=>command('/delete-photo');
$('undo').onclick=()=>command('/undo');document.addEventListener('keydown',e=>{
if(['INPUT','SELECT'].includes(e.target.tagName)||e.ctrlKey||e.metaKey||e.altKey||e.repeat)return;
if(e.key==='1')command('/capture/good');if(e.key==='2')command('/capture/bad');if(e.key==='3')command('/capture/circle');});
async function cameras(){const devices=await navigator.mediaDevices.enumerateDevices();
const chosen=stream?.getVideoTracks()[0]?.getSettings().deviceId||$('camera').value;
$('camera').replaceChildren();for(const [i,d] of devices.filter(d=>d.kind==='videoinput').entries()){
const option=new Option(d.label||'Camera '+(i+1),d.deviceId);$('camera').add(option);}
if(chosen)$('camera').value=chosen;}
async function startCamera(){if(busy||switching)return;switching=true;ready=false;controls();
$('message').textContent='Allow camera access in your browser…';
try{const id=$('camera').value;if(stream)stream.getTracks().forEach(t=>t.stop());
stream=await navigator.mediaDevices.getUserMedia({audio:false,video:{
...(id?{deviceId:{exact:id}}:{}),width:{ideal:1280},height:{ideal:720}}});
const track=stream.getVideoTracks()[0];track.onended=()=>{ready=false;controls();$('message').textContent='Camera disconnected. Choose a camera and enable it again.';};
$('preview').srcObject=stream;await $('preview').play();await cameras();ready=true;
$('message').textContent='Ready · '+track.label;$('connect').textContent='Restart camera';
}catch(e){$('message').textContent=e.name==='NotAllowedError'?'Camera access denied. Allow camera access for localhost in your browser, then click Enable camera.':e.name==='NotFoundError'?'No camera found. Connect a camera and try again.':e.message;}
finally{switching=false;controls();}}
$('connect').onclick=startCamera;$('camera').onchange=startCamera;
window.addEventListener('beforeunload',()=>stream?.getTracks().forEach(t=>t.stop()));
controls();state();if(navigator.mediaDevices?.getUserMedia){
navigator.mediaDevices.addEventListener('devicechange',()=>cameras().catch(()=>{}));startCamera();
}else $('message').textContent='Open this dashboard at http://localhost:8003 to use your camera.';
</script></main></html>"""


class CaptureStore:
    def __init__(self, source, out, fetch=None):
        self.source, self.out = source, Path(out).expanduser().resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        for kind in CATEGORIES:
            (self.out / kind).mkdir(exist_ok=True)
        self.fetch = fetch or self.fetch_frame
        self.lock = threading.Lock()
        self.history = []

    def fetch_frame(self):
        with urllib.request.urlopen(self.source, timeout=3) as response:
            return response.read()

    def snapshot(self):
        with self.lock:
            counts = {kind: len(list((self.out / kind).glob("*_f0.jpg"))) for kind in CATEGORIES}
            return {"counts": counts, "folder": str(self.out), "can_undo": bool(self.history),
                    "can_delete_photo": bool(self._photos())}

    def save(self, kind):
        if kind not in CATEGORIES:
            raise ValueError("Choose plain, cross-marked, or circle-marked")
        with self.lock:
            frames, previous = [], None
            deadline = time.monotonic() + 5
            while len(frames) < 3 and time.monotonic() < deadline:
                data = self.fetch()
                image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("Inspection feed did not return a valid image")
                if data != previous:
                    frames.append(data)
                    previous = data
                if len(frames) < 3:
                    time.sleep(0.1)
            if len(frames) != 3:
                raise ValueError("Camera frames are frozen; nothing saved")
            return self._write_frames(kind, frames, "teleop-inspect-snapshots", "inspect", self.source)

    def save_upload(self, kind, payload):
        if kind not in CATEGORIES:
            raise ValueError("Choose plain, cross-marked, or circle-marked")
        encoded = payload.get("frames") if isinstance(payload, dict) else None
        if not isinstance(encoded, list) or len(encoded) != 3:
            raise ValueError("A capture needs exactly three camera frames")
        if any(not isinstance(frame, str) for frame in encoded):
            raise ValueError("Invalid camera frames")
        frames = [base64.b64decode(frame, validate=True) for frame in encoded]
        camera = payload.get("camera", "Laptop camera")
        if not isinstance(camera, str) or len(camera) > 500:
            raise ValueError("Invalid camera name")
        with self.lock:
            return self._write_frames(kind, frames, "browser-camera", camera)

    def _write_frames(self, kind, frames, source, camera, source_url=None):
        shapes = []
        for data in frames:
            image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None
            if image is None:
                raise ValueError("Invalid camera image; nothing saved")
            shapes.append(image.shape[:2])
        if len(set(shapes)) != 1:
            raise ValueError("Camera changed resolution during capture; nothing saved")
        key = datetime.now(timezone.utc).strftime("cap%Y%m%dT%H%M%S") + uuid.uuid4().hex[:8]
        paths = [self.out / kind / f"{key}_f{i}.jpg" for i in range(3)]
        capture_meta = {"source": source, "camera": camera, "source_url": source_url,
                        "frame_size": list(reversed(shapes[0])), "views": 3, "source_format": "jpeg"}
        paths.append(self.out / kind / f"{key}.json")
        try:
            for path, data in zip(paths[:3], frames):
                path.write_bytes(data)
            paths[-1].write_text(json.dumps(capture_meta, indent=2) + "\n")
            meta_path = self.out / "dataset.json"
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"crop": None}
            meta.update(capture_meta)
            temp = meta_path.with_suffix(".tmp")
            temp.write_text(json.dumps(meta, indent=2) + "\n")
            temp.replace(meta_path)
        except Exception:
            for path in paths:
                path.unlink(missing_ok=True)
            raise
        self.history.append(paths)
        return f"Saved {CATEGORIES[kind]} brick · 3 views"

    def _photos(self):
        return [path for kind in CATEGORIES for path in (self.out / kind).glob("*_f*.jpg")]

    def delete_photo(self):
        with self.lock:
            photos = self._photos()
            if not photos:
                return "No photos to delete"
            latest = max(photos, key=lambda p: (p.stat().st_mtime_ns, p.name))
            key = latest.stem.rsplit("_f", 1)[0]
            remaining = [p for p in latest.parent.glob(f"{key}_f*.jpg") if p != latest]
            meta_path = latest.parent / f"{key}.json"
            if remaining and meta_path.exists():
                meta = json.loads(meta_path.read_text())
                meta["views"] = len(remaining)
                temp = meta_path.with_suffix(".tmp")
                temp.write_text(json.dumps(meta, indent=2) + "\n")
                temp.replace(meta_path)
            latest.unlink()
            if not remaining:
                meta_path.unlink(missing_ok=True)
                self.history = [paths for paths in self.history if meta_path not in paths]
            return f"Deleted last photo · {CATEGORIES[latest.parent.name]} brick · {len(remaining)} views remain"

    def undo(self):
        with self.lock:
            if not self.history:
                return "Nothing to undo in this session"
            for path in self.history.pop():
                path.unlink(missing_ok=True)
            return "Removed the last capture · 3 views"


def make_handler(store):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, status, data, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def json(self, status, data):
            self.send(status, json.dumps(data).encode(), "application/json")

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/":
                self.send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif path == "/api/state":
                self.json(200, store.snapshot())
            elif path == "/frame.jpg":
                try:
                    self.send(200, store.fetch(), "image/jpeg")
                except (OSError, ValueError) as exc:
                    self.json(503, {"error": str(exc)})
            else:
                self.json(404, {"error": "Not found"})

        def do_POST(self):
            try:
                if self.path in tuple(f"/capture/{kind}" for kind in CATEGORIES):
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 16 * 1024 * 1024:
                        return self.json(400, {"error": "Send three camera frames (maximum 16 MB)"})
                    payload = json.loads(self.rfile.read(length))
                    message = store.save_upload(self.path.rsplit("/", 1)[1], payload)
                elif self.path == "/delete-photo":
                    message = store.delete_photo()
                elif self.path == "/undo":
                    message = store.undo()
                else:
                    return self.json(404, {"error": "Not found"})
                self.json(200, {"message": message})
            except (OSError, ValueError) as exc:
                self.json(503, {"error": str(exc)})
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="http://127.0.0.1:8000/snapshot/inspect.jpg")
    parser.add_argument("--out", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8003)
    args = parser.parse_args()
    store = CaptureStore(args.source, args.out)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(store))
    print(f"Anomaly capture: http://{args.host}:{args.port} -> {store.out}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
