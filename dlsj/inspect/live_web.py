"""Local camera tester for a model trained in Anomalib Studio.

python -m dlsj.inspect.live_web --run-dir /path/to/training/run
"""

import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import urllib.error
import urllib.request
import uuid

import cv2
import numpy as np

PAGE = r'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Live brick inspection</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#101827;color:#f8fafc;font:16px system-ui}main{max-width:1120px;margin:auto;padding:24px}h1{margin:0 0 8px}p{color:#cbd5e1;line-height:1.5}button,select,input{font:inherit}button,select{padding:12px;border-radius:9px;border:1px solid #64748b;background:#25344b;color:white}button{cursor:pointer}button:disabled{opacity:.5}label{display:block;margin:14px 0 8px}.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}select{flex:1;min-width:180px}.layout{display:grid;grid-template-columns:2fr 1fr;gap:20px;margin-top:20px}.stage{position:relative;align-self:start;background:#030712;border-radius:12px;overflow:hidden}video,#remote{display:block;width:100%;min-height:160px}[hidden]{display:none!important}.stage canvas{position:absolute;inset:0;width:100%;height:100%;touch-action:none;cursor:crosshair}aside{background:#1e293b;border-radius:12px;padding:18px}#result{font-size:28px;font-weight:750;margin:10px 0}.normal{color:#86efac}.marked{color:#fda4af}.muted{color:#cbd5e1}#view{width:100%;background:#030712;border-radius:8px}#status{min-height:24px;margin-top:12px}small{color:#94a3b8;line-height:1.5}input[type=range]{width:180px}#score{font-variant-numeric:tabular-nums}@media(max-width:750px){.layout{grid-template-columns:1fr}}
</style><main><h1>Live brick inspection</h1>
<p id="instructions">Choose a camera and place one brick inside the square, with its marked face visible. Click the preview to move the square; adjust its size to fit the whole brick face.</p>
<div class="row"><select id="camera" aria-label="Camera"><option value="">Default laptop camera</option></select><button id="connect">Enable camera</button><button id="toggle" disabled>Start detection</button></div>
<label id="autoLabel" hidden><input id="auto" type="checkbox"> <span id="autoText">Auto-detect brick (crop preview)</span></label>
<div class="layout"><div><div class="stage"><video id="video" autoplay playsinline muted></video><img id="remote" alt="Intel XPS inspection camera" hidden><canvas id="overlay" aria-label="Click to position the inspection square"></canvas></div>
<div id="manualCropControls"><label for="size">Inspection square size</label><input id="size" type="range" min="15" max="95" value="60"><span id="sizeValue">60%</span></div></div>
<aside><small id="model">Loading model…</small><div id="result" class="muted" role="status">Waiting for camera</div><div id="score">Score —</div>
<p><small id="classification">YOLO: not loaded</small></p>
<p><small id="detail">Plain = normal · Cross or circle = marked</small></p><canvas id="view" width="320" height="320" aria-label="Last inspected crop with heatmap"></canvas>
<label><input id="heatmap" type="checkbox" checked> Show anomaly heatmap</label>
<label for="threshold">Decision threshold <span id="thresholdValue"></span></label><input id="threshold" type="range" min="0" max="1" step="any"><button id="reset">Reset threshold</button>
<p><small>Scores above the threshold are marked. The score is averaged over three frames; it is not a probability. Keep the brick steady while testing. With YOLO loaded, weak or conflicting predictions request reinspection.</small></p></aside></div>
<p id="evaluationNotice" hidden><small>Experimental crop ensemble: validation target not met. Check predictions against the actual brick.</small></p><div id="status" role="status">Connecting…</div><small>Inspect the whole face. Hidden marks and empty inspection squares are not reliable tests.</small>
<script>
const $=id=>document.getElementById(id);let stream=null,ready=false,running=false,inflight=false,switching=false,revision=0,center={x:.5,y:.5},scores=[],defaultThreshold=.683024,last=null,modelReady=false,classFrames=[],classifierSettings=null,remoteConfigured=false,remoteActive=false,remoteGeneration=0,remoteUpdated=0,detectorReady=false,detectedBox=null,previousBox=null,detectedBoxes=[];
function reset(message='Collecting frames…'){revision++;detectedBox=null;previousBox=null;detectedBoxes=[];scores=[];classFrames=[];last=null;$('classification').textContent=classifierSettings?'YOLO: waiting for three frames':'YOLO: not loaded';$('view').getContext('2d').clearRect(0,0,320,320);$('result').className='muted';$('result').textContent=message;$('score').textContent='Score —';$('detail').textContent='Plain = normal · Cross or circle = marked';}
function controls(){$('toggle').disabled=!ready||!modelReady||switching;$('connect').disabled=switching;$('camera').disabled=switching;$('toggle').textContent=running?'Pause detection':'Start detection';}
function frame(){return remoteActive?$('remote'):$('video');}
function dimensions(){const v=frame();return {width:remoteActive?v.naturalWidth:v.videoWidth,height:remoteActive?v.naturalHeight:v.videoHeight};}
function roi(){const {width,height}=dimensions(),side=Math.min(width,height)*Number($('size').value)/100;return {x:Math.max(0,Math.min(width-side,center.x*width-side/2)),y:Math.max(0,Math.min(height-side,center.y*height-side/2)),side};}
function drawBox(){
 if(!ready)return;const c=$('overlay'),d=dimensions();c.width=d.width;c.height=d.height;const ctx=c.getContext('2d');ctx.lineWidth=Math.max(3,c.width/300);
 if(!$('auto').checked){const r=roi();ctx.fillStyle='rgba(0,0,0,.3)';ctx.fillRect(0,0,c.width,c.height);ctx.clearRect(r.x,r.y,r.side,r.side);ctx.strokeStyle='#38bdf8';ctx.strokeRect(r.x,r.y,r.side,r.side);}
 for(const box of detectedBoxes){const [x0,y0,x1,y1]=box;ctx.strokeStyle='#94a3b8';ctx.strokeRect(x0,y0,x1-x0,y1-y0);}
 if(detectedBox){const [x0,y0,x1,y1]=detectedBox;ctx.strokeStyle='#86efac';ctx.strokeRect(x0,y0,x1-x0,y1-y0);}
}
function mode(){const auto=$('auto').checked;$('manualCropControls').hidden=auto;$('instructions').textContent=auto?'Brick boxes are detected automatically. If several bricks are visible, click the box of the brick you want to inspect. Present its marked face to the camera.':'Choose a camera and place one brick inside the square, with its marked face visible. Click the preview to move the square; adjust its size to fit the whole brick face.';}

async function enumerate(){const selected=remoteActive?'remote':stream?.getVideoTracks()[0]?.getSettings().deviceId;const devices=await navigator.mediaDevices.enumerateDevices();$('camera').replaceChildren();if(remoteConfigured)$('camera').add(new Option('Intel XPS inspection camera','remote'));$('camera').add(new Option('Default laptop camera',''));for(const [i,d]of devices.filter(d=>d.kind==='videoinput').entries())$('camera').add(new Option(d.label||'Camera '+(i+1),d.deviceId));if(selected)$('camera').value=selected;}
async function pollRemote(generation){
 if(!remoteActive||generation!==remoteGeneration)return;
 try{
  const response=await fetch('/camera/frame.jpg',{cache:'no-store',signal:AbortSignal.timeout(5000)});
  if(!response.ok)throw Error('Intel camera unavailable ('+response.status+')');
  const blob=await response.blob(),url=URL.createObjectURL(blob),image=new Image();
  try{image.src=url;await image.decode();if(!remoteActive||generation!==remoteGeneration)return;$('remote').src=url;await $('remote').decode();if(!remoteActive||generation!==remoteGeneration)return;remoteUpdated=performance.now();const first=!ready;ready=true;drawBox();if(first){$('result').textContent='Ready';$('status').textContent='Camera: Intel XPS inspection camera';}controls();}finally{URL.revokeObjectURL(url);}
 }catch(e){if(generation!==remoteGeneration)return;running=false;ready=false;reset('Camera disconnected');$('status').textContent=e.message+' · retrying';controls();}
 if(remoteActive&&generation===remoteGeneration)setTimeout(()=>pollRemote(generation),200);
}
async function camera(){if(switching)return;switching=true;running=false;ready=false;remoteGeneration++;reset('Connecting camera…');controls();try{const id=$('camera').value;stream?.getTracks().forEach(t=>{t.onended=null;t.stop();});stream=null;remoteActive=id==='remote';$('video').hidden=remoteActive;$('remote').hidden=!remoteActive;if(remoteActive){await pollRemote(remoteGeneration);}else{stream=await navigator.mediaDevices.getUserMedia({audio:false,video:{...(id?{deviceId:{exact:id}}:{}),width:{ideal:1280},height:{ideal:720}}});const track=stream.getVideoTracks()[0];track.onended=()=>{running=false;ready=false;reset('Camera disconnected');controls();};$('video').srcObject=stream;await $('video').play();await enumerate();ready=true;drawBox();$('result').textContent='Ready';$('status').textContent='Camera: '+track.label;}$('connect').textContent='Restart camera';}catch(e){$('status').textContent=e.name==='NotAllowedError'?'Allow camera access for localhost, then click Enable camera.':e.message;$('result').textContent='Camera unavailable';}finally{switching=false;controls();}}

function classify(){
 if(!scores.length)return;
 const mean=scores.reduce((a,b)=>a+b,0)/scores.length;
 $('score').textContent='Anomalib '+mean.toFixed(6)+' · '+scores.length+'/3 frames';
 if(scores.length<3){$('result').className='muted';$('result').textContent='Collecting '+scores.length+'/3';return;}
 const marked=mean>=Number($('threshold').value);
 if(classifierSettings){
  if(classFrames.length!==3){$('result').className='muted';$('result').textContent='Waiting for classifier';return;}
  const probabilities=Object.keys(classFrames[0]).map(k=>[k,classFrames.reduce((sum,p)=>sum+p[k],0)/3]).sort((a,b)=>b[1]-a[1]);
  const [label,confidence]=probabilities[0],margin=confidence-probabilities[1][1];
  $('classification').textContent='YOLO: '+label.toUpperCase()+' · '+(confidence*100).toFixed(1)+'% confidence';
  const reliable=confidence>=classifierSettings.min_confidence&&margin>=classifierSettings.min_margin;
  const agreement=marked?(label==='cross'||label==='circle'):label==='plain';
  if(!reliable||!agreement){$('result').className='muted';$('result').textContent='UNCERTAIN · INSPECT AGAIN';return;}
  $('result').className=marked?'marked':'normal';
  $('result').textContent={plain:'PLAIN / NORMAL',cross:'CROSS / DEFECTIVE',circle:'CIRCLE / MINOR DEFECT'}[label];
 }else{$('result').className=marked?'marked':'normal';$('result').textContent=marked?'MARKED / DEFECTIVE':'PLAIN / NORMAL';}
}
function render(){if(!last)return;const ctx=$('view').getContext('2d');ctx.clearRect(0,0,320,320);ctx.drawImage(last.crop,0,0,320,320);if($('heatmap').checked&&last.map)ctx.drawImage(last.map,0,0,320,320);}
async function tick(){
 let generation=revision;
 try{
  if(!running||!ready||inflight)return;
  inflight=true;generation=revision;const v=frame();
  if(remoteActive){if(performance.now()-remoteUpdated>2000)throw Error('Intel camera frame is stale');}else if(v.readyState<2)return;
  const r=roi(),crop=document.createElement('canvas');crop.width=crop.height=320;
  let result;
  if($('auto').checked){
   const full=document.createElement('canvas'),d=dimensions();full.width=d.width;full.height=d.height;full.getContext('2d').drawImage(v,0,0);
   const response=await fetch('/detect',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({frame:full.toDataURL('image/jpeg',.95).split(',')[1],zone:[0,0,d.width,d.height],previous_box:previousBox})});
   result=await response.json();if(!response.ok)throw Error(result.error||'Brick detection failed');if(generation!==revision||!running)return;
   if(!result.selected){reset(result.detections.length?'SELECT A BRICK':'NO BRICK DETECTED');detectedBoxes=result.detections.map(d=>d.box);drawBox();$('status').textContent=result.detections.length?'Click one detected brick to select it':'Present a brick to the camera';return;}
   const next=result.selected.box;
   if(previousBox&&!sameTarget(previousBox,next)){reset('Target changed');generation=revision;}
   previousBox=next;detectedBox=next;detectedBoxes=result.detections.map(d=>d.box);drawBox();
   const image=new Image();image.src='data:image/jpeg;base64,'+result.crop;await image.decode();if(generation!==revision||!running)return;
   crop.getContext('2d').drawImage(image,0,0);
   last={crop,map:null};render();
   if(!result.inspection_ready){$('result').textContent='BRICK CROP PREVIEW';$('status').textContent=result.inspection_reason||'Automatic crop calibration pending · anomaly decisions disabled';$('detail').textContent='Detector: '+result.latency_ms+' ms · exact model crop 320 × 320';return;}
  }else{
   crop.getContext('2d').drawImage(v,r.x,r.y,r.side,r.side,0,0,320,320);
   const response=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({frame:crop.toDataURL('image/jpeg',.95).split(',')[1]})});
   result=await response.json();if(!response.ok)throw Error(result.error||'Prediction failed');
  }
  if(generation!==revision||!running)return;
  if(classifierSettings){if(!result.class_probabilities)throw Error('YOLO prediction missing');classFrames.push(result.class_probabilities);if(classFrames.length>3)classFrames.shift();}
  scores.push(result.score);if(scores.length>3)scores.shift();last={crop,map:null};classify();
  $('detail').textContent='Last prediction: '+result.latency_ms+' ms · model input 320 × 320';$('status').textContent='Live detection running';render();
  const map=new Image();map.onload=()=>{if(generation===revision&&last?.crop===crop){last.map=map;render();}};map.src='data:image/png;base64,'+result.anomaly_map;
 }catch(e){if(generation===revision){reset('Prediction unavailable');running=false;$('status').textContent=e.message;controls();}}
 finally{inflight=false;setTimeout(tick,350);}
}
function sameTarget(a,b){const area=r=>(r[2]-r[0])*(r[3]-r[1]);const intersection=Math.max(0,Math.min(a[2],b[2])-Math.max(a[0],b[0]))*Math.max(0,Math.min(a[3],b[3])-Math.max(a[1],b[1]));return intersection/(area(a)+area(b)-intersection)>=.3;}
$('auto').onchange=()=>{mode();reset(running?'Collecting frames…':'Ready');drawBox();};

$('connect').onclick=camera;$('camera').onchange=camera;$('toggle').onclick=()=>{running=!running;reset(running?'Collecting frames…':'Paused');controls();};$('overlay').onpointerdown=e=>{const r=$('overlay').getBoundingClientRect(),x=(e.clientX-r.left)/r.width,y=(e.clientY-r.top)/r.height;if($('auto').checked){const d=dimensions(),boxes=detectedBoxes.slice(),matches=boxes.filter(b=>x*d.width>=b[0]&&x*d.width<=b[2]&&y*d.height>=b[1]&&y*d.height<=b[3]);reset(matches.length===1?'Tracking selected brick':'Select one brick');detectedBoxes=boxes;if(matches.length===1){previousBox=matches[0];detectedBox=matches[0];}}else{center={x,y};reset(running?'Collecting frames…':'Ready');}drawBox();};$('size').oninput=()=>{$('sizeValue').textContent=$('size').value+'%';reset(running?'Collecting frames…':'Ready');drawBox();};$('heatmap').onchange=render;$('threshold').oninput=()=>{$('thresholdValue').textContent=Number($('threshold').value).toFixed(6);classify();};$('reset').onclick=()=>{$('threshold').value=defaultThreshold;$('threshold').oninput();};
window.addEventListener('beforeunload',()=>stream?.getTracks().forEach(t=>t.stop()));
(async()=>{try{const r=await fetch('/health'),s=await r.json();if(!r.ok)throw Error(s.error);detectorReady=Boolean(s.detector);$('autoLabel').hidden=!detectorReady;$('auto').checked=detectorReady;$('autoText').textContent=s.auto_inspection_ready?'Automatically detect and inspect the brick':'Auto-detect brick (crop preview)';mode();remoteConfigured=Boolean(s.camera_source);if(remoteConfigured){$('camera').add(new Option('Intel XPS inspection camera','remote'));$('camera').value='remote';}classifierSettings=s.classifier||null;$('evaluationNotice').hidden=!(s.auto_inspection_ready&&classifierSettings?.calibration_target_met===false);defaultThreshold=s.threshold;$('threshold').value=defaultThreshold;$('threshold').oninput();$('model').textContent=s.model+(classifierSettings?' + '+classifierSettings.model:'');modelReady=true;controls();}catch(e){$('model').textContent='Model unavailable';$('status').textContent=e.message;}if(remoteConfigured||navigator.mediaDevices?.getUserMedia)await camera();else $('status').textContent='Open this page at http://localhost:8006 to enable your camera.';tick();})();
</script></main></html>'''


def decode_frame(payload):
    encoded = payload.get('frame') if isinstance(payload, dict) else None
    if not isinstance(encoded, str):
        raise ValueError('Send a JPEG inspection crop')
    frame = base64.b64decode(encoded, validate=True)
    image = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR) if frame else None
    if not frame.startswith(b'\xff\xd8') or image is None or image.shape[:2] != (320, 320):
        raise ValueError('The inspection crop must be a 320 × 320 JPEG')
    return frame, image


class StudioModel:
    def __init__(self, run_dir, studio_url='http://127.0.0.1:8005'):
        run_dir = Path(run_dir)
        meta = json.loads((run_dir / 'studio_model.json').read_text())
        evaluation = json.loads((run_dir / 'evaluation.json').read_text())
        if meta['id'] != evaluation['model_id']:
            raise ValueError('Model and calibration IDs do not match')
        self.threshold = evaluation['validation_selected_threshold']
        self.name = meta['name']
        if evaluation.get('profile'):
            self.name += f" · {evaluation['profile']} augmentation"
        self.model_id = meta['id']
        self.url = f"{studio_url.rstrip('/')}/api/projects/{meta['project_id']}/models/{meta['id']}"

    def health(self):
        with urllib.request.urlopen(self.url, timeout=5) as response:
            meta = json.load(response)
        if not meta.get('is_ready'):
            raise ValueError('Studio model is not ready')
        return {'model': self.name, 'model_id': self.model_id, 'threshold': self.threshold}

    def predict(self, payload):
        import time
        frame, image = decode_frame(payload)
        boundary = uuid.uuid4().hex
        body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="crop.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'.encode()
                + frame + f'\r\n--{boundary}--\r\n'.encode())
        request = urllib.request.Request(self.url + ':predict?device=CPU', data=body,
                                        headers={'Content-Type': f'multipart/form-data; boundary={boundary}'})
        start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=45) as response:
            result = json.load(response)
        return {'score': float(result['score']), 'anomaly_map': result['anomaly_map'],
                'latency_ms': round((time.perf_counter() - start) * 1000)}


class LocalModel(StudioModel):
    """Run the released Anomalib OpenVINO export without a Studio database."""

    def __init__(self, model_dir, device='CPU'):
        import openvino as ov

        super().__init__(model_dir)
        self.device = device
        model_dir = Path(model_dir)
        self.lock = threading.Lock()
        self.compiled = ov.Core().compile_model(
            str(model_dir / 'model.xml'), device,
            {'PERFORMANCE_HINT': 'LATENCY', 'CACHE_DIR': str(model_dir / 'runtime-cache')},
        )

    def health(self):
        return {'model': self.name, 'model_id': self.model_id,
                'threshold': self.threshold, 'device': self.device, 'backend': 'local OpenVINO'}

    def predict(self, payload):
        import time

        _, image = decode_frame(payload)
        # Match Anomalib's OpenVINOInferencer. Resize and ImageNet normalization
        # are already embedded in this export; its input is RGB float NCHW [0,1].
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])
        start = time.perf_counter()
        with self.lock:
            outputs = self.compiled([tensor])
            values = {port.any_name: value.copy() for port, value in outputs.items()}
        score = float(values['pred_score'].item())
        scaled = (values['anomaly_map'].squeeze() * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
        # OpenCV encodes BGRA as PNG RGBA, matching Studio's PIL heatmap.
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2BGRA)
        heatmap[:, :, 3] = scaled
        ok, encoded = cv2.imencode('.png', heatmap)
        if not ok:
            raise ValueError('Could not encode anomaly heatmap')
        return {'score': score, 'anomaly_map': base64.b64encode(encoded).decode(),
                'latency_ms': round((time.perf_counter() - start) * 1000)}


def fetch_camera_frame(source):
    with urllib.request.urlopen(source, timeout=3) as response:
        frame = response.read(5 * 1024 * 1024 + 1)
    if len(frame) > 5 * 1024 * 1024 or not frame.startswith(b'\xff\xd8'):
        raise ValueError('Camera did not return a JPEG')
    image = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Camera returned an invalid JPEG')
    return frame


class AutoCropService:
    """Stateless per-frame target selection; client supplies its own previous box."""
    def __init__(self, detector, model, inspection_ready=False, min_inspection_side=64):
        self.detector, self.model, self.inspection_ready = detector, model, inspection_ready
        self.min_inspection_side = min_inspection_side

    def predict(self, payload):
        import time
        from dataclasses import asdict
        from dlsj.inspect.brick_detector import inspection_crop, select_target
        start = time.perf_counter()
        if not isinstance(payload, dict) or not isinstance(payload.get('frame'), str):
            raise ValueError('Send a JPEG camera frame')
        data = base64.b64decode(payload['frame'], validate=True)
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None
        if not data.startswith(b'\xff\xd8') or frame is None or max(frame.shape[:2]) > 4096:
            raise ValueError('Send a JPEG camera frame up to 4096 pixels per side')
        h, w = frame.shape[:2]
        zone = payload.get('zone', [0, 0, w, h])
        previous = payload.get('previous_box')
        for box in [zone] + ([previous] if previous is not None else []):
            if (not isinstance(box, list) or len(box) != 4 or not all(isinstance(v, (int, float)) and np.isfinite(v) for v in box)
                    or not (0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h)):
                raise ValueError('Inspection area and previous box must be within the frame')
        detections = self.detector.detect(frame)
        selected = select_target(detections, zone, previous)
        result = {'detections': [asdict(d) for d in detections], 'selected': asdict(selected) if selected else None,
                  'inspection_ready': self.inspection_ready}
        if selected:
            x0, y0, x1, y1 = selected.box
            if min(x1-x0, y1-y0) < self.min_inspection_side:
                result['inspection_ready'] = False
                result['inspection_reason'] = 'Move the brick closer for a readable inspection crop'
            crop, bounds = inspection_crop(frame, selected.box)
            ok, encoded = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                raise ValueError('Could not encode inspection crop')
            result.update(crop=base64.b64encode(encoded).decode(), crop_box=bounds)
            if result['inspection_ready']:
                result.update(self.model.predict({'frame': result['crop']}))
        result['latency_ms'] = round((time.perf_counter()-start)*1000)
        return result


def make_handler(model, camera_source=None, auto_crop=None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, status, data, content_type='application/json'):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def error(self, exc):
            self.send(503, json.dumps({'error': f'Prediction service unavailable: {exc}. Check Anomalib Studio at localhost:8005.'}).encode())

        def do_GET(self):
            if self.path == '/':
                self.send(200, PAGE.encode(), 'text/html; charset=utf-8')
            elif self.path == '/camera/frame.jpg' and camera_source:
                try:
                    self.send(200, fetch_camera_frame(camera_source), 'image/jpeg')
                except (OSError, ValueError) as exc:
                    self.send(503, json.dumps({'error': f'Camera unavailable: {exc}'}).encode())
            elif self.path == '/health':
                try:
                    self.send(200, json.dumps({**model.health(), 'camera_source': bool(camera_source), 'detector': bool(auto_crop), 'auto_inspection_ready': bool(auto_crop and auto_crop.inspection_ready)}).encode())
                except (OSError, ValueError) as exc:
                    self.error(exc)
            else:
                self.send(404, b'{"error":"Not found"}')

        def do_POST(self):
            if self.path not in ('/predict', '/detect') or (self.path == '/detect' and auto_crop is None):
                return self.send(404, b'{"error":"Not found"}')
            try:
                length = int(self.headers.get('Content-Length', 0))
                if not 0 < length <= 3 * 1024 * 1024:
                    raise ValueError('Send one JPEG frame or crop (maximum 3 MB)')
                predictor = auto_crop if self.path == '/detect' else model
                result = predictor.predict(json.loads(self.rfile.read(length)))
                self.send(200, json.dumps(result).encode())
            except ValueError as exc:
                self.send(400, json.dumps({'error': str(exc)}).encode())
            except OSError as exc:
                self.error(exc)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--run-dir', help='Studio run with matching model/calibration metadata')
    source.add_argument('--model-dir', help='Extracted release with model.xml, model.bin and metadata')
    parser.add_argument('--device', default='CPU', help='OpenVINO device for --model-dir (default CPU)')
    parser.add_argument('--classifier-dir', help='Optional exported YOLO directory containing classifier.json')
    parser.add_argument('--studio-url', default='http://127.0.0.1:8005')
    parser.add_argument('--detector-weights', help='Custom one-class YOLO brick detection weights or OpenVINO directory')
    parser.add_argument('--detector-device', default='cpu')
    parser.add_argument('--min-inspection-side', type=int, default=64, help='Minimum short box side in source pixels for anomaly decisions')
    parser.add_argument('--auto-inspection-calibrated', action='store_true', help='Enable automatic-crop decisions only after validating models/thresholds on matching crops')
    parser.add_argument('--auto-inspection', action='store_true', help='Enable the existing PatchCore/YOLO ensemble on detected crops for live testing; does not assert calibration quality')
    parser.add_argument('--camera-source', help='Remote HTTP JPEG snapshot URL (proxied locally)')
    parser.add_argument('--port', type=int, default=8006)
    args = parser.parse_args()
    model = LocalModel(args.model_dir, args.device) if args.model_dir else StudioModel(args.run_dir, args.studio_url)
    if args.classifier_dir:
        from dlsj.inspect.yolo_classifier import CombinedModel, YoloClassifier
        model = CombinedModel(model, YoloClassifier(args.classifier_dir, args.device))
    auto_crop = None
    if args.detector_weights:
        from dlsj.inspect.brick_detector import BrickDetector
        auto_crop = AutoCropService(BrickDetector(args.detector_weights, args.detector_device), model, args.auto_inspection or args.auto_inspection_calibrated, args.min_inspection_side)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(model, args.camera_source, auto_crop))
    print(f'Live inspection: http://localhost:{args.port} · {model.name}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
