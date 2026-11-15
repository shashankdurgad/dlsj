"""Rig data collection for the YOLO brick classifier (plain / circle / cross).

Every spin already saves its 12 classifier crops (320x320, exactly what YOLO sees), full frames and per-frame
predictions under `outputs/auto_inspect/<time>_aN/` (`auto_inspect.AutoInspector._save`). Collecting = recording the
TRUE label of each saved spin, plus per-frame exclusions (a marked brick's mark is not visible on every face).

Store: `~/inference/datasets/rig_yolo/collect.json` = {"entries": [{id, t, label, dirs, pred, excluded: [i...]}]}.
Review page at /collect on the loop dashboard: click a crop to exclude / include it, relabel or delete a spin.
`build_dataset()` turns the store into train/val/test/<label>/ folders split BY SPIN (no spin crosses splits).
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import threading
import time
from pathlib import Path

LABELS = ("plain", "circle", "cross")
ROOT = Path("~/inference/datasets/rig_yolo").expanduser()


class Store:
    def __init__(self, root: Path | str = ROOT):
        self.root = Path(root).expanduser()
        self.path = self.root / "collect.json"
        self.lock = threading.Lock()
        self.data = {"entries": []}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                self.path.replace(self.path.with_suffix(".json.corrupt"))

    def _write(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=1) + "\n")
        tmp.replace(self.path)

    def add(self, label: str, dirs: list[str], pred: str | None = None, source: str = "") -> dict:
        if label not in LABELS:
            raise ValueError(f"label must be one of {LABELS}")
        dirs = [str(Path(d).expanduser().resolve()) for d in dirs if d]
        with self.lock:
            e = {"id": hashlib.sha1(("|".join(dirs) + str(time.time())).encode()).hexdigest()[:10],
                 "t": round(time.time(), 1), "label": label, "dirs": dirs, "pred": pred, "source": source,
                 "excluded": []}
            self.data["entries"].append(e)
            self._write()
            return e

    def update(self, entry_id: str, *, toggle: int | None = None, label: str | None = None,
               delete: bool = False) -> None:
        with self.lock:
            es = self.data["entries"]
            for e in es:
                if e["id"] != entry_id:
                    continue
                if delete:
                    es.remove(e)
                elif label is not None:
                    if label not in LABELS:
                        raise ValueError(label)
                    e["label"] = label
                elif toggle is not None:
                    ex = set(e.get("excluded", []))
                    ex.symmetric_difference_update({int(toggle)})
                    e["excluded"] = sorted(ex)
                self._write()
                return
            raise KeyError(entry_id)

    def counts(self) -> dict:
        out = {k: {"spins": 0, "crops": 0} for k in LABELS}
        for e in self.data["entries"]:
            n = sum(len(crops(d)) for d in e["dirs"]) - len(e.get("excluded", []))
            out[e["label"]]["spins"] += 1
            out[e["label"]]["crops"] += max(0, n)
        return out

    def view(self, limit: int = 60) -> list[dict]:
        """Newest first, with crop file names and the model's per-frame guess."""
        out = []
        for e in reversed(self.data["entries"][-limit:]):
            frames = []
            k = 0
            for d in e["dirs"]:
                preds = per_frame(d)
                for c in crops(d):
                    i = int(c.stem.split("_")[1])
                    p = preds.get(i, {})
                    frames.append({"k": k, "dir": d, "file": c.name, "pred": p.get("mark") or
                                   ("plain" if p.get("label") == "perfect" else p.get("label")),
                                   "conf": p.get("score")})
                    k += 1
            out.append({**e, "frames": frames})
        return out


def crops(d) -> list[Path]:
    return sorted(Path(d).glob("crop_*.jpg"))


def per_frame(d) -> dict:
    try:
        j = json.loads((Path(d) / "sweep.json").read_text())
        return {int(f["i"]): f for f in (j.get("inspection") or {}).get("per_frame", [])}
    except Exception:  # noqa: BLE001
        return {}


def build_dataset(out: Path | str, store: Store | None = None, split=(0.7, 0.15, 0.15), seed: int = 2026) -> dict:
    """train/val/test/<label>/ crops, split by spin (stratified per label), excluded crops left out."""
    store = store or Store()
    out = Path(out).expanduser()
    if out.exists():
        raise FileExistsError(f"{out} exists: use a new folder")
    rng = random.Random(seed)
    manifest = {"split_unit": "spin", "seed": seed, "images": []}
    for label in LABELS:
        es = [e for e in store.data["entries"] if e["label"] == label]
        rng.shuffle(es)
        n = len(es)
        n_tr, n_va = round(n * split[0]), round(n * split[1])
        for j, e in enumerate(es):
            part = "train" if j < n_tr else "val" if j < n_tr + n_va else "test"
            k = 0
            for d in e["dirs"]:
                for c in crops(d):
                    if k not in e.get("excluded", []):
                        dst = out / part / label / f"{e['id']}_{k:02d}.jpg"
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(c, dst)
                        manifest["images"].append({"spin": e["id"], "split": part, "label": label,
                                                   "path": str(dst), "source": str(c)})
                    k += 1
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Rig data review</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#121417;--panel:#1b1e22;--line:#2c3137;--fg:#eef0f2;--mute:#8a919a;--ok:#3ecf8e;--red:#e5484d}
*{box-sizing:border-box;margin:0;padding:0}body{background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif;padding:16px}
h1{font-size:18px;margin-bottom:4px}.mute{color:var(--mute)}.counts{margin:8px 0 16px;display:flex;gap:16px;flex-wrap:wrap}
.spin{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:10px;margin-bottom:12px}
.head{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
.lab{font-weight:600;padding:2px 10px;border-radius:99px;background:#2a2f36}
button{font:inherit;color:var(--fg);background:#22262b;border:1px solid var(--line);border-radius:8px;padding:4px 10px;cursor:pointer}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(110px,1fr));gap:6px}
.tile{position:relative;cursor:pointer;border-radius:8px;overflow:hidden;border:2px solid transparent}
.tile img{width:100%;display:block;aspect-ratio:1}.tile span{position:absolute;left:0;right:0;bottom:0;font-size:11px;background:rgba(0,0,0,.65);padding:2px 4px}
.tile.ex{opacity:.25;border-color:var(--red)}.tile.ex::after{content:"excluded";position:absolute;top:4px;left:4px;font-size:11px;background:var(--red);padding:1px 5px;border-radius:4px}
.tile.bad span{color:#ffb4b4}
</style></head><body>
<h1>Rig data for the YOLO classifier</h1>
<div class="mute">Click a crop to exclude it (e.g. a circle/cross brick whose mark is not visible on that face). Red text = the current model disagrees with the label.</div>
<div class="counts" id="counts"></div><div id="list"></div>
<script>
const $=id=>document.getElementById(id);
async function post(p){await fetch(p,{method:'POST'});load()}
async function load(){const r=await fetch('/collect/state').then(x=>x.json());
 $('counts').innerHTML=Object.entries(r.counts).map(([k,v])=>`<span><b>${k}</b>: ${v.spins} spins, ${v.crops} crops</span>`).join('');
 $('list').innerHTML=r.entries.map(e=>`<div class="spin"><div class="head"><span class="lab">${e.label}</span>
  <span class="mute">${new Date(e.t*1000).toLocaleTimeString()} · model said ${e.pred||'?'} · ${e.frames.length-e.excluded.length}/${e.frames.length} crops used</span>
  ${['plain','circle','cross'].filter(l=>l!==e.label).map(l=>`<button onclick="post('/collect/label/${e.id}/${l}')">relabel ${l}</button>`).join('')}
  <button onclick="if(confirm('Delete this spin from the dataset? (images stay on disk)'))post('/collect/delete/${e.id}')">delete</button></div>
  <div class="tiles">${e.frames.map(f=>`<div class="tile${e.excluded.includes(f.k)?' ex':''}${f.pred&&f.pred!==e.label?' bad':''}" onclick="post('/collect/toggle/${e.id}/${f.k}')">
   <img loading="lazy" src="/collect/img?d=${encodeURIComponent(f.dir)}&f=${f.file}"><span>${f.pred||'?'} ${f.conf!=null?(+f.conf).toFixed(2):''}</span></div>`).join('')}</div></div>`).join('')||'<p class="mute">No labelled spins yet.</p>'}
load();setInterval(load,4000);
</script></body></html>"""
