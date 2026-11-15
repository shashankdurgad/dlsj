"""Fan twin jobs (sweeps, render batches) out over Modal GPUs, one container per job, results back on the box.

    U="uv run --no-project --with modal --with pyyaml python sim/twin/modal/fanout.py"   # MODAL_PROFILE=desmond-zee
    $U run --cmd "sim/twin/run.sh sim/twin/grasp_gate/gate.py --rung 3 --set gripper.finray.effort_nm={0.4,1.2}" \
        --grid seed=0,1 --dry-run          # 4 jobs ({a,b} inline lists and --grid key=... -> {key} expand, cartesian)
    $U run --jobs jobs.yaml                # [{name, cmd, gpu?, timeout?, cpu?, memory?, env?}, ...]
    $U data                                # make sure the data Volumes are complete (run does this too)
    $U stop                                # stop any fan-out app still running (run does this at its end)

run: 1. snapshots the box's working tree (default watson:~/Documents/dlsj; tracked + untracked, git-ignored files
excluded, files > --max-file-mb dropped) + small inputs from sim/twin/out (< 2 MB json / npz / npy / pt / ...,
sim/twin/out/scene, visual_fit/wood, visuals/top_median.png, and any --include path) into Volume dlsj-twin-fanout;
2. checks the data Volumes (dlsj-twin-data: cal2 recordings, post-train v3.1, MolmoAct2 8k fine-tune, station photos;
dlsj-twin-hf: base MolmoAct2 snapshots) and fetches what is missing (HF downloads inside Modal, CPU containers);
3. runs every job in its own container (default RTX-PRO-6000, 8 cores, 32 GiB, --timeout 1800 s), at most
--max-containers (4; up to 8 with --wide) at once. `sim/twin/run.sh X.py args` and `X.py args` run as
`isaaclab.sh -p X.py args` (as run.sh / app.py do), `python ...` in the Isaac venv, anything else through bash. The
repo layout is the box's (sim/twin/data/* are symlinks to the Volumes, HF_HOME = the HF Volume, offline);
4. every file the job wrote under sim/twin/out + its stdout land in dlsj-twin-fanout:/runs/<run>/<job>/ and are copied
to <box>:~/Documents/dlsj/sim/twin/out/modal_fanout/<run>/<job>/ (out/<path under sim/twin/out>, stdout.log,
job.json); a summary table (job, exit code, wall, key result lines) is printed and saved there too.
Safety: cost estimate before launch (refused above $15 without --yes-cost), --dry-run, and the app is stopped at the
end, on errors and on Ctrl-C / SIGTERM (then every fan-out app still listed as running is stopped). Each run appends
its GPU-minutes and estimated cost to <box>:~/Documents/dlsj/sim/twin/out/modal_fanout/usage.csv.
"""
from __future__ import annotations

import os

if "MODAL_TASK_ID" not in os.environ:  # local client: the team workspace, never the CLI's active profile
    if os.environ.get("MODAL_PROFILE", "desmond-zee") != "desmond-zee":
        raise SystemExit(f"MODAL_PROFILE={os.environ['MODAL_PROFILE']}: fan-out runs bill desmond-zee only")
    os.environ["MODAL_PROFILE"] = "desmond-zee"

import argparse
import itertools
import json
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
APP = "dlsj-twin-fanout"
WORK = "/tmp/repo"
ISAAC_SH = "/workspace/isaaclab/isaaclab.sh"
VENV_PY = "/opt/isaaclab-venv/bin/python"
CACHE, HF, DATA, FAN = "/cache", "/hf", "/data", "/fan"
EULA = {"ACCEPT_EULA": "Y", "PRIVACY_CONSENT": "N", "OMNI_KIT_ACCEPT_EULA": "YES"}
PRIVACY_TOML = "[privacy]\nperformance = false\npersonalization = false\nusage = false\n"
TELEMETRY = "omni.telemetry.transmitter"
SCRATCH = Path(os.environ.get("DLSJ_FANOUT_TMP", "/private/tmp/claude-501/-Users-desmondzee-Hardware-dlsj/"
                              "4698d3ab-97f9-4aa3-a532-a5fb532f6b27/scratchpad/fanout"))
SSH = ["perl", "-e", "alarm 900; exec @ARGV", "ssh", "-F", os.path.expanduser("~/.brev/ssh_config"), "-o",
       "BatchMode=yes", "-o", "LogLevel=ERROR"]
REMOTE_REPO = "~/Documents/dlsj"
HF_SECRET = os.environ.get("DLSJ_FANOUT_HF_SECRET", "hf-akoniti")

# Modal list prices (https://modal.com/pricing, read 2026-11-14), USD per second. Other GPUs: pass --gpu-price.
GPU_PRICE = {"RTX-PRO-6000": 0.000842, "L40S": 0.000542, "B200": 0.001736}
CPU_PRICE, MEM_PRICE = 0.0000131, 0.00000222  # per physical core / s, per GiB / s
COST_LIMIT = 15.0
MAX_DEFAULT, MAX_HARD = 4, 8

# Data the twin scripts read under sim/twin/data/ (Volume dlsj-twin-data, same layout) or HF_HOME (dlsj-twin-hf).
HF_DATA = {
    "canonical/duplo_teleop_cal2_rgb_456": dict(repo="akoniti/dlsj-canonical", type="dataset", local="canonical",
                                                allow=["duplo_teleop_cal2_rgb_456/*"]),
    "posttrain_v3_1": dict(repo="akoniti/dlsj-posttrain-v3-1", type="dataset", local="posttrain_v3_1"),
    "models/molmoact2_ft_8k": dict(repo="akoniti/dlsj-molmoact2-ft-8k", type="model", local="models/molmoact2_ft_8k"),
    "hf:allenai/MolmoAct2-SO100_101": dict(repo="allenai/MolmoAct2-SO100_101", type="model",
                                           rev="152569fe57914d97be91055800035f54e250d009"),
    "hf:allenai/MolmoAct2": dict(repo="allenai/MolmoAct2", type="model",
                                 rev="e432d85f6e039edca44afb93c262f3084ab72a9c"),
}
BOX_DATA = ["photos"]  # copied from the box (private Volume only)
# sim/twin/out inputs some scripts read (rung0.json, layouts, textures, ...): small data files + these paths
OUT_EXT = {".json", ".jsonl", ".npz", ".npy", ".pt", ".usd", ".usda", ".txt", ".csv", ".yaml", ".md"}
OUT_PATHS = ["sim/twin/out/scene", "sim/twin/out/visual_fit/wood", "sim/twin/out/visuals/top_median.png"]
# key result lines: GATE_PASS ..., REPLAY {...}, SMOKE_OK, ... (Kit's own log lines start with a timestamp), errors
KEY_RE = r"^[A-Z][A-Z0-9_]{2,}(\s|:|$)|^Traceback|^\w+(Error|Exception): "

app = modal.App(APP)
cache_vol = modal.Volume.from_name("dlsj-twin-cache", create_if_missing=True)
hf_vol = modal.Volume.from_name("dlsj-twin-hf", create_if_missing=True)
data_vol = modal.Volume.from_name("dlsj-twin-data", create_if_missing=True)
fan_vol = modal.Volume.from_name("dlsj-twin-fanout", create_if_missing=True)

if modal.is_local():
    sys.path.insert(0, str(HERE))
    if not (HERE.parent / ".build/IsaacLab/docker/Dockerfile.base").exists():
        subprocess.run(["bash", str(HERE / "prepare.sh")], check=True)
    import app as twin_app  # noqa: E402  (the twin image of app.py, without its code layer)

    IMAGE = twin_app.base_image()
else:
    IMAGE = modal.Image.debian_slim()  # never used inside a container
HF_IMAGE = modal.Image.debian_slim(python_version="3.12").uv_pip_install("huggingface_hub==0.34.4")


# ----------------------------------------------------------------------------- container side
def _sh(cmd: str, timeout: float = 600, env: dict | None = None, cwd: str | None = None,
        log: Path | None = None) -> tuple[int, str]:
    p = subprocess.Popen(["bash", "-lc", cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         env={**os.environ, **(env or {})}, cwd=cwd, start_new_session=True)
    killed = []

    def _kill():
        killed.append(1)
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    timer = threading.Timer(timeout, _kill)
    timer.start()
    out = []
    f = open(log, "w") if log else None
    for line in p.stdout:
        out.append(line)
        sys.stdout.write(line)
        if f:
            f.write(line)
    p.wait()
    timer.cancel()
    if killed:
        out.append(f"\n[fanout: killed after the {timeout:.0f} s job timeout]\n")
        if f:
            f.write(out[-1])
    if f:
        f.close()
    return (-9 if killed else p.returncode), "".join(out)


def _command(cmd: str) -> tuple[str, dict]:
    """sim/twin/run.sh X.py args / X.py args -> isaaclab.sh -p; python -> the Isaac venv; else bash as is."""
    tok = shlex.split(cmd)
    env = {}
    while tok and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tok[0]):
        k, v = tok.pop(0).split("=", 1)
        env[k] = v
    if tok and tok[0].lstrip("./") == "sim/twin/run.sh":
        tok = tok[1:]
    if tok and tok[0].endswith(".py"):
        return f"{ISAAC_SH} -p {shlex.join(tok)}", env
    if tok and tok[0] in ("python", "python3"):
        return f"{VENV_PY} {shlex.join(tok[1:])}", env
    return shlex.join(tok), env


@app.function(image=IMAGE, gpu="RTX-PRO-6000", timeout=2100, cpu=8.0, memory=32768, max_containers=MAX_HARD,
              single_use_containers=True,
              volumes={CACHE: cache_vol, HF: hf_vol, DATA: data_vol, FAN: fan_vol})
def job(spec: dict) -> dict:
    t0 = time.monotonic()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    run, name = spec["run"], spec["name"]
    dst = Path(FAN) / "runs" / run / name
    for d in ("/root/.nvidia-omniverse/config", "/isaac-sim/.nvidia-omniverse/config"):
        Path(d).mkdir(parents=True, exist_ok=True)
        (Path(d) / "privacy.toml").write_text(PRIVACY_TOML)
    fan_vol.reload()
    so101 = f"{WORK}/sim/twin/assets/so101"
    prep = (f"rm -rf {WORK} && mkdir -p {WORK} && tar -xzf {FAN}/runs/{run}/code.tar.gz -C {WORK} "
            f"&& mkdir -p {so101} && cp -a /opt/twin/assets/so101/urdf {so101}/urdf "
            f"&& {{ [ ! -f {CACHE}/so101_usd.tar ] || tar -xf {CACHE}/so101_usd.tar -C {so101}; }} "
            f"&& mkdir -p {WORK}/sim/twin/data {WORK}/sim/twin/out && ln -sfn {HF} {WORK}/sim/twin/data/hf "
            f"&& for d in {DATA}/*; do ln -sfn \"$d\" {WORK}/sim/twin/data/; done")
    rc, txt = _sh(prep, timeout=600)
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                         text=True).stdout.strip().splitlines()
    tag = "".join(c if c.isalnum() else "_" for c in (gpu[0] if gpu else "unknown"))
    kit = Path(CACHE) / f"kitcache_{tag}.tar"
    if kit.exists():
        _sh(f"tar -xf {kit} -C / 2>/dev/null || true")
    prep_s = time.monotonic() - t0
    marker = Path("/tmp/fanout.marker")
    marker.write_text(started)
    time.sleep(1.1)  # out files from the snapshot keep their mtimes; anything newer than the marker is the job's
    cmd, cenv = _command(spec["cmd"])
    env = {"PYTHONPATH": WORK, "HF_HOME": HF, "HF_HUB_OFFLINE": "1",
           # the twin's render path (Fabric scene delegate off + VisualSync)
           "DLSJ_KIT_ARGS": "--/app/useFabricSceneDelegate=false "
                            "--/rtx/hydra/readTransformsFromFabricInRenderDelegate=false",
           **EULA, **cenv, **(spec.get("env") or {})}
    dst.mkdir(parents=True, exist_ok=True)
    log = Path("/tmp/stdout.log")
    t1 = time.monotonic()
    if rc == 0:
        print(f"$ {cmd}", flush=True)
        rc, txt = _sh(cmd, timeout=spec["timeout"], env=env, cwd=WORK, log=log)
    else:
        log.write_text(txt)
    wall = time.monotonic() - t1
    tel = subprocess.run(["pgrep", "-f", TELEMETRY], capture_output=True, text=True).stdout.split()
    if tel:
        subprocess.run(["pkill", "-f", TELEMETRY])
    # collect: everything the job wrote under sim/twin/out (up to max_out_mb), stdout, job.json
    files, total, skipped = [], 0, []
    out_root = Path(WORK) / "sim/twin/out"
    for f in sorted(out_root.rglob("*")):
        if not f.is_file() or f.stat().st_mtime < marker.stat().st_mtime:
            continue
        sz = f.stat().st_size
        rel = f.relative_to(out_root)
        if total + sz > spec.get("max_out_mb", 2048) * 2**20:
            skipped.append(str(rel))
            continue
        (dst / "out" / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst / "out" / rel)
        files.append(str(rel))
        total += sz
    shutil.copy2(log, dst / "stdout.log")
    keys = [ln.strip()[:2000] for ln in txt.splitlines() if re.search(spec.get("key_re") or KEY_RE, ln.strip())]
    res = {"job": name, "cmd": spec["cmd"], "exec": cmd, "rc": rc, "wall_s": round(wall, 1),
           "prep_s": round(prep_s, 1), "gpu": tag, "started_utc": started, "key_lines": keys[-40:],
           "tail": txt[-3000:], "out_files": files, "out_bytes": total, "out_skipped": skipped,
           "telemetry_killed": bool(tel), "kit_cache": kit.exists()}
    (dst / "job.json").write_text(json.dumps(res, indent=1))
    fan_vol.commit()
    if rc == 0 and not kit.exists():  # first run on this GPU type: keep its shader cache for the next ones
        dirs = " ".join(d.lstrip("/") for d in ("isaac-sim/kit/cache", "root/.cache/ov", "root/.cache/nvidia/GLCache",
                                                "root/.nv/ComputeCache", "root/.local/share/ov/data")
                        if Path("/" + d).exists())
        tmp = f"{kit}.{os.getpid()}.tmp"
        _sh(f"tar -cf {tmp} -C / {dirs} && mv -n {tmp} {kit}; rm -f {tmp}", timeout=600)
        cache_vol.commit()
    res["container_s"] = round(time.monotonic() - t0, 1)
    return res


@app.function(image=HF_IMAGE, cpu=4.0, memory=8192, timeout=3600, volumes={DATA: data_vol, HF: hf_vol},
              secrets=[modal.Secret.from_name(HF_SECRET)])
def fetch(key: str, item: dict) -> dict:
    """One HF snapshot into the data Volume (local= path) or the HF cache Volume (rev=), token from the secret."""
    from huggingface_hub import snapshot_download

    names = [k for k in os.environ if ("HF" in k or "HUGGING" in k) and os.environ[k].startswith("hf_")]
    pref = [k for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN") if k in names]
    var = (pref or names or [None])[0]
    t0 = time.monotonic()
    kw = dict(repo_id=item["repo"], repo_type=item["type"], token=os.environ.get(var) if var else None,
              max_workers=16, allow_patterns=item.get("allow"))
    if "local" in item:
        path = snapshot_download(local_dir=f"{DATA}/{item['local']}", **kw)
        vol = data_vol
    else:
        path = snapshot_download(revision=item["rev"], cache_dir=f"{HF}/hub", **kw)
        vol = hf_vol
    n, size = 0, 0
    for f in Path(path).rglob("*"):
        if f.is_file() and ".cache" not in f.parts:
            n, size = n + 1, size + f.stat().st_size
    vol.commit()
    return {"key": key, "files": n, "bytes": size, "s": round(time.monotonic() - t0, 1), "token_var": var}


# ----------------------------------------------------------------------------- local side
SNAP_PY = r'''
import json, os, subprocess, sys, tarfile, io, time
repo, max_mb, out_mb, list_only = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4] == "1"
inc, out_ext, out_paths = json.loads(sys.argv[5]), set(json.loads(sys.argv[6])), json.loads(sys.argv[7])
keep = sys.argv[8]  # also keep the tar on the box (repo-relative path), "" = no
os.chdir(os.path.expanduser(repo))
g = subprocess.run(["git", "ls-files", "-z", "-co", "--exclude-standard", "--", "sim", "dlsj", "deploy"],
                   capture_output=True, check=True).stdout.decode().split("\0")
drop = []
def ok(f, lim):
    if not os.path.isfile(f) or "__pycache__" in f or "/.pytest_cache/" in f:
        return False
    if os.path.getsize(f) > lim * 2**20:
        drop.append(f)
        return False
    return True
files = [f for f in g if f and ok(f, max_mb)]
for root, _, fs in os.walk("sim/twin/out"):
    if root.startswith("sim/twin/out/modal_fanout"):
        continue
    for x in fs:
        f = os.path.join(root, x)
        if os.path.splitext(x)[1] in out_ext and os.path.getsize(f) <= out_mb * 2**20 and ok(f, max_mb):
            files.append(f)
for p in out_paths + inc:
    if os.path.isdir(p):
        files += [os.path.join(r, x) for r, _, fs in os.walk(p) for x in fs if ok(os.path.join(r, x), 2048)]
    elif os.path.isfile(p):
        files.append(p)
    elif p in inc:
        sys.stderr.write(f"WARNING include {p} not found\n")
files = sorted(set(files))
head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
br = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True).stdout.strip()
dirty = subprocess.run(["git", "status", "--short", "--", "sim", "dlsj", "deploy"], capture_output=True,
                       text=True).stdout.splitlines()
meta = {"repo": os.getcwd(), "head": head, "branch": br, "dirty_files": len(dirty), "files": len(files),
        "bytes": sum(os.path.getsize(f) for f in files if os.path.exists(f)), "dropped_large": drop[:50],
        "utc": time.time()}
class Tee:
    def __init__(self, *fs):
        self.fs = fs
    def write(self, b):
        for f in self.fs:
            f.write(b)
        return len(b)
    def flush(self):
        for f in self.fs:
            f.flush()
if not list_only:
    sinks = [sys.stdout.buffer]
    if keep:
        os.makedirs(os.path.dirname(keep), exist_ok=True)
        sinks.append(open(keep, "wb"))
    with tarfile.open(fileobj=Tee(*sinks), mode="w|gz", compresslevel=3) as t:
        for f in files:
            try:
                t.add(f, recursive=False)
            except FileNotFoundError:  # a live tree: removed since it was listed
                pass
        b = json.dumps(meta, indent=1).encode()
        ti = tarfile.TarInfo("sim/twin/SNAPSHOT.json"); ti.size = len(b); ti.mtime = 0
        t.addfile(ti, io.BytesIO(b))
sys.stderr.write("SNAP " + json.dumps(meta) + "\n")
'''


def _ssh(box: str, cmd: str, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(SSH + [box, cmd], **kw)


def snapshot(box: str, dst: Path | None, a, keep: str = "") -> dict:
    """Stream the box's code (+ small out inputs) as a tar.gz to dst (None: stats only); keep = box copy path."""
    args = [REMOTE_REPO, str(a.max_file_mb), "2", "0" if dst else "1", json.dumps(a.include),
            json.dumps(sorted(OUT_EXT)), json.dumps(OUT_PATHS), keep]
    cmd = "python3 - " + " ".join(shlex.quote(x) for x in args)
    with open(dst, "wb") if dst else open(os.devnull, "wb") as f:
        p = subprocess.run(SSH + [box, cmd], input=SNAP_PY.encode(), stdout=f, stderr=subprocess.PIPE)
    err = p.stderr.decode()
    line = [ln for ln in err.splitlines() if ln.startswith("SNAP ")]
    if p.returncode or not line:
        raise SystemExit(f"snapshot from {box} failed (rc {p.returncode}):\n{err[-2000:]}")
    for ln in err.splitlines():
        if ln.startswith("WARNING"):
            print(ln)
    return json.loads(line[-1][5:])


def _expand(template: str, grid: list[str]) -> list[tuple[str, str]]:
    """{a,b,c} inline lists and {key} with --grid key=v1,v2 -> cartesian product of (name, command)."""
    named = {}
    for g in grid:
        k, v = g.split("=", 1)
        named[k] = v.split(",")
    axes = []  # (span text, label, values)
    for m in re.finditer(r"\{([^{}\"':\s]+)\}", template):
        body = m.group(1)
        if body in named:
            if body not in [x[1] for x in axes]:
                axes.append((m.group(0), body, named[body]))
        elif "," in body:
            pre = re.search(r"(?:([A-Za-z0-9_]+)=|--([A-Za-z0-9_-]+)\s+)$", template[:m.start()].split(".")[-1])
            label = (pre.group(1) or pre.group(2)) if pre else f"p{len(axes)}"
            axes.append((m.group(0), label, body.split(",")))
    missing = set(named) - {a[1] for a in axes}
    if missing:
        raise SystemExit(f"--grid keys not in --cmd as {{key}}: {sorted(missing)}")
    out = []
    for combo in itertools.product(*[a[2] for a in axes]) if axes else [()]:
        cmd, parts = template, []
        for (span, label, _), v in zip(axes, combo, strict=True):
            cmd = cmd.replace(span, v, 1) if span.strip("{}") not in named else cmd.replace(span, v)
            parts.append(f"{label}={v}")
        out.append(("_".join(parts) or "job", cmd))
    return out


def _jobs(a) -> list[dict]:
    if a.jobs:
        txt = Path(a.jobs).read_text()
        if a.jobs.endswith((".yaml", ".yml")):
            import yaml

            raw = yaml.safe_load(txt)
        else:
            raw = json.loads(txt)
        raw = raw["jobs"] if isinstance(raw, dict) else raw
        jobs = [j if isinstance(j, dict) else {"cmd": j} for j in raw]
    elif a.cmd:
        jobs = [{"name": n, "cmd": c} for n, c in _expand(a.cmd, a.grid)]
    else:
        raise SystemExit("give --cmd TEMPLATE [--grid k=v1,v2] or --jobs FILE")
    env = dict(e.split("=", 1) for e in a.env)
    seen = set()
    for i, j in enumerate(jobs):
        j.setdefault("name", f"j{i:02d}")
        n = f"j{i:02d}_" + re.sub(r"[^A-Za-z0-9_.=-]+", "-", j["name"])[:80]
        assert n not in seen
        seen.add(n)
        j["name"] = n
        j.setdefault("gpu", a.gpu)
        j.setdefault("timeout", a.timeout)
        j.setdefault("cpu", a.cpu)
        j.setdefault("memory", a.memory)
        j["env"] = {**env, **(j.get("env") or {})}
        if a.key:
            j["key_re"] = a.key
    return jobs


def _rate(j: dict, a) -> float:
    g, _, n = j["gpu"].partition(":")
    price = a.gpu_price if a.gpu_price else GPU_PRICE.get(g.rstrip("+"))
    if price is None:
        raise SystemExit(f"no list price for GPU {g}; pass --gpu-price USD_PER_S (see https://modal.com/pricing)")
    return price * int(n or 1) + CPU_PRICE * j["cpu"] + MEM_PRICE * j["memory"] / 1024


def _gpus(j: dict) -> int:
    return int(j["gpu"].partition(":")[2] or 1)


def _live_apps() -> list[dict]:
    r = subprocess.run([sys.executable, "-m", "modal", "app", "list", "--json"], capture_output=True, text=True,
                       timeout=120)
    return [x for x in json.loads(r.stdout or "[]") if x.get("description") == APP and "stopped" not in
            str(x.get("state"))]


def _stop_leftovers(wait_s: float = 120, only: str | None = None) -> list[str]:
    """Stop apps of this name Modal still lists as running (belt and braces after app.run), wait until none is.

    only = an app id: stop just that one (a run cleans up after itself without killing other fan-out runs in
    flight); None (`fanout.py stop`) stops all of them."""
    stopped = []
    try:
        for x in [x for x in _live_apps() if only is None or x["app_id"] == only]:
            subprocess.run([sys.executable, "-m", "modal", "app", "stop", "-y", x["app_id"]], timeout=120,
                           capture_output=True)
            stopped.append(f"{x['app_id']} ({x.get('state')}, {x.get('tasks')} tasks)")
        t0 = time.time()
        while stopped and time.time() - t0 < wait_s:
            live = [x for x in _live_apps() if only is None or x["app_id"] == only]
            if not live:
                break
            time.sleep(5)
        else:
            if stopped:
                print(f"[fanout] WARNING still listed as running after {wait_s:.0f} s: {live}; run `fanout.py stop`")
    except Exception as e:  # noqa: BLE001
        print(f"[fanout] WARNING could not check for running apps: {e}; run `fanout.py stop`")
    return stopped


def _manifest() -> dict:
    try:
        return json.loads(b"".join(data_vol.read_file("MANIFEST.json")))
    except Exception:  # noqa: BLE001
        return {}


def ensure_data(box: str, force: bool = False) -> dict:
    """Fetch missing data into the Volumes (must run inside app.run())."""
    man = _manifest()
    todo = {k: v for k, v in HF_DATA.items() if force or k not in man}
    calls = {k: fetch.spawn(k, v) for k, v in todo.items()}
    changed = bool(todo)
    for d in BOX_DATA:
        if force or d not in man:
            changed = True
            tmp = SCRATCH / "data" / d
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["scp", "-F", os.path.expanduser("~/.brev/ssh_config"), "-q", "-r",
                            f"{box}:{REMOTE_REPO}/sim/twin/data/{d}", str(tmp)], check=True)
            with data_vol.batch_upload(force=True) as up:
                up.put_directory(str(tmp), f"/{d}")
            n = sum(1 for f in tmp.rglob("*") if f.is_file())
            shutil.rmtree(tmp, ignore_errors=True)
            man[d] = {"from": f"{box}:sim/twin/data/{d}", "files": n,
                      "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            print(f"[data] {d}: {n} files from {box}")
    for k, c in calls.items():
        try:
            r = c.get()
            man[k] = {**HF_DATA[k], **r, "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            print(f"[data] {k}: {r['files']} files, {r['bytes'] / 1e9:.2f} GB in {r['s']} s")
        except Exception as e:  # noqa: BLE001
            print(f"[data] {k}: FAILED {type(e).__name__}: {str(e)[:500]}")
    if changed:
        p = SCRATCH / "MANIFEST.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(man, indent=1))
        with data_vol.batch_upload(force=True) as up:
            up.put_file(str(p), "/MANIFEST.json")
        p.unlink()
    missing = [k for k in [*HF_DATA, *BOX_DATA] if k not in man]
    if missing:
        raise SystemExit(f"data still missing: {missing}")
    return man


def _download_run(run: str, local: Path) -> None:
    for e in fan_vol.listdir(f"runs/{run}", recursive=True):
        if e.type != modal.volume.FileEntryType.FILE or e.path.endswith("code.tar.gz"):
            continue
        rel = Path(e.path).relative_to(f"runs/{run}")
        (local / rel).parent.mkdir(parents=True, exist_ok=True)
        with open(local / rel, "wb") as f:
            fan_vol.read_file_into_fileobj(e.path, f)


def _table(res: list[dict]) -> str:
    rows = [f"{'job':<44} {'rc':>4} {'wall_s':>7} {'cont_s':>7}  key result lines"]
    for r in res:
        keys = r.get("key_lines") or [ln for ln in (r.get("tail") or "").strip().splitlines()[-2:]]
        keys = [k for k in keys if "{" not in k] or keys  # e.g. GATE_FAIL ... over the GATE {json} line
        rows.append(f"{r['job'][:44]:<44} {r.get('rc', '-')!s:>4} {r.get('wall_s', '-')!s:>7} "
                    f"{r.get('container_s', '-')!s:>7}  " + " | ".join(k[:160] for k in keys[-2:]))
    return "\n".join(rows)


def _sigterm(*_) -> None:
    raise KeyboardInterrupt


def cmd_run(a) -> None:
    jobs = _jobs(a)
    if a.max_containers > MAX_HARD:
        raise SystemExit(f"--max-containers is capped at {MAX_HARD}")
    if a.max_containers > MAX_DEFAULT and not a.wide:
        raise SystemExit(f"--max-containers > {MAX_DEFAULT} needs --wide (hard maximum {MAX_HARD})")
    conc = min(a.max_containers, len(jobs))
    bound = [(j["timeout"] + 300) * _rate(j, a) for j in jobs]  # Modal timeout = job timeout + prep / collect
    exp = [(min(a.est_min * 60, j["timeout"]) + 120) * _rate(j, a) for j in jobs]
    waves = -(-len(jobs) // conc)
    print(f"[fanout] {len(jobs)} jobs on {sorted({j['gpu'] for j in jobs})}, {conc} at a time ({waves} waves); "
          f"cost estimate ${sum(exp):.2f} at ~{a.est_min:g} min/job (+2 min start), upper bound ${sum(bound):.2f} "
          f"(every job running to its {a.timeout} s timeout); GPU ${GPU_PRICE.get(a.gpu.split(':')[0], 0) * 3600:.2f}"
          f"/h list price")
    for j in jobs:
        print(f"  {j['name']}: [{j['gpu']}, {j['cpu']:g} cpu, {j['memory'] // 1024} GiB, {j['timeout']} s] {j['cmd']}")
    if sum(bound) > COST_LIMIT and not a.yes_cost:
        raise SystemExit(f"[fanout] upper bound ${sum(bound):.2f} > ${COST_LIMIT:.0f}: lower --timeout / the job "
                         f"count, or pass --yes-cost")
    if a.dry_run:
        print("[fanout] dry run: snapshot stats only, nothing launched")
        m = snapshot(a.box, None, a)
        print(f"[fanout] snapshot {a.box}:{m['repo']} {m['branch']}@{m['head']} (+{m['dirty_files']} dirty): "
              f"{m['files']} files, {m['bytes'] / 1e6:.1f} MB; dropped (> {a.max_file_mb} MB): {m['dropped_large']}")
        return
    run = a.name or datetime.now().strftime("%Y%m%d-%H%M%S")
    run = re.sub(r"[^A-Za-z0-9_.-]+", "-", run)
    local = SCRATCH / run
    shutil.rmtree(local, ignore_errors=True)
    local.mkdir(parents=True)
    t_start = time.time()
    results: dict[str, dict] = {}
    stop_ev = threading.Event()
    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, signal.default_int_handler)  # also when started in the background (SIGINT ignored)

    def _t(stage: str) -> None:
        print(f"[fanout] t+{time.time() - t_start:.0f}s {stage}", flush=True)

    m: dict = {}
    try:
        tar = local / "code.tar.gz"
        m = snapshot(a.box, tar, a, keep=f"sim/twin/out/modal_fanout/{run}/code.tar.gz")
        _t("snapshot")
        print(f"[fanout] snapshot {a.box}:{m['repo']} {m['branch']}@{m['head']} (+{m['dirty_files']} dirty files): "
              f"{m['files']} files, {m['bytes'] / 1e6:.1f} MB -> {tar.stat().st_size / 1e6:.1f} MB gz")
        app_id = None
        with fan_vol.batch_upload(force=True) as up:
            up.put_file(str(tar), f"/runs/{run}/code.tar.gz")
        tar.unlink()
        _t("upload")
        with modal.enable_output(), app.run():
            app_id = app.app_id
            _t("app started")
            if not a.skip_data:
                ensure_data(a.box)
                _t("data checked")
            sem = threading.Semaphore(conc)
            fns = {}

            def one(j):
                with sem:
                    if stop_ev.is_set():
                        return
                    key = (j["gpu"], j["timeout"], j["cpu"], j["memory"])
                    if key not in fns:
                        fns[key] = job.with_options(gpu=j["gpu"], timeout=j["timeout"] + 300, cpu=float(j["cpu"]),
                                                    memory=int(j["memory"]), max_containers=conc,
                                                    secrets=[modal.Secret.from_name(x) for x in a.secret])
                    t0 = time.time()
                    try:
                        r = fns[key].remote({**j, "run": run})
                    except Exception as e:  # noqa: BLE001
                        r = {"job": j["name"], "rc": "ERR", "key_lines": [f"{type(e).__name__}: {str(e)[:300]}"]}
                    r["client_s"] = round(time.time() - t0, 1)
                    results[j["name"]] = r
                    print(f"[fanout] t+{time.time() - t_start:.0f}s done {j['name']}: rc {r.get('rc')} "
                          f"wall {r.get('wall_s')} s, container {r.get('container_s')} s, client {r['client_s']} s "
                          f"({len(results)}/{len(jobs)})", flush=True)

            ths = [threading.Thread(target=one, args=(j,), daemon=True) for j in jobs]
            for t in ths:
                t.start()
            for t in ths:
                while t.is_alive():
                    t.join(1.0)
    except KeyboardInterrupt:
        stop_ev.set()
        print("\n[fanout] interrupted: stopping the app (all containers)")
    finally:
        stop_ev.set()
        _t("app stopped")
        left = _stop_leftovers(only=app_id) if app_id else []
        if left:
            print(f"[fanout] stopped leftover apps: {left}")
    res = [results.get(j["name"], {"job": j["name"], "rc": "n/a"}) for j in jobs]
    wall = time.time() - t_start
    # container time measured inside each job; an interrupted / failed call counts its client time + 30 s shutdown
    cont = {j["name"]: res[i].get("container_s") or (res[i].get("client_s") or 0) + 30 for i, j in enumerate(jobs)}
    gpu_min = sum(cont[j["name"]] * _gpus(j) for j in jobs) / 60
    cost = sum(cont[j["name"]] * _rate(j, a) for j in jobs)
    table = _table(res)
    head = (f"run {run}: {len(jobs)} jobs from {a.box} ({m.get('branch')}@{m.get('head')} +{m.get('dirty_files')} "
            f"dirty) | wall {wall / 60:.1f} min | {gpu_min:.1f} GPU-min | ~${cost:.2f} (container time x list price, "
            f"excludes image load)")
    print("\n" + head + "\n" + table)
    try:
        _download_run(run, local)
        fan_vol.remove_file(f"runs/{run}/code.tar.gz")
        _t("downloaded")
    except Exception as e:  # noqa: BLE001
        print(f"[fanout] WARNING download from dlsj-twin-fanout:/runs/{run} failed: {e}")
    (local / "summary.txt").write_text(head + "\n" + table + "\n")
    (local / "summary.json").write_text(json.dumps({"run": run, "box": a.box, "snapshot": m,
                                                    "wall_s": round(wall, 1), "gpu_min": round(gpu_min, 2),
                                                    "cost_usd": round(cost, 3), "jobs": res}, indent=1))
    rdir = f"{REMOTE_REPO}/sim/twin/out/modal_fanout"
    tar = subprocess.Popen(["tar", "-C", str(local), "-cf", "-", "."], stdout=subprocess.PIPE)
    p = _ssh(a.box, f"mkdir -p {rdir}/{run} && tar -xf - -C {rdir}/{run}", stdin=tar.stdout)
    tar.wait()
    _t("copied to box")
    ok = sum(1 for r in res if r.get("rc") == 0)
    row = ",".join(str(x) for x in [datetime.fromtimestamp(t_start, timezone.utc).isoformat(timespec="seconds"), run,
                                    a.box, len(jobs), ok, "+".join(sorted({j["gpu"] for j in jobs})), conc,
                                    round(wall / 60, 2), round(gpu_min, 2), round(cost, 3), os.environ.get("USER", "")])
    hdr = "utc_start,run,box,jobs,jobs_ok,gpu,max_containers,wall_min,gpu_min,est_cost_usd,user"
    _ssh(a.box, f"f={rdir}/usage.csv; [ -s $f ] || echo {hdr} > $f; echo {shlex.quote(row)} >> $f")
    if p.returncode == 0:
        shutil.rmtree(local, ignore_errors=True)
        print(f"[fanout] results -> {a.box}:{rdir}/{run}/  (usage logged in {rdir}/usage.csv)")
    else:
        print(f"[fanout] WARNING copy to {a.box} failed; results kept in {local}")


def cmd_data(a) -> None:
    with modal.enable_output(), app.run():
        app_id = app.app_id
        man = ensure_data(a.box, force=a.force)
    _stop_leftovers(only=app_id)
    print(json.dumps({k: {x: v.get(x) for x in ("files", "bytes", "from", "rev")} for k, v in man.items()}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="sub", required=True)
    r = sub.add_parser("run", help="fan jobs out")
    r.add_argument("--cmd", help="template command; {a,b} lists and {key} (with --grid) expand")
    r.add_argument("--grid", action="append", default=[], help="key=v1,v2 for {key} in --cmd (repeatable)")
    r.add_argument("--jobs", help="YAML / JSON list of {name, cmd, gpu, timeout, cpu, memory, env}")
    r.add_argument("--box", default="watson", help="code snapshot from and results to this box (watson / jarvis)")
    r.add_argument("--gpu", default="RTX-PRO-6000")
    r.add_argument("--gpu-price", type=float, default=None, help="USD / s, for GPUs not in the built-in list")
    r.add_argument("--timeout", type=int, default=1800, help="per-job command timeout, s")
    r.add_argument("--cpu", type=float, default=8.0)
    r.add_argument("--memory", type=int, default=32768, help="MiB")
    r.add_argument("--env", action="append", default=[], help="K=V for every job (repeatable)")
    r.add_argument("--secret", action="append", default=[], help="Modal secret for every job, e.g. dlsj-wandb")
    r.add_argument("--max-containers", type=int, default=MAX_DEFAULT)
    r.add_argument("--wide", action="store_true", help=f"allow up to {MAX_HARD} containers at once")
    r.add_argument("--est-min", type=float, default=5.0, help="expected minutes per job, for the cost estimate")
    r.add_argument("--yes-cost", action="store_true", help=f"allow an upper-bound estimate above ${COST_LIMIT:.0f}")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--name", help="run name (default: timestamp)")
    r.add_argument("--include", action="append", default=[], help="extra box path (file / dir) for the snapshot")
    r.add_argument("--max-file-mb", type=float, default=20.0, help="drop larger code-tree files from the snapshot")
    r.add_argument("--key", help="regex for the key result lines (default: GATE_*, *_OK, RESULT, Traceback, ...)")
    r.add_argument("--skip-data", action="store_true", help="do not check the data Volumes")
    d = sub.add_parser("data", help="make sure the data Volumes are complete")
    d.add_argument("--box", default="watson")
    d.add_argument("--force", action="store_true", help="re-fetch everything")
    sub.add_parser("stop", help="stop any running fan-out app")
    a = ap.parse_args()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    if a.sub == "run":
        cmd_run(a)
    elif a.sub == "data":
        cmd_data(a)
    else:
        print(f"stopped: {_stop_leftovers()}")


if __name__ == "__main__":
    main()
