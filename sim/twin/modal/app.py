"""The Isaac twin on Modal GPUs (x86_64): image, render checks, env throughput, MolmoAct2 latency.

    sim/twin/modal/prepare.sh            # Isaac Lab v3.0.0 checkout -> sim/twin/.build/IsaacLab (as build.sh)
    MODAL_PROFILE=desmond-zee uv run --no-project --with modal python -m modal run sim/twin/modal/app.py \
        --task <task> [--gpu ...] [...]

Tasks (each calls one function with a hard timeout; results -> sim/twin/out/modal/<task>_<gpu>.json + PNGs):
    diag         --gpu RTX-PRO-6000      nvidia-smi, mounted NVIDIA userspace libs, vulkaninfo
    render       --gpu RTX-PRO-6000      smoke.py + check_render_motion.py, with and without the GB10 KIT_ARGS
    bench-env    --gpu RTX-PRO-6000:2 --procs-per-gpu 1 --steps 300   TwinEnv steps/s, N Isaac processes at once
    fetch-molmo                          MolmoAct2-SO100_101 -> Volume dlsj-twin-hf (CPU only)
    bench-molmo  --gpu B200              MolmoAct2 latency per 30-action chunk (bf16, 10 flow steps), batch 1 / 8

Image = Isaac Lab v3.0.0's own docker/Dockerfile.base on nvcr.io/nvidia/isaac-sim:6.2.0 (the file build.sh builds
on the GB10) + the twin Dockerfile's Studio layer (Physical AI Studio 8f132416 against Isaac Lab's torch / numpy pins) +
the pinned SO-101 URDF (fetch_so101.sh). The repo's sim/ and deploy/ are added last (not copied into a layer), so code
edits do not rebuild the image. Everything runs as root (Modal's runtime); Isaac runs through isaaclab.sh -p exactly
as sim/twin/run.sh does, in a subprocess of Modal's own Python (add_python), so Modal's client never touches the
Isaac venv.

Volumes: dlsj-twin-cache (Kit / shader / compute caches as one tarball per GPU type, the URDF -> USD conversion),
dlsj-twin-hf (HF_HOME for MolmoAct2).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
LOCAL_REPO = HERE.parents[2] if modal.is_local() else Path("/nonexistent")  # in a container: /root/app.py
ISAACLAB = LOCAL_REPO / "sim/twin/.build/IsaacLab"

APP = "dlsj-twin-modal"
REPO = "/repo"  # code as added to the image (read-only mount)
WORK = "/tmp/repo"  # writable copy each container runs from
ISAAC_SH = "/workspace/isaaclab/isaaclab.sh"
VENV_PY = "/opt/isaaclab-venv/bin/python"
CACHE = "/cache"
HF = "/hf"
KIT_CACHE_DIRS = ["/isaac-sim/kit/cache", "/root/.cache/ov", "/root/.cache/nvidia/GLCache", "/root/.nv/ComputeCache",
                  "/root/.local/share/ov/data"]
EULA = {"ACCEPT_EULA": "Y", "PRIVACY_CONSENT": "N", "OMNI_KIT_ACCEPT_EULA": "YES"}  # telemetry consent off (run.sh)
# Omniverse's documented opt-out file; with it (and PRIVACY_CONSENT=N) Kit should not start omni.telemetry.transmitter.
PRIVACY_TOML = "[privacy]\nperformance = false\npersonalization = false\nusage = false\n"
TELEMETRY = "omni.telemetry.transmitter"

app = modal.App(APP)
cache_vol = modal.Volume.from_name("dlsj-twin-cache", create_if_missing=True)
hf_vol = modal.Volume.from_name("dlsj-twin-hf", create_if_missing=True)
# W&B for DICE-RL training (WANDB_API_KEY / WANDB_ENTITY / WANDB_PROJECT): every training function in this app takes
# `secrets=TRAIN_SECRETS`. The smoke / benchmark functions below do not log to W&B, so they do not attach it.
TRAIN_SECRETS = [modal.Secret.from_name("dlsj-wandb")]


def _modal_dockerfile() -> Path:
    """Isaac Lab's docker/Dockerfile.base with three mechanical rewrites Modal's builder needs (same result):
    `COPY --parents ../source/*/pyproject.toml DST/` -> one COPY per file; `../` source prefixes (BuildKit resolves
    them to the context root) -> context-relative; `RUN --mount=type=cache,...` (a uv download cache only) dropped."""
    import re

    src = (ISAACLAB / "docker/Dockerfile.base").read_text().splitlines()
    out = []
    for ln in src:
        m = re.match(r"^COPY --parents \.\./(source/\*/pyproject\.toml) (\S+)/$", ln)
        if m:
            for f in sorted(ISAACLAB.glob(m.group(1))):
                rel = f.relative_to(ISAACLAB).as_posix()
                out.append(f"COPY {rel} {m.group(2)}/{rel}")
            continue
        if ln.startswith("COPY "):
            ln = re.sub(r"(?<=\s)\.\./", "", ln).replace("isaaclab  ${", "isaaclab . ${")
        ln = re.sub(r"^RUN --mount=type=cache,target=\S+ (?=\\$)", "RUN ", ln)
        out.append(ln)
    text = "\n".join(out) + "\n"
    assert "--parents" not in text and "--mount" not in text
    dst = ISAACLAB / "docker/Dockerfile.modal"
    if not dst.exists() or dst.read_text() != text:
        dst.write_text(text)
    return dst


def base_image() -> modal.Image:
    """The twin image without the repo code (fanout.py adds its own code snapshot at container start)."""
    if not (ISAACLAB / "docker/Dockerfile.base").exists():
        if modal.is_local():
            raise SystemExit(f"missing {ISAACLAB}; run sim/twin/modal/prepare.sh first")
        return modal.Image.debian_slim()  # inside a container the image is already built; this is never used
    base = modal.Image.from_dockerfile(
        _modal_dockerfile(),
        context_dir=ISAACLAB,
        ignore=modal.FilePatternMatcher.from_file(ISAACLAB / ".dockerignore"),
        build_args={"ISAACSIM_VERSION_ARG": "6.2.0", "ISAACSIM_ROOT_PATH_ARG": "/isaac-sim",
                    "ISAACLAB_PATH_ARG": "/workspace/isaaclab", "DOCKER_USER_HOME_ARG": "/root"},
        add_python="3.12",
    )
    # sim/twin/docker/Dockerfile (Studio layer), unchanged except: root user (Modal), no BuildKit cache mount.
    studio = base.dockerfile_commands([
        "USER root",
        "ENTRYPOINT []",
        "RUN apt-get update && apt-get install -y --no-install-recommends clang libclang-dev libv4l-dev pkg-config "
        "vulkan-tools pciutils && rm -rf /var/lib/apt/lists/*",
        "ARG STUDIO_REPO=https://github.com/open-edge-platform/physical-ai-studio.git",
        "ARG STUDIO_REF=8f132416",
        "RUN git clone -q --filter=blob:none ${STUDIO_REPO} /opt/physical-ai-studio "
        "&& git -C /opt/physical-ai-studio checkout -q ${STUDIO_REF}",
        "RUN uv pip freeze --python ${VIRTUAL_ENV}/bin/python "
        "| grep -iE '^(torch|torchvision|torchaudio|numpy|warp-lang|gymnasium|opencv-python|opencv-python-headless)==' "
        "> /opt/isaaclab-pins.txt "
        "&& uv pip install --python ${VIRTUAL_ENV}/bin/python -c /opt/isaaclab-pins.txt "
        "-e /opt/physical-ai-studio/library",
    ])
    # SO-101 URDF + meshes (fetch_so101.sh, pinned ref) -> /opt/twin/assets/so101/urdf
    img = (studio
           .add_local_file(LOCAL_REPO / "sim/twin/scripts/fetch_so101.sh", "/opt/twin/scripts/fetch_so101.sh",
                           copy=True)
           .run_commands("bash /opt/twin/scripts/fetch_so101.sh")
           # Telemetry off: Kit still spawns omni.telemetry.transmitter (enableAnonymousData=true) with
           # PRIVACY_CONSENT=N and an opt-out privacy.toml, so its binary becomes a no-op.
           .run_commands("find /isaac-sim -type f -name omni.telemetry.transmitter -print "
                         "-exec sh -c 'printf \"#!/bin/sh\\nexit 0\\n\" > \"$1\"' _ {} \\; "
                         "| tee /opt/telemetry-disabled.txt && test -s /opt/telemetry-disabled.txt")
           .env(EULA | {"HF_HOME": HF, "TOKENIZERS_PARALLELISM": "false"}))
    return img


def _image() -> modal.Image:
    img = base_image()
    if not modal.is_local():
        return img
    ign = ["twin/.build", "twin/assets/so101", "twin/out", "twin/data", "**/__pycache__", "**/.pytest_cache"]
    return (img.add_local_dir(LOCAL_REPO / "sim", f"{REPO}/sim", ignore=ign)
            .add_local_dir(LOCAL_REPO / "deploy", f"{REPO}/deploy", ignore=["**/__pycache__"]))


image = _image()


# ----------------------------------------------------------------------------- container helpers
def _sh(cmd: str, timeout: float = 600, env: dict | None = None, cwd: str | None = None) -> tuple[int, str]:
    """Run a shell command, stream its output, return (rc, output)."""
    print(f"$ {cmd}", flush=True)
    p = subprocess.Popen(["bash", "-lc", cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         env={**os.environ, **(env or {})}, cwd=cwd)
    out = []
    t0 = time.monotonic()
    try:
        for line in p.stdout:
            out.append(line)
            sys.stdout.write(line)
            if time.monotonic() - t0 > timeout:
                p.kill()
                out.append(f"\n[timeout after {timeout}s]\n")
                break
        p.wait(timeout=30)
    except Exception as e:  # noqa: BLE001
        p.kill()
        out.append(f"\n[{type(e).__name__}: {e}]\n")
    return (p.returncode if p.returncode is not None else -9), "".join(out)


def _gpu_tag() -> str:
    try:
        n = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                           text=True, timeout=30).stdout.strip().splitlines()[0]
    except Exception:  # noqa: BLE001
        n = "unknown"
    return "".join(c if c.isalnum() else "_" for c in n)


def _prep(restore_cache: bool = True) -> dict:
    """Writable repo copy at WORK with the URDF (and a cached USD), Kit caches restored for this GPU type."""
    t0 = time.monotonic()
    for d in ("/root/.nvidia-omniverse/config", "/isaac-sim/.nvidia-omniverse/config"):
        Path(d).mkdir(parents=True, exist_ok=True)
        (Path(d) / "privacy.toml").write_text(PRIVACY_TOML)
    _sh(f"rm -rf {WORK} && cp -a {REPO} {WORK} && mkdir -p {WORK}/sim/twin/assets/so101 "
        f"&& cp -a /opt/twin/assets/so101/urdf {WORK}/sim/twin/assets/so101/urdf")
    usd_tar = Path(CACHE) / "so101_usd.tar"
    if usd_tar.exists():
        _sh(f"tar -xf {usd_tar} -C {WORK}/sim/twin/assets/so101")
    tag = _gpu_tag()
    kit_tar = Path(CACHE) / f"kitcache_{tag}.tar"
    if restore_cache and kit_tar.exists():
        _sh(f"tar -xf {kit_tar} -C / 2>/dev/null || true")
    return {"gpu": tag, "prep_s": round(time.monotonic() - t0, 1), "kit_cache_restored": kit_tar.exists()}


def _save_cache(save_usd: bool = True) -> None:
    tag = _gpu_tag()
    dirs = " ".join(d.lstrip("/") for d in KIT_CACHE_DIRS if Path(d).exists())
    if dirs:
        _sh(f"tar -cf {CACHE}/kitcache_{tag}.tar.tmp -C / {dirs} && mv {CACHE}/kitcache_{tag}.tar.tmp "
            f"{CACHE}/kitcache_{tag}.tar && ls -la {CACHE}")
    usd = Path(WORK) / "sim/twin/assets/so101/usd"
    if save_usd and usd.exists() and not (Path(CACHE) / "so101_usd.tar").exists():
        _sh(f"tar -cf {CACHE}/so101_usd.tar -C {WORK}/sim/twin/assets/so101 usd")
    cache_vol.commit()


def _telemetry_check() -> str:
    """Report (and stop) any Omniverse telemetry transmitter; '' when none is running."""
    r = subprocess.run(["pgrep", "-af", TELEMETRY], capture_output=True, text=True).stdout.strip()
    r = "\n".join(ln for ln in r.splitlines() if "pgrep" not in ln)
    if r:
        subprocess.run(["pkill", "-f", TELEMETRY])
        print(f"[telemetry] {TELEMETRY} was running; killed:\n{r}", flush=True)
    return r


def _isaac(script: str, args: str = "", timeout: float = 900, env: dict | None = None) -> tuple[int, str]:
    rc, txt = _sh(f"{ISAAC_SH} -p {script} {args}", timeout=timeout, cwd=WORK,
                  env={"PYTHONPATH": WORK, **EULA, **(env or {})})
    tel = _telemetry_check()
    return rc, txt + (f"\n[telemetry transmitter found and killed: {tel}]\n" if tel else "")


def _grep(text: str, *keys: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if any(k in ln for k in keys)]


# ----------------------------------------------------------------------------- GPU functions
@app.function(image=image, gpu="RTX-PRO-6000", timeout=600, volumes={CACHE: cache_vol})
def diag() -> dict:
    """What the container sees: driver, capabilities, NVIDIA userspace libs (graphics ones matter), Vulkan."""
    out = {}
    for k, c in {
        "nvidia_smi": "nvidia-smi",
        "env": "env | grep -E '^(NVIDIA_|VK_|MUJOCO|HOME=|USER=)' | sort",
        "driver_version": "cat /proc/driver/nvidia/version 2>&1 | head -3",
        "dev": "ls -la /dev/nvidia* /dev/dri 2>&1",
        "libs": "ls /usr/lib/x86_64-linux-gnu | grep -iE 'nvidia|libcuda|libGLX|libEGL_nv|rtcore|glcore|vk' | sort",
        "icd": "cat /etc/vulkan/icd.d/*.json /usr/share/vulkan/icd.d/*.json 2>&1",
        "vulkaninfo": "timeout 120 vulkaninfo --summary 2>&1 | tail -40",
        "cpu_mem": "nproc; free -g | head -2; df -h / /tmp | tail -2",
    }.items():
        out[k] = _sh(c, timeout=150)[1]
    return out


@app.function(image=image, gpu="RTX-PRO-6000", timeout=2400, volumes={CACHE: cache_vol}, cpu=8.0, memory=32768)
def render(variants: list[str] | None = None) -> dict:
    """smoke.py and check_render_motion.py with ("gb10fix") and without ("nofix") the GB10 KIT_ARGS."""
    info = _prep()
    res = {"prep": info, "runs": {}, "images": {}}
    kit = {"gb10fix": None, "nofix": ""}
    for v in variants or ["gb10fix", "nofix"]:
        env = {} if kit[v] is None else {"DLSJ_KIT_ARGS": kit[v]}
        for script in ("sim/twin/scripts/smoke.py", "sim/twin/scripts/check_render_motion.py"):
            t0 = time.monotonic()
            rc, txt = _isaac(script, timeout=1200, env=env)
            key = f"{v}:{Path(script).stem}"
            res["runs"][key] = {"rc": rc, "wall_s": round(time.monotonic() - t0, 1),
                                "key_lines": _grep(txt, "SMOKE_OK", "RENDER_", "camera ", "cuda:", "top image change",
                                                   "physics gripper", "isaaclab ", "isaacsim ", "physicalai", "torch ",
                                                   "Error", "error:", "Traceback", "vkCreate", "Vulkan", "GPU"),
                                "telemetry_killed": "[telemetry transmitter found" in txt, "tail": txt[-4000:]}
            if Path(script).stem == "smoke":
                for png in sorted((Path(WORK) / "sim/twin/out").glob("smoke_*.png")):
                    res["images"][f"{v}_{png.name}"] = png.read_bytes()
                    png.unlink()
            if rc == 0:
                _save_cache()
    return res


@app.function(image=image, gpu="RTX-PRO-6000", timeout=2400, volumes={CACHE: cache_vol}, cpu=8.0, memory=32768)
def bench_env(procs_per_gpu: int = 1, steps: int = 300, warmup: int = 30, kit_args: str | None = None) -> dict:
    """TwinEnv (3 cameras, 640x480) steps/s: procs_per_gpu Isaac processes on every visible GPU, in parallel."""
    info = _prep()
    n_gpu = int(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.count("GPU "))
    env = {} if kit_args is None else {"DLSJ_KIT_ARGS": kit_args}
    smi_peak: dict = {}
    stop = threading.Event()

    def sample():  # peak memory / utilisation per GPU while the benchmark runs
        while not stop.wait(3.0):
            r = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True)
            for ln in r.stdout.strip().splitlines():
                i, mem, util = (int(x.strip()) for x in ln.split(","))
                d = smi_peak.setdefault(i, {"mem_mib_max": 0, "util_samples": []})
                d["mem_mib_max"] = max(d["mem_mib_max"], mem)
                d["util_samples"].append(util)

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    if n_gpu * procs_per_gpu == 1:  # single process: warm the cache in the foreground, keep the log
        rc, txt = _isaac("sim/twin/modal/bench_env.py", f"--gpu 0 --steps {steps} --warmup {warmup} "
                         f"--out /tmp/bench_0.json --png-dir /tmp", timeout=1800, env=env)
        logs = {0: txt[-3000:]}
        if rc == 0:
            _save_cache()
    else:
        procs, logs = [], {}
        t0 = time.monotonic()
        for i in range(n_gpu * procs_per_gpu):
            g = i % n_gpu
            log = open(f"/tmp/bench_{i}.log", "w")
            cmd = (f"{ISAAC_SH} -p sim/twin/modal/bench_env.py --gpu {g} --steps {steps} --warmup {warmup} "
                   f"--out /tmp/bench_{i}.json --barrier /tmp/barrier --n-procs {n_gpu * procs_per_gpu}"
                   + (" --png-dir /tmp" if i == 0 else ""))
            procs.append(subprocess.Popen(["bash", "-lc", cmd], stdout=log, stderr=subprocess.STDOUT, cwd=WORK,
                                          env={**os.environ, "PYTHONPATH": WORK, **EULA, **env}))
        for i, p in enumerate(procs):
            try:
                p.wait(timeout=max(60, 1700 - (time.monotonic() - t0)))
            except subprocess.TimeoutExpired:
                p.kill()
            logs[i] = Path(f"/tmp/bench_{i}.log").read_text()[-3000:]
    stop.set()
    telemetry = _telemetry_check()
    for d in smi_peak.values():  # utilisation over the busiest half of the run (timed loops, not startup)
        u = sorted(d.pop("util_samples"))
        d["util_pct_p50_top_half"] = u[len(u) * 3 // 4] if u else None
        d["util_pct_max"] = u[-1] if u else None
    results = []
    for i in range(n_gpu * procs_per_gpu):
        f = Path(f"/tmp/bench_{i}.json")
        results.append(json.loads(f.read_text()) if f.exists() else {"proc": i, "error": "no result",
                                                                    "log_tail": logs.get(i, "")[-1500:]})
    ok = [r for r in results if "steps_per_s_render" in r]
    agg = {"n_gpu": n_gpu, "procs_per_gpu": procs_per_gpu, "n_procs": len(results), "n_ok": len(ok),
           "sum_steps_per_s_render": round(sum(r["steps_per_s_render"] for r in ok), 2),
           "sum_steps_per_s_norender": round(sum(r["steps_per_s_norender"] for r in ok), 2)}
    return {"prep": info, "agg": agg, "per_proc": results, "nvidia_smi_during": smi_peak, "telemetry": telemetry,
            "log_tail_0": logs.get(0, "")[-2500:],
            "images": {p.name: p.read_bytes() for p in sorted(Path("/tmp").glob("bench_*.png"))}}


@app.function(image=image, timeout=3600, volumes={HF: hf_vol}, cpu=4, memory=8192)
def fetch_molmo(repo_id: str = "allenai/MolmoAct2-SO100_101") -> str:
    """Download the MolmoAct2 checkpoint into the HF volume (CPU container: no GPU billed while downloading)."""
    rc, txt = _sh(f"{VENV_PY} -c \"from huggingface_hub import snapshot_download as s; "
                  f"print(s('{repo_id}'))\" && du -sh {HF}/hub/*", timeout=3300,
                  env={"HF_HOME": HF, "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    hf_vol.commit()
    return txt[-2000:]


@app.function(image=image, gpu="RTX-PRO-6000", timeout=1500, volumes={HF: hf_vol}, cpu=8.0, memory=65536)
def bench_molmo(n_calls: int = 20, batches: list[int] | None = None) -> dict:
    """MolmoAct2 latency per chunk through sim/twin/policies/molmoact2.py's worker (bf16, 10 flow steps)."""
    _sh(f"rm -rf {WORK} && cp -a {REPO} {WORK}")
    b = ",".join(str(x) for x in (batches or [1, 8]))
    rc, txt = _sh(f"{VENV_PY} sim/twin/modal/bench_molmo.py --calls {n_calls} --batches {b} --out /tmp/molmo.json",
                  timeout=1400, cwd=WORK, env={"PYTHONPATH": WORK, "HF_HOME": HF, "HF_HUB_OFFLINE": "1"})
    f = Path("/tmp/molmo.json")
    return {"rc": rc, "result": json.loads(f.read_text()) if f.exists() else None, "tail": txt[-3000:]}


# ----------------------------------------------------------------------------- local entrypoints
OUT = LOCAL_REPO / "sim/twin/out/modal"


def _dump(name: str, res: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    imgs = res.pop("images", {}) if isinstance(res, dict) else {}
    for k, v in imgs.items():
        (OUT / k).write_bytes(v)
    (OUT / f"{name}.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str)[:12000])
    print(f"-> {OUT / name}.json" + (f" + {len(imgs)} images" if imgs else ""))


@app.local_entrypoint()
def main(task: str = "diag", gpu: str = "RTX-PRO-6000", procs_per_gpu: int = 1, steps: int = 300,
         variants: str = "gb10fix,nofix", batches: str = "1,8", calls: int = 20, kit_args: str = "<default>"):
    t0 = time.time()
    tag = gpu.replace(":", "x")
    if task == "diag":
        res = diag.with_options(gpu=gpu).remote()
    elif task == "render":
        res = render.with_options(gpu=gpu).remote(variants.split(","))
    elif task == "bench-env":
        n = procs_per_gpu * (int(gpu.split(":")[1]) if ":" in gpu else 1)  # Modal's default is ~1 core: Isaac
        res = bench_env.with_options(gpu=gpu, cpu=min(64.0, 6.0 * n), memory=min(262144, 24576 * n)).remote(
            procs_per_gpu, steps,
                                                     kit_args=None if kit_args == "<default>" else kit_args)
        tag += f"_p{procs_per_gpu}" + ("" if kit_args == "<default>" else "_nofix" if not kit_args else "_kitargs")
    elif task == "fetch-molmo":
        res = {"log": fetch_molmo.remote()}
    elif task == "bench-molmo":
        res = bench_molmo.with_options(gpu=gpu).remote(calls, [int(x) for x in batches.split(",")])
    else:
        raise SystemExit(f"unknown task {task}")
    res = res if isinstance(res, dict) else {"result": res}
    res["client_wall_s"] = round(time.time() - t0, 1)
    _dump(f"{task}_{tag}", res)
