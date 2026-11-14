"""Residual-RL demo on Modal (RTX PRO 6000): Isaac Lab 3.0 sim + the real MolmoAct2 (frozen base) in one container.

Same scripts as on the Spark (demo_train.py, demo_video.py, run_real_demo.sh); this file only builds the container:
  /opt/isaac        Isaac Sim 6.2.0 + Isaac Lab 3.0.0 (as ~/isaaclab-venv on the Spark), torch 2.11 cu128
  /opt/studio       Physical AI Studio at the training commit with the molmoact2 extra (as training/molmoact2/modal_app.py)
Volumes (all ours): rl-demo-assets (robot USDs from Joe's sim), rl-demo-hf (HF cache: base MolmoAct2),
         rl-demo-ckpt (the LoRA checkpoint, copied in), rl-demo-isaac-cache (shader/kit caches), rl-demo-runs (outputs).

    # once: robot USDs (so101_new_calib_2 / _11 from /home/joe-spark-2/pi05-isaac-so101/assets/converted on the Spark)
    modal volume create rl-demo-assets
    modal volume put rl-demo-assets <dir containing joe/assets/converted> /
    modal run sim/residual_env/modal_demo.py::fetch_base         # base MolmoAct2 weights (pinned) into rl-demo-hf
    modal volume put rl-demo-ckpt <local lora.ckpt> /<run>/<file>.ckpt   # when the LoRA lands
    # 1. can the container render? (short; prints SMOKE_RESULT, verdict OK / BLANK / FROZEN)
    modal run sim/residual_env/modal_demo.py::smoke
    # 2. throughput with the real policy (zero-shot weights, same compute as the LoRA)
    modal run sim/residual_env/modal_demo.py::throughput --envs 8,16
    # 3. the demo, once the LoRA lands (detached; outputs in rl-demo-runs:/<name>/)
    modal run --detach sim/residual_env/modal_demo.py::train --ckpt /<run>/<file>.ckpt \
        --prompts "pick up the brick|...|..." --envs 16 --hours 2
    modal volume get rl-demo-runs /demo_molmo_lora ./rl_demo
"""
import os
import subprocess
import threading
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
CODE = "/opt/residual_env"
STUDIO = "/opt/studio"
ISAAC_PY = "/opt/isaac/bin/python"
MOLMO_PY = f"{STUDIO}/library/.venv/bin/python"
GPU = os.environ.get("RL_DEMO_GPU", "RTX-PRO-6000")
_versions = REPO / "training/molmoact2/versions.env"
VERSIONS = (dict(line.split("=", 1) for line in _versions.read_text().splitlines()
                 if line.strip() and not line.startswith("#")) if _versions.exists() else {})

# Vulkan / EGL ICDs for the NVIDIA driver (the driver libraries themselves come from the host at run time)
_ICD = ('{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.0"}}')
_EGL = '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}'

isaac_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "ca-certificates", "ffmpeg", "libgl1", "libglu1-mesa", "libegl1", "libvulkan1",
                 "vulkan-tools", "libxt6", "libxrandr2", "libxinerama1", "libxcursor1", "libxi6", "libsm6", "libice6",
                 "libxkbcommon0", "libxext6", "libx11-6", "libgomp1", "pciutils")
    .run_commands(
        "mkdir -p /etc/vulkan/icd.d /usr/share/vulkan/icd.d /usr/share/glvnd/egl_vendor.d",
        f"echo '{_ICD}' > /etc/vulkan/icd.d/nvidia_icd.json",
        f"echo '{_ICD}' > /usr/share/vulkan/icd.d/nvidia_icd.json",
        f"echo '{_EGL}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json",
    )
    .pip_install("uv")
    # The Spark's working ~/isaaclab-venv, pinned exactly (pip freeze minus the aarch64 torch/CUDA wheels): isaacsim
    # 6.2.0.0 pins urllib3 2.7 while isaaclab 3.0.0 asks for >=2.8, so a joint resolve fails; isaaclab goes in second
    # with the pins as overrides.
    .add_local_file(HERE / "modal_isaac_overrides.txt", "/opt/isaac_pins.txt", copy=True)
    .run_commands(
        "uv venv /opt/isaac --python 3.12",
        "uv pip install --python /opt/isaac/bin/python --index-strategy unsafe-best-match "
        "--extra-index-url https://pypi.nvidia.com --extra-index-url https://download.pytorch.org/whl/cu128 "
        "-r /opt/isaac_pins.txt --override /opt/isaac_pins.txt 'torch==2.11.0+cu128' 'torchvision==0.26.0+cu128'",
        # isaaclab with its dependencies, conflicts forced to the Spark's working versions (urllib3 2.7, warp 1.19)
        "uv pip install --python /opt/isaac/bin/python --index-strategy unsafe-best-match "
        "--extra-index-url https://pypi.nvidia.com --extra-index-url https://download.pytorch.org/whl/cu128 "
        "'isaaclab==3.0.0' 'torch==2.11.0+cu128' 'torchvision==0.26.0+cu128' --override /opt/isaac_pins.txt",
        # present on the Spark outside pip's metadata (pandas) or as an aarch64-only build (OpenCV without FFmpeg)
        "uv pip install --python /opt/isaac/bin/python 'pandas==3.1.1' pyarrow 'opencv-python-headless<5' "
        "--override /opt/isaac_pins.txt",
        f"OMNI_KIT_ACCEPT_EULA=YES {ISAAC_PY} -c 'import isaaclab, isaacsim, torch; import isaaclab.app; "
        "print(isaaclab.__file__, torch.__version__)'",
    )
    .env({"NVIDIA_DRIVER_CAPABILITIES": "all", "NVIDIA_VISIBLE_DEVICES": "all", "OMNI_KIT_ACCEPT_EULA": "YES",
          "ACCEPT_EULA": "Y", "PYTHONUNBUFFERED": "1", "DLSJ_JOE_ROOT": "/vol/assets/joe"})
)
demo_image = (
    isaac_image
    .run_commands(
        f"GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none --no-checkout "
        f"https://github.com/open-edge-platform/physical-ai-studio.git {STUDIO}",
        f"git -C {STUDIO} fetch --depth 1 origin {VERSIONS.get('STUDIO_COMMIT', 'main')}",
        f"GIT_LFS_SKIP_SMUDGE=1 git -C {STUDIO} checkout --detach {VERSIONS.get('STUDIO_COMMIT', 'main')}",
        f"uv sync --project {STUDIO}/library --frozen --python 3.12 --extra cu128 --extra molmoact2",
        f"uv pip uninstall --python {MOLMO_PY} opencv-python opencv-python-headless",
        f"uv pip install --python {MOLMO_PY} --no-deps 'opencv-python-headless==4.13.0.92'",
        f"{MOLMO_PY} -c 'import peft; from physicalai.policies import MolmoAct2'",
    )
    .env({"HF_HOME": "/vol/hf", "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
          "PYTHONPATH": "/opt/molmoact2"})  # policy_text (128-token text) for LoRA checkpoints
)


def _with_code(img):
    img = img.add_local_dir(HERE, CODE, ignore=["__pycache__", "*.pyc", "out", "results"])
    for f in ("scene_params.json", "servo_params.json", "joint_offsets_fit.json"):  # app.py / calib.py look next to the code
        img = img.add_local_file(REPO / "sim/realism" / f, f"{CODE}/{f}")
    return img.add_local_dir(REPO / "training/molmoact2", "/opt/molmoact2", ignore=["__pycache__", "*.pyc", "runs"])


isaac_image, demo_image = _with_code(isaac_image), _with_code(demo_image)
app = modal.App("dlsj-rl-demo")
assets_vol = modal.Volume.from_name("rl-demo-assets", create_if_missing=True)
out_vol = modal.Volume.from_name("rl-demo-runs", create_if_missing=True)
hf_vol = modal.Volume.from_name("rl-demo-hf", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("rl-demo-ckpt", create_if_missing=True)
cache_vol = modal.Volume.from_name("rl-demo-isaac-cache", create_if_missing=True)
SMOKE_VOLS = {"/vol/assets": assets_vol.read_only(), "/vol/out": out_vol, "/vol/cache": cache_vol}
DEMO_VOLS = {**SMOKE_VOLS, "/vol/hf": hf_vol.read_only(), "/vol/ckpt": ckpt_vol.read_only()}
CACHE_DIRS = ["/root/.cache/nvidia/GLCache", "/root/.cache/ov", "/root/.local/share/ov/data", "/root/.nv/ComputeCache"]


def _link_caches():
    """Kit / shader caches on a volume: the first start compiles RTX shaders (minutes); later starts reuse them."""
    for d in CACHE_DIRS:
        tgt = Path("/vol/cache") / d.strip("/").replace("/", "_")
        tgt.mkdir(parents=True, exist_ok=True)
        p = Path(d)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.is_symlink() or not p.exists():
            p.unlink(missing_ok=True)
            p.symlink_to(tgt)


def _sh(cmd, timeout=None, **kw):
    print("+", cmd if isinstance(cmd, str) else " ".join(cmd), flush=True)
    return subprocess.run(cmd, shell=isinstance(cmd, str), timeout=timeout, **kw)


class _Committer:
    """Commit the output volume every few minutes so a long detached run is visible (and survives a crash)."""

    def __init__(self, every=300):
        self.every, self.stop = every, threading.Event()

    def __enter__(self):
        def loop():
            while not self.stop.wait(self.every):
                try:
                    out_vol.commit()
                except Exception as e:  # noqa: BLE001
                    print("commit failed:", e, flush=True)
        threading.Thread(target=loop, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        out_vol.commit()


def _check_assets():
    usd = Path("/vol/assets/joe/assets/converted/so101_new_calib_2/so101_new_calib.usda")
    if not usd.exists():
        raise SystemExit(f"{usd} missing: `modal volume put rl-demo-assets <dir with joe/assets/converted> /`")


@app.function(image=isaac_image, gpu=GPU, volumes=SMOKE_VOLS, timeout=40 * 60, cpu=8, memory=32768)
def smoke(num_envs: int = 8):
    """Can Isaac Sim render cameras here? Diagnostics + smoke_render.py (verdict OK / BLANK / FROZEN)."""
    _check_assets()
    _link_caches()
    _sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv")
    _sh("ldconfig -p | grep -iE 'GLX_nvidia|EGL_nvidia|libnvidia-gl|libnvidia-vulkan|libnvidia-rtcore|libnvoptix' "
        "|| echo 'NO NVIDIA GRAPHICS LIBS'")
    _sh("vulkaninfo --summary 2>&1 | grep -iE 'deviceName|driverName|apiVersion|ERROR' | head -12")
    out = "/vol/out/smoke"
    t = time.time()
    r = _sh([ISAAC_PY, f"{CODE}/smoke_render.py", "--out", out, "--num_envs", str(num_envs)], timeout=30 * 60,
            cwd=CODE, capture_output=True, text=True)
    lines = (r.stdout + r.stderr).splitlines()
    keep = [x for x in lines if x.startswith(("SMOKE_RESULT", "Traceback", "RuntimeError", "Exception")) or
            "[Error]" in x or "vulkan" in x.lower() or "rtx" in x.lower() and "error" in x.lower()]
    print("\n".join(keep[-60:]))
    print("wall_s", round(time.time() - t), "exit", r.returncode)
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "smoke.log").write_text("\n".join(lines))
    out_vol.commit()
    cache_vol.commit()


def _demo(args, timeout):
    _check_assets()
    _link_caches()
    if not any(Path("/vol/hf/hub").glob("models--allenai--MolmoAct2-SO100_101/refs/main")):
        raise SystemExit("base weights missing: `modal run sim/residual_env/modal_demo.py::fetch_base` first")
    env = dict(os.environ, ISAAC_PY=ISAAC_PY, MOLMO_PY=MOLMO_PY, OUT_ROOT="/vol/out", IPC_ROOT="/tmp/ipc")
    with _Committer():
        r = _sh(["bash", f"{CODE}/run_real_demo.sh", *args], timeout=timeout, cwd=CODE, env=env)
    return r.returncode


@app.function(image=demo_image, gpu=GPU, volumes=DEMO_VOLS, timeout=90 * 60, cpu=16, memory=65536)
def throughput(envs: str = "8,16", minutes: float = 3.0):
    """Env ticks/s with the real policy (zero-shot weights; LoRA has the same compute) and rendered cameras."""
    for n in envs.split(","):
        _demo(["--pretrained", "--envs", n, "--minutes", str(minutes), "--eval_every_min", "1000", "--no_video",
               "--name", f"throughput_N{n}"], timeout=40 * 60)
        _sh(f"grep -E '^TRAIN' /vol/out/throughput_N{n}/train.log | tail -1; "
            f"grep -E 'WORKER_DONE' /vol/out/throughput_N{n}/worker.log | tail -1")


@app.function(image=demo_image, gpu=GPU, volumes=DEMO_VOLS, timeout=24 * 3600, cpu=16, memory=65536)
def train(ckpt: str = "", pretrained: bool = False, prompts: str = "", envs: int = 16, hours: float = 2.0,
          name: str = ""):
    """The demo: run_real_demo.sh (worker + SAC trainer + video). --ckpt is a path inside rl-demo-ckpt."""
    if not ckpt and not pretrained:
        raise SystemExit("--ckpt /<run>/<file>.ckpt (in rl-demo-ckpt) or --pretrained")
    args = ["--envs", str(envs), "--hours", str(hours)]
    if ckpt:
        p = Path("/vol/ckpt") / ckpt.lstrip("/")
        if not p.exists():
            raise SystemExit(f"{p} not found in rl-demo-ckpt (`modal volume put rl-demo-ckpt <file> /<path>`)")
        args += ["--ckpt", str(p)]
    else:
        args += ["--pretrained"]
    if prompts:
        args += ["--prompts", prompts]
    if name:
        args += ["--name", name]
    rc = _demo(args, timeout=int(hours * 3600 + 2 * 3600))
    print("exit", rc, "outputs in rl-demo-runs:/" + (name or ("demo_molmo_lora" if ckpt else "demo_molmo_zeroshot")))


@app.function(image=demo_image, volumes={"/vol/hf": hf_vol}, timeout=60 * 60, cpu=4, memory=16384)
def fetch_base():
    """Download allenai/MolmoAct2-SO100_101 `main` into rl-demo-hf (refs/main recorded for offline use) and check it
    is the revision pinned in training/molmoact2/versions.env. CPU only; the model is public (no token)."""
    os.environ.pop("HF_HUB_OFFLINE", None)
    repo, rev = VERSIONS.get("MOLMOACT2_REPO", "allenai/MolmoAct2-SO100_101"), VERSIONS.get("MOLMOACT2_REVISION", "")
    r = _sh([MOLMO_PY, "-c", f"""
import os
os.environ.pop("HF_HUB_OFFLINE", None)
from huggingface_hub import snapshot_download
p = snapshot_download("{repo}")
print("BASE_READY", p)
if "{rev}" and not p.rstrip("/").endswith("{rev}"):
    print("WARNING: main is", p.rsplit("/", 1)[-1], "not the pinned {rev}")
"""], env={k: v for k, v in os.environ.items() if k != "HF_HUB_OFFLINE"})
    hf_vol.commit()
    return r.returncode
