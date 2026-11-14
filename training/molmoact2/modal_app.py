"""MolmoAct2 fine-tune on Modal (4x RTX PRO 6000 in one container), same code path as run.sh on GCP.

The image is Studio at the pinned commit with the molmoact2 extra (as setup.sh). train.py / phase_data.py /
methods.py / augment.py / progress.py run unchanged; Lightning DDP runs inside one container as a subprocess
(`physicalai fit`), which is what Modal recommends for frameworks that re-execute their entrypoint.

Persistent volumes:  molmoact2-hf (Hugging Face cache), molmoact2-data (prepared snapshots),
                     molmoact2-runs (run outputs/checkpoints; resume state lives here).

    # 1. stage a prepared snapshot (prepare_data.py output) once
    modal volume create molmoact2-data
    modal volume put molmoact2-data <local snapshot dir> /molmoact2-cal2-456-3stage-20261114
    # 2. gate: checksum check, smoke run, 200-step timing (timing.json) and OpenVINO export of step 200
    modal run training/molmoact2/modal_app.py::gate --run m2a-modal-20261114 --method ai2 \
        --dataset molmoact2-cal2-456-3stage-20261114
    # 3. full run (detached; re-run the same command to resume after a preemption or the 24 h limit)
    modal run --detach training/molmoact2/modal_app.py::train --run m2a-modal-20261114 --method ai2 \
        --dataset molmoact2-cal2-456-3stage-20261114 --steps 12000
    # 4. fetch results
    modal volume get molmoact2-runs /m2a-modal-20261114/full/checkpoints ./checkpoints

GPU: B300:4 by default (torch swapped to cu130 for it); MOLMOACT2_GPU="H100!:8" / "RTX-PRO-6000:4" (or any Modal GPU string) overrides it at deploy time.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time

import modal

HERE = Path(__file__).resolve().parent
STUDIO = "/opt/studio"
CODE = "/opt/molmoact2"
# Locally this file sits next to versions.env; inside the container Modal runs it from /root, and the code bundle
# (with versions.env) is at CODE.
_versions = HERE / "versions.env" if (HERE / "versions.env").exists() else Path(CODE) / "versions.env"
VERSIONS = dict(line.split("=", 1) for line in _versions.read_text().splitlines()
                if line.strip() and not line.startswith("#"))
GPU = os.environ.get("MOLMOACT2_GPU", "B300:4")
# B300 (sm_103) needs CUDA 13: Studio's lock installs torch 2.11.0+cu128, which has no sm_103 kernels, so for
# B300 the same torch/torchvision versions are swapped for their cu130 builds (one extra image layer).
CUDA13 = GPU.upper().startswith("B300")
TIMEOUT = 24 * 3600  # Modal's maximum function timeout

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libgl1", "curl", "ca-certificates")
    .pip_install("uv")
    .run_commands(
        f"GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none --no-checkout "
        f"https://github.com/open-edge-platform/physical-ai-studio.git {STUDIO}",
        f"git -C {STUDIO} fetch --depth 1 origin {VERSIONS['STUDIO_COMMIT']}",
        f"GIT_LFS_SKIP_SMUDGE=1 git -C {STUDIO} checkout --detach {VERSIONS['STUDIO_COMMIT']}",
        # --frozen, not --locked: Studio's pyproject has a relative `exclude-newer = "7 days"`, so newer uv sees the
        # lock's absolute cutoff as changed and re-resolves (which fails); --frozen installs the lock exactly as written.
        f"uv sync --project {STUDIO}/library --frozen --python 3.12 --extra cu128 --extra molmoact2 --extra loggers",
        # the lock installs opencv-python 5.0 AND opencv-python-headless 4.13 into one cv2/ (import fails): keep headless
        f"uv pip uninstall --python {STUDIO}/library/.venv/bin/python opencv-python opencv-python-headless",
        f"uv pip install --python {STUDIO}/library/.venv/bin/python --no-deps 'opencv-python-headless==4.13.0.92'",
        f"{STUDIO}/library/.venv/bin/python -c 'import peft; from physicalai.policies import MolmoAct2'",
    )
)
if CUDA13:
    image = image.run_commands(
        f"uv pip install --python {STUDIO}/library/.venv/bin/python --index-url https://download.pytorch.org/whl/cu130 "
        "--extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match "
        "'torch==2.11.0+cu130' 'torchvision==0.26.0+cu130'",
        # compiled arch flags are readable without a GPU (get_arch_list() is empty at build time)
        f"{STUDIO}/library/.venv/bin/python -c 'import torch; f = torch._C._cuda_getArchFlags(); "
        "print(torch.__version__, torch.version.cuda, f); assert \"sm_103\" in f or \"sm_100\" in f, f'",
        f"{STUDIO}/library/.venv/bin/python -c 'import peft; from physicalai.policies import MolmoAct2'",
        # NVRTC (PyTorch's runtime-compiled "jiterator" kernels) dlopens libnvrtc-builtins.so.13.* by name; the cu13
        # pip wheels keep it in a nested lib dir that is not on the loader path. Install if absent, then register it.
        f"find {STUDIO}/library/.venv -name 'libnvrtc-builtins.so.13*' | grep -q . "
        f"|| uv pip install --python {STUDIO}/library/.venv/bin/python 'nvidia-cuda-nvrtc>=13.0,<13.1'",
        f"find {STUDIO}/library/.venv -name 'libnvrtc*.so*' -exec dirname {{}} \\; | sort -u > /etc/ld.so.conf.d/nvrtc13.conf "
        "&& cat /etc/ld.so.conf.d/nvrtc13.conf && ldconfig && ldconfig -p | grep nvrtc-builtins",
        f"{STUDIO}/library/.venv/bin/python -c 'import ctypes; ctypes.CDLL(\"libnvrtc-builtins.so.13.0\"); print(\"nvrtc-builtins ok\")'",
    )
image = (
    image.env({"STUDIO_DIR": STUDIO, "HF_HOME": "/vol/hf", "HF_HUB_DISABLE_TELEMETRY": "1", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(HERE, CODE, ignore=["__pycache__", ".studio", "runs", "*.pyc"])
)
hf_vol = modal.Volume.from_name("molmoact2-hf", create_if_missing=True)
data_vol = modal.Volume.from_name("molmoact2-data", create_if_missing=True)
runs_vol = modal.Volume.from_name("molmoact2-runs", create_if_missing=True)
VOLUMES = {"/vol/hf": hf_vol, "/vol/data": data_vol, "/vol/runs": runs_vol}
app = modal.App("molmoact2-finetune", image=image)
PY = f"{STUDIO}/library/.venv/bin/python"


def sh(cmd, **kw):
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(list(map(str, cmd)), **kw)


def pinned_model():
    """Download the pinned MolmoAct2 revision into the HF volume, then go offline (as run.sh)."""
    sh([PY, "-c", f"""
from huggingface_hub import snapshot_download
# Download `main` (as run.sh) so the cache records refs/main, which Studio resolves offline; then check it is the pin.
path = snapshot_download("{VERSIONS['MOLMOACT2_REPO']}")
if not path.rstrip("/").endswith("{VERSIONS['MOLMOACT2_REVISION']}"):
    raise SystemExit(f"{VERSIONS['MOLMOACT2_REPO']} main moved to {{path.rsplit('/', 1)[-1]}}; update versions.env")
print("MODEL_PINNED", path)
"""], check=True)
    hf_vol.commit()
    os.environ["HF_HUB_OFFLINE"] = "1"


def dataset_path(name):
    """The snapshot on the data volume, checksum-verified once (a .verified marker is kept on the volume)."""
    root = Path("/vol/data") / name
    if not (root / "READY.json").exists():
        raise SystemExit(f"/vol/data/{name}/READY.json missing: `modal volume put molmoact2-data <snapshot> /{name}`")
    marker = root / ".verified"
    if not marker.exists():
        sh(["sha256sum", "--quiet", "-c", "SHA256SUMS.txt"], cwd=root, check=True)
        marker.touch()
        data_vol.commit()
    return root


class Committer:
    """Commit the runs volume every few minutes so a preempted container loses at most that much."""

    def __init__(self, every=300):
        self.every, self.stop = every, threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def loop(self):
        while not self.stop.wait(self.every):
            try:
                runs_vol.commit()
            except Exception as exc:  # noqa: BLE001 - keep training if a commit fails
                print("volume commit failed:", exc, flush=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        runs_vol.commit()


def train_py(dataset, output, method, extra=(), wandb_name=None):
    gpus = int(GPU.split(":")[1]) if ":" in GPU else 1
    cmd = [PY, f"{CODE}/train.py", dataset, "--output", output, "--gpus", gpus, "--method", method, *extra]
    return cmd + (["--wandb-name", wandb_name] if wandb_name else [])


WANDB = [modal.Secret.from_name("wandb")]  # `modal secret create wandb WANDB_API_KEY=...`


@app.function(gpu=GPU, volumes=VOLUMES, timeout=4 * 3600, cpu=64, memory=393216, secrets=WANDB)
def gate(run: str, method: str = "ai2", dataset: str = "molmoact2-cal2-456-3stage-20261114", steps: int = 12000):
    if method not in ("ai2", "studio"):
        raise SystemExit("method must be ai2 or studio")
    sh(["nvidia-smi", "-L"], check=True)
    sh([PY, "-c", "import torch; n = torch.cuda.device_count(); x = torch.randn(256, 256, device='cuda', "
        "dtype=torch.bfloat16); print('CUDA_OK', n, torch.cuda.get_device_name(0), "
        "torch.cuda.get_device_capability(0), float((x @ x).float().abs().mean()))"], check=True)
    pinned_model()
    data = dataset_path(dataset)
    out = Path("/vol/runs") / run
    out.mkdir(parents=True, exist_ok=True)
    with Committer():
        if not (out / "smoke.done").exists():
            sh(train_py(data, out / "smoke", method, ["--smoke"]), check=True)
            (out / "smoke.done").touch()
        snap = out / "timing/checkpoints/snapshots/step000200.ckpt"
        if not snap.exists():
            shutil.rmtree(out / "timing", ignore_errors=True)
            sh(train_py(data, out / "timing", method, ["--steps", 200, "--snapshot-every", 200],
                        wandb_name=f"{run}-timing"), check=True)
        p = json.loads((out / "timing/progress.json").read_text())
        p["projected_hours_for_steps"] = {"steps": steps, "hours": round(steps * p["median_s_per_step"] / 3600, 2)}
        p["gpu"] = GPU
        (out / "timing/timing.json").write_text(json.dumps(p, indent=2))
        print("TIMING", json.dumps(p), flush=True)
        if not (out / "openvino/.done").exists():
            shutil.rmtree(out / "openvino", ignore_errors=True)
            sh([f"{STUDIO}/library/.venv/bin/physicalai", "export", "--policy", "physicalai.policies.MolmoAct2",
                "--ckpt_path", snap, "--backend", "openvino", "--output_dir", out / "openvino"], check=True)
            (out / "openvino/.done").touch()
    print(f"GATE_READY: modal volume get molmoact2-runs /{run}/openvino ./openvino-step200 and time it on the XPS GPU")
    return p


@app.function(gpu=GPU, volumes=VOLUMES, timeout=TIMEOUT, cpu=64, memory=393216, secrets=WANDB,
              retries=modal.Retries(max_retries=5, initial_delay=30.0, backoff_coefficient=1.0))
def train(run: str, method: str = "ai2", dataset: str = "molmoact2-cal2-456-3stage-20261114", steps: int = 12000):
    """Full run; resumes from /vol/runs/<run>/full/checkpoints/resume/last.ckpt after preemption/timeout."""
    if method not in ("ai2", "studio"):
        raise SystemExit("method must be ai2 or studio")
    started = time.monotonic()
    sh(["nvidia-smi", "-L"], check=True)
    sh([PY, "-c", "import torch; n = torch.cuda.device_count(); x = torch.randn(256, 256, device='cuda', "
        "dtype=torch.bfloat16); print('CUDA_OK', n, torch.cuda.get_device_name(0), "
        "torch.cuda.get_device_capability(0), float((x @ x).float().abs().mean()))"], check=True)
    pinned_model()
    data = dataset_path(dataset)
    out = Path("/vol/runs") / run / "full"
    runs_vol.reload()
    last = out / "checkpoints/resume/last.ckpt"
    extra = ["--steps", steps]
    if last.exists():
        extra += ["--resume", last]
        print("RESUMING from", last, flush=True)
    elif out.exists():
        shutil.rmtree(out)  # started but no resume checkpoint yet: start clean (train.py refuses an existing output)
    left = TIMEOUT - (time.monotonic() - started) - 1200  # stop cleanly 20 min before the function timeout
    extra += ["--deadline-seconds", int(left)]
    with Committer():
        code = sh(train_py(data, out, method, extra, wandb_name=run)).returncode
    print("TRAINING_FINISHED exit", code, flush=True)
    if code:
        raise RuntimeError(f"training exited {code}")  # triggers a Modal retry, which resumes
    return json.loads((out / "progress.json").read_text()) if (out / "progress.json").exists() else None
