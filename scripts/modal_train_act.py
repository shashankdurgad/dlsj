"""Train a LeRobot ACT policy on the teleop dataset on a multi-GPU Modal node, tracked in W&B.

The run covers one policy for the whole take: pick, present to the inspection camera, place on a plate.
ACT ignores the task text. The model is LeRobot's standard ACT (ResNet-18 + CVAE, 100-step chunks). It is
the proven, simple baseline, and dlsj.deploy.act_export exports it to OpenVINO.

1. Upload a colour-only copy of the dataset. Depth isn't used by ACT and is ~85% of the size:

       MODAL_PROFILE=desmond-zee modal volume put dlsj-teleop-data data/duplo_teleop_rgb /duplo_teleop_rgb

2. Train detached, so the run keeps going if this laptop sleeps (smoke test: --steps 200 --run act_smoke):

       MODAL_PROFILE=desmond-zee modal run --detach scripts/modal_train_act.py --run act_v1 --gpus 4

   It uses DDP via torchrun on L40S:<gpus>. The batch size is per GPU.

   W&B tracking: LeRobot 0.7 logs to Trackio, not W&B, so this wrapper parses LeRobot's training log lines
   (step, loss, grad norm, lr, timings) and sends them to W&B. It uses the Modal secret `wandb`
   (WANDB_API_KEY). Logging problems never stop training.

3. Fetch the final checkpoint and export it for the XPS:

       MODAL_PROFILE=desmond-zee modal volume get dlsj-act-runs /act_v1/checkpoints/last/pretrained_model ./outputs/act_v1
       python -m dlsj.deploy.act_export --policy_path ./outputs/act_v1/pretrained_model --out exports/act_v1
"""

import re
import subprocess
import threading
import time

import modal

app = modal.App("dlsj-act")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .uv_pip_install("lerobot[dataset,training]==0.7.0", "wandb")
)

data = modal.Volume.from_name("dlsj-teleop-data", create_if_missing=True)
runs = modal.Volume.from_name("dlsj-act-runs", create_if_missing=True)

# LeRobot log line, e.g. "step:1K smpl:32K ep:150 epch:0.68 loss:0.412 grdn:12.3 lr:1.0e-05 updt_s:0.21 data_s:0.01"
_KV = re.compile(r"(?:^|\s)([a-z_]+):(-?[0-9.]+(?:e[-+]?\d+)?)([KMB]?)(?=\s|$)")
LOG_FREQ = 50
_SCALE = {"": 1, "K": 1e3, "M": 1e6, "B": 1e9}


def parse_log_line(line: str) -> dict[str, float] | None:
    """Metrics from a LeRobot training log line, or None if the line isn't one."""
    if "step:" not in line or "loss:" not in line:
        return None
    out = {}
    for key, num, suffix in _KV.findall(line):
        try:
            out[key] = float(num) * _SCALE[suffix]
        except ValueError:
            continue
    return out if "step" in out and "loss" in out else None


@app.function(image=image, gpu="L40S:4", cpu=32, memory=131072, timeout=12 * 3600,
              volumes={"/data": data, "/runs": runs}, secrets=[modal.Secret.from_name("wandb")])
def train(dataset: str = "duplo_teleop_rgb", run: str = "act_v1", steps: int = 30_000, batch_size: int = 16,
          gpus: int = 4, save_freq: int = 5_000, wandb_project: str = "dlsj-act") -> str:
    import wandb

    workers = max(2, 28 // gpus)
    cmd = [
        "torchrun", "--standalone", f"--nproc_per_node={gpus}", "-m", "lerobot.scripts.lerobot_train",
        f"--dataset.repo_id=local/{dataset}", f"--dataset.root=/data/{dataset}", "--dataset.video_backend=pyav",
        "--policy.type=act", "--policy.device=cuda", "--policy.push_to_hub=false",
        f"--output_dir=/runs/{run}", f"--job_name={run}", f"--batch_size={batch_size}", f"--steps={steps}",
        f"--save_freq={save_freq}", f"--log_freq={LOG_FREQ}", f"--num_workers={workers}",
    ]
    print(" ".join(cmd), flush=True)
    config = {"dataset": dataset, "steps": steps, "batch_size_per_gpu": batch_size, "gpus": gpus,
              "global_batch": batch_size * gpus, "policy": "act", "gpu": "L40S"}
    try:
        wb = wandb.init(project=wandb_project, name=run, config=config)
    except Exception as e:  # noqa: BLE001  never let tracking stop training
        print(f"W&B disabled: {e}", flush=True)
        wb = None

    stop = threading.Event()

    def commit_periodically():  # make checkpoints visible outside the container during training
        while not stop.wait(600):
            runs.commit()

    threading.Thread(target=commit_periodically, daemon=True).start()
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    logged = 0
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
            m = parse_log_line(line)
            if wb is not None and m:
                logged += 1  # LeRobot rounds the displayed step (1500 -> "2K"); one line per LOG_FREQ steps
                try:
                    wb.log({k: v for k, v in m.items() if k != "step"}, step=logged * LOG_FREQ)
                except Exception as e:  # noqa: BLE001
                    print(f"W&B log failed: {e}", flush=True)
        code = proc.wait()
    finally:
        stop.set()
        runs.commit()
        if wb is not None:
            wb.finish()
    if code != 0:
        raise RuntimeError(f"lerobot-train exited with {code}")
    return f"{run}: {steps} steps on {gpus}x L40S in {(time.time() - t0) / 60:.1f} min -> dlsj-act-runs:/{run}"


@app.local_entrypoint()
def main(dataset: str = "duplo_teleop_rgb", run: str = "act_v1", steps: int = 30_000, batch_size: int = 16,
         gpus: int = 4, save_freq: int = 5_000):
    fn = train.with_options(gpu=f"L40S:{gpus}")
    print(fn.remote(dataset=dataset, run=run, steps=steps, batch_size=batch_size, gpus=gpus, save_freq=save_freq))
