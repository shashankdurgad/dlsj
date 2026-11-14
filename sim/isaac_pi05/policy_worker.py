"""π0.5 policy worker for the Isaac SO-101 benchmark (separate process; file IPC with rollout.py).

--model ft   : our fine-tune w3joe/pi05-junction-so101-20261114 (Joe's pinned checkpoint best.ckpt, rev 34e0f9d)
--model base : lerobot/pi05_base (pretrained only). It has no SO-101 normalisation stats, so it gets the SO-101
               dataset stats from the fine-tune checkpoint (exactly what physicalai's fine-tuning path does before
               training: `_update_preprocessor_stats`), i.e. the zero-shot starting point of our fine-tune.
Protocol: rollout writes ipc/obs-NNNN.npz (state, wrist, top, inspect, instruction); worker answers
ipc/actions-NNNN.npz (actions (50,6), latency_s). ipc/done ends the worker.
"""
import argparse, json, os, pickle, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import joeenv  # noqa: F401  (torch from ~/isaacsim-venv + Joe's physicalai/lerobot site-packages)
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["ft", "base"], required=True)
ap.add_argument("--ipc", type=Path, required=True)
ap.add_argument("--hparams", default=str(Path(__file__).parent / "ft_hparams.pkl"))
ap.add_argument("--base-path", default="")
args = ap.parse_args()
args.ipc.mkdir(parents=True, exist_ok=True)

from physicalai.data import Observation
from physicalai.policies import Pi05

torch.manual_seed(42)
t0 = time.perf_counter()
hp = pickle.load(open(args.hparams, "rb"))
stats = hp["dataset_stats"]
if args.model == "ft":
    import inspect
    from transformers.initialization import no_init_weights

    saved = torch.load("/home/joe-spark-2/pi05-isaac-so101/model/best.ckpt", map_location="cpu", mmap=True,
                       weights_only=False)
    hparams = {k: v for k, v in saved["hyper_parameters"].items() if k in inspect.signature(Pi05.__init__).parameters}
    hparams.update(pretrained_name_or_path=None, compile_model=False, gradient_checkpointing=False)
    with no_init_weights():
        policy = Pi05(**hparams)
    policy.on_load_checkpoint(saved)
    policy.load_state_dict(saved["state_dict"], strict=True, assign=True)
    del saved
else:
    path = Path(args.base_path)
    policy = Pi05(pretrained_name_or_path=str(path), compile_model=False, gradient_checkpointing=False,
                  num_inference_steps=hp.get("num_inference_steps", 10))
    base_stats_found = policy.model is not None
    if policy.model is None:  # the base ships no dataset stats -> build the model with the SO-101 stats
        policy._initialize_model(stats, path / "model.safetensors")
    policy._update_preprocessor_stats(stats)
    print("BASE", json.dumps({"base_shipped_stats": base_stats_found, "normalization_mode": policy.config.normalization_mode,
                              "image_resolution": policy.config.image_resolution}), flush=True)
import gc
gc.collect()
policy = policy.eval().to("cuda")
policy.requires_grad_(False)
print("POLICY_READY", args.model, f"load {time.perf_counter() - t0:.1f}s", torch.cuda.get_device_name(0), flush=True)
(args.ipc / "ready").write_text(json.dumps({"model": args.model, "load_s": time.perf_counter() - t0}))

i = 0
while True:
    f = args.ipc / f"obs-{i:04d}.npz"
    while not f.exists():
        if (args.ipc / "done").exists():
            print("WORKER_DONE", flush=True)
            sys.exit(0)
        time.sleep(0.02)
    d = np.load(f, allow_pickle=False)
    obs = Observation(
        state=torch.from_numpy(d["state"].copy()).unsqueeze(0),
        images={n: torch.from_numpy(d[n].copy()).permute(2, 0, 1).unsqueeze(0).float() / 255
                for n in ["wrist", "top", "inspect"]},
        task=[str(d["instruction"].item())],
    )
    torch.cuda.synchronize()
    t = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        a = policy.predict_action_chunk(obs)
    torch.cuda.synchronize()
    lat = time.perf_counter() - t
    a = a.float().cpu().numpy()[0]
    assert a.shape == (50, 6) and np.isfinite(a).all(), a.shape
    tmp = args.ipc / f"actions-{i:04d}.tmp.npz"
    np.savez(tmp, actions=a, latency_s=lat)
    os.replace(tmp, args.ipc / f"actions-{i:04d}.npz")
    print(json.dumps({"i": i, "latency_s": round(lat, 4), "min": np.round(a.min(0), 1).tolist(),
                      "max": np.round(a.max(0), 1).tolist()}), flush=True)
    i += 1
