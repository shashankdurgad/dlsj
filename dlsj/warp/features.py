"""Cache frozen per-frame image features for every episode of a LeRobot v3 dataset.

    python -m dlsj.warp.features --root ~/desmond/data/duplo_teleop --out ~/desmond/warp/feats \
        --cameras top,wrist

Writes <out>/<camera>/episode_XXXXXX.npy (float16 [T, D], one row per dataset frame) and <out>/meta.json.
Backbone: DINOv3 ViT-B/16 as in the paper (gated on the Hub: needs an accepted licence + HF_TOKEN); falls back
to DINOv2 ViT-B/14 automatically. The CLS/pooled embedding is used. Re-running skips finished episodes.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger("warp.features")
BACKBONES = ["facebook/dinov3-vitb16-pretrain-lvd1689m", "facebook/dinov2-base"]


def load_backbone(name: str | None, device: str):
    from transformers import AutoImageProcessor, AutoModel

    for model_id in ([name] if name else BACKBONES):
        try:
            proc = AutoImageProcessor.from_pretrained(model_id)
            model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device).eval()
            log.info("backbone: %s", model_id)
            return model_id, proc, model
        except Exception as e:  # noqa: BLE001  (gated / offline)
            log.warning("could not load %s: %s", model_id, str(e).splitlines()[0][:160])
    raise SystemExit("no backbone could be loaded")


@torch.no_grad()
def embed(frames: torch.Tensor, proc, model, device: str, batch: int = 128) -> np.ndarray:
    """uint8 [T, C, H, W] -> float16 [T, D] pooled embeddings at 224x224."""
    out = []
    for i in range(0, len(frames), batch):
        x = proc(images=list(frames[i: i + batch]), return_tensors="pt", size={"height": 224, "width": 224},
                 do_center_crop=False)["pixel_values"].to(device, torch.bfloat16)
        o = model(pixel_values=x)
        pooled = o.pooler_output if getattr(o, "pooler_output", None) is not None else o.last_hidden_state[:, 0]
        out.append(pooled.float().cpu().numpy().astype(np.float16))
    return np.concatenate(out)


def episode_frames(meta, ep: int, key: str) -> torch.Tensor:
    """All frames of episode `ep` for video `key`, one per dataset frame, uint8 [T, C, H, W].

    Decodes with PyAV (bundles its own FFmpeg; torchcodec needs system FFmpeg libraries). Seeks to the
    episode start inside the concatenated v3 video file, then takes, for each dataset timestamp, the first
    decoded frame at or after it (within half a frame).
    """
    import av

    row = meta.episodes[ep]
    n = int(row["length"])
    t0 = float(row[f"videos/{key}/from_timestamp"])
    want = t0 + np.arange(n) / meta.fps
    half = 0.5 / meta.fps
    out: list[np.ndarray] = []
    with av.open(str(meta.root / meta.get_video_file_path(ep, key))) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        c.seek(int(max(0.0, t0 - 1.0) / s.time_base), stream=s, backward=True)
        last = None
        for fr in c.decode(s):
            t = float(fr.pts * s.time_base)
            if t < want[len(out)] - half:
                continue
            img = fr.to_ndarray(format="rgb24")
            while len(out) < n and t >= want[len(out)] - half:
                out.append(img)
            last = img
            if len(out) >= n:
                break
        while len(out) < n and last is not None:  # video ends a frame early: repeat the last frame
            out.append(last)
    if len(out) < n:
        raise RuntimeError(f"episode {ep} {key}: decoded {len(out)} of {n} frames")
    return torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2)


def main() -> None:
    from lerobot.datasets import LeRobotDatasetMetadata

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="local LeRobot dataset directory")
    p.add_argument("--repo-id", default=None, help="defaults to local/<root name>")
    p.add_argument("--out", required=True)
    p.add_argument("--cameras", default="top,wrist", help="short names; matched against observation.images.<name>")
    p.add_argument("--backbone", default=None, help="Hub id; default tries DINOv3 then DINOv2")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--episodes", default=None, help="e.g. 0-49 for a quick pass")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    root = Path(args.root).expanduser()
    meta = LeRobotDatasetMetadata(args.repo_id or f"local/{root.name}", root=root)
    keys = {c: f"observation.images.{c}" for c in args.cameras.split(",")}
    missing = [k for k in keys.values() if k not in meta.video_keys]
    if missing:
        raise SystemExit(f"cameras {missing} not in dataset video keys {meta.video_keys}")
    eps = range(meta.total_episodes)
    if args.episodes:
        a, b = (int(x) for x in args.episodes.split("-"))
        eps = range(a, b + 1)

    model_id, proc, model = load_backbone(args.backbone, args.device)
    out = Path(args.out).expanduser()
    for c in keys:
        (out / c).mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps({
        "backbone": model_id, "fps": meta.fps, "cameras": list(keys), "root": str(root),
        "total_episodes": meta.total_episodes}, indent=2))

    t_start = time.time()
    for i, ep in enumerate(eps):
        for c, key in keys.items():
            f = out / c / f"episode_{ep:06d}.npy"
            if f.exists():
                continue
            frames = episode_frames(meta, ep, key)
            np.save(f, embed(frames, proc, model, args.device))
        if i % 10 == 0:
            log.info("episode %d/%d (%.0fs)", i + 1, len(eps), time.time() - t_start)
    log.info("done: %s", out)


if __name__ == "__main__":
    main()
