"""LeRobot 0.7 ACT checkpoint -> Physical AI Studio model folder (`model.ckpt` + `exports/openvino`).

Run in the Studio library env on the XPS (`physicalai`, LeRobot is not needed):

    cd ~/inference/physical-ai-studio/library
    nice -n 5 .venv/bin/python ~/inference/dlsj-act/deploy/act/studio_convert.py \
        --policy_path ~/inference/models/act-warp-duplo-417-20261114 \
        --out ~/inference/exports/act_warp_417_studio

What it does, all with Studio's own classes:

1. Builds Studio's native `physicalai.policies.ACT` (the class Studio's backend uses for policy "act") with
   the checkpoint's architecture and the normalisation stats from LeRobot's saved processor files, named the
   Studio way (`state`, `action`, cameras `wrist`/`top`/`inspect`). `ACT(pretrained_name_or_path=...)` is not
   used because it only reads `model.safetensors`; with a LeRobot >= 0.5 checkpoint it would keep placeholder
   stats (mean 0, std 1) and only log a warning.
2. Loads the LeRobot weights with Studio's own key remap (`_remap_lerobot_act_state_dict`) and requires that
   every network weight is matched (only normaliser buffers, set from the stats, may be absent from the file).
3. Saves `model.ckpt` the way Studio's `to_torch` writes Lightning checkpoints, then reloads it with
   `ACT.load_from_checkpoint` and checks the reloaded policy gives the same actions.
4. Exports with Studio's `policy.export(..., backend="openvino")`, with the top-camera training crop baked
   into the graph: Studio's inference letterbox-resizes every camera to one `image_size` and has no
   per-camera crop, so `image_size` is set to the full 480x640 frame (a no-op resize for all three cameras),
   the graph's `images.top` input takes the full 640x480 frame, and the first op slices x 0-560.
   `crop.json` next to the manifest records that.
5. Runs OpenVINO's ConstantFolding pass on the saved IR. Studio's ACT trace (torch 2.11) leaves the decoder's
   input-independent first self-attention (it runs on a zeros tensor) and the sinusoidal position embeddings as
   live subgraphs, and the Arc B390 GPU plugin (OpenVINO 2026.4.1) then fails to compile the model:
   `[GPU] ProgramBuilder build failed! ... primitive_onednn_base.h:559: could not execute a primitive`
   (bisected to `decoder.layers.0/aten::add/Add_1`, zeros + self-attention). Folding those constants is output-
   equivalent and the folded IR compiles on GPU; CPU parity is re-checked after it.
6. Collapses the encoder-token re-stacking (`collapse_unstack_restack`): Studio's ACT traces its 872 encoder
   tokens as one 872-input Concat of single rows, which then fails at GPU *inference* with the same onednn error
   (bisected to `_model/aten::stack/Concat`). Concatenating all rows of a tensor in order is the tensor, so the
   three camera runs are replaced by the camera token tensors themselves (exact; CPU output unchanged, 0.0).
"""

import argparse
import copy
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for dlsj.deploy.act_studio

from dlsj.deploy.act_studio import (  # noqa: E402
    STUDIO_IMAGE_PREFIX,
    Crop,
    check_studio_manifest,
    collapse_unstack_restack,
    studio_dataset_stats,
    studio_image_size,
)


def _read_normaliser(policy_path: Path) -> dict[str, np.ndarray]:
    from safetensors.numpy import load_file

    pre = json.loads((policy_path / "policy_preprocessor.json").read_text())
    step = next(s for s in pre["steps"] if s["registry_name"] == "normalizer_processor")
    return load_file(str(policy_path / step["state_file"]))


def build_policy(policy_path: Path, crops: dict[str, Crop]):
    """Studio ACT with the LeRobot checkpoint's architecture, stats and weights (CPU, eval)."""
    from physicalai.data import FeatureType
    from physicalai.policies.act.policy import ACT, _remap_lerobot_act_state_dict
    from safetensors.torch import load_file

    cfg = json.loads((policy_path / "config.json").read_text())
    if cfg.get("type") != "act":
        raise ValueError(f"not an ACT checkpoint: {cfg.get('type')}")
    stats = studio_dataset_stats(cfg, _read_normaliser(policy_path))
    image_size = studio_image_size(stats, crops)
    # feature types stay plain strings, as Studio's own `Dataset.stats` stores them: an enum in the hparams would
    # make the checkpoint unloadable with torch.load(weights_only=True), which Lightning uses (FeatureType is a StrEnum)
    arch = {k: cfg[k] for k in (
        "n_obs_steps", "chunk_size", "n_action_steps", "vision_backbone", "replace_final_stride_with_dilation",
        "pre_norm", "dim_model", "n_heads", "dim_feedforward", "feedforward_activation", "n_encoder_layers",
        "n_decoder_layers", "use_vae", "latent_dim", "n_vae_encoder_layers", "temporal_ensemble_coeff",
        "dropout", "kl_weight", "optimizer_lr", "optimizer_weight_decay")}
    arch["optimizer_lr_backbone"] = cfg.get("optimizer_lr_backbone", cfg["optimizer_lr"])
    policy = ACT(
        **arch,
        image_size=image_size,
        # weights come from the checkpoint; None avoids downloading torchvision's ImageNet ResNet-18 first
        pretrained_backbone_weights=None,
        use_imagenet_stats=False,  # use the checkpoint's own image stats as saved (they are ImageNet's anyway)
        dataset_stats=stats,
    )
    sd = load_file(str(policy_path / "model.safetensors"))
    images = [n for n, s in stats.items() if s["type"] == FeatureType.VISUAL.value]
    remapped = _remap_lerobot_act_state_dict(sd, images)
    missing, unexpected = policy.model.load_state_dict(remapped, strict=False)
    # `explain_attention_layer` is Studio's alias of the last decoder layer's cross-attention (same module,
    # so it is filled by the decoder weights); check that rather than trusting the name.
    alias = policy.model._model.explain_attention_layer
    if alias is not policy.model._model.decoder.layers[-1].multihead_attn:
        raise RuntimeError("explain_attention_layer is no longer an alias of the decoder cross-attention")
    net_missing = [k for k in missing if not k.startswith(
        ("_input_normalizer.", "_output_denormalizer.", "_model.explain_attention_layer."))]
    if unexpected or net_missing:
        raise RuntimeError(f"weights don't match Studio's ACT: unexpected={unexpected} missing={net_missing}")
    loaded = sum(int(v.numel()) for k, v in remapped.items())
    return policy.eval(), {"image_size": list(image_size), "weights_loaded": len(remapped),
                           "params_loaded": loaded, "normaliser_buffers_from_stats": sorted(missing)}


class CropInGraph(torch.nn.Module):
    """Wraps Studio's ACTModel so the export takes full camera frames and crops them as the first op."""

    def __init__(self, inner: torch.nn.Module, crops: dict[str, Crop]) -> None:
        super().__init__()
        self.inner = inner
        self.crops = crops  # Studio camera name -> Crop
        feats = dict(inner._config.input_features)
        for cam, crop in crops.items():
            f = feats[cam]
            feats[cam] = dataclasses.replace(f, shape=(f.shape[0], *crop.source_hw))
        self._config = copy.copy(inner._config)
        object.__setattr__(self._config, "input_features", feats)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        batch = dict(batch)
        for cam, c in self.crops.items():
            batch[STUDIO_IMAGE_PREFIX + cam] = batch[STUDIO_IMAGE_PREFIX + cam][..., c.y:c.y + c.height,
                                                                                 c.x:c.x + c.width]
        return self.inner(batch)


def fold_constants(xml: Path) -> dict:
    """Constant-fold a saved IR and collapse its token re-stacking, in place (keeps its precision)."""
    import collections

    import openvino as ov
    from openvino.passes import ConstantFolding, Manager

    core = ov.Core()
    model = core.read_model(str(xml))
    before = collections.Counter(o.get_type_name() for o in model.get_ordered_ops())
    manager = Manager()
    manager.register_pass(ConstantFolding())
    manager.run_passes(model)
    collapsed = collapse_unstack_restack(model)
    after = collections.Counter(o.get_type_name() for o in model.get_ordered_ops())
    tmp = xml.with_name("folded.xml")
    ov.save_model(model, str(tmp), compress_to_fp16=False)  # Studio's ACT export is FP32; INT8 constants stay INT8
    del model
    tmp.replace(xml)
    tmp.with_suffix(".bin").replace(xml.with_suffix(".bin"))
    changed = {k: [before[k], after[k]] for k in sorted(set(before) | set(after)) if before[k] != after[k]}
    return {"restack_runs_collapsed": collapsed, "op_counts_before_after": changed}


def save_checkpoint(policy, path: Path) -> None:
    """Same dict `ExportablePolicyMixin.to_torch` writes; `ACT.load_from_checkpoint` reads it."""
    import lightning

    ckpt = {"state_dict": policy.state_dict(), "epoch": 0, "global_step": 0,
            "pytorch-lightning_version": lightning.__version__, "loops": {}, "hparams_name": "kwargs",
            "hyper_parameters": dict(policy.hparams)}
    policy.on_save_checkpoint(ckpt)
    torch.save(ckpt, str(path))


def _sample(policy, crops: dict[str, Crop], full: bool, seed: int = 0) -> dict[str, torch.Tensor]:
    """A deterministic float observation in Studio's model-input format (images in [0,1], CHW)."""
    g = torch.Generator().manual_seed(seed)
    batch = {"state": torch.randn(1, 6, generator=g) * 30}
    for cam, f in policy.model._config.image_features.items():
        h, w = crops[cam].source_hw if (full and cam in crops) else f.shape[1:]
        batch[STUDIO_IMAGE_PREFIX + cam] = torch.rand(1, 3, h, w, generator=g)
    return batch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy_path", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="model folder to create (model.ckpt + exports/)")
    ap.add_argument("--crop", default=None, help="top crop x,y,w,h@WxH (default: checkpoint-metadata.json)")
    ap.add_argument("--int8", action="store_true", help="also apply Studio's INT8_SYM weight-compression hook")
    args = ap.parse_args()

    from physicalai.policies.act.policy import ACT

    pp = args.policy_path.expanduser()
    meta = json.loads((pp / "checkpoint-metadata.json").read_text())
    crops = {"top": Crop.parse(args.crop) if args.crop else Crop.from_metadata(meta["top_camera_crop"])}
    t0 = time.perf_counter()
    policy, info = build_policy(pp, crops)

    out = args.out.expanduser()
    (out / "exports" / "openvino").mkdir(parents=True, exist_ok=True)
    save_checkpoint(policy, out / "model.ckpt")
    reloaded = ACT.load_from_checkpoint(str(out / "model.ckpt"), map_location="cpu").eval()
    sample = _sample(policy, crops, full=False)
    with torch.no_grad():
        a = policy.model.predict_action_chunk({k: v.clone() for k, v in sample.items()})
        b = reloaded.model.predict_action_chunk({k: v.clone() for k, v in sample.items()})
    info["ckpt_reload_max_abs_diff"] = float((a - b).abs().max())
    if info["ckpt_reload_max_abs_diff"] > 1e-4:
        raise RuntimeError(f"model.ckpt does not reproduce the policy: {info['ckpt_reload_max_abs_diff']}")

    # crop-in-graph equivalence in torch before exporting
    wrapper = CropInGraph(policy.model, crops).eval()
    full = _sample(policy, crops, full=True, seed=1)
    cropped = {k: (crops[k[len(STUDIO_IMAGE_PREFIX):]].apply_chw(v) if k[len(STUDIO_IMAGE_PREFIX):] in crops else v)
               for k, v in full.items()}
    with torch.no_grad():
        info["crop_wrapper_max_abs_diff"] = float((wrapper({k: v.clone() for k, v in full.items()})
                                                  - policy.model.predict_action_chunk(cropped)).abs().max())

    hooks = []
    if args.int8:
        from physicalai.export import compress_weights_openvino_int8_sym

        hooks.append(compress_weights_openvino_int8_sym)
    inner = policy.model
    policy.model = wrapper
    try:
        policy.export(str(out / "exports" / "openvino"), backend="openvino", post_export_hooks=hooks)
    finally:
        policy.model = inner
    manifest = json.loads((out / "exports" / "openvino" / "manifest.json").read_text())
    info["constant_folding"] = fold_constants(out / "exports" / "openvino" / manifest["model"]["artifacts"]["openvino"])
    cams = {cam: crops[cam].source_hw if cam in crops else tuple(f.shape[1:])
            for cam, f in inner._config.image_features.items()}
    problems = check_studio_manifest(manifest, cams, 6, policy.config.chunk_size, 6)
    if problems:
        raise RuntimeError(f"Studio manifest problems: {problems}")
    (out / "exports" / "openvino" / "crop.json").write_text(json.dumps(
        {"note": "top-camera training crop is applied inside the graph; feed full frames",
         **{f"{STUDIO_IMAGE_PREFIX}{k}": c.as_dict() for k, c in crops.items()}}, indent=2))
    info["seconds"] = round(time.perf_counter() - t0, 1)
    info["files"] = {str(p.relative_to(out)): p.stat().st_size for p in sorted(out.rglob("*")) if p.is_file()}
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
