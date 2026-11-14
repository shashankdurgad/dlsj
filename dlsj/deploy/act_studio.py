"""Helpers for putting a LeRobot ACT checkpoint into Intel Physical AI Studio.

Studio's own `physicalai.policies.ACT` re-implements LeRobot's ACT network with the same module names, so
LeRobot weights load into it after a key rename. What does not carry over by itself:

* LeRobot 0.7 keeps normalisation statistics in the saved processor files
  (`policy_preprocessor_step_*_normalizer_processor.safetensors`), not in `model.safetensors`. Studio's
  `ACT(pretrained_name_or_path=...)` only reads `model.safetensors`, so it would silently keep placeholder
  stats (mean 0, std 1). `studio_dataset_stats` turns the processor stats into the `dataset_stats` that
  Studio's `ACT(dataset_stats=...)` takes.
* Studio names cameras after the environment (`top`, not `observation.images.top`) and its inference
  pipeline resizes every camera to one `image_size` with letterboxing. It has no per-camera crop.
  The deploy scripts therefore bake the training crop into the exported graph (see deploy/act/README.md).

Only numpy is imported here, so the logic is testable without torch, LeRobot or Studio.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LEROBOT_IMAGE_PREFIX = "observation.images."
LEROBOT_STATE = "observation.state"
LEROBOT_ACTION = "action"
STUDIO_STATE = "state"
STUDIO_ACTION = "action"
STUDIO_IMAGE_PREFIX = "images."


def camera_name(feature_key: str) -> str:
    """`observation.images.top` -> `top` (Studio's camera/feature name)."""
    if not feature_key.startswith(LEROBOT_IMAGE_PREFIX):
        raise ValueError(f"{feature_key!r} is not a LeRobot camera feature")
    name = feature_key[len(LEROBOT_IMAGE_PREFIX):]
    if not name or "." in name:
        raise ValueError(f"can't derive a Studio camera name from {feature_key!r}")
    return name


def studio_dataset_stats(config: dict, norm: dict[str, np.ndarray]) -> dict[str, dict]:
    """LeRobot `config.json` + normaliser tensors -> Studio `ACT(dataset_stats=...)` argument.

    `norm` is the flat tensor dict of the saved normaliser (keys like `observation.state.mean`).
    Only MEAN_STD checkpoints are handled, which is what LeRobot ACT uses by default; anything else is refused
    rather than half-converted. Feature types are returned as strings ("STATE", "VISUAL", "ACTION").
    """
    norm_map = config.get("normalization_mapping", {})
    for ftype in ("STATE", "VISUAL", "ACTION"):
        if norm_map.get(ftype, "MEAN_STD") != "MEAN_STD":
            raise ValueError(f"only MEAN_STD normalisation is supported, {ftype} uses {norm_map[ftype]}")
    features = {**config["input_features"], **config["output_features"]}
    stats: dict[str, dict] = {}
    for key, feat in features.items():
        ftype = feat["type"]
        if ftype == "STATE" and key == LEROBOT_STATE:
            name = STUDIO_STATE
        elif ftype == "ACTION" and key == LEROBOT_ACTION:
            name = STUDIO_ACTION
        elif ftype == "VISUAL":
            name = camera_name(key)
        else:
            raise ValueError(f"unsupported feature {key!r} of type {ftype}")
        shape = tuple(int(s) for s in feat["shape"])
        mean, std = norm.get(f"{key}.mean"), norm.get(f"{key}.std")
        if mean is None or std is None:
            raise ValueError(f"normaliser has no mean/std for {key!r}")
        mean, std = np.asarray(mean, np.float64).ravel(), np.asarray(std, np.float64).ravel()
        if mean.size != shape[0] or std.size != shape[0]:
            raise ValueError(f"{key}: stats have {mean.size} values, feature has {shape[0]} channels")
        if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(std)) and np.all(std > 0)):
            raise ValueError(f"{key}: non-finite or non-positive stats")
        if name in stats:
            raise ValueError(f"two features map to Studio name {name!r}")
        stats[name] = {"name": name, "type": ftype, "shape": shape, "mean": mean.tolist(), "std": std.tolist()}
    if STUDIO_STATE not in stats or STUDIO_ACTION not in stats:
        raise ValueError("expected observation.state and action features")
    return stats


@dataclass(frozen=True)
class Crop:
    """A training crop: box (x, y, width, height) inside a full camera frame of `source_hw` (height, width)."""

    x: int
    y: int
    width: int
    height: int
    source_hw: tuple[int, int]

    def __post_init__(self) -> None:
        sh, sw = self.source_hw
        if min(self.x, self.y) < 0 or self.width <= 0 or self.height <= 0 \
                or self.x + self.width > sw or self.y + self.height > sh:
            raise ValueError(f"crop box {self.x},{self.y},{self.width},{self.height} doesn't fit a {sw}x{sh} frame")

    @classmethod
    def from_metadata(cls, meta: dict) -> Crop:
        """From the model repo's `checkpoint-metadata.json` `top_camera_crop` entry."""
        x, y, w, h = (int(v) for v in meta["box_xywh"])
        sh, sw = (int(v) for v in meta["source_hw"])
        return cls(x, y, w, h, (sh, sw))

    @classmethod
    def parse(cls, spec: str) -> Crop:
        """"0,0,560,480@640x480" (same syntax as `dlsj.deploy.act_export --crop`, without the key)."""
        try:
            box, src = spec.split("@", 1)
            x, y, w, h = (int(v) for v in box.split(","))
            sw, sh = (int(v) for v in src.lower().split("x"))
        except ValueError:
            raise ValueError(f"bad crop {spec!r}; expected x,y,width,height@WIDTHxHEIGHT") from None
        return cls(x, y, w, h, (sh, sw))

    @property
    def cropped_hw(self) -> tuple[int, int]:
        return self.height, self.width

    def apply_hwc(self, img: np.ndarray) -> np.ndarray:
        """Crop an (H, W, C) full frame; refuses frames of any other size."""
        if tuple(img.shape[:2]) != tuple(self.source_hw):
            raise ValueError(f"expected a {self.source_hw[0]}x{self.source_hw[1]} (HxW) frame, got {img.shape[:2]}")
        return img[self.y:self.y + self.height, self.x:self.x + self.width]

    def apply_chw(self, img: np.ndarray) -> np.ndarray:
        """Crop a (..., C, H, W) full frame; refuses frames of any other size."""
        if tuple(img.shape[-2:]) != tuple(self.source_hw):
            raise ValueError(f"expected a {self.source_hw[0]}x{self.source_hw[1]} (HxW) frame, got {img.shape[-2:]}")
        return img[..., self.y:self.y + self.height, self.x:self.x + self.width]

    def as_dict(self) -> dict:
        return {"box_xywh": [self.x, self.y, self.width, self.height], "source_hw": list(self.source_hw)}


def studio_image_size(stats: dict[str, dict], crops: dict[str, Crop]) -> tuple[int, int]:
    """The single `image_size` Studio's letterbox resize must use so that it never changes a camera frame.

    Studio resizes every camera to one (H, W). Every camera's *full* frame (the crop's source size for cropped
    cameras, the trained size otherwise) must already be that size, or the resize would distort it.
    """
    sizes = set()
    for name, s in stats.items():
        if s["type"] != "VISUAL":
            continue
        trained = tuple(s["shape"][1:3])
        crop = crops.get(name)
        if crop is not None:
            if tuple(crop.cropped_hw) != trained:
                raise ValueError(f"{name}: crop gives {crop.cropped_hw}, the policy was trained on {trained}")
            sizes.add(tuple(crop.source_hw))
        else:
            sizes.add(trained)
    unknown = set(crops) - {n for n, s in stats.items() if s["type"] == "VISUAL"}
    if unknown:
        raise ValueError(f"crops for cameras the policy doesn't use: {sorted(unknown)}")
    if len(sizes) != 1:
        raise ValueError(f"cameras have different full-frame sizes {sorted(sizes)}; Studio resizes them all to one")
    return sizes.pop()


def check_studio_manifest(manifest: dict, cameras: dict[str, tuple[int, int]], state_dim: int,
                          chunk: int, action_dim: int) -> list[str]:
    """Problems with a Studio `manifest.json` for an ACT export that should take full camera frames.

    `cameras` maps Studio camera name -> full frame (H, W). Returns an empty list when it fits.
    """
    problems = []
    if manifest.get("format") != "policy_package":
        problems.append("format is not policy_package")
    if manifest.get("policy", {}).get("name") != "act":
        problems.append(f"policy.name is {manifest.get('policy', {}).get('name')!r}, not 'act'")
    model = manifest.get("model", {})
    if not model.get("artifacts", {}).get("openvino"):
        problems.append("no openvino artifact")
    feats = {f["init_args"]["name"]: f["init_args"] for f in model.get("input_features", [])}
    if STUDIO_STATE not in feats or list(feats[STUDIO_STATE]["shape"]) != [state_dim]:
        problems.append(f"state input is {feats.get(STUDIO_STATE, {}).get('shape')}, expected [{state_dim}]")
    for cam, (h, w) in cameras.items():
        f = feats.get(STUDIO_IMAGE_PREFIX + cam)
        if f is None:
            problems.append(f"no input for camera {cam!r}")
        elif list(f["shape"]) != [3, h, w]:
            problems.append(f"camera {cam!r} input is {f['shape']}, expected [3, {h}, {w}]")
    extra = {k for k in feats if k.startswith(STUDIO_IMAGE_PREFIX)} - {STUDIO_IMAGE_PREFIX + c for c in cameras}
    if extra:
        problems.append(f"unexpected camera inputs {sorted(extra)}")
    outs = [f["init_args"] for f in model.get("output_features", [])]
    if len(outs) != 1 or list(outs[0]["shape"]) != [chunk, action_dim]:
        problems.append(f"output is {[o.get('shape') for o in outs]}, expected [[{chunk}, {action_dim}]]")
    for spec in model.get("preprocessors", []):
        if spec.get("type") == "resize":
            size = spec.get("image_resolution")
            if size is not None and {tuple(size)} != set(cameras.values()):
                problems.append(f"resize preprocessor targets {size}, camera frames are {sorted(set(cameras.values()))}")
    return problems


def sample_frames(lengths: dict[int, int], n: int, seed: int = 0, horizon: int = 0) -> list[tuple[int, int]]:
    """Pick `n` distinct (episode, frame) pairs, spread over episodes, deterministic for a seed.

    Frames are drawn so that at least `horizon` future frames remain in the episode when possible
    (episodes shorter than that are skipped), which lets predictions be compared with recorded actions.
    """
    rng = np.random.default_rng(seed)
    eps = sorted(e for e, length in lengths.items() if length > horizon)
    if not eps:
        raise ValueError(f"no episode is longer than {horizon} frames")
    if n > sum(lengths[e] - horizon for e in eps):
        raise ValueError("not enough frames")
    order = list(rng.permutation(eps))
    picks: set[tuple[int, int]] = set()
    i = 0
    while len(picks) < n:
        ep = int(order[i % len(order)])
        picks.add((ep, int(rng.integers(0, lengths[ep] - horizon))))
        i += 1
    return sorted(picks)


def chunk_mae(pred: np.ndarray, future: np.ndarray, valid: np.ndarray, steps: int) -> np.ndarray:
    """Per-joint MAE between predicted chunks and recorded future actions over the first `steps` steps.

    pred, future: (N, chunk, A); valid: (N, chunk) bool (False past the episode end). Returns (A,).
    """
    pred, future, valid = np.asarray(pred), np.asarray(future), np.asarray(valid, bool)
    if pred.shape != future.shape or valid.shape != pred.shape[:2]:
        raise ValueError(f"shape mismatch {pred.shape} {future.shape} {valid.shape}")
    m = valid[:, :steps]
    if not m.any():
        raise ValueError("no valid steps")
    err = np.abs(pred[:, :steps] - future[:, :steps])
    return (err * m[..., None]).sum(axis=(0, 1)) / m.sum()


def diff_summary(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """max/mean absolute difference and the largest per-chunk max (robot units)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    d = np.abs(a - b)
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()), "p99_abs": float(np.percentile(d, 99))}


def _unstacked_row(src) -> tuple | None:
    """(X, i) if `src` is Unsqueeze(Gather(X, i, axis=0), axis=0) with constant i, else None."""
    node = src.get_node()
    if node.get_type_name() != "Unsqueeze":
        return None
    axes = node.input_value(1).get_node()
    gather = node.input_value(0).get_node()
    if axes.get_type_name() != "Constant" or np.asarray(axes.get_data()).ravel().tolist() != [0] \
            or gather.get_type_name() != "Gather":
        return None
    idx, axis = gather.input_value(1).get_node(), gather.input_value(2).get_node()
    if idx.get_type_name() != "Constant" or axis.get_type_name() != "Constant" \
            or np.asarray(axis.get_data()).ravel().tolist() != [0] or np.asarray(idx.get_data()).size != 1:
        return None
    return gather.input_value(0), int(np.asarray(idx.get_data()).ravel()[0])


def collapse_unstack_restack(model) -> int:
    """Rewrite `Concat(axis 0)` inputs of the form `X[0], X[1], ..., X[N-1]` (each Unsqueeze(Gather(X, i)))
    back to `X`, in place. Returns how many runs were collapsed.

    Studio's ACT builds its encoder tokens with `tokens.extend(list(cam_features))` + `torch.stack`, which traces
    into one Concat with ~870 single-row inputs (2 + 300 + 270 + 300 tokens for our cameras). On the Arc B390
    GPU plugin (OpenVINO 2026.4.1) that Concat fails at inference with `primitive_onednn_base.h:559: could not
    execute a primitive`. Re-stacking all rows of X in order is X itself, so the rewrite is exact.
    """
    from openvino import opset13 as ops

    collapsed = 0
    for op in model.get_ordered_ops():
        if op.get_type_name() != "Concat" or op.get_attributes()["axis"] != 0 or op.get_input_size() < 3:
            continue
        srcs = [i.get_source_output() for i in op.inputs()]
        new_inputs, j, changed = [], 0, False
        while j < len(srcs):
            row = _unstacked_row(srcs[j])
            if row is not None and row[1] == 0:
                x = row[0]
                n = x.get_partial_shape()[0]
                if n.is_static and j + n.get_length() <= len(srcs):
                    k = n.get_length()
                    rows = [_unstacked_row(s) for s in srcs[j:j + k]]
                    if all(r is not None and r[0] == x and r[1] == t for t, r in enumerate(rows)):
                        new_inputs.append(x)
                        j += k
                        collapsed += 1
                        changed = True
                        continue
            new_inputs.append(srcs[j])
            j += 1
        if changed:
            new = ops.concat(new_inputs, 0) if len(new_inputs) > 1 else None
            out = new.output(0) if new is not None else new_inputs[0]
            if new is not None:
                new.set_friendly_name(op.get_friendly_name())
            op.output(0).replace(out)
    model.validate_nodes_and_infer_types()
    return collapsed
