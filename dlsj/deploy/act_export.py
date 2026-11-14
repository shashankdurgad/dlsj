"""Export a LeRobot ACT checkpoint to a single OpenVINO model, normalisation included.

The exported graph maps raw robot data straight to raw robot actions:

    state  float32 (1, S)        joint positions in the robot's calibrated units
    <cam>  uint8   (1, H, W, 3)  one RGB frame per camera, in `manifest.json` order
    ->
    actions float32 (1, chunk, A) joint targets in the robot's calibrated units

LeRobot's own normaliser and unnormaliser are traced into the graph, so the laptop runtime needs
neither LeRobot nor torch, and the numbers match `preprocessor -> ACT -> postprocessor` exactly.
Shapes are static (batch 1, fixed image sizes), which the NPU needs.

    uv run python -m dlsj.deploy.act_export --policy_path outputs/train/act/checkpoints/last/pretrained_model \
        --out exports/act --precision fp16            # or fp32 / int8 (NNCF weight-only INT8)

    uv run python -m dlsj.deploy.act_export --benchmark exports/act --device GPU   # on the laptop

Cropping: if a camera was cropped for training (e.g. the overhead camera to x 0-560), pass the same crop at
export, e.g. `--crop observation.images.top=0,0,560,480@640x480` (x, y, width, height of the box, then the
full camera frame size). It is stored in the manifest, and `OpenVINOACT` applies it to full camera frames.
A frame whose size is neither that full size nor the cropped size is rejected rather than fed to the policy.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from lerobot.configs.types import FeatureType
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor.normalize_processor import NormalizerProcessorStep, UnnormalizerProcessorStep
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

MODEL_XML = "act.xml"
MANIFEST = "manifest.json"


class ACTEndToEnd(nn.Module):
    """Raw observation -> raw action chunk, with LeRobot's normalisation inside the graph."""

    def __init__(self, policy: ACTPolicy, normalizer: NormalizerProcessorStep, unnormalizer: UnnormalizerProcessorStep):
        super().__init__()
        cfg = policy.config
        if cfg.env_state_feature:
            raise ValueError("ACT checkpoints with environment-state inputs are not supported")
        if not cfg.robot_state_feature:
            raise ValueError("expected an ACT checkpoint with observation.state")
        self.model = policy.model
        self.image_keys = list(cfg.image_features)
        self.normalizer = normalizer
        self.unnormalizer = unnormalizer

    def forward(self, state: torch.Tensor, *images: torch.Tensor) -> torch.Tensor:
        batch = {OBS_STATE: self.normalizer._apply_transform(state, OBS_STATE, FeatureType.STATE)}
        frames = []
        for key, img in zip(self.image_keys, images, strict=True):
            x = img.permute(0, 3, 1, 2).to(torch.float32) / 255.0  # (B,H,W,3) uint8 -> (B,3,H,W) [0,1]
            frames.append(self.normalizer._apply_transform(x, key, FeatureType.VISUAL))
        batch[OBS_IMAGES] = frames
        actions = self.model(batch)[0]
        return self.unnormalizer._apply_transform(actions, ACTION, FeatureType.ACTION, inverse=True)


def _step(pipeline, cls):
    for step in pipeline.steps:
        if isinstance(step, cls):
            return step
    raise ValueError(f"no {cls.__name__} in the saved processor pipeline")


def load_act(policy_path: str | Path) -> tuple[ACTPolicy, ACTEndToEnd]:
    policy = ACTPolicy.from_pretrained(str(policy_path))
    policy.to("cpu").eval()
    # checkpoints trained on CUDA save device_processor(device="cuda"), which fails to load on the laptop
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(policy_path),
                                         preprocessor_overrides={"device_processor": {"device": "cpu"}})
    norm, unnorm = _step(pre, NormalizerProcessorStep), _step(post, UnnormalizerProcessorStep)
    for step in (norm, unnorm):  # make sure stats live on CPU/float32 before tracing
        step.to(device="cpu", dtype=torch.float32)
    return policy, ACTEndToEnd(policy, norm, unnorm).eval()


def _manifest(policy: ACTPolicy) -> dict:
    cfg = policy.config
    images = {k: list(ft.shape) for k, ft in cfg.image_features.items()}  # (C,H,W)
    return {
        "policy": "act",
        "inputs": {
            "state": {"name": OBS_STATE, "shape": [1, cfg.robot_state_feature.shape[0]], "dtype": "float32"},
            "images": [{"name": k, "shape": [1, s[1], s[2], 3], "dtype": "uint8"} for k, s in images.items()],
        },
        "output": {"name": ACTION, "shape": [1, cfg.chunk_size, cfg.action_feature.shape[0]], "dtype": "float32"},
        "chunk_size": cfg.chunk_size,
        "n_action_steps": cfg.n_action_steps,
    }


def example_inputs(manifest: dict) -> tuple[np.ndarray, ...]:
    ins = manifest["inputs"]
    state = np.zeros(ins["state"]["shape"], dtype=np.float32)
    imgs = [np.zeros(i["shape"], dtype=np.uint8) for i in ins["images"]]
    return (state, *imgs)


def parse_crop(spec: str) -> tuple[str, dict]:
    """"observation.images.top=0,0,560,480@640x480" -> (key, {"box": [x, y, w, h], "source": [H, W]})."""
    try:
        key, rest = spec.split("=", 1)
        box, src = rest.split("@", 1)
        x, y, w, h = (int(v) for v in box.split(","))
        sw, sh = (int(v) for v in src.lower().split("x"))
    except ValueError:
        raise ValueError(f"bad crop {spec!r}; expected key=x,y,width,height@WIDTHxHEIGHT") from None
    if min(x, y) < 0 or w <= 0 or h <= 0 or x + w > sw or y + h > sh:
        raise ValueError(f"crop box {x},{y},{w},{h} doesn't fit a {sw}x{sh} frame")
    return key, {"box": [x, y, w, h], "source": [sh, sw]}


def crop_image(img: np.ndarray, key: str, want_hw: tuple[int, int], crops: dict) -> np.ndarray:
    """Apply the training crop for `key` to a full camera frame; pass through frames already the right size."""
    if tuple(img.shape[:2]) == tuple(want_hw):
        return img
    c = crops.get(key)
    if c is not None and list(img.shape[:2]) == list(c["source"]):
        x, y, w, h = c["box"]
        return img[y:y + h, x:x + w]
    raise ValueError(f"{key}: got a {img.shape[0]}x{img.shape[1]} (HxW) frame, but the policy expects "
                     f"{want_hw[0]}x{want_hw[1]}" + (f", or a {c['source'][0]}x{c['source'][1]} frame to crop"
                                                     if c else " and the export has no crop for this camera"))


def export(policy_path: str | Path, out_dir: str | Path, precision: str = "fp16",
           crops: dict[str, dict] | None = None) -> Path:
    import openvino as ov

    if precision not in ("fp32", "fp16", "int8"):
        raise ValueError("precision must be fp32, fp16 or int8")
    policy, wrapper = load_act(policy_path)
    manifest = _manifest(policy)
    crops = crops or {}
    for key, c in crops.items():
        _x, _y, w, h = c["box"]
        want = next((i["shape"][1:3] for i in manifest["inputs"]["images"] if i["name"] == key), None)
        if want is None:
            raise ValueError(f"crop given for {key}, which the policy doesn't use")
        if [h, w] != list(want):
            raise ValueError(f"crop for {key} is {w}x{h} but the policy was trained on {want[1]}x{want[0]}")
    manifest["crops"] = crops
    ex = tuple(torch.from_numpy(x) for x in example_inputs(manifest))
    shapes = [(list(x.shape), ov.Type.f32 if x.dtype == torch.float32 else ov.Type.u8) for x in ex]
    with torch.no_grad():
        ov_model = ov.convert_model(wrapper, example_input=ex, input=shapes)
    names = ["state"] + [i["name"] for i in manifest["inputs"]["images"]]
    for port, name in zip(ov_model.inputs, names, strict=True):
        port.get_tensor().set_names({name})
    ov_model.outputs[0].get_tensor().set_names({"actions"})
    if precision == "int8":
        import nncf

        ov_model = nncf.compress_weights(ov_model)  # weight-only INT8, no calibration data needed
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ov.save_model(ov_model, out / MODEL_XML, compress_to_fp16=precision == "fp16")
    manifest["precision"] = precision
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2))
    return out / MODEL_XML


class OpenVINOACT:
    """Laptop-side runtime: no torch or LeRobot needed, only openvino + numpy."""

    def __init__(self, export_dir: str | Path, device: str = "GPU", cache_dir: str | None = "ov_cache",
                 inference_precision: str | None = None):
        """`inference_precision` ("f32", "f16", "bf16") overrides the device default. GPUs and ARM CPUs run
        f16 by default, which is normally fine for control; pass "f32" for exact parity checks."""
        import openvino as ov

        export_dir = Path(export_dir)
        self.manifest = json.loads((export_dir / MANIFEST).read_text())
        core = ov.Core()
        if cache_dir:
            core.set_property({"CACHE_DIR": cache_dir})  # avoids slow recompiles, especially on the NPU
        config = {"PERFORMANCE_HINT": "LATENCY"}
        if inference_precision:
            config["INFERENCE_PRECISION_HINT"] = inference_precision
        self.compiled = core.compile_model(str(export_dir / MODEL_XML), device, config)
        self.request = self.compiled.create_infer_request()
        self.image_keys = [i["name"] for i in self.manifest["inputs"]["images"]]
        self.image_hw = {i["name"]: tuple(i["shape"][1:3]) for i in self.manifest["inputs"]["images"]}
        self.crops = self.manifest.get("crops", {})

    def __call__(self, state: np.ndarray, images: dict[str, np.ndarray]) -> np.ndarray:
        """state (S,) float; images {key: (H,W,3) uint8 RGB, full camera frames} -> actions (chunk, A)."""
        inputs = [np.asarray(state, dtype=np.float32)[None]]
        inputs += [np.ascontiguousarray(crop_image(images[k], k, self.image_hw[k], self.crops), dtype=np.uint8)[None]
                   for k in self.image_keys]
        res = self.request.infer(inputs)
        return np.asarray(next(iter(res.values())))[0]


def benchmark(export_dir: str | Path, device: str, iters: int = 50, warmup: int = 5,
              cache_dir: str | None = "ov_cache") -> dict:
    t0 = time.perf_counter()
    runner = OpenVINOACT(export_dir, device, cache_dir=cache_dir)
    compile_s = time.perf_counter() - t0
    ex = example_inputs(runner.manifest)
    rng = np.random.default_rng(0)
    state = rng.normal(size=ex[0].shape[1:]).astype(np.float32)
    images = {k: rng.integers(0, 255, size=x.shape[1:], dtype=np.uint8) for k, x in zip(runner.image_keys, ex[1:], strict=True)}
    for _ in range(warmup):
        runner(state, images)
    times = []
    for _ in range(iters):
        t = time.perf_counter()
        runner(state, images)
        times.append((time.perf_counter() - t) * 1e3)
    t = np.array(times)
    return {"device": device, "compile_s": round(compile_s, 2), "median_ms": round(float(np.median(t)), 2),
            "p90_ms": round(float(np.percentile(t, 90)), 2), "min_ms": round(float(t.min()), 2)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy_path", help="LeRobot ACT pretrained_model directory")
    ap.add_argument("--out", default="exports/act")
    ap.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "int8"])
    ap.add_argument("--crop", action="append", default=[],
                    help="training crop for a camera, key=x,y,width,height@WIDTHxHEIGHT (repeatable)")
    ap.add_argument("--benchmark", help="export directory to benchmark instead of exporting")
    ap.add_argument("--device", default="CPU", help="CPU, GPU, NPU or AUTO (benchmark only)")
    args = ap.parse_args()
    if args.benchmark:
        for dev in args.device.split(","):
            print(json.dumps(benchmark(args.benchmark, dev.strip())))
        return
    if not args.policy_path:
        ap.error("--policy_path is required to export")
    crops = dict(parse_crop(c) for c in args.crop)
    print(f"wrote {export(args.policy_path, args.out, args.precision, crops)}")


if __name__ == "__main__":
    main()
