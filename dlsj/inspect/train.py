"""Train PaDiM/PatchCore on plain bricks, export a static-shape OpenVINO IR for the NPU, calibrate the threshold.

    python -m dlsj.inspect.train --data ~/inference/inspect_data/real --out ~/inference/exports/inspect/padim

`--data` holds full `inspect` frames (what `capture.py` saves): `good/` (plain bricks, training + held-out
calibration) and optionally `bad/` (cross-marked) and `minor/` (circle-marked): both are "marked" for stage 1 and
are ONLY used to calibrate the threshold, never trained on.
Files `<brick>_f<k>.png` are views of one brick; the held-out split is done per brick so views of one brick
never land on both sides. The ROI comes from `--crop` or the dataset's `dataset.json`.

Steps:
1. crop + resize the training frames to the model input with the same code the detector uses (cv2, uint8);
2. `anomalib.engine.Engine.fit` on `anomalib.data.Folder` (normalisation/thresholding in the post-processor
   are off: the IR outputs the raw score, we calibrate it ourselves);
3. `model.to_openvino(input_size=...)`, then make it fully static ([1,H,W,3]) and fold u8→f32, /255 and
   NHWC→NCHW into the graph with `PrePostProcessor`; optional NNCF INT8 (`--int8`);
4. score the held-out good (and bad) bricks with `AnomalibInspector` on the deploy device, median over views;
5. threshold + uncertainty band (`core.calibrate`), written to `manifest.json` with the model path, input size,
   crop, model name and training date.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from dlsj.inspect import core
from dlsj.inspect.core import Manifest

log = logging.getLogger("inspect.train")


def split_bricks(paths, holdout: float, seed: int):
    groups = core.group_by_brick(paths)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n_hold = max(1, round(len(keys) * holdout)) if holdout > 0 else 0
    hold = {k: groups[k] for k in keys[:n_hold]}
    train = {k: groups[k] for k in keys[n_hold:]}
    return train, hold


def write_crops(groups: dict, dst: Path, crop, input_hw) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for paths in groups.values():
        for p in paths:
            core.write_rgb(dst / f"{p.stem}.png", core.prepare(core.read_rgb(p), crop, input_hw))
            n += 1
    return n


def build_model(name: str, backbone: str | None, input_hw, n_features: int | None = None,
                coreset_ratio: float = 0.1):
    from anomalib.models import Padim, Patchcore
    from anomalib.post_processing import PostProcessor

    cls = {"padim": Padim, "patchcore": Patchcore}[name]
    kw = {
        "pre_processor": cls.configure_pre_processor(image_size=tuple(input_hw)),
        "post_processor": PostProcessor(enable_normalization=False, enable_thresholding=False),
        "evaluator": False,
        "visualizer": False,
    }
    if name == "padim":
        kw["backbone"] = backbone or "resnet18"
        kw["n_features"] = n_features  # None = Anomalib default (100 for resnet18); smaller = smaller covariance
    else:
        kw["backbone"] = backbone or "resnet18"  # wide_resnet50_2 is Anomalib's default; resnet18 fits the NPU better
        kw["layers"] = ("layer2", "layer3")
        kw["coreset_sampling_ratio"] = coreset_ratio
    return cls(**kw), kw["backbone"]


def make_blur_separable(model) -> int:
    """Swap Anomalib's 2-D Gaussian blur on the anomaly map (33x33 conv for sigma=4) for two 1-D convs.

    Same result (a Gaussian kernel is the outer product of two 1-D kernels and reflect padding is per-axis),
    but ~16x fewer MACs: on the NPU the 2-D blur alone took 4.5 ms per 3-frame batch.
    """
    import torch
    import torch.nn.functional as F
    from anomalib.models.components.filters import GaussianBlur2d

    class SeparableBlur(torch.nn.Module):
        def __init__(self, blur: GaussianBlur2d):
            super().__init__()
            k = blur.kernel[0, 0]  # (kh, kw), normalised
            self.register_buffer("kx", k.sum(0).view(1, 1, 1, -1).contiguous())
            self.register_buffer("ky", k.sum(1).view(1, 1, -1, 1).contiguous())
            self.mode = blur.border_type
            self.ph, self.pw = (k.shape[0] - 1) // 2, (k.shape[1] - 1) // 2

        def forward(self, x):
            b, c, h, w = x.shape
            x = x.reshape(b * c, 1, h, w)
            x = F.conv2d(F.pad(x, (self.pw, self.pw, 0, 0), mode=self.mode), self.kx)
            x = F.conv2d(F.pad(x, (0, 0, self.ph, self.ph), mode=self.mode), self.ky)
            return x.reshape(b, c, h, w)

    n = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, GaussianBlur2d) and child.kernel.shape[-1] % 2 == 1:
                new = SeparableBlur(child)
                probe = torch.rand(2, child.kernel.shape[0], 64, 64, device=child.kernel.device) * 100
                err = (new(probe) - child(probe)).abs().max().item()
                if err > 1e-3:
                    log.warning("separable blur differs by %.3g from the 2-D blur; keeping the original", err)
                    continue
                setattr(parent, name, new)
                n += 1
    return n


def export_static(model, work: Path, input_hw, int8_crops: np.ndarray | None) -> Path:
    """Anomalib OpenVINO export → static [1,H,W,3] uint8 RGB input with pre-processing in the graph."""
    import openvino as ov
    import torch

    h, w = input_hw
    model = model.cpu().eval()
    log.info("separable blur: replaced %d Gaussian blur module(s)", make_blur_separable(model))

    def direct():
        class Wrap(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                out = self.m(x)
                return out.pred_score, out.anomaly_map

        with torch.no_grad():
            m = ov.convert_model(Wrap(model), example_input=torch.zeros(1, 3, h, w), input=[1, 3, h, w])
        m.outputs[0].get_tensor().set_names({"pred_score"})
        m.outputs[1].get_tensor().set_names({"anomaly_map"})
        return m

    try:
        raw_xml = model.to_openvino(work / "export", input_size=(h, w))
        ov_model = ov.Core().read_model(str(raw_xml))
        # Anomalib's dynamo ONNX export marks the batch dynamic, but for some models (PatchCore) the traced example
        # batch (2) gets baked into internal shapes; then this reshape fails and we convert directly instead.
        ov_model.reshape({ov_model.input(0).get_any_name(): [1, 3, h, w]})
        log.info("anomalib to_openvino: %s, reshaped to static [1,3,%d,%d]", raw_xml, h, w)
    except Exception as e:  # noqa: BLE001
        log.warning("anomalib to_openvino + static reshape failed (%s); converting the PyTorch module directly",
                    str(e).strip().splitlines()[0][:200])
        ov_model = direct()

    log.info("outputs: %s", [(o.get_any_name(), o.get_partial_shape()) for o in ov_model.outputs])

    if int8_crops is not None:
        import nncf

        data = [(c.astype(np.float32) / 255.0).transpose(2, 0, 1)[None] for c in int8_crops]
        # Quantise the CNN backbone; keep MatMuls (PaDiM's Mahalanobis distance, PatchCore's kNN distances) in
        # float: their inputs have a large dynamic range and they decide the score.
        ov_model = nncf.quantize(ov_model, nncf.Dataset(data), subset_size=len(data),
                                 ignored_scope=nncf.IgnoredScope(types=["MatMul"], validate=False))
        log.info("NNCF INT8 post-training quantisation on %d crops (MatMul kept in float)", len(data))

    ppp = ov.preprocess.PrePostProcessor(ov_model)
    ppp.input().tensor().set_element_type(ov.Type.u8).set_layout(ov.Layout("NHWC")).set_shape([1, h, w, 3])
    ppp.input().model().set_layout(ov.Layout("NCHW"))
    ppp.input().preprocess().convert_element_type(ov.Type.f32).scale(255.0)
    ov_model = ppp.build()
    xml = work.parent / "model.xml"
    ov.save_model(ov_model, str(xml), compress_to_fp16=False)
    return xml


def score_groups(insp, groups: dict) -> tuple[list[float], list[tuple[float, ...]], list[float]]:
    """Per-brick median score, per-frame scores and per-brick map medians."""
    med, frames, map_meds = [], [], []
    for paths in groups.values():
        imgs = [core.read_rgb(p) for p in paths]
        crops = np.stack([core.prepare(f, insp.manifest.crop, insp.manifest.input_size) for f in imgs])
        s, m = insp.score_crops(crops)
        med.append(core.aggregate(s))
        frames.append(tuple(float(v) for v in s))
        map_meds.append(float(np.median(m)))
    return med, frames, map_meds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True,
                    help="folder with good/ (and optional bad/ cross, minor/ circle) inspect frames")
    ap.add_argument("--out", type=Path, required=True, help="export folder: model.xml/.bin + manifest.json")
    ap.add_argument("--model", default="padim", choices=["padim", "patchcore"])
    ap.add_argument("--backbone", help="default resnet18")
    ap.add_argument("--n-features", type=int, help="PaDiM: random feature subset size (default 100 for resnet18)")
    ap.add_argument("--coreset-ratio", type=float, default=0.1, help="PatchCore memory-bank fraction kept")
    ap.add_argument("--input-size", type=int, nargs=2, default=[256, 256], metavar=("H", "W"))
    ap.add_argument("--crop", help="ROI x0,y0,x1,y1 in inspect-frame pixels (default: dataset.json, else full frame)")
    ap.add_argument("--holdout", type=float, default=0.2, help="fraction of good bricks held out for calibration")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--accelerator", default="cpu", choices=["cpu", "xpu"],
                    help="training device (cpu default: PaDiM fits in seconds and leaves the GPU to pi0.5)")
    ap.add_argument("--calib-device", default="NPU", help="OpenVINO device used to score the calibration set")
    ap.add_argument("--band-frac", type=float, default=0.25)
    ap.add_argument("--int8", action="store_true", help="EXPERIMENTAL NNCF INT8 PTQ (backbone only). On the XPS NPU (driver 1.38) the INT8 PaDiM IR "
                         "hangs the NPU (ZE_RESULT_ERROR_DEVICE_LOST) and gave no speed-up; it only helps the CPU")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from anomalib.data import Folder
    from anomalib.engine import Engine

    from dlsj.inspect.detector import AnomalibInspector, OpenVINOBackend

    data, out = args.data.expanduser(), args.out.expanduser()
    input_hw = tuple(args.input_size)
    crop = core.parse_box(args.crop)
    meta = data / "dataset.json"
    if crop is None and meta.exists():
        c = json.loads(meta.read_text()).get("crop")
        crop = tuple(c) if c else None
    good = core.list_images(data / "good")
    bad = core.list_images(data / "bad") + core.list_images(data / "minor")  # stage 1: any mark = anomalous
    if not good:
        raise SystemExit(f"no images in {data / 'good'}")
    train_g, hold_g = split_bricks(good, args.holdout, args.seed)
    bad_g = core.group_by_brick(bad)
    log.info("bricks: %d train good, %d held-out good, %d bad; crop=%s input=%s",
             len(train_g), len(hold_g), len(bad_g), crop, input_hw)

    work = out / "work"
    if work.exists():
        shutil.rmtree(work)
    n = write_crops(train_g, work / "crops" / "good", crop, input_hw)
    log.info("wrote %d training crops", n)
    # Lightning's fit loop needs a validation set; give it the held-out plain bricks (no labels are used).
    write_crops(hold_g or dict(list(train_g.items())[:2]), work / "crops" / "holdout_good", crop, input_hw)

    dm = Folder(name="bricks", root=work / "crops", normal_dir="good", abnormal_dir=None,
                normal_test_dir="holdout_good", test_split_mode="from_dir", val_split_mode="same_as_test",
                train_batch_size=32, eval_batch_size=32, num_workers=4, seed=args.seed)
    from lightning.pytorch import seed_everything

    seed_everything(args.seed, workers=True)  # PaDiM's random feature subset and PatchCore's coreset sampling
    model, backbone = build_model(args.model, args.backbone, input_hw, args.n_features, args.coreset_ratio)
    engine_kw = {"default_root_dir": str(work / "results"), "logger": False}
    if args.accelerator == "xpu":
        from anomalib.engine import SingleXPUStrategy, XPUAccelerator

        engine_kw.update(strategy=SingleXPUStrategy(), accelerator=XPUAccelerator())
    else:
        engine_kw.update(accelerator="cpu")
    t0 = time.perf_counter()
    Engine(**engine_kw).fit(model=model, datamodule=dm)
    t_fit = time.perf_counter() - t0
    log.info("fit: %.1f s", t_fit)

    int8_crops = None
    if args.int8:
        import cv2

        int8_crops = np.stack([cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
                               for p in core.list_images(work / "crops" / "good")[:300]])
    t0 = time.perf_counter()
    xml = export_static(model, work, input_hw, int8_crops)
    log.info("export: %s (%.1f s)", xml, time.perf_counter() - t0)

    # Calibrate on the exported IR, on the deploy device, exactly as the detector will run it.
    manifest = Manifest(model=xml.name, model_name=type(model).__name__, input_size=input_hw, crop=crop,
                        threshold=0.0, band_low=0.0, band_high=0.0, backbone=backbone,
                        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    backend = OpenVINOBackend(xml, input_hw, 1, args.calib_device)
    insp = AnomalibInspector(manifest, backend, batch=1, device=backend.device, heatmap=False)
    if not hold_g:
        raise SystemExit("no held-out good bricks to calibrate on: use --holdout > 0 and more good bricks")
    g_scores, _, g_maps = score_groups(insp, hold_g)
    b_scores, _, _ = score_groups(insp, bad_g) if bad_g else ([], [], [])
    cal = core.calibrate(g_scores, b_scores, band_frac=args.band_frac)
    manifest.threshold, manifest.band_low, manifest.band_high = cal["threshold"], cal["band_low"], cal["band_high"]
    manifest.map_min = float(np.median(g_maps))
    manifest.pixel_threshold = cal["threshold"]
    manifest.calibration = {**cal["stats"], "device": backend.device, "band_frac": args.band_frac,
                            "train_bricks": len(train_g), "train_images": n, "fit_seconds": round(t_fit, 2),
                            "int8": args.int8, "n_features": args.n_features, "data": str(data)}
    manifest.save(out / "manifest.json")
    shutil.rmtree(work, ignore_errors=True)  # crops, intermediate export, Lightning output

    def acc(scores, positive):
        lab = [core.decide(s, (s,), manifest.band_low, manifest.band_high) for s in scores]
        want = core.Label.DEFECTIVE if positive else core.Label.PERFECT
        return sum(x is want for x in lab), sum(x is core.Label.UNCERTAIN for x in lab), len(lab)

    log.info("calibration (%s on %s): threshold %.4g band [%.4g, %.4g]", cal["stats"]["method"], backend.device,
             manifest.threshold, manifest.band_low, manifest.band_high)
    (gc, gu, gn), (bc, bu, bn) = acc(g_scores, False), acc(b_scores, True)
    log.info("calibration set (in-sample for the threshold): good %d/%d perfect (%d uncertain); "
             "bad %d/%d defective (%d uncertain)", gc, gn, gu, bc, bn, bu)
    log.info("wrote %s", out / "manifest.json")


if __name__ == "__main__":
    main()
