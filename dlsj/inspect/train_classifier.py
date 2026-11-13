"""Train the stage-2 mark classifier (circle vs cross), export it to OpenVINO for the NPU, evaluate it.

    python -m dlsj.inspect.train_classifier --data data/mark_crops --out ~/inference/exports/inspect/mark_cnn \\
        --arch mobilenet_v3_small --epochs 15 --device NPU [--test data/mark_crops_test]

`--data` (one or more folders) holds `circle/` and `cross/` crops (any depth), e.g. made by `crops.py` from iPhone
photos and from inspect-camera frames. Crops whose names share a `<brick>_f<k>` stem stay on one side of the
train/val split. Augmentation bridges the iPhone -> webcam gap: rotation (any angle), scale/shift, colour jitter
(brightness, contrast, saturation, hue), marker-colour inversion, grey, blur, downscale, JPEG and sensor noise.

Model: torchvision MobileNetV3-small (default) or ResNet18 with ImageNet weights, new 2-class head, AdamW, cosine LR,
CPU by default (the GPU belongs to pi0.5; `--accelerator xpu` to use it). Export: `ov.convert_model` with a static
[1,3,H,W] input and softmax in the graph, then `PrePostProcessor` folds u8 -> f32, /255, ImageNet mean/std and
NHWC -> NCHW into the graph: the IR takes [1,H,W,3] u8 RGB like stage 1 and compiles on the NPU. Writes
`model.xml/.bin` and `classifier.json` (classes, input size, pad_frac, min_confidence, metrics) into `--out`.
Point the stage-1 manifest's `"stage2"` at it ({"backend": "cnn", "manifest": "../mark_cnn/classifier.json"}) or
pass `--stage2 <classifier.json>` to the detector / loop.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from dlsj.inspect import classify, core

log = logging.getLogger("inspect.train_classifier")

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


# ----------------------------------------------------------------------------- data
def load_items(folders) -> list[tuple[Path, int]]:
    items = []
    for f in folders:
        items += classify.labelled_crops(f)
    return items


def split(items, val_frac: float, seed: int):
    groups: dict[str, list] = {}
    for p, y in items:
        key = f"{y}:{core._VIEW_RE.match(p.stem)['brick'] if core._VIEW_RE.match(p.stem) else p.stem}"
        groups.setdefault(key, []).append((p, y))
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n_val = int(round(len(keys) * val_frac))
    val = [it for k in keys[:n_val] for it in groups[k]]
    train = [it for k in keys[n_val:] for it in groups[k]]
    return train, val


def augment(img: np.ndarray, rng: np.random.Generator, size: int) -> np.ndarray:
    """uint8 RGB crop -> augmented uint8 RGB `size`x`size` (cv2/numpy only)."""
    img = classify.to_input(img, size)
    # geometry: any rotation, scale 0.75-1.25, shift +-10 %, flips
    a = rng.uniform(0, 360)
    s = rng.uniform(0.75, 1.25)
    m = cv2.getRotationMatrix2D((size / 2, size / 2), a, s)
    m[:, 2] += rng.uniform(-0.1, 0.1, 2) * size
    img = cv2.warpAffine(img, m, (size, size), borderMode=cv2.BORDER_REFLECT_101)
    if rng.random() < 0.5:
        img = img[:, ::-1]
    f = img.astype(np.float32)
    # colour: marker-colour inversion, hue/sat jitter, brightness/contrast, grey
    if rng.random() < 0.2:
        f = 255 - f
    if rng.random() < 0.8:
        hsv = cv2.cvtColor(np.clip(f, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + rng.uniform(-12, 12)) % 180
        hsv[..., 1] *= rng.uniform(0.6, 1.4)
        f = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    f = (f - 128) * rng.uniform(0.6, 1.4) + 128 + rng.uniform(-40, 40)
    f = f * rng.uniform(0.9, 1.1, 3)  # white-balance shift
    if rng.random() < 0.1:
        f = np.repeat(f.mean(2, keepdims=True), 3, 2)
    img = np.clip(f, 0, 255).astype(np.uint8)
    # optics / sensor: blur, downscale (webcam resolution), JPEG, noise
    if rng.random() < 0.4:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.8))
    if rng.random() < 0.4:
        k = rng.uniform(0.3, 0.8)
        small = cv2.resize(img, (max(8, int(size * k)),) * 2, interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (size, size), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.5:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(30, 95))])
        if ok:
            img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if rng.random() < 0.5:
        img = np.clip(img + rng.normal(0, rng.uniform(2, 10), img.shape), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)


def to_tensor(batch_u8: np.ndarray):
    import torch

    x = torch.from_numpy(batch_u8).permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    return (x - mean) / std


# ----------------------------------------------------------------------------- model
def build(arch: str, pretrained: bool = True):
    import torch
    import torchvision

    if arch == "mobilenet_v3_small":
        w = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        m = torchvision.models.mobilenet_v3_small(weights=w)
        m.classifier[-1] = torch.nn.Linear(m.classifier[-1].in_features, 2)
    elif arch == "resnet18":
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = torchvision.models.resnet18(weights=w)
        m.fc = torch.nn.Linear(m.fc.in_features, 2)
    else:
        raise ValueError(arch)
    return m


def export(model, out: Path, size: int) -> Path:
    import openvino as ov
    import torch

    class WithSoftmax(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return torch.softmax(self.m(x), dim=1)

    model = model.cpu().eval()
    with torch.no_grad():
        ovm = ov.convert_model(WithSoftmax(model), example_input=torch.zeros(1, 3, size, size),
                               input=[1, 3, size, size])
    ovm.outputs[0].get_tensor().set_names({"probs"})
    ppp = ov.preprocess.PrePostProcessor(ovm)
    ppp.input().tensor().set_element_type(ov.Type.u8).set_layout(ov.Layout("NHWC")).set_shape([1, size, size, 3])
    ppp.input().model().set_layout(ov.Layout("NCHW"))
    ppp.input().preprocess().convert_element_type(ov.Type.f32).scale(255.0).mean(list(MEAN)).scale(list(STD))
    ovm = ppp.build()
    xml = out / "model.xml"
    ov.save_model(ovm, str(xml), compress_to_fp16=True)
    return xml


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, nargs="+", required=True, help="crop folder(s) with circle/ and cross/")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--test", type=Path, nargs="*", default=[], help="held-out crop folder(s) to report on")
    ap.add_argument("--arch", default="mobilenet_v3_small", choices=["mobilenet_v3_small", "resnet18"])
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8, help="torch CPU threads")
    ap.add_argument("--accelerator", default="cpu", choices=["cpu", "xpu"])
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--device", default="NPU", help="OpenVINO device for the post-export evaluation")
    ap.add_argument("--min-confidence", type=float, default=0.7)
    ap.add_argument("--pad-frac", type=float, default=0.3, help="crop padding the crops were made with")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    import torch

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)
    items = load_items(args.data)
    if not items:
        raise SystemExit(f"no crops in {args.data} (expected circle/ and cross/)")
    train, val = split(items, args.val_frac, args.seed)
    log.info("crops: %d train, %d val (circle %d, cross %d)", len(train), len(val),
             sum(y == 0 for _, y in items), sum(y == 1 for _, y in items))
    imgs = {p: classify.to_input(core.read_rgb(p), args.size) for p, _ in items}

    dev = torch.device("xpu" if args.accelerator == "xpu" else "cpu")
    model = build(args.arch, not args.no_pretrained).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * math.ceil(len(train) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(steps, 1), pct_start=0.15)
    n_c = sum(y == 1 for _, y in train)
    w = torch.tensor([len(train) / max(2 * (len(train) - n_c), 1), len(train) / max(2 * n_c, 1)], dtype=torch.float32)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w.to(dev), label_smoothing=0.05)

    def evaluate_torch(subset):
        if not subset:
            return float("nan")
        model.eval()
        correct = 0
        with torch.no_grad():
            for i in range(0, len(subset), 128):
                chunk = subset[i:i + 128]
                x = to_tensor(np.stack([imgs[p] for p, _ in chunk])).to(dev)
                pred = model(x).argmax(1).cpu().numpy()
                correct += int((pred == np.array([y for _, y in chunk])).sum())
        return correct / len(subset)

    t0 = time.perf_counter()
    for ep in range(args.epochs):
        model.train()
        order = rng.permutation(len(train))
        tot, nb = 0.0, 0
        for i in range(0, len(order), args.batch):
            idx = order[i:i + args.batch]
            xb = np.stack([augment(imgs[train[j][0]], rng, args.size) for j in idx])
            yb = torch.tensor([train[j][1] for j in idx])
            loss = loss_fn(model(to_tensor(xb).to(dev)), yb.to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot, nb = tot + float(loss.detach()), nb + 1
        log.info("epoch %d/%d loss %.4f val acc %.3f (%.0f s)", ep + 1, args.epochs, tot / max(nb, 1),
                 evaluate_torch(val), time.perf_counter() - t0)
    t_fit = time.perf_counter() - t0
    val_acc = evaluate_torch(val)

    out = args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    xml = export(model, out, args.size)
    cfg = {"backend": "cnn", "model": xml.name, "arch": args.arch, "input_size": [args.size, args.size],
           "classes": list(classify.CLASSES), "min_confidence": args.min_confidence, "pad_frac": args.pad_frac,
           "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "metrics": {"train_crops": len(train), "val_crops": len(val), "val_acc_torch": round(val_acc, 4),
                       "epochs": args.epochs, "fit_seconds": round(t_fit, 1)},
           "data": [str(d) for d in args.data]}
    (out / "classifier.json").write_text(json.dumps(cfg, indent=2) + "\n")
    log.info("exported %s; torch val acc %.3f", xml, val_acc)

    # Exported IR on the deploy device, crop by crop (batch 1) like at run time.
    clf = classify.CNNMarkClassifier.from_config(out / "classifier.json", device=args.device)
    for name, subset in [("val", val)] + [(f"test:{t}", load_items([t])) for t in args.test]:
        if not subset:
            continue
        x = np.stack([imgs[p] if p in imgs else classify.to_input(core.read_rgb(p), args.size) for p, _ in subset])
        lat = []
        probs = []
        for xi in x:
            t1 = time.perf_counter()
            probs.append(clf.probs(xi[None])[0])
            lat.append((time.perf_counter() - t1) * 1e3)
        pred = np.argmax(np.array(probs), 1)
        y = np.array([yy for _, yy in subset])
        conf = np.max(np.array(probs), 1)
        unk = conf < clf.min_confidence
        cm = np.zeros((2, 3), int)
        for t, pp, u in zip(y, pred, unk):
            cm[t, 2 if u else pp] += 1
        acc = float((pred == y).mean())
        print(f"[{name}] OpenVINO {clf.device}: {len(y)} crops, argmax acc {100 * acc:.1f}%; with min_confidence "
              f"{clf.min_confidence}: rows circle/cross, cols circle/cross/unknown = {cm.tolist()}; "
              f"latency per crop median {np.median(lat):.3f} ms p90 {np.percentile(lat, 90):.3f} ms")
        cfg["metrics"][f"{name}_acc_ov_{clf.device}"] = round(acc, 4)
        cfg["metrics"][f"{name}_latency_ms_median_{clf.device}"] = round(float(np.median(lat)), 3)
    (out / "classifier.json").write_text(json.dumps(cfg, indent=2) + "\n")


if __name__ == "__main__":
    main()
