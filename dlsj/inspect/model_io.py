"""Load any stage-1 anomaly model the team has: our `manifest.json`, or a standard Anomalib / Anomalib Studio
OpenVINO export, and (re)calibrate its threshold.

`resolve(path)` auto-detects what `path` is and returns a `core.Manifest`:

* a `manifest.json` written by `dlsj.inspect.train` or by `calibrate` below (uint8 NHWC IR, raw scores, our band);
* a folder or `.xml` of a standard export (`Engine.export(..., ExportType.OPENVINO)` writes
  `<root>/weights/openvino/model.xml`; Anomalib Studio's download is a flat `model.xml` + `evaluation.json` +
  `studio_model.json`). Such an IR takes RGB float32 NCHW in [0, 1] (`[?,3,H,W]`, often dynamic), has resize +
  ImageNet normalisation in the graph, and outputs `pred_score` (min-max normalised to [0, 1], threshold 0.5,
  clamped), `pred_label`, `anomaly_map`, `pred_mask`. On some devices the output names change (`clamp`), so outputs
  are matched by rank/type, not name.

How an export is run (`prepare_model`, used by `detector.OpenVINOBackend`):
* dynamic batch / H / W are reshaped to a static `[B, H, W]` (the NPU needs static shapes);
* the u8 -> f32, /255 and NHWC -> NCHW conversion is folded into the graph with `PrePostProcessor`, so the CPU only
  crops + resizes (cv2, uint8), exactly as for our own exports. Resize/normalisation inside the export stay;
* `score_space="raw"` cuts the graph before the post-processor: the outputs become the model's raw image score and
  raw anomaly map (the input of `Clamp(Multiply(x, a) + b, 0, 1)`), which do not saturate at 1.0. The
  normalisation constants `a, b` are read from the graph, so thresholds given in normalised units (e.g.
  `evaluation.json` `validation_selected_threshold`) are mapped to raw units: raw = (norm - b) / a.

Threshold of an export without our calibration, in order: `evaluation.json` `validation_selected_threshold`
(Anomalib Studio + external validation), `metadata.json` `image_threshold` / `pixel_threshold`, the post-processor
threshold (0.5 normalised). The uncertainty band is threshold +- `band` (normalised units, default 0.05; in raw
space scaled by 1/a). Calibrate it on labelled data instead (writes a manifest.json pointing at the export):

    python -m dlsj.inspect.model_io info      --model ~/inference/models/downloads/brick-patchcore-geometry-v1
    python -m dlsj.inspect.model_io calibrate --model ~/inference/models/downloads/brick-patchcore-geometry-v1 \
        --data outputs/sweeps --crop 200,120,520,440 --input-size 320 --score-space raw --device GPU \
        --out ~/inference/exports/inspect/patchcore_geometry/manifest.json

`--data` is either a folder of saved sweeps (`sweep.json` with a `label`, from `dlsj.loop.sweep_test`) or an image
dataset (`good/`, `minor/`, `bad/`). Sweeps: good samples = every frame of a plain sweep except its single highest
(the glare frame the `topk` rule tolerates); marked samples = the 2nd-highest frame of a marked sweep (the `topk`
rule needs 2 marked frames). Image folders: per frame for `good/`; for marked bricks with >= 3 views the 2nd-highest
view (the mark may be on one face only), else per frame.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import deque
from dataclasses import replace
from pathlib import Path

import numpy as np

from dlsj.inspect import core
from dlsj.inspect.core import Manifest

log = logging.getLogger("inspect.model_io")

DEFAULT_BAND = 0.05  # normalised units, half-width of the uncertainty band around an export's threshold
EXPORT_SIDECARS = ("evaluation.json", "metadata.json", "studio_model.json", "training_config.json")


# ----------------------------------------------------------------------------- graph helpers (need openvino)
def _rank(p) -> int | None:
    return None if p.rank.is_dynamic else p.rank.get_length()


def pick_outputs(model) -> dict:
    """Indices of the score / map / label / mask outputs, by name when it helps, else by rank and type."""
    import openvino as ov

    outs = list(model.outputs)
    names = [" ".join(sorted(o.get_names())).lower() for o in outs]
    is_float = [o.get_element_type() in (ov.Type.f32, ov.Type.f16) for o in outs]
    ranks = [_rank(o.get_partial_shape()) for o in outs]

    def first(pred):
        return next((i for i in range(len(outs)) if pred(i)), None)

    score = first(lambda i: is_float[i] and "score" in names[i] and (ranks[i] or 0) <= 2)
    score = score if score is not None else first(lambda i: is_float[i] and ranks[i] is not None and ranks[i] <= 2)
    amap = first(lambda i: is_float[i] and "map" in names[i] and ranks[i] == 4)
    amap = amap if amap is not None else first(lambda i: is_float[i] and ranks[i] in (3, 4) and i != score)
    label = first(lambda i: not is_float[i] and ranks[i] is not None and ranks[i] <= 2)
    mask = first(lambda i: not is_float[i] and ranks[i] in (3, 4))
    if score is None or amap is None:
        raise ValueError(f"cannot find score/map outputs among {[(n, r) for n, r in zip(names, ranks)]}")
    return {"score": score, "map": amap, "label": label, "mask": mask}


def _const(node) -> float | None:
    if node.get_type_name() != "Constant":
        return None
    d = np.asarray(node.get_data(), dtype=np.float64)
    return float(d.ravel()[0]) if d.size and np.allclose(d, d.ravel()[0]) else None


def _linear_input(node):
    """For `Multiply(x, c)` / `Multiply(c, x)` return (x_output, c)."""
    if node.get_type_name() != "Multiply":
        return None
    a, b = node.input_value(0), node.input_value(1)
    for x, c in ((a, b), (b, a)):
        cv = _const(c.get_node())
        if cv is not None and x.get_node().get_type_name() != "Constant":
            return x, cv
    return None


def find_normalisation(output) -> dict | None:
    """Walk up from an output to Anomalib's min-max normalisation `Clamp(x * a + b, 0, 1)`.

    Returns {"raw": <raw Output>, "scale": a, "offset": b} or None when the pattern is not there.
    """
    seen, todo = set(), deque([(output.get_node(), 0)])
    while todo:
        node, depth = todo.popleft()
        if id(node) in seen or depth > 12:
            continue
        seen.add(id(node))
        if node.get_type_name() == "Clamp":
            attrs = node.get_attributes()
            if abs(float(attrs.get("min", -1)) - 0.0) < 1e-6 and abs(float(attrs.get("max", -1)) - 1.0) < 1e-6:
                add = node.input_value(0).get_node()
                if add.get_type_name() == "Add":
                    for xi, ci in ((0, 1), (1, 0)):
                        off = _const(add.input_value(ci).get_node())
                        lin = _linear_input(add.input_value(xi).get_node())
                        if off is not None and lin is not None:
                            raw, scale = lin
                            return {"raw": raw, "scale": scale, "offset": off}
        for j in range(node.get_input_size()):
            todo.append((node.input_value(j).get_node(), depth + 1))
    return None


def input_format(model) -> str:
    """"u8_nhwc" (our exports: uint8 [B,H,W,3]) or "f32_nchw" (standard Anomalib: float [B,3,H,W] in [0, 1])."""
    import openvino as ov

    p = model.inputs[0]
    shape = p.get_partial_shape()
    if p.get_element_type() == ov.Type.u8:
        return "u8_nhwc"
    if shape.rank.get_length() == 4 and shape[1].is_static and shape[1].get_length() == 3:
        return "f32_nchw"
    if shape.rank.get_length() == 4 and shape[3].is_static and shape[3].get_length() == 3:
        return "f32_nhwc"
    raise ValueError(f"unsupported model input {p.get_element_type()} {shape}")


def cut_raw(model):
    """New model whose outputs are the raw (pre-normalisation) score and map. Returns (model, norm dict)."""
    import openvino as ov

    idx = pick_outputs(model)
    found = {k: find_normalisation(model.outputs[idx[k]]) for k in ("score", "map")}
    missing = [k for k, v in found.items() if v is None]
    if missing:
        raise ValueError(f"no Clamp(x*a+b, 0, 1) normalisation found above the {missing} output(s); the model "
                         "already outputs raw values or uses another post-processor")
    raw_score, raw_map = found["score"]["raw"], found["map"]["raw"]
    raw_score.get_tensor().set_names({"raw_score"})
    raw_map.get_tensor().set_names({"raw_map"})
    new = ov.Model([raw_score, raw_map], model.get_parameters(), "raw_cut")
    return new, {k: [found[k]["scale"], found[k]["offset"]] for k in found}


def read_normalisation(xml: str | Path) -> dict | None:
    import openvino as ov

    model = ov.Core().read_model(str(xml))
    try:
        return cut_raw(model)[1]
    except ValueError:
        return None


def prepare_model(ov_core, xml: str | Path, input_hw: tuple[int, int], batch: int, score_space: str = "raw",
                  fmt: str | None = None):
    """Read an IR and make it run like ours: static [batch, H, W, 3] uint8 RGB input, outputs (score, map).

    `score_space` "normalized" keeps an export's own outputs; "raw" on a normalised export cuts the graph to its
    raw score/map (our own exports already output raw values and are left alone).
    Returns (model, info) with info = {"format", "batch", "outputs": {score, map, label}, "normalisation"}.
    """
    import openvino as ov

    model = ov_core.read_model(str(xml))
    fmt = fmt or input_format(model)
    norm = None
    if score_space == "raw" and fmt != "u8_nhwc":
        try:
            model, norm = cut_raw(model)
        except ValueError as e:
            log.warning("raw scores requested but %s; using the model's own outputs", e)
    h, w = input_hw
    name = model.inputs[0].get_any_name()
    shape = {"u8_nhwc": [batch, h, w, 3], "f32_nhwc": [batch, h, w, 3], "f32_nchw": [batch, 3, h, w]}[fmt]
    try:
        model.reshape({name: shape})
    except Exception as e:  # noqa: BLE001  some graphs (e.g. old PatchCore exports) only work at their exported batch
        exported = model.inputs[0].get_partial_shape()
        if not exported.is_static:
            raise
        log.warning("cannot reshape %s to batch %d (%s); using its exported shape %s", Path(xml).name, batch,
                    str(e).strip().splitlines()[0][:120], exported)
        batch = exported[0].get_length()
    if fmt != "u8_nhwc":
        ppp = ov.preprocess.PrePostProcessor(model)
        ppp.input().tensor().set_element_type(ov.Type.u8).set_layout(ov.Layout("NHWC")).set_shape([batch, h, w, 3])
        ppp.input().model().set_layout(ov.Layout("NCHW" if fmt == "f32_nchw" else "NHWC"))
        ppp.input().preprocess().convert_element_type(ov.Type.f32).scale(255.0)
        model = ppp.build()
    outs = pick_outputs(model)
    return model, {"format": fmt, "batch": batch, "outputs": outs, "normalisation": norm}


# ----------------------------------------------------------------------------- locating + describing a model
def find_xml(path: Path) -> Path:
    if path.suffix == ".xml":
        return path
    for cand in (path / "model.xml", path / "weights" / "openvino" / "model.xml", path / "openvino" / "model.xml"):
        if cand.exists():
            return cand
    xmls = sorted(path.rglob("*.xml"))
    if len(xmls) == 1:
        return xmls[0]
    raise FileNotFoundError(f"no OpenVINO model.xml under {path}" if not xmls else
                            f"several .xml under {path}: {[str(x) for x in xmls]}; pass the .xml")


def _sidecars(xml: Path) -> dict[str, dict]:
    """evaluation.json / metadata.json / ... next to the IR or up to two folders above it (export root)."""
    out: dict[str, dict] = {}
    for d in (xml.parent, xml.parent.parent, xml.parent.parent.parent):
        for name in EXPORT_SIDECARS:
            p = d / name
            if name not in out and p.is_file():
                try:
                    out[name] = json.loads(p.read_text())
                except (OSError, json.JSONDecodeError) as e:
                    log.warning("ignoring %s: %s", p, e)
    return out


def _band(thr: float, w: float, normalised: bool) -> tuple[float, float]:
    lo, hi = thr - w, thr + w
    if normalised:  # clamped to [0, 1]: keep band_high reachable for a saturated score of 1.0
        hi = min(hi, thr + (1.0 - thr) / 2) if thr < 1.0 else thr
        lo = max(lo, 0.0)
    return lo, hi


def export_manifest(xml: Path, crop=None, input_size: tuple[int, int] | None = None, score_space: str = "normalized",
                    band: float = DEFAULT_BAND) -> Manifest:
    """A `Manifest` for a standard export (thresholds from its sidecars; see the module docstring)."""
    import openvino as ov

    side = _sidecars(xml)
    model = ov.Core().read_model(str(xml))
    fmt = input_format(model)
    shape = model.inputs[0].get_partial_shape()
    hw_idx = (1, 2) if fmt.endswith("nhwc") else (2, 3)
    static_hw = all(shape[i].is_static for i in hw_idx)
    if input_size is None:
        if static_hw:
            input_size = (shape[hw_idx[0]].get_length(), shape[hw_idx[1]].get_length())
        else:
            tc = side.get("training_config.json", {})
            sz = tc.get("image_size") or tc.get("input_size")
            input_size = tuple(sz) if isinstance(sz, (list, tuple)) else ((int(sz),) * 2 if sz else (320, 320))
            log.info("model input H/W are dynamic; feeding %s crops (set --input-size to change)", input_size)
    ev, meta, studio = side.get("evaluation.json", {}), side.get("metadata.json", {}), side.get("studio_model.json", {})
    norm = None
    try:
        norm = cut_raw(model)[1]
    except ValueError:
        pass
    has_label = pick_outputs(model)["label"] is not None
    normalised = norm is not None or has_label
    if "validation_selected_threshold" in ev:
        thr_n, src = float(ev["validation_selected_threshold"]), "evaluation.json validation_selected_threshold"
    elif "image_threshold" in meta:
        thr_n, src = float(meta["image_threshold"]), "metadata.json image_threshold"
    elif normalised:
        thr_n, src = float(studio.get("threshold", 0.5)), "post-processor threshold (normalised 0.5)"
    else:
        raise ValueError(f"{xml}: raw-score export with no threshold metadata; run `python -m dlsj.inspect.model_io "
                         "calibrate` on labelled data first")
    pix_n = float(meta.get("pixel_threshold", 0.5)) if normalised else float(meta.get("pixel_threshold", thr_n))
    calib = {"method": "export", "threshold_source": src, "threshold_normalised": thr_n, "band_normalised": band,
             "aggregation": ev.get("aggregation", "")}
    if score_space == "raw" and norm is not None:
        (sa, sb), (ma, mb) = norm["score"], norm["map"]
        thr, w = (thr_n - sb) / sa, band / sa
        lo, hi = _band(thr, w, normalised=False)
        pix, map_min = (pix_n - mb) / ma, (0.0 - mb) / ma
    else:
        if score_space == "raw":
            log.warning("%s has no normalisation to cut; using its own outputs", xml.name)
            score_space = "normalized" if normalised else "raw"
        thr = thr_n
        lo, hi = _band(thr, band, normalised)
        pix, map_min = pix_n, 0.0
    return Manifest(model=str(xml), model_name=str(studio.get("architecture") or meta.get("model") or "anomalib-export"),
                    input_size=tuple(int(v) for v in input_size), crop=crop, threshold=float(thr),
                    band_low=float(lo), band_high=float(hi), map_min=float(map_min), pixel_threshold=float(pix),
                    backbone=str(studio.get("backbone", "")), calibration=calib, input_format=fmt,
                    score_space=score_space, source="anomalib_export", normalisation=norm or {})


def resolve(path: str | Path, crop=None, input_size: tuple[int, int] | None = None, score_space: str | None = None,
            band: float = DEFAULT_BAND) -> tuple[Manifest, Path]:
    """(Manifest, base dir its `model` path is relative to) for a manifest.json, an export folder or a .xml.

    `crop` / `input_size` / `score_space` override what the manifest says (None = keep).
    """
    path = Path(path).expanduser()
    mpath = path if path.suffix == ".json" else (path / "manifest.json" if (path / "manifest.json").is_file() else None)
    if mpath is not None:
        m = Manifest.load(mpath)
        base = mpath.parent
        if score_space and score_space != m.score_space and m.source == "dlsj":
            log.warning("%s: our own exports always output raw scores; ignoring --score-space %s", mpath, score_space)
        elif score_space and score_space != m.score_space:
            raise ValueError(f"{mpath} was calibrated in {m.score_space} space; re-run calibrate with --score-space "
                             f"{score_space}")
    else:
        xml = find_xml(path)
        m = export_manifest(xml, crop, input_size, score_space or "normalized", band)
        base = xml.parent
    if crop is not None:
        m = replace(m, crop=tuple(crop))
    if input_size is not None and tuple(input_size) != tuple(m.input_size):
        m = replace(m, input_size=tuple(input_size))
    return m, base


# ----------------------------------------------------------------------------- calibration
def _second_highest(scores) -> float:
    s = sorted(scores, reverse=True)
    return float(s[1] if len(s) > 1 else s[0])


def sample_scores(insp, data: Path) -> tuple[list[float], list[float], str]:
    """(good scores, marked scores, description) for `calibrate`, from saved sweeps or an image dataset."""
    good, bad = [], []
    sweeps = sorted(p.parent for p in data.rglob("sweep.json"))
    if sweeps:
        for d in sweeps:
            meta = json.loads((d / "sweep.json").read_text())
            label = meta.get("label")
            if label not in ("plain", "circle", "cross"):
                continue
            frames = [core.read_rgb(d / f["file"]) for f in meta["frames"]]
            s = [float(v) for v in insp.score_frames(frames)]
            if label == "plain":
                good += sorted(s)[:-1] if len(s) > 2 else s
            else:
                bad.append(_second_highest(s))
        return good, bad, f"{len(sweeps)} sweeps"
    for kind in ("good", "minor", "bad"):
        for _, paths in sorted(core.group_by_brick(core.list_images(data / kind)).items()):
            s = [float(v) for v in insp.score_frames([core.read_rgb(p) for p in paths])]
            if kind == "good":
                good += s
            elif len(s) >= 3:
                bad.append(_second_highest(s))
            else:
                bad += s
    return good, bad, "image folders"


def calibrate_model(model: str | Path, data: Path, out: Path, device: str = "CPU", crop=None,
                    input_size=None, score_space: str = "raw", band_frac: float = 0.25, stage2: str | None = None
                    ) -> Manifest:
    from dlsj.inspect.detector import AnomalibInspector

    insp = AnomalibInspector.from_model(model, device=device, batch=1, crop=crop, input_size=input_size,
                                        score_space=score_space, stage2="none")
    good, bad, what = sample_scores(insp, data)
    if not good:
        raise SystemExit(f"no plain samples found in {data}")
    cal = core.calibrate(good, bad or None, band_frac=band_frac)
    m = insp.manifest
    xml = Path(m.model) if Path(m.model).is_absolute() else (insp.base_dir / m.model)
    out.parent.mkdir(parents=True, exist_ok=True)
    stats = dict(cal["stats"], data=str(data), samples=what, device=insp.device, previous=m.calibration)
    new = replace(m, model=str(xml.resolve()), threshold=cal["threshold"], band_low=cal["band_low"],
                  band_high=cal["band_high"], calibration=stats,
                  pixel_threshold=m.pixel_threshold if m.pixel_threshold is not None else cal["threshold"],
                  stage2=({"backend": stage2} if stage2 in ("shape", "none") else
                          {"backend": "cnn", "manifest": stage2} if stage2 else m.stage2))
    new.save(out)
    print(f"{what}: {len(good)} good / {len(bad)} marked samples -> threshold {cal['threshold']:.6g} band "
          f"[{cal['band_low']:.6g}, {cal['band_high']:.6g}] ({stats['method']}); wrote {out}")
    return new


# ----------------------------------------------------------------------------- CLI
def _hw(text: str | None):
    if not text:
        return None
    v = [int(x) for x in text.split(",")]
    return (v[0], v[0]) if len(v) == 1 else (v[0], v[1])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("info", "calibrate"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True, help="manifest.json, export folder or model.xml")
        p.add_argument("--crop", default=None, help="x0,y0,x1,y1 ROI in inspect-frame pixels (square for 320x320)")
        p.add_argument("--input-size", default=None, help="H,W (or one number) fed to the model")
        p.add_argument("--score-space", choices=["raw", "normalized"], default=None)
        p.add_argument("--band", type=float, default=DEFAULT_BAND, help="export band half-width (normalised units)")
        p.add_argument("--device", default="CPU")
    c = sub.choices["calibrate"]
    c.add_argument("--data", type=Path, required=True, help="saved sweeps or good/ minor/ bad/ image folders")
    c.add_argument("--out", type=Path, required=True, help="manifest.json to write")
    c.add_argument("--band-frac", type=float, default=0.25)
    c.add_argument("--stage2", default=None, help='"shape" | "none" | CNN classifier.json to record in the manifest')
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    crop = core.parse_box(args.crop)
    if args.cmd == "info":
        m, base = resolve(args.model, crop, _hw(args.input_size), args.score_space, args.band)
        print(json.dumps({"base_dir": str(base), **m.__dict__}, indent=2, default=str))
        return
    calibrate_model(args.model, args.data.expanduser(), args.out.expanduser(), args.device, crop, _hw(args.input_size),
                    args.score_space or "raw", args.band_frac, args.stage2)


if __name__ == "__main__":
    main()
