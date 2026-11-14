"""Spin-and-inspect test pipeline: record sweeps on the arm, collect Anomalib training data, replay/evaluate.

    python -m dlsj.loop.sweep_test hardware ...   # sweep a held brick on the real arm, save + inspect it
    python -m dlsj.loop.sweep_test collect  ...   # the same sweep, frames saved as Anomalib training data
    python -m dlsj.loop.sweep_test eval     ...   # (alias: replay) re-run the inspector + every aggregation strategy
    python -m dlsj.loop.sweep_test synth    ...   # synthetic labelled sweeps (dlsj.inspect.synthetic)

hardware (connects the follower with the repo calibration and the cameras; NEVER while someone else uses them):

    python -m dlsj.loop.sweep_test hardware --cameras inspect=4 --pose inspect --close-gripper --label cross \\
        --model ~/inference/models/downloads/brick-patchcore-geometry-v1 --crop 200,120,520,440 --device GPU \\
        --sweep-frames 12 --sweep-range 180 --sweep-settle-ms 150 --repeat 5

  ramps to the saved `inspect` pose (`dlsj.loop.run --save-pose inspect`; `--pose current` holds where the arm is),
  with --close-gripper opens the jaws, waits for Enter while you put a brick in, closes them; then runs the sweep
  (`dlsj.loop.sweep`), infers each frame while the wrist moves on, and saves `outputs/sweeps/<time>_<label>/`:
  frame_XX.png, sweep.json (angles, timing, per-frame results, the decision of every strategy), contact.jpg
  (frames with heatmap, label, score, mark) and scores.png (score per angle with threshold + band).
  Add `--dry-run` to run the same code on the fake workcell (`--fake-brick side_cross|side_circle|glint|plain|...`).

collect (Anomalib's normal set must cover every rotation, so collect plain bricks at all spin angles):

    python -m dlsj.loop.sweep_test collect --cameras inspect=4 --pose inspect --close-gripper --label plain \\
        --dataset ~/inference/inspect_data/real_sweeps --repeat 20
    python -m dlsj.loop.sweep_test collect --from-sweeps outputs/sweeps --dataset ~/inference/inspect_data/real_sweeps

  writes `<dataset>/{good,minor,bad}/<sweep>_f<k>.png` (plain / circle / cross; `_f<k>` groups the views of one brick
  for `dlsj.inspect.train`'s per-brick split) and `<dataset>/collect.jsonl` (angles per sweep).

eval / replay over saved, labelled sweeps (any folder tree containing sweep.json files):

    python -m dlsj.loop.sweep_test eval --sweeps outputs/sweeps --model <manifest.json | export dir | model.xml> \\
        --device NPU [--stage2 shape | --stage2 .../classifier.json] [--crop x0,y0,x1,y1] --report report.html

  prints, per strategy (topk, majority, max, median), the confusion matrix (rows = truth plain/circle/cross,
  columns = plain/circle/cross/unknown), accuracy, unknown rate and "unknown -> purple" accuracy, plus the inference
  latency per sweep; writes an HTML report with every sweep's score strip.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from dlsj.contracts import InspectionResult, Label, MarkType
from dlsj.inspect import aggregate as agg_mod
from dlsj.inspect.aggregate import STRATEGIES, AggregateConfig
from dlsj.loop.sweep import SweepConfig, Sweeper, SweepResult

log = logging.getLogger("sweep_test")

LABELS = ("plain", "circle", "cross")
TRUTH = {"plain": MarkType.NONE, "circle": MarkType.CIRCLE, "cross": MarkType.CROSS}
FOLDER = {"plain": "good", "circle": "minor", "cross": "bad"}
OUTCOMES = (MarkType.NONE, MarkType.CIRCLE, MarkType.CROSS, MarkType.UNKNOWN)
COLOUR = {Label.PERFECT: (62, 207, 142), Label.DEFECTIVE: (255, 122, 200), Label.UNCERTAIN: (255, 200, 87)}  # RGB


# ----------------------------------------------------------------------------- drawing (cv2 + numpy only)
def _cv2():
    import cv2

    return cv2


def contact_sheet(frames, recs, heats=None, cols: int = 4, thumb_w: int = 320) -> np.ndarray:
    """RGB grid of the sweep's frames, heatmap blended in, each captioned angle / score / label / mark."""
    cv2 = _cv2()
    h0, w0 = frames[0].shape[:2]
    th = int(round(thumb_w * h0 / w0))
    tiles = []
    for i, f in enumerate(frames):
        img = f.astype(np.float32)
        if heats is not None and heats[i] is not None and heats[i].shape[:2] == f.shape[:2]:
            hm = np.clip(heats[i], 0, 1)
            col = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1].astype(np.float32)
            a = (0.45 * hm)[..., None]
            img = img * (1 - a) + col * a
        img = np.clip(img, 0, 255).astype(np.uint8)
        r = recs[i] if recs is not None else None
        if r is not None and r.region is not None:
            x0, y0, x1, y1 = r.region
            cv2.rectangle(img, (x0, y0), (x1, y1), COLOUR[r.label], 3)
        img = np.ascontiguousarray(cv2.resize(img, (thumb_w, th), interpolation=cv2.INTER_AREA))
        cv2.rectangle(img, (0, 0), (thumb_w, 24), (0, 0, 0), -1)
        if r is not None:
            ang = "" if r.angle_deg is None else f"{r.angle_deg:+.0f}deg "
            txt = f"{i:02d} {ang}{r.score:.4g} {r.label.value[:4]}"
            if r.mark is not None:
                txt += f" {r.mark.value} {r.mark_confidence:.2f}"
            cv2.putText(img, txt, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOUR[r.label], 1, cv2.LINE_AA)
            cv2.rectangle(img, (0, 0), (thumb_w - 1, th - 1), COLOUR[r.label], 2)
        tiles.append(img)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    return np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])


def score_chart(recs, threshold: float, band_low: float, band_high: float, title: str = "",
                size: tuple[int, int] = (900, 320)) -> np.ndarray:
    """RGB bar chart: one bar per angle (colour = per-frame label), dashed threshold, shaded uncertainty band."""
    cv2 = _cv2()
    W, H = size
    img = np.full((H, W, 3), 255, np.uint8)
    pl, pr, pt, pb = 60, 16, 30, 40
    scores = [r.score for r in recs]
    lo = min(0.0, min(scores), band_low)
    hi = max(max(scores), band_high) * 1.08 if max(max(scores), band_high) > 0 else 1.0

    def y(v):
        return int(round(pt + (H - pt - pb) * (1 - (v - lo) / max(hi - lo, 1e-9))))

    cv2.rectangle(img, (pl, y(band_high)), (W - pr, y(band_low)), (255, 236, 190), -1)
    for x in range(pl, W - pr, 12):
        cv2.line(img, (x, y(threshold)), (min(x + 6, W - pr), y(threshold)), (90, 90, 90), 1)
    n = len(recs)
    bw = (W - pl - pr) / max(n, 1)
    for i, r in enumerate(recs):
        x0, x1 = int(pl + i * bw + 2), int(pl + (i + 1) * bw - 2)
        cv2.rectangle(img, (x0, y(r.score)), (x1, y(lo)), COLOUR[r.label], -1)
        if r.mark is not None:
            m = {"cross": "X", "circle": "O"}.get(r.mark.value, "?")
            cv2.putText(img, m, ((x0 + x1) // 2 - 5, y(r.score) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1,
                        cv2.LINE_AA)
        if r.angle_deg is not None and (n <= 16 or i % 2 == 0):
            cv2.putText(img, f"{r.angle_deg:+.0f}", (x0, H - pb + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 60, 60), 1,
                        cv2.LINE_AA)
    for v in (lo, threshold, hi / 1.08):
        cv2.putText(img, f"{v:.4g}", (2, y(v) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.line(img, (pl, pt), (pl, y(lo)), (0, 0, 0), 1)
    cv2.putText(img, title or "score per wrist_roll angle (deg)", (pl, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (20, 20, 20), 1, cv2.LINE_AA)
    return img


# ----------------------------------------------------------------------------- sweep folders
def strategies_record(results: dict[str, agg_mod.Aggregate]) -> dict:
    return {s: {"outcome": a.outcome().value, "label": a.label.value, "score": round(float(a.score), 6),
                "mark": None if a.mark is None else a.mark.value, "mark_confidence": round(a.mark_confidence, 4),
                "summary": a.summary} for s, a in results.items()}


def save_sweep(out: Path, frames, angles, label: str | None, sweep: SweepResult | None = None,
               insp_info: dict | None = None, recs=None, heats=None, results: dict | None = None,
               meta: dict | None = None) -> Path:
    """Write a sweep folder: frame_XX.png, sweep.json, and (with inspection results) contact.jpg + scores.png."""
    from dlsj.inspect.core import write_rgb

    out.mkdir(parents=True, exist_ok=True)
    fr = []
    stops = {s.index: s for s in sweep.stops} if sweep is not None else {}
    for k, (f, a) in enumerate(zip(frames, angles)):
        name = f"frame_{k:02d}.png"
        write_rgb(out / name, f)
        d = {"i": k, "file": name, "angle": None if a is None else round(float(a), 2)}
        if k in stops:
            s = stops[k]
            d |= {"target": round(s.target_deg, 2), "settled": s.settled, "move_ms": round(s.move_ms, 1),
                  "settle_ms": round(s.settle_ms, 1), "grab_ms": round(s.grab_ms, 1)}
        fr.append(d)
    doc = {"version": 1, "label": label, "created": datetime.now().isoformat(timespec="seconds"),
           "camera": "inspect", "frames": fr, **(meta or {})}
    if sweep is not None:
        doc["sweep"] = sweep.record()
    if recs is not None:
        doc["inspection"] = {**(insp_info or {}), "per_frame": agg_mod.frames_record(recs),
                             "strategies": strategies_record(results or {})}
        _cv2().imwrite(str(out / "contact.jpg"), contact_sheet(frames, recs, heats)[..., ::-1],
                       [_cv2().IMWRITE_JPEG_QUALITY, 85])
        title = f"{label or '?'}: " + ", ".join(f"{s}={a.outcome().value}" for s, a in (results or {}).items())
        info = insp_info or {}
        _cv2().imwrite(str(out / "scores.png"), score_chart(recs, info.get("threshold", 0.5), info.get("band_low", 0.5),
                                                           info.get("band_high", 0.5), title)[..., ::-1])
    (out / "sweep.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    return out


def load_sweep(d: Path) -> tuple[dict, list[np.ndarray], list[float | None]]:
    from dlsj.inspect.core import read_rgb

    meta = json.loads((d / "sweep.json").read_text())
    frames = [read_rgb(d / f["file"]) for f in meta["frames"]]
    return meta, frames, [f.get("angle") for f in meta["frames"]]


def find_sweeps(root: Path) -> list[Path]:
    return sorted(p.parent for p in Path(root).expanduser().rglob("sweep.json"))


def inspector_info(insp) -> dict:
    m = insp.sweep_meta()
    man = getattr(insp, "manifest", None)
    info = {k: m[k] for k in ("threshold", "band_low", "band_high", "device", "mark_device")}
    if man is not None:
        info |= {"model": str(man.model), "model_name": man.model_name, "score_space": man.score_space,
                 "crop": list(man.crop) if man.crop else None, "input_size": list(man.input_size)}
    clf = getattr(insp, "classifier", None)
    info["stage2"] = getattr(clf, "name", "none") if clf is not None else "none"
    return info


def inspect_all(insp, frames, angles, cfg: AggregateConfig, strategies=STRATEGIES):
    """(per-frame records, heatmaps, {strategy: Aggregate}, latency ms) for one sweep."""
    t0 = time.perf_counter()
    out = insp.inspect_frames(frames, angles)
    ms = (time.perf_counter() - t0) * 1e3
    recs, heats = [o[0] for o in out], [o[1] for o in out]
    m = insp.sweep_meta()
    results = agg_mod.aggregate_all(recs, m["threshold"], m["band_low"], m["band_high"], cfg, strategies)
    return recs, heats, results, ms


# ----------------------------------------------------------------------------- collect
def collect_frames(dataset: Path, sweep_id: str, frames, angles, label: str, crop=None) -> list[Path]:
    from dlsj.inspect.core import write_rgb

    d = Path(dataset).expanduser() / FOLDER[label]
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for k, f in enumerate(frames):
        p = d / f"{sweep_id}_f{k}.png"
        write_rgb(p, f)
        paths.append(p)
    root = Path(dataset).expanduser()
    with open(root / "collect.jsonl", "a") as fh:
        fh.write(json.dumps({"sweep": sweep_id, "label": label, "folder": FOLDER[label], "n": len(frames),
                             "angles": [None if a is None else round(float(a), 2) for a in angles],
                             "t": time.time()}) + "\n")
    ds = root / "dataset.json"
    if crop is not None and not ds.exists():
        ds.write_text(json.dumps({"source": "dlsj.loop.sweep_test collect", "crop": list(crop)}, indent=1) + "\n")
    return paths


def collect_from_sweeps(src: Path, dataset: Path, crop=None) -> int:
    n = 0
    for d in find_sweeps(src):
        meta, frames, angles = load_sweep(d)
        if meta.get("label") not in LABELS:
            log.warning("skipping %s: no plain/circle/cross label", d)
            continue
        collect_frames(dataset, d.name, frames, angles, meta["label"], crop)
        n += 1
    return n


# ----------------------------------------------------------------------------- eval / replay
def evaluate_sweeps(insp, sweeps: list[Path], cfg: AggregateConfig = AggregateConfig(),
                    strategies=STRATEGIES, write_back: bool = False) -> dict:
    """Run the inspector once per sweep and every strategy on its frames. Returns tables + rows + latency."""
    rows = []
    for d in sweeps:
        meta, frames, angles = load_sweep(d)
        label = meta.get("label")
        if label not in LABELS:
            log.warning("skipping %s: no plain/circle/cross label", d)
            continue
        recs, heats, results, ms = inspect_all(insp, frames, angles, cfg, strategies)
        rows.append({"sweep": str(d), "name": d.name, "label": label, "latency_ms": ms, "n": len(frames),
                     "frame_ms": [r.latency_ms for r in recs], "recs": recs,
                     "outcomes": {s: a.outcome() for s, a in results.items()},
                     "results": results, "synthetic": meta.get("synthetic")})
        if write_back:
            save_sweep(d, frames, angles, label, None, inspector_info(insp), recs, heats, results,
                       {k: v for k, v in meta.items() if k not in ("frames", "inspection", "label", "version",
                                                                    "created", "camera")})
    tables = {}
    for s in strategies:
        t = {lab: {o: 0 for o in OUTCOMES} for lab in LABELS}
        for r in rows:
            t[r["label"]][r["outcomes"][s]] += 1
        n = len(rows)
        correct = sum(r["outcomes"][s] is TRUTH[r["label"]] for r in rows)
        unknown = sum(r["outcomes"][s] is MarkType.UNKNOWN for r in rows)
        safe = sum((MarkType.CROSS if r["outcomes"][s] is MarkType.UNKNOWN else r["outcomes"][s]) is TRUTH[r["label"]]
                   for r in rows)
        missed = sum(r["label"] != "plain" and r["outcomes"][s] is MarkType.NONE for r in rows)
        tables[s] = {"table": t, "n": n, "correct": correct, "unknown": unknown, "wrong": n - correct - unknown,
                     "safe_correct": safe, "missed_marks": missed}
    lat = np.array([r["latency_ms"] for r in rows]) if rows else np.zeros(0)
    fr = np.concatenate([r["frame_ms"] for r in rows]) if rows else np.zeros(0)
    latency = {}
    if lat.size:
        latency = {"sweeps": int(lat.size), "frames_per_sweep": int(np.median([r["n"] for r in rows])),
                   "sweep_median_ms": float(np.median(lat)), "sweep_p90_ms": float(np.percentile(lat, 90)),
                   "sweep_p99_ms": float(np.percentile(lat, 99)), "frame_median_ms": float(np.median(fr)),
                   "frame_p90_ms": float(np.percentile(fr, 90))}
    return {"tables": tables, "rows": rows, "latency": latency, "strategies": list(strategies)}


def print_eval(ev: dict, info: dict) -> None:
    print(f"model={info.get('model', '?')} device={info.get('device')} stage2={info.get('stage2')}@"
          f"{info.get('mark_device') or '-'} threshold={info.get('threshold', float('nan')):.6g} "
          f"band=[{info.get('band_low', float('nan')):.6g}, {info.get('band_high', float('nan')):.6g}]")
    for s in ev["strategies"]:
        t = ev["tables"][s]
        print(f"\n[{s}]  rows = truth; unknown = re-inspect, then the purple (defective) bowl")
        print(f"{'truth':8s} {'plain':>6s} {'circle':>7s} {'cross':>6s} {'unknown':>8s}")
        for lab in LABELS:
            row = t["table"][lab]
            if sum(row.values()):
                print(f"{lab:8s} {row[MarkType.NONE]:6d} {row[MarkType.CIRCLE]:7d} {row[MarkType.CROSS]:6d} "
                      f"{row[MarkType.UNKNOWN]:8d}")
        n = max(t["n"], 1)
        print(f"correct {t['correct']}/{t['n']} ({100 * t['correct'] / n:.1f}%), wrong {t['wrong']}, unknown "
              f"{t['unknown']}, marked->plain {t['missed_marks']}; unknown->purple accuracy "
              f"{100 * t['safe_correct'] / n:.1f}%")
    L = ev["latency"]
    if L:
        print(f"\ninference per sweep ({L['frames_per_sweep']} frames, all stages, offline batch): median "
              f"{L['sweep_median_ms']:.1f} ms p90 {L['sweep_p90_ms']:.1f} p99 {L['sweep_p99_ms']:.1f}; per frame "
              f"median {L['frame_median_ms']:.2f} ms p90 {L['frame_p90_ms']:.2f}")


def _svg_strip(recs, threshold: float, w: int = 260, h: int = 54) -> str:
    smax = max([r.score for r in recs] + [threshold * 1.5, 1e-6])
    smin = min(0.0, min(r.score for r in recs))
    n = len(recs)
    bw = w / n
    col = {Label.PERFECT: "#2f9e6e", Label.DEFECTIVE: "#d6458f", Label.UNCERTAIN: "#d99a1c"}

    def y(v):
        return h - (h - 4) * (v - smin) / max(smax - smin, 1e-9)

    parts = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="score per angle">']
    for i, r in enumerate(recs):
        t = f"{r.angle_deg if r.angle_deg is not None else i}: {r.score:.4g} {r.label.value}" + (
            f" {r.mark.value} p(cross)={r.p_cross}" if r.mark is not None else "")
        parts.append(f'<rect x="{i * bw + 0.5:.1f}" y="{y(r.score):.1f}" width="{bw - 1:.1f}" '
                     f'height="{h - y(r.score):.1f}" fill="{col[r.label]}"><title>{html.escape(t)}</title></rect>')
    parts.append(f'<line x1="0" x2="{w}" y1="{y(threshold):.1f}" y2="{y(threshold):.1f}" stroke="#555" '
                 'stroke-dasharray="3 3"/></svg>')
    return "".join(parts)


def write_report(ev: dict, info: dict, path: Path) -> Path:
    """Self-contained HTML: per-strategy confusion matrices + latency + a row per sweep with its score strip."""
    css = ("body{font:14px/1.45 system-ui,sans-serif;margin:16px;color:#1d2330;background:#fafbfc}"
           "h1{font-size:20px}h2{font-size:16px;margin-top:24px}table{border-collapse:collapse;margin:6px 0}"
           "td,th{border:1px solid #d8dce4;padding:3px 8px;text-align:right;font-variant-numeric:tabular-nums}"
           "td:first-child,th:first-child{text-align:left}.ok{background:#e3f6ec}.bad{background:#fde4ea}"
           ".unk{background:#fff4d6}.grid{display:flex;flex-wrap:wrap;gap:16px}.card{background:#fff;border:1px "
           "solid #e3e6ec;border-radius:8px;padding:8px 12px}code{font-size:12px}"
           "@media (prefers-color-scheme: dark){body{background:#14171c;color:#e6e8ee}.card{background:#1c2027;"
           "border-color:#2a2f38}td,th{border-color:#333a45}.ok{background:#173a2a}.bad{background:#41202b}"
           ".unk{background:#3d3218}}")
    out = [f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,"
           f"initial-scale=1'><title>Sweep eval</title><style>{css}</style></head><body>",
           "<h1>Spin-and-inspect evaluation</h1>",
           f"<p><code>{html.escape(json.dumps(info, default=str))}</code></p>"]
    L = ev["latency"]
    if L:
        out.append(f"<p>Inference per sweep ({L['frames_per_sweep']} frames, offline): median "
                   f"<b>{L['sweep_median_ms']:.1f} ms</b>, p90 {L['sweep_p90_ms']:.1f} ms; per frame median "
                   f"{L['frame_median_ms']:.2f} ms.</p>")
    out.append("<div class='grid'>")
    for s in ev["strategies"]:
        t = ev["tables"][s]
        n = max(t["n"], 1)
        out.append(f"<div class='card'><h2>{s}</h2><table><tr><th>truth</th><th>plain</th><th>circle</th>"
                   "<th>cross</th><th>unknown</th></tr>")
        for lab in LABELS:
            row = t["table"][lab]
            if not sum(row.values()):
                continue
            cells = []
            for o in OUTCOMES:
                cls = "unk" if o is MarkType.UNKNOWN else ("ok" if o is TRUTH[lab] else "bad")
                cells.append(f"<td class='{cls if row[o] else ''}'>{row[o]}</td>")
            out.append(f"<tr><td>{lab}</td>{''.join(cells)}</tr>")
        out.append(f"</table>correct {t['correct']}/{t['n']} ({100 * t['correct'] / n:.0f}%), unknown "
                   f"{t['unknown']}, marked&rarr;plain {t['missed_marks']}<br>unknown&rarr;purple accuracy "
                   f"{100 * t['safe_correct'] / n:.0f}%</div>")
    out.append("</div><h2>Sweeps</h2><table><tr><th>sweep</th><th>truth</th><th>scores per angle</th>"
               + "".join(f"<th>{s}</th>" for s in ev["strategies"]) + "<th>ms</th></tr>")
    for r in ev["rows"]:
        cells = []
        for s in ev["strategies"]:
            o = r["outcomes"][s]
            cls = "unk" if o is MarkType.UNKNOWN else ("ok" if o is TRUTH[r["label"]] else "bad")
            cells.append(f"<td class='{cls}'>{o.value}</td>")
        out.append(f"<tr><td>{html.escape(r['name'])}</td><td>{r['label']}</td>"
                   f"<td>{_svg_strip(r['recs'], info.get('threshold', 0.5))}</td>{''.join(cells)}"
                   f"<td>{r['latency_ms']:.0f}</td></tr>")
    out.append("</table></body></html>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out))
    return path


def eval_json(ev: dict, info: dict) -> dict:
    return {"info": info, "latency": ev["latency"],
            "tables": {s: {**{k: v for k, v in t.items() if k != "table"},
                           "table": {lab: {o.value: c for o, c in row.items()} for lab, row in t["table"].items()}}
                       for s, t in ev["tables"].items()},
            "sweeps": [{"name": r["name"], "label": r["label"], "latency_ms": round(r["latency_ms"], 2),
                        "outcomes": {s: o.value for s, o in r["outcomes"].items()}} for r in ev["rows"]]}


# ----------------------------------------------------------------------------- building the inspector
def load_inspector(args, cfg: AggregateConfig):
    if args.model in (None, "fake"):
        return None
    from dlsj.inspect.core import parse_box
    from dlsj.inspect.detector import AnomalibInspector

    return AnomalibInspector.from_model(args.model, device=args.device, batch=args.batch, stage2=args.stage2,
                                        stage2_device=args.stage2_device, crop=parse_box(args.crop),
                                        input_size=_hw(args.input_size), score_space=args.score_space, agg=cfg,
                                        cache_dir=args.cache_dir)


def _hw(text):
    if not text:
        return None
    v = [int(x) for x in str(text).split(",")]
    return (v[0], v[0]) if len(v) == 1 else (v[0], v[1])


# ----------------------------------------------------------------------------- hardware / collect
class Rig:
    """Robot + cameras + a scripted-move executor, real or fake."""

    def __init__(self, args):
        from dlsj.loop.executor import ExecConfig, SkillExecutor
        from dlsj.loop.skills import SkillLibrary

        self.args, self.cleanup, self.world, self.home = args, [], None, None
        if args.dry_run:
            from dlsj.loop import fakes

            self.world = fakes.FakeWorld(bricks=[args.fake_brick] * max(args.repeat, 1))
            self.world.held, self.world.on_table = self.world.on_table, None
            start = self.world.poses["inspect"].copy()
            start[5] = fakes.HOLD_GRIP
            self.robot = fakes.FakeRobot(self.world, start=start)
            self.cameras = fakes.FakeCameras(self.world, self.robot)
            self.pose = self.world.poses["inspect"].copy()
            self.pose[5] = 0.0
            self.limits = None
        else:
            from dlsj import arms
            from dlsj.loop import poses as P
            from dlsj.loop.hardware import LeRobotArm, StreamCameras

            self.robot = LeRobotArm(args.follower_port, args.follower_id)
            self.robot.connect()
            self.cleanup.append(self.robot.disconnect)
            self.cameras = StreamCameras.open(args.cameras)
            self.cleanup.insert(0, self.cameras.stop)
            self.cameras.wait_ready()
            if args.camera not in self.cameras.names:
                raise SystemExit(f"--cameras must include '{args.camera}' (got {self.cameras.names})")
            self.pose = P.load_pose(args.follower_id, "inspect") if args.pose == "inspect" else None
            if args.pose == "inspect" and self.pose is None:
                raise SystemExit("no saved inspect pose: `python -m dlsj.loop.run --save-pose inspect`, or --pose "
                                 "current")
            self.home = P.load_pose(args.follower_id, "home")
            self.limits = arms.joint_limits_deg(args.follower_id, "wrist_roll")
            log.info("wrist_roll limits from calibration: %s", self.limits)
        self.ex = SkillExecutor(self.robot, self.cameras, None, SkillLibrary(),
                                ExecConfig(fps=args.fps, ramp_speed_deg_s=args.ramp_speed, async_inference=False))

    def close(self) -> None:
        for fn in self.cleanup:
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                log.warning("cleanup: %s", e)


def _ask(args, prompt: str) -> bool:
    if args.yes:
        return True
    try:
        return input(prompt).strip().lower() not in ("q", "quit", "n", "no")
    except EOFError:
        return False


def sweep_config(args, limits) -> SweepConfig:
    return SweepConfig(frames=args.sweep_frames, range_deg=args.sweep_range, settle_ms=args.sweep_settle_ms,
                       max_speed_deg_s=args.sweep_speed, discard_frames=args.sweep_discard, limits=limits,
                       grasp_current_min=args.grasp_current_min, grasp_empty_pos=args.grasp_empty_pos,
                       continuous=args.continuous, camera=args.camera, fps=args.fps,
                       gripper_hold=args.grip_close if args.close_gripper else None)


def run_hardware(args, collect: bool = False) -> int:
    cfg = AggregateConfig(strategy=args.aggregate, k=args.topk)
    from dlsj.inspect.core import parse_box

    if collect and args.label not in LABELS:
        raise SystemExit("collect needs --label plain|circle|cross")
    insp = load_inspector(args, cfg)
    rig = Rig(args)
    if insp is None and args.dry_run and not collect:
        from dlsj.loop import fakes

        insp = fakes.FakeInspector(rig.cameras.roi, agg=cfg)
    abort = threading.Event()
    out_root = Path(args.out).expanduser()
    saved = []
    try:
        ex = rig.ex
        if rig.pose is not None:
            pose = rig.pose.copy()
            ex._state()
            pose[5] = ex.cmd[5]  # keep whatever the gripper is doing; --close-gripper handles it below
            log.info("ramping to the inspect pose")
            ex.move_to(pose, abort)
        else:
            ex._state()  # hold the current pose
        for k in range(max(args.repeat, 1)):
            if args.close_gripper:
                ex.set_gripper(args.grip_open, abort)
                if not _ask(args, f"\nsweep {k + 1}/{args.repeat}: put a brick between the jaws, Enter (q = quit) "):
                    break
                ex.set_gripper(args.grip_close, abort, timeout_s=2.0)
                ex.hold_until_still(5, 3.0, 1.0, abort)
            elif k and not _ask(args, f"\nsweep {k + 1}/{args.repeat}: Enter for the next sweep (q = quit) "):
                break
            session = insp.start_sweep(args.aggregate) if insp is not None and not args.sweep_sync else None
            sres = Sweeper(rig.robot, rig.cameras, sweep_config(args, rig.limits), abort).run(
                ex.cmd, on_frame=session.add if session is not None else None)
            ex.cmd = sres.final_cmd.copy()
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            sid = f"{stamp}_{k:02d}_{args.label or 'unlabelled'}"
            print(f"sweep {sid}: {sres.status}, {len(sres.frames)} frames, {sres.duration_ms:.0f} ms "
                  f"(+{sres.return_ms:.0f} ms back to {sres.final_roll:.1f} deg)")
            if sres.status != "ok":
                if session is not None:
                    session.cancel()
                if sres.status == "grasp_lost":
                    print("  grasp lost during the sweep: the brick dropped or slipped")
                if not sres.frames:
                    continue
            recs = heats = results = None
            info = None
            if insp is not None and sres.frames:
                t0 = time.perf_counter()
                if session is not None and sres.status == "ok":
                    r: InspectionResult = session.result()
                    recs, heats = session.records, session.heats
                    m = insp.sweep_meta()
                    results = agg_mod.aggregate_all(recs, m["threshold"], m["band_low"], m["band_high"], cfg)
                    tail = (time.perf_counter() - t0) * 1e3
                    print(f"  inference overlapped with the motion; {tail:.1f} ms left after the last frame "
                          f"(compute {r.stage_latency_ms.get('stage1+2', 0):.0f} ms total)")
                else:
                    recs, heats, results, ms = inspect_all(insp, sres.frames, sres.angles, cfg)
                    print(f"  inference after the sweep: {ms:.1f} ms")
                info = inspector_info(insp)
                for s, a in results.items():
                    print(f"  {s:8s} -> {a.outcome().value:8s} {a.summary.get('reason', '')} "
                          f"{a.summary.get('mark_reason', '')}")
            d = save_sweep(out_root / sid, sres.frames, sres.angles, args.label, sres, info, recs, heats, results,
                           {"source": "dry-run" if args.dry_run else "hardware",
                            "args": {k2: v for k2, v in vars(args).items() if k2 != "func"}})
            saved.append(d)
            print(f"  saved {d}")
            if collect:
                collect_frames(args.dataset, sid, sres.frames, sres.angles, args.label, parse_box(args.crop))
                print(f"  + {len(sres.frames)} frames -> {Path(args.dataset).expanduser() / FOLDER[args.label]}")
    except KeyboardInterrupt:
        abort.set()
        print("\ninterrupted")
    finally:
        if not args.dry_run:
            if args.close_gripper and _ask(args, "take the brick out: Enter opens the gripper (q = leave closed) "):
                rig.ex.set_gripper(args.grip_open)
            if rig.home is not None and not args.no_home:
                log.info("ramping home")
                rig.ex.move_to(rig.home)
        rig.close()
    print(f"\n{len(saved)} sweep(s) in {out_root}")
    return 0


# ----------------------------------------------------------------------------- CLI
def add_model_args(p) -> None:
    p.add_argument("--model", default=None, help="manifest.json, Anomalib export folder or model.xml (stage 1)")
    p.add_argument("--device", default="NPU", help="stage-1 OpenVINO device (NPU / GPU / CPU)")
    p.add_argument("--batch", type=int, default=1, help="static batch (1 is fastest on the NPU)")
    p.add_argument("--stage2", default=None, help='"shape" | "none" | CNN classifier.json (default: manifest / shape)')
    p.add_argument("--stage2-device", default=None)
    p.add_argument("--crop", default=None, help="x0,y0,x1,y1 inspect ROI (needed for an export without manifest)")
    p.add_argument("--input-size", default=None, help="H,W fed to the model (exports with dynamic H/W)")
    p.add_argument("--score-space", choices=["raw", "normalized"], default=None,
                   help="exports: raw = cut before the normaliser (no saturation at 1.0)")
    p.add_argument("--cache-dir", default=None, help="OpenVINO compile cache")
    p.add_argument("--aggregate", default="topk", choices=list(STRATEGIES))
    p.add_argument("--topk", type=int, default=3)


def add_rig_args(p) -> None:
    p.add_argument("--dry-run", action="store_true", help="fake arm + cameras (no hardware)")
    p.add_argument("--fake-brick", default="side_cross", help="dry run: brick kind (see dlsj.loop.fakes)")
    p.add_argument("--follower-port", default="auto")
    p.add_argument("--follower-id", default="follower")
    p.add_argument("--cameras", default="inspect=4", help="name=index,... (must include the inspect camera)")
    p.add_argument("--camera", default="inspect")
    p.add_argument("--pose", choices=["inspect", "current"], default="inspect")
    p.add_argument("--close-gripper", action="store_true", help="hand-feed a brick, then close the jaws on it")
    p.add_argument("--grip-open", type=float, default=60.0)
    p.add_argument("--grip-close", type=float, default=5.0)
    p.add_argument("--grasp-empty-pos", type=float, default=4.0)
    p.add_argument("--grasp-current-min", type=float, default=None)
    p.add_argument("--label", choices=list(LABELS), default=None, help="ground truth of the brick")
    p.add_argument("--repeat", type=int, default=1, help="sweeps (one brick each, Enter between)")
    p.add_argument("--yes", action="store_true", help="don't wait for Enter")
    p.add_argument("--out", default="outputs/sweeps")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--ramp-speed", type=float, default=30.0)
    p.add_argument("--no-home", action="store_true", help="don't ramp home at the end")
    p.add_argument("--sweep-frames", type=int, default=12)
    p.add_argument("--sweep-range", type=float, default=180.0)
    p.add_argument("--sweep-settle-ms", type=float, default=150.0)
    p.add_argument("--sweep-speed", type=float, default=90.0)
    p.add_argument("--sweep-discard", type=int, default=1)
    p.add_argument("--continuous", action="store_true")
    p.add_argument("--sweep-sync", action="store_true", help="infer after the sweep instead of during it")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    hw = sub.add_parser("hardware", help="sweep a held brick, save + inspect")
    add_rig_args(hw)
    add_model_args(hw)
    co = sub.add_parser("collect", help="sweep and save the frames as Anomalib training data")
    add_rig_args(co)
    add_model_args(co)
    co.add_argument("--dataset", required=True, help="dataset root: good/ minor/ bad/ are created inside")
    co.add_argument("--from-sweeps", default=None, help="convert existing labelled sweep folders instead")
    for name in ("eval", "replay"):
        ev = sub.add_parser(name, help="re-run inspector + all strategies over saved sweeps")
        ev.add_argument("--sweeps", required=True)
        add_model_args(ev)
        ev.add_argument("--strategies", default=",".join(STRATEGIES))
        ev.add_argument("--report", default=None, help="HTML report path (default <sweeps>/eval_report.html)")
        ev.add_argument("--json", default=None, help="write the tables as JSON here")
        ev.add_argument("--write-back", action="store_true", help="store the results + charts in each sweep folder")
    sy = sub.add_parser("synth", help="synthetic labelled sweeps (dlsj.inspect.synthetic.generate_sweeps)")
    sy.add_argument("--out", required=True)
    sy.add_argument("--plain", type=int, default=20)
    sy.add_argument("--circle", type=int, default=20)
    sy.add_argument("--cross", type=int, default=20)
    sy.add_argument("--frames", type=int, default=12)
    sy.add_argument("--range", type=float, default=180.0)
    sy.add_argument("--seed", type=int, default=7)
    sy.add_argument("--glare-prob", type=float, default=0.4)
    sy.add_argument("--layout", choices=["sweeps", "dataset"], default="sweeps")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.cmd == "synth":
        from dlsj.inspect.synthetic import generate_sweeps

        generate_sweeps(Path(args.out).expanduser(), args.plain, args.circle, args.cross, args.frames, args.range,
                        args.seed, args.layout, args.glare_prob)
        return 0
    if args.cmd == "collect" and args.from_sweeps:
        from dlsj.inspect.core import parse_box

        n = collect_from_sweeps(Path(args.from_sweeps), Path(args.dataset), parse_box(args.crop))
        print(f"{n} sweeps -> {Path(args.dataset).expanduser()}")
        return 0
    if args.cmd in ("hardware", "collect"):
        return run_hardware(args, collect=args.cmd == "collect")
    # eval / replay
    strategies = tuple(s.strip() for s in args.strategies.split(",") if s.strip())
    cfg = AggregateConfig(strategy=args.aggregate, k=args.topk)
    insp = load_inspector(args, cfg)
    if insp is None:
        raise SystemExit("eval needs --model")
    sweeps = find_sweeps(Path(args.sweeps))
    if not sweeps:
        raise SystemExit(f"no sweep.json under {args.sweeps}")
    _, frames, angles = load_sweep(sweeps[0])  # warm-up: the first inference on a device is slow
    insp.inspect_frames(frames[:2], angles[:2])
    ev = evaluate_sweeps(insp, sweeps, cfg, strategies, args.write_back)
    info = inspector_info(insp)
    print_eval(ev, info)
    report = Path(args.report) if args.report else Path(args.sweeps).expanduser() / "eval_report.html"
    write_report(ev, info, report)
    print(f"\nreport: {report}")
    if args.json:
        Path(args.json).write_text(json.dumps(eval_json(ev, info), indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
