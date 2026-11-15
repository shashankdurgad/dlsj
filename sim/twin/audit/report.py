"""Audit report: report.md + evidence images for one run, and the trend table over all runs.

    python sim/twin/audit/report.py --run sim/twin/out/audit/runs/<stamp>          # plain python, CPU

Reads the run's state.json / render_meta.json / policy_<model>.json / perceptual.json(.npz) / sim.npz and the fixed
set; writes <run>/report.md and <run>/img/*, and copies both to sim/twin/out/audit/ (report.md, img/ = latest run).
The hand-written ranked list sim/twin/out/audit/top10.md (if present) is included verbatim; the automatic candidate
ranking below it is recomputed every run.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from sim.twin.audit.common import BITS, CAMS, OUT, PHASES, RUNS, SET, load_set

AREAS = {"sim/twin/visuals/": "visual fit / lighting", "sim/twin/env/randomize": "lighting / DR",
         "sim/twin/env/": "env", "sim/twin/scene/": "scene", "sim/twin/gripper/": "gripper",
         "sim/twin/robot/": "robot model", "sim/twin/align/joint_fit/": "joint fit", "sim/twin/calib/": "calibration",
         "sim/twin/align/": "camera fit", "sim/twin/physics/": "physics", "sim/twin/policies/": "policies",
         "sim/twin/rl/": "rl", "sim/twin/sensor/": "camera sensor", "sim/residual_env/": "residual env"}
RENDER_AREAS = ("visual fit / lighting", "lighting / DR", "env", "scene", "gripper", "robot model", "joint fit",
                "calibration", "camera fit", "physics", "camera sensor", "residual env")


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def f(v, n=2) -> str:
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{n}f}"


def area_of(path: str) -> str:
    for k, v in AREAS.items():
        if path.startswith(k):
            return v
    return "other"


def hashes(run: Path) -> dict:
    p = run / "files.sha256"
    if not p.exists():
        return {}
    return {ln.split(None, 1)[1].strip(): ln.split()[0] for ln in p.read_text().splitlines() if ln.strip()}


def changed(prev: dict, cur: dict) -> dict:
    """Files that differ between two snapshots, grouped by agent area."""
    out: dict[str, list] = {}
    for k in sorted(set(prev) | set(cur)):
        if prev.get(k) != cur.get(k):
            out.setdefault(area_of(k), []).append(k)
    return out


# ------------------------------------------------------------------------------------------------ images
def put(img, text, org=(6, 18), scale=0.5):
    img = np.ascontiguousarray(img)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def save(path: Path, rgb, q=88):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.ascontiguousarray(rgb)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])


def heat(m, shape, vmax):
    h = cv2.resize(np.clip(m / max(vmax, 1e-6), 0, 1).astype(np.float32), shape[::-1], interpolation=cv2.INTER_CUBIC)
    return cv2.applyColorMap((np.clip(h, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1]


def overlay(img, m, vmax, alpha=0.55):
    return (img * (1 - alpha) + heat(m, img.shape[:2], vmax) * alpha).astype(np.uint8)


def make_images(run: Path, S, real, sim, pm) -> list[str]:
    img = run / "img"
    made = []
    N = len(sim["top"])
    for cam in CAMS:
        R, Sm = real[cam][:N], sim[cam]
        rows = []
        for p in PHASES:  # one pair per phase: real | sim | 3 x |diff|
            i = next((k for k, s in enumerate(S[:N]) if s["phase"] == p), None)
            if i is None:
                continue
            d = np.clip(np.abs(R[i].astype(np.int16) - Sm[i].astype(np.int16)) * 3, 0, 255).astype(np.uint8)
            rows.append(np.concatenate([put(R[i], f"real {cam} | {p} | ep {S[i]['ep']} f {S[i]['frame']}"),
                                        put(Sm[i], "sim"), put(d, "3 x |real - sim|")], 1))
        big = np.concatenate(rows, 0)
        save(img / f"pairs_{cam}.jpg", cv2.resize(big, (big.shape[1] // 2, big.shape[0] // 2),
                                                 interpolation=cv2.INTER_AREA))
        made.append(f"pairs_{cam}.jpg")
        cells = []
        for p in PHASES:  # all pairs: real above sim, one row of pairs per phase
            ids = [k for k, s in enumerate(S[:N]) if s["phase"] == p]
            row = [np.concatenate([cv2.resize(R[i], (200, 150), interpolation=cv2.INTER_AREA),
                                   cv2.resize(Sm[i], (200, 150), interpolation=cv2.INTER_AREA)], 0) for i in ids]
            row[0] = put(row[0], p)
            cells.append(np.concatenate(row, 1))
        w = max(c.shape[1] for c in cells)
        cells = [np.pad(c, ((0, 0), (0, w - c.shape[1]), (0, 0))) for c in cells]
        save(img / f"all_{cam}.jpg", np.concatenate(cells, 0), 82)
        made.append(f"all_{cam}.jpg")
        if pm is not None and f"patchdist_{cam}" in pm:
            pd = pm[f"patchdist_{cam}"]
            mean_r = R.mean(0).astype(np.uint8)
            mean_s = Sm.mean(0).astype(np.uint8)
            a = put(overlay(mean_r, pd.mean(0), 0.8), f"{cam}: mean DINOv2 patch distance real-vs-sim (max = 0.8)")
            b = put(overlay(mean_r, pm[f"patchdist_nbr_{cam}"], 0.8), "floor: real vs next real frame")
            c = put(overlay(mean_s, pm[f"probe_sim_{cam}"].mean(0), 1.0), "linear probe: mean held-out P(sim) on sim")
            save(img / f"where_{cam}.jpg", np.concatenate([a, b, c], 1))
            made.append(f"where_{cam}.jpg")
            conf = pm[f"probe_sim_{cam}"].reshape(N, -1).mean(1)
            top = np.argsort(-conf)[:4]
            low = np.argsort(conf)[:2]
            tiles = []
            for i in list(top) + list(low):
                tiles.append(np.concatenate([
                    put(R[i], f"real | {S[i]['phase']} ep {S[i]['ep']}"),
                    put(Sm[i], f"sim | mean P(sim) {conf[i]:.2f}"),
                    put(overlay(Sm[i], pm[f"probe_sim_{cam}"][i], 1.0), "DINO probe P(sim)"),
                    put(overlay(Sm[i], pm[f"cnn_highpass_sim_{cam}"][i], 1.0), "high-pass CNN P(sim)")], 1))
            big = np.concatenate(tiles, 0)
            save(img / f"giveaway_{cam}.jpg", cv2.resize(big, (big.shape[1] // 2, big.shape[0] // 2),
                                                        interpolation=cv2.INTER_AREA))
            made.append(f"giveaway_{cam}.jpg")
        # mean colour per matched region: real swatch | sim swatch
        mk = sim["mask_" + cam]
        sw = []
        k = np.ones((7, 7), np.uint8)
        for name, bit in list(BITS.items()) + [("background", 0)]:
            cr, cs = [], []
            for i in range(N):
                m = cv2.erode(((mk[i] & bit) > 0 if bit else mk[i] == 0).astype(np.uint8), k) > 0
                if m.sum() >= 300:
                    cr.append(R[i][m].mean(0))
                    cs.append(Sm[i][m].mean(0))
            if not cr:
                continue
            t = np.zeros((60, 240, 3), np.uint8)
            t[:, :120], t[:, 120:] = np.mean(cr, 0), np.mean(cs, 0)
            sw.append(put(t, f"{name}: real | sim", (4, 14), 0.42))
        if sw:
            save(img / f"swatch_{cam}.jpg", np.concatenate(sw, 0), 95)
            made.append(f"swatch_{cam}.jpg")
    return made


# ------------------------------------------------------------------------------------------------ tables
def policy_tables(pol: dict) -> list[str]:
    o, ct = pol["overall"], pol["close_timing"]
    names = [("sim", "**sim (wrist + top)**"), ("sim_top", "sim top only (wrist real)"),
             ("sim_wrist", "sim wrist only (top real)"), ("nbr", "floor: next real frame"),
             ("ref", "ceiling: other episode, same phase"), ("seed_noise", "flow-noise: two seeds, same real frame")]
    L = [f"Model `{pol['model']}` ({pol['model_key']}), {pol['n']} samples x {pol['k']} seeds; "
         f"reference re-run determinism max |delta| {max(pol['determinism_max_abs']):.3f}.", "",
         "| images fed (vs the real frames, same seed) | joints, first 10 (deg) | joints, all 30 (deg) | "
         "gripper, first 10 | gripper, all 30 | feature cosine | centred cosine |", "|---|---|---|---|---|---|---|"]
    for k, lab in names:
        m = o[k]
        L.append(f"| {lab} | {f(m['joint10']['mean'])} | {f(m['joint30']['mean'])} | {f(m['grip10']['mean'])} | "
                 f"{f(m['grip30']['mean'])} | {f(m.get('cos', {}).get('mean'), 4)} | "
                 f"{f(m.get('cos_centred', {}).get('mean'), 3)} |")
    e = o["equivalence_index"]
    L += ["", "Equivalence index (0 = like the next real frame, 1 = like a wrong scene): "
          + ", ".join(f"{k} **{f(v)}**" for k, v in e.items()) + ".",
          f"Error against the human demo (mean chunk, first 10 actions, deg): real frames "
          f"{f(o['demo_error']['real']['joint10'])}, sim frames {f(o['demo_error']['sim']['joint10'])}, "
          f"holding still {f(o['demo_error']['hold']['joint10'])}.",
          "Signed bias of the sim images (sim - real, first 10 actions; pan, lift, elbow, wrist flex, wrist roll deg, "
          "gripper): overall " + " ".join(f"{v:+.2f}" for v in pol["bias_sim_minus_real_first10"]["overall"])
          + "; pregrasp " + " ".join(f"{v:+.2f}" for v in pol["bias_sim_minus_real_first10"].get("pregrasp", []))
          + f". Gripper-close timing on reach + pregrasp samples (chunk step of the first command < 12; 30 = none): "
          f"real {f(ct['real'], 1)}, sim {f(ct['sim'], 1)}; |sim - real| "
          f"{f(ct['abs_diff_sim'], 1)} steps vs {f(ct['abs_diff_next_real_frame'], 1)} "
          f"for the next real frame; sim closes earlier in {f(100 * ct['sim_closes_earlier_frac'], 0)} % "
          f"and later in {f(100 * ct['sim_closes_later_frac'], 0)} % of them.", "",
          "| phase | sim joints10 | sim top only | sim wrist only | floor | ceiling | index joints10 | sim gripper10 | "
          "floor | ceiling | index feature |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for p, t in pol["by_phase"].items():
        L.append(f"| {p} | {f(t['sim']['joint10']['mean'])} | {f(t['sim_top']['joint10']['mean'])} | "
                 f"{f(t['sim_wrist']['joint10']['mean'])} | {f(t['nbr']['joint10']['mean'])} | "
                 f"{f(t['ref']['joint10']['mean'])} | {f(t['equivalence_index']['joint10'])} | "
                 f"{f(t['sim']['grip10']['mean'])} | {f(t['nbr']['grip10']['mean'])} | "
                 f"{f(t['ref']['grip10']['mean'])} | "
                 f"{f(t['equivalence_index'].get('feature'))} |")
    return L


def control_tables(ctl: dict) -> list[str]:
    """Positive control: the twin's closed-loop outcome vs the real robot's (control.py)."""
    t, real = ctl["twin_ft8k"], ctl["real"]
    L = ["Does the policy DO in the twin what it does on the robot? (A sim that passes replays can still score every "
         "policy 0.) Twin: the 8k fine-tune closed-loop in TwinEnv, one pick attempt from 0.7-1.2 s before the "
         "recorded close (RL harness `sim/twin/rl/molmo_eval.py`, logs " + ", ".join(f"`{x}`" for x in t["logs"])
         + "). Real: the XPS controller traces, first close of each pick episode (empty = jaws shut on nothing).", "",
         "| | episodes | first attempt on the brick | 95 % CI | all closes empty | outcomes |",
         "|---|---|---|---|---|---|",
         f"| **twin, 8k** | {t['episodes']} | {f(t['success_rate'])} (lifted) | {t['success_ci95'][0]}-"
         f"{t['success_ci95'][1]} | n/a (one attempt) | {t['outcomes']} |"]
    for k in ("ft8k", "ft5.5k"):
        r = real.get(k)
        if r:
            L.append(f"| real, {k} | {r['episodes']} | {f(r['first_close_brick_rate'])} (jaws on a brick) | "
                     f"{r['first_close_ci95'][0]}-{r['first_close_ci95'][1]} | {f(r['empty_close_frac'])} of "
                     f"{r['closes']} | first close {r['first_close']}, no close {r['no_close']} |")
    L += ["", "Real run outcomes: " + "; ".join(f"{k}: {v}" for k, v in ctl["real_reference"].items()) + ".",
          "Caveat: the twin episode starts with the approach already done by the demo (near the close pose), the real "
          "first close follows the policy's own approach from the start pose; so the twin rate is an upper bound for "
          "the comparison and a twin far ABOVE the real rate means the twin forgives the real failure (empty closes)."]
    return L


def perceptual_tables(per: dict) -> list[str]:
    L = ["| camera | KID real-vs-sim | KID real-vs-real | CLS dist sim | floor | ceiling | patch dist sim | floor | "
         "ceiling |", "|---|---|---|---|---|---|---|---|---|"]
    for cam, c in per["cams"].items():
        if "kid" not in c:
            L.append(f"| {cam} | DINO failed: {c.get('dino_error')} | | | | | | | |")
            continue
        k, pc, pp = c["kid"], c["paired_cls"], c["paired_patch"]
        L.append(f"| {cam} | {f(k['real_vs_sim']['kid'], 3)} +- {f(k['real_vs_sim']['sd'], 3)} | "
                 f"{f(k['real_vs_real']['kid'], 3)} +- {f(k['real_vs_real']['sd'], 3)} | {f(pc['sim']['mean'], 3)} | "
                 f"{f(pc['nbr']['mean'], 3)} | {f(pc['ref']['mean'], 3)} | {f(pp['sim']['mean'], 3)} | "
                 f"{f(pp['nbr']['mean'], 3)} | {f(pp['ref']['mean'], 3)} |")
    L += ["", "Real-vs-sim classifiers, held-out episodes (50 % = indistinguishable; control = the same classifier on "
          "real audit frames vs real reference frames):", "",
          "| camera | DINO patch probe: patch acc | image acc | control image acc | CNN 32 px crops RGB: crop acc | "
          "control | CNN high-pass only: crop acc | control |", "|---|---|---|---|---|---|---|---|"]
    for cam, c in per["cams"].items():
        pd = c.get("probe_dino", {})
        a, b = pd.get("real_vs_sim", {}), pd.get("control_real_vs_real", {})
        L.append(f"| {cam} | {f(a.get('patch_acc'), 3)} | {f(a.get('image_acc'), 3)} | {f(b.get('image_acc'), 3)} | "
                 f"{f(c['cnn_rgb']['real_vs_sim']['crop_acc'], 3)} | "
                 f"{f(c['cnn_rgb']['control_real_vs_real']['crop_acc'], 3)} | "
                 f"{f(c['cnn_highpass']['real_vs_sim']['crop_acc'], 3)} | "
                 f"{f(c['cnn_highpass']['control_real_vs_real']['crop_acc'], 3)} |")
    L += ["", "Matched regions (sim geometry masks, medians over the samples; d = sim - real in CIELAB; ratios = "
          "sim / real):", "",
          "| camera | region | n | dE | dL | da | db | chroma ratio | contrast ratio | sharpness ratio | "
          "noise sigma real | sim | highlight L* real | sim |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for cam, c in per["cams"].items():
        for reg, m in c["regions"].items():
            if reg == "_global":
                continue
            L.append(f"| {cam} | {reg} | {m['n']} | {f(m['dE'], 1)} | {f(m['dL'], 1)} | {f(m['da'], 1)} | "
                     f"{f(m['db'], 1)} | {f(m['chroma_ratio'])} | {f(m['contrast_ratio'])} | {f(m['sharp_ratio'])} | "
                     f"{f(m['noise_real'])} | {f(m['noise_sim'])} | {f(m.get('hl_real'), 1)} | "
                     f"{f(m.get('hl_sim'), 1)} |")
    geo = [f"{cam} IoU {c['plate_geo']['iou']:.2f}, sim - real centroid {c['plate_geo']['dx']:+.0f} / "
           f"{c['plate_geo']['dy']:+.0f} px (median |d| {c['plate_geo']['offset']:.0f} px)"
           for cam, c in per["cams"].items() if c.get("plate_geo")]
    if geo:
        L += ["", "Plate placement (twin plate mask vs the saturated real plate blobs near it): " + "; ".join(geo)
              + "."]
    L += ["", "Whole image (medians; real / sim):", "",
          "| camera | L* | highlights >= 250 (%) | saturation | sharpness | chroma sharpness | high-freq power (%) | "
          "MAE (floor) | image shift px (floor) |", "|---|---|---|---|---|---|---|---|---|"]
    for cam, c in per["cams"].items():
        g, fl = c["regions"]["_global"], c["floor_next_frame"]
        L.append(f"| {cam} | {f(g['lum_real'], 1)} / {f(g['lum_sim'], 1)} | {f(100 * g['clip_hi_real'], 2)} / "
                 f"{f(100 * g['clip_hi_sim'], 2)} | {f(g['sat_real'], 0)} / {f(g['sat_sim'], 0)} | "
                 f"{f(g['sharp_real'])} / {f(g['sharp_sim'])} | {f(g['chroma_sharp_real'])} / "
                 f"{f(g['chroma_sharp_sim'])} | {f(100 * g['hf_real'])} / {f(100 * g['hf_sim'])} | "
                 f"{f(g['mae'], 1)} ({f(fl['mae'], 1)}) | {f(g['shift_px'], 1)} ({f(fl['shift_px'], 1)}) |")
    return L


def candidates(per: dict | None, pol: dict | None, rm: dict | None) -> list[dict]:
    """Automatic discrepancy candidates: severity in 'units of clearly visible' x the camera's policy weight."""
    w = {"top": 1.0, "wrist": 1.0, "inspect": 0.3}
    if pol:
        o = pol["overall"]
        fl = o["nbr"]["joint10"]["mean"]
        ex = {k: max(o["sim_" + k]["joint10"]["mean"] - fl, 0.0) for k in ("top", "wrist")}
        tot = max(ex["top"] + ex["wrist"], 1e-6)
        w.update({k: 0.5 + v / tot for k, v in ex.items()})
    out = []
    reg_area = {"table": "visual fit (wood / table texture)", "plates": "visual fit (plate material) / scene",
                "bricks": "scene (Duplo) / visual fit", "robot": "robot model / visual fit (robot paint)",
                "props": "scene (props)", "background": "visual fit (backdrop) / lighting"}
    if per:
        for cam, c in per["cams"].items():
            for reg, m in c["regions"].items():
                if reg == "_global" or m["dE"] is None:
                    continue
                out.append(dict(cam=cam, what=f"{reg} colour", area=reg_area.get(reg, "?"), sev=m["dE"] / 6.0,
                                ev=f"dE {m['dE']:.1f} (dL {m['dL']:+.1f}, da {m['da']:+.1f}, db {m['db']:+.1f}), "
                                   f"chroma x{m['chroma_ratio']:.2f}"))
                if m.get("hl_real") is not None and m.get("hl_sim") is not None and reg != "background":
                    out.append(dict(cam=cam, what=f"{reg} highlights (specular / light hardness)", area="lighting",
                                    sev=abs(np.log2((m["hl_sim"] + 2) / (m["hl_real"] + 2))) * 1.5,
                                    ev=f"L* p99.5 - median: real {m['hl_real']:.1f} vs sim {m['hl_sim']:.1f}"))
                if m["sharp_ratio"]:
                    out.append(dict(cam=cam, what=f"{reg} sharpness / texture",
                                    sev=abs(np.log2(m["sharp_ratio"])) * 1.5,
                                    area="camera sensor / codec" if reg != "table" else "visual fit (wood) / sensor",
                                    ev=f"|Laplacian| sim / real {m['sharp_ratio']:.2f}, contrast "
                                       f"x{m['contrast_ratio']:.2f}"))
            g = c["regions"]["_global"]
            t = c["regions"].get("table")
            if t and t["noise_real"] and t["noise_sim"] is not None:
                r = (t["noise_sim"] + 0.05) / (t["noise_real"] + 0.05)
                out.append(dict(cam=cam, what="fine noise on the table", area="camera sensor / codec",
                                sev=abs(np.log2(r)),
                                ev=f"sigma real {t['noise_real']:.2f} vs sim {t['noise_sim']:.2f}"))
            r = (g["chroma_sharp_sim"] + 1e-3) / (g["chroma_sharp_real"] + 1e-3)
            out.append(dict(cam=cam, what="chroma detail (4:2:0 / codec softness)", area="camera sensor / codec",
                            sev=abs(np.log2(r)) * 1.5,
                            ev=f"chroma |Laplacian| real {g['chroma_sharp_real']:.2f} vs sim "
                               f"{g['chroma_sharp_sim']:.2f}"))
            pg = c.get("plate_geo")
            if pg and cam == "top":
                out.append(dict(cam=cam, what="plate placement (layout / camera geometry)",
                                area="scene (layout_from_episode) / top camera fit",
                                sev=pg["offset"] / 8.0 + (1 - pg["iou"]) * 2,
                                ev=f"real-vs-sim plate IoU {pg['iou']:.2f}, sim centroid {pg['dx']:+.0f} / "
                                   f"{pg['dy']:+.0f} px from real (median |d| {pg['offset']:.0f} px, n {pg['n']})"))
            reliable = g["shift_resp"] is not None and g["shift_resp"] >= 0.1
            out.append(dict(cam=cam, what="image shift (geometry)" + ("" if reliable else ", UNRELIABLE (low "
                                                                      "phase-correlation response)"),
                            sev=max(g["shift_px"] - c["floor_next_frame"]["shift_px"], 0) / 6.0 if reliable else 0.0,
                            area={"top": "joint fit / top camera pose", "wrist": "robot model (wrist camera mount) / "
                                  "gripper", "inspect": "inspect camera pose / joint fit"}[cam],
                            ev=f"median phase-correlation shift {g['shift_px']:.1f} px (dx {g['shift_dx']:+.1f}, dy "
                               f"{g['shift_dy']:+.1f}; floor {c['floor_next_frame']['shift_px']:.1f})"))
            out.append(dict(cam=cam, what="exposure / highlights", area="lighting",
                            sev=abs(g["lum_sim"] - g["lum_real"]) / 5.0
                            + 20 * abs(g["clip_hi_sim"] - g["clip_hi_real"]),
                            ev=f"L* real {g['lum_real']:.1f} vs sim {g['lum_sim']:.1f}; clipped highlights "
                               f"{100 * g['clip_hi_real']:.2f} % vs {100 * g['clip_hi_sim']:.2f} %"))
    for d in out:
        d["score"] = float(d["sev"] * w.get(d["cam"], 1.0))
    if rm and rm.get("samples"):
        key = "hold_err" if any("hold_err" in r for r in rm["samples"]) else "state_err"
        he = np.array([r[key] for r in rm["samples"] if key in r])
        if len(he):
            mx = np.abs(he[:, :5]).mean(0)
            out.append(dict(cam="all", what="servo cannot hold the recorded pose (sag after reset)",
                            area="joint fit / physics (servo model)", sev=float(mx.max()) / 1.5,
                            score=float(mx.max()) / 1.5,
                            ev="mean |measured - recorded| per joint " + " ".join(f"{v:.2f}" for v in mx)
                               + f" deg (max {np.abs(he[:, :5]).max():.1f})"))
    if rm and rm.get("grasp"):
        miss = [v["miss_m"] for v in rm["grasp"].values() if v.get("miss_m") is not None]
        if miss:
            out.append(dict(cam="all", what="fingertip-to-brick miss at the grasp frame",
                            area="joint fit / scene layout",
                            sev=float(np.median(miss)) / 0.008, score=float(np.median(miss)) / 0.008,
                            ev=f"median {1000 * np.median(miss):.1f} mm, max {1000 * np.max(miss):.1f} mm over "
                               f"{len(miss)} episodes (twin fingertip at the recorded close pose vs the detected "
                               "brick)"))
    return sorted(out, key=lambda d: -d["score"])


def trend_row(run: Path, prev_hash: dict) -> tuple[str, dict]:
    st, per = load(run / "state.json") or {}, load(run / "perceptual.json")
    pz, pf = load(run / "policy_zero.json"), load(run / "policy_ft.json")
    ctl = load(run / "control.json")
    cur = hashes(run)
    ch = changed(prev_hash, cur) if prev_hash else {}
    bad = run / "INVALID"
    if bad.exists():  # a run whose inputs were broken (e.g. all-black renders): listed, not scored
        return f"| {run.name[4:]} | {st.get('twin_hash', '?')[:8]} | INVALID: {bad.read_text().strip()} |" \
            + " |" * 18, cur
    chs = ", ".join(f"{a} ({len(v)})" for a, v in ch.items() if a in RENDER_AREAS) or ("first run" if not prev_hash
                                                                                         else "none")

    def pv(p, *keys):
        try:
            for k in keys:
                p = p[k]
            return p
        except (KeyError, TypeError):
            return None

    def cv(cam, *keys):
        return pv(per, "cams", cam, *keys)

    cells = [run.name[4:], st.get("twin_hash", "?")[:8] + " / " + st.get("audit_hash", "v1"), chs,
             f(pv(pz, "overall", "sim", "joint10", "mean")), f(pv(pz, "overall", "equivalence_index", "joint10")),
             f(pv(pz, "overall", "equivalence_index", "feature")),
             f(pv(pf, "overall", "sim", "joint10", "mean")), f(pv(pf, "overall", "equivalence_index", "joint10")),
             f(pv(pf, "overall", "equivalence_index", "feature")),
             f(pv(pf, "overall", "sim", "grip10", "mean"), 1) + " (" + f(pv(pf, "overall", "nbr", "grip10", "mean"), 1)
             + ")",
             f(pv(pf, "overall", "sim", "cos", "mean"), 3),
             (f"{pv(ctl, 'twin_ft8k', 'success')}/{pv(ctl, 'twin_ft8k', 'episodes')}" if ctl else "n/a"),
             " / ".join(f(cv(c, "kid", "real_vs_sim", "kid"), 2) for c in CAMS),
             " / ".join(f(cv(c, "paired_patch", "sim", "mean"), 2) for c in CAMS),
             " / ".join(f(cv(c, "cnn_highpass", "real_vs_sim", "crop_acc"), 2) for c in CAMS),
             " / ".join(f(cv(c, "regions", "table", "dE"), 1) for c in CAMS),
             f(cv("top", "plate_geo", "iou")) + " / " + f(cv("top", "plate_geo", "offset"), 0),
             " / ".join(f(cv(c, "regions", c2, "dE"), 1) for c, c2 in (("wrist", "bricks"), ("wrist", "plates"))),
             " / ".join(f(cv(c, "regions", "_global", "sat_sim"), 0) + ":"
                        + f(cv(c, "regions", "_global", "sat_real"), 0) for c in ("top", "wrist")),
             f(cv("wrist", "regions", "table", "contrast_ratio")),
             f(pv(pf, "close_timing", "sim"), 1) + " vs " + f(pv(pf, "close_timing", "real"), 1)]
    return "| " + " | ".join(cells) + " |", cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    run = Path(ap.parse_args().run)
    S = load_set()["samples"]
    st, rm, per = load(run / "state.json") or {}, load(run / "render_meta.json"), load(run / "perceptual.json")
    pols = [p for p in (load(run / "policy_ft.json"), load(run / "policy_zero.json")) if p]
    lays = load(run / "layouts.json") or {}
    made = []
    if (run / "sim.npz").exists():
        pm = np.load(run / "perceptual.npz") if (run / "perceptual.npz").exists() else None
        made = make_images(run, S, np.load(SET / "real.npz"), np.load(run / "sim.npz"), pm)
    runs = sorted(d for d in RUNS.iterdir() if d.is_dir() and d.name[:2] == "20" and (d / "state.json").exists())
    prev = [d for d in runs if d.name < run.name]
    L = [f"# Twin realism audit {run.name}", "",
         f"Host {st.get('host')}, {st.get('time')}; branch `{st.get('branch')}` @ `{str(st.get('commit'))[:7]}` + "
         f"{len(st.get('dirty', []))} dirty / untracked paths; twin source hash `{st.get('twin_hash')}` over "
         f"{st.get('n_files')} files (`runs/{run.name}/files.sha256`).",
         f"Audit set: {len(S)} timesteps of {len({s['ep'] for s in S})} held-out cal2 episodes "
         f"({', '.join(PHASES)}; 9 each), rendered by the default `TwinEnv` (colour correction on, randomisation off) "
         "at the recorded joint state with the episode's detected layout."]
    if prev:
        ch = changed(hashes(prev[-1]), hashes(run))
        L += ["", f"Changed since the previous audit ({prev[-1].name}): "
              + ("; ".join(f"**{a}** {len(v)} files ({', '.join(Path(x).name for x in v[:6])}"
                           f"{', ...' if len(v) > 6 else ''})" for a, v in ch.items()) or "nothing") + "."]
    if rm:
        err = np.abs(np.array([r["state_err"] for r in rm["samples"] if "state_err" in r]))
        miss = [v["miss_m"] for v in rm["grasp"].values() if v.get("miss_m") is not None]
        src = lays.get("source", {})
        ps = [x for x in (rm.get("physics_set") or []) if x] if isinstance(rm.get("physics_set"), list) else []
        L += ["", f"Render: {rm['n']} samples in {rm['render_s']} s ({rm.get('batches', 1)} batches"
              f"{', some with ' + ', '.join(ps) if ps else ''}), {len(rm['errors'])} errors; layouts "
              f"{sum(v == 'detected' for v in src.values())} detected / {sum(v == 'frozen' for v in src.values())} "
              f"frozen / {sum(v == 'none' for v in src.values())} missing. Sim joint state minus recorded after the "
              f"reset: mean |err| {f(float(err[:, :5].mean()))} deg (max {f(float(err[:, :5].max()))}), gripper "
              f"{f(float(err[:, 5].mean()))} (max {f(float(err[:, 5].max()))}). Fingertip-to-detected-brick miss at "
              f"the gripper-close frame: median {f(1000 * float(np.median(miss)), 1) if miss else 'n/a'} mm, max "
              f"{f(1000 * float(np.max(miss)), 1) if miss else 'n/a'} mm."]
        he = [r["hold_err"] for r in rm["samples"] if "hold_err" in r]
        if he:
            he = np.array(he)
            L += ["", "Servo hold error of a plain reset at the recorded pose (twin measured minus recorded after the "
                  "reset's settle ticks, before the audit's correction; pan, lift, elbow, wrist flex, roll, gripper): "
                  "mean " + " ".join(f"{v:+.2f}" for v in he.mean(0)) + "; max |.| "
                  + " ".join(f"{v:.2f}" for v in np.abs(he).max(0)) + " deg."]
        else:
            L += ["", "Harness note: this run rendered at the plain reset pose (no hold-error correction), so the "
                  "sim arm sat at the state error above."]
    L += ["", "## 1. Policy equivalence (headline)", ""]
    if not pols:
        L.append("Policy stage missing for this run.")
    for p in pols:
        L += policy_tables(p) + [""]
    ctl = load(run / "control.json")
    L += ["## 1b. Positive control (closed loop vs the real robot)", ""]
    L += control_tables(ctl) if ctl and ctl.get("twin_ft8k", {}).get("episodes") else ["Control stage missing."]
    L += ["", "## 2. Perceptual distances, classifiers, matched regions", ""]
    L += perceptual_tables(per) if per else ["Perceptual stage missing for this run."]
    L += ["", "## 3. Ranked discrepancies", ""]
    top10 = OUT / "top10.md"
    if top10.exists():
        L += [top10.read_text().strip(), ""]
    cand = candidates(per, pols[0] if pols else None, rm)
    L += ["### Automatic candidates (this run)", "",
          "Score = severity (1 = clearly visible: dE 6, a 1.6x sharpness ratio, a 2x noise ratio, 6 px shift) x the "
          "camera's policy weight (0.5 + its share of the single-camera action gap; inspect 0.3, not a policy input).",
          "", "| # | camera | discrepancy | evidence | score | area |", "|---|---|---|---|---|---|"]
    for k, d in enumerate(cand[:20]):
        L.append(f"| {k + 1} | {d['cam']} | {d['what']} | {d['ev']} | {d['score']:.2f} | {d['area']} |")
    L += ["", "## 4. Evidence images", ""] + [f"![{m}](img/{m})" for m in made]
    L += ["", "## 5. Trend", "",
          "| run | twin hash / audit harness | render-relevant areas changed (files) | zero-shot joints10 (deg) | "
          "index | feature index | fine-tune joints10 | index | feature index | fine-tune gripper10 (floor) | "
          "fine-tune feature "
          "cosine | twin closed-loop picks | KID top / wrist / inspect | DINO patch dist | high-pass CNN acc | "
          "table dE | top plate IoU / offset px | wrist bricks / plates dE | saturation sim:real top / wrist | "
          "wrist table contrast ratio | pregrasp close step sim vs real |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    ph: dict = {}
    for d in runs:
        row, ph = trend_row(d, ph)
        L.append(row)
    (run / "report.md").write_text("\n".join(L) + "\n")
    shutil.copy(run / "report.md", OUT / "report.md")
    if (run / "img").exists():
        (OUT / "img").mkdir(exist_ok=True)
        for p in (run / "img").iterdir():
            shutil.copy(p, OUT / "img" / p.name)
    print("REPORT_OK", run / "report.md", len(made), "images", flush=True)


if __name__ == "__main__":
    main()
