"""Residual-RL demo video (1280x720, 30 fps): title card -> learning curve -> before/after on fixed eval seeds -> end card.

Same seeded episodes are run twice: frozen base alone ("before") and base + the trained residual ("after"), rendered
from the overview camera. Shown: the first episodes of seed `--seed` (not selected), then (--select changed, default)
episodes where the residual turned a failure into a success, labelled SELECTED with the fixed/broken counts; each panel
shows its real outcome. Numbers on the cards come from the trainer's curve.json / eval files, not from this script.

    python demo_video.py --run ~/residual_env/out/demo_scripted --base scripted --out demo_scripted.mp4
    python demo_video.py --run ~/residual_env/out/demo_molmo --base molmo --ipc ~/residual_env/ipc_molmo --latency 0.31
"""
import argparse
import json
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True, help="demo_train.py output dir (meta.json, curve.json, ckpt_final.pt)")
ap.add_argument("--ckpt", default="ckpt_final.pt")
ap.add_argument("--base", choices=["scripted", "molmo"], default="scripted")
ap.add_argument("--ipc", default=str(Path.home() / "residual_env/ipc_molmo"))
ap.add_argument("--latency", type=float, default=None, help="default: meta.json latency_s")
ap.add_argument("--seed", type=int, default=None, help="default: meta.json eval_seed")
ap.add_argument("--n_show", type=int, default=4)
ap.add_argument("--select", choices=["first", "changed"], default="changed",
                help="first: the first n_show episodes; changed: the first n_show/2 episodes, then n_show/2 episodes "
                     "where the residual turned a failure into a success (labelled as selected, with fixed/broken counts)")
ap.add_argument("--base_label", default="", help="e.g. 'MolmoAct2 LoRA (ours), frozen'")
ap.add_argument("--crop", default="390,205,800,450", help="x,y,w,h of the 1280x720 overview kept per panel ('' = all)")
ap.add_argument("--heldout", default="", help="demo_eval.py --out JSON: cards quote base vs residual on these held-out "
                                             "episodes instead of the trainer's eval")
ap.add_argument("--heldout_key", default="", help="which residual in --heldout (default: the last one)")
ap.add_argument("--out", default="")
args = ap.parse_args()
sys.path.insert(0, str(Path(__file__).resolve().parent))
run = Path(args.run).expanduser()
meta = json.loads((run / "meta.json").read_text())
curve = json.loads((run / "curve.json").read_text())
_log = json.loads((run / "train_log.json").read_text()) if (run / "train_log.json").exists() else []
for c in curve:  # older runs logged the replay-buffer size (capped at 500k) as "transitions": use env steps instead
    if c["tag"] != "base" and _log:
        steps = max([r["steps"] for r in _log if r["t_min"] <= c["t_min"] + 1e-6] or [0])
        c["transitions"] = max(c["transitions"], steps * meta["num_envs"])
seed = args.seed if args.seed is not None else meta["eval_seed"]
latency = args.latency if args.latency is not None else meta["latency_s"]
out_path = Path(args.out).expanduser() if args.out else run / f"demo_{args.base}.mp4"

from app import launch  # noqa: E402

app = launch(cameras=True)  # overview camera always; policy cameras only for the molmo base

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from app import make  # noqa: E402
from bases import IPCVLABase, ScriptedFailBase  # noqa: E402
from demo_lib import HOLD_TICKS, LIFT_M, demo_cfg, rollout  # noqa: E402
from sac import SAC  # noqa: E402

W, H, FPS = 1280, 720, 30
TW, TH = 640, 360
BG = (24, 22, 20)
FG = (235, 235, 235)
DIM = (150, 150, 150)
GOOD = (90, 200, 90)
BAD = (70, 70, 220)

n = max(args.n_show, 1)
base = ScriptedFailBase() if args.base == "scripted" else IPCVLABase(args.ipc)
cfg = demo_cfg(meta["num_envs"], cameras=args.base == "molmo", latency=latency)
cfg.overview = True
cfg.instructions = meta.get("instructions", cfg.instructions)
sim, scene, layout, env = make(cfg, base)
if hasattr(base, "env"):
    base.env = env
agent = SAC(env.obs_dim, env.act_dim, env.device)
agent.load(run / args.ckpt)
policy = lambda o: agent.act(o, deterministic=True)  # noqa: E731
groups = [("first", list(range(min(n, env.n))))]
counts = None
if args.select == "changed":  # unrecorded pass over all envs to find outcome changes, then record only the shown ones
    pb, pa = rollout(env, base, None, seed=seed), rollout(env, base, policy, seed=seed)
    fixed = [int(i) for i in np.where(~pb["success"] & pa["success"])[0]]
    broken = [int(i) for i in np.where(pb["success"] & ~pa["success"])[0]]
    counts = {"episodes": env.n, "fixed": len(fixed), "broken": len(broken),
              "both_success": int((pb["success"] & pa["success"]).sum())}
    k = max(1, n // 2)
    first = list(range(k))
    sel = [i for i in fixed if i not in first][:n - k]
    groups = [("first", first)] + ([("changed", sel)] if sel else [])
    print("SELECT", json.dumps({**counts, "first": first, "selected_fixed": sel}), flush=True)
show_list = [i for _, g in groups for i in g]
show = torch.as_tensor(show_list, device=env.device)


def capture(store):
    def rec(t):
        f = env.overview_frames(show).cpu().numpy()
        if args.crop:
            x0, y0, cw, ch = (int(v) for v in args.crop.split(","))
            f = f[:, y0:y0 + ch, x0:x0 + cw]
        store.append(np.stack([cv2.resize(cv2.cvtColor(x, cv2.COLOR_RGB2BGR), (TW, TH), interpolation=cv2.INTER_AREA)
                               for x in f]))
    return rec


frames_before, frames_after = [], []
m_before = rollout(env, base, None, seed=seed, record=capture(frames_before))
m_after = rollout(env, base, policy, seed=seed, record=capture(frames_after))


def txt(img, s, org, scale=0.8, col=FG, th=2):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, th, cv2.LINE_AA)


def card(lines, secs=4.0):
    img = np.full((H, W, 3), BG, np.uint8)
    y = 200
    for s, sc, col in lines:
        th = 2 if sc >= 0.8 else 1
        words, cur = s.split(" "), ""
        for w in words + [None]:  # wrap to the frame width
            nxt = w if not cur else (cur + " " + w if w is not None else cur)
            if w is not None and cv2.getTextSize(nxt, cv2.FONT_HERSHEY_SIMPLEX, sc, th)[0][0] <= W - 180:
                cur = nxt
                continue
            txt(img, cur, (90, y), sc, col, th)
            y += int(60 * max(sc, 0.6))
            cur = w or ""
    return [img] * int(secs * FPS)


def curve_card(secs=6.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [c["transitions"] for c in curve]
    ys = [100 * c["success_rate"] for c in curve]
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
    fig.patch.set_facecolor(np.array(BG[::-1]) / 255)
    ax.set_facecolor(np.array(BG[::-1]) / 255)
    ax.plot(xs, ys, "-o", color="#5ab4ff", lw=3, ms=7)
    ax.axhline(ys[0], color="#999999", ls="--", lw=1.5, label=f"frozen base alone: {ys[0]:.0f}%")
    ax.set_ylim(0, 100)
    ax.set_xlabel("environment transitions collected (sim)", color="w", fontsize=14)
    ax.set_ylabel(f"success on {curve[0]['episodes']} fixed eval episodes (%)", color="w", fontsize=14)
    ax.set_title(f"Residual SAC learning curve (task: grasp + lift >= {LIFT_M * 100:.0f} cm, hold {HOLD_TICKS / 30:.0f} s)",
                 color="w", fontsize=16)
    ax.tick_params(colors="w")
    for s in ax.spines.values():
        s.set_color("#777777")
    ax.legend(facecolor="#333333", labelcolor="w", fontsize=12)
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    plt.close(fig)
    img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), (W, H))
    return [img] * int(secs * FPS)


def panel(tile, label, ok, t, t_succ):
    tile = tile.copy()
    cv2.rectangle(tile, (0, 0), (TW, 34), (0, 0, 0), -1)
    txt(tile, label, (10, 24), 0.65)
    txt(tile, f"t={t / 30:4.1f}s", (TW - 110, 24), 0.6, DIM)
    if t_succ >= 0 and t >= t_succ:
        txt(tile, "SUCCESS", (TW - 140, TH - 14), 0.8, GOOD)
    elif t == len(frames_before) - 1 and not ok:
        txt(tile, "FAIL", (TW - 90, TH - 14), 0.8, BAD)
    return tile


base_label = args.base_label or ("scripted VLA-failure stand-in (NOT a learned VLA)" if args.base == "scripted"
                                 else "MolmoAct2, frozen")
video = []
s0, sf = curve[0], curve[-1]
eval_note = "fixed-seed episodes"
if args.heldout:
    ho = json.loads(Path(args.heldout).expanduser().read_text())
    key = args.heldout_key or [k for k in ho if k != "base"][-1]
    s0 = dict(ho["base"]["summary"])
    sf = dict(ho[key]["summary"], transitions=curve[-1]["transitions"], t_min=curve[-1]["t_min"])
    eval_note = "held-out episodes (seeds unseen in training)"
video += card([("Residual RL on a frozen robot policy (Isaac Lab sim)", 1.1, FG),
               (f"Base: {base_label}", 0.75, FG),
               ("SAC learns small bounded corrections (gripper, joints) added to the base's actions.", 0.65, DIM),
               (f"Sim twin of our SO-101 rig ({meta['calib']} calibration). Not deployed to the real robot.", 0.65, DIM),
               (f"Eval success: base alone {100 * s0['success_rate']:.0f}%  ->  base + residual "
                f"{100 * sf['success_rate']:.0f}%  ({sf['episodes']} {eval_note})", 0.75, FG)], 5.0)
video += curve_card()
for gname, gids in groups:
    for k in range(0, len(gids), 2):
        pair = gids[k:k + 2]
        if gname == "first":
            cap = (f"Same episodes, same seed: before (left) vs after (right). Episodes {', '.join(str(i + 1) for i in pair)}"
                   f" of seed {seed}: the first ones, NOT selected.")
        else:
            cap = (f"SELECTED episodes {', '.join(str(i + 1) for i in pair)}: ones where the residual turned a failure into a "
                   f"success. In these {counts['episodes']}: {counts['fixed']} fixed, {counts['broken']} broken.")
        video += card([(cap, 0.62, FG)], 2.5)
        T = len(frames_before)
        for t in range(T):
            img = np.full((H, W, 3), BG, np.uint8)
            for r, i in enumerate(pair):
                j = show_list.index(i)
                tag = "" if gname == "first" else " [selected]"
                img[r * TH:(r + 1) * TH, :TW] = panel(frames_before[t][j], f"ep {i + 1}{tag}  BEFORE: base alone",
                                                      bool(m_before["success"][i]), t, int(m_before["t_success"][i]))
                img[r * TH:(r + 1) * TH, TW:] = panel(frames_after[t][j], f"ep {i + 1}{tag}  AFTER: + residual",
                                                      bool(m_after["success"][i]), t, int(m_after["t_success"][i]))
            video.append(img)
        video += [video[-1]] * FPS
video += card([("What this shows", 1.0, FG),
               (f"Base alone {100 * s0['success_rate']:.0f}%  ->  with residual {100 * sf['success_rate']:.0f}% "
                f"on {sf['episodes']} {eval_note}, simulation.", 0.7, FG),
               (f"Training: {sf['transitions']:,} transitions, {sf['t_min']:.0f} min wall-clock on one GPU.", 0.65, DIM),
               (f"Base: {base_label}.", 0.65, DIM),
               ((f"Shown: the first {len(groups[0][1])} episodes of seed {seed}"
                 + (f", then {len(groups[1][1])} SELECTED where the residual fixed a failure "
                    f"({counts['fixed']} fixed / {counts['broken']} broken of {counts['episodes']})" if len(groups) > 1 else "")
                 + "."), 0.6, DIM),
               ("Not run on the real arm; sim-to-real transfer is untested.", 0.65, DIM)], 5.0)

wr = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
if not wr.isOpened():  # OpenCV without FFmpeg (Isaac venv on the Spark): built-in MJPEG/AVI writer, transcode later
    out_path = out_path.with_suffix(".avi")
    wr = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
assert wr.isOpened(), "no usable video writer"
for f in video:
    wr.write(f)
wr.release()
summary = {"video": str(out_path), "frames": len(video), "seed": seed, "groups": groups, "outcome_changes": counts,
           "before_success": m_before["success"][show_list].tolist(), "after_success": m_after["success"][show_list].tolist(),
           "eval_before": s0["success_rate"], "eval_after": sf["success_rate"]}
(out_path.with_suffix(".json")).write_text(json.dumps(summary, indent=1))
print("VIDEO_DONE", json.dumps(summary), flush=True)
if hasattr(base, "ipc"):
    (Path(base.ipc) / "done").write_text("1")
app.close()
