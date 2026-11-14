"""One command: export MolmoAct2 optimisation variants for a checkpoint, test them, recommend one.

    cd ~/inference/physical-ai-studio/library && HF_HOME=~/inference/hf-molmo HF_HUB_OFFLINE=1 \
    .venv/bin/python ~/inference/deploy/molmoact2/optimize.py \
        --name lora_v1 --ckpt runs/molmoact2/.../last.ckpt --prompts three_phase \
        --variants fp16_ref,int8_sym,int8_keep_ae,steps5_int8,steps3_int8 [--frames 60]

For each variant (built once, reused across runs): latency on the XPS GPU and agreement with the FP16 / 10-step
reference of the SAME checkpoint on recorded frames from every phase, plus error vs the human demo. PASS = within the
INT8-vs-FP16 noise floor (harness.JOINT_TOL / GRIP_TOL) and demo error not worse than harness.DEMO_TOL. Writes
<workdir>/report.md + report.json and prints the fastest PASS variant and its run_molmo.sh launch line.

Variant names:
  fp16_ref                reference export (traced from the checkpoint, FP16 weights, 10 flow steps)
  int8_sym | int8_asym | int8_keep_ae | int8_keep_ae_vit | int4_lm      quantize.py variants of fp16_ref
  steps{K}_fp16           re-traced with K flow steps; steps{K}_int8 = its int8_sym
  res{S}_fp16 / res{S}_int8   square image size S (zero-shot = experiment; adopt only for a model trained at S)
  text{N}_fp16 / text{N}_int8 static text length N (default 256, ~98 used): removes pad tokens from the prefix
  s{K}t{N}_fp16 / s{K}t{N}_int8 / s{K}t{N}_rtc_int8   combined steps K + text length N (+ RTC inputs)
  rtc_fp16 / rtc_int8     RTC-patched export (rtc_patch.py); predictions are compared with guidance ~0
  notext_<base>           <base> with the setup/control text tokens switched off in the manifest (no re-trace)
  <base>@KEY=VALUE[,..]   <base> with OpenVINO GPU compile options (e.g. int8_sym@DYNAMIC_QUANTIZATION_GROUP_SIZE=0)
Heavy steps (re-traces, ~44 GB RAM) run one at a time under ~/inference/tmp/heavy.lock with a memory watchdog.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness as H  # noqa: E402

LOCK = Path("~/inference/tmp/heavy.lock").expanduser()
MIN_FREE_GB = 6


def mem_available_gb() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1048576
    return 99.0


def arm_busy() -> list[str]:
    """Live robot controllers (pause GPU work while the arm is driven)."""
    pat = re.compile(r"(pi05_control|rtc_control|smolvla_control|teleop_record|sweep_live|act_rollout)")
    busy = []
    for p in Path("/proc").glob("[0-9]*"):
        try:
            argv = (p / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        exe = os.path.basename(argv[0].decode(errors="ignore")) if argv and argv[0] else ""
        # only real python processes (not shells/ssh whose command text merely mentions a controller name)
        if exe.startswith("python") and pat.search(b" ".join(argv[1:4]).decode(errors="ignore")) \
                and str(os.getpid()) != p.name:
            busy.append(b" ".join(argv).decode(errors="ignore")[:120])
    return busy


def wait_arm_free(log) -> None:
    while (b := arm_busy()):
        log(f"arm in use ({b[0][:60]}...): waiting")
        time.sleep(60)


def run_heavy(cmd: list[str], log) -> None:
    """Run a memory-heavy subprocess under the shared lock, killing it if RAM runs out."""
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK, "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        wait_arm_free(log)
        log("$ " + " ".join(cmd))
        p = subprocess.Popen(cmd, start_new_session=True)
        killed = threading.Event()

        def guard():
            while p.poll() is None:
                if mem_available_gb() < MIN_FREE_GB:
                    os.killpg(p.pid, 15)
                    killed.set()
                    return
                time.sleep(1)

        threading.Thread(target=guard, daemon=True).start()
        rc = p.wait()
        if killed.is_set():
            raise RuntimeError(f"killed: MemAvailable < {MIN_FREE_GB} GB")
        if rc:
            raise RuntimeError(f"command failed ({rc}): {' '.join(cmd)}")


class Builder:
    def __init__(self, args, log):
        self.a, self.log = args, log
        self.work = Path(args.workdir).expanduser() / args.name
        self.work.mkdir(parents=True, exist_ok=True)
        self.py = sys.executable

    def path(self, name: str) -> Path:
        return self.work / name

    def build(self, name: str) -> Path:
        base, _, _ = name.partition("@")
        out = self.path(base)
        if (out / "manifest.json").exists():
            return out
        if base == "fp16_ref":
            self._export(out, [])
        elif m := re.fullmatch(r"steps(\d+)_fp16", base):
            self._export(out, ["--steps", m.group(1)])
        elif m := re.fullmatch(r"res(\d+)_fp16", base):
            self._export(out, ["--image-size", m.group(1)])
        elif m := re.fullmatch(r"text(\d+)_fp16", base):
            self._export(out, ["--text-len", m.group(1)])
        elif m := re.fullmatch(r"s(\d+)t(\d+)(_rtc)?_fp16", base):  # combined: steps + text length (+ RTC)
            self._export(out, ["--steps", m.group(1), "--text-len", m.group(2)] + (["--rtc"] if m.group(3) else []))
        elif base == "rtc_fp16":
            self._export(out, ["--rtc"])
        elif m := re.fullmatch(r"(steps\d+|res\d+|text\d+|s\d+t\d+(?:_rtc)?|rtc)_int8", base):
            self._quant(self.build(f"{m.group(1)}_fp16"), out, "int8_sym")
        elif base in ("int8_sym", "int8_asym", "int8_keep_ae", "int8_keep_ae_vit", "int4_lm"):
            self._quant(self.build("fp16_ref"), out, base)
        elif base.startswith("notext_"):
            src = self.build(base.removeprefix("notext_"))
            out.mkdir(parents=True, exist_ok=True)
            for f in src.iterdir():
                if f.name != "manifest.json":
                    (out / f.name).symlink_to(f.resolve())
            man = json.loads((src / "manifest.json").read_text())
            for p in man["model"]["preprocessors"]:
                if p.get("type") == "molmoact2":
                    p["add_setup_tokens"] = False
                    p["add_control_tokens"] = False
            (out / "manifest.json").write_text(json.dumps(man, indent=1))
        else:
            raise SystemExit(f"unknown variant {name}")
        return out

    def _export(self, out: Path, extra: list[str]) -> None:
        cmd = [self.py, str(HERE / "export.py"), "--out", str(out), "--cams", self.a.cams]
        cmd += ["--ckpt", self.a.ckpt] if self.a.ckpt else ["--pretrained", self.a.pretrained]
        run_heavy(cmd + extra, self.log)

    def _quant(self, src: Path, out: Path, variant: str) -> None:
        run_heavy([self.py, str(HERE / "quantize.py"), str(src), str(out), "--variant", variant], self.log)


def parse_cfg(name: str) -> dict:
    _, _, tail = name.partition("@")
    return dict(kv.split("=", 1) for kv in tail.split(",") if kv) if tail else {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="run name (folder under --workdir)")
    ap.add_argument("--ckpt", help="Studio Lightning checkpoint (LoRA fine-tune)")
    ap.add_argument("--pretrained", default="allenai/MolmoAct2-SO100_101")
    ap.add_argument("--data", default="~/inference/data/duplo_teleop_cal2_3phase")
    ap.add_argument("--prompts", default="zeroshot", help="zeroshot | three_phase | path to a json {grasp,present,place}")
    ap.add_argument("--cams", default="wrist,top")
    ap.add_argument("--crop", default="", help="cam=x,y,w,h[;...] as the training set (MolmoAct2 fine-tune: top=0,0,560,480)")
    ap.add_argument("--brick", default="red", help="{brick} for episodes whose label doesn't name the brick colour")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--variants", default="fp16_ref,int8_sym,int8_keep_ae,steps5_int8")
    ap.add_argument("--workdir", default="~/inference/exports/molmo_opt")
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--cache-dir", default="~/inference/ov_cache")
    a = ap.parse_args()

    prompts = {"zeroshot": H.ZEROSHOT, "three_phase": H.THREE_PHASE}.get(a.prompts) or json.loads(Path(a.prompts).read_text())
    b = Builder(a, lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True))
    log = b.log
    crops = {k: tuple(int(v) for v in xywh.split(",")) for k, xywh in
             (part.split("=", 1) for part in filter(None, a.crop.split(";")))}
    tag = a.prompts if a.prompts in ('zeroshot', 'three_phase') else Path(a.prompts).stem
    frames = H.load_frames(a.data, a.frames, prompts, cams=tuple(a.cams.split(",")), crops=crops, brick=a.brick,
                           cache=b.work / f"frames_{tag}{'_crop' if crops else ''}_{a.frames}.npz")
    log(f"{len(frames)} frames")
    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    if "fp16_ref" not in variants:
        variants.insert(0, "fp16_ref")
    results, preds = {}, {}
    for v in variants:
        try:
            exp = b.build(v)
            pfile = b.work / f"pred_{v.replace('@', '__').replace('=', '-').replace(',', '_')}.npz"
            if pfile.exists():
                z = np.load(pfile, allow_pickle=True)
                pred, lat = z["pred"], z["lat"].item()
            else:
                wait_arm_free(log)
                with open(LOCK, "w") as lk:
                    fcntl.flock(lk, fcntl.LOCK_EX)
                    pred, lat = H.predict(exp, frames, a.device, a.cache_dir, **parse_cfg(v))
                np.savez(pfile, pred=pred, lat=np.asarray(lat, dtype=object))
            preds[v] = pred
            results[v] = {"export": str(exp), **lat}
            log(f"{v}: {lat}")
        except Exception as e:  # noqa: BLE001
            results[v] = {"error": f"{type(e).__name__}: {e}"[:300]}
            log(f"{v}: FAILED {results[v]['error']}")
    ref = preds.get("fp16_ref")
    if ref is None:
        raise SystemExit("no reference predictions; see errors above")
    for v, p in preds.items():
        results[v]["quality"] = H.compare(p, ref, frames)
    ok = [v for v, r in results.items() if "quality" in r and r["quality"]["all"]["pass"]]
    best = min(ok, key=lambda v: results[v]["median_ms"]) if ok else None

    lines = [f"# MolmoAct2 optimisation: {a.name}", "", f"checkpoint: {a.ckpt or a.pretrained}; prompts: {a.prompts}; "
             f"frames: {len(frames)}; reference: fp16_ref", "",
             "| variant | median ms | p90 ms | vs ref joints | vs ref gripper | demo err (ref) | PASS |", "|---|---|---|---|---|---|---|"]
    for v, r in results.items():
        if "quality" not in r:
            lines.append(f"| {v} | - | - | - | - | - | error: {r.get('error', '')[:60]} |")
            continue
        q = r["quality"]["all"]
        lines.append(f"| {v} | {r['median_ms']} | {r['p90_ms']} | {q['vs_ref_joints']} | {q['vs_ref_gripper']} | "
                     f"{q['demo_err_joints']} ({q['demo_err_ref_joints']}) | {'PASS' if q['pass'] else 'fail'} |")
    if best:
        cfg = parse_cfg(best)
        lines += ["", f"**Recommended: `{best}`** ({results[best]['median_ms']} ms).",
                  f"Launch: `EXPORT={results[best]['export']} ~/inference/deploy/smolvla/run_molmo.sh`"
                  + (f"  (+ compile options {cfg}: add to the policy wrapper's config)" if cfg else "")]
    report = "\n".join(lines)
    (b.work / "report.md").write_text(report + "\n")
    (b.work / "report.json").write_text(json.dumps({"results": results, "recommended": best}, indent=1, default=str))
    print(report)


if __name__ == "__main__":
    main()
