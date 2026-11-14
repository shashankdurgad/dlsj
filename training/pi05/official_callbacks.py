"""Checkpoints, GCS uploads and progress for the official full fine-tune.

* resume checkpoint (weights + optimizer + loop state) every ``resume_every``
  steps; only the latest is kept locally and in GCS (``resume/last.ckpt``);
* weights-only snapshots at fixed steps, each with a JSON sidecar
  (step, sha256, val loss) so early ones can be tried on the robot;
* weights-only best ``val/loss`` checkpoint (``best/best.ckpt``);
* ``progress.json`` with measured s/step and peak GPU memory.

Uploads run on a single background thread on rank 0. Every save is written
to ``*.pending`` first and renamed, so a preempted VM never leaves a
truncated file under a final name.
"""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import torch
from lightning.pytorch import Callback

SNAPSHOT_STEPS = (3000, 6000, 10000, 15000, 20000)


def sha256(path, chunk=1 << 24):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def metric(trainer, name):
    value = trainer.callback_metrics.get(name)
    return None if value is None else float(value)


class OfficialCheckpoints(Callback):
    def __init__(self, resume_every=2000, snapshot_steps=SNAPSHOT_STEPS):
        self.resume_every = int(resume_every)
        self.snapshot_steps = tuple(int(s) for s in snapshot_steps)
        self.best = None

    # Lightning stores this in resume checkpoints, so "best" survives preemption.
    def state_dict(self):
        return {"best": self.best}

    def load_state_dict(self, state):
        self.best = state.get("best")

    def on_fit_start(self, trainer, pl_module):
        self.root = Path(trainer.default_root_dir)
        self.uri = os.environ.get("RUN_ARTIFACTS")
        self.started = time.monotonic()
        self.times = []
        self.last_step = -1
        self.snapshotted = set()
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1) if trainer.is_global_zero else None
        self.pending = []
        torch.cuda.reset_peak_memory_stats()

    # -- saving helpers ---------------------------------------------------
    def _save(self, trainer, relative, weights_only, extra=None, keep_latest_only=False):
        destination = self.root / "checkpoints" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".pending")
        trainer.save_checkpoint(str(temporary), weights_only=weights_only)
        trainer.strategy.barrier()
        if trainer.is_global_zero:
            if keep_latest_only:
                for old in destination.parent.glob("*.ckpt"):
                    old.unlink()
            temporary.replace(destination)
            sidecar = destination.with_suffix(".json")
            info = {"checkpoint": relative, "step": trainer.global_step, "epoch": trainer.current_epoch,
                    "weights_only": weights_only, "bytes": destination.stat().st_size,
                    "sha256": sha256(destination), "val_loss": metric(trainer, "val/loss"),
                    "train_loss": metric(trainer, "train/loss"),
                    "elapsed_seconds": time.monotonic() - self.started,
                    "normalization": "TRAIN-NORMALIZATION.json (embedded in checkpoint)"}
            info.update(extra or {})
            sidecar.write_text(json.dumps(info, indent=2) + "\n")
            print("CHECKPOINT_READY " + json.dumps(info), flush=True)
            if self.uri:
                remote = f"{self.uri}/checkpoints/{Path(relative).parent.as_posix()}/".replace("/./", "/")

                def upload(paths=(destination, sidecar), remote=remote, name=relative):
                    subprocess.run(["gcloud", "storage", "cp", *map(str, paths), remote], check=True)
                    print("CHECKPOINT_UPLOADED", remote + Path(name).name, flush=True)
                self.pending.append(self.pool.submit(upload))
        trainer.strategy.barrier()

    def _snapshot(self, trainer, reason):
        step = trainer.global_step
        if step in self.snapshotted:
            return
        self.snapshotted.add(step)
        self._save(trainer, f"snapshots/step{step:06d}.ckpt", True, {"reason": reason})

    # -- hooks ------------------------------------------------------------
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step == self.last_step:
            return
        self.last_step = step
        now = time.monotonic()
        self.times = (self.times + [(step, now)])[-101:]
        if self.resume_every and step % self.resume_every == 0:
            self._save(trainer, "resume/last.ckpt", False, keep_latest_only=False)
        if trainer.is_global_zero and (step <= 1 or step % 25 == 0):
            first, last = self.times[0], self.times[-1]
            speed = (last[1] - first[1]) / (last[0] - first[0]) if last[0] > first[0] else None
            report = {"global_step": step, "max_steps": trainer.max_steps, "epoch": trainer.current_epoch,
                      "elapsed_seconds": now - self.started, "recent_seconds_per_step": speed,
                      "world_size": trainer.world_size,
                      "peak_gpu_allocated_gib_rank0": torch.cuda.max_memory_allocated() / 2**30,
                      "peak_gpu_reserved_gib_rank0": torch.cuda.max_memory_reserved() / 2**30,
                      "lr": trainer.optimizers[0].param_groups[0]["lr"] if trainer.optimizers else None,
                      "loss": float(outputs["loss"].detach().cpu())
                      if isinstance(outputs, dict) and "loss" in outputs else None,
                      "val_loss": metric(trainer, "val/loss"), "best_val_loss": self.best}
            (self.root / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
            print("TRAIN_PROGRESS " + json.dumps(report), flush=True)
            if step % 100 == 0 and self.uri:
                self.pending.append(self.pool.submit(
                    subprocess.run, ["gcloud", "storage", "cp", str(self.root / "progress.json"),
                                     self.uri + "/progress.json"], check=False))

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking or trainer.global_step == 0:
            return
        loss = metric(trainer, "val/loss")
        improved = trainer.strategy.broadcast(
            loss is not None and (self.best is None or loss < self.best["val_loss"]))
        if improved:
            self.best = {"val_loss": loss, "step": trainer.global_step}
            self._save(trainer, "best/best.ckpt", True, {"reason": "best val/loss"})
        if trainer.global_step in self.snapshot_steps:
            self._snapshot(trainer, "scheduled")

    def on_train_end(self, trainer, pl_module):
        step = trainer.global_step
        if step in self.snapshot_steps:
            self._snapshot(trainer, "scheduled")
        elif step < trainer.max_steps:
            # Stopped early (deadline): keep a weights-only copy and a resume point.
            self._save(trainer, f"snapshots/stopped-step{step:06d}.ckpt", True, {"reason": "stopped early"})
        if self.resume_every and step % self.resume_every:
            self._save(trainer, "resume/last.ckpt", False)
        if trainer.is_global_zero:
            for future in self.pending:
                future.result()
            self.pool.shutdown(wait=True)
