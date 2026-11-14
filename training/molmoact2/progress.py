"""Lightning callback: write progress.json (step, loss, s/step, peak GPU memory) every 25 optimizer steps."""
import json
from pathlib import Path
import statistics
import time

import torch
from lightning.pytorch import Callback


class Progress(Callback):
    def __init__(self, every: int = 25, warmup_steps: int = 20):
        self.every, self.warmup_steps = every, warmup_steps

    def on_fit_start(self, trainer, pl_module):
        self.started, self.last, self.intervals, self.step = time.monotonic(), None, [], -1
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step == self.step:  # gradient accumulation: only count optimizer steps
            return
        now = time.monotonic()
        if self.last is not None and step > self.warmup_steps:
            self.intervals.append((now - self.last[1]) / max(1, step - self.last[0]))
        self.last, self.step = (step, now), step
        if trainer.is_global_zero and (step % self.every == 0 or step == trainer.max_steps):
            loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
            report = {
                "global_step": step, "max_steps": trainer.max_steps, "elapsed_s": round(now - self.started, 1),
                "median_s_per_step": statistics.median(self.intervals) if self.intervals else None,
                "loss": float(loss.detach().cpu()) if torch.is_tensor(loss) else None,
                "world_size": trainer.world_size,
                "peak_gpu_gib_rank0": round(torch.cuda.max_memory_allocated() / 2**30, 2)
                if torch.cuda.is_available() else None,
                "trainable_params": sum(p.numel() for p in pl_module.parameters() if p.requires_grad),
                "total_params": sum(p.numel() for p in pl_module.parameters()),
            }
            if report["median_s_per_step"]:
                left = max(0, trainer.max_steps - step) * report["median_s_per_step"]
                report["eta_hours"] = round(left / 3600, 2)
            Path(trainer.default_root_dir, "progress.json").write_text(json.dumps(report, indent=2) + "\n")
            print("PROGRESS " + json.dumps(report), flush=True)
