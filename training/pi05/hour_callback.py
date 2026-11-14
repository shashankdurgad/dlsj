"""Progress and final recovery checkpoint for the bounded one-hour run."""
import json
from pathlib import Path
import time

import torch
from lightning.pytorch import Callback


class HourProgress(Callback):
    def on_fit_start(self, trainer, pl_module):
        self.started = time.monotonic()
        self.times = []
        self.last_step = -1
        torch.cuda.reset_peak_memory_stats()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step == self.last_step:
            return
        self.last_step = step
        now = time.monotonic()
        self.times.append((step, now))
        if step == 100:
            # Measure the warm run before setting the final update budget. The
            # wall-clock Timer remains authoritative if throughput later slows.
            recent = self.times[-81:]
            seconds_per_step = (recent[-1][1] - recent[0][1]) / (recent[-1][0] - recent[0][0])
            remaining = max(0, 3000 - (now - self.started) - 120)
            # Never exceed the launcher's configured cap (it also sets the LR decay length).
            target = min(trainer.max_steps, step + max(1, int(remaining / seconds_per_step)))
            target = trainer.strategy.broadcast(target if trainer.is_global_zero else None)
            trainer.fit_loop.epoch_loop.max_steps = target
            if trainer.is_global_zero:
                Path(trainer.default_root_dir, "measured-budget.json").write_text(
                    json.dumps({"seconds_per_step": seconds_per_step,
                                "target_optimizer_steps": target}, indent=2) + "\n")
                print(f"MEASURED_UPDATE_BUDGET {target} steps at {seconds_per_step:.3f} s/update", flush=True)
        if trainer.is_global_zero and (step == 1 or step % 25 == 0):
            recent = self.times[-81:]
            seconds_per_step = ((recent[-1][1] - recent[0][1]) /
                                (recent[-1][0] - recent[0][0])) if len(recent) > 1 else None
            report = {
                "global_step": step, "elapsed_seconds": now - self.started,
                "recent_seconds_per_step": seconds_per_step,
                "world_size": trainer.world_size,
                "peak_gpu_allocated_gib_rank0": torch.cuda.max_memory_allocated() / 2**30,
                "trainable_parameters": sum(p.numel() for p in pl_module.parameters() if p.requires_grad),
                "total_parameters": sum(p.numel() for p in pl_module.parameters()),
                "loss": float(outputs["loss"].detach().cpu()) if isinstance(outputs, dict) and "loss" in outputs else None,
            }
            Path(trainer.default_root_dir, "progress.json").write_text(json.dumps(report, indent=2) + "\n")
            print("HOUR_PROGRESS " + json.dumps(report), flush=True)

    def on_train_end(self, trainer, pl_module):
        # Every rank participates in distributed checkpoint synchronization.
        trainer.save_checkpoint(str(Path(trainer.default_root_dir) / "final.ckpt"))
