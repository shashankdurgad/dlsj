import json
from pathlib import Path
import statistics
import time
import torch
from lightning.pytorch import Callback

class StepTiming(Callback):
    def __init__(self):
        self.previous = None
        self.intervals = []
        self.validation_batches = 0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        torch.cuda.synchronize()
        now = time.perf_counter()
        if self.previous is not None and trainer.global_step >= 10:
            self.intervals.append(now - self.previous)
        self.previous = now

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.validation_batches += 1

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs['loss'] if isinstance(outputs, dict) else outputs
        assert torch.isfinite(loss).all(), 'Nonfinite training loss'

    def on_train_end(self, trainer, pl_module):
        torch.cuda.synchronize()
        result = {'gpu': torch.cuda.get_device_name(), 'gpus': trainer.num_devices,
            'per_gpu_batch': 8, 'precision': trainer.precision,
            'optimizer_steps': trainer.global_step, 'validation_batches': self.validation_batches, 'timed_intervals': len(self.intervals),
            'median_seconds_per_update': statistics.median(self.intervals),
            'mean_seconds_per_update': statistics.mean(self.intervals),
            'max_seconds_per_update': max(self.intervals),
            'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
            'peak_reserved_gib': torch.cuda.max_memory_reserved()/2**30,
            'synthetic_data': True, 'validation_and_checkpointing_included': False}
        Path('/tmp/pi05-benchmark-result.json').write_text(json.dumps(result, indent=2)+'\n')
        print('PI05_BENCHMARK_RESULT '+json.dumps(result), flush=True)
