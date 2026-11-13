"""Latency of lerobot/pi05_base action-chunk inference on the Intel GPU (PyTorch, no OpenVINO)."""
import sys, time, numpy as np, torch
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy

dtype = sys.argv[1] if len(sys.argv) > 1 else "bfloat16"
dev = "xpu"
from lerobot.configs.policies import PreTrainedConfig
cfg = PreTrainedConfig.from_pretrained("lerobot/pi05_base"); cfg.device = dev; cfg.dtype = getattr(torch, dtype); cfg.num_inference_steps = int(sys.argv[2]) if len(sys.argv) > 2 else cfg.num_inference_steps
policy = PI05Policy.from_pretrained("lerobot/pi05_base", config=cfg).to(dev).eval()
pre, post = make_pre_post_processors(policy.config, "lerobot/pi05_base", preprocessor_overrides={"device_processor": {"device": dev}})
obs = {k: torch.rand(1, 3, 224, 224) for k in ("observation.images.base_0_rgb", "observation.images.left_wrist_0_rgb", "observation.images.right_wrist_0_rgb")}
obs["observation.state"] = torch.zeros(1, 32)
obs["task"] = ["place the brick on the blue plate"]

def run():
    with torch.inference_mode():
        b = pre(dict(obs))
        a = policy.predict_action_chunk(b)
        torch.xpu.synchronize()
    return a

t0 = time.perf_counter(); a = run(); print(f"first call (incl. warmup) {time.perf_counter()-t0:.1f}s, chunk shape {tuple(a.shape)}")
for _ in range(3): run()
ts = []
for _ in range(20):
    t = time.perf_counter(); run(); ts.append((time.perf_counter() - t) * 1e3)
print(f"pi05_base PyTorch-XPU {dtype}: median {np.median(ts):.0f} ms  p90 {np.percentile(ts,90):.0f} ms  min {min(ts):.0f} ms  (3 cams 224^2, {cfg.num_inference_steps} steps, chunk 50)")
print(f"XPU memory peak {torch.xpu.max_memory_allocated()/2**30:.1f} GiB")
