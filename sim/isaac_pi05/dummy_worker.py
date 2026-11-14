import sys, time, os, numpy as np
from pathlib import Path
ipc = Path(sys.argv[1]); i = 0
while not (ipc / "done").exists():
    f = ipc / f"obs-{i:04d}.npz"
    if not f.exists(): time.sleep(0.02); continue
    time.sleep(0.05); s = np.load(f)["state"]
    a = np.repeat(s[None], 50, 0); a[:, 1] += np.linspace(0, 10, 50)  # small shoulder_lift motion
    np.savez(ipc / f"actions-{i:04d}.tmp.npz", actions=a.astype(np.float32), latency_s=0.0)
    os.replace(ipc / f"actions-{i:04d}.tmp.npz", ipc / f"actions-{i:04d}.npz"); i += 1
