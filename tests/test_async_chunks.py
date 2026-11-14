import threading
import time
from collections import deque

import numpy as np

from dlsj.deploy.async_chunks import AsyncChunker


class SlowChunker(AsyncChunker):
    """AsyncChunker with a fake 0.1 s 'policy' whose action i in a chunk equals (snapshot tick + i)."""

    def __init__(self, n_obs=2, n_act=15, delay=0.1):  # bypass the policy-based constructor
        self.n_obs, self.n_act, self.delay = n_obs, n_act, delay
        self.hist = deque(maxlen=n_obs)
        self.lock = threading.Lock()
        self.chunk = None
        self.latencies_ms = []
        self._stop = threading.Event()
        self._new_obs = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.snapshots = []

    def predict(self, snapshot):
        self.snapshots.append([t for t, _ in snapshot])
        time.sleep(self.delay)
        t0 = snapshot[-1][0]
        return np.array([[t0 + i] * 6 for i in range(self.n_act)], dtype=np.float32)


def test_async_chunks_never_freeze_and_skip_stale_actions():
    ch = SlowChunker().start()
    got = {}
    for tick in range(60):  # 2 s of a 30 Hz loop
        ch.push(tick, {"t": tick})
        a = ch.action(tick)
        if a is not None:
            got[tick] = a[0]
        time.sleep(1 / 30)
    ch.stop()
    first = min(got)
    assert first <= 8  # first chunk arrives after ~0.1 s (3 ticks) plus startup
    ticks = list(range(first, 60))
    assert all(t in got for t in ticks)  # never None once running: no freezes
    # the action executed at each tick is the one planned for that tick (stale ones skipped)
    assert np.mean([got[t] == t for t in ticks]) > 0.9
    # the observation history handed to the policy is consecutive 30 Hz ticks, as in training
    assert all(b - a == 1 for a, b in ch.snapshots)
