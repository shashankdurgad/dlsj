"""Per-stage timing (OPTIMIZE, PLAYBOOK 8.1) and a structured event log, both JSONL.

    <run_dir>/timings.jsonl  {"t": unix, "stage": "inference", "ms": 171.2, ...meta}   one line per sample,
                             then one {"type": "summary", "stages": {...}} line on close
    <run_dir>/events.jsonl   {"t": unix, "event": "inspection", ...}                   decisions, retries, results
    <run_dir>/summary.json   median / p90 / p99 / mean / n per stage

Stages written by the loop: perceive (state + camera grab -> Observation), inference (one policy chunk),
tick (control-loop work per tick, aggregated per skill), stall (per skill: time the arm had no fresh
target, i.e. starved ticks or inline sync inference), skill.<NAME> (wall time of a skill), settle,
detect (frame grab + inspector), inspector (InspectionResult.latency_ms), reason, pick, place, brick
(one whole brick cycle).
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if hasattr(o, "value"):  # Enum
        return o.value
    return str(o)


def summarize(values: list[float]) -> dict:
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "median": round(float(np.median(a)), 3), "p90": round(float(np.percentile(a, 90)), 3),
            "p99": round(float(np.percentile(a, 99)), 3), "mean": round(float(a.mean()), 3),
            "max": round(float(a.max()), 3)}


class RunLog:
    """Thread-safe JSONL writer for timings and events. `run_dir=None` keeps everything in memory only."""

    def __init__(self, run_dir: str | Path | None):
        self.run_dir = Path(run_dir) if run_dir else None
        self.samples: dict[str, list[float]] = defaultdict(list)
        self.events: list[dict] = []
        self._lock = threading.Lock()
        self._tf = self._ef = None
        if self.run_dir:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._tf = open(self.run_dir / "timings.jsonl", "a", buffering=1)
            self._ef = open(self.run_dir / "events.jsonl", "a", buffering=1)

    def timing(self, stage: str, ms: float, **meta) -> None:
        rec = {"t": round(time.time(), 4), "stage": stage, "ms": round(float(ms), 3), **meta}
        with self._lock:
            self.samples[stage].append(float(ms))
            if self._tf:
                self._tf.write(json.dumps(rec, default=_jsonable) + "\n")

    def event(self, event: str, **data) -> dict:
        rec = {"t": round(time.time(), 4), "event": event, **data}
        with self._lock:
            self.events.append(rec)
            if self._ef:
                self._ef.write(json.dumps(rec, default=_jsonable) + "\n")
        return rec

    def summary(self) -> dict:
        with self._lock:
            return {k: summarize(v) for k, v in sorted(self.samples.items())}

    def close(self) -> dict:
        s = self.summary()
        with self._lock:
            if self._tf:
                self._tf.write(json.dumps({"t": round(time.time(), 4), "type": "summary", "stages": s}) + "\n")
                self._tf.close()
                self._tf = None
            if self._ef:
                self._ef.close()
                self._ef = None
        if self.run_dir:
            (self.run_dir / "summary.json").write_text(json.dumps(s, indent=2) + "\n")
        return s


class Stopwatch:
    """`with Stopwatch(log, "detect", brick=3) as sw: ...` records the block's wall time in ms."""

    def __init__(self, log: RunLog, stage: str, **meta):
        self.log, self.stage, self.meta = log, stage, meta
        self.ms = 0.0

    def __enter__(self) -> "Stopwatch":
        self._t = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.ms = (time.perf_counter() - self._t) * 1000.0
        self.log.timing(self.stage, self.ms, **self.meta)
