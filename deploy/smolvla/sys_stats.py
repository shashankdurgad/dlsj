"""Hardware utilisation sampler for the deploy laptop (stdlib only, standalone).

Samples CPU, memory, the robot-server process, the Intel Arc iGPU (xe driver),
the Intel NPU and a few temperatures from /proc and /sys on a background
thread, so a dashboard can poll ``snapshot()`` at ~2 Hz.  Every value is None
when its source is missing (e.g. on macOS), and nothing here raises.

This file deliberately imports nothing from the dlsj package so it can be
copied next to a web server on its own.

    stats = SysStats()
    stats.start()
    ...
    stats.snapshot()   # {"cpu_pct": 12.3, "gpu_pct": 41.0, ..., "history": {...}}
    stats.stop()

``root`` prefixes every /proc and /sys path so tests can use a fake tree.
Percentages derived from counters (cpu, gpu, npu, process cpu) need two
samples, so they are None until the second sample lands.
"""

from __future__ import annotations

import collections
import glob
import os
import threading
import time
from typing import Any, Deque, Dict, List, Optional, Tuple

HISTORY_LEN = 120
HISTORY_KEYS = ("cpu", "gpu", "npu", "mem_pct")

_SCALAR_KEYS = (
    "cpu_pct", "cpu_cores", "load1",
    "mem_used_gb", "mem_total_gb", "mem_pct",
    "proc_rss_gb", "proc_cpu_pct",
    "gpu_pct", "gpu_media_pct", "gpu_freq_mhz", "gpu_max_mhz",
    "npu_pct",
    "cpu_temp_c", "gpu_temp_c",
)


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def _round(x: Optional[float], nd: int = 1) -> Optional[float]:
    return None if x is None else round(x, nd)


class SysStats:
    def __init__(self, root: str = "/", period: float = 0.5, pid: Optional[int] = None):
        self.root = str(root)
        self.period = float(period)
        self.pid = os.getpid() if pid is None else int(pid)
        try:
            self._clk_tck = int(os.sysconf("SC_CLK_TCK"))
        except (AttributeError, ValueError, OSError):
            self._clk_tck = 100

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._latest: Dict[str, Any] = {k: None for k in _SCALAR_KEYS}
        self._latest["ts"] = None
        self._history: Dict[str, Deque[Optional[float]]] = {
            k: collections.deque(maxlen=HISTORY_LEN) for k in HISTORY_KEYS
        }

        # Previous raw counters, for deltas.
        self._prev_t: Optional[float] = None
        self._prev_cpu: Optional[Tuple[float, float]] = None      # (total, idle) jiffies
        self._prev_proc: Optional[float] = None                   # utime+stime jiffies
        self._prev_gt: Dict[str, float] = {}                      # gt dir -> idle_residency_ms
        self._prev_npu: Optional[float] = None                    # busy us

    # ------------------------------------------------------------------ API

    def start(self) -> "SysStats":
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="sys-stats", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=2.0)
        self._thread = None

    def snapshot(self) -> Dict[str, Any]:
        try:
            with self._lock:
                snap = dict(self._latest)
                snap["history"] = {k: list(v) for k, v in self._history.items()}
            return snap
        except Exception:  # never raise
            out: Dict[str, Any] = {k: None for k in _SCALAR_KEYS}
            out["ts"] = None
            out["history"] = {k: [] for k in HISTORY_KEYS}
            return out

    # ------------------------------------------------------------- internals

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._sample()
            except Exception:
                pass
            self._stop.wait(self.period)

    def _p(self, path: str) -> str:
        return os.path.join(self.root, path.lstrip("/"))

    def _read(self, path: str) -> Optional[str]:
        try:
            with open(self._p(path), "r") as f:
                return f.read()
        except Exception:
            return None

    def _read_num(self, path: str) -> Optional[float]:
        s = self._read(path)
        if s is None:
            return None
        try:
            return float(s.strip().split()[0])
        except Exception:
            return None

    def _glob(self, pattern: str) -> List[str]:
        """Sorted glob of a root-relative pattern, returned as root-relative paths."""
        try:
            hits = sorted(glob.glob(self._p(pattern)))
        except Exception:
            return []
        prefix = self._p("")
        out = []
        for h in hits:
            rel = os.path.relpath(h, prefix) if prefix else h
            out.append("/" + rel)
        return out

    def _sample(self, now: Optional[float] = None) -> Dict[str, Any]:
        """Take one sample, update latest + history, return the new values.

        ``now`` is a monotonic timestamp in seconds; tests pass it explicitly
        instead of sleeping.
        """
        if now is None:
            now = time.monotonic()
        dt = None if self._prev_t is None else now - self._prev_t
        if dt is not None and dt <= 0:
            dt = None

        v: Dict[str, Any] = {k: None for k in _SCALAR_KEYS}
        for fn in (self._s_cpu, self._s_mem, self._s_proc, self._s_gpu,
                   self._s_npu, self._s_temps):
            try:
                v.update(fn(dt))
            except Exception:
                pass

        self._prev_t = now
        v["ts"] = time.time()
        with self._lock:
            self._latest = v
            self._history["cpu"].append(v["cpu_pct"])
            self._history["gpu"].append(v["gpu_pct"])
            self._history["npu"].append(v["npu_pct"])
            self._history["mem_pct"].append(v["mem_pct"])
        return v

    # CPU ---------------------------------------------------------------

    def _s_cpu(self, dt: Optional[float]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        stat = self._read("/proc/stat")
        if stat:
            lines = stat.splitlines()
            cores = sum(1 for ln in lines if ln.startswith("cpu") and ln[3:4].isdigit())
            out["cpu_cores"] = cores or None
            for ln in lines:
                if ln.startswith("cpu "):
                    f = [float(x) for x in ln.split()[1:9]]
                    f += [0.0] * (8 - len(f))
                    total = sum(f)
                    idle = f[3] + f[4]
                    prev = self._prev_cpu
                    self._prev_cpu = (total, idle)
                    if prev is not None and total > prev[0]:
                        busy = 1.0 - (idle - prev[1]) / (total - prev[0])
                        out["cpu_pct"] = _round(_clamp(100.0 * busy))
                    break
        la = self._read("/proc/loadavg")
        if la:
            try:
                out["load1"] = round(float(la.split()[0]), 2)
            except Exception:
                pass
        return out

    # Memory ------------------------------------------------------------

    def _s_mem(self, dt: Optional[float]) -> Dict[str, Any]:
        txt = self._read("/proc/meminfo")
        if not txt:
            return {}
        info: Dict[str, float] = {}
        for ln in txt.splitlines():
            parts = ln.replace(":", " ").split()
            if len(parts) >= 2:
                try:
                    info[parts[0]] = float(parts[1])  # kB
                except ValueError:
                    pass
        total = info.get("MemTotal")
        avail = info.get("MemAvailable")
        if total is None or avail is None or total <= 0:
            return {}
        used = total - avail
        gb = 1024.0 * 1024.0  # kB -> GiB
        return {
            "mem_used_gb": round(used / gb, 2),
            "mem_total_gb": round(total / gb, 2),
            "mem_pct": _round(_clamp(100.0 * used / total)),
        }

    # Process -----------------------------------------------------------

    def _s_proc(self, dt: Optional[float]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        status = self._read(f"/proc/{self.pid}/status")
        if status:
            for ln in status.splitlines():
                if ln.startswith("VmRSS:"):
                    try:
                        out["proc_rss_gb"] = round(float(ln.split()[1]) / (1024.0 * 1024.0), 3)
                    except Exception:
                        pass
                    break
        st = self._read(f"/proc/{self.pid}/stat")
        if st:
            try:
                rest = st[st.rindex(")") + 2:].split()
                # rest[0] is field 3 (state); utime/stime are fields 14/15.
                jiffies = float(rest[11]) + float(rest[12])
            except Exception:
                jiffies = None
            if jiffies is not None:
                prev = self._prev_proc
                self._prev_proc = jiffies
                if prev is not None and dt and jiffies >= prev:
                    cpu_s = (jiffies - prev) / self._clk_tck
                    out["proc_cpu_pct"] = _round(max(0.0, 100.0 * cpu_s / dt))
        return out

    # GPU ---------------------------------------------------------------

    def _gt_dirs(self) -> Dict[str, str]:
        """Map role ('rc' / 'mc') -> root-relative gt dir, first card that has it."""
        found: Dict[str, str] = {}
        seen = set()
        for name_path in self._glob("/sys/class/drm/card*/device/tile0/gt*/gtidle/name"):
            try:
                real = os.path.realpath(self._p(name_path))
            except Exception:
                real = name_path
            if real in seen:
                continue
            seen.add(real)
            name = (self._read(name_path) or "").strip()
            role = name.rsplit("-", 1)[-1] if "-" in name else ""
            if role in ("rc", "mc") and role not in found:
                found[role] = os.path.dirname(os.path.dirname(name_path))  # .../gtN
        return found

    def _gt_busy(self, gt_dir: str, dt: Optional[float]) -> Optional[float]:
        idle = self._read_num(f"{gt_dir}/gtidle/idle_residency_ms")
        if idle is None:
            return None
        prev = self._prev_gt.get(gt_dir)
        self._prev_gt[gt_dir] = idle
        if prev is None or not dt or idle < prev:
            return None
        return _round(_clamp(100.0 * (1.0 - (idle - prev) / (dt * 1000.0))))

    def _s_gpu(self, dt: Optional[float]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        gts = self._gt_dirs()
        if "rc" in gts:
            gt = gts["rc"]
            out["gpu_pct"] = self._gt_busy(gt, dt)
            f = self._read_num(f"{gt}/freq0/act_freq")
            if f is not None:
                out["gpu_freq_mhz"] = int(f)
            m = self._read_num(f"{gt}/freq0/rp0_freq")
            if m is None:
                m = self._read_num(f"{gt}/freq0/max_freq")
            if m is not None:
                out["gpu_max_mhz"] = int(m)
        if "mc" in gts:
            out["gpu_media_pct"] = self._gt_busy(gts["mc"], dt)
        return out

    # NPU ---------------------------------------------------------------

    def _s_npu(self, dt: Optional[float]) -> Dict[str, Any]:
        for p in self._glob("/sys/class/accel/accel*/device/npu_busy_time_us"):
            busy = self._read_num(p)
            if busy is None:
                continue
            prev = self._prev_npu
            self._prev_npu = busy
            if prev is None or not dt or busy < prev:
                return {}
            return {"npu_pct": _round(_clamp(100.0 * (busy - prev) / (dt * 1e6)))}
        return {}

    # Temperatures ------------------------------------------------------

    def _s_temps(self, dt: Optional[float]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for hw in self._glob("/sys/class/hwmon/*"):
            name = (self._read(f"{hw}/name") or "").strip()
            if name == "coretemp" and "cpu_temp_c" not in out:
                best = None
                for inp in self._glob(f"{hw}/temp*_input"):
                    label = (self._read(inp.replace("_input", "_label")) or "").strip()
                    t = self._read_num(inp)
                    if t is None:
                        continue
                    if label == "Package id 0":
                        best = t
                        break
                    if best is None:
                        best = t
                if best is not None:
                    out["cpu_temp_c"] = round(best / 1000.0, 1)
            elif name == "xe" and "gpu_temp_c" not in out:
                best = None
                for inp in self._glob(f"{hw}/temp*_input"):
                    label = (self._read(inp.replace("_input", "_label")) or "").strip().lower()
                    t = self._read_num(inp)
                    if t is None:
                        continue
                    if "pkg" in label:
                        best = t
                        break
                    if best is None:
                        best = t
                if best is not None:
                    out["gpu_temp_c"] = round(best / 1000.0, 1)
        if "cpu_temp_c" not in out:
            for z in self._glob("/sys/class/thermal/thermal_zone*"):
                if (self._read(f"{z}/type") or "").strip() == "x86_pkg_temp":
                    t = self._read_num(f"{z}/temp")
                    if t is not None:
                        out["cpu_temp_c"] = round(t / 1000.0, 1)
                        break
        return out
