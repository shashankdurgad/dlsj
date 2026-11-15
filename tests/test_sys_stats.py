import os

from dlsj.sys_stats import SysStats


def w(root, rel, text):
    p = os.path.join(str(root), rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(text)


def build(root, pid, *, cpu, mem_avail, proc_jiffies, idle0, idle1, act, npu, rss=2 * 1024 * 1024):
    w(root, "proc/stat",
      f"cpu  {cpu[0]} 0 {cpu[1]} {cpu[2]} 0 0 0 0 0 0\ncpu0 1 0 1 1 0 0 0 0 0 0\n"
      "cpu1 1 0 1 1 0 0 0 0 0 0\ncpu2 1 0 1 1 0 0 0 0 0 0\nintr 5\n")
    w(root, "proc/loadavg", "1.25 0.9 0.5 1/200 1234\n")
    w(root, "proc/meminfo",
      f"MemTotal:       {64 * 1024 * 1024} kB\nMemFree: 5 kB\nMemAvailable:   {mem_avail} kB\n")
    w(root, f"proc/{pid}/status", f"Name:\tx\nVmRSS:\t   {rss} kB\n")
    w(root, f"proc/{pid}/stat",
      f"{pid} (robot (srv)) S 1 1 1 0 -1 0 0 0 0 0 {proc_jiffies[0]} {proc_jiffies[1]} 0 0 20 0\n")
    gt = "sys/class/drm/card1/device/tile0"
    w(root, f"{gt}/gt0/gtidle/name", "gt0-rc\n")
    w(root, f"{gt}/gt0/gtidle/idle_residency_ms", f"{idle0}\n")
    w(root, f"{gt}/gt1/gtidle/name", "gt1-mc\n")
    w(root, f"{gt}/gt1/gtidle/idle_residency_ms", f"{idle1}\n")
    w(root, f"{gt}/gt0/freq0/act_freq", f"{act}\n")
    w(root, f"{gt}/gt0/freq0/rp0_freq", "2400\n")
    w(root, "sys/class/accel/accel0/device/npu_busy_time_us", f"{npu}\n")
    w(root, "sys/class/hwmon/hwmon3/name", "coretemp\n")
    w(root, "sys/class/hwmon/hwmon3/temp1_label", "Core 0\n")
    w(root, "sys/class/hwmon/hwmon3/temp1_input", "40000\n")
    w(root, "sys/class/hwmon/hwmon3/temp2_label", "Package id 0\n")
    w(root, "sys/class/hwmon/hwmon3/temp2_input", "61500\n")
    w(root, "sys/class/hwmon/hwmon5/name", "xe\n")
    w(root, "sys/class/hwmon/hwmon5/temp1_input", "55000\n")


def test_two_samples_compute_percentages(tmp_path):
    pid = 4242
    s = SysStats(root=str(tmp_path), pid=pid)
    tck = s._clk_tck
    build(tmp_path, pid, cpu=(100, 100, 800), mem_avail=48 * 1024 * 1024,
          proc_jiffies=(10, 10), idle0=1000, idle1=1000, act=800, npu=5_000_000)

    first = s._sample(now=100.0)
    assert first["cpu_pct"] is None and first["gpu_pct"] is None and first["npu_pct"] is None
    assert first["proc_cpu_pct"] is None
    assert first["cpu_cores"] == 3
    assert first["load1"] == 1.25
    assert first["mem_total_gb"] == 64.0
    assert first["mem_used_gb"] == 16.0
    assert first["mem_pct"] == 25.0
    assert first["proc_rss_gb"] == 2.0
    assert first["gpu_freq_mhz"] == 800 and first["gpu_max_mhz"] == 2400
    assert first["cpu_temp_c"] == 61.5 and first["gpu_temp_c"] == 55.0

    # 2 s later: cpu busy 50% (dtotal 400, didle 200), render GT 25% busy (idle +1500 of 2000ms),
    # media GT 100% idle, NPU busy 1.0 s of 2 s, process used 1.0 cpu-second (50% of one core * 2 s -> 50%).
    build(tmp_path, pid, cpu=(300, 100, 1000), mem_avail=32 * 1024 * 1024,
          proc_jiffies=(10 + tck // 2, 10 + tck // 2), idle0=2500, idle1=3000, act=1600,
          npu=6_000_000)
    second = s._sample(now=102.0)
    assert second["cpu_pct"] == 50.0
    assert second["gpu_pct"] == 25.0
    assert second["gpu_media_pct"] == 0.0
    assert second["npu_pct"] == 50.0
    assert second["proc_cpu_pct"] == 50.0
    assert second["mem_pct"] == 50.0
    assert second["gpu_freq_mhz"] == 1600

    # Process using 3 cores for 2 s -> 300%; GPU idle counter ahead of wall clamps to 0; NPU saturates.
    build(tmp_path, pid, cpu=(300, 100, 1000), mem_avail=32 * 1024 * 1024,
          proc_jiffies=(10 + tck // 2 + tck * 2, 10 + tck // 2 + tck * 4), idle0=2500 + 5000, idle1=3000,
          act=1600, npu=6_000_000 + 9_000_000)
    third = s._sample(now=104.0)
    assert third["proc_cpu_pct"] == 300.0
    assert third["gpu_pct"] == 0.0
    assert third["npu_pct"] == 100.0

    snap = s.snapshot()
    assert snap["gpu_pct"] == 0.0
    hist = snap["history"]
    assert set(hist) == {"cpu", "gpu", "npu", "mem_pct"}
    assert hist["gpu"] == [None, 25.0, 0.0]
    assert hist["npu"] == [None, 50.0, 100.0]
    assert hist["cpu"][1] == 50.0


def test_history_is_capped(tmp_path):
    s = SysStats(root=str(tmp_path))
    for i in range(130):
        s._sample(now=float(i))
    assert len(s.snapshot()["history"]["cpu"]) == 120


def test_thermal_zone_fallback(tmp_path):
    w(tmp_path, "sys/class/thermal/thermal_zone2/type", "x86_pkg_temp\n")
    w(tmp_path, "sys/class/thermal/thermal_zone2/temp", "47000\n")
    s = SysStats(root=str(tmp_path))
    assert s._sample(now=1.0)["cpu_temp_c"] == 47.0


def test_empty_tree_returns_nones(tmp_path):
    s = SysStats(root=str(tmp_path))
    snap = s.snapshot()
    assert snap["cpu_pct"] is None and snap["history"]["cpu"] == []
    s._sample(now=1.0)
    s._sample(now=2.0)
    snap = s.snapshot()
    for k in ("cpu_pct", "cpu_cores", "load1", "mem_used_gb", "mem_total_gb", "mem_pct",
              "proc_rss_gb", "proc_cpu_pct", "gpu_pct", "gpu_media_pct", "gpu_freq_mhz",
              "gpu_max_mhz", "npu_pct", "cpu_temp_c", "gpu_temp_c"):
        assert snap[k] is None, k


def test_start_stop_thread(tmp_path):
    s = SysStats(root=str(tmp_path), period=0.01)
    s.start()
    try:
        import time
        time.sleep(0.1)
        assert len(s.snapshot()["history"]["cpu"]) >= 2
    finally:
        s.stop()
