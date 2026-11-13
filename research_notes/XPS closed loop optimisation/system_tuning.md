# System-level tuning of the closed-loop robot pipeline on the Dell XPS 16 (DA16260, Core Ultra X9 388H, Ubuntu 26.04, kernel 7.0, xe)

Scope: OS, hardware and concurrency settings for 3 UVC/RealSense cameras, pi0.5 on the iGPU (OpenVINO), Anomalib on the NPU, and the SO-101 Feetech loop at 30-60 Hz in Python 3.12. Research date: 2026-11-14.

How to read this file: claims under "Cited Findings" are sourced. Commands and facts marked "(standard practice, not re-verified this session)" come from long-standing kernel and tool behaviour that I could not re-fetch within the tool budget. The report writer should present them as recommendations to verify on the machine, not as sourced facts. I ran about 20 tool calls. Several primary docs (the OpenVINO thread-scheduling page body, the LeRobot SerialMotorsBus source, compute-runtime debug keys) did not render or have moved, so they are listed as gaps.

---

## 1. Panther Lake core topology, pinning, RT scheduling, Thread Director on kernel 7.0

### Takeaway
Linux on kernel 7.0 sees three core classes, with different `cpu_capacity` values and max frequencies. It fills P-cores, then E-cores, then LP E-cores, but it does not reliably keep foreground or latency-critical threads on P-cores when the machine is loaded. Intel's cluster-scheduling fix for hybrid P/E/LPE parts was queued for Linux 7.3, so the stock 7.0 kernel in Ubuntu 26.04 does not have it. Explicit pinning is therefore needed: servo loop on a dedicated P-core, inference and runtime threads on the other P-cores and E-cores, and housekeeping on LP E-cores.

### Cited Findings
- On a sibling Panther Lake part (Core Ultra 7 356H, 4P+8E+4LPE), Linux CPUs 0-3 are P-cores, 4-11 are E-cores and 12-15 are LP E-cores. Measured `cpu_capacity` is 1004-1024 for P, 714 for E and 637 for LP E, and `cpuinfo_max_freq` is 4.7 / 3.7 / 3.3 GHz. dmesg shows "Hybrid CPU capacity scaling enabled". — [LinuxLinks, Does Linux schedule Panther Lake correctly (p1)](https://www.linuxlinks.com/minisforum-m2-does-linux-schedule-panther-lake-correctly/)
- Commands to identify core types: `lscpu -e=CPU,CORE,SOCKET,NODE,ONLINE,MAXMHZ,MINMHZ`, plus a sysfs loop over `/sys/devices/system/cpu/cpu*/{topology/core_id,topology/cluster_id,cpufreq/cpuinfo_max_freq,cpu_capacity}`, plus `sudo dmesg | grep -Ei 'hybrid|capacity|intel_pstate|hfi'`. — [LinuxLinks p1](https://www.linuxlinks.com/minisforum-m2-does-linux-schedule-panther-lake-correctly/)
- Per-core SHA-256 throughput: an E-core gives about 80% and an LP E-core about 75% of a P-core. — [LinuxLinks p1](https://www.linuxlinks.com/minisforum-m2-does-linux-schedule-panther-lake-correctly/)
- With 12 low-priority workers on all P and E cores, a normal-priority foreground thread was placed on an idle LP E-core and ran about 24% below its P-core result. With all 16 CPUs busy it spent only 29% of its time on P-cores and migrated 827 times across all three classes. The author concludes Linux "does not consistently protect foreground single-thread performance". Niceness decides CPU-time share, not core class. — [LinuxLinks p9 (conclusions)](https://www.linuxlinks.com/minisforum-m2-does-linux-schedule-panther-lake-correctly/9/); [Test 5](https://www.linuxlinks.com/minisforum-m2-does-linux-schedule-panther-lake-correctly/7/); [Test 6](https://www.linuxlinks.com/minisforum-m2-does-linux-schedule-panther-lake-correctly/8/)
- Intel patches to improve cluster-aware load balancing on Core Ultra hybrid CPUs that mix P with E/LPE cores were queued for the Linux 7.3 merge window, so kernel 7.0 does not include them. — [Phoronix: Linux 7.3 better Intel hybrid](https://www.phoronix.com/news/Linux-7.3-Better-Intel-Hybrid) (search snippet; the page returned 403 on fetch); [hwbusters](https://hwbusters.com/news/intel-says-linux-cluster-scheduling-fails-on-its-hybrid-cpus-the-fix-is-queued-for-7-3/); [PCGH](https://www.pcgameshardware.de/Linux-Software-26761/News/Kernel-7-3-Intel-Scheduler-P-Cores-E-Cores-1550686/)
- The Core Ultra X7 358H has 4P + 8E + 4LPE (16 cores, no SMT). — [Phoronix X7 358H Linux review (via search snippet)](https://scour.ing/@plooh/p/https://www.phoronix.com/review/intel-core-ultra-x7-358h-linux)

### Commands (standard practice, not re-verified this session; check the CPU numbering on the actual 388H first)
```bash
# 1. Map the topology. Expect 4P + 8E + 4LPE = 16 CPUs, no SMT; confirm on the 388H.
lscpu -e=CPU,CORE,CLUSTER,MAXMHZ
for c in /sys/devices/system/cpu/cpu[0-9]*; do echo "$(basename $c) cap=$(cat $c/cpu_capacity) max=$(cat $c/cpufreq/cpuinfo_max_freq) cluster=$(cat $c/topology/cluster_id)"; done
cat /sys/devices/cpu_core/cpus /sys/devices/cpu_atom/cpus   # perf PMU split. LP E-cores are also "atom"; tell them apart by the lower max_freq/capacity.

# 2. Suggested partition (assumes P = 0-3, E = 4-11, LPE = 12-15):
#    CPU 3      -> servo control loop (SCHED_FIFO), alone
#    CPU 2      -> camera capture threads + frame assembly
#    CPU 0-1    -> OpenVINO host threads (GPU/NPU submission, pre/post-processing)
#    CPU 4-11   -> OpenVINO CPU-side ops, tokenisation, Anomalib pre-processing, logging, recording
#    CPU 12-15  -> OS housekeeping, IRQs, desktop
taskset -c 3 python control_loop.py         # or os.sched_setaffinity(0, {3}) inside Python
sudo chrt -f 80 taskset -c 3 python control_loop.py
# Inside Python (needs CAP_SYS_NICE, or an rtprio entry in /etc/security/limits.d/):
#   os.sched_setaffinity(0, {3}); os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(80))
# Per thread: threading.get_native_id() then os.sched_setaffinity(tid, {...})

# 3. Optional harder isolation (kernel cmdline in /etc/default/grub, then update-grub):
#    isolcpus=managed_irq,domain,3 nohz_full=3 rcu_nocbs=3 irqaffinity=12-15
#    Lighter alternative without a reboot: a cpuset cgroup or a systemd slice with AllowedCPUs=
sudo systemctl set-property --runtime user.slice AllowedCPUs=0-2,4-15   # keep desktop work off CPU 3
# 4. Move IRQs (USB xHCI in particular) off the control core:
grep -E 'xhci|i915|xe|vpu' /proc/interrupts; echo 12-15 | sudo tee /proc/irq/<N>/smp_affinity_list
# 5. RT throttling: the default sched_rt_runtime_us=950000 keeps a safety margin. Leave it.
# 6. Check timer jitter on the pinned core:
sudo cyclictest -m -p80 -a3 -i1000 -D60 --histogram=200
```

### Inferences
- Do not rely on EAS, HFI or Thread Director to keep the 30-60 Hz servo loop on a P-core under combined iGPU, NPU and camera load. Given the LinuxLinks results and the missing 7.3 fix, pin it explicitly. Keep it off the LP E-cores. LP E-cores on Meteor Lake and Lunar Lake sit in the SoC/low-power island with lower frequency and a separate cache path, and I expect the same on Panther Lake (not verified).
- `isolcpus`/`nohz_full` is probably more than needed at 30-60 Hz, where the period is 16-33 ms. Pinning plus SCHED_FIFO plus `irqaffinity` typically gives sub-millisecond scheduling jitter on PREEMPT_DYNAMIC kernels. Use `cyclictest` to decide. Ubuntu kernels are usually PREEMPT_DYNAMIC, so `preempt=full` on the cmdline is a cheap upgrade. PREEMPT_RT has been mainline since 6.12, and Ubuntu ships a separate real-time kernel flavour, but I did not verify that Ubuntu 26.04 offers one.
- SCHED_FIFO on a Python process only helps the thread that holds the GIL. Put the servo loop in its own process so camera or inference threads in the same interpreter cannot block it via the GIL.
- Thread Director / HFI: dmesg mentions HFI, but I found no evidence that kernel 7.0 uses ITD per-thread classification (IPC classes) for placement. Treat ITD as having no effect on Linux 7.0 placement for this workload. See Gaps.

### Gaps
- No source confirming the exact CPU numbering of the X9 388H. It is presumably the same 4P+8E+4LPE, but it must be checked with the commands above.
- No source on whether the IPC-classes / ITD scheduler patches landed in mainline by 7.0. My understanding is they had not, but I could not confirm.
- No Panther Lake `cyclictest` latency data found.
- Whether Ubuntu 26.04 will get a 7.3+ HWE kernel and when: not found.

---

## 2. Power and performance: platform profile, EPP, GPU frequency/RC6, NPU turbo, Dell thermal modes, throttling detection

### Takeaway
Run on AC, set the platform profile to performance (this drives Dell's firmware fan/thermal mode via `dell-pc`), set EPP to `performance`, and pin the xe GT min frequency up to cut GPU ramp-up latency on bursty 150-470 ms inference. Detect throttling via xe `freq0/throttle/reasons`, `turbostat`, and thermal-zone temperatures. The 2026 XPS 16 runs warmer than its predecessor and throttles under some sustained loads (Windows testing). Expect sustained iGPU + NPU + CPU load to hit PL1/thermal limits, so log throttle reasons during every run.

### Cited Findings
- The kernel driver `dell-pc` (drivers/platform/x86/dell/dell-pc.c, author Lyndon Sanche, merged around Linux 6.11) exposes Dell SMBIOS fan/thermal modes via ACPI `platform_profile`, mapping to quiet / cool / balanced / performance. It only advertises modes the firmware reports, and does nothing if the firmware lacks the thermal commands. — [Phoronix: Dell PC driver for 6.11](https://www.phoronix.com/news/Dell-PC-Driver-For-Linux-6.11); [dell-pc.c source](https://git.zx2c4.com/wireguard-linux/tree/drivers/platform/x86/dell/dell-pc.c); [patch v8](https://patchew.org/linux/20240529174843.13226-1-lsanche@lyndeno.ca/20240529174843.13226-4-lsanche@lyndeno.ca)
- xe GT frequency sysfs on kernel 7.0 lives at `<device>/tile#/gt#/freq0/`:
  - Read-only: `act_freq` (what PCODE resolved), `cur_freq` (what GuC requested), `rpn_freq` (min), `rpe_freq` (efficient), `rpa_freq` (achievable), `rp0_freq` (max).
  - Read-write: `min_freq`, `max_freq`. "If max <= min, then freq_min becomes a fixed frequency request." GuC only requests within [min, max], and PCODE has the final say based on thermals.
  
  — [kernel.org v7.0 Xe GT Frequency Management](https://docs.kernel.org/7.0/gpu/xe/xe_gt_freq.html)
- xe throttle sysfs: `freq0/throttle/status` (0/1) and `freq0/throttle/reasons`, a space-separated list or "none". The docs recommend reading `reasons` rather than polling `status` and then the individual files, which is racy. On non-Crescent-Island platforms the per-reason files are `reason_pl1`, `reason_pl2`, `reason_pl4`, `reason_thermal`, `reason_prochot`, `reason_ratl`, `reason_vr_thermalert` and `reason_vr_tdc`. Some may be hidden per platform. — [kernel.org v7.0 xe_gt_freq](https://docs.kernel.org/7.0/gpu/xe/xe_gt_freq.html)
- NPU turbo: OpenVINO property `NPU_TURBO` (`ov::intel_npu::turbo(true)`) is a hint to keep maximum NPU frequency and memory throughput within platform TDP. The docs warn it may increase compile time and memory footprint, affect workload latency, and cause compatibility issues with older NPU drivers. Nightly docs say it is not recommended for sustained workloads because of higher power. — [OpenVINO 2026 NPU device](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html); [nightly](https://docs.openvino.ai/nightly/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html)
- The NPU kernel driver is `intel_vpu`, exposed as `/dev/accel/accel0`. The user-space stack is intel/linux-npu-driver (Level Zero NPU plugin, compiler, firmware). openSUSE packaged NPU driver 1.35.0 with OpenVINO 2026.3.1 in August 2026, which gives a current-version reference point. — [intel/linux-npu-driver](https://github.com/intel/linux-npu-driver); [openSUSE news 2026-08-31](https://news.opensuse.org/2026/08/31/NPU-openVINO/)
- LinuxLinks also tested the Panther Lake NPU under CachyOS. — [LinuxLinks NPU on Panther Lake](https://www.linuxlinks.com/minisforum-m2-intel-panther-lake-mini-pc-running-linux-npu-cachyos/) (not fetched)
- XPS 16 2026 thermals (Windows): Notebookcheck reports it runs noticeably warmer than the previous generation. HotHardware says it "does throttle under sustained load, but not always" in Cinebench 2026 multi-core. — [Notebookcheck](https://www.notebookcheck.net/2026-Dell-XPS-16-runs-noticeably-warmer-than-last-generation-model.1255741.0.html); [HotHardware review p3](https://hothardware.com/reviews/new-dell-xps-16-da16260-review?page=3)
- Linux status on the DA16260: Dell does not officially support Linux on the 2026 XPS. On Ubuntu 26.04 LTS, Wi-Fi, trackpad and speakers work, and the built-in webcam is not detected (irrelevant here because external UVC cameras are used). After resume, the OLED panel can stick at 20 Hz. — [techtest.org, Dell XPS 14/16 2026 + Linux](https://techtest.org/dell-xps-14-16-2026-linux/)
- An Ubuntu certification entry exists at ubuntu.com/certified/202603-38515. The search surfaced it alongside XPS 16 results, but I did not confirm which model it covers. — [ubuntu.com certified](https://ubuntu.com/certified/202603-38515)

### Commands (standard practice, not re-verified this session)
```bash
# Platform profile (ACPI; the dell-pc driver feeds this). Kernel >= 6.14 also has /sys/class/platform-profile/*
cat /sys/firmware/acpi/platform_profile_choices
echo performance | sudo tee /sys/firmware/acpi/platform_profile
# Or via power-profiles-daemon (Ubuntu default), which also sets EPP:
powerprofilesctl list; powerprofilesctl set performance
#   (ppd may override manual EPP writes. Either use only ppd, or stop it: sudo systemctl mask --now power-profiles-daemon)
# intel_pstate (active mode, HWP):
cat /sys/devices/system/cpu/intel_pstate/status            # expect "active"
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference
# Optionally raise the floor on the control core only:
echo 3000000 | sudo tee /sys/devices/system/cpu/cpu3/cpufreq/scaling_min_freq
# Disable deep C-states on the control core only (cuts wake-up latency without burning the package budget):
for s in /sys/devices/system/cpu/cpu3/cpuidle/state*/disable; do echo 1 | sudo tee $s; done   # keep state0/1 enabled if preferred
# xe GPU frequency (the card index may differ; find it via /sys/class/drm/card*/device/driver -> xe)
G=/sys/class/drm/card0/device/tile0/gt0
cat $G/freq0/{rpn_freq,rpe_freq,rpa_freq,rp0_freq,act_freq}
echo $(cat $G/freq0/rpa_freq) | sudo tee $G/freq0/min_freq   # raise the floor to cut ramp latency
watch -n0.5 "cat $G/freq0/act_freq $G/freq0/throttle/reasons"
# RC6 / GT idle (xe gtidle):
cat $G/gtidle/idle_status $G/gtidle/idle_residency_ms
# Whole-system telemetry while running:
sudo turbostat --quiet --show Busy%,Bzy_MHz,PkgTmp,PkgWatt,GFXWatt,CoreTmp --interval 1
sudo intel_gpu_top     # in recent igt-gpu-tools; or `nvtop`, `qmassa`, xe's fdinfo
cat /sys/class/thermal/thermal_zone*/type; cat /sys/class/thermal/thermal_zone*/temp
# Optional: Intel RAPL PL1/PL2 (powercap) for visibility only:
cat /sys/class/powercap/intel-rapl:0/constraint_*_power_limit_uw
```

### Inferences
- With the GPU floor raised, each action chunk avoids the GuC ramp from idle. Raising `min_freq` to `rpe`/`rpa` (not `rp0`) is a reasonable compromise on a laptop: `rp0` with sustained NPU and CPU load is likely to trigger PL1/thermal throttling, which shows up as `reason_pl1`/`reason_thermal`. Measure p50/p99 chunk latency at both settings.
- A deliberate trade-off between CPU and GPU: the CPU, iGPU and NPU share one package power budget. EPP=performance on all 16 cores can take power away from the iGPU. Consider EPP=performance only on the P-cores used for the control loop and host threads, and `balance_performance` on the E/LP-E cores.
- NPU_TURBO: in this pipeline the NPU (Anomalib) is not the critical path. Leave turbo off unless Anomalib latency matters, because it competes for package power and memory bandwidth with the iGPU running pi0.5.
- AC vs battery: on battery, ppd and firmware will clamp PL1/PL2 and Dell modes may change. Do all evaluation on AC with the profile set to performance, and log `/sys/class/power_supply/AC*/online` alongside results.
- Some Dell firmware may expose only a subset of modes, or `platform_profile` may not appear at all if the BIOS does not report thermal SMBIOS tokens on this new model. Check `platform_profile_choices`. If it is missing, use Dell BIOS setup ("Thermal Management": Ultra Performance, Optimized, Cool, Quiet) as the fallback (not verified for the DA16260 BIOS).

### Gaps
- Not verified that `dell-pc` binds on the DA16260 or which modes it exposes.
- No Linux sustained-load thermal data for the XPS 16 2026 found. Only Windows reviews.
- No source for exact xe GT idle (RC6) sysfs paths on 7.0 (fetched docs covered frequency only). Disabling RC6 is not exposed as a simple sysfs knob in xe, as far as I know. Raising `min_freq` is the practical lever.
- No Linux sysfs control for NPU frequency found beyond the OpenVINO `NPU_TURBO` hint.

---

## 3. Memory-bandwidth contention: LPDDR5X shared by CPU, iGPU and NPU

### Takeaway
I found no published Linux measurements of concurrent iGPU + NPU inference contention on Lunar Lake or Panther Lake. Treat it as a measurement task. pi0.5 decode on the iGPU is likely memory-bandwidth bound, and the NPU and camera DMA/decode traffic share the same LPDDR5X, so keep the NPU job small and duty-cycled, and avoid unnecessary large copies on the CPU.

### Cited Findings
- `NPU_TURBO` explicitly raises "memory throughput" for the NPU within TDP. Turning it on therefore increases NPU demand on the shared memory system. — [OpenVINO 2026 NPU device](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html)
- xe throttle reasons include package PL1/PL2/PL4 and thermal, which shows the iGPU frequency is governed by package-wide power limits shared with the CPU and NPU. — [kernel.org v7.0 xe_gt_freq](https://docs.kernel.org/7.0/gpu/xe/xe_gt_freq.html)

### Inferences (measurement plan)
- Measure pi0.5 chunk latency (a) alone, (b) with Anomalib looping on the NPU, (c) with the cameras running, and (d) with all three. Log `act_freq`, `throttle/reasons`, and `turbostat` PkgWatt/GFXWatt in every run. Read memory bandwidth with `intel_gpu_top` (IMC read/write counters where supported) or `perf stat -e uncore_imc*/...` if exposed on Panther Lake (unverified).
- Mitigations: run the NPU detector at a lower rate than the camera (for example every 3rd frame) and not synchronised with VLA chunk start. Keep inputs at the native 640x480 and resize once. Use OpenVINO remote tensors or zero-copy where possible (out of scope here: model export).

### Gaps
- No Lunar Lake or Panther Lake measurements of GPU+NPU concurrency or memory-bandwidth interference found in the time available. The report should say this explicitly.
- Theoretical LPDDR5X bandwidth for the 388H configuration (speed grade in the DA16260) was not verified.

---

## 4. Camera pipeline: format, USB bandwidth, decode, buffering, timestamps, exposure

### Takeaway
Use MJPEG for the two UVC webcams and put the RealSense on its own USB 3 root port, ideally its own controller. Three YUYV 640x480@30 streams (about 147 Mbit/s each) on one USB 2 hub will fail with ENOSPC/-28. At 640x480, CPU MJPEG decode via libjpeg-turbo in OpenCV costs about a millisecond or two per frame, so VA-API hardware JPEG decode is not worth the complexity. Use one capture thread per camera that always keeps only the newest frame, timestamp with the V4L2 buffer timestamp or `time.monotonic_ns()` at grab, and lock exposure and white balance.

### Cited Findings
- `VIDIOC_STREAMON` failing with "No space left on device" / `uvcvideo: Failed to submit URB 0 (-28)` is a USB bandwidth reservation failure from the host controller, not a disk issue. — [The Good Penguin](https://www.thegoodpenguin.co.uk/?p=251); [Arch forum](https://bbs.archlinux.org/viewtopic.php?id=278537)
- 640x480 @ 30 fps uncompressed at 24 bpp is about 221 Mbit/s (YUYV at 16 bpp is about 147 Mbit/s, by the same arithmetic), so uncompressed webcams quickly exhaust USB 2.0. — [StackOverflow answer (archived)](https://kaweah.freedombox.rocks/kiwix/content/stackoverflow.com_en_all_2023-11/a/35161718)
- Running several UVC cameras on one USB root requires MJPEG mode rather than YUV. — [OpenPnP group thread](https://groups.google.com/g/openpnp/c/gSSFehc1cwI)
- `uvcvideo quirks=128` (UVC_QUIRK_FIX_BANDWIDTH) makes the driver compute bandwidth instead of trusting the camera's maximum request. Results are mixed. The more reliable fix is to spread cameras across separate root hubs or controllers. — [Arch forum](https://bbs.archlinux.org/viewtopic.php?id=278537); [Launchpad bug 1014749](https://bugs.launchpad.net/bugs/1014749)

### Commands (standard practice, not re-verified this session)
```bash
lsusb -t                                  # check speed (480M vs 5000M) and which root hub each camera sits on
v4l2-ctl --list-devices
v4l2-ctl -d /dev/videoX --list-formats-ext   # confirm MJPG 640x480@30 on the UVC webcams
# Fix exposure, white balance and power-line frequency (control names vary by camera; list them with -L):
v4l2-ctl -d /dev/videoX -L
v4l2-ctl -d /dev/videoX -c auto_exposure=1 -c exposure_time_absolute=150 -c white_balance_automatic=0 \
         -c white_balance_temperature=4500 -c power_line_frequency=1 -c focus_automatic_continuous=0
# Last resort if bandwidth errors persist:
echo 'options uvcvideo quirks=128' | sudo tee /etc/modprobe.d/uvcvideo.conf && sudo modprobe -r uvcvideo && sudo modprobe uvcvideo
# Stop USB autosuspend from adding resume latency on the cameras and the servo adapter:
echo -1 | sudo tee /sys/module/usbcore/parameters/autosuspend   # or a per-device power/control=on udev rule
# Stable device names: udev rules on ID_SERIAL / ID_PATH -> /dev/cam_wrist, /dev/cam_top, /dev/so101
```
OpenCV (standard practice):
```python
cap = cv2.VideoCapture("/dev/cam_wrist", cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # set FOURCC before size and fps
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480); cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)        # V4L2 backend: fewer queued buffers, less stale latency
# One thread per camera: loop { ok = cap.grab(); t = time.monotonic_ns(); ok, f = cap.retrieve(); latest[cam] = (t, f) }
# grab() and retrieve() release the GIL during the blocking ioctl and the JPEG decode.
```

### Inferences
- RealSense D415 colour over V4L2 is uncompressed (YUYV, no MJPEG on that sensor as far as I know), about 147 Mbit/s at 640x480@30. It must sit on a USB 3 port to enumerate at 5 Gbit/s. A D415 on USB 2 is limited to reduced modes. Using pyrealsense2 instead of V4L2 gives hardware/global timestamps (`frame.get_frame_metadata(rs.frame_metadata_value.sensor_timestamp)` / `get_timestamp()` with global time enabled), which are better for cross-camera alignment.
- The "Pro QHD 2K/60fps" webcam must be forced to 640x480 MJPEG. Otherwise OpenCV can negotiate a high-bandwidth default. If its 640x480 mode is YUYV-only, the shared hub will overflow.
- Synchronisation: these UVC cameras are free-running with no hardware sync. Assemble an observation from the latest frame of each camera, stamped at `grab()`, and record the per-camera age (`now - t`). Discard or flag observations where the spread exceeds about one frame period (33 ms). Note that V4L2 buffer timestamps (CLOCK_MONOTONIC at the start or end of frame, depending on the driver) can be read via `v4l2-ctl --stream-mmap -d X --verbose`. OpenCV does not expose them directly except via `CAP_PROP_POS_MSEC` (backend-dependent).
- Exposure: auto-exposure lengthens exposure in dim light, which drops the effective frame rate to 15 fps on many UVC cameras and adds latency. Fixed manual exposure ≤ 16 ms keeps 30 fps steady.
- VA-API / oneVPL JPEG decode on Arc exists via the iHD media driver, but at 640x480 the upload, sync and download overhead likely exceeds the libjpeg-turbo CPU cost. Pin decode threads to E-cores instead.
- When using MJPEG through a hub, decode cost runs on the capture threads. Pin them away from the servo core.

### Gaps
- No primary source verified for the exact D415 V4L2 colour formats on current librealsense. Check with `v4l2-ctl --list-formats-ext`.
- No measured libjpeg-turbo vs VA-API decode times at 640x480 on Panther Lake.
- Whether the XPS 16 DA16260's USB-C/Thunderbolt ports are on separate xHCI controllers: not found. Check with `lsusb -t`.

---

## 5. Python concurrency design: threads vs processes, OpenVINO async, shared memory, GC, OpenVINO CPU threads

### Takeaway
Use separate processes for (1) the servo control loop (SCHED_FIFO, pinned P-core) and (2) perception and inference (cameras plus OpenVINO GPU/NPU via async APIs), connected via `multiprocessing.shared_memory` ring buffers or a small lock-free action buffer. Inside the inference process, threads are fine because OpenCV capture/decode and OpenVINO inference release the GIL. Limit OpenVINO CPU threads and use its core-type controls so it cannot spill onto the control core.

### Cited Findings
- OpenVINO CPU plugin properties: `ov::hint::scheduling_core_type` (`ANY_CORE` / `PCORE_ONLY` / `ECORE_ONLY`) selects core types on hybrid CPUs. `ov::hint::enable_hyper_threading` sets 1 vs 2 logical processors per core. `ov::inference_num_threads` caps threads. `ov::hint::enable_cpu_pinning` is enabled by default on Linux, disabled on Windows and macOS. scheduling_core_type and enable_hyper_threading are supported only on Intel x86-64 on Linux and Windows. — [OpenVINO 2025: Performance Hints and Thread Scheduling](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device/performance-hint-and-thread-scheduling.html) (via search snippet; full page did not render on fetch); [2024 version](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device/performance-hint-and-thread-scheduling.html)
- LeRobot has an official async inference design ("Asynchronous Robot Inference: Decoupling Action Prediction and Execution"). The robot client executes the current action chunk while the policy server computes the next one. — [Hugging Face blog: async robot inference](https://huggingface.co/blog/async-robot-inference) (not fetched in detail)

### Recommended pattern (standard practice, not re-verified this session)
```python
import openvino as ov, gc, os
core = ov.Core()
# Keep OpenVINO's CPU-side work off the control and capture cores:
core.set_property("CPU", {"INFERENCE_NUM_THREADS": 4, "ENABLE_CPU_PINNING": False,   # let the process affinity mask decide
                          "SCHEDULING_CORE_TYPE": "ECORE_ONLY"})                    # only if the CPU device is actually used
os.sched_setaffinity(0, set(range(0, 3)) | set(range(4, 12)))   # inference process: never CPU 3
gpu = core.compile_model(model, "GPU", {"PERFORMANCE_HINT": "LATENCY"})
npu = core.compile_model(det, "NPU", {"PERFORMANCE_HINT": "LATENCY"})
# Async: req.start_async(inputs); ...; req.wait()  — or ov.AsyncInferQueue(npu, 2) with a callback.
# The GIL is released during infer and wait, so camera threads keep running.
gc.collect(); gc.freeze()          # after model load: move long-lived objects out of GC generations
gc.disable()                       # in the control process; call gc.collect() in idle slack each cycle, or raise thresholds:
gc.set_threshold(100_000, 50, 50)
```
- Control process loop: an absolute-deadline loop using `time.clock_nanosleep(time.CLOCK_MONOTONIC, ...)` is not available in Python. Use `time.sleep()` to about 1 ms before the deadline, then a short spin, or `select` on timerfd via `os.timerfd_create` (Python 3.13+, not available in 3.12). Read the action chunk from shared memory, interpolate, sync_write, and sync_read every N cycles.
- Shared memory: `multiprocessing.shared_memory.SharedMemory` plus a numpy view, with a sequence counter (seqlock) per slot. Avoid `multiprocessing.Queue` pickling of 640x480x3 frames (about 0.9 MB each) on the hot path.
- Pre-allocate numpy buffers and avoid per-cycle allocations in the control loop (this also reduces GC pressure). Run `mlockall` via `ctypes` (`libc.mlockall(3)`) in the control process to avoid page faults (needs a memlock limit).

### Inferences
- Python 3.12 has no free-threaded build (that is 3.13t+). Processes remain the only way to isolate the control loop from GIL contention.
- If `ENABLE_CPU_PINNING=True` (the Linux default), OpenVINO pins its CPU streams itself. My understanding is that it respects the process affinity mask, but it may still choose cores that clash with capture threads. Disabling pinning and using an explicit process affinity is the safer combination. Verify with `ps -L -o tid,psr,comm -p <pid>`.
- For GPU and NPU devices, OpenVINO's host-side CPU threads are mainly submission and waiting. `INFERENCE_NUM_THREADS` matters mostly when a model or subgraph falls back to CPU (for example tokeniser or pre/post-processing ops), so check with `ov::hint` / `exec_graph` which device each node runs on.

### Gaps
- Full text of the OpenVINO 2025/2026 thread-scheduling page could not be retrieved (page body did not render). In particular, which cores the LATENCY hint uses by default on hybrid parts, and how LP E-cores are treated, are unverified.
- No benchmark data found on Python GC pause magnitudes in this kind of loop.

---

## 6. Feetech / LeRobot serial latency

### Takeaway
At 1 Mbps the wire time for a sync_read of 6 STS3215 positions is under about 1 ms. Real cost is dominated by the USB-serial adapter's latency and the servo's Return Delay Time. If the adapter is FTDI, set `latency_timer` to 1 ms (default 16 ms). If it is CH340/CH343 (common on SO-101 Waveshare-style boards), the latency_timer knob does not exist under `cdc_acm`/`ch341`. Use `setserial low_latency` where supported, and make sure Return_Delay_Time is 0. Budget at 60 Hz is 16.7 ms per cycle, so sync_write every cycle and sync_read every cycle are both affordable once the adapter latency is low.

### Cited Findings
- Robotis Dynamixel SDK documents a default USB latency of 16 ms and tells users to lower `LATENCY_TIMER` for faster communication. The Feetech SDK lineage (scservo_sdk) is derived from the Dynamixel SDK. — [Robotis Dynamixel SDK FAQ](https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_sdk/faq/)
- A Feetech JS SDK port uses the same 16 ms latency default and adds 2× latency into its packet timeout. — [feetech.js lowLevelSDK.mjs (HF space)](https://huggingface.co/spaces/blanchon/RobotHub-Frontend/blob/31657459874ab25f81442bcb5a0260da955c3f8a/packages/feetech.js/lowLevelSDK.mjs)
- The FTDI latency timer is configurable between 2 and 255 ms through the D2XX API (the Linux `ftdi_sio` sysfs `latency_timer` accepts 1). — [FTDI FT_SetLatencyTimer](https://www.ftdichip.com/Support/Knowledgebase/ft_setlatencytimer.htm)
- Feetech SYNC WRITE requires identical data length and start address for all servos. — [Feetech SCS protocol notes](https://aifitlab-wiki.super.site/feetech-servo-motor-docs/feetech-scs-protocol)
- LeRobot recommends a 1,000,000 baud rate for Feetech motors. — [LeRobot Feetech docs](https://huggingface.co/docs/lerobot/feetech)
- In current LeRobot main, `FeetechMotorsBus` is a deprecated alias of a generic `SerialMotorsBus` that finds each motor's family from its model, and `protocol_version` is ignored. Code and register defaults moved, so older blog posts about `feetech.py` may be stale. — [LeRobot feetech.py (main)](https://raw.githubusercontent.com/huggingface/lerobot/main/src/lerobot/motors/feetech/feetech.py)
- The CH343 is CDC-compliant and works with the in-kernel `cdc_acm` driver (`/dev/ttyACM*`). WCH also ships an out-of-tree `ch343ser` driver. — [Waveshare CH343 wiki](https://waveshare.com/wiki/CH343_USB_UART_Board); [ch343ser_linux](https://gitee.com/lyjunnan/ch343ser_linux)

### Commands (standard practice, not re-verified this session)
```bash
# Identify the adapter chip:
lsusb | grep -Ei 'ftdi|1a86|wch|qinheng'; udevadm info -q property -n /dev/ttyACM0 | grep -E 'ID_VENDOR|DRIVER'
# FTDI (ftdi_sio) only:
cat /sys/bus/usb-serial/devices/ttyUSB0/latency_timer; echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
# persistent: /etc/udev/rules.d/99-ftdi-lowlat.rules
#   ACTION=="add", SUBSYSTEM=="usb-serial", DRIVER=="ftdi_sio", ATTR{latency_timer}="1"
# Generic ASYNC_LOW_LATENCY flag (supported by some drivers):
sudo setserial /dev/ttyUSB0 low_latency; setserial -g /dev/ttyUSB0
# Disable USB autosuspend for the adapter (ATTR{power/control}="on" in udev).
```
Timing budget (inference from protocol arithmetic, 1 Mbps, 8N1 = 10 µs/byte):
- sync_write goal-position to 6 servos: header + 6×(ID + 2 bytes) ≈ 26 bytes ≈ 0.26 ms wire, no response.
- sync_read present-position for 6 servos: request ≈ 14 bytes, plus 6 responses × 8 bytes = 48 bytes, ≈ 0.6 ms wire, plus 6 × Return_Delay_Time (STS default is reported as about 500 µs, giving about 3 ms, if not set to 0), plus the USB round trip (FTDI at default 16 ms latency timer can add up to 16 ms; about 1 ms at latency_timer=1; CDC-ACM full-speed about 1 ms frame granularity).
- Measure directly: time `bus.sync_read("Present_Position")` and `bus.sync_write("Goal_Position", ...)` over 1000 iterations and report p50/p99. Target p99 < 3 ms read and < 1 ms write.

### Inferences
- If p99 sync_read is about 16 ms, it is almost certainly an FTDI latency timer at the default. If it is about 3-4 ms on CDC-ACM, check Return_Delay_Time.
- At 60 Hz, consider writing every cycle and reading every 2nd cycle if the read budget is tight. Position feedback for logging does not need to be 60 Hz.
- Retries: LeRobot's sync_read retries on comm errors. Noise on a long bus or under-powered servos shows up as p99 spikes. Count retries and failures.

### Gaps
- Could not verify the exact USB-serial chip on the user's SO-101 controller board (Waveshare bus servo adapter variants use CH343 or CH340; others use FTDI). Check with `lsusb`.
- Could not retrieve LeRobot's current `SerialMotorsBus` source to confirm that it sets Return_Delay_Time=0 and what timeout and retry defaults it uses.
- No LeRobot issue with measured SO-101 sync_read latencies was found in this session.
- Whether `cdc_acm` honours ASYNC_LOW_LATENCY: not confirmed.

---

## 7. Level Zero / compute-runtime env vars on Xe3 + kernel 7.0; Ubuntu 26.04 specifics

### Takeaway
Immediate command lists are now the default path for newer Intel GPUs in the oneAPI Level Zero stack. The L0 V2 adapter is the default on Arc B-series and Lunar Lake from oneAPI 2025.3 and only supports immediate command lists. The old advice to export immediate-command-list env vars is mostly stale for Xe2/Xe3. OpenVINO's GPU plugin uses OpenCL/L0 via its own paths, so these env vars rarely matter. Prefer up-to-date compute-runtime (NEO), IGC and NPU driver packages over env-var tweaks.

### Cited Findings
- The Level Zero V2 adapter became the default with the 2025.3 release for newer platforms including Arc B-series (Xe2) and Core Ultra 200V (Lunar Lake), and it only supports immediate command lists. Overrides: `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0` (SYCL) and `LIBOMPTARGET_LEVEL_ZERO_USE_IMMEDIATE_COMMAND_LIST=0` (OpenMP). — [Intel: Level Zero Immediate Command Lists](https://www.intel.com/content/www/us/en/developer/articles/guide/level-zero-immediate-command-lists.html); [oneAPI GPU optimization guide 2025-2](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/immediate-command-lists.html)
- The Unified Runtime var is `UR_L0_USE_IMMEDIATE_COMMANDLISTS` (0 = off, 1 = per queue, 2 = per thread per queue). — [UR Level Zero reference](https://oneapi-src.github.io/unified-runtime/core/LEVEL_ZERO.html)
- NEO debug env vars are ignored on Linux release builds unless `NEOReadDebugKeys=1` is set. The full list is in `debug_variables_base.inl`. — [compute-runtime FAQ (Debian 26.05 package)](https://sources.debian.org/src/intel-compute-runtime/26.05.37020.3-1/FAQ.md)
- Ubuntu 26.04 LTS works on the DA16260 apart from the internal webcam. Dell does not officially support Linux on it. — [techtest.org](https://techtest.org/dell-xps-14-16-2026-linux/)

### Inferences
- Do not set NEO debug keys in production. If experimenting, `NEOReadDebugKeys=1` plus specific keys (for example `EnableDirectSubmission=1`) is required before any key takes effect. Direct submission defaults vary by platform, so verify any gain with p99 chunk latency.
- On a new platform like Panther Lake (Xe3), freshness of compute-runtime, IGC, the NPU driver and OpenVINO matters more than env vars. Use Intel's PPA or GitHub releases if Ubuntu 26.04's archive versions lag. Check with `clinfo | grep -i driver`, `python -c "import openvino as ov; print(ov.Core().available_devices)"`, and `ls /dev/dri /dev/accel`.
- Ensure the user is in the `render` and `video` groups (GPU/NPU access) and `dialout` (serial).

### Gaps
- No source found listing env vars specifically recommended for Xe3 (Panther Lake) with OpenVINO GPU plugin on kernel 7.0.
- No Ubuntu 26.04 package versions for intel-opencl-icd, level-zero or intel-npu-driver were verified.
- No source on xe driver behaviour differences specific to Xe3 on 7.0 vs later kernels (for example any Panther Lake-specific xe fixes in 7.1-7.3).
