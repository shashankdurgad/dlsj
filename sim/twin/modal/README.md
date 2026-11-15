# The Isaac twin on Modal GPUs (x86_64)

Smoke test and throughput measurement (2026-11-14, workspace `desmond-zee`): the Isaac Lab 3.0 twin
(`sim/twin`, built for the GB10 in Docker) runs unchanged on Modal's RTX-PRO-6000, with RTX camera rendering, the
twin env and the MolmoAct2 prior. No training here; the numbers below size the DICE-RL run.

## Build and run
```
sim/twin/modal/prepare.sh        # Isaac Lab v3.0.0 checkout -> sim/twin/.build/IsaacLab (same ref / place as build.sh)
export MODAL_PROFILE=desmond-zee
M="uv run --no-project --with modal python -m modal run sim/twin/modal/app.py"
$M --task diag        --gpu RTX-PRO-6000                     # driver, mounted graphics libs, vulkaninfo
$M --task render      --gpu RTX-PRO-6000                     # smoke.py + check_render_motion.py, with / without GB10 KIT_ARGS
$M --task bench-env   --gpu RTX-PRO-6000:2 --procs-per-gpu 3 # TwinEnv steps/s, N Isaac processes in parallel
$M --task fetch-molmo                                        # MolmoAct2-SO100_101 -> Volume dlsj-twin-hf (CPU only)
$M --task bench-molmo --gpu RTX-PRO-6000                     # MolmoAct2 latency per chunk, batch 1 / 8
```
(in zsh call the command directly instead of through `$M`). Results land in `sim/twin/out/modal/<task>_<gpu>.json`
(+ PNGs); every function has a hard `timeout=`, `modal run` apps stop when the entrypoint returns.

* **Image** (`app.py`): Isaac Lab v3.0.0's own `docker/Dockerfile.base` on `nvcr.io/nvidia/isaac-sim:6.2.0` (amd64,
  anonymous NGC pull) via `modal.Image.from_dockerfile`, then the twin `docker/Dockerfile`'s Studio layer (Physical
  AI Studio `8f132416` installed against Isaac Lab's torch / numpy / warp pins), the pinned SO-101 URDF
  (`fetch_so101.sh`), and `sim/` + `deploy/` added at container start (code edits do not rebuild). Cold build
  ~9 min (base layer 503 s), cached afterwards. Three mechanical Dockerfile rewrites Modal needs
  (`_modal_dockerfile`, same result): Modal rejects `COPY --parents`, so it becomes one `COPY` per
  `source/*/pyproject.toml`; `../` COPY prefixes become context-relative; the BuildKit uv cache mount is dropped.
  Modal skips `USER isaaclab`, so everything runs as root; `ENTRYPOINT []` replaces Isaac Sim's `runheadless.sh`.
  Isaac runs through `isaaclab.sh -p` in a subprocess of Modal's own Python (`add_python="3.12"`), as `run.sh` does.
* **Telemetry off**: `PRIVACY_CONSENT=N` (image + every Isaac subprocess) and an opt-out `privacy.toml` were not
  enough: Kit still spawned `omni.telemetry.transmitter` with `--/telemetry/enableAnonymousData=true` (seen once, in
  the first smoke run, and killed). The image now replaces that binary with a no-op (`/opt/telemetry-disabled.txt`
  lists it); every Isaac run checks for (and would kill) a transmitter: none since.
* **Volumes**: `dlsj-twin-cache` (Kit / RTX shader / compute caches as one tar per GPU type, plus the URDF -> USD
  conversion), `dlsj-twin-hf` (HF_HOME, MolmoAct2 21 GB, 65 s download in a CPU container).
* **CPU**: Modal's default is ~1 core (`nproc` = 1); Isaac functions request 8 cores, `bench-env` 6 per process.
* **W&B**: training functions take `secrets=TRAIN_SECRETS` (`modal.Secret.from_name("dlsj-wandb")`, exists since
  2026-11-15); fan-out jobs get it with `--secret dlsj-wandb`.

## Fan-out: many short twin jobs on many GPUs (`fanout.py`)
For sweeps / render batches that would otherwise queue on watson / jarvis. Use it when parallel GPUs really save
time (a sweep of independent runs); quick single checks stay on the GB10. Run from any checkout that has
`sim/twin/modal/` (it runs `prepare.sh` itself if `sim/twin/.build/IsaacLab` is missing), on the Mac (Modal + brev ssh):
```
F="uv run --no-project --with modal --with pyyaml python sim/twin/modal/fanout.py"   # MODAL_PROFILE=desmond-zee (forced)
$F run --dry-run --cmd "sim/twin/run.sh sim/twin/grasp_gate/gate.py --rung 3 --n 60 --set gripper.finray.effort_nm={0.4,1.2} --seed {0,1}"
$F run --name effort-sweep --cmd "...same..."        # 4 jobs, results -> watson:sim/twin/out/modal_fanout/effort-sweep/
$F run --cmd "sim/twin/run.sh X.py --a {a} --b {1,2}" --grid a=0.1,0.2   # {a,b} inline lists and --grid key=... -> {key}
$F run --jobs jobs.yaml       # [{name, cmd, gpu?, timeout?, cpu?, memory?, env?}, ...] (YAML or JSON)
$F run ... --secret dlsj-wandb --env WANDB_MODE=online   # secrets / env for every job
$F data                       # (re)check the data Volumes; `run` does this too
$F stop                       # stop any fan-out app still running (`run` does this at its end)
```
* **What it does**: (1) snapshots the box's live working tree (`--box watson` default, or `jarvis`): `git ls-files -co
  --exclude-standard` of `sim/ dlsj/ deploy/` (so git-ignored data, `out/`, `.build/`, `assets/so101` never go),
  files > `--max-file-mb 20` dropped, plus small inputs from `sim/twin/out` (json / npz / npy / pt / ... < 2 MB,
  `out/scene`, `out/visual_fit/wood`, `out/visuals/top_median.png`; add more with `--include sim/twin/out/...`).
  ~110 MB gz today, streamed through the Mac's scratchpad (deleted after the upload) into Volume `dlsj-twin-fanout`,
  and kept on the box as `modal_fanout/<run>/code.tar.gz` (rerun the exact code on a GB10 from it). (2) checks the data
  Volumes and fetches anything missing. (3) one Modal call + container per job (`single_use_containers`), default
  `RTX-PRO-6000`, 8 cores, 32 GiB, `--timeout 1800` s per job (+ 300 s for prep / collection as the Modal timeout).
  The repo is extracted at `/tmp/repo` (same layout as the box), `sim/twin/data/<x>` symlinks to the data Volume,
  `sim/twin/data/hf` = `HF_HOME` = Volume `dlsj-twin-hf` (offline), Kit shader cache restored from `dlsj-twin-cache`,
  GB10 Kit flags as default (`DLSJ_KIT_ARGS` unset; pass `--env DLSJ_KIT_ARGS=` to change). `sim/twin/run.sh X.py
  args` / `X.py args` -> `isaaclab.sh -p X.py args`; `python ...` -> the Isaac venv; anything else -> bash.
  (4) every file the job wrote under `sim/twin/out` (newer than its start), its `stdout.log` and `job.json` go to
  `dlsj-twin-fanout:/runs/<run>/<job>/` and then to `<box>:~/Documents/dlsj/sim/twin/out/modal_fanout/<run>/<job>/`
  (`out/<path under sim/twin/out>`), with `summary.txt` / `summary.json`; the summary table (job, exit code, command
  wall, container time, key lines: `GATE_PASS ...`, `REPLAY {...}`, `*_OK`, tracebacks; `--key REGEX`) is printed.
* **Spend guard**: at most `--max-containers 4` at once (5-8 need `--wide`, never more than 8); the cost estimate is
  always printed, and a run whose upper bound (every job to its timeout, list prices) is above **$15** is refused
  without `--yes-cost`; `--dry-run` prints jobs, estimate and snapshot size, launches nothing. Each run appends
  `utc_start,run,box,jobs,jobs_ok,gpu,max_containers,wall_min,gpu_min,est_cost_usd,user` to
  `<box>:sim/twin/out/modal_fanout/usage.csv` (container time measured in the job x list price; excludes the few
  seconds of image load).
* **Stopping**: the app is ephemeral (`app.run()`): it stops when the run ends, on errors and on Ctrl-C / SIGINT /
  SIGTERM (also when started in the background); afterwards every `dlsj-twin-fanout` app still listed as running is
  stopped with `modal app stop -y` and polled until none is. Tested: SIGINT and SIGTERM mid-job -> containers gone
  within ~30 s, `modal app list` clean.
* **Data Volumes** (`$F data`, HF downloads inside Modal on CPU containers, token from secret `hf-akoniti`, never
  printed): `dlsj-twin-data` = `canonical/duplo_teleop_cal2_rgb_456` (`akoniti/dlsj-canonical`, 250 files 1.8 GB,
  35 s), `posttrain_v3_1` (`akoniti/dlsj-posttrain-v3-1`, 2.4 GB, 43 s), `models/molmoact2_ft_8k`
  (`akoniti/dlsj-molmoact2-ft-8k`, 22.4 GB, 116 s), `photos` (from the box's `sim/twin/data/photos`, private Volume
  only), `MANIFEST.json`; `dlsj-twin-hf` += `allenai/MolmoAct2@e432d85f` (101 s) next to
  `allenai/MolmoAct2-SO100_101@152569fe`. Generated textures live in the code tree (`sim/twin/visuals/textures`) or
  the small `sim/twin/out` inputs above, so they travel with the snapshot. Other box data (e.g. `real_rollouts`) is
  not mirrored; add it to `BOX_DATA` in `fanout.py` if a job needs it.
* **Caveats**: the snapshot is the box's *live* tree, so a half-edited file from another agent goes too (the first
  validation run hit a `NameError` in `grasp_gate/metrics.py` mid-edit; rerun after the edit). Outputs a job writes
  outside `sim/twin/out` are not collected. Each job is a fresh Kit process (~10-20 s startup with the cached shaders).

**Validation (2026-11-15, watson snapshots, RTX-PRO-6000)**:
| run | jobs | wall (whole run) | per job: command / container | GPU-min | est. cost |
|---|---|---|---|---|---|
| `validate-gate-effort-2`: gate rung 3, n 60, effort {0.4, 1.2} x seed {0, 1} | 4 | 6.0 min (first, slow app start) | 24-29 s / 27-38 s | 2.1 | $0.13 |
| `consistency-1`: gate e1.2 s0 twice, e0.4 s0, `validate_twin.py --only replay --episodes 12` | 4 | **1.8 min** (snapshot 14 s, upload 13 s, app 19 s, jobs 38-56 s, download + copy back 21 s) | 28 s (replay 48 s) / 33 s (52 s) | 2.5 | $0.15 |

All tests incl. the signal tests came to ~$1.3 of GPU time (`usage.csv`). Consistency against a GB10 on the *same*
snapshot (`consistency-1/code.tar.gz` extracted on watson, `run.sh`, docker): gate e0.4 s0 **50.0 % on both**, the
same 60/60 per-episode success flags (one failure labelled tipped on the GB10, pushed on Modal); e1.2 s0 50.0 % GB10
vs 48.3 % Modal (one episode of 60 flips), joint RMSE equal to 0.1 deg. Modal is deterministic run to run (the two
e1.2 s0 jobs: identical 48.3 %). GB10 wall for one gate job: 1 min 55 s - 2 min 3 s through `run.sh` on a busy box;
4 jobs on Modal in parallel ~1 min. So: x86 RTX-PRO-6000 results match the GB10 to within contact-sim noise
(about one episode in 60); compare sweep arms within one platform.

## What works, what doesn't
| | RTX-PRO-6000 (Blackwell server, driver 610.57.04, CUDA 13.3 UMD) |
|---|---|
| Vulkan / RTX in Modal's gVisor container | **works out of the box**: `NVIDIA_DRIVER_CAPABILITIES=all` is already set, graphics userspace libs (`libGLX_nvidia`, `libnvidia-rtcore`, `-glcore`, `-eglcore`) are mounted at the driver version, `vulkaninfo` lists the GPU (Vulkan 1.4.341). No extra libs needed. |
| `smoke.py` (both variants) | `SMOKE_OK`; top / wrist render (means 197 / 206), isaaclab 28.0.0 (v3.0.0), physicalai-train 0.3.0, torch 2.11 cu130 |
| `check_render_motion.py`, Isaac Lab default (Fabric) render path, `DLSJ_KIT_ARGS=""` | **`RENDER_SEES_MOTION`** (13.0 % of top pixels change between poses): on x86 the plain Isaac Lab path follows physics, the GB10 bug does not occur |
| `check_render_motion.py` with the GB10 KIT_ARGS | `RENDER_FROZEN` (0.2 %), as on the GB10: with the Fabric delegate off the renderer never sees PhysX poses |
| `TwinEnv` as on the branch (GB10 KIT_ARGS + `VisualSync`) | all three cameras follow physics (start / moved frames checked) |
| `TwinEnv` with `DLSJ_KIT_ARGS=""` | top and inspect move, **wrist camera frozen** (image change 0.0): `VisualSync`'s USD writes to the wrist camera are not picked up by the Fabric path. Keep the GB10 KIT_ARGS for `TwinEnv` on x86 too (no speed gain from dropping them: 14.7 vs 14.3 steps/s). |
| MolmoAct2 worker (`MolmoAct2Client`) | works; bf16, 10 flow steps, 30-action chunks, finite |
| L40S fallback | not needed, not tested |
| B200 (MolmoAct2) | not measured: two requests (2026-11-14 23:37 UTC, 17 min; 23:55 UTC, 12 min) never got a container (no B200 capacity for this workspace at the time) and were stopped; nothing was billed. Retry with `--gpu B200` later; `B200+` may land on a B300, which needs CUDA 13.1+ (the image's torch is cu130) |

`sim/residual_env/app.py` gained a backwards-compatible `DLSJ_KIT_ARGS` override (unset = the GB10 flags) for the
check above. Kit cold start without the shader cache: 100 s to the first frame; with the cached tar: 13-18 s.

## Measured (RTX-PRO-6000, 640x480 x 3 cameras, one `TwinEnv` station per Isaac process, 300 timed ticks)
| layout | env ticks/s (total) | per process | ms / tick p50 (p95) | GPU util | GPU mem / GPU |
|---|---|---|---|---|---|
| 1 GPU, 1 process | **14.3** | 14.3 | 65.6 (82) | ~51 % | 6.4 GB |
| 1 GPU, 4 processes | **24.0** | 5.8-6.4 | 158-166 (218-228) | 97 % | 25.6 GB |
| 2 GPUs, 3 processes each | **40.3** (20.1 / GPU) | 6.1-8.2 | 120-141 (238-357) | 88-89 % | 19.3 GB |

* one tick = `TwinEnv.step` at 30 Hz: 4 PhysX substeps (120 Hz) + servo model + `VisualSync` + 3 RTX cameras + colour
  correction + GPU -> CPU copies. Without rendering: 30.3 ticks/s for 1 process (so ~33 ms physics / tick); render
  alone 35.5 ms (1 process), 37-49 ms with 4-6 processes. `reset()` 0.16-0.7 s.
* a GPU saturates at ~3-4 processes (~20-24 ticks/s per RTX-PRO-6000); per-process speed drops as they share the GPU.
* startup: 18-21 s for one env; 100-180 s when 4-6 processes start together (stagger them, or keep envs alive).

**MolmoAct2** (`allenai/MolmoAct2-SO100_101`, bf16, 10 flow steps, wrist + top 480x640, 20 timed calls after 3 warm-up):
| GPU | batch 1 (client, worker time) | batch 1 roundtrip / with RTC | batch 8 (in-process) | load | peak mem |
|---|---|---|---|---|---|
| RTX-PRO-6000 | **0.232 s** (p95 0.243) | 0.234 / 0.233 s | **0.409 s** = 51 ms per chunk | 37 s (Volume) | 14.4 GB |
| GB10 (for reference) | 0.37 s | | | | |
| B200 | not measured (no capacity in two attempts) | | | | |

The client has no batched call; batch 8 loads the worker's model (`_Worker`) in-process and feeds a stacked batch
through the same pre-processor -> `predict_action_chunk` -> post-processor (output (8, 30, 6)).

## Recommended training layout
Throughput planning numbers: **~20-24 env ticks/s per RTX-PRO-6000 with 3-4 twin processes**, and the prior at
**4.3 chunks/s (batch 1) or 19.6 chunk-samples/s (batch 8) per RTX-PRO-6000**. With RTC re-planning every 15 ticks,
each env needs one prior call per 15 ticks; DICE-style K-seed sampling multiplies the samples per call by K.

**A. One container, `RTX-PRO-6000:8`** (default): GPU 0 = MolmoAct2 prior (batched across envs) + learner;
GPUs 1-7 = 4 `TwinEnv` processes each (28 envs, ~7 x 24 = **~170 ticks/s**, extrapolated from the 1- and 2-GPU runs).
Simple local IPC (the client already talks over pipes), one place for the replay buffer. Prior demand at 170 ticks/s:
11 calls/s; with K = 1 that is ~0.6 of GPU 0; with K = 8 it is ~90 samples/s, ~4.6 GPUs at batch 8, so move GPUs
from envs to the prior (e.g. 4 env GPUs at ~96 ticks/s + 4 prior / learner GPUs at ~78 samples/s), or measure larger
batches first (only 1 and 8 were measured). Caveats: Modal's GPU guide says more than 2 GPUs per container usually
means longer waits, and it does not state a maximum count for RTX-PRO-6000 (2 worked here; 8 is untested). The CPU
per Isaac process was set to 6 cores and not profiled lower; 28 processes at 6 cores would exceed a single
container, so profile CPU before the run.

**B. Split containers**: 7 x `RTX-PRO-6000:1` env containers (4 envs, ~24 cores each) + 1 prior / learner container,
talking over Modal networking or a `modal.Queue`. Easier to schedule (single-GPU requests), CPU no longer a
per-container limit, an env crash does not take down the learner; costs a network hop per chunk (milliseconds against
a 230 ms model call) and more plumbing. Choose B if `RTX-PRO-6000:8` does not schedule or CPU becomes the limit.

## Cost (Modal list prices, https://modal.com/pricing, read 2026-11-14)
RTX-PRO-6000 $0.000842 / s ($3.03 / h); L40S $0.000542 / s ($1.95 / h); B200 $0.001736 / s ($6.25 / h);
CPU $0.0000131 / physical core / s ($0.047 / h); memory $0.00000222 / GiB / s ($0.008 / h).
* **A** (`RTX-PRO-6000:8`, 64 cores, 256 GiB): 8 x 3.03 + 64 x 0.047 + 256 x 0.008 = **~$29 / h**.
* **B** (7 x [1 GPU, 24 cores, 96 GiB] + 1 x [1 GPU, 8 cores, 64 GiB]): 24.25 + 8.3 + 5.9 = **~$38 / h**
  (the CPU / memory requests are what this test used, not a measured need).

## Estimated wall-clock for ~300k env ticks of DICE-RL
| layout | env ticks/s | env time | + startup | cost (GPU + CPU + memory) |
|---|---|---|---|---|
| A, K = 1 (7 env GPUs) | ~170 | ~30 min | ~3-5 min | ~$17 |
| A, K = 8 (4 env + 4 prior GPUs) | ~96 | ~52 min | ~3-5 min | ~$27 |
| 1 x `RTX-PRO-6000:2`, 36 cores (dev / debug) | ~40 | ~2.1 h | ~3 min | ~$19 ($8.9 / h), prior not included |

Learner updates are not included (unknown until the DICE-RL learner exists); they add on top unless overlapped with
collection. Env time = 300k / ticks/s, prior capacity checked as above.
