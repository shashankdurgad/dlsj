# SO-101 digital twin (Isaac Lab 3.0 + Physical AI Studio, Docker, GB10)

Goal: a twin of the Duplo station accurate enough to RL in (residual / speed policies on top of a frozen VLA),
built on the residual env in `sim/residual_env` (servo model, scene distributions, calibration profiles).

## Environment
```
sim/twin/docker/build.sh                 # dlsj/isaaclab-base:3.0.0 (Isaac Lab v3.0.0 official image on isaac-sim:6.2.0)
                                         #   + dlsj/twin:latest (Physical AI Studio library, pinned to its 8f132416)
sim/twin/scripts/fetch_so101.sh          # SO-101 URDF + meshes (TheRobotStudio/SO-ARM100, pinned) -> assets/so101/urdf
sim/twin/run.sh sim/twin/scripts/smoke.py  # URDF -> USD, residual-env scene, one frame per camera -> sim/twin/out/
sim/twin/run.sh --shell                  # interactive shell in the same container
```
Every run is a throwaway container (`--rm --init`, named `dlsj-twin-*`); Kit / shader caches persist in the
`dlsj-twin-*` docker volumes. Generated files (`.build/`, `assets/so101/`, `out/`) are git-ignored.
Verified 2026-11-14 on watson (GB10): isaaclab 28.0.0 (v3.0.0), physicalai-train 0.3.0, torch 2.11 cu130, both cameras render.

## Calibration status
* `calib/realsense_intrinsics.json`: factory colour intrinsics of the top and inspect D415s at the recording
  settings (fx ~605 px, top cx 304). The current sim profile (`joe_cal2`) uses fitted 583 / 527 px with a centred
  principal point, so its fitted camera poses absorb that error.
* `calib/charuco_7x5_35mm_A4.pdf`: board for measured extrinsics (top / inspect on the table, wrist hand-eye),
  with the board pose in the robot frame from touching its corners with the gripper tip.

## Next
1. measured camera extrinsics (ChArUco) + render-vs-real overlay check on recorded episodes;
2. appearance: table texture from the top camera, plate / Duplo meshes, phone scan (USDZ) of the surroundings;
3. a Studio `Gym` wrapper around the env so Studio policies roll out in the twin.
