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

## Camera check from recorded data (target-free)
`align/make_samples.py` -> `align/export_points.py` (Isaac: robot surface points at recorded joint states, the sim's
own kinematics) -> `align/fit_cameras.py` (silhouette fit, factory intrinsics, 2:1 episode split). Results in
`calib/cameras_fit.json`, overlays in `out/align/overlay_{top,inspect}.jpg` (red = joe_cal2, green = fitted).
* top: joe_cal2 is already right (held-out IoU 0.363 -> 0.369 fitted; pose change 3 cm / 3 deg).
* inspect: fitted is better (held-out loss 12.8 -> 4.2) but tentative; measure with the screen board.
* wrist: not fitted (unknown intrinsics; needs the board).
Screen boards for a phone / laptop instead of the printed PDF: `calib/make_screen_boards.py`, `calib/screen_boards.json`.

## Next
1. measured camera extrinsics (ChArUco) + render-vs-real overlay check on recorded episodes;
2. appearance: table texture from the top camera, plate / Duplo meshes, phone scan (USDZ) of the surroundings;
3. a Studio `Gym` wrapper around the env so Studio policies roll out in the twin.

## Visual realism (`visuals/`)
Makes the twin's camera images look like the recorded cal2 frames (pi0.5 rollouts / RL in the twin). Physics is not changed.
```
python sim/twin/visuals/table_texture.py     # top/inspect medians -> textures/table_top.jpg, textures/backdrop_inspect.jpg (plain python, docker image venv)
python sim/twin/visuals/measure_real.py      # plate diameters, brick / plate / jaw colours from real frames -> out/visuals/real_appearance.json
python sim/twin/visuals/measure_real.py --ccm   # per-camera colour transforms (top -> wrist / inspect) -> visuals/camera_color.json
sim/twin/run.sh sim/twin/visuals/render_compare.py [--calibrate] [--episodes ...] [--frame N]   # real | sim | blend -> out/visuals/compare_*.jpg
```
* `twin_scene.py`: `twin_scene_cfg` (residual env cfg + lights + inspect camera) and `apply_twin_visuals` (after `InteractiveScene`, before
  `sim.reset()`); `color_correct(img, "wrist"|"inspect")` for the per-camera colour response.
* **GB10 render path: the cameras do not see physics motion by themselves.** With the Fabric scene delegate off (KIT_ARGS) the renderer reads
  USD, and PhysX never writes poses back to USD (also not with `use_fabric=False`), so every camera keeps rendering the spawn pose. Writing the
  physics prims' xforms invalidates the PhysX tensor views. The twin therefore renders physics-free visual proxies (`RobotVisual`,
  `TwinVis/*`) that `usd_sync.VisualSync` moves from the physics state before each render; the wrist camera's local xform is corrected the same
  way. Call `sync(); sim.render(); scene.update(0)` for every rendered frame.
* `lighting.json` (`render_compare.py --calibrate`): dome + key light scale so the textured table renders like the real top-camera median
  (linear tonemap); `backdrop_gain` for the inspect backdrop.
