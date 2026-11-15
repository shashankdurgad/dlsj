# SO-101 synthetic pickup recordings

This project generates physical pick-and-inspect demonstrations in the existing
Isaac Sim installation on Joe Spark 2. A privileged scripted controller uses the
known brick pose and URDF inverse kinematics. The π0.5 policy does not generate
these demonstrations.

Open [the three-camera replay](previews/successful-pickup-three-cameras.mp4) to see
the first successful recorded episode. The views are top, wrist, and inspection.

## Pilot dataset

The pilot target is **50 accepted episodes: 40 training and 10 validation**. Each
episode contains three synchronized RGB videos at 640 × 480, 30 fps, plus robot
state, commanded joint targets, timestamps, and the task instruction. There are
150 camera clips across the accepted raw episodes. Most episodes last 5.33 seconds;
the first, slower recording lasts 12 seconds.

The target is a mint-green 2 × 2 Duplo-sized brick. Layouts contain three distractor
bricks and blue/orange plates. Target position, yaw, and illumination vary. Training
target X positions are 0.162–0.178 m; validation uses a separate 0.183–0.191 m band.
Both use Y positions from -0.117 to -0.097 m. Target yaw varies from 48 to 68 degrees.
This is a small, deliberately bounded pilot, rather than broad task coverage.

An episode is accepted only if all recorded checks pass:

- The target remains at least 5 cm above its initial resting height for the final second.
- It stays within 4 cm horizontally of the inspection target position.
- The force-limited gripper stalls against the object.
- Opening the gripper after recording makes the target drop at least 3 cm.
- Arm tracking, target/table clearance, camera content, and physics-step counts pass.

No attachment constraint, kinematic brick motion, or teleport is used during a
demonstration. Teleports reset the robot and objects before recording begins.
The post-recording release test is excluded from the demonstration videos/actions.
Rejected attempts stay in `raw/` with `success: false` and are omitted from export.

## Files

- `raw/candidate-NNN/`: three individual camera clips, first/last images,
  `trajectory.npz`, and `result.json` for each attempt.
- `dataset/train/` and `dataset/validation/`: separate native LeRobot v3 datasets.
- `dataset/manifest.json`: episode provenance and the split-to-candidate mapping.
- `dataset/validation.json`: native-loader and source-array verification results.
- `dataset/training-smoke.json`: a test through Physical AI Studio's training adapter.
- `repeatability.json`: the 20-reset physical-controller gate.
- `plans.json`: exact numeric commands and randomized layout parameters.

Each frame records the observation/images at time t, then applies its action over
the following 1/30 second. Physics runs at 60 Hz, two steps per action. Replicator
captures at zero time advance and waits for rendering; the recorder checks that
capture adds zero physics steps. States and actions share the original six-joint
ordering. Five arm angles are in degrees after subtracting the estimated joint
offsets; gripper radians are mapped from 0–upper-limit to 0–100. Raw radians are
also retained. `inspect.prob` is a binary scripted inspection-hold phase marker,
not an estimated visual probability.

## Runtime and reproduction

The Spark working directory is `/home/joe-spark-2/pi05-isaac-so101/synthetic`.
The existing `pi05-isaac-runtime` container mounts the task directory at `/work`.
Isaac Sim stays in its existing persistent process with a loopback Python server.
Export and dataset checks run in the separate container. Nothing is uploaded and
no GCP training job is started.

`run.sh` reproduces a fresh pilot on that configured Spark. It requires the existing
Isaac installation, runtime container, robot-source checkout, and bundled assets.
It preserves an existing run and exits unless `--new-run` is supplied; that option
archives existing raw data and datasets before starting another run. Do not invoke
it while the current generator is running. The script performs scene setup,
repeatability checks, recording, native export, and validation.

## Limits before real-robot fine-tuning

The camera mounting, joint offsets, gripper conversion, materials, and actuator
parameters are approximate. Physical success here does not establish successful
hardware transfer. Some synthetic wrist-flex angles exceed the range observed in
the real training dataset. Calibrate the real/sim mapping before relying on these
demonstrations for deployment, retain real demonstrations during fine-tuning, and
measure performance on untouched real examples.

The scene uses estimated geometry and a static photographic hall backdrop. The
pilot varies one target geometry over a small region. Larger batches should expand
the layout and appearance coverage and be checked independently; repeating the
same trajectory with new rendering settings should stay within the same split.
