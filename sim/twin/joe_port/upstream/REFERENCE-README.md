# Video-matched SO-101 simulation

The scene is configured on Joe Spark 2 in `/home/joe-spark-2/pi05-isaac-so101`.
It uses the existing Isaac Sim 6.2 runtime and a separate policy container.

## Recording reference

Training dataset: `gs://onflow-30k-pi05-artifacts/datasets/intel-20261114-212episodes-training/duplo_teleop_rgb/`.
Metadata reports 212 episodes, 47,079 frames, 30 Hz, and four tasks.
The three RGB streams are `wrist`, `top`, and `inspect`, each 640 × 480.
I inspected frames across episodes 0, 6, and 211; the scene layout is based on episode 0.
Downloaded source recordings and metadata are in `../reference-data/`.

The scene includes a wooden tabletop, round blue/orange plates, pastel Duplo-sized
bricks with hollow studs, a black SO-101 with red fingers, an orange camera bracket,
and a side inspection camera. The starting joint state comes from the recording.
The conference background uses a distant textured plane sampled from the recording;
the policy receives rendered simulator images, with dynamic robot and objects.

## Calibration limits

This is a visual approximation, not a measured digital twin. A pinhole top camera
and three pitch-joint offsets were jointly fitted to one wrist-housing landmark
over 12 frames. The approximately six-pixel fit error does not establish correct
robot calibration: several camera/robot configurations can explain one landmark.
The wrist mount, inspection view, materials, geometry, and gripper conversion are
estimated. The side and wrist images still differ visibly from the recordings.
See `recording-vs-simulation.jpg` and `assets/camera-fit.json`.

Joint observations subtract the estimated offsets before conversion to degrees;
policy targets add the offsets before applying simulator limits. The gripper uses
the provisional 0–100 to 0–upper-limit conversion. This needs real calibration.

## Policy and reproduction

Model: `w3joe/pi05-junction-so101-20261114` at revision
`34e0f9d11f9b91fa07743493b11b76ac80961675`.
Checkpoint SHA256: `27218d38cf69df81a448794a8484dc0e7cc6f2b15151ddde5ac5625ec9381d85`.
The previous baseline rollout is preserved locally in the parent directory and
remotely under `runs/outputs-*`.

On the already configured Spark, run `bash ~/pi05-isaac-so101/run-reference.sh`.
It preserves existing outputs, rebuilds the scene in the persistent Isaac session,
loads the pinned model, and runs ten 50-action chunks at 30 Hz. Simulation pauses
during inference, so video playback does not establish real-time performance.
The model worker and Isaac dependencies remain isolated. The script relies on the
existing container, Python environment, robot-source checkout, and checkpoint.

The `scripts/` and `assets/` copies capture this configuration. Exported USD layers
retain Spark asset paths and should be opened there.

## Observed result

The revised rollout completed 500 actions (16.67 simulated seconds). The target
brick did not move or lift, and the reviewed video showed no pickup. Warm median
inference time was approximately 0.372 seconds per 50-action chunk. This is one
failed rollout in an approximate scene; it does not establish real-robot performance
or prove that camera calibration alone caused the failure. The raw results are in
`rollout-result.json` and `rollout-arrays.npz`; the review is in `evaluation.json`.
