# Live brick inspection on the Mac

Open http://localhost:8006 or double-click `Start Live Brick Inspection.command`
in `~/Desktop/hackjunction`. The launcher starts Studio if needed.

Allow camera access, select a camera, and fit one brick face inside the square.
Click the preview to move the square and use its size slider to fit the brick.
Click **Start detection** and hold the brick steady for three predictions.
The right panel shows the actual inspected crop and optional anomaly heatmap.

For a fresh machine, download the published weights and run with `--model-dir`.
The [model download and setup instructions](../models/brick_patchcore_geometry/README.md)
use direct OpenVINO inference and do not require Studio or this Mac's launcher.

The tester uses geometry-augmented PatchCore
`c01c0cdc-80b5-44b4-a8ab-bdbc5db6eb52`, with mean score over three frames
and validation-calibrated threshold 0.9991861979166667. The slider preserves the
calibrated value; displayed scores and threshold use six decimal places.
This is an anomaly score, not a probability. The threshold slider affects the
live tester only. It does not modify the model or Studio calibration.
Moving the inspection square, switching cameras, or pausing clears old scores.
No camera frames are saved by this tester.

Run against another Studio model/calibration pair with:

```sh
python -m dlsj.inspect.live_web --run-dir /path/to/training/run --port 8006
```

The run directory must contain matching `studio_model.json` and `evaluation.json`.
Studio must be running at localhost:8005. Model predictions stay on this Mac.

The launcher reads `../anomaly-training/live_model_run.txt` to select the saved
model/calibration pair. It currently points to
`../anomaly-training/augmentation-20261114T031256/geometry`.

The cleaned-data held-out PatchCore results were 15/16 plain accepted, 54/58 crosses
and 50/55 circles detected as anomalous. The Mac launcher now also loads the
[trained YOLO classifier](yolo-brick-classification.md), showing plain, cross,
circle or uncertain/reinspect. It averages three YOLO probability vectors and
requires agreement with the Anomalib decision. The local classifier selection is
stored in `../anomaly-training/live_classifier_dir.txt`.
Empty crops, obscured marks, or different camera
conditions are not validated uses. This is an interactive evaluation tool.
