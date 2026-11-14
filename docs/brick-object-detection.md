# Automatic brick cropping

The brick **detector** is a separate one-class YOLO model. The existing
plain/cross/circle YOLO **classifier** cannot return bounding boxes.

```
full camera frame -> brick detector -> target selection -> 5% padded crop
                  -> aspect-preserving 320 x 320 letterbox -> Anomalib + classifier
```

The live tester supports the existing PatchCore plus whole-brick YOLO ensemble
on automatic crops. Both models have been trained on matching detector crops.
The ensemble averages three frames and requests reinspection when models disagree
or classifier confidence is weak. Automatic inspection is enabled for experimental
live testing with `--auto-inspection`; the page displays the unmet validation
target. Omitting this flag keeps automatic mode as crop preview only. Manual
inspection remains available.

## Annotation contract

Canonical model-reviewed annotations use this format (coordinates refer to the
original full-resolution image):

```json
{
  "version": 1,
  "class_names": ["brick"],
  "images": [{
    "path": "/absolute/path/to/original.jpg",
    "capture": "capture-id",
    "split": "train",
    "width": 1280,
    "height": 720,
    "boxes": [{"xyxy": [400, 200, 650, 400], "class_id": 0}],
    "review_status": "reviewed",
    "notes": "Model-reviewed; mark visible"
  }]
}
```

Use one class for all physical bricks, including plain, circle and cross. Label
all visible bricks; mark occlusion/truncation and flag uncertain images as
`needs_review` or `unusable`. Only `reviewed` images enter training. Reviewed
negative images may have an empty box list. Preserve capture/session splits;
near-identical frames must not cross splits. Include Intel inspection-camera
views and negative examples before claiming deployment accuracy. Model review is
not human ground truth; audit the box contact sheets before relying on metrics.

## Prepare and train

Use `requirements-yolo-training.txt` in an isolated environment:

```sh
python -m dlsj.inspect.brick_detection_train \
  --annotations /path/to/annotations.json --out /path/to/new-run \
  --prepare-only
```

The importer checks box bounds, image dimensions, capture split leakage and
identical images across splits. Source images are copied unchanged. It refuses
nonempty output datasets to avoid mixing stale labels into a new run.

To train, choose a **different, empty run directory**, omit `--prepare-only`, and
optionally set `--device mps`, `cpu` or a CUDA device. Defaults use pretrained
`yolo26n.pt`, 640-pixel input, 100 epochs with validation early stopping, batch 8,
and seed 2026. The selected checkpoint is evaluated on test once, then exported
as static batch-1 OpenVINO FP32. Keep the source annotations, generated manifest,
checkpoint, test metrics and export together. Do not repeatedly select settings
using test results.

## Live preview

```sh
python -m dlsj.inspect.live_web \
  --run-dir /path/to/anomaly/run \
  --classifier-dir /path/to/whole-brick-classifier \
  --detector-weights /path/to/detector/train/weights/best.pt \
  --camera-source http://100.77.191.42:8000/snapshot/inspect.jpg
```

Automatic cropping is selected by default when detector weights are supplied.
Click **Start detection**; uncheck **Auto-detect brick (crop preview)** only to
use the manual crop fallback. Grey boxes show detected bricks; the green rectangle is the selected brick.
The manual square and its size control are hidden in automatic mode.
The right preview is the exact JPEG crop that can be passed to Anomalib, including
the neutral grey letterbox. The box is computed from the same frame as the crop.
A box includes some background; this is detection, not instance segmentation.

Automatic detection covers the full frame. A single detected brick is selected
automatically. With multiple candidates, click a detected box to choose a target;
otherwise the result remains ambiguous unless exactly one overlaps the previous
target at IoU >= 0.3.
This is short-term box association, not persistent physical identity tracking.
Missing/ambiguous targets clear decisions and crops. Boxes with a short side
below 64 source pixels remain visible as previews but cannot trigger Anomalib;
the page asks the user to move the brick closer. This is an initial image-quality
heuristic, configurable with `--min-inspection-side`, not a guarantee that marks
are readable. Use the same setting when building crop datasets. A substantial box change
also clears the three-frame scores. The browser still supports local cameras
and the Intel snapshot feed. No robot-control commands are issued.

`--detector-device cpu` is the default and is separate from the anomaly/classifier
device. OpenVINO detector exports can also be supplied through Ultralytics; this
requires its runtime dependencies, unlike the standalone classifier runtime.
Runtime uses fixed square 640-pixel letterboxing for both PyTorch and OpenVINO
and explicitly requests FP32 for OpenVINO CPU inference. Benchmark the real
device before selecting CPU/GPU/NPU placement.

## Enable automatic decisions after validation

Use `inspection_crop` from `dlsj.inspect.brick_detector` for training preparation
and live inference. It pads each detected box by 5%, clips to the camera frame,
resizes without distortion and letterboxes to 320 x 320 with value 127. Retrain
PatchCore on matching plain crops if the existing model does not transfer, and
validate/retrain the whole-brick classifier on matching crops too. Keep the
robot pipeline's separate **mark-region classifier** contract unchanged.

Calibrate anomaly thresholds and classifier gates using validation sessions only.
Measure end-to-end outcomes, missed/ambiguous target coverage, complete-mark
retention and latency on a separate camera session. Only then start the tester
with the matching model files and `--auto-inspection-calibrated`. This flag does
not perform calibration: it declares that the supplied models/calibration were
already validated. Until then, auto mode shows crops and never calls Anomalib.

## Matching training/evaluation crop datasets

The annotation workflow also records `target_box_index` for the intended held
brick; an incidental background brick must never inherit its mark class.
`brick_crop_dataset` runs the trained detector over all three original views,
associates it with the reviewed target, and uses the runtime crop function:

```sh
python -m dlsj.inspect.brick_crop_dataset \
  --annotations /path/to/annotations.json \
  --source-manifest /path/to/input_manifest.json \
  --detector-weights /path/to/best.pt --out /path/to/new-crops
```

The source manifest contains one original `_f0.jpg` representative per capture,
with `category` (`good`, `bad`, `circle`) and the fixed split. Output includes a
supervised manifest, a normal-only-training anomaly manifest, source/crop hashes,
box coordinates and `coverage.json`. Ambiguous targets, missing annotations or
missed views remain explicit misses. Retained-crop accuracy alone excludes these
failures and must not be presented as end-to-end accuracy. Training recipes that
need a validation-only experiment can use `--validation-only` on the detector
trainer; that skips test evaluation and export.

## Local experiment results (14 November 2026)

Artifacts are outside Git under `../anomaly-training/`. The current automatic
preview at port 8007 uses the detector from
`brick-detector-audited-20261114/run/train/weights/best_openvino_model`.
Gemini/OpenCode reviewed 330 original representatives (315 accepted, 15 flagged)
and 16 Intel camera training frames. These are model-reviewed annotations, not
independent human ground truth.

The selected detector scored test mAP50 0.919 and mAP50-95 0.535 on 122
reviewed test images. OpenVINO FP32 matched all 40 validation images with no box
count changes and zero coordinate differences. Intel examples were training-only;
these numbers do not establish accuracy on a fresh Intel camera session.

`brick-auto-crops-20261114` retained 307 of 330 captures, requiring all three
views to have usable matching crops. It retained 116 of 129 test captures;
the other 13 remain explicit misses, not correct predictions.

Matching PatchCore training is in `brick-auto-inspection-20261114/baseline`.
The matching YOLO classifier is in `brick-auto-classifier-20261114`, including
`deployment.json`, `joint_evaluation.json`, and `export_validation.json`.
Classifier-only accuracy was 113/116 (97.4%) on retained test captures.
OpenVINO export changed no frame labels across validation and test.

The combined Anomalib/classifier gate produced 88 correct decisions, one wrong
decision and 27 uncertain results among the 116 retained test captures. Counting
all 129 original test captures, correct decisions covered 68.2%. The validation
calibration target was not met: it still labelled one circle as plain. The candidate is enabled with `--auto-inspection` for user-requested live testing
of the same ensemble workflow. Do not pass `--auto-inspection-calibrated` for this
candidate. The remaining quality work is improving mark detection and checking a
fresh labelled camera session.

All 64 relevant unit tests passed. The Intel camera subsequently reconnected.
A browser check on the actual stream detected a 60 x 57 pixel brick and correctly
requested a closer view, without triggering downstream inference on this
undersized target. No JavaScript errors were observed. A fresh labelled Intel
camera session is still needed for accuracy evaluation.

## Run the crop ensemble locally

Use `../Start Auto Brick Inspection.command` to start Studio and the auto-crop
ensemble at http://localhost:8007. It reads the exact model command saved in
`../anomaly-training/brick-auto-inspection-20261114/live_command.json`.
The original manual tester at port 8006 is available separately.

```sh
python -m dlsj.inspect.live_web \
  --run-dir ../anomaly-training/brick-auto-inspection-20261114/baseline \
  --classifier-dir ../anomaly-training/brick-auto-classifier-20261114/train/weights/best_openvino_model \
  --detector-weights ../anomaly-training/brick-detector-audited-20261114/run/train/weights/best_openvino_model \
  --auto-inspection --port 8007 \
  --camera-source http://100.77.191.42:8000/snapshot/inspect.jpg
```

The existing classifier architecture is YOLO26n, not YOLOv8. The bounding-box
detector is also YOLO26n. Detection supplies the crop; PatchCore supplies the
anomaly score/heatmap, and the separate whole-brick YOLO supplies mark class
probabilities. Both downstream models receive exactly the same crop JPEG.
API smoke checks on saved validation source frames covered plain, cross and
circle, with identical scores/probabilities between automatic cropping and
direct prediction on the returned crop. This is not a fresh live-camera test.
