# YOLO brick classification alongside Anomalib

The Mac live tester at http://localhost:8006 now runs geometry-augmented Anomalib
PatchCore plus a trained YOLO26n image classifier. It shows the anomaly score and
heatmap together with the YOLO class/confidence. Final outcomes are plain, cross,
circle or uncertain/reinspect. This is a camera tester; it does not issue robot
commands or replace the team's region-based stage-2 robot interface.

For a fresh machine, [download the released weights and run the combined tester](../models/brick_yolo_classifier/README.md).

## Data and training

The source is the frozen cleaned set of 331 captures (three frames each). The
explicit supervised split from the frozen-feature experiment is reused:

| Capture split | Plain | Cross | Circle | Total |
|---|---:|---:|---:|---:|
| Train | 69 | 44 | 45 | 158 |
| Validation | 14 | 15 | 15 | 44 |
| Test | 16 | 58 | 55 | 129 |

All three frames remain with their parent capture. Preparation verifies source
hashes, rejects cross-split capture groups and copies the original images without
changing held-out bytes. There are 474 training, 132 validation and 387 test images.
Physical brick/session identities were unavailable. The existing test benchmark
has already been examined during earlier experiments; a new camera session is
needed for an independent final assessment.

YOLO26n classification starts from ImageNet-pretrained weights. The input is the
complete 320×320 RGB brick crop, converted to float32 NCHW /255. The trainer
overrides Ultralytics' default cropping transforms. Training-only augmentation is
safe rotation ±15°, padding, small scale/translation, mild brightness/contrast,
noise, blur and JPEG compression. There is no random erasing or mark-removing crop.
Validation and inference use the complete source crop with no augmentation.

The five-epoch smoke run completed in 33.9 seconds on the Mac's Apple M4 Pro MPS
GPU. The full run used AdamW, learning rate 0.001, weight decay 0.0005, batch 16,
seed 2026, 100 maximum epochs and patience 15. It stopped after 18 epochs in 66.1
seconds; the best checkpoint was epoch 3, selected by frame-level validation
accuracy. These timings exclude environment setup and pretrained-weight download.

## Results

Per-capture predictions average the three frame probability vectors. On the
existing 129-capture test set, standalone YOLO accuracy was **90.7%** and macro F1
was **0.896**, compared with **87.6%** and **0.832** for the frozen-feature probe.
This small benchmark improvement does not establish performance on new sessions.

| Actual / predicted | Circle | Cross | Plain |
|---|---:|---:|---:|
| Circle | 50 | 2 | 3 |
| Cross | 5 | 52 | 1 |
| Plain | 1 | 0 | 15 |

The classifier-only search did not achieve its target of 95% validation accuracy
at >=50% coverage. The live configuration therefore uses the declared fallback
confidence threshold 0.8 and top-two margin 0.2, without a claim that softmax
confidence is calibrated or guarantees correctness. Settings were fixed before
test evaluation.

The combined gate also requires agreement with Anomalib. Its calibrated anomaly
threshold remains 0.9991861979166667 on the mean of three frame scores:

- Anomalib normal + confident YOLO plain: plain.
- Anomalib anomalous + confident YOLO cross/circle: the corresponding mark.
- Weak confidence, a small top-two margin, or disagreement: uncertain/reinspect.

On the test set this produced 107 decisions, of which 104 were correct, and 22
uncertain outcomes: **97.2% accuracy among decisions at 82.9% coverage**. Correct
decisions were 80.6% of all captures. Three wrong decisions remain: circle→cross,
cross→circle and cross→plain, one each. Unknown defects are not guaranteed to be
rejected merely because the classifier has confidence thresholds.

## Export and verification

The best model was exported to static batch-1 OpenVINO FP32. The CPU runtime
explicitly requests FP32: the default reduced-precision CPU path changed some
probabilities by up to 0.037 during the initial check. With FP32, the maximum
absolute difference from the saved PyTorch outputs was 0.00000769 across all 519
validation/test frames, with zero frame-label changes.

Tests cover split leakage, unchanged test bytes, RGB/edge preservation, three-frame
aggregation, agreement/disagreement and weak-confidence decisions. The browser
was checked with a synthetic camera stream using real combined inference; all
six decision cases and clearing stale classifier scores were also checked. The
physical camera remains for the user to evaluate. No Intel GPU/NPU benchmark or
robot execution was performed for this YOLO model.

## Files and reproduction

Current artifacts:
`../anomaly-training/yolo-20261114T014100`. They include the explicit dataset
manifest, training logs, `train/weights/best.pt`, the OpenVINO directory,
`evaluation.json`, `export_validation.json`, `joint_evaluation.json`, all frame and
capture predictions, and `training_preview.jpg`.

The exported directory is `train/weights/best_openvino_model` and contains
`classifier.json`, `best.xml` and `best.bin`. The live launcher reads
`../anomaly-training/live_classifier_dir.txt` for this path. Runtime inference
needs OpenVINO, NumPy and OpenCV; it does not need Ultralytics or PyTorch.

Create an isolated Python 3.13 environment and install
`requirements-yolo-training.txt`. From the repository root, train using a new
output directory and the existing explicit supervised manifest:

```sh
python -m scripts.train_yolo_bricks \
  --source-manifest ../anomaly-training/supervised-probe-20261114/dataset.json \
  --out ../anomaly-training/yolo-new --device mps --epochs 100
python -m scripts.evaluate_yolo_export \
  --run-dir ../anomaly-training/yolo-new \
  --anomaly-run ../anomaly-training/augmentation-20261114T031256/geometry
```

Use `--device cpu` if MPS is unavailable, or `--device 0` for a CUDA GPU. The source
dataset is local and is not included in the repository. Training code is
`dlsj/inspect/yolo_train.py`; deployment and gate logic are in
`dlsj/inspect/yolo_classifier.py`.

To load the exported classifier alongside the existing Studio anomaly model:

```sh
python -m dlsj.inspect.live_web \
  --run-dir ../anomaly-training/augmentation-20261114T031256/geometry \
  --classifier-dir ../anomaly-training/yolo-20261114T014100/train/weights/best_openvino_model
```

For standalone anomaly inference use `--model-dir` with the extracted PatchCore
release instead of `--run-dir`. The YOLO classifier was trained on whole brick
crops, so do not feed it the tighter anomaly-region crops used by the team's
existing stage-2 classifier without a matching training/evaluation change.
