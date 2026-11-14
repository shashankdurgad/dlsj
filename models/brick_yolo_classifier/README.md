# Trained YOLO brick classifier

Download the trained **YOLO26n classifier** from the
[GitHub Release](https://github.com/shashankdurgad/dlsj/releases/tag/brick-yolo-classifier-v1).
It predicts circle, cross and plain from a complete 320×320 brick-face crop.
The live tester combines it with the separately released Anomalib PatchCore
model and returns uncertain/reinspect for weak or conflicting predictions.

## Run the combined live tester

From the repository root, download both release bundles into separate folders:

```sh
gh release download brick-patchcore-geometry-v1 --repo shashankdurgad/dlsj \
  --pattern 'brick-patchcore-geometry-v1.tar.gz' --pattern SHA256SUMS \
  --dir models/downloads/patchcore
gh release download brick-yolo-classifier-v1 --repo shashankdurgad/dlsj \
  --pattern 'brick-yolo-classifier-v1.tar.gz' --pattern SHA256SUMS \
  --dir models/downloads/yolo

cd models/downloads/patchcore
shasum -a 256 -c SHA256SUMS
tar -xzf brick-patchcore-geometry-v1.tar.gz
cd ../yolo
shasum -a 256 -c SHA256SUMS
tar -xzf brick-yolo-classifier-v1.tar.gz
cd ../../..

python3 -m venv .venv-inspect
.venv-inspect/bin/python -m pip install -r requirements-anomaly-live.txt
.venv-inspect/bin/python -m dlsj.inspect.live_web \
  --model-dir models/downloads/patchcore/brick-patchcore-geometry-v1 \
  --classifier-dir models/downloads/yolo/brick-yolo-classifier-v1
```

Open http://localhost:8006, allow camera access, select the camera and start
detection. This path requires OpenVINO, NumPy and OpenCV, with no running Studio
database, Ultralytics or PyTorch. CPU FP32 inference is the verified YOLO path.
Intel GPU/NPU operation and robot motion are not validated by this release.

The bundle contains `best.xml`, `best.bin`, the matching `classifier.json`, a
PyTorch `best.pt` checkpoint, evaluation reports and per-weight hashes. Loading
the PyTorch checkpoint requires the repository and its custom transform class,
plus the dependencies in `requirements-yolo-training.txt`. The OpenVINO export
is sufficient for live inference. Camera images, datasets and runtime caches are
excluded.

## Results and decision settings

The explicit split has 158 training, 44 validation and 129 test captures, each
with three frames. Fine-tuning ran on the Mac's MPS GPU with early stopping after
18 epochs, selecting epoch 3 by validation accuracy. Training-only transforms
preserve the whole brick crop. See
[training and evaluation notes](../../docs/yolo-brick-classification.md).

- Standalone three-class accuracy: **117/129 (90.7%)**, macro F1 **0.896**.
- Combined Anomalib/YOLO gate: **104/107 decisions correct (97.2%)**, with
  **22/129 uncertain**. Coverage is **82.9%**; correct decisions account for
  **80.6% of all captures**. Three wrong decisions remain.
- Average three YOLO probability vectors. Require confidence >=0.8 and top-two
  margin >=0.2, plus agreement with Anomalib's calibrated anomaly decision.
  The classifier-only validation confidence search missed its target; these are
  the declared fallback settings, not a guarantee of calibrated confidence.
- FP32 OpenVINO parity: no label changes across 519 validation/test frames;
  maximum probability difference below 0.000008.

The existing test benchmark has been inspected in previous experiments, physical
brick/session identities were unavailable, and only 16 test captures are plain.
Validate the complete pipeline on a fresh camera session before drawing broader
conclusions. This model expects whole-brick crops, rather than the tight mark
regions consumed by the team's separate robot stage-2 classifier.
