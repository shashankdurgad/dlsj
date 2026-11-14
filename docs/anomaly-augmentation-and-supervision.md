# Brick anomaly detection and mark classification

Use Anomalib PatchCore as the normal-versus-anomalous detector. Train its normal
memory bank on unmarked bricks. Crosses and circles are both anomalies. A separate
supervised classifier can identify the mark type; PatchCore scores do not encode
cross versus circle.

Run the detector and classifier on the same brick face. Anomaly detector says
normal and classifier says unmarked: accept. Detector says anomalous and classifier
confidently identifies cross or circle: send to the corresponding bin. Conflicting
or weak evidence: return uncertain and inspect again. An anomalous object need not
contain either known mark, so retain an unknown-defect outcome. Show the anomaly
score and heatmap in the demo, together with the robot's resulting action.

This is an interpretation of the supplied rubric, which emphasizes Anomalib
detection quality, normal/defective handling, integration, and downstream robotics.
It is not an assurance from the judges about auxiliary classifiers.

## Reproducible augmentation experiment

Implementation: `dlsj/inspect/augment.py`. Dependency: Albumentations 2.0.8 in a
separate environment; it does not change the robot runtime environment.

The frozen cleaned source is
`../anomaly-training/cleaned-20261114T024821/studio_dataset.json`.
It contains 331 captures, each with three frames. Only its 69 normal training
captures (207 images) are augmented. Validation and test images remain unchanged.
Every derivative retains its parent capture identifier, source hash and seed.
Transformation metadata is saved alongside the generated JPEG files.

| Profile | Changes | Normal training images |
|---|---|---:|
| baseline | Originals | 207 |
| geometry | Safe rotation up to 15 degrees, padding, small scale and translation | 1,035 |
| photometric | Mild brightness, contrast, gamma, colour, sensor noise, blur and compression | 1,035 |
| combined | Both groups | 1,035 |

Each augmented profile adds four variants per original image. The three views
share geometric and lighting draws; sensor noise varies by view. Padding and safe
rotation preserve the complete source crop. Do not use erasing or fabricated dark
marks on normal examples, because those could teach the model to accept defects.

Generate a new output directory:

```sh
python -m pip install 'albumentations==2.0.8'
python -m dlsj.inspect.augment \
  --source-manifest ../anomaly-training/cleaned-20261114T024821/studio_dataset.json \
  --out ../anomaly-training/augmentation-new --variants 4 --seed 2026
```

`scripts/train_augmentation_studio.py` imports each normal training set into a
separate Studio project and trains PatchCore. Use a Python environment containing
`httpx` and a running Studio at localhost:8005:

```sh
python scripts/train_augmentation_studio.py \
  --run-dir ../anomaly-training/augmentation-new
```

For repeatability, the local Studio API has been extended with `training_seed`
and `coreset_sampling_ratio`. The job schema is
`application/backend/src/pydantic_models/job.py`; the training service passes the
seed to Lightning and the ratio to PatchCore. These changes live in the sibling
Studio checkout. Unmodified upstream Studio does not accept these options.

All runs use seed 2026. The baseline coreset ratio is 0.1 and augmented ratios are
0.02, giving approximately the same memory-bank capacity. This controls storage
and inference cost while changing the diversity of the candidate normal patches.
Studio internally splits the imported images and creates synthetic validation
examples; those internal scores are not the external evaluation.

For external evaluation, average the three frame scores per capture. Select a
threshold for each candidate on validation only. The selection objective averages
normal acceptance and the mean of cross/circle detection recalls. Select the best
profile on validation, then evaluate only that winner and the controlled baseline
on the untouched test split. `comparison.json` records results; individual
`evaluation.json` files include thresholds. The thresholds are external calibration
and do not change Studio's default threshold or the existing live tester.

The active experiment is
`../anomaly-training/augmentation-20261114T031256`. Its progress is recorded in
`training_progress.json`; `augmentation_preview.jpg` shows representative draws.
The live tester now uses the validation-selected geometry model and its matching
external threshold. Its launcher records the selection in
`../anomaly-training/live_model_run.txt`.

All four runs completed. Validation selected geometry (objective 0.786), ahead of
combined (0.784), baseline (0.702) and photometric (0.693). On the held-out test:

| Correct decision | Controlled baseline | Geometry winner |
|---|---:|---:|
| Accept unmarked | 9/16 | 15/16 |
| Flag cross as anomalous | 57/58 | 54/58 |
| Flag circle as anomalous | 54/55 | 50/55 |
| Selection objective on test | 0.772 | 0.929 |

Geometry improves normal acceptance with some loss of marked-brick recall. These
are anomaly decisions, not cross-versus-circle classification. The selected
external threshold is 0.9991861979166667, averaging three frame scores, and model
ID is `c01c0cdc-80b5-44b4-a8ab-bdbc5db6eb52`. The previously deployed model accepted
12/16 normal and detected 55/58 crosses and 53/55 circles on the same test split;
the new candidate therefore represents a different false-reject/missed-mark
tradeoff. The small normal test group and unknown physical identities limit the
conclusion. The geometry candidate is now loaded for interactive live testing;
the original frozen-feature classifier remains an offline probe. A later
[YOLO classifier](yolo-brick-classification.md) is now connected to the tester.

Calibration limitation observed in the controlled baseline: 27/42 normal,
173/177 cross and 177/180 circle validation frame scores were clipped to 1.0.
The installed Anomalib post-processor explicitly clamps normalized scores to
[0, 1], using statistics from Studio's internal validation. A threshold close to
1.0 is not a probability. A follow-up export should retain raw anomaly scores and
calibrate them on real validation captures, then validate the exported decision
against the evaluation pipeline. Augmentation alone does not fix this loss of
score resolution.

## Supervised follow-up

Start with a pretrained lightweight image classifier and the existing image-level
labels: unmarked, cross, circle. Compare a frozen-feature linear classifier with
fine-tuning the final backbone block, using class balancing, modest augmentation
and early stopping.

An exploratory frozen-feature baseline has now been trained by
`scripts/train_brick_classifier_probe.py`: pretrained WideResNet50-2 layer2/layer3
global mean/max features, averaged across three views, then a class-balanced scaled
logistic regression. It selects regularization on validation only. Its supervised
split has 158 training captures, 44 validation captures and the original 129 test
captures. No augmentation was used for this probe.

Test accuracy was 87.6%, with macro F1 0.832. Confusion rows are actual labels and
columns are predictions:

| Actual / predicted | Unmarked | Cross | Circle |
|---|---:|---:|---:|
| Unmarked | 12 | 3 | 1 |
| Cross | 3 | 52 | 3 |
| Circle | 4 | 2 | 49 |

Artifacts are in `../anomaly-training/supervised-probe-20261114`, including a split
manifest, frozen features, classifier, per-capture test predictions and evaluation.
This is a classification feasibility result, not a validated combined detector,
confidence-gated model, live deployment or robot integration. The next supervised
experiment should fine-tune the final backbone block and add training-only mild
augmentation, without selecting hyperparameters using these test results.

To reproduce the frozen probe using an environment with torch, torchvision, timm,
scikit-learn, joblib, NumPy and Pillow installed:

```sh
python scripts/train_brick_classifier_probe.py \
  --source-manifest ../anomaly-training/cleaned-20261114T024821/studio_dataset.json \
  --out ../anomaly-training/supervised-probe-new
```

The anomaly split currently has no marked training examples. Create a separate
supervised manifest that divides the existing marked validation captures into
training and validation. Keep the original test captures untouched. Split by
physical brick and recording session if their identities can be recovered;
capture-only grouping cannot rule out the same physical brick occurring in more
than one split. Keep every augmented view with its parent.

Evaluate three-class confusion, macro F1, each mark's recall, normal false rejects,
unknown-defect handling and the complete combined robot decision. Subtype accuracy
on correctly detected anomalies alone hides missed marks in the first stage.
Use confidence thresholds calibrated on validation and report the proportion of
uncertain decisions, rather than forcing all images into the three known classes.

For supervised anomaly localization, annotate actual ink masks on marked faces.
Whole-brick crop boxes are not defect masks. A supervised anomaly architecture
would require a validated mask-aware data/training path; the current Studio
PatchCore workflow does not gain semantic mark labels just by importing bad images.

Sources: [Anomalib PatchCore](https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/patchcore.html),
[Albumentations replay](https://albumentations.ai/docs/4-advanced-guides/replay-and-applied-params/).
