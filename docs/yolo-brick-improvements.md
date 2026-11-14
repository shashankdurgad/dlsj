# YOLO improvement experiment

The stronger classifier is available for comparison at http://localhost:8007.
The main tester at http://localhost:8006 remains on v1: the new classifier improves
three-class accuracy, but the complete gate has a coverage/error tradeoff.

## What changed

Reviewed thumbnails for all 44 cross and 45 circle training captures. One cross
capture (`cap20261114T001136ea561cbf`) had no readable cross in any of its three
frames; all three views were inspected and excluded from a new training copy.
Original data and all validation/test examples remain unchanged. The revised
training set has 157 captures, 471 images: 69 plain, 43 cross and 45 circle.

The validation error sheet also shows two circle-labelled captures whose marked
faces are hidden or covered. They remain in the unchanged benchmark. A useful
future dataset needs per-view mark visibility annotations: the physical brick's
label does not guarantee the mark is visible in every photograph.

Two training recipes were compared on the same validation split:

| Recipe | Learning rate | Frozen initial modules | Safe rotation | Capture validation accuracy |
|---|---:|---:|---:|---:|
| Published v1 reference | 0.001 | 0 | ±15° | 40/44 (90.9%) |
| Lower LR | 0.0001 | 0 | ±15° | 37/44 (84.1%) |
| Partial freeze + rotations | 0.0003 | 5 | ±180° | 41/44 (93.2%) |

The new runs also use a warmup bias learning rate matching the chosen learning
rate, rather than the original 0.1 warmup default. Augmentation retains the full
source crop, without erasing marks. The winning training recipe stopped after 29
epochs, selecting epoch 9; training took 103.4 seconds on the Mac MPS GPU. Recipes
were run alongside each other, so these are measured run times, not isolated GPU
benchmark timings. Several settings changed together; their individual effects
have not been isolated.

## Selecting the complete decision

At the original confidence/margin settings (0.8/0.2), the candidate made 30 correct
joint validation decisions, versus v1's 31, with no wrong decisions in either.
This exposed a limitation of calibrating the classifier independently from the
Anomalib gate. Both models' complete gates were therefore calibrated using only
the 44 supervised validation captures and the matching Anomalib scores.

The fixed search tries confidence thresholds 0.5–0.95 and margins 0–0.3. Among
settings giving no observed validation errors and >=50% correct coverage, it
maximizes correct decisions, breaking ties toward higher thresholds. This is a
selection rule, not a statistical guarantee of zero error on new data.

The candidate reached 34/44 correct joint validation decisions with confidence
0.5 and margin 0.1; v1 reached 31/44. This candidate/settings selection was saved
before evaluating the candidate on the existing test benchmark. The rejected
lower-LR model was not evaluated on the test split.

An equal-weight v1/v2 probability average was subsequently checked on validation
only. It tied the candidate's 41/44 classifier accuracy and 34 correct joint
decisions; it was not exported or tested, avoiding added inference complexity
without a validation benefit.

## Existing test benchmark results

| Measure | Published v1 | New candidate |
|---|---:|---:|
| Classifier correct | 117/129 (90.7%) | 122/129 (94.6%) |
| Classifier macro F1 | 0.896 | 0.931 |
| Circle correctly labelled | 50/55 | 52/55 |
| Cross correctly labelled | 52/58 | 55/58 |
| Plain correctly labelled | 15/16 | 15/16 |
| Combined correct decisions | 104 | 110 |
| Combined wrong decisions | 3 | 6 |
| Combined uncertain | 22 | 13 |
| Accuracy among decisions | 97.2% | 94.8% |
| Decision coverage | 82.9% | 89.9% |

The candidate's wrong joint decisions comprise circle→cross (2), circle→plain
(1), cross→circle (1), and cross→plain (2). It makes more correct decisions but
also more wrong decisions, so it has not replaced v1 in the main tester.

As a fixed-rule comparison after candidate test evaluation, the new classifier
was also scored with the previous 0.8/0.2 confidence rule: 102 correct, 3 wrong,
24 uncertain (97.1% accuracy among decisions). That also does not dominate v1.
No new threshold was fitted using test labels. These comparisons reuse an already
examined benchmark; fresh camera-session evaluation is required before broader
claims or deployment decisions.

FP32 OpenVINO matched PyTorch labels on all 519 validation/test frames. Maximum
probability difference was below 0.000006. The candidate tester retains Anomalib's
score/heatmap and uses YOLO's complete brick crop, matching training.

## Reproduction and artifacts

Artifacts are in `../anomaly-training/yolo-improvement-20261114`: training review,
reviewed manifest, both recipes, default-threshold and calibrated comparison
reports, best checkpoint, OpenVINO export and all selected-candidate predictions.
Validation errors are visualized in the original run's `validation_errors.jpg`.
The main launcher's model selection files remain on v1.

Training now accepts `--lr0`, `--warmup-bias-lr`, `--freeze`, `--rotation` and
`--patience`. Use `--validation-only` for candidate selection; this skips all test
prediction and export. `--checkpoint` evaluates an existing best checkpoint
without training again. `--joint-anomaly-run` calibrates the complete gate using
matching anomaly-validation predictions.

```sh
python -m scripts.train_yolo_bricks \
  --source-manifest ../anomaly-training/yolo-improvement-20261114/reviewed_dataset.json \
  --out ../anomaly-training/yolo-new --weights /path/to/yolo26n-cls.pt \
  --lr0 .0003 --warmup-bias-lr .0003 --freeze 5 --rotation 180 \
  --patience 20 --validation-only
```

To evaluate/export a validation-selected checkpoint, repeat with the same source
and output, add `--checkpoint /path/to/best.pt` and
`--joint-anomaly-run /path/to/anomaly/run`, and omit `--validation-only`.
Then run `scripts.evaluate_yolo_export` as documented for v1.

The highest-priority next data collection is the actual inspection-camera pose,
with the complete mark visible and fingers outside it. Record new sessions under
different lighting and track physical brick identities, including unmarked bricks
in every colour/shape used for marked examples. Capture a separate final evaluation
session before selecting further model settings.
