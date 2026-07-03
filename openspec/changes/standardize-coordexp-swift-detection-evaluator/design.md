# CoordExp-Swift Detection Evaluator Design

## Context

CoordExp-swift inference owns a rebuilt output stack:

- `src/inference/parsing.py` parses generated compact object-box text into
  structured prediction objects and drop diagnostics.
- `src/inference/scoring.py` computes row-local confidence scores from selected
  generated-token logprobs.
- `src/inference/artifacts.py` writes `gt_vs_pred.jsonl`,
  `gt_vs_pred_scored.jsonl`, token traces, parser diagnostics, and provenance.
- `src/eval/detection_consumer.py` currently exists as a minimal scored-artifact
  consumer, but it is count-oriented and not yet a standardized mAP/mRecall
  evaluator.

The immediate user need is fast re-evaluation of retrained checkpoints without
re-running inference. The long-term need is a crisp owner boundary so future
agents do not mix parsing, scoring, metric eligibility, and official metric
reduction.

The primary design question is where salvage happens. V1 answer:

```text
Generated Text
  -> prediction parser performs span salvage/drop diagnostics
  -> scoring keeps only finite score-bearing parsed predictions
  -> evaluator validates artifacts and reduces already-scored predictions
```

The evaluator never reparses raw decode text and never recovers predictions
from dropped spans.

## Goals / Non-Goals

**Goals:**

- define a standard CoordExp-swift evaluator for `gt_vs_pred_scored.jsonl`;
- make parser/scoring/evaluator ownership explicit and testable;
- compute official COCO bbox mAP/mRecall with `pycocotools`;
- preserve one-row-per-input evaluation so empty-pred rows count as false
  negatives rather than disappearing;
- align category mapping with the COCO-80 closed prompt vocabulary;
- write auditable COCO conversion artifacts next to `metrics.json`;
- provide a simple direct CLI for immediate re-evaluation of existing artifact
  directories.

**Non-Goals:**

- no implementation before user approval after OpenSpec review;
- no raw generated-text parsing inside the evaluator;
- no semantic category remapping, aliases, or description embedding matching in
  V1;
- no custom F1-ish matcher, LVIS/proxy evaluator, segmentation evaluator, or
  per-image/category report in V1;
- no training-time eval integration change;
- no legacy evaluator bridge as the default path.

## Decisions

### Parser Owns Salvage

`src/inference/parsing.py` owns all generated-text recovery. It may accept valid
compact object spans, drop malformed spans, and record `parse_status`,
`dropped_prediction_count`, and `dropped_predictions`. This is the only place
where “salvage” happens.

Rationale: salvage depends on template syntax, coordinate-token boundaries, and
generated text spans. Putting it in eval would create a second parser and make
benchmark numbers depend on evaluator-specific recovery logic.

Alternative considered: let evaluator recover objects from raw decode text.
Rejected because it fragments parsing semantics and makes re-evaluation
non-idempotent.

### Scoring Owns Score-Bearing Prediction Eligibility

`src/inference/scoring.py` and `src/inference/artifacts.py` own whether a parsed
prediction enters `gt_vs_pred_scored.jsonl`. A prediction enters scored `pred`
only when selected-token score evidence is valid, finite, row-local, and
provenance-bearing. Rows remain present even when no prediction is scoreable.

Rationale: official AP depends on prediction ranking. Scores must be produced
from generation trace evidence, not inferred by the evaluator.

Alternative considered: evaluator assigns default score `1.0` to unscored
predictions. Rejected because it would create fake AP comparability.

### Evaluator Owns Metric Normalization, Not Salvage

`src/eval/detection_consumer.py` validates artifact integrity and turns scored
rows into COCO GT/prediction JSON. It may exclude predictions from COCO
conversion when they are not valid benchmark detections, for example unknown
COCO-80 category names or invalid prediction bboxes. That exclusion is metric
normalization, not text salvage; it must be counted in `metrics.json`.

Evaluator responsibilities:

- validate raw/scored row parity and provenance binding;
- validate row-local score provenance and score range;
- read raw rows only for parser/drop normalization counters;
- map GT and prediction descriptions to canonical COCO-80 class ids;
- fail fast on unknown GT categories;
- count and exclude unknown prediction categories;
- preserve empty prediction rows as zero detections;
- run official COCO bbox reduction.

Alternative considered: evaluator drops rows with malformed parse status or
empty predictions. Rejected because it hides false negatives and inflates
metrics.

### Official V1 Metric Scope

V1 uses official COCO bbox metrics through `pycocotools.COCOeval`:

- `mAP`: bbox AP averaged over IoU `.50:.95`;
- `mAP_50`;
- `mAP_75`;
- `mRecall`: bbox AR@100;
- raw COCO aliases such as `bbox_AP`, `bbox_AP50`, `bbox_AR100`.

Rationale: this directly supports checkpoint benchmarking with familiar
semantics and avoids inventing a parallel detector metric.

Alternative considered: add a custom matcher in V1. Rejected for simplicity; it
can be added later as diagnostics if the official metric needs explanation.

### COCO-80 Category Registry

V1 uses a canonical COCO-80 registry aligned with the prompt vocabulary.
Normalization is limited to lowercasing and whitespace collapse. Unknown GT
categories are artifact contract failures. Unknown prediction categories are
excluded from COCO predictions and counted.

Rationale: the training/inference prompt is closed-class COCO-80. Dynamic
category creation or semantic remapping would make metrics sensitive to model
wording instead of the agreed benchmark class set.

Alternative considered: build categories dynamically from GT and predictions.
Rejected because it can hide category drift and make runs incomparable.

### Output And CLI

Evaluator outputs:

- `metrics.json`;
- `coco_gt.json`;
- `coco_predictions.json`.

The operator path is:

```bash
python scripts/evaluate_detection.py \
  --artifact-dir RUN_DIR \
  --out-dir RUN_DIR/eval
```

`--pred-jsonl` is allowed as a compatibility alias for users who have the
scored file path in hand. No YAML eval config is added in V1.

Rationale: this is an offline artifact reducer, not a full run launcher. The
direct artifact-dir CLI is the smallest useful surface for immediate
checkpoint re-evaluation.

Alternative considered: YAML-only eval config. Rejected for V1 because there is
no repeated config family yet and the user needs fast re-eval over existing
artifact directories.

## Risks / Trade-offs

- COCO-80 prompt vocabulary and evaluator vocabulary drift -> keep the
  evaluator registry derived from or explicitly parity-tested against the
  prompt/source list.
- Rows with `pred: []` may surprise users who expect `metric_bearing=false` to
  drop rows -> spec clarifies benchmark eligibility is row/provenance/GT based,
  not prediction-presence based.
- Unknown prediction categories are counted but excluded, which can make AP
  look like both category failure and missing detection -> metrics must include
  `unknown_category_pred_count`.
- `pycocotools` can print noisy summaries -> acceptable for V1 CLI; tests
  should assert JSON outputs, not stdout formatting.
- Minimal V1 lacks per-image/category diagnostics -> accepted trade-off for
  simplicity; add later only if model-diagnosis needs it.

## Migration Plan

1. Keep existing inference artifacts unchanged.
2. Replace the minimal count-only consumer with the standardized COCO bbox
   evaluator after approval.
3. Add tests that first fail on the current count-only behavior.
4. Preserve current artifact names and provenance checks.
5. Add the direct evaluator CLI.
6. Run targeted eval tests and one tiny artifact smoke.
7. Re-evaluate retrained checkpoints from existing artifact directories without
   re-running inference.

Rollback is simple: restore the previous count-only consumer and old script.
No training or inference artifacts need migration.

## Open Questions

- None blocking for V1 after the approved grill decisions.
