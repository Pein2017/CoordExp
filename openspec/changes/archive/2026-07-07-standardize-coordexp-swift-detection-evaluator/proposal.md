## Why

CoordExp-swift inference now writes scored detection artifacts, but the current
evaluation boundary is still only a minimal artifact consumer rather than a
standardized benchmark evaluator. Retrained checkpoints need prompt-aligned
offline re-evaluation now, and mAP/mRecall must be computed from the rebuilt
artifact contract without importing legacy evaluator authority as runtime
infrastructure.

This change protects accuracy/precision first by making parser salvage,
prediction scoring, metric eligibility, COCO category mapping, and official
COCO bbox reduction explicit. It preserves efficiency by evaluating existing
scored artifacts without re-running inference, preserves simplicity by keeping
V1 to one official bbox evaluator, and leaves clear extension points for future
diagnostic match traces, category-level reports, vLLM outputs, and rollout
evaluation.

## What Changes

- Add a standardized CoordExp-swift detection evaluator contract for
  `gt_vs_pred_scored.jsonl` plus its provenance sidecar.
- Define the role split:
  - inference parser/scoring owns generated-text parsing, span salvage,
    prediction dropping, score construction, and score provenance;
  - evaluator owns artifact validation, metric normalization, COCO-80 mapping,
    conversion to COCO JSON, and official metric reduction.
- Require official COCO bbox metrics for V1:
  `mAP = AP@[.50:.95]`, `mAP_50`, `mAP_75`, and
  `mRecall = AR@100`, with raw COCO aliases preserved.
- Require rows with `pred: []` to remain metric-bearing false negatives when
  GT and image identity are valid.
- Require dropped/salvaged generated spans to remain diagnostic-only and never
  be reparsed by the evaluator.
- Require a canonical COCO-80 category registry aligned with the prompt
  vocabulary; unknown GT categories fail fast, unknown prediction categories
  are excluded from COCO predictions and counted.
- Require evaluator outputs:
  - `metrics.json`;
  - `coco_gt.json`;
  - `coco_predictions.json`.
- Require a direct operator CLI:
  `python scripts/evaluate_detection.py --artifact-dir RUN_DIR --out-dir RUN_DIR/eval`,
  with `--pred-jsonl` as a compatibility alias for the scored artifact path.
- No implementation begins under this change until the OpenSpec artifacts are
  reviewed and the user explicitly approves implementation.

## Capabilities

### New Capabilities

- `coordexp-swift-detection-evaluator`: Defines the standardized offline
  detection evaluator for CoordExp-swift scored inference artifacts, including
  parser/evaluator role boundaries, COCO-80 normalization, official bbox
  mAP/mRecall semantics, output artifacts, CLI surface, and acceptance checks.

### Modified Capabilities

- None. There are no current stable `openspec/specs/` capabilities to modify in
  this worktree. The completed inference OpenSpec remains historical planning
  context; this change introduces the stricter evaluator contract as a narrow
  follow-up.

## Impact

- Affected code surfaces after approval:
  - `src/eval/detection_consumer.py`;
  - possible small `src/eval/*` helper modules for COCO categories/conversion;
  - `scripts/evaluate_detection.py`;
  - `tests/eval/test_detection_consumer.py`.
- Affected artifact contracts:
  - `gt_vs_pred.jsonl` is diagnostic context and raw parser evidence;
  - `gt_vs_pred_scored.jsonl` is the metric input;
  - `gt_vs_pred_scored.jsonl.provenance.json` gates metric computation;
  - evaluator writes `metrics.json`, `coco_gt.json`, and
    `coco_predictions.json`.
- Dependencies:
  - `pycocotools` is the official COCO bbox reduction dependency;
  - no MS-Swift runtime dependency is introduced;
  - legacy/current main evaluator code may be used as source-study guidance but
    is not copied as a runtime dependency.
