## 1. Approval And Source Study

- [ ] 1.1 Obtain explicit user approval to implement this evaluator change after OpenSpec review.
- [ ] 1.2 Re-check `main` evaluator behavior for COCO bbox conversion, empty prediction handling, score validation, category mapping, and metric key names; record only implementation-relevant findings in the work notes or commit message.
- [ ] 1.3 Verify current CoordExp-swift inference artifact fields in `gt_vs_pred.jsonl`, `gt_vs_pred_scored.jsonl`, `parse_diagnostics.jsonl`, and provenance sidecar before coding.
- [ ] 1.4 Confirm `pycocotools` is available in the target `ms` runtime and decide the failure message if it is missing.

## 2. Failing Tests First

- [ ] 2.1 Add a failing test for a perfect one-object scored fixture producing `mAP=1.0`, `mAP_50=1.0`, and `mRecall=1.0`.
- [ ] 2.2 Add a failing test proving an empty scored `pred: []` row with GT remains in evaluation and produces zero AP/recall rather than being dropped.
- [ ] 2.3 Add a failing test proving parser-dropped spans are counted from raw artifact diagnostics but are not reparsed by the evaluator.
- [ ] 2.4 Add a failing test proving unknown prediction categories are excluded from `coco_predictions.json` and counted, while unknown GT categories fail fast.
- [ ] 2.5 Add failing tests for missing score provenance, invalid scores, raw/scored row mismatch, and provenance SHA mismatch.
- [ ] 2.6 Add a failing CLI test for `scripts/evaluate_detection.py --artifact-dir ... --out-dir ...`.

## 3. Evaluator Implementation

- [ ] 3.1 Add or deepen a small COCO-80 category registry under `src/eval` with prompt-vocabulary parity tests or explicit derivation evidence.
- [ ] 3.2 Implement artifact normalization that validates scored rows, reads raw rows for counters, enforces row parity, and never reparses raw decode text.
- [ ] 3.3 Implement COCO GT conversion preserving every valid image row and every valid GT object.
- [ ] 3.4 Implement COCO prediction conversion from already-scored predictions, including unknown-category counters and invalid-prediction diagnostics.
- [ ] 3.5 Implement official COCO bbox metric reduction through `pycocotools`, including deterministic zero-metric behavior for valid GT with no predictions.
- [ ] 3.6 Write `metrics.json`, `coco_gt.json`, and `coco_predictions.json` with `allow_nan=False`.

## 4. CLI And Operator Surface

- [ ] 4.1 Wire `scripts/evaluate_detection.py --artifact-dir RUN_DIR --out-dir RUN_DIR/eval` to the standardized evaluator.
- [ ] 4.2 Support `--pred-jsonl` as a compatibility alias for paths ending in `gt_vs_pred_scored.jsonl`.
- [ ] 4.3 Ensure the CLI can run from the repository root without relying on legacy `src.eval.detection` or `src.infer` imports.
- [ ] 4.4 Print the metrics path and compact JSON metrics summary without treating stdout as the artifact source of truth.

## 5. Verification

- [ ] 5.1 Run `pytest tests/eval/test_detection_consumer.py -q`.
- [ ] 5.2 Run a tiny local artifact smoke through the CLI and inspect `metrics.json`, `coco_gt.json`, and `coco_predictions.json`.
- [ ] 5.3 Run `openspec validate standardize-coordexp-swift-detection-evaluator --strict`.
- [ ] 5.4 Run `git diff --check`.
- [ ] 5.5 Confirm no implementation imports MS-Swift or legacy `src.eval.detection` as runtime dependencies.

## 6. Re-Eval Readiness

- [ ] 6.1 Provide the exact command template for re-evaluating a retrained checkpoint's existing inference artifact directory.
- [ ] 6.2 State evidence scope clearly: evaluator tests and tiny smoke validate the reducer; full checkpoint mAP/mRecall claims require running the command on the target artifact directory.
