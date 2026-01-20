# Change: Add F1-ish detection evaluator (greedy matching + semantic scoring)

## Why
COCO-style mAP is a poor fit for the current CoordExp detection workflow:
- Generation is greedy and does not provide calibrated confidence scores (pred scores are effectively constant).
- Open-vocabulary desc strings are not canonicalized, so exact-string category mapping over-penalizes near-synonyms (e.g., `armchair` vs `chair`).
- For research iteration and EM-ish post-training readiness, we need a metric that explicitly quantifies:
  - **missing objects** (false negatives),
  - **matched objects** (true positives),
  - **hallucinations / extra objects** (false positives),
  while still reporting whether the matched objects are semantically correct.

## What Changes
- Add a new evaluator metric mode: **F1-ish** (set-level matching + counts).
  - Per-image greedy 1:1 assignment by **location first** using IoU.
  - Use **segmentation IoU when polygons exist**, and always support **bbox ↔ poly** matching by converting bboxes to rectangle segmentations.
  - After location matching, score **semantic correctness** using exact match and/or embedding similarity.
- Keep existing COCO artifacts and COCOeval metrics available for comparability; evaluator can run **COCO**, **F1-ish**, or **both** in one invocation.
- Emit additional diagnostics artifacts for EM-ish workflows:
  - `matches.jsonl` containing per-image matched pairs with IoU and semantic similarity.
  - Per-image counts for `matched`, `missing`, and `hallucination`.

## Impact
- Affected specs: `detection-evaluator`
- Affected code: `src/eval/detection.py`, `scripts/evaluate_detection.py`, and docs under `docs/` for runbook updates.
- Backward compatibility:
  - COCO outputs remain unchanged when running in COCO-only mode.
  - F1-ish metrics add new keys/files but do not change the prediction JSONL contract.

