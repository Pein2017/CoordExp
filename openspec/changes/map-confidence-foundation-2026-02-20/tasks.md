## 1. Config + Contract Plumbing

- [ ] 1.1 Update evaluator + pipeline YAML plumbing to remove any fixed-score mode: evaluator always honors `pred[*].score` and fails fast if missing/non-numeric.
- [ ] 1.2 Remove `eval.use_pred_score` from configs/templates/docs and add fail-fast rejection if present (no legacy toggles).
- [ ] 1.3 Add/verify artifact path conventions for confidence outputs in the offline post-op config (no new CLI flags; `--config` only).

## 2. Confidence Core (Pure, CPU-only)

- [ ] 2.1 Add `src/eval/bbox_confidence.py` with deterministic confidence math: bbox coord-token logprob reduction (`mean_logprob`) + mapping (`exp`).
- [ ] 2.2 Implement deterministic span resolution (subsequence matching over `generated_token_text`, tie-break earliest unused, left-to-right assignment).
- [ ] 2.3 Add unit tests for reducers/mappings + repeated-pattern tie-breaks (`tests/test_bbox_confidence.py`).

## 3. Offline Confidence Post-Op (Artifacts)

- [ ] 3.1 Add `src/eval/confidence_postop.py` to join `gt_vs_pred.jsonl` + `pred_token_trace.jsonl` by `line_idx` and emit `pred_confidence.jsonl`.
- [ ] 3.2 Implement merge writer to produce `gt_vs_pred_scored.jsonl` without mutating the base artifact:
  - keep only objects with a computed finite confidence-derived `score`,
  - drop objects with `confidence=null` / failures,
  - add `pred_score_source` and `pred_score_version` per record.
- [ ] 3.3 Emit `confidence_postop_summary.json` (counts dropped by reason + kept fraction) to make dropping auditable without inspecting JSONL.
- [ ] 3.4 Add YAML-first entrypoint `scripts/postop_confidence.py` (CPU-only) and a small example config under `configs/` (or document in `docs/` if config placement is ambiguous).
- [ ] 3.5 Add integration test with synthetic JSONL fixtures (`tests/test_confidence_postop.py`) asserting deterministic outputs and failure_reason codes.

## 4. Evaluator: Always Honor Scores (mAP)

- [ ] 4.1 Update `src/eval/detection.py` to always export COCO `score=float(pred['score'])` and fail fast on missing/non-numeric/NaN/inf scores.
- [ ] 4.2 Update tests to assert the new contract:
  - missing/out-of-range scores raise,
  - unscored legacy artifacts (missing `pred_score_source` / `pred_score_version`) are rejected,
  - score ordering is deterministic and stable under ties.

## 5. Docs + Verification

- [ ] 5.1 Update `docs/eval/README.md` to document the confidence workflow: run inference → run confidence post-op → evaluate the scored JSONL (evaluator is score-aware by default).
- [ ] 5.2 Run targeted tests (CPU): `PYTHONPATH=. conda run -n ms python -m pytest -q tests/test_bbox_confidence.py tests/test_confidence_postop.py tests/test_detection_eval_output_parity.py`.
