## 1. Specs
- [x] 1.1 Add spec delta for `trainer-metrics-components` covering canonical aliases and sparse-gauge aggregation semantics.

## 2. Implementation
- [x] 2.1 Update Stage-2 AB pending logger aggregation to use per-key observation counts for average-style keys.
- [x] 2.2 Add canonical decode-count metric keys in rollout-matching payload and remove legacy aliases.

## 3. Docs
- [x] 3.1 Update `docs/training/METRICS_LOSSES.md` to mark canonical decode keys and minimal metric contract.
- [x] 3.2 Update `docs/training/STAGE2_RUNBOOK.md` with canonical decode metric guidance.

## 4. Validation
- [x] 4.1 Run focused unit tests for Stage-2 AB trainer and rollout-matching decoding metrics.
- [x] 4.2 Re-check representative `logging.jsonl` rows for removed GAS-dilution artifacts.
