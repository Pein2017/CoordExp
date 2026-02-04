## 1. Implementation
- [x] 1.1 Add typed `stage2_ab` config classes in `src/config/schema.py`
- [x] 1.2 Update `src/config/loader.py` / training materialization to parse top-level `stage2_ab`
- [x] 1.3 Replace `schedule.pattern` logic with deterministic `b_ratio` schedule in the trainer
- [x] 1.4 Refactor Channel-A softctx loop to default `unroll` (no detach); keep `em_detach` as opt-in ablation
- [x] 1.5 Split Channel-A CE anchor (CE@A1 logits, geo@final logits)
- [x] 1.6 Replace bbox loss with SmoothL1 + CIoU; remove all GIoU config knobs and code paths
- [x] 1.7 Implement bbox canonicalization for all bbox losses (avoid NaN CIoU)
- [x] 1.8 Implement Channel-B stop-neutral CE (mask top-level `}` and `<|im_end|>`)
- [x] 1.9 Implement Channel-B strict-drop diagnostics: `N_valid_pred`, `N_drop_invalid`, reason buckets
- [x] 1.10 Implement Channel-B semantic gating for matched desc CE (sentence-transformer; require pinned `revision` when enabled; disable gating + warn/metric if unavailable)
- [x] 1.11 Ensure FN append always in Channel-B B3 target construction
- [x] 1.12 Ensure B2 forward is skipped when no valid matched pairs exist

## 2. Config + Docs Migration
- [x] 2.1 Migrate `configs/stage2_ab/**` to the new typed `stage2_ab` section
- [x] 2.2 Update `docs/training/STAGE2_RUNBOOK.md` to match the new Channel-B stop-neutral + semantic gating contract
- [x] 2.3 Update any example scripts/runbooks that mention `schedule.pattern` or GIoU

## 3. Tests + Validation
- [x] 3.1 Update/add unit tests for `b_ratio` schedule determinism
- [x] 3.2 Update/add unit tests for Channel-A unroll (no detach) and CE anchor split
- [x] 3.3 Update/add unit tests for Channel-B stop-neutral masking (`}` + `<|im_end|>`)
- [x] 3.4 Update/add unit tests for semantic gating behavior (thresholded matched desc masking)
- [x] 3.5 Run `conda run -n ms python -m pytest -q tests/test_stage2_ab_training.py`
- [x] 3.6 Run `openspec validate 2026-02-03-refactor-stage2-ab-training-contract --strict`
