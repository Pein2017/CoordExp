## 1. Stage-2 AB Contract Alignment (Channel-B Unified Semantics)
- [x] Update `src/trainers/stage2_ab_training.py` to enforce unified Channel-B CE mask policy:
  - [x] matched prefix: structure CE on, desc CE off, coord CE off
  - [x] FP prefix: structure/desc/coord CE all off
  - [x] FN-injected spans: structure+desc CE on, coord CE off
- [x] Update Channel-B geometry accumulation to include matched + FN-injected objects, and exclude FP geometry loss.
- [x] Ensure Channel-B stop-neutral masking applies to:
  - [x] outermost top-level close brace used by injection
  - [x] `<|im_end|>`
- [x] Keep `reordered_gt_sft` behavior available only as explicit legacy/ablation path (not default contract behavior).

## 2. Deterministic FN Injection + Key Allocation
- [x] Implement deterministic outermost-brace resolution in Channel-B construction (brace-depth or equivalent deterministic parser).
- [x] Implement deterministic FN insertion before that outermost close brace with comma insertion rules:
  - [x] insert comma iff retained prefix body already has object entries
  - [x] otherwise inject without leading comma
- [x] Update FN key allocation to derive `start_id` from all retained `object_N` keys, including keys from invalid/dropped entries.
- [x] Ensure malformed keys are ignored (do not crash; do not reserve ids).

## 3. Rollout Prefix Key-Accounting Clarification
- [x] Update `src/trainers/rollout_matching_sft.py` prefix metadata so `max_object_index_in_prefix` reflects retained `object_N` keys independent of strict-validation survival.
- [x] Keep ordering/tokenization invariants unchanged (no pre-cut decode+re-encode).

## 4. Dataset Runtime Contract Hardening
- [x] Update `src/datasets/utils.py` geometry extraction validation to fail fast on:
  - [x] missing geometry
  - [x] multiple geometry fields
  - [x] invalid bbox/poly arity
- [x] Update `src/datasets/builders/jsonlines.py` group payload emission to reject geometry-less/ambiguous objects instead of serializing partial objects.
- [x] Ensure `desc` remains non-empty string and enforce exact geometry invariants at runtime builder boundary.

## 5. Config + Docs Alignment
- [x] Align relevant Stage-2 configs under `configs/stage2_ab/**` so Unified Channel-B semantics are the default path.
- [x] Update `docs/training/STAGE2_RUNBOOK.md` to reflect final Channel-B CE/geometry/stop-neutral contracts.
- [x] Update `docs/training/METRICS_LOSSES.md` with corrected Channel-B geometry and masking semantics.
- [x] Update `docs/data/JSONL_CONTRACT.md` references if runtime fail-fast behavior needs clarifying language.

## 6. Tests (Acceptance + Regression)
- [x] Extend/add Stage-2 tests for non-reordered Channel-B CE spans:
  - [x] matched structure supervised + matched desc masked
  - [x] FP fully masked
  - [x] FN struct+desc supervised
- [x] Add test for collision-safe key numbering with invalid high-index prefix keys.
- [x] Add test for deterministic brace-depth + comma insertion behavior.
- [x] Add test ensuring stop-neutral masks target the same outermost close brace used by FN injection plus `<|im_end|>`.
- [x] Add dataset tests for fail-fast object invariants:
  - [x] desc-only object rejected
  - [x] multi-geometry object rejected
  - [x] invalid bbox/poly arity rejected
- [x] Run regression suites for affected trainers/datasets and record outcomes.

## 7. Pre-Implementation Governance Checks
- [x] Verify OpenSpec deltas and tasks remain in sync after any scope adjustment.
- [x] Keep implementation changes scoped to this change (no unrelated trainer refactors).
- [x] Re-run `openspec status --change align-stage2-full-idea-contracts-2026-02-09 --json` before code implementation starts.
