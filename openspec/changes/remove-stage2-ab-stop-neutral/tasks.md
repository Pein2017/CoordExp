## 0. Prerequisites (Refactor First)

- [x] 0.1 Land the schema/module refactor work that clarifies canonical helper ownership and leaves compatibility shims as needed (e.g., `openspec/changes/src-ambiguity-cleanup-2026-02-11`).
- [ ] 0.2 When implementing this change, avoid introducing new parallel helpers/regexes; reuse canonical helper surfaces under `src/common/*` and canonical coord-token helpers under `src/coord_tokens/codec.py` where applicable.

## 1. Contract and Config Update

- [ ] 1.1 Remove stop-neutral masking from Stage-2 AB Channel-B (no runtime toggle; fixed contract behavior changes).
- [ ] 1.2 Remove/adjust stop-neutral-specific counters and code branches that are no longer normative (including `stage2_ab/channel_b/stop_neutral/N_skip`).
- [ ] 1.3 Add replacement counter for closure marker resolution failures: `stage2_ab/channel_b/closure_supervision/N_drop` (global aggregate via the neutral metrics payload).
- [ ] 1.4 Verify FP-neutral masking logic remains unchanged for unmatched predicted objects.
- [ ] 1.5 Refresh `configs/stage2_ab/**` to the latest contract (remove any stop-neutral keys/comments; no legacy support).

## 2. Channel-B CE Mask Implementation

- [ ] 2.1 Modify Channel-B CE mask builder to keep supervision on top-level `}` and `<|im_end|>`.
- [ ] 2.2 Keep deterministic outermost-brace identification for injection/structure-token handling.
- [ ] 2.3 Ensure no regression in FN append/key allocation logic while removing stop-neutral masking.
- [ ] 2.4 Add global aggregated rollout truncation metric in Stage-2 AB logs: `rollout/parse_truncated_rate`.

## 3. Tests

- [ ] 3.1 Replace stop-neutral assertions with closure-supervision assertions in Stage-2 AB tests.
- [ ] 3.2 Add regression test showing FP spans remain CE/geometry neutral after stop-neutral removal.
- [ ] 3.3 Run targeted tests: `conda run -n ms python -m pytest tests/test_stage2_ab_training.py`.
- [ ] 3.4 Run boundary regression test: `conda run -n ms python -m pytest tests/test_stage2_rollout_import_boundaries.py`.

## 4. Docs and Runbook Alignment

- [ ] 4.1 Update `docs/training/STAGE2_RUNBOOK.md` to remove stop-neutral contract wording and document the new Channel-B stop/closure supervision behavior (`}` + `<|im_end|>` are supervised; FP-neutral remains).
- [ ] 4.2 Update `docs/training/METRICS_LOSSES.md` to remove `stage2_ab/channel_b/stop_neutral/N_skip`, add `stage2_ab/channel_b/closure_supervision/N_drop`, and document global aggregation semantics for Stage-2 AB training metrics (including `rollout/parse_truncated_rate`).
- [x] 4.3 Update `progress/full_idea.md` references where stop-neutral is described as default policy.
- [x] 4.4 Sync proposal/design/spec wording with the same rationale (rollout-length inflation without reliable TP-hit gain under stop-neutral).

## 5. Lightweight Validation (No Baseline Gate)

- [ ] 5.1 Run a bounded Stage-2 AB smoke under the updated contract (stop/closure supervised; FP-neutral unchanged).
- [ ] 5.2 Verify audit metrics keys are present and have sane ranges:
  - `stage2_ab/channel_b/closure_supervision/N_drop` is `>= 0`,
  - `rollout/parse_truncated_rate` is in `[0, 1]`.
- [ ] 5.3 Record reproducibility metadata (config path, run name, seed, output artifacts, git SHA) in change notes.
