## Context

`progress/full_idea.md` defines a stricter Stage-2 AB contract than what is currently encoded by implementation + main specs.
The main drifts are:
- Channel-B CE/geometry semantics are partially inconsistent with unified one-pass policy.
- FN key allocation in code can under-estimate the next key when high-index `object_N` entries are invalid/dropped.
- Deterministic FN injection + stop-neutral masking need a shared outermost-brace anchor.
- Dataset object emission still allows geometry-less objects in runtime builder paths.

This change is cross-cutting across trainer semantics, parser/append helpers, dataset builder validation, and governance artifacts.
Because this repository is in OpenSpec workflow migration, artifact order matters: proposal -> specs + design -> tasks -> implementation.

## Goals / Non-Goals

**Goals:**
- Make Channel-B normative behavior match updated `full_idea`:
  - matched struct-on/desc-off,
  - FP fully CE+geometry neutral,
  - FN-injected struct+desc supervised,
  - geometry on matched + FN only.
- Make FN injection and stop-neutral masking deterministic via the same outermost-brace resolution.
- Make FN key allocation collision-safe using all retained `object_N` keys, not only valid matched objects.
- Enforce runtime object contract invariants (non-empty `desc`, exactly one geometry, valid arity) before assistant payload emission.
- Keep traceability via delta specs + tasks before touching implementation.

**Non-Goals:**
- No new architecture branch, no RL loop, no upstream HF model file edits.
- No new CLI flags.
- No rewrite of rollout generation backend contracts beyond scoped clarifications.

## Decisions

### Decision 1: Treat unified Channel-B as normative default; keep reordered path as legacy/ablation only
- Rationale: current direction is explicit in `full_idea`; keeping old behavior as default would continue contract drift.
- Alternative considered: keep dual defaults for backward compatibility.
- Why rejected: dual defaults complicate reproducibility and interpretation of reported metrics.

### Decision 2: Use deterministic brace-depth resolution as the single source of truth for both FN injection and stop-neutral masking
- Rationale: a shared anchor prevents subtle mismatches where injection and CE masking target different braces.
- Alternative considered: token-level "last brace" heuristic.
- Why rejected: tokenizer-dependent and brittle with fused punctuation tokens.

### Decision 3: Compute FN `start_id` from retained prefix key scan, independent of strict-validation survival
- Rationale: key-space collisions are a text-level correctness issue; validity for matching and key reservation are different concerns.
- Alternative considered: derive max key from valid parsed objects only.
- Why rejected: can reuse already-present key ids when high-index objects are invalid/dropped.

### Decision 4: Enforce strict object invariants at runtime builder boundary (fail-fast)
- Rationale: silently emitting malformed objects harms training reproducibility and corrupts downstream assumptions.
- Alternative considered: best-effort normalization with silent drops.
- Why rejected: hides data quality errors and produces nondeterministic sample quality.

### Decision 5: Keep rollout-matching-sft delta narrowly scoped to ambiguity fixes
- Rationale: rollout main spec already contains key-allocation intent; observed drift is primarily implementation.
- Alternative considered: broad rollout spec rewrite.
- Why rejected: unnecessary scope expansion and higher merge/conflict risk.

## Risks / Trade-offs

- **[Risk] Stricter data validation can break previously tolerated datasets** -> Mitigation: provide actionable error messages + preflight validation tests.
- **[Risk] Brace-depth/token-position mapping may be fragile for tokenizer edge cases** -> Mitigation: add deterministic unit tests for fused punctuation and nested braces.
- **[Risk] Loss-mix change may shift historical metric baselines** -> Mitigation: document contract change in runbook/metrics docs and compare with pinned pre-change checkpoints.
- **[Risk] Cross-change interaction with async actor-learner work** -> Mitigation: keep this change scoped to semantic correctness and avoid overlapping queue/sync logic edits.

## Migration Plan

1. Finalize delta specs (`stage2-ab-training`, `rollout-matching-sft`, `public-data-pipeline`) and this design.
2. Create tasks mapped one-to-one to spec deltas and required tests.
3. Implementation order:
   - `src/trainers/rollout_matching_sft.py`: prefix key accounting helper contract.
   - `src/trainers/stage2_ab_training.py`: CE/geometry masking policy, FN injection + brace targeting.
   - `src/datasets/utils.py` and `src/datasets/builders/jsonlines.py`: fail-fast object/geometry invariants.
   - docs/config alignment.
4. Validation order:
   - unit tests for CE spans, key collision safety, brace/comma behavior, dataset invariants.
   - existing Stage-2 AB and rollout unit suites as regression checks.
5. Rollback strategy:
   - revert implementation commits while preserving spec artifacts,
   - temporarily run legacy reordered behavior explicitly as an opt-in ablation path if urgent unblock is needed.

## Open Questions

- Runtime builder invariants are scoped under the `public-data-pipeline` delta in this change to keep conversion/runtime contract ownership unified.
- Do any active datasets intentionally contain geometry-less objects that should be filtered upstream rather than fail-fast in conversion/runtime?
- Should we add a dedicated metric for per-batch FN-injected geometry contribution to make post-migration comparisons easier?
