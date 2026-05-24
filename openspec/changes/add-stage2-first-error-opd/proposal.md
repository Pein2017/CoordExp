## Why

Online compact-full Stage-2 training needs to correct the model on the
self-prefix states it actually visits without reverting to broad Stage-1 SFT.
The current residual-set adapter can accidentally treat a same-role live token
as positive even when it is not one of the oracle valid actions, which can let
format or type drift reinforce itself as training steps increase.

## What Changes

- Add a first-error OPD contract for Channel-B residual-set training: rollout
  prefixes may be used as context, but the supervised target at a correction
  point comes from oracle `ValidAction` records, not from the live token.
- Allow multiple-positive marginal likelihood only at genuinely ambiguous trie
  positions where multiple valid next-token actions remain reasonable under the
  current residual state.
- Collapse to singleton strict CE as soon as the teacher-forced prefix selects a
  concrete branch; deterministic schema, stop, and object-internal singleton
  positions are normal CE cases.
- Keep malformed or unlocatable spans masked/dropped as context, while emitting
  first-error diagnostics that distinguish oracle corrections from malformed
  fallback/drop handling.
- Add stable diagnostics for ambiguous-token targets, singleton/strict targets,
  corrected live-token mismatches, and coordinate ambiguity exposure.

## Capabilities

### New Capabilities

- `stage2-first-error-opd`: Defines online self-prefix first-error correction,
  ambiguity-scoped multiple-positive trie supervision, strict CE after branch
  collapse, live-token mismatch handling, and diagnostics.

### Modified Capabilities

- `stage2-ab-training`: Adds first-error OPD semantics to the active online
  residual-state trie Channel-B path.
- `teacher-forcing-unified-loss-registry`: Allows target atoms whose supervised
  selected token intentionally differs from the live self-prefix token when the
  atom provenance marks it as a first-error correction.

## Impact

- Affected code:
  - `src/trainers/stage2_two_channel/teacher_forcing_adapter.py`
  - `src/trainers/teacher_forcing/modules/residual_set_correction.py`
  - `src/training/teacher_forcing/validation.py`
  - Stage-2 residual-set tests and config contract tests
- Affected configs/docs:
  - `configs/stage2_two_channel/prod/*`
  - `configs/stage2_two_channel/smoke/*`
  - `docs/training/STAGE2_RUNBOOK.md`
  - `openspec/specs/stage2-ab-training/spec.md`
  - `openspec/specs/teacher-forcing-unified-loss-registry/spec.md`
- No new CLI flags or production dependencies are introduced.
