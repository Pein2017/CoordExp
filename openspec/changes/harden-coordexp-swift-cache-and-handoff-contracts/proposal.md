## Why

CoordExp-Swift V1 has a functional local training and inference backbone, but
two correctness boundaries still depend too much on operator memory: stale
packing caches can survive semantic Qwen-forward code changes, and
checkpoint-to-inference composition can be assembled manually from loosely
related artifacts. This change hardens those boundaries without starting a
broad refactor.

The priority order is accuracy/precision first, then efficiency, then
simplicity, then extension. The implementation should standardize the
canonical pipeline and lose weight by replacing ad hoc handoff surfaces with
small, auditable contracts.

## What Changes

- Harden packing-cache identity so cache fingerprints include the Qwen
  forward-side source producers that affect packed position, FA2, and model
  forward semantics.
- Preserve the current packed-cache payload shape for this wave; do not
  redesign the cache into lower-level partial artifacts unless later evidence
  shows that broader invalidation is insufficient.
- Add a production-readiness and checkpoint-handoff contract centered on
  `checkpoint_handoff.json`.
- Define production inference as handoff-driven by default, with explicit
  manual path composition treated as research/dev evidence rather than
  canonical production evidence.
- Keep worker count as packing-cache provenance only, not part of semantic
  cache identity.
- Defer `run_training_pipeline` refactoring, loss-plan authority changes, and
  broad docs cleanup to later waves.

No breaking change is intended for valid fresh runs. Existing cache entries may
be invalidated when the added source identities are introduced, which is an
intentional correctness trade-off.

## Capabilities

### New Capabilities

- `coordexp-swift-pack-cache-semantic-identity`: source-code identity,
  provenance, and invalidation behavior for deterministic supervised packing
  caches.
- `coordexp-swift-checkpoint-handoff-readiness`: canonical checkpoint handoff
  manifest and read-only production-readiness validation for training-to-
  inference continuity.

### Modified Capabilities

None. The stable `openspec/specs/` tree does not yet contain archived
CoordExp-Swift specs. This change hardens contracts from the active
`rebuild-coordexp-swift-training-infra` baseline without treating archived or
legacy OpenSpec material as current authority.

## Impact

Affected surfaces:

- `src/training/pack_cache.py` and related packing-cache tests.
- `src/qwen/positions.py`, `src/qwen/fa2.py`, and `src/qwen/forward.py` as
  cache identity determinants, not behavior changes.
- Checkpoint metadata and handoff artifacts under the training artifact stack.
- Inference config/runtime loading paths that distinguish canonical
  production handoff from explicit research/dev manual composition.
- OpenSpec/docs surfaces that define the first two upgrade waves from
  `docs/superpowers/plans/2026-07-06-coordexp-swift-architecture-upgrade-roadmap.md`.

The change must not alter prompt rendering, tokenization, packing order,
position-id computation, FA2 execution, loss math, optimizer behavior,
distributed training semantics, or evaluator metric reduction except where
tests explicitly prove cache identity and handoff contract behavior.
