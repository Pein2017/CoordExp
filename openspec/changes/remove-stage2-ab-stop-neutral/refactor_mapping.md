# Refactor Alignment (Post-Refactor `src/` Layering)

This change set is intended to preserve **public behavior intent** while aligning spec language and references to the post-refactor codebase as defined by `openspec/changes/src-ambiguity-cleanup-2026-02-11`.

This file is *mapping-only*: it records how older internal names/paths map onto the refactored layering and neutral contracts.

## Spec Terminology / Metric Name Alignment (Intent Preserved)

- Stop-neutral toggles vs fixed contract behavior:
  - Old wording referenced a "stop-neutral runtime mode".
  - Current codebase behavior is fixed-by-contract stop-neutral masking (no `stage2_ab.channel_b.stop_neutral` knob in the typed schema).
  - Updated wording: "stop-neutral masking" (and its removal) rather than a runtime toggle.

- Truncation-tail KPI naming:
  - Old design text used a placeholder key `truncation_tail_rate`.
  - Updated to the canonical rollout truncation indicator name: `rollout/parse_truncated_rate`.
  - Notes: acceptance intent is unchanged: reduce long-tail truncation/over-generation.

## Module / Contract Mapping (Pre-Refactor -> Current)

- Stage-2 AB boundary decomposition:
  - Entry surface remains: `src/trainers/stage2_ab_training.py`
  - Orchestration components extracted to:
    - `src/trainers/stage2_ab/scheduler.py`
    - `src/trainers/stage2_ab/executors.py`
    - `src/trainers/stage2_ab/async_queue.py`

- Rollout parsing/matching contracts consumed by Stage-2:
  - Public contract modules (import-light, Stage-2 safe):
    - `src/trainers/rollout_matching/contracts.py`
    - `src/trainers/rollout_matching/parsing.py`
    - `src/trainers/rollout_matching/matching.py`
    - `src/trainers/rollout_matching/packing.py`
    - `src/trainers/rollout_matching/telemetry.py`
  - Boundary rule: Stage-2 AB SHOULD NOT import underscore-private rollout helpers from trainer monoliths.

- Metrics contracts (neutral `src.metrics` abstraction):
  - Canonical payload contract:
    - `src/metrics/payload_contract.py` (`schema_version=1`)
  - Stop-neutral diagnostic key currently emitted by the baseline contract:
    - `stage2_ab/channel_b/stop_neutral/N_skip`
  - This change set removes stop-neutral masking and replaces stop-neutral skip accounting with a closure-supervision drop counter:
    - `stage2_ab/channel_b/stop_neutral/N_skip` -> `stage2_ab/channel_b/closure_supervision/N_drop`
  - Metrics are required to be emitted as global aggregates after grad-accum aggregation and DDP all-reduce.

- Coord utilities vs dataset-layer geometry:
  - Canonical coord-token helpers (single source of truth):
    - `src/coord_tokens/codec.py` (coord-token regex + encode/decode for `<|coord_k|>`)
  - Compatibility surface (may be a shim after ambiguity cleanup refactors):
    - `src/common/geometry/coord_utils.py` (re-exported via `src.common.geometry`)
  - Dataset-owned transform/resize semantics:
    - `src/datasets/geometry.py`
  - Boundary rule: stop/closure marker identification and masking MUST NOT introduce dataset-layer dependencies.

## No Functional Semantics Changed By This Alignment

These refactor-alignment edits only adjust naming and contract/module references to match the refactored layering.
The original behavioral intent and acceptance criteria of the change set remain unchanged.
