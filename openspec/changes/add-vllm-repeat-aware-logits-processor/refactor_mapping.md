# Refactor Alignment (Post-Refactor `src/` Layering)

This change set is intended to preserve **public behavior** while aligning spec language and references to the post-refactor codebase as defined by `openspec/changes/src-ambiguity-cleanup-2026-02-11`.

This file is *mapping-only*: it records how older internal names/paths map onto the refactored layering and neutral contracts.

## Naming / Key Updates (Intent Preserved)

- Change id + directory name:
  - Old: `add-vllm-repeat-awared-logits-processor`
  - New: `add-vllm-repeat-aware-logits-processor`
  - Notes: typo fix only; the change intent is unchanged.

- vLLM processor label:
  - Old spec text used `repeat_awared`.
  - New spec text describes **repeat-aware processing controlled by** `custom.extra.rollout_matching.repeat_terminate`, without requiring a specific string identifier.
  - Notes: activation semantics (startup-time server injection; no request-time dependency) remain the same.

- Telemetry keys (repeat-aware activation + trigger audit):
  - Old: `rollout/repeat_awared_active` -> New: `rollout/repeat_terminate_active`
  - Old: `rollout/repeat_awared_triggered_sequences` -> New: `rollout/repeat_terminate_triggered_sequences`
  - Notes: keys are entries in `trainer_metrics.metrics` per the neutral metrics payload contract (`src/metrics/payload_contract.py`, `schema_version=1`). These keys are expected to be emitted as global aggregates after grad-accum aggregation and DDP all-reduce.

- Global tail metrics (distributed):
  - `rollout/parse_truncated_rate`: global sums (not mean of rank-local ratios)
  - `rollout/gen_new_tokens_p99`: simple global proxy via all-reduce max over rank-local p99

## Module / Contract Mapping (Pre-Refactor -> Current)

- Rollout parsing/matching/packing contracts:
  - Old pattern (pre-refactor): consumers importing helper logic directly from trainer monoliths (often underscore-private).
  - Current public contract modules (import-light, Stage-2 safe):
    - `src/trainers/rollout_matching/contracts.py`
    - `src/trainers/rollout_matching/parsing.py`
    - `src/trainers/rollout_matching/matching.py`
    - `src/trainers/rollout_matching/packing.py`
    - `src/trainers/rollout_matching/telemetry.py`
  - Boundary rule: Stage-2 AB SHOULD consume rollout helpers through these public modules only.

- Stage-2 AB boundary decomposition:
  - Entry surface remains: `src/trainers/stage2_ab_training.py`
  - Orchestration components extracted to:
    - `src/trainers/stage2_ab/scheduler.py`
    - `src/trainers/stage2_ab/executors.py`
    - `src/trainers/stage2_ab/async_queue.py`

- Metrics contracts (neutral `src.metrics` abstraction):
  - Old (pre-ambiguity-cleanup): trainer-side mixins commonly imported from `src/metrics/dataset_metrics.py`.
  - Current canonical split:
    - trainer-side mixins live in `src/trainers/metrics/mixins.py`,
    - neutral payload/reporting contracts live under `src/metrics/*`:
      - `src/metrics/payload_contract.py` (neutral trainer-metrics payload contract),
      - `src/metrics/reporter.py` (reporter helpers).
  - Compatibility shims:
    - `src/metrics/dataset_metrics.py` re-exports `src.trainers.metrics.mixins`,
    - `src/trainers/metrics/reporter.py` re-exports `src.metrics.reporter`.

- Coord utilities vs dataset-layer geometry:
  - Canonical coord-token helpers (single source of truth):
    - `src/coord_tokens/codec.py` (coord-token regex + encode/decode for `<|coord_k|>`)
  - Compatibility surface (may be a shim after ambiguity cleanup refactors):
    - `src/common/geometry/coord_utils.py` (re-exported via `src.common.geometry`)
  - Dataset-owned transform/resize semantics:
    - `src/datasets/geometry.py`
  - Boundary rule: repeat-aware processing and rollout parsing MUST NOT introduce dataset-layer dependencies.

## No Functional Semantics Changed By This Alignment

These refactor-alignment edits only adjust naming and contract/module references to match the refactored layering.
The original behavioral intent and acceptance criteria of the change set remain unchanged.
