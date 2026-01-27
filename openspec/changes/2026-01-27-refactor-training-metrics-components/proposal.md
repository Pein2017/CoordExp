# Change: Refactor Training Metrics Components

## Why

CoordExp currently implements training-time metrics, loss augmentation, and monitoring via a mix of:
- collator-side enrichment (`dataset_labels`, `token_types`, `pack_num_samples`, `instability_meta_json`)
- trainer mixins that pop these fields and write into ms-swift `custom_metrics`

This works, but has growing costs for research iteration:
- implicit, stringly-typed batch-key contracts between collators and trainers
- order-sensitive Trainer MRO (mixin ordering changes behavior)
- duplicated logic for logging/reporting and pack-normalized metrics
- low testability (metric/loss code is embedded in `Trainer.compute_loss` with broad `try/except`)

We need small, reusable abstractions so adding new diagnostics or objectives (multi-task / multi-objective) does not require copying patterns across mixins or entangling with collation.

## Scope

In-scope (Stage-1 / standard SFT only):
- The collator wrapper that attaches batch metadata/diagnostics (e.g. token types, packing metadata).
- Trainer mixins in `src/metrics/dataset_metrics.py` used when `custom.trainer_variant != rollout_matching_sft`.
- Aggregate-only metrics logging (no per-dataset buckets).
- Coord loss augmentation (`coord_soft_ce_w1`) and instability monitoring behavior, preserving current semantics.

Out-of-scope:
- Stage-2 `rollout_matching_sft` training-time logging and packing diagnostics (that trainer uses its own collator and logging paths).

Impacted existing capabilities (behavior preserved; no spec changes intended unless explicitly required during implementation):
- `vl-token-type-metrics` (token-type telemetry + packing alignment)
- `coord-token-mode` (coord-token supervision/loss composition)
- `packing-dataset` (pack-size metadata surfaced for per-sample-normalized telemetry)

## What Changes

This change refactors the implementation into minimal, composable components while preserving behavior:

- Introduce a centralized **Batch Extras** contract (stable keys + pop helpers) so extra fields are not forwarded to `model(**inputs)`.
- Extract **metrics reporting** into a thin ms-swift-only adapter (centralized mode resolution + `custom_metrics` writes).
- Extract **metric computation** into pure functions / small modules (aggregate-only; no per-dataset buckets).
- Extract **loss computation** into explicit components with clear failure semantics.
- Refactor collator-side enrichment into composable **enrichers** (dataset meta, token types, instability meta).

## Constraints / Non-Goals

- Aggregate-only metrics remain the default; this change MUST NOT introduce per-dataset buckets.
- Preserve existing metric key names (see `docs/TRAINING_METRICS_AND_LOSSES.md`) and existing batch keys.
- Do not edit upstream HF model files (e.g. `modeling_qwen3_vl.py`).
- Avoid new CLI flags; configuration stays YAML-first.

## Failure Semantics

- **Best-effort** for diagnostics-only code paths (metrics/logging/monitoring):
  - failures MUST NOT block training
  - expected skip conditions (missing required inputs, known alignment mismatches) remain per-batch skips
  - unexpected exceptions MUST emit a warning once per diagnostic (and MAY disable the diagnostic to avoid repeated overhead/log spam)
- **Fail-fast** for objective-changing code paths (loss composition / label-masking):
  - if an enabled loss component fails, training MUST raise to avoid silently changing the training objective

## Impact

Expected impact is internal-only (architecture):
- no config migrations required
- no change to metric key names
- no change to dataset formats

Potential risks:
- subtle ordering differences in mixin execution if glue code changes
- accidentally changing which labels are used for token-accuracy metrics when coord-loss masking is enabled

Mitigations:
- keep existing public entrypoints (`build_dataset_metrics_collator`, mixin class names)
- add focused unit tests around packing, masking, and metrics key emission

## Validation

Refactor parity is validated via:
- Unit tests (fast): token-type alignment (packed + padded), pack-size metadata, grad-accum loss scaling, coord loss masking/composition.
- Key parity: ensure the emitted metric keys match `docs/TRAINING_METRICS_AND_LOSSES.md` and no per-dataset buckets appear.
- NaN safety: ensure all diagnostics skip cleanly on empty supervision and do not introduce NaNs into metrics.

## Migration Strategy

Backwards compatibility is preserved by:
- keeping existing top-level batch keys (`token_types`, `pack_num_samples`, `instability_meta_json`, etc.)
- keeping existing mixin names and wiring in `src/sft.py`
- moving logic behind these stable interfaces into new internal modules
