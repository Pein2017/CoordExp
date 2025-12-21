# vl-token-type-metrics Specification

## Purpose
TBD - created by archiving change add-token-type-metrics-packed. Update Purpose after archive.
## Requirements
### Requirement: Aggregate token-type telemetry
The system SHALL log aggregate (not per-dataset) metrics: `loss`, `token_acc`, and per-token-type (`desc`, `coord`, `format`) accuracy. Per-dataset buckets MUST NOT be emitted. Metrics computation MUST be NaN-safe: when a sample or batch has zero supervised tokens, the metric is skipped or logged with zero count so no NaN appears in logs.

#### Scenario: Empty supervision safe
- WHEN a batch arrives where all labels are `-100`
- THEN token-type metrics are skipped or reported with zero denominators without producing NaN values
- AND no per-dataset metric keys are added

### Requirement: Stable metric key names (shared parity)
The system SHALL emit stable aggregate metric keys (no prefix): `loss`, `token_acc`, and per-type keys `desc_token_acc`, `coord_token_acc`, `format_token_acc`. These keys SHALL remain aligned with the upstream Qwen3-VL implementation when both are enabled to ease cross-repo dashboarding.

#### Scenario: Key parity
- WHEN token-type metrics are enabled in CoordExp
- THEN the emitted keys match the names above and align with the Qwen3-VL metric names for the same feature

### Requirement: Packing-compatible alignment
The system SHALL support packed batching by computing token types per raw sample before packing, concatenating them in the same order as packed labels, and emitting metrics only when lengths match. If alignment fails, metrics for that batch SHALL be skipped with a debug notice (no crash).

#### Scenario: Two samples packed
- GIVEN two samples of lengths 80 and 40 packed into one sequence of length 120
- WHEN token types are concatenated in the pack order
- THEN token-type metrics use the concatenated types aligned to packed labels

#### Scenario: Misalignment skip
- WHEN concatenated token_types length differs from packed labels length
- THEN the batch’s token-type metrics are skipped and a debug log notes the mismatch

### Requirement: Configurable include/exclude
The system SHALL expose a `token_type_metrics` config with fields `enabled`, `include`, and `exclude`; defaults target `lvis` only. Metrics are computed only for samples whose normalized dataset label is in `include` and not in `exclude`.

#### Scenario: Include-only lvis
- GIVEN defaults `include=["lvis"]` and `exclude=[]`
- WHEN a batch contains lvis and coco samples
- THEN only lvis samples contribute to token-type metrics

#### Scenario: Packed include/exclude
- GIVEN a packed batch containing samples from lvis and coco
- AND include=["lvis"], exclude=[]
- WHEN the batch is processed
- THEN token types for lvis tokens are computed and concatenated, while coco token positions are IGNORE so aggregate metrics count only lvis tokens

### Requirement: Eval parity with packing toggle
The system SHALL support token-type metrics in eval for both padded and packed modes. By default (eval packing off) padded eval batches are supported; when eval packing is enabled, the packing-compatible alignment logic SHALL be reused.

#### Scenario: Eval padded default
- GIVEN eval_packing is false
- WHEN eval runs
- THEN token-type metrics are computed on padded batches

#### Scenario: Eval with packing
- GIVEN eval_packing is true and packing is enabled
- WHEN eval runs
- THEN token-type metrics use packing alignment and skip on mismatch rather than failing

### Requirement: Graceful fallback
If required inputs are missing (assistant payload, tokenizer, or attention/labels), the system SHALL skip token-type metrics for that batch and continue training without error.

#### Scenario: Missing payload
- WHEN a batch lacks `assistant_payload`
- THEN token-type metrics are skipped and training proceeds
