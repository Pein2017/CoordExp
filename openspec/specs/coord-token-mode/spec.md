# coord-token-mode Specification

## Purpose
TBD - created by archiving change add-augment-coord-token-support. Update Purpose after archive.
## Requirements
### Requirement: Coord-token augmentation compatibility
The system SHALL support data augmentation on records whose geometries are expressed as coord tokens (`<|coord_k|>`, 0–999), converting to numeric values before augmentation and restoring coord tokens after augmentation when coord-token mode is enabled. Public geometry fields (`bbox_2d`, `poly`) MUST remain token strings after augmentation.

#### Scenario: Coord tokens round-trip through identity augmentation
- **GIVEN** `custom.coord_tokens.enabled` is true
- **AND** an object geometry is provided as coord tokens
- **WHEN** augmentation runs with no-op/identity transforms
- **THEN** the geometry remains tokenized with identical token values after preprocessing/augmentation.

#### Scenario: Coord tokens with active affine augmentation
- **GIVEN** `custom.coord_tokens.enabled` is true
- **AND** an object geometry is provided as coord tokens
- **WHEN** an affine/geometry-changing augmentation runs
- **THEN** the system converts tokens to numeric values for the transform
- **AND** rounds transformed coordinates to the nearest integer (consistent with existing numeric path)
- **AND** converts the transformed integer values back to coord tokens in the output record, keeping geometry fields as tokens.

### Requirement: Numeric path remains unchanged
The augmentation pipeline SHALL preserve current behaviour for datasets that are not in coord-token mode.

#### Scenario: Coord tokens disabled
- **GIVEN** `custom.coord_tokens.enabled` is false
- **WHEN** augmentation runs on a record (token or numeric geometries)
- **THEN** the existing numeric-only augmentation behaviour is used and no token↔int conversion occurs.

### Requirement: Clear failure on invalid coord tokens
The system SHALL emit a clear validation error before augmentation if coord-token inputs fall outside 0–999 or have odd-length coordinate lists.

#### Scenario: Out-of-range token
- **GIVEN** `custom.coord_tokens.enabled` is true
- **AND** a geometry contains `<|coord_1000|>` or any value outside 0–999
- **WHEN** preprocessing runs
- **THEN** a ValueError is raised indicating the token exceeds the allowed range for the current config.

### Requirement: Coord-token mode is gated
- The system SHALL expose a config flag (e.g., `coord_tokens.enabled`) that, when false or absent, preserves the current numeric geometry workflow unchanged.
- When the flag is true, coord-token handling is enabled across loader, template, and coord-token supervision helpers.

#### Scenario: Default path unchanged
- GIVEN coord-token mode is disabled
- WHEN a numeric JSONL sample is loaded
- THEN validation, template normalization, and losses behave exactly as today (pixel → norm1000 in template; text untouched).

### Requirement: Coord token codec utilities
- The system SHALL provide a reusable codec that maps `<|coord_k|>` ↔ int k ↔ normalized float k/999 and builds a coord-token id mask for CE/logit restriction.
- The supported k range SHALL be 0..999 inclusive.

#### Scenario: Token round-trip
- GIVEN a coord token string `<|coord_123|>`
- WHEN passed through the codec
- THEN it returns int 123 and normalized 123/999 (≈ 0.123123...), and converting back yields the same token string.

### Requirement: Token-aware validation
- The loader/validator SHALL accept geometry expressed as coord tokens (arrays of `<|coord_k|>`), provided width/height metadata is present to allow pixel recovery.
- Numeric geometry remains supported.

#### Scenario: Token geometry accepted
- GIVEN a JSONL object with `bbox_2d: ["<|coord_10|>", "<|coord_20|>", "<|coord_200|>", "<|coord_220|>"]` and width/height
- WHEN validated
- THEN it is accepted and numeric equivalents are available for downstream loss.

### Requirement: Template bypass for pre-quantized coords
- In coord-token mode, the template SHALL skip bbox re-normalization when data is already quantized to norm1000 tokens, while keeping the existing normalization path for numeric data.

#### Scenario: No double normalization
- GIVEN coord-token mode is enabled and a sample already encoded as coord tokens
- WHEN the template processes the sample
- THEN it does not rescale bbox values and leaves the coord tokens untouched in text.

### Requirement: Offline numeric→coord-token converter
- The system SHALL provide a CLI/utility to convert numeric JSONL geometry to `<|coord_k|>` tokens using the same rounding rule as the current pipeline (`round(999 * x / max(1, width-1))` for x and `round(999 * y / max(1, height-1))` for y), optionally preserving a numeric copy for losses. Output tokens SHALL lie in 0..999 inclusive (no 1000 bin exists under this convention).

#### Scenario: Conversion produces tokenized JSONL
- GIVEN a numeric JSONL file
- WHEN the converter is run
- THEN the output JSONL has geometry/text coords as coord tokens and includes width/height so pixels can be reconstructed.

### Requirement: Distributional coord-token supervision helpers
- Coord-token mode SHALL include helpers that:
  - restrict logits to the coord-token sub-vocabulary for coord supervision at `<|coord_*|>` positions,
  - compute per-token distribution losses `softCE(Gaussian kernel) + 1D W1(CDF)` on the ordered coord bins,
  - optionally apply a coord-vocab gate loss that penalizes probability mass outside the coord vocab at coord positions.
- When distributional coord-token supervision is enabled, the system SHALL ensure coord-token targets do not contribute to the base full-vocab CE loss by masking coord targets to `ignore_index` (or an equivalent mechanism with zero gradient), while still using the same forward logits to compute coord losses.
- Legacy expectation-decoding and box-level regression losses are not supported. The system SHALL fail fast with a clear error if legacy config keys are provided (e.g. `custom.coord_loss` or `custom.coord_expectation_metrics`), directing users to `custom.coord_soft_ce_w1`.

#### Scenario: Coord tokens supervised with softCE+W1 from one forward
- **GIVEN** coord-token mode is enabled
- **AND** distributional coord-token supervision is enabled
- **WHEN** `Trainer.compute_loss` is called
- **THEN** coord-token targets do not contribute to the base full-vocab CE loss
- **AND** coord `softCE+W1` is computed from the same forward logits restricted to the coord vocab.

#### Scenario: Legacy coord-loss config rejected
- **GIVEN** a YAML config that provides `custom.coord_loss` or `custom.coord_expectation_metrics`
- **WHEN** the training config is loaded
- **THEN** the load fails with a clear error message
- **AND** the error indicates the supported replacement is `custom.coord_soft_ce_w1`.

### Requirement: Coord-gated soft-label losses for Stage-1
When an opt-in Stage-1 coord-loss mode is enabled, the system SHALL support computing coordinate supervision strictly at `<|coord_*|>` token positions by:
- restricting logits to the coord vocab (coord-vocab gate) at those positions, and
- applying a per-token loss of `softCE(Gaussian kernel) + 1D W1(CDF)` against a unimodal soft target over bins.

The system SHALL apply standard full-vocab CE only to non-coord tokens (text + JSON structure) under this mode, to avoid double-supervision at coord positions.

#### Scenario: Coord tokens supervised without full-vocab CE
- **GIVEN** coord-token mode is enabled and a training sample contains `<|coord_k|>` tokens in the assistant JSON
- **AND** Stage-1 coord-gated `softCE+W1` mode is enabled
- **WHEN** the training step computes losses
- **THEN** non-coord tokens contribute to the standard CE loss
- **AND** coord-token positions contribute only to the coord-gated `softCE+W1` loss
- **AND** logits at coord-token positions are restricted to the coord vocab for the coord loss computation.

### Requirement: Single-forward Stage-1 loss composition
When Stage-1 coord-gated `softCE+W1` mode is enabled, the system MUST compute:
- the non-coord CE loss and
- the coord `softCE+W1` loss
from a **single** model forward pass.

The system SHALL ensure coord-token targets do not contribute to the base full-vocab CE loss by masking coord targets to `ignore_index` (or an equivalent mechanism with zero gradient), while still using the same forward logits to compute coord losses.

#### Scenario: No second forward is required
- **GIVEN** a training batch with at least one supervised coord token position
- **AND** Stage-1 coord-gated `softCE+W1` mode is enabled
- **WHEN** `Trainer.compute_loss` is called
- **THEN** the model forward is executed exactly once for that batch
- **AND** the base CE loss ignores coord-token targets
- **AND** coord `softCE+W1` is computed from the same forward logits.

### Requirement: Unimodal soft targets and Wasserstein-1 via CDF
The system SHALL provide utilities to build unimodal soft targets over ordered coord bins and compute 1D Wasserstein-1 loss via CDF differences so that:
- `W1(p, q) = 0` when `p == q`, and
- the loss increases when probability mass shifts farther away in bin order.

#### Scenario: W1 is zero for identical distributions
- **GIVEN** two identical normalized distributions over coord bins
- **WHEN** W1(CDF) is computed
- **THEN** the result is exactly zero (up to numerical tolerance).
