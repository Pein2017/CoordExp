# coord-token-mode Specification

## Purpose
Define the CoordExp coord-token-only contract: how JSONL geometry is represented/consumed as `<|coord_k|>` tokens (0-999), how augmentation/template/training treat those tokens, and which legacy coord-loss knobs are supported.

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

### Requirement: Clear failure on invalid coord tokens
The system SHALL emit a clear validation error before augmentation if coord-token inputs fall outside 0–999 or have odd-length coordinate lists.

#### Scenario: Out-of-range token
- **GIVEN** `custom.coord_tokens.enabled` is true
- **AND** a geometry contains `<|coord_1000|>` or any value outside 0–999
- **WHEN** preprocessing runs
- **THEN** a ValueError is raised indicating the token exceeds the allowed range for the current config.

### Requirement: Coord-token mode is mandatory
- The system SHALL require `custom.coord_tokens.enabled: true`.
- The system SHALL require `custom.coord_tokens.skip_bbox_norm: true` to avoid double normalization on pre-quantized coordinates.
- Configs that disable coord-token mode (or disable skip-bbox-norm) MUST fail fast with actionable errors.

#### Scenario: Coord-token mode cannot be disabled
- GIVEN a training config with `custom.coord_tokens.enabled: false`
- WHEN config parsing or prompt resolution runs
- THEN loading fails with a clear error indicating coord-token mode is mandatory.

### Requirement: Coord token codec utilities
- The system SHALL provide a reusable codec that maps `<|coord_k|>` ↔ int k ↔ normalized float k/999 and builds a coord-token id mask for CE/logit restriction.
- The supported k range SHALL be 0..999 inclusive.

#### Scenario: Token round-trip
- GIVEN a coord token string `<|coord_123|>`
- WHEN passed through the codec
- THEN it returns int 123 and normalized 123/999 (≈ 0.123123...), and converting back yields the same token string.

### Requirement: Token-aware validation
- The loader/validator SHALL accept geometry expressed as coord tokens (arrays of `<|coord_k|>`), provided width/height metadata is present to allow pixel recovery.
- Numeric geometry inputs may be accepted internally but SHALL be converted/emitted under the coord-token contract for model-facing payloads.

#### Scenario: Token geometry accepted
- GIVEN a JSONL object with `bbox_2d: ["<|coord_10|>", "<|coord_20|>", "<|coord_200|>", "<|coord_220|>"]` and width/height
- WHEN validated
- THEN it is accepted and numeric equivalents are available for downstream loss.

### Requirement: Template bypass for pre-quantized coords
- In coord-token mode, the template SHALL skip bbox re-normalization when data is already quantized to norm1000 tokens.

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
- Legacy expectation-decoding and box-level regression losses are not supported:
  - `custom.coord_expectation_metrics` is removed and MUST fail fast with guidance to use distribution-based coord losses (`custom.coord_soft_ce_w1`).
  - `custom.coord_loss` is a deprecated no-op retained for config refactors: configuration parsing MUST NOT raise, and the value MUST be ignored.

#### Scenario: Coord tokens supervised with softCE+W1 from one forward
- **GIVEN** coord-token mode is enabled
- **AND** distributional coord-token supervision is enabled
- **WHEN** `Trainer.compute_loss` is called
- **THEN** coord-token targets do not contribute to the base full-vocab CE loss
- **AND** coord `softCE+W1` is computed from the same forward logits restricted to the coord vocab.

#### Scenario: custom.coord_expectation_metrics rejected
- **GIVEN** a YAML config that provides `custom.coord_expectation_metrics`
- **WHEN** the training config is loaded
- **THEN** the load fails with a clear error message
- **AND** the error indicates the supported replacement is `custom.coord_soft_ce_w1`.

#### Scenario: custom.coord_loss ignored
- **GIVEN** a YAML config that provides `custom.coord_loss` (legacy)
- **WHEN** the training config is loaded
- **THEN** parsing succeeds and the legacy field is ignored.

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


### Requirement: Numeric path remains unchanged
The augmentation pipeline SHALL preserve current behaviour for datasets that are not in coord-token mode.

#### Scenario: Coord tokens disabled
- **GIVEN** `custom.coord_tokens.enabled` is false
- **WHEN** augmentation runs on a record (token or numeric geometries)
- **THEN** the existing numeric-only augmentation behaviour is used and no token↔int conversion occurs.


### Requirement: Coord-token mode is gated
- The system SHALL expose a config flag (e.g., `coord_tokens.enabled`) that, when false or absent, preserves the current numeric geometry workflow unchanged.
- When the flag is true, coord-token handling is enabled across loader, template, and loss helpers.

#### Scenario: Default path unchanged
- GIVEN coord-token mode is disabled
- WHEN a numeric JSONL sample is loaded
- THEN validation, template normalization, and losses behave exactly as today (pixel → norm1000 in template; text untouched).


### Requirement: Loss helpers for coord tokens
- Coord-token mode SHALL include helpers that (a) restrict logits to coord tokens, (b) expectation-decode coords using the ordered coord-token id list (0..999), (c) assemble decoded boxes from coord positions, and (d) provide numeric targets and coord-position masks for CE/L1/GIoU.
- Expectation decoding SHALL support top-k selection where `top_k` may be a fraction (0 < top_k < 1) of the coord vocab or an integer count; fractional values SHALL be converted to a count via `ceil(top_k * 1000)` and clamped to [1, 1000]. Top-k selection SHALL use the highest logits; tie order is implementation-defined.

#### Scenario: Expectation decoding ready for loss
- GIVEN logits over vocab and coord-position indices
- WHEN passed to the helper
- THEN it returns decoded boxes in normalized space and masks for CE/geom losses without needing the template to renormalize.

#### Scenario: Top-k expectation decoding uses coord token ids
- **GIVEN** coord token ids for `<|coord_0|>`..`<|coord_999|>` are available
- **WHEN** expectation decoding runs with `top_k = 0.1`
- **THEN** it uses the top 10% (100 tokens) of coord-token logits mapped to bins 0..999 and returns normalized coords in [0,1].


### Requirement: Loss-scale weighting for coord/text CE
The system SHALL apply coord/text CE weights via ms-swift `loss_scale` when `coord_loss` is enabled and `coord_ce_weight` or `non_coord_ce_weight` differs from 1.0. The `loss_scale` tensor SHALL be built from labels as follows: 0.0 for `labels == -100`, `coord_ce_weight` for coord-token labels, and `non_coord_ce_weight` for all other supervised labels. The tensor length SHALL match `labels` for both padded and packed batches so that ms-swift’s shift (`roll(-1)`) aligns weights to target tokens.

#### Scenario: Weighted coord CE
- **GIVEN** `coord_loss.enabled` is true and `coord_ce_weight=2.0`, `non_coord_ce_weight=1.0`
- **WHEN** a batch contains both coord and text labels
- **THEN** `loss_scale` is attached with 2.0 at coord-label positions, 1.0 at non-coord label positions, and 0.0 at masked positions
- **AND** ms-swift applies the weights via the loss_scale path in CE computation.

#### Scenario: Default weights avoid loss_scale
- **GIVEN** `coord_ce_weight=1.0` and `non_coord_ce_weight=1.0`
- **WHEN** a batch is collated
- **THEN** `loss_scale` is omitted and the default CE loss path is used.

#### Scenario: Packed batch alignment
- **GIVEN** packing is enabled for training or eval
- **WHEN** `loss_scale` is attached
- **THEN** its length matches packed labels length and weights correspond to packed label positions.


### Requirement: Aux losses remain coord-only
Auxiliary coord losses (L1, GIoU, poly mask/smooth) SHALL continue to be computed only on coord-token positions, independent of `loss_scale` values.

#### Scenario: Aux loss excludes text
- **GIVEN** a batch with both coord and text labels
- **WHEN** auxiliary coord losses are computed
- **THEN** only coord-token positions contribute to aux losses and text tokens do not.
