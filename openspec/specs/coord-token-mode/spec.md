# coord-token-mode Specification

## Purpose
TBD - created by archiving change add-augment-coord-token-support. Update Purpose after archive.
## Requirements
### Requirement: Coord-token augmentation compatibility
The system SHALL support data augmentation on records whose geometries are expressed as coord tokens (`<|coord_k|>`, 0–999), converting to numeric values before augmentation and restoring coord tokens after augmentation when coord-token mode is enabled. Public geometry fields (`bbox_2d`, `poly`, `line`) MUST remain token strings after augmentation.

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
- When the flag is true, coord-token handling is enabled across loader, template, and loss helpers.

#### Scenario: Default path unchanged
- GIVEN coord-token mode is disabled
- WHEN a numeric JSONL sample is loaded
- THEN validation, template normalization, and losses behave exactly as today (pixel → norm1000 in template; text untouched).

### Requirement: Coord token codec utilities
- The system SHALL provide a reusable codec that maps `<|coord_k|>` ↔ int k ↔ normalized float k/1000 and builds a coord-token id mask for CE/logit restriction.
- The default supported k range SHALL be 0..999 inclusive; the codec MAY optionally allow 1000 when configured.

#### Scenario: Token round-trip
- GIVEN a coord token string `<|coord_123|>`
- WHEN passed through the codec
- THEN it returns int 123 and normalized 0.123, and converting back yields the same token string.

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
- The system SHALL provide a CLI/utility to convert numeric JSONL geometry to `<|coord_k|>` tokens using the same rounding rule as the current pipeline (`round(x/width*1000)`), optionally preserving a numeric copy for losses. Default output tokens SHALL lie in 0..999 inclusive; if any value would round to 1000 the converter SHALL either allow it via a flag or raise clearly.

#### Scenario: Conversion produces tokenized JSONL
- GIVEN a numeric JSONL file
- WHEN the converter is run
- THEN the output JSONL has geometry/text coords as coord tokens and includes width/height so pixels can be reconstructed.

### Requirement: Loss helpers for coord tokens
- Coord-token mode SHALL include helpers that (a) restrict logits to coord tokens, (b) expectation-decode boxes, and (c) provide numeric targets for CE/L1/GIoU.

#### Scenario: Expectation decoding ready for loss
- GIVEN logits over vocab and coord-position indices
- WHEN passed to the helper
- THEN it returns decoded boxes in normalized space and masks for CE/geom losses without needing the template to renormalize.

