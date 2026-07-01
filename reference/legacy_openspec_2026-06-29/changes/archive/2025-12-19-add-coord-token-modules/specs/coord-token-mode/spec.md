## ADDED Requirements

### Requirement: Coord-token mode is gated
- The system SHALL expose a config flag (e.g., `coord_tokens.enabled`) that, when false or absent, preserves the current numeric geometry workflow unchanged.
- When the flag is true, coord-token handling is enabled across loader, template, and loss helpers.

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

### Requirement: Loss helpers for coord tokens
- Coord-token mode SHALL include helpers that (a) restrict logits to coord tokens, (b) expectation-decode boxes, and (c) provide numeric targets for CE/L1/GIoU.

#### Scenario: Expectation decoding ready for loss
- GIVEN logits over vocab and coord-position indices
- WHEN passed to the helper
- THEN it returns decoded boxes in normalized space and masks for CE/geom losses without needing the template to renormalize.
