## MODIFIED Requirements
### Requirement: Loss helpers for coord tokens
- Coord-token mode SHALL include helpers that (a) restrict logits to coord tokens, (b) expectation-decode coords using the ordered coord-token id list (0..999), (c) assemble decoded boxes from coord positions, and (d) provide numeric targets and coord-position masks for CE/L1/GIoU.
- Expectation decoding SHALL support top-k selection where `top_k` may be a fraction (0 < top_k < 1) of the coord vocab or an integer count; fractional values SHALL be converted to a count via `ceil(top_k * 1000)` and clamped to [1, 1000]. Top-k selection SHALL use the highest logits; tie order is implementation-defined.

#### Scenario: Top-k expectation decoding uses coord token ids
- **GIVEN** coord token ids for `<|coord_0|>`..`<|coord_999|>` are available
- **WHEN** expectation decoding runs with `top_k = 0.1`
- **THEN** it uses the top 10% (100 tokens) of coord-token logits mapped to bins 0..999 and returns normalized coords in [0,1].
