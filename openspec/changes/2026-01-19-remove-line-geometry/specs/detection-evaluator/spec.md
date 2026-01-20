## MODIFIED Requirements

### Requirement: Parsing and coordinate handling
- The evaluator SHALL reuse the shared coord-processing module (via `src/common/geometry`/`src/common/schemas`) used by inference/visualization, supporting coord tokens or ints in 0–999 with one geometry per object.
- Supported geometries SHALL be limited to `bbox_2d` and `poly`.
- Any object containing `line` or `line_points` SHALL be treated as invalid geometry and dropped with an `invalid_geometry`-style counter.

#### Scenario: Line geometry is rejected
- GIVEN a prediction object that contains `line`
- WHEN the evaluator ingests the predictions JSONL
- THEN that object is dropped, increments an `invalid_geometry` counter, and does not contribute to matches/metrics.

