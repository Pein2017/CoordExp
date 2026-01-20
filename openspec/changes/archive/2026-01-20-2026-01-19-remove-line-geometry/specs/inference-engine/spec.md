## MODIFIED Requirements

### Requirement: Unified output schema
Each output line in `pred.jsonl` SHALL contain `gt` and `pred` arrays of objects with fields:
- `type` (`bbox_2d` or `poly` only),
- `points` (absolute pixel coordinates),
- `desc` (label string),
- `score` fixed at 1.0,

plus top-level `width`, `height`, `image`, `mode`, optional `coord_mode` for trace/debug, `raw_output`, and an `errors` list (empty when none).

Any `line` geometry encountered during parsing/validation SHALL be treated as invalid geometry and MUST NOT be emitted in `pred` outputs.

#### Scenario: Line in raw output is dropped
- **GIVEN** a generated JSON object that includes a `line` geometry
- **WHEN** the inference engine parses and validates predictions
- **THEN** the `line` object is excluded from `pred` and an error/counter reflects invalid geometry.

