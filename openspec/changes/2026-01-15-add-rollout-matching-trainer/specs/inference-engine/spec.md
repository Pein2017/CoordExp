## MODIFIED Requirements

### Requirement: Unified output schema
Each output line in `pred.jsonl` SHALL contain `gt` and `pred` arrays of objects with fields `type` (`bbox_2d` or `poly`), `points` (absolute pixel coordinates), `desc` (label string), and `score` fixed at 1.0, plus top-level `width`, `height`, `image`, `mode`, optional `coord_mode` for trace/debug, `raw_output`, and an `errors` list (empty when none). Legacy mixed-format fields (e.g., raw norm `predictions`/dual schemas) SHALL NOT be emitted.

#### Scenario: Successful sample output
- **WHEN** a sample is processed without errors
- **THEN** the JSONL line includes `gt` and `pred` arrays with pixel `points`, `desc`, `type`, `score:1.0`, along with `width`, `height`, `image`, `mode`, and an empty `errors` list.

#### Scenario: Error sample output
- **WHEN** a sample fails validation (e.g., missing height)
- **THEN** the JSONL line contains an `errors` list describing the issue, `pred` is empty, and processing continues for subsequent samples.

## REMOVED Requirements

### Requirement: Line tolerance without metrics
**Reason**: `line` geometries are deprecated and not present in the supported dataset corpus; keeping them in the inference artifact format increases downstream ambiguity without providing value.

