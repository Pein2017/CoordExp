## ADDED Requirements

### Requirement: Encoded-sample caches distinguish geometry-expression modes
Encoded-sample and packing cache identity SHALL include the resolved
geometry-expression mode.

Normative behavior:
- cache fingerprints MUST distinguish at least:
  - `coord_tokens`
  - `norm1000_text`
- cache reuse across these two modes MUST be treated as invalid even when the
  canonical source dataset lineage is otherwise the same,
- resolved metadata artifacts for cache-bearing runs SHOULD expose the same
  geometry-expression identity for auditability.

#### Scenario: Coord-token and raw-text caches do not collide
- **GIVEN** two Stage-1 runs that share the same canonical preset lineage
- **AND** one uses `train.coord.jsonl`
- **AND** the other uses `train.norm.jsonl`
- **WHEN** encoded-sample or static-packing cache keys are computed
- **THEN** the keys differ
- **AND** stale cache reuse across the two runs does not occur.
