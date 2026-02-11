# vl-token-type-metrics Spec Delta

This is a delta spec for change `src-ambiguity-cleanup-2026-02-11`.

## ADDED Requirements

### Requirement: Coord-token identification uses canonical helper
Token-type telemetry MUST identify coord tokens using the canonical coord-token helper implementation (single source of truth) so coord-token classification is consistent with other consumers (datasets, parsing, evaluation).
Implementations MUST NOT define parallel coord-token regex/constants that can drift.
Token format/range semantics are owned by the `coord-utils` capability contract; this capability MUST reuse those semantics and MUST NOT redefine conflicting coord-token validity rules.

#### Scenario: Coord tokens are classified as coord type
- **GIVEN** an assistant payload containing a coordinate token string `<|coord_5|>`
- **WHEN** token-type telemetry computes per-token types
- **THEN** the token corresponding to `<|coord_5|>` is classified as `coord`.
