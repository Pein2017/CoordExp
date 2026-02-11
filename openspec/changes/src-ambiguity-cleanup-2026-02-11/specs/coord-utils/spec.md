# coord-utils Spec Delta

This is a delta spec for change `src-ambiguity-cleanup-2026-02-11`.

## ADDED Requirements

### Requirement: Canonical coord-token helpers are single-source-of-truth
Coord token detection, encode, and decode helpers (including the underlying coord-token regex) SHALL have a single canonical implementation.
All consumers (datasets, token-type telemetry, inference/eval parsing, visualization) MUST reuse the canonical helpers instead of defining parallel regex/constants.

#### Scenario: Coord-token detection is consistent across consumers
- **GIVEN** a string token `<|coord_12|>` and a non-token string `coord_12`
- **WHEN** different components check whether the value is a coord token
- **THEN** they agree that `<|coord_12|>` is a coord token
- **AND** they agree that `coord_12` is not a coord token.

### Requirement: Single-geometry extraction helper is reused and order-preserving
Shared geometry extraction/shape validation helpers SHALL provide a canonical way to:
- enforce exactly one geometry kind per object (`bbox_2d` xor `poly`),
- flatten nested point containers into a flat coordinate sequence when permitted,
- enforce arity invariants (bbox len=4; poly even-length and >= 6),
- preserve coordinate ordering (non-destructive).

Dataset preprocessing, coordinate standardization, and coord-token annotation paths MUST reuse these shared helpers.

#### Scenario: Object with both geometry keys is rejected
- **GIVEN** an object containing both `bbox_2d` and `poly`
- **WHEN** shared extraction/validation runs
- **THEN** validation fails explicitly (no silent selection).

#### Scenario: Polygon ordering is preserved through validation
- **GIVEN** a valid polygon coordinate list in canonical order
- **WHEN** it is validated via the shared helper
- **THEN** the returned coordinate sequence retains the same order and values.
