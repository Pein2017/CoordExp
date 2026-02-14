# coord-utils Spec Delta

This is a delta spec for change `2026-02-11-src-ambiguity-cleanup`.

## ADDED Requirements

This change is the authoritative helper-contract delta for coord/geometry helper ownership, range validation, and nested-point boundaries across active overlaps.

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

### Requirement: Canonical module ownership map is explicit
The canonical ownership map for shared coord/geometry helpers SHALL be:
- coord-token regex + encode/decode/range checks: `src/coord_tokens/codec.py`,
- geometry extraction and key/arity validation: `src/common/geometry/object_geometry.py`,
- flattening + generic coord conversion helpers: `src/common/geometry/coord_utils.py`,
- transform/resize operations: `src/datasets/geometry.py` (authoritative transforms surface).

Legacy module paths MAY remain for compatibility but MUST re-export canonical implementations and MUST NOT carry divergent behavior.

#### Scenario: Canonical helper imports resolve without ambiguity
- **GIVEN** a consumer imports coord-token detection and geometry extraction helpers
- **WHEN** it uses canonical module paths
- **THEN** each concern resolves to one authoritative implementation
- **AND** legacy paths, if imported, behave identically via re-export.

### Requirement: Coord token/numeric range rules are deterministic
Canonical coord conversion helpers SHALL enforce:
- coord-token integer bins in range `0..999`,
- numeric coords accepted only when integer-valued within tolerance (`abs(v - round(v)) <= 1e-6`) and within `0..999`,
- out-of-range or non-integer values rejected explicitly.

#### Scenario: Out-of-range and non-integer numeric coords are rejected
- **GIVEN** numeric values `1000`, `-1`, and `12.3`
- **WHEN** canonical coord conversion validates them
- **THEN** each invalid value is rejected with explicit diagnostics
- **AND** valid integer-valued in-range values are accepted.

### Requirement: Nested point flattening is opt-in and shape-bounded
Nested point flattening MUST be opt-in. Default shared extraction behavior is flat-only.
When nested flattening is enabled, only sequence-of-pairs forms (`[[x, y], ...]`) are accepted; malformed nested shapes are rejected.

#### Scenario: Nested points require explicit opt-in
- **GIVEN** polygon points represented as `[[x, y], ...]`
- **WHEN** extraction runs in default flat-only mode
- **THEN** validation fails explicitly
- **AND WHEN** extraction runs with nested-point opt-in
- **THEN** the points are flattened deterministically with order preserved.
