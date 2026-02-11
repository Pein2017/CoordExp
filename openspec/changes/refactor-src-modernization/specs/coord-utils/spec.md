# coord-utils Spec Delta

This is a delta spec for change `refactor-src-modernization`.

## ADDED Requirements

### Requirement: Canonical coordinate conversion helpers are shared across data, inference, and evaluation
Coord-utils SHALL provide canonical conversion helpers for coord-token and numeric coordinate representations used by dataset, inference, and evaluator paths.
Consumers MUST reuse canonical helpers instead of defining parallel conversion implementations.
For overlapping active deltas, detailed helper ownership/shape semantics are authoritative in `src-ambiguity-cleanup-2026-02-11`; this change MUST stay compatible with that contract.
Ownership boundaries MUST remain explicit: transform/resize semantics remain authoritative in `src/datasets/geometry.py`, while shared coord-utils helpers remain pure/import-light and MUST NOT introduce dataset<->eval dependency cycles.

#### Scenario: Shared conversion path yields identical numeric coordinates across consumers
- **GIVEN** identical coord-token inputs and image dimensions
- **WHEN** dataset, inference, and evaluator components convert coordinates via canonical helpers
- **THEN** each consumer obtains the same numeric coordinate outputs.

#### Scenario: Canonical helper reuse does not violate geometry ownership boundaries
- **GIVEN** shared coord-utils helpers are consumed by dataset and evaluator paths
- **WHEN** dependency boundaries are inspected
- **THEN** transform-authoritative logic remains in `src/datasets/geometry.py`
- **AND** shared coord-utils modules do not introduce dataset<->eval import cycles.

### Requirement: Coordinate validation invariants are centralized and explicit
Coord-utils SHALL define centralized validation invariants for coordinate sequences, including arity, even-point structure, and allowed value ranges.
Validation outcomes MUST be deterministic and consistent across all consumers.

#### Scenario: Invalid odd-length coordinate sequence fails consistently
- **GIVEN** a geometry sequence with odd coordinate count
- **WHEN** any consumer validates via coord-utils
- **THEN** validation fails deterministically with a consistent invariant violation classification.

### Requirement: Geometry helper behavior remains non-destructive and order-preserving
Shared coordinate and geometry helper flows MUST preserve coordinate ordering semantics and MUST NOT reorder or drop valid points during conversion/validation.

#### Scenario: Valid polygon point order is preserved through helper pipeline
- **GIVEN** a valid polygon coordinate list in canonical order
- **WHEN** it is passed through shared validation/conversion helpers
- **THEN** the output retains the same point order and geometry semantics.

### Requirement: Canonical geometry keys and arity invariants are enforced
Geometry dicts SHALL use canonical keys `bbox_2d` or `poly` (exactly one per object geometry mapping).
Validation helpers MUST reject legacy keys (`bbox`, `polygon`) to prevent ambiguous dual schema behavior.

Arity invariants:
- For `bbox_2d`, the sequence MUST contain exactly 4 values (expected ordering `[x1, y1, x2, y2]`).
- For `poly`, the sequence MUST be flat (not nested), MUST contain an even number of values, and MUST contain at least 6 coordinates (>= 3 points).

#### Scenario: Legacy geometry keys are rejected explicitly
- **GIVEN** a geometry mapping containing `bbox` or `polygon`
- **WHEN** it is validated via the dataset contract validation helper
- **THEN** validation fails with explicit diagnostics indicating canonical keys are required.

#### Scenario: Invalid geometry arity fails deterministically
- **GIVEN** a `bbox_2d` geometry with len != 4 OR a `poly` geometry with odd length / length < 6
- **WHEN** any consumer validates via canonical geometry validation helpers
- **THEN** validation fails deterministically with a consistent invariant violation classification.

### Requirement: Canonical module map for coord/geometry helpers is explicit
The canonical ownership map SHALL be:
- coord-token regex/encode/decode/range checks: `src/coord_tokens/codec.py`,
- single-object geometry extraction + key/arity validation: `src/common/geometry/object_geometry.py`,
- generic coord/point flattening + conversion helpers: `src/common/geometry/coord_utils.py`,
- transform/resize geometry operations: `src/datasets/geometry.py` (authoritative transforms surface).

Legacy helper paths MAY remain as compatibility shims, but MUST re-export canonical behavior and MUST NOT carry divergent logic.

#### Scenario: Canonical helper import paths remain stable and unambiguous
- **GIVEN** a consumer needs coord-token parsing and geometry extraction
- **WHEN** it uses canonical helper imports
- **THEN** it resolves to one authoritative implementation per concern
- **AND** legacy shim paths (if used) preserve identical behavior.

### Requirement: Coord numeric range/coercion rules are deterministic
Coord-token and numeric-coordinate conversion helpers SHALL enforce:
- integer bin range `0..999` for coord-token values,
- numeric inputs accepted only when integer-valued within tolerance (`abs(v - round(v)) <= 1e-6`),
- out-of-range or non-integer values rejected explicitly.

#### Scenario: Numeric coordinate coercion rejects out-of-range and non-integer values
- **GIVEN** numeric inputs `1000`, `-1`, and `12.3`
- **WHEN** canonical coord conversion/validation runs
- **THEN** all are rejected with explicit diagnostics
- **AND** an in-range integer-valued input (for example `12` or `12.0`) is accepted.

### Requirement: Nested point flattening is opt-in and shape-bounded
Canonical extraction/validation helpers MUST default to flat coordinate sequences.
Nested point containers are allowed only when the caller explicitly opts in (for example `allow_nested_points=true`), and only for sequence-of-pairs shapes (`[[x, y], ...]`).

#### Scenario: Nested point containers are rejected unless explicitly allowed
- **GIVEN** polygon points in nested-pairs form
- **WHEN** extraction runs with default flat-only behavior
- **THEN** validation fails explicitly
- **AND** when extraction runs with explicit nested-point opt-in, the sequence is flattened deterministically while preserving order.
