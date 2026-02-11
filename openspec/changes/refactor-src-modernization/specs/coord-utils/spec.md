# coord-utils Spec Delta

This is a delta spec for change `refactor-src-modernization`.

## ADDED Requirements

### Requirement: Canonical coordinate conversion helpers are shared across data, inference, and evaluation
Coord-utils SHALL provide canonical conversion helpers for coord-token and numeric coordinate representations used by dataset, inference, and evaluator paths.
Consumers MUST reuse canonical helpers instead of defining parallel conversion implementations.
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
