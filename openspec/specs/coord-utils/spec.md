# coord-utils Specification

## Purpose
Define shared geometry helper utilities used across evaluator/visualization for bridging bbox and polygon representations without duplicating geometry logic.

## Requirements
### Requirement: Mixed-geometry bridging
- The shared module SHALL include helpers to convert a polygon to its tight bbox and a bbox to a minimal quadrilateral segmentation so that bbox GT can be matched against polygon predictions via IoU.
- Detection evaluator and visualization SHALL use these helpers to keep bbox–poly matching feasible without bespoke logic in each tool and SHALL import them via `src/common/geometry` to avoid parallel implementations.

#### Scenario: Polygon prediction matches bbox GT
- GIVEN a GT bbox and a prediction polygon covering the same region
- WHEN the evaluator uses the shared helper to derive a bbox/segmentation for IoU
- THEN the prediction can be paired and scored correctly instead of being dropped for geometry mismatch.


### Requirement: Canonical coordinate conversion helpers are shared across data, inference, and evaluation
Coord-utils SHALL provide canonical conversion helpers for coord-token and numeric coordinate representations used by dataset, inference, and evaluator paths.
Consumers MUST reuse canonical helpers instead of defining parallel conversion implementations.
For overlapping active deltas, detailed helper ownership/shape semantics are authoritative in `2026-02-11-src-ambiguity-cleanup`; this change MUST stay compatible with that contract.
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
Nested point flattening MUST be opt-in. Default shared extraction behavior is flat-only.
When nested flattening is enabled, only sequence-of-pairs forms (`[[x, y], ...]`) are accepted; malformed nested shapes are rejected.

#### Scenario: Nested points require explicit opt-in
- **GIVEN** polygon points represented as `[[x, y], ...]`
- **WHEN** extraction runs in default flat-only mode
- **THEN** validation fails explicitly
- **AND WHEN** extraction runs with nested-point opt-in
- **THEN** the points are flattened deterministically with order preserved.

#### Scenario: Nested point containers are rejected unless explicitly allowed
- **GIVEN** polygon points in nested-pairs form
- **WHEN** extraction runs with default flat-only behavior
- **THEN** validation fails explicitly
- **AND** when extraction runs with explicit nested-point opt-in, the sequence is flattened deterministically while preserving order.


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
This requirement MUST be interpreted as the post-refactor restatement of `Requirement: Canonical module map for coord/geometry helpers is explicit` and preserves the same ownership contract under updated naming.
Any future contract edit to either restatement MUST be mirrored in the paired ownership requirement section within the same change to avoid drift.

The canonical ownership map for shared coord/geometry helpers SHALL remain:
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
This requirement MUST be interpreted as the post-refactor restatement of `Requirement: Coord numeric range/coercion rules are deterministic` and keeps the same invariant scope.
Any future invariant edit to either restatement MUST be mirrored in the paired range/coercion requirement section within the same change to avoid drift.

Canonical coord conversion helpers SHALL enforce:
- coord-token integer bins in range `0..999`,
- numeric coords accepted only when integer-valued within tolerance (`abs(v - round(v)) <= 1e-6`) and within `0..999`,
- out-of-range or non-integer values rejected explicitly.

#### Scenario: Out-of-range and non-integer numeric coords are rejected
- **GIVEN** numeric values `1000`, `-1`, and `12.3`
- **WHEN** canonical coord conversion validates them
- **THEN** each invalid value is rejected with explicit diagnostics
- **AND** valid integer-valued in-range values are accepted.
