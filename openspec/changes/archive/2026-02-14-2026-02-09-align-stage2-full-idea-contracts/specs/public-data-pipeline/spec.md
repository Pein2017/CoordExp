# public-data-pipeline Spec Delta

This is a delta spec for change `2026-02-09-align-stage2-full-idea-contracts`.

## MODIFIED Requirements

### Requirement: Geometry Compatibility (bbox_2d Required, poly Optional)
For compatibility across diverse sources, dataset conversion outputs and runtime assistant-payload emission paths consuming contract records SHALL enforce contract-valid object payloads with exactly one geometry field and a valid description.

Dataset conversion SHALL normalize source-specific geometry keys and formats into the CoordExp contract keys:
- Output geometry keys SHALL be only `bbox_2d` or `poly` (plus optional `poly_points`).
- Legacy source keys (e.g., `bbox`, `polygon`, `segmentation`, `coords`, etc.) SHALL NOT appear in the emitted contract JSONL.

Object contract invariants (normative, shared by conversion and runtime payload emission):
- Each emitted object MUST contain a non-empty string `desc`.
- Each emitted object MUST contain exactly one geometry field (`bbox_2d` xor `poly`).
- If geometry is `bbox_2d`, it MUST have exactly 4 coordinates.
- If geometry is `poly`, the flattened coordinate list MUST have even length and length >= 6.
- Conversion and runtime payload emission MUST fail fast on missing geometry, ambiguous multi-geometry objects, or invalid geometry arity. They MUST NOT emit geometry-less objects.

Polygon geometry (`poly`) support is OPTIONAL and MAY be implemented per dataset when the source provides polygon annotations.

#### Scenario: Source bbox key is not `bbox_2d`
- **GIVEN** a source dataset stores bbox geometry under a different field name (e.g., `bbox` or `coords`)
- **WHEN** the dataset plugin runs `convert`
- **THEN** the emitted contract JSONL uses `bbox_2d` with the correct `[x1,y1,x2,y2]` ordering.

#### Scenario: Source provides polygon only
- **GIVEN** a source dataset provides polygon geometry but not bounding boxes
- **WHEN** the dataset plugin runs `convert`
- **THEN** the plugin derives `bbox_2d` from the polygon (or emits `poly` when supported and enabled), and the record remains compliant with `docs/data/JSONL_CONTRACT.md`.

#### Scenario: Source provides both bbox and polygon
- **GIVEN** a source dataset provides both bbox and polygon geometry
- **WHEN** the dataset plugin runs `convert`
- **THEN** the plugin emits exactly one geometry per object.
- **AND** by default, the plugin emits `bbox_2d` (bbox-only is the default geometry mode).
- **AND** `poly` emission is opt-in and occurs only when explicitly enabled by the user via dataset-specific passthrough options.

#### Scenario: Converter fails on missing geometry or invalid arity
- **GIVEN** a source object with empty `desc`, missing geometry, multiple geometry fields, or invalid geometry arity
- **WHEN** the dataset plugin runs `convert`
- **THEN** conversion fails fast with an actionable validation error
- **AND** no geometry-less/ambiguous object is emitted into the contract JSONL.

#### Scenario: Runtime builder fails on missing geometry or invalid arity
- **GIVEN** a record reaching runtime assistant-payload emission with an object that has empty `desc`, missing geometry, multiple geometry fields, or invalid geometry arity
- **WHEN** runtime payload construction executes
- **THEN** runtime validation fails fast with an actionable error
- **AND** no geometry-less/ambiguous object is serialized into assistant payload output.

#### Scenario: User explicitly enables poly emission for a dataset that supports polygons
- **GIVEN** a dataset plugin supports polygon emission
- **WHEN** the user runs `public_data/run.sh <dataset> convert -- <dataset-specific-flag-to-enable-poly>`
- **THEN** the emitted contract JSONL contains `poly` geometries where available (still exactly one geometry per object).
