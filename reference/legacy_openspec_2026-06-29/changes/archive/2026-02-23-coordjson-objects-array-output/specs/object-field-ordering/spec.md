## MODIFIED Requirements

### Requirement: Object field order is config-driven and strict
The system SHALL expose `custom.object_field_order` as the single source of truth for **per-record key order** inside the assistant `"objects"` array in:
- serialized assistant payload records, and
- rendered assistant message CoordJSON text that is fed into the Qwen3-VL chat template.

Allowed values:
- `desc_first` (default)
- `geometry_first`

Unknown values MUST fail fast during config loading with actionable guidance.

Normative clarification:
- `geometry_first` means "emit the record's concrete geometry key (`bbox_2d` or `poly`) before `desc`."
- The contract MUST NOT introduce or require a synthetic key literally named `geometry`.
- `geometry_first` is the config name; when a record’s geometry key is `bbox_2d`, it is equivalent to the direction doc’s “bbox-first” wording, but the config also applies to `poly` geometries.
- For a given sample, assistant payload per-record key order and rendered assistant CoordJSON per-record key order MUST be equivalent (no split-brain ordering between structured payload and emitted text).
- `custom.object_field_order` governs only per-record key order; it MUST NOT change the order of records in the `"objects"` array (instance ordering is controlled by `custom.object_ordering` only).

#### Scenario: Default behavior remains desc-first
- **GIVEN** `custom.object_field_order` is omitted
- **WHEN** training config is loaded and dataset serialization runs
- **THEN** per-record payload order is `desc` before the concrete geometry key (`bbox_2d` or `poly`)
- **AND** behavior matches the legacy baseline ordering semantics.

#### Scenario: geometry-first is accepted
- **GIVEN** `custom.object_field_order: geometry_first`
- **WHEN** training config is loaded
- **THEN** config parsing succeeds
- **AND** per-record payloads serialize the concrete geometry key (`bbox_2d` or `poly`) before `desc`.

#### Scenario: geometry-first preserves concrete geometry key names
- **GIVEN** `custom.object_field_order: geometry_first`
- **WHEN** a record payload is serialized
- **THEN** the first semantic key is `bbox_2d` or `poly` (whichever exists on the record)
- **AND** no key named `geometry` is emitted.

#### Scenario: Rendered assistant CoordJSON uses the same configured order
- **GIVEN** `custom.object_field_order: geometry_first`
- **WHEN** a sample is rendered into assistant message CoordJSON text for chat-template encoding
- **THEN** each rendered record uses concrete geometry key then `desc`
- **AND** the rendered per-record key order matches the corresponding structured assistant payload per-record key order.

#### Scenario: Invalid object field order fails fast
- **GIVEN** `custom.object_field_order: foo_first`
- **WHEN** config is loaded
- **THEN** loading fails fast with an error listing allowed values.

### Requirement: Object instance ordering remains independent
`custom.object_field_order` SHALL control only key order within each record and SHALL NOT alter object instance sequence.

Object instance sequence SHALL remain controlled exclusively by `custom.object_ordering` (`sorted` or `random`).

#### Scenario: geometry-first does not reorder object instances
- **GIVEN** `custom.object_ordering: sorted`
- **AND** `custom.object_field_order: geometry_first`
- **WHEN** objects are serialized into the `"objects"` array
- **THEN** the `"objects"` array elements follow sorted instance order
- **AND** only per-record key order changes to geometry-before-desc.

### Requirement: Geometry-first supports bbox and poly payloads
When `custom.object_field_order: geometry_first`, the geometry key (`bbox_2d` or `poly`) SHALL appear before `desc` for each record payload.

#### Scenario: geometry-first for bbox payload
- **GIVEN** a record with `bbox_2d`
- **AND** `custom.object_field_order: geometry_first`
- **WHEN** payload is serialized
- **THEN** record field order is `bbox_2d` then `desc`.

#### Scenario: geometry-first for poly payload
- **GIVEN** a record with `poly`
- **AND** `custom.object_field_order: geometry_first`
- **WHEN** payload is serialized
- **THEN** record field order is `poly` then `desc`.

### Requirement: Assistant outputs do not emit `poly_points` metadata fields
For dense assistant outputs used in training targets, per-record payloads SHALL contain:
- `desc`, and
- exactly one geometry key (`bbox_2d` or `poly`).

`poly_points` is JSONL-side optional metadata and MUST NOT be emitted in assistant payloads or rendered assistant CoordJSON text.

#### Scenario: poly record output excludes poly_points
- **GIVEN** a record whose source object contains `poly` and optional `poly_points`
- **WHEN** assistant payload and rendered assistant CoordJSON text are produced
- **THEN** emitted record keys include `poly` and `desc` only
- **AND** `poly_points` is absent from emitted assistant outputs.

### Requirement: Prompt wording aligns with configured field order
Dense prompt templates SHALL align instruction wording with `custom.object_field_order`.

Normative behavior:
- `desc_first`: prompts request desc plus geometry (baseline wording).
- `geometry_first`: prompts request the geometry key (`bbox_2d` or `poly`) before `desc`.
- Object instance ordering wording (sorted/random) remains unchanged and continues to be governed by `custom.object_ordering`.

#### Scenario: geometry-first prompt wording is selected
- **GIVEN** `custom.object_field_order: geometry_first`
- **AND** `custom.object_ordering: random`
- **WHEN** dense prompt pair is resolved
- **THEN** prompt text requests the geometry key (`bbox_2d` or `poly`) before `desc`
- **AND** still states object instance order is random.
