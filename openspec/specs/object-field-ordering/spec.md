# object-field-ordering Specification

## Purpose
Define a config-driven contract for per-object JSON field order in detection training targets, enabling controlled autoregressive ablations while preserving object instance ordering and geometry semantics.
## Requirements
### Requirement: Object field order is config-driven and strict
The system SHALL expose `custom.object_field_order` as the single source of truth for per-object field order in:
- serialized assistant payload objects, and
- rendered assistant message JSON text that is fed into the Qwen3-VL chat template.

Allowed values:
- `desc_first`
- `geometry_first`

Unknown values MUST fail fast during config loading with actionable guidance.
Omission MUST fail fast; `custom.object_field_order` is required and must be set explicitly.

Normative clarification:
- `geometry_first` means "emit the object's existing geometry key (`bbox_2d` or `poly`) before `desc`."
- The contract MUST NOT introduce or require a synthetic key literally named `geometry`.
- `geometry_first` is the generalized term for the direction doc's "bbox-first"; they are equivalent when geometry is `bbox_2d`.
- For a given sample, assistant payload per-object key order and rendered assistant JSON text per-object key order MUST be equivalent (no split-brain ordering between structured payload and emitted text).

#### Scenario: Omitted object field order fails fast
- **GIVEN** `custom.object_field_order` is omitted
- **WHEN** training config is loaded
- **THEN** config validation fails fast with guidance to set `custom.object_field_order` explicitly (`desc_first` or `geometry_first`).

#### Scenario: Explicit desc-first preserves baseline layout
- **GIVEN** `custom.object_field_order: desc_first`
- **WHEN** dataset serialization runs
- **THEN** per-object payload order is `desc` before the concrete geometry key (`bbox_2d` or `poly`).

#### Scenario: geometry-first is accepted
- **GIVEN** `custom.object_field_order: geometry_first`
- **WHEN** training config is loaded
- **THEN** config parsing succeeds
- **AND** per-object payloads serialize the concrete geometry key (`bbox_2d` or `poly`) before `desc`.

#### Scenario: geometry-first preserves concrete geometry key names
- **GIVEN** `custom.object_field_order: geometry_first`
- **WHEN** an object payload is serialized
- **THEN** the first semantic key is `bbox_2d` or `poly` (whichever exists on the object)
- **AND** no key named `geometry` is emitted.

#### Scenario: Rendered assistant JSON text uses the same configured order
- **GIVEN** `custom.object_field_order: geometry_first`
- **WHEN** a sample is rendered into assistant message JSON text for chat-template encoding
- **THEN** each rendered object uses concrete geometry key then `desc`
- **AND** the rendered order matches the corresponding structured assistant payload order.

#### Scenario: Invalid object field order fails fast
- **GIVEN** `custom.object_field_order: foo_first`
- **WHEN** config is loaded
- **THEN** loading fails fast with an error listing allowed values.

### Requirement: Object instance ordering remains independent
`custom.object_field_order` SHALL control only key order within each object payload and SHALL NOT alter object instance sequence.

Object instance sequence SHALL remain controlled exclusively by the active ordering configuration:
- dataset-backed training and evaluation serialization MUST use `custom.object_ordering`
- standalone inference dense prompt construction MUST use `infer.object_ordering`

Allowed ordering values:
- `sorted`
- `random`

Default behavior:
- `custom.object_ordering` defaults to `sorted`
- `infer.object_ordering` defaults to `sorted`

Normative behavior:
- `sorted` means canonical top-left ordering by `(minY, minX)` (top-to-bottom, then left-to-right).
- `random` means deterministic per-epoch reshuffle for dataset-backed training/evaluation, derived from stable sample identity plus epoch.
- On standalone inference surfaces, `random` controls prompt wording and expected ordering policy only; it MUST NOT be reinterpreted as a field-order change.
- `custom.object_field_order` and inference field-order config MUST NOT change object instance sequence.

#### Scenario: geometry-first does not reorder object instances
- **GIVEN** `custom.object_ordering: sorted`
- **AND** `custom.object_field_order: geometry_first`
- **WHEN** objects are serialized
- **THEN** object keys (`object_1`, `object_2`, ...) follow sorted instance order
- **AND** only per-object key order changes to geometry-before-desc.

#### Scenario: Inference ordering defaults to sorted
- **GIVEN** standalone dense inference omits `infer.object_ordering`
- **WHEN** dense prompts are resolved
- **THEN** the resolved ordering policy defaults to `sorted`

#### Scenario: Random training ordering reshuffles each epoch deterministically
- **GIVEN** `custom.object_ordering: random`
- **WHEN** the same base sample is fetched in two different epochs
- **THEN** object instance order MAY differ between epochs
- **AND** for a fixed seed, sample identity, and epoch, the order MUST remain deterministic

### Requirement: Geometry-first supports bbox and poly payloads
When `custom.object_field_order: geometry_first`, the geometry key (`bbox_2d` or `poly`) SHALL appear before `desc` for each object payload.

#### Scenario: geometry-first for bbox payload
- **GIVEN** an object with `bbox_2d`
- **AND** `custom.object_field_order: geometry_first`
- **WHEN** payload is serialized
- **THEN** object field order is `bbox_2d` then `desc`.

#### Scenario: geometry-first for poly payload
- **GIVEN** an object with `poly`
- **AND** `custom.object_field_order: geometry_first`
- **WHEN** payload is serialized
- **THEN** object field order is `poly` then `desc`.

### Requirement: Assistant outputs do not emit `poly_points` metadata fields
For dense assistant outputs used in training targets, per-object payloads SHALL contain:
- `desc`, and
- exactly one geometry key (`bbox_2d` or `poly`).

`poly_points` is JSONL-side optional metadata and MUST NOT be emitted in assistant payloads or rendered assistant JSON text.

#### Scenario: poly object output excludes poly_points
- **GIVEN** a record whose source object contains `poly` and optional `poly_points`
- **WHEN** assistant payload and rendered assistant JSON text are produced
- **THEN** emitted object keys include `poly` and `desc` only
- **AND** `poly_points` is absent from emitted assistant outputs.

### Requirement: Prompt wording aligns with configured field order
Dense prompt templates SHALL align instruction wording with the configured field order on the active surface.

Normative behavior:
- `desc_first`: prompts request desc plus geometry (baseline wording).
- `geometry_first`: prompts request the geometry key (`bbox_2d` or `poly`) before `desc`.
- Training and trainer-driven rollout/eval prompt wording SHALL derive instance-order wording from `custom.object_ordering`.
- Standalone inference prompt wording SHALL derive instance-order wording from `infer.object_ordering`.

#### Scenario: geometry-first prompt wording is selected
- **GIVEN** `custom.object_field_order: geometry_first`
- **AND** `custom.object_ordering: random`
- **WHEN** a dense prompt pair is resolved for training
- **THEN** prompt text requests the geometry key (`bbox_2d` or `poly`) before `desc`
- **AND** still states object instance order is random.

#### Scenario: Standalone inference preserves independent ordering and field-order wording
- **GIVEN** `infer.object_field_order: geometry_first`
- **AND** `infer.object_ordering: random`
- **WHEN** a dense prompt pair is resolved for standalone inference
- **THEN** prompt text requests the geometry key (`bbox_2d` or `poly`) before `desc`
- **AND** still states object instance order is random.

