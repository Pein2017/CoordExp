# object-field-ordering Spec Delta

This is a delta spec for change `epoch-random-ordering-ablation`.

## MODIFIED Requirements

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
