# stage2-ab-training Spec Delta

This is a delta spec for change `epoch-random-ordering-ablation`.

## MODIFIED Requirements

### Requirement: Stage-2 object instance ordering contract is unchanged
`custom.object_field_order` SHALL NOT modify stage-2 object instance ordering behavior.

Normative behavior:
- When Channel-A is selected, teacher-forced assistant payload construction SHALL follow `custom.object_ordering`.
- When Channel-A is selected, canonical clean-prefix construction SHALL serialize objects in the same effective order as the Channel-A teacher-forced assistant payload.
- For Channel-A, `custom.object_ordering: sorted` means canonical top-left ordering by `(minY, minX)`.
- For Channel-A, `custom.object_ordering: random` means the trainer SHALL use the current epoch’s deterministic dataset order for that sample.
- Channel-A object key numbering (`object_1`, `object_2`, ...`) SHALL follow the effective configured instance order.
- Channel-B object sequence remains determined by existing pipeline semantics (parsed rollout appearance order, matching/index continuation logic, and FN append order).
- Only intra-object field order is configurable through `custom.object_field_order`.

#### Scenario: geometry-first does not change rollout appearance order handling
- **GIVEN** rollout parsed objects appear in a specific raw-text order
- **AND** `custom.object_field_order: geometry_first`
- **WHEN** Stage-2 performs matching and FN append
- **THEN** parsed predicted order remains the same as raw-text appearance
- **AND** only field order inside serialized object payloads differs.

#### Scenario: Channel-A sorted ordering uses canonical top-left order
- **GIVEN** Channel-A is selected
- **AND** `custom.object_ordering: sorted`
- **WHEN** the trainer builds the teacher-forced assistant payload and canonical clean prefix
- **THEN** both surfaces serialize GT objects in canonical top-left order
- **AND** object key numbering follows that same order.

#### Scenario: Channel-A random ordering reuses the sample’s current epoch order
- **GIVEN** Channel-A is selected
- **AND** `custom.object_ordering: random`
- **WHEN** the trainer builds the teacher-forced assistant payload and canonical clean prefix
- **THEN** both surfaces serialize GT objects in the sample’s current epoch order
- **AND** object key numbering follows that same order.
