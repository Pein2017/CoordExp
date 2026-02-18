## ADDED Requirements

### Requirement: Stage-2 serialized object field order follows shared config
Stage-2 AB serialization paths SHALL honor `custom.object_field_order` exactly as stage-1 serialization does.

Scope:
- Channel-A teacher-forced assistant payload construction.
- Channel-B FN append serialization path used to build `Y_train`.

Normative behavior:
- `desc_first`: per-object payload order is `desc` then concrete geometry key (`bbox_2d` or `poly`).
- `geometry_first`: per-object payload order is concrete geometry key (`bbox_2d` or `poly`) then `desc`.
- Object instance ordering and object key numbering remain unchanged.
- The serializer MUST NOT emit a synthetic key literally named `geometry`.

#### Scenario: Channel-A uses geometry-first payload when configured
- **GIVEN** `custom.object_field_order: geometry_first`
- **WHEN** Channel-A constructs teacher-forced assistant payload text
- **THEN** each serialized object payload places its concrete geometry key before `desc`
- **AND** object keys remain sequential (`object_1`, `object_2`, ...).

#### Scenario: Channel-B uses geometry-first for FN append when configured
- **GIVEN** `custom.object_field_order: geometry_first`
- **AND** Channel-B appends unmatched GT objects
- **WHEN** `Y_train` is constructed
- **THEN** appended object payloads place their concrete geometry key before `desc`
- **AND** matching/masking logic remains unchanged.

#### Scenario: Default desc-first behavior is preserved in both channels
- **GIVEN** `custom.object_field_order` is omitted
- **WHEN** Channel-A or Channel-B serializes object payloads
- **THEN** payloads remain `desc` before the concrete geometry key (`bbox_2d` or `poly`).

### Requirement: Stage-2 object instance ordering contract is unchanged
`custom.object_field_order` SHALL NOT modify stage-2 object instance ordering behavior.

Normative behavior:
- Object sequence remains determined by existing pipeline semantics (GT order, parsed rollout appearance order, and current matching/index continuation logic).
- Only intra-object field order is configurable.

#### Scenario: geometry-first does not change rollout appearance order handling
- **GIVEN** rollout parsed objects appear in a specific raw-text order
- **AND** `custom.object_field_order: geometry_first`
- **WHEN** Stage-2 performs matching and FN append
- **THEN** parsed predicted order remains the same as raw-text appearance
- **AND** only field order inside serialized object payloads differs.
