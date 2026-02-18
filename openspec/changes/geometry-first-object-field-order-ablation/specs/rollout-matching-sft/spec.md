## ADDED Requirements

### Requirement: FN append serialization honors configured object field order
When rollout-matching builds `Y_train` via mandatory FN append, each appended object payload SHALL follow `custom.object_field_order`.

Normative behavior:
- `desc_first`: append payload uses `{desc, bbox_2d}` or `{desc, poly}` depending on object geometry type.
- `geometry_first`: append payload uses `{bbox_2d, desc}` or `{poly, desc}` depending on object geometry type.
- Geometry key can be `bbox_2d` or `poly`.
- The serializer MUST NOT emit a synthetic key literally named `geometry`.

This requirement applies only to field order within each appended object payload and MUST NOT alter:
- object key numbering (`object_{n}` continuation),
- predicted object appearance-order parsing,
- matching order semantics.

#### Scenario: geometry-first changes only per-object field order in FN append
- **GIVEN** `custom.object_field_order: geometry_first`
- **AND** Channel-B has unmatched GT objects to append
- **WHEN** `SerializeAppend(FN_gt_objects)` is produced
- **THEN** each appended object places its concrete geometry key (`bbox_2d` or `poly`) before `desc`
- **AND** object keys still start at `max_object_index_in_prefix + 1`.

#### Scenario: desc-first remains baseline append layout
- **GIVEN** `custom.object_field_order` is omitted or set to `desc_first`
- **WHEN** FN append fragment is serialized
- **THEN** appended object payloads keep `desc` before the concrete geometry key (`bbox_2d` or `poly`).

### Requirement: Field-order variation is schema-equivalent for strict parsing
Strict parsing for rollout matching SHALL treat `desc_first` and `geometry_first` object payloads as schema-equivalent.

Normative behavior:
- Reordering `{desc, bbox_2d}` to `{bbox_2d, desc}` or `{desc, poly}` to `{poly, desc}` MUST NOT by itself invalidate an object.
- Existing strict checks (missing desc, invalid geometry, wrong arity, bad coord tokens, etc.) remain unchanged.

#### Scenario: geometry-first object remains valid under strict parse
- **GIVEN** a rollout object encoded as `{\"bbox_2d\": [...], \"desc\": \"...\"}`
- **WHEN** strict parsing runs
- **THEN** the object is considered valid if all existing schema constraints pass
- **AND** it is not dropped solely due to field order.
