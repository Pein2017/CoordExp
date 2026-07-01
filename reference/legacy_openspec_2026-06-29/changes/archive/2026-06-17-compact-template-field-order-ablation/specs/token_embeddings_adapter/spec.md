## ADDED Requirements

### Requirement: Compact token rows are independent of field order
Compact token-row adaptation SHALL derive required structural rows from
`detection_template.id` and SHALL NOT alter the row set based on
`custom.object_field_order`.

The implementation MUST validate the exact structural row set for the selected
template id before training or adapter-checkpoint use when compact token-row
adaptation is active.

#### Scenario: Bbox-first and desc-first require identical rows
- **GIVEN** two compact configs that differ only by `custom.object_field_order`
- **AND** both use `detection_template.id: compact_object_box_closed`
- **WHEN** token-row adaptation resolves required rows
- **THEN** both require the same coord rows plus `<|object_ref_start|>`,
  `<|object_ref_end|>`, `<|box_start|>`, and `<|box_end|>`.

#### Scenario: Object-closed row set excludes box end
- **GIVEN** `detection_template.id: compact_object_closed`
- **WHEN** compact structural rows are resolved
- **THEN** `<|object_ref_end|>` is required
- **AND** `<|box_end|>` is not required.
