## ADDED Requirements

### Requirement: Compact prompt examples compose template id and field order
Dense prompt resolution SHALL derive compact output examples from both the
resolved compact template id and the resolved object field order.

Prompt hashes MUST differ when either `detection_template.id` or
`custom.object_field_order` changes.  Prompt text MUST NOT describe desc-first
rows when the resolved field order is `geometry_first`.

#### Scenario: Bbox-first prompt pattern matches renderer
- **GIVEN** `detection_template.id: compact_object_box_closed`
- **AND** `custom.object_field_order: geometry_first`
- **WHEN** dense compact prompt text is resolved
- **THEN** the example row places the box segment before the object-ref segment
- **AND** the example row includes `<|box_end|>` and `<|object_ref_end|>`.

#### Scenario: Prompt hash changes with compact field order
- **GIVEN** two compact prompt configs that differ only by
  `custom.object_field_order`
- **WHEN** prompt hashes are computed
- **THEN** the hashes differ.

#### Scenario: Backend prompt parity includes both axes
- **GIVEN** a fixed inference config with a compact template id and
  `object_field_order`
- **WHEN** HF, local vLLM, and server-backed vLLM prepare backend requests
- **THEN** each request carries byte-equivalent system/user prompt text
- **AND** each request records the same template id, object field order, and
  prompt hash.
