## MODIFIED Requirements

### Requirement: Compact token rows are independent of field order

Compact token embedding adaptation SHALL be authored through the high-level
`token_embeddings_adapter` namespace, derive required structural rows from
`detection_template.id`, and SHALL NOT alter the row set based on
`sample_factory.target_sequence.object_field_order`.

The implementation MUST validate the exact structural row set for the selected
template id before training or adapter-checkpoint use when compact token-row
adaptation is active.

Flat `token_rows` and `custom.token_embeddings_adapter` are deprecated by this
program. Active migrated configs MUST reject both authoring paths and point to
`token_embeddings_adapter`.

#### Scenario: Bbox-first and desc-first require identical rows

- **GIVEN** two compact configs that differ only by
  `sample_factory.target_sequence.object_field_order`
- **AND** both use `detection_template.id: compact_object_box_closed`
- **WHEN** token-row adaptation resolves required rows
- **THEN** both require the same coord rows plus `<|object_ref_start|>`,
  `<|object_ref_end|>`, `<|box_start|>`, and `<|box_end|>`.

#### Scenario: Legacy token-row authoring is rejected

- **GIVEN** an active migrated config with flat `token_rows`
- **WHEN** token embedding adapter configuration is parsed
- **THEN** loading fails before adapter setup
- **AND** the error directs the author to `token_embeddings_adapter`.

#### Scenario: Custom token adapter authoring is rejected

- **GIVEN** an active migrated config with `custom.token_embeddings_adapter`
- **WHEN** token embedding adapter configuration is parsed
- **THEN** loading fails before adapter setup
- **AND** the error directs the author to `token_embeddings_adapter`.
