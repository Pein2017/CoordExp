## MODIFIED Requirements

### Requirement: Compact prompt examples compose template id and field order

Dense prompt resolution SHALL derive compact output examples from both the
resolved compact template id and the resolved object field order.

For training and trainer-driven evaluation, prompt hashes MUST differ when
either `detection_template.id` or
`sample_factory.target_sequence.object_field_order` changes. Prompt text MUST
NOT describe desc-first rows when the resolved field order is `geometry_first`.

Stage-1 training prompt variant authoring SHALL use `prompt.variant`.
Historical `custom.extra.prompt_variant` authoring MUST NOT be accepted for
active target-hierarchy configs outside explicit rejection/migration tests.
Historical flat `prompt.prompt_variant_enabled` authoring MUST migrate as:

- `true` -> `prompt.variant: coco_80`
- `false` -> no authored prompt variant

After migration, active target-hierarchy configs MUST reject
`prompt.prompt_variant_enabled`.
Stage-2 rollout-correction training and eval rollout prompt variants remain
owned by `rollout_matching.prompt_variant` and
`rollout_matching.eval_prompt_variant`.

Standalone inference prompt hashing MAY continue to use inference-specific
field-order/config vocabulary, but metric-bearing artifacts MUST persist the
resolved template id and resolved object field order.

#### Scenario: Bbox-first prompt pattern matches renderer

- **GIVEN** `detection_template.id: compact_object_box_closed`
- **AND** `sample_factory.target_sequence.object_field_order: geometry_first`
- **WHEN** dense compact prompt text is resolved for training
- **THEN** the example row places the box segment before the object-ref segment
- **AND** the example row includes `<|box_end|>` and `<|object_ref_end|>`.

#### Scenario: Prompt hash changes with compact field order

- **GIVEN** two compact training prompt configs that differ only by
  `sample_factory.target_sequence.object_field_order`
- **WHEN** prompt hashes are computed
- **THEN** the hashes differ.

#### Scenario: Stage-1 prompt variant moves out of custom extra

- **GIVEN** an active Stage-1 target-hierarchy config with
  `prompt.variant: coco_80`
- **WHEN** training prompts are resolved
- **THEN** the shared prompt registry resolves the `coco_80` variant
- **AND** no `custom.extra.prompt_variant` authoring is required.

#### Scenario: Historical prompt boolean maps to prompt variant during migration

- **GIVEN** the flat canonical Stage-1 research-TF config authors
  `prompt.prompt_variant_enabled: true`
- **WHEN** it is migrated into the target hierarchy
- **THEN** the migrated config authors `prompt.variant: coco_80`
- **AND** active migrated configs reject `prompt.prompt_variant_enabled`.

#### Scenario: Backend prompt parity includes both axes

- **GIVEN** a fixed inference config with a compact template id and resolved
  object field order
- **WHEN** HF, local vLLM, and server-backed vLLM prepare backend requests
- **THEN** each request carries byte-equivalent system/user prompt text
- **AND** each request records the same template id, object field order, and
  prompt hash.
