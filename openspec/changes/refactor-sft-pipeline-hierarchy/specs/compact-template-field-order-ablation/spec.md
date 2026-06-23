## MODIFIED Requirements

### Requirement: Compact rows are derived from template id and object field order

The system SHALL derive compact assistant row bytes from both the top-level
`detection_template.id` and
`sample_factory.target_sequence.object_field_order`.

The compact template id SHALL select closure and separator structure.

`sample_factory.target_sequence.object_field_order` SHALL select semantic
segment order:

- `desc_first`: object-ref segment before box segment.
- `geometry_first`: box segment before object-ref segment.

The renderer, strict parser, prompt example, render spans, target construction,
and artifact provenance MUST consume the same resolved pair. The system MUST NOT
infer either axis from generated text for metric-bearing workflows.
`custom.object_field_order` and `custom.detection_template_id` are historical
authoring names after this migration and MUST NOT be active config sources
outside explicit rejection/migration tests.

#### Scenario: Rich geometry-first compact row is canonical

- **GIVEN** `detection_template.id: compact_object_box_closed`
- **AND** `sample_factory.target_sequence.object_field_order: geometry_first`
- **WHEN** one object with desc `cat` is rendered with four coord tokens
- **THEN** the row is
  `<|box_start|><|coord_x1|><|coord_y1|><|coord_x2|><|coord_y2|><|box_end|><|object_ref_start|>cat<|object_ref_end|>`
- **AND** it does not contain a newline unless the selected template id is the
  line variant.

#### Scenario: Rich desc-first compact row remains canonical

- **GIVEN** `detection_template.id: compact_object_box_closed`
- **AND** `sample_factory.target_sequence.object_field_order: desc_first`
- **WHEN** one object with desc `cat` is rendered with four coord tokens
- **THEN** the row is
  `<|object_ref_start|>cat<|object_ref_end|><|box_start|><|coord_x1|><|coord_y1|><|coord_x2|><|coord_y2|><|box_end|>`.

#### Scenario: Strict parser rejects opposite field order

- **GIVEN** `detection_template.id: compact_object_box_closed`
- **AND** `sample_factory.target_sequence.object_field_order: geometry_first`
- **WHEN** generated text uses a desc-first row
- **THEN** strict parsing fails before normalized predictions are materialized.

### Requirement: Closure-family template ids have exact structural-row contracts

The compact template registry SHALL expose exact required structural token rows
for each template id. Object field order MUST NOT change the required token-row
set.

#### Scenario: Field order does not affect row count

- **GIVEN** two configs that differ only by
  `sample_factory.target_sequence.object_field_order`
- **AND** both use `detection_template.id: compact_object_box_closed`
- **WHEN** required trainable rows are resolved
- **THEN** both configs require the same structural token rows.

### Requirement: Final ablation pair is sorted standard Stage-1 SFT

The bbox-first versus desc-first final ablation SHALL be expressed as paired
standard Stage-1 SFT configs, not as a rollout-only trainer migration.

Normative ablation settings:

- `pipeline.id: stage1_standard_sft`
- `objective.id: standard_ce`
- `sample_factory.id: detection_sequence`
- `sample_factory.target_sequence.object_ordering: sorted`
- `sample_factory.target_sequence.object_field_order: desc_first` for one arm
- `sample_factory.target_sequence.object_field_order: geometry_first` for the
  other arm
- `detection_template.id: compact_object_box_closed`
- `global_max_length: 12000`
- LLM-only trainability
- checkpoint
  `model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`
- standard Stage-1 SFT launch path

#### Scenario: Paired ablation configs differ only by row field order identity

- **GIVEN** the desc-first and bbox-first ablation configs
- **WHEN** their resolved configs are compared
- **THEN** the intentional differences include
  `sample_factory.target_sequence.object_field_order`, run identity,
  output/logging paths, and cache fingerprints
- **AND** the template id, checkpoint, object ordering, packing length, trainable
  module policy, and dataset identity match.
