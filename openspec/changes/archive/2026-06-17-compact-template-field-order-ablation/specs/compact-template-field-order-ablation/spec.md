## ADDED Requirements

### Requirement: Compact rows are derived from template id and object field order
The system SHALL derive compact assistant row bytes from both the resolved
compact template id and `custom.object_field_order`.

The compact template id SHALL select closure and separator structure.  Supported
compact template ids for this change SHALL be:

- `compact`
- `compact_object_closed`
- `compact_box_closed`
- `compact_object_box_closed`
- `compact_object_box_closed_lines`

`custom.object_field_order` SHALL select semantic segment order:

- `desc_first`: object-ref segment before box segment.
- `geometry_first`: box segment before object-ref segment.

The renderer, strict parser, prompt example, render spans, target construction,
and artifact provenance MUST consume the same resolved pair.  The system MUST
NOT infer either axis from generated text for metric-bearing workflows.

#### Scenario: Rich geometry-first compact row is canonical
- **GIVEN** `detection_template.id: compact_object_box_closed`
- **AND** `custom.object_field_order: geometry_first`
- **WHEN** one object with desc `cat` is rendered with four coord tokens
- **THEN** the row is
  `<|box_start|><|coord_x1|><|coord_y1|><|coord_x2|><|coord_y2|><|box_end|><|object_ref_start|>cat<|object_ref_end|>`
- **AND** it does not contain a newline unless the selected template id is the
  line variant.

#### Scenario: Rich desc-first compact row remains canonical
- **GIVEN** `detection_template.id: compact_object_box_closed`
- **AND** `custom.object_field_order: desc_first`
- **WHEN** one object with desc `cat` is rendered with four coord tokens
- **THEN** the row is
  `<|object_ref_start|>cat<|object_ref_end|><|box_start|><|coord_x1|><|coord_y1|><|coord_x2|><|coord_y2|><|box_end|>`.

#### Scenario: Strict parser rejects opposite field order
- **GIVEN** `detection_template.id: compact_object_box_closed`
- **AND** `custom.object_field_order: geometry_first`
- **WHEN** generated text uses a desc-first row
- **THEN** strict parsing fails before normalized predictions are materialized.

### Requirement: Closure-family template ids have exact structural-row contracts
The compact template registry SHALL expose exact required structural token rows
for each template id.  Object field order MUST NOT change the required token-row
set.

Normative row counts:

- `compact`: 1002 trainable rows when token-row adaptation is enabled.
- `compact_object_closed`: 1003 trainable rows.
- `compact_box_closed`: 1003 trainable rows.
- `compact_object_box_closed`: 1004 trainable rows.
- `compact_object_box_closed_lines`: 1004 trainable rows.

#### Scenario: Object-only closure requires object-ref-end row
- **GIVEN** `detection_template.id: compact_object_closed`
- **AND** compact token-row adaptation is enabled
- **WHEN** required trainable rows are resolved
- **THEN** the row set contains `<|object_ref_start|>`,
  `<|object_ref_end|>`, `<|box_start|>`, and the 1000 coord-token rows
- **AND** it does not require `<|box_end|>`.

#### Scenario: Field order does not affect row count
- **GIVEN** two configs that differ only by `custom.object_field_order`
- **AND** both use `detection_template.id: compact_object_box_closed`
- **WHEN** required trainable rows are resolved
- **THEN** both configs require the same 1004 token rows.

### Requirement: Final ablation pair is sorted standard Stage-1 SFT
The bbox-first versus desc-first final ablation SHALL be expressed as paired
standard Stage-1 SFT configs, not as a rollout-only trainer migration.

Normative ablation settings:

- `custom.object_ordering: sorted`
- `custom.object_field_order: desc_first` for one arm
- `custom.object_field_order: geometry_first` for the other arm
- `custom.detection_template_id: compact_object_box_closed` or the resolved
  equivalent template surface used by the active standard SFT path
- `global_max_length: 12000`
- LLM-only trainability
- checkpoint
  `model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`
- standard Stage-1 SFT launch path

#### Scenario: Paired ablation configs differ only by row field order identity
- **GIVEN** the desc-first and bbox-first ablation configs
- **WHEN** their resolved configs are compared
- **THEN** the intentional differences include `custom.object_field_order`,
  run identity, output/logging paths, and cache fingerprints
- **AND** the template id, checkpoint, object ordering, packing length, trainable
  module policy, and dataset identity match.
