## ADDED Requirements

### Requirement: Semantic detection template ids
The system SHALL expose detection assistant serialization through a semantic
`detection_template.id` value.

Allowed ids SHALL be:

- `stage1_json_pretty`
- `compact`
- `compact_box_closed`
- `compact_object_box_closed`
- `compact_object_box_closed_lines`

The compact ids SHALL render object rows canonically as:

- `compact`: `<|object_ref_start|>{desc}<|box_start|>{coords}`
- `compact_box_closed`:
  `<|object_ref_start|>{desc}<|box_start|>{coords}<|box_end|>`
- `compact_object_box_closed`:
  `<|object_ref_start|>{desc}<|object_ref_end|><|box_start|>{coords}<|box_end|>`
- `compact_object_box_closed_lines`:
  `<|object_ref_start|>{desc}<|object_ref_end|><|box_start|>{coords}<|box_end|>\n`

For `compact_object_box_closed_lines`, the canonical renderer MUST include
exactly one newline after every object row, including the final row.

`compact_full` SHALL NOT be an accepted schema value after this change.

`stage1_json_pretty` SHALL remain the strict non-compact JSON template id. It
MUST participate in resolved config and artifact provenance when serialization
depends on it, but it MUST NOT use compact token-row adaptation or compact row
patterns.

#### Scenario: Current compact bytes use semantic name
- **GIVEN** a detection config with `detection_template.id: compact`
- **WHEN** one object is rendered with desc `cat` and four coord tokens
- **THEN** the assistant row starts with `<|object_ref_start|>cat<|box_start|>`
- **AND** it does not include `<|object_ref_end|>`, `<|box_end|>`, or a newline.

#### Scenario: Closed line variant has final newline
- **GIVEN** a detection config with
  `detection_template.id: compact_object_box_closed_lines`
- **WHEN** assistant text is rendered for one or more objects
- **THEN** every object row ends with `<|box_end|>\n`
- **AND** the final object row also ends with that newline.

#### Scenario: Old compact name is rejected
- **WHEN** config parsing sees `detection_template.id: compact_full`
- **THEN** parsing fails with an error that lists the supported semantic
  template ids.

#### Scenario: Json template stays non-compact
- **GIVEN** `detection_template.id: stage1_json_pretty`
- **WHEN** template resolution runs
- **THEN** the resolver selects the strict JSON assistant template
- **AND** compact token-row adaptation is not required for that template id.

### Requirement: Template id is the single authored serialization surface
For compact detection templates, `detection_template.id` SHALL be the only
authored config choice that selects assistant serialization.

The canonical YAML path SHALL be `detection_template.id` in both training and
inference configs. Inference-specific compact controls such as
`infer.detection_sequence_format`, `infer.row_separator`,
`infer.compact_full_parse_mode`, and `infer.parsing.compact_full` SHALL NOT be
accepted as aliases for this field.

The system MUST derive all of the following from `detection_template.id`:

- renderer,
- strict parser,
- prompt row pattern,
- row separator,
- required structural tokens,
- required trainable token rows,
- artifact metadata value,
- inference materialization and post-hoc mAP preflight policy.

Configs MUST reject independently authored compact parse modes, row separators,
serialization policies, or compact-format aliases when they duplicate or
contradict the selected template id.

#### Scenario: Duplicate row separator knob is rejected
- **GIVEN** a config with
  `detection_template.id: compact_object_box_closed_lines`
- **AND** the config also authors an independent compact row separator value
- **WHEN** config parsing or runtime resolution runs
- **THEN** the run fails before training or inference starts
- **AND** the error points to `detection_template.id` as the supported source of
  truth.

#### Scenario: Old inference compact knobs are rejected
- **GIVEN** an inference config with a supported `detection_template.id`
- **AND** the config authors `infer.detection_sequence_format`,
  `infer.row_separator`, `infer.compact_full_parse_mode`, or
  `infer.parsing.compact_full`
- **WHEN** config parsing or runtime resolution runs
- **THEN** validation fails before generation starts
- **AND** the error points to `detection_template.id` as the supported source of
  truth.

#### Scenario: Derived prompt pattern matches renderer
- **GIVEN** any supported compact `detection_template.id`
- **WHEN** dense prompts are resolved
- **THEN** the compact example row in the prompt matches the canonical renderer
  for that id.

### Requirement: Strict template-specific parsing
The system SHALL select the parser used for training validation, inference
materialization, and post-hoc mAP from `detection_template.id`, and SHALL
enforce that template's canonical structural tokens.

Compact parsers MUST NOT auto-detect another compact variant from generated
text. Missing closure tokens in a closed variant, unexpected closure tokens in
`compact`, or newline separators in non-line variants MUST produce a structured
parse failure.

#### Scenario: Closed variant rejects missing box end
- **GIVEN** `detection_template.id: compact_box_closed`
- **WHEN** generated compact text omits `<|box_end|>` after the four coord
  tokens
- **THEN** parsing fails with a template-specific structural error.

#### Scenario: Compact rejects closure-token output
- **GIVEN** `detection_template.id: compact`
- **WHEN** generated compact text includes `<|box_end|>` after a row
- **THEN** parsing fails instead of accepting the output as another variant.

### Requirement: Template-derived tokenizer and token-row contract
The tokenizer MUST contain the structural tokens required by the selected
template, and each required structural token MUST resolve to a single token id.
The native-token constants for compact structural rows SHALL include:

- `<|object_ref_start|>` id `151646`,
- `<|object_ref_end|>` id `151647`,
- `<|box_start|>` id `151648`,
- `<|box_end|>` id `151649`.

When compact detection token-row adaptation is enabled, the trainable row set
MUST be derived from `detection_template.id`:

- `compact`: 1002 rows, consisting of 1000 coord-token rows plus
  `<|object_ref_start|>` and `<|box_start|>`.
- `compact_box_closed`: 1003 rows, adding `<|box_end|>`.
- `compact_object_box_closed`: 1004 rows, adding `<|object_ref_end|>` and
  `<|box_end|>`.
- `compact_object_box_closed_lines`: 1004 rows, adding
  `<|object_ref_end|>` and `<|box_end|>`.

#### Scenario: Box-closed variant requires box-end row
- **GIVEN** `detection_template.id: compact_box_closed`
- **AND** token-row adaptation is enabled
- **WHEN** token-row validation runs
- **THEN** the required row set contains `<|box_end|>`
- **AND** validation fails if that structural row is not trainable.

#### Scenario: Object-box-closed variants require four structural rows
- **GIVEN** `detection_template.id` is either `compact_object_box_closed` or
  `compact_object_box_closed_lines`
- **AND** token-row adaptation is enabled
- **WHEN** token-row validation runs
- **THEN** the required row set contains `<|object_ref_start|>`,
  `<|object_ref_end|>`, `<|box_start|>`, and `<|box_end|>`
- **AND** the total required row count is 1004.

### Requirement: Template id provenance in hashes and artifacts
The system SHALL include the resolved `detection_template.id` in the bounded
metadata carriers that depend on compact assistant serialization:

- encoded-sample cache fingerprints,
- prompt hashes,
- training resolved config where detection templates are resolved,
- inference `resolved_config.json`,
- inference `summary.json`,
- inference `gt_vs_pred.jsonl` records as `detection_template_id`.

Metric output artifacts such as `metrics.json`, `per_image.json`, `matches.jsonl`,
and `per_class.csv` MUST NOT gain new template fields unless they also carry raw
compact generated text.

Artifacts missing a required template id MUST fail post-hoc mAP preflight with
an actionable error. The system MUST NOT infer the template from generated text
when scoring post-hoc mAP.

Metadata-only migration from historical artifacts SHALL be permitted only when
rewriting legacy `compact_full` metadata to `compact` for artifacts whose
serialized assistant bytes and 1002-row adapter contract already match
`compact`. Closed compact variants MUST require regenerated artifacts or
checkpoints validated with their template-derived 1003 or 1004 row contract.

#### Scenario: Prompt hashes differ by template id
- **GIVEN** two inference configs that differ only by compact
  `detection_template.id`
- **WHEN** prompt metadata hashes are computed
- **THEN** the hashes differ.

#### Scenario: Training cache fingerprints differ by template id
- **GIVEN** two training configs that differ only by compact
  `detection_template.id`
- **WHEN** encoded-sample cache or training packing fingerprints are computed
- **THEN** the fingerprints differ.

#### Scenario: Missing artifact template id fails post-hoc mAP preflight
- **GIVEN** an inference artifact family produced after this change
- **AND** the artifact family does not record the resolved
  `detection_template.id`
- **WHEN** post-hoc mAP evaluation is requested
- **THEN** evaluation fails before scoring
- **AND** the error instructs the operator to edit or regenerate the artifact
  metadata.

#### Scenario: Metadata-only migration is compact-only
- **GIVEN** a historical artifact whose metadata says `compact_full`
- **WHEN** the operator manually repairs metadata without regenerating model
  output
- **THEN** the only valid semantic target is `compact`
- **AND** `compact_box_closed`, `compact_object_box_closed`, and
  `compact_object_box_closed_lines` require regenerated output or checkpoint
  validation for their row contracts.
