## ADDED Requirements

### Requirement: Detection template metadata and parsing
Inference SHALL resolve compact output parsing from `detection_template.id` and
SHALL persist the resolved detection template id in inference artifacts.

Normative behavior:

- inference configs MUST author detection serialization through
  top-level `detection_template.id` rather than independent compact parse modes
  or row separator knobs under `infer`,
- `infer.detection_sequence_format`, `infer.row_separator`,
  `infer.compact_full_parse_mode`, and `infer.parsing.compact_full` MUST fail
  validation when authored after this change,
- generated compact text MUST be parsed by the strict parser derived from the
  resolved template id,
- `resolved_config.json` MUST record the resolved `detection_template.id`,
- `summary.json` MUST record the resolved `detection_template.id`,
- emitted `gt_vs_pred.jsonl` records MUST include the resolved
  `detection_template_id`,
- artifacts that contain compact generated text but lack a resolved template id
  MUST fail inference materialization or post-hoc preflight with an actionable
  error.

#### Scenario: Inference artifact records compact template id
- **GIVEN** an inference config with
  `detection_template.id: compact_box_closed`
- **WHEN** inference writes resolved config, summary, and `gt_vs_pred.jsonl`
- **THEN** those artifacts record the resolved compact template id
- **AND** downstream post-hoc mAP can recover the template contract without a
  CLI override.

#### Scenario: Generated text parsed by selected template
- **GIVEN** an inference config with
  `detection_template.id: compact_object_box_closed`
- **WHEN** generation returns compact text
- **THEN** the inference materialization path parses that text with the
  `compact_object_box_closed` parser
- **AND** it does not auto-detect or silently accept another compact variant.

#### Scenario: Post-hoc mAP consumes normalized predictions
- **GIVEN** inference has materialized `gt_vs_pred.jsonl` with
  `detection_template_id` and normalized pixel-space `pred` objects
- **WHEN** post-hoc mAP evaluation consumes that artifact
- **THEN** mAP scoring uses the normalized `pred` objects
- **AND** it does not reparse raw compact text.

#### Scenario: Old parse-mode knob is rejected
- **GIVEN** an inference config with a compact `detection_template.id`
- **AND** the config authors `infer.detection_sequence_format`,
  `infer.row_separator`, `infer.compact_full_parse_mode`, or
  `infer.parsing.compact_full`
- **WHEN** config parsing or runtime resolution runs
- **THEN** validation fails before generation starts.
