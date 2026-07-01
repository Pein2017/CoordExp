## ADDED Requirements

### Requirement: Template-aware post-hoc mAP over normalized artifacts
The detection evaluator SHALL support post-hoc mAP for all supported compact
detection template variants by accepting inference artifacts whose generated
compact text was already parsed and normalized during inference materialization.

Normative behavior:

- the evaluator MUST read the resolved template metadata from the inference
  artifact family during preflight,
- the evaluator MUST require post-change `gt_vs_pred.jsonl` records to include
  `detection_template_id`,
- standard post-hoc mAP MUST score the normalized `gt` and `pred` object arrays
  already present in `gt_vs_pred.jsonl`,
- standard post-hoc mAP MUST NOT reparse raw compact generated text,
- the evaluator MUST NOT infer compact template variants from raw generated text,
- artifacts that lack a required template id MUST fail before scoring with an
  actionable metadata error,
- mAP metric semantics MUST remain independent of the compact wrapper variant
  because scoring receives the same normalized object schema.

#### Scenario: Post-hoc mAP accepts materialized box-closed output
- **GIVEN** an inference artifact family that records
  `detection_template_id: compact_box_closed`
- **AND** `gt_vs_pred.jsonl` contains normalized pixel-space `pred` objects
- **WHEN** post-hoc mAP evaluation runs
- **THEN** the evaluator scores the normalized objects through the existing mAP
  path
- **AND** it does not reparse raw compact text.

#### Scenario: Missing template metadata fails before scoring
- **GIVEN** a post-change inference artifact family
- **AND** no resolved detection template id is recorded
- **WHEN** post-hoc mAP evaluation runs
- **THEN** evaluation fails before metric computation
- **AND** the error says that artifact metadata must be edited or regenerated.

#### Scenario: Wrapper variant does not change mAP semantics
- **GIVEN** two artifact families whose parsed predictions normalize to the same
  object list and pixel geometry
- **AND** the artifact families use different supported compact template ids
- **WHEN** post-hoc mAP evaluation runs on each artifact family
- **THEN** the metric computation receives equivalent normalized predictions
- **AND** the resulting mAP values are equal for the same ground truth.

#### Scenario: Raw compact text alone is not a post-hoc mAP input
- **GIVEN** an artifact contains raw compact generated text but does not contain
  normalized `pred` objects
- **WHEN** standard post-hoc mAP evaluation runs
- **THEN** evaluation fails before scoring
- **AND** the error instructs the operator to run inference materialization or a
  separately specified raw-generation parser workflow.
