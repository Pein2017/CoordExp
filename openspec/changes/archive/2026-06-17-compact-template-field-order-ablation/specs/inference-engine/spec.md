## ADDED Requirements

### Requirement: Inference materialization uses compact template and field order
Inference SHALL parse compact generated text using the resolved compact template
id and object field order recorded for the run.

Artifacts that contain compact generated text MUST persist both axes before
metric-bearing materialization:

- `detection_template_id`
- `object_field_order`

Inference MUST NOT auto-detect bbox-first versus desc-first from raw generated
text when producing comparable `gt_vs_pred.jsonl` artifacts.

#### Scenario: Bbox-first inference parses by configured order
- **GIVEN** an inference run with `detection_template.id:
  compact_object_box_closed`
- **AND** resolved object field order `geometry_first`
- **WHEN** generation returns a rich bbox-first compact row
- **THEN** materialization parses it into normalized prediction objects
- **AND** a desc-first row is rejected as a contract mismatch.

#### Scenario: Inference artifact records both axes
- **GIVEN** a compact inference run
- **WHEN** `resolved_config.json`, `summary.json`, and `gt_vs_pred.jsonl` are
  written
- **THEN** the artifacts record both detection template id and object field
  order.
