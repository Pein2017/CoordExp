## MODIFIED Requirements

### Requirement: Inference runner emits robust JSONL
- Each output line SHALL include `preds` as parsed objects.
- Supported geometry types in `preds` SHALL be limited to `bbox_2d` and `poly`.
- Any `line` geometry in parsed output SHALL be treated as invalid geometry and excluded from `preds`.

#### Scenario: Line in generation does not appear in preds
- GIVEN a sample whose generated text contains a `line` object
- WHEN the inference runner processes it
- THEN it excludes the `line` object from `preds` while retaining the verbatim `raw_output` for debugging.

