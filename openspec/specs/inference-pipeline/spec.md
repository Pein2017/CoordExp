# inference-pipeline Specification

## Purpose
TBD - created by archiving change refactor-inference-viz-decouple. Update Purpose after archive.
## Requirements
### Requirement: Inference runner emits robust JSONL
- Each output line SHALL include `preds` as parsed objects.
- Supported geometry types in `preds` SHALL be limited to `bbox_2d` and `poly`.
- Any `line` geometry in parsed output SHALL be treated as invalid geometry and excluded from `preds`.

#### Scenario: Line in generation does not appear in preds
- GIVEN a sample whose generated text contains a `line` object
- WHEN the inference runner processes it
- THEN it excludes the `line` object from `preds` while retaining the verbatim `raw_output` for debugging.

### Requirement: Staged pipeline (inference → eval and/or viz)
- Evaluation and visualization tools SHALL consume the inference JSONL directly using the emitted `preds`; they MAY reparse from `raw_output` via the shared parser as a fallback but shall not require re-running the model.
- The pipeline SHALL support running evaluation and visualization in parallel off the same inference artifact (`predictions.jsonl`) to enable checkpoint ablations.
- The inference runner SHALL expose baseline decoding parameters (e.g., temperature, top_p, repetition_penalty, max_new_tokens, seed, device selection) and an output path so runs are reproducible and comparable.

#### Scenario: One inference feed drives both eval and viz
- GIVEN an inference run that produces `predictions.jsonl`
- WHEN the user runs the detection evaluator and the visualization tool against that file
- THEN both complete without invoking the model, using the same parsed geometry and metadata, enabling apples-to-apples comparison across checkpoints.

