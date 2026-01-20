# inference-pipeline Specification

## Purpose
TBD - created by archiving change refactor-inference-viz-decouple. Update Purpose after archive.
## Requirements
### Requirement: Inference runner emits robust JSONL
- The system SHALL provide an inference-only CLI (same input schema as `vis_coordexp`) that loads a checkpoint, runs generation over the input JSONL (single-image records only), and writes exactly one valid JSON object per input record (no visualization side effects, no batching required).
- Each line SHALL include: `image_id` (input order), `images` (original list), `width`, `height`, `coord_mode` set to `"norm1000"` (model always outputs 0–999), `raw_output` (verbatim model text), mandatory `preds` (parsed objects) with `score: 1.0`, and optional `error` when parsing/validation fails; `preds` SHALL be `[]` on errors while retaining the flawed raw output for debugging.
- The runner SHALL attempt aggressive JSON repair (quotes/brackets/trailing commas) on malformed/truncated generations; if still invalid, it SHALL emit the line with `error` populated and continue without aborting the batch.
- Extra `score` hints from the model are ignored; generation order defines export order when scores tie.
- Parsed geometries in `preds` SHALL be limited to `bbox_2d` or `poly`; any `line` geometry in the raw output is invalid and MUST be excluded from `preds` with an error recorded.
- If a multi-image record is encountered, the runner SHALL mark the record with an `error` (e.g., `multi_image_not_supported`), skip generation, and continue.

#### Scenario: Malformed generation still yields JSON
- GIVEN a sample whose generated text is truncated mid-JSON
- WHEN the inference runner processes it
- THEN it writes a valid JSON line with `preds: []`, retains `raw_output`, sets `error`, `coord_mode: "norm1000"`, and the job continues.

#### Scenario: Coord mode recorded for downstream scaling
- GIVEN a model that outputs 0–999 coord tokens
- WHEN the inference runner writes the JSONL
- THEN it sets `coord_mode: "norm1000"` and includes `width`/`height`, enabling downstream tools to convert to pixels without guessing.

### Requirement: Staged pipeline (inference → eval and/or viz)
- Evaluation and visualization tools SHALL consume the inference JSONL directly using the emitted `preds`; they MAY reparse from `raw_output` via the shared parser as a fallback but shall not require re-running the model.
- The pipeline SHALL support running evaluation and visualization in parallel off the same inference artifact (`predictions.jsonl`) to enable checkpoint ablations.
- The inference runner SHALL expose baseline decoding parameters (e.g., temperature, top_p, repetition_penalty, max_new_tokens, seed, device selection) and an output path so runs are reproducible and comparable.

#### Scenario: One inference feed drives both eval and viz
- GIVEN an inference run that produces `predictions.jsonl`
- WHEN the user runs the detection evaluator and the visualization tool against that file
- THEN both complete without invoking the model, using the same parsed geometry and metadata, enabling apples-to-apples comparison across checkpoints.
