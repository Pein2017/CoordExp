# inference-engine Specification

## Purpose
TBD - created by archiving change add-centralized-inference-engine. Update Purpose after archive.
## Requirements
### Requirement: Unified inference CLI
The system SHALL provide an inference entrypoint that requires `--gt_jsonl`, `--model_checkpoint`, `--mode` (`coord` or `text`), output path, device, limit, seed, and generation flags (`--temperature`, `--top_p`, `--max_new_tokens`, `--repetition_penalty`). Generation config MUST be supplied via CLI flags (no external config files), and `--mode` MUST be provided explicitly (no auto-detect); missing `--mode` SHALL fail fast.

#### Scenario: Run inference with required flags
- **WHEN** a user runs the inference CLI with `--gt_jsonl`, `--model_checkpoint`, `--mode text`, output path, and generation flags
- **THEN** the run succeeds without prompting for missing mode/config files and processes samples up to the optional `--limit`.

### Requirement: Coord-mode scaling and validation
In `coord` mode, the engine SHALL treat both GT and predictions as normalized (0-999) coordinates, enforce the 0-999 range, require `width` and `height`, denormalize to absolute pixels using those dimensions, and skip any sample with invalid coords or missing size while recording an error.

#### Scenario: Coord mode with valid tokens
- **WHEN** GT and predictions contain coord tokens within 0-999 and width/height are present
- **THEN** the engine denormalizes both GT and preds to pixel-space and emits a valid output line.

#### Scenario: Coord mode with out-of-range token
- **WHEN** a prediction contains a coord value outside 0-999
- **THEN** the sample is skipped for predictions, an error is recorded for that sample, and processing continues.

### Requirement: Text-mode scaling and validation
In `text` mode, the engine SHALL treat GT coordinates as absolute pixels (no scaling) and denormalize predictions that are expressed in norm1000 (tokens or 0-999 ints); predictions already in pixel-space MUST be accepted. Samples with malformed geometry (odd point counts, missing size) MUST be skipped with an error recorded. If GT is detected as normalized in text mode (or pixel GT in coord mode), the sample SHALL record a `mode_gt_mismatch` error and skip predictions.

#### Scenario: Text mode with normalized preds
- **WHEN** GT boxes are in pixels and predictions are 0-999 ints/tokens
- **THEN** predictions are denormalized to pixels using GT width/height and emitted; GT remains unchanged.

#### Scenario: Text mode with pixel preds
- **WHEN** GT is in pixels and predictions are pixel values (no tokens)
- **THEN** predictions are accepted as-is (clamped to image bounds) and emitted; GT remains unchanged.

### Requirement: Unified output schema
Each output line in `pred.jsonl` SHALL contain `gt` and `pred` arrays of objects with fields `type` (`bbox_2d` or `poly` only), `points` (absolute pixel coordinates), `desc` (label string), and `score` fixed at 1.0, plus top-level `width`, `height`, `image`, `mode`, optional `coord_mode` for trace/debug, `raw_output`, and an `errors` list (empty when none). Legacy mixed-format fields (e.g., raw norm `predictions`/dual schemas) SHALL NOT be emitted.

#### Scenario: Successful sample output
- **WHEN** a sample is processed without errors
- **THEN** the JSONL line includes `gt` and `pred` arrays with pixel `points`, `desc`, `type`, `score:1.0`, along with `width`, `height`, `image`, `mode`, and an empty `errors` list.

#### Scenario: Error sample output
- **WHEN** a sample fails validation (e.g., missing height)
- **THEN** the JSONL line contains an `errors` list describing the issue, `pred` is empty, and processing continues for subsequent samples.

### Requirement: Polygon preservation and evaluation
Polygons (`poly`) SHALL be preserved in outputs and evaluated via COCO-style polygon segmentation (mask IoU) derived from the vertex list (single ring, clamped, non-degenerate). Bounding boxes MAY be derived for ancillary needs but SHALL NOT replace the polygon geometry in the output schema.

#### Scenario: Polygon prediction
- **WHEN** a prediction is a `poly` with valid vertices
- **THEN** the output keeps the polygon vertex list in pixels, and downstream evaluation uses COCO segmentation/mask IoU derived from those vertices.

### Requirement: Deterministic generation
When `--seed` is provided, the engine SHALL seed torch (and CUDA) generators and pass a seeded `torch.Generator` into `model.generate` so that repeated runs with the same inputs and flags yield identical `pred.jsonl` outputs.

#### Scenario: Reproducible runs
- **WHEN** inference is executed twice with identical inputs and `--seed 1234`
- **THEN** the produced `pred.jsonl` files are byte-identical (ordering preserved, floating points only from deterministic scaling/rounding).

### Requirement: Limit handling
If `--limit N` is set, the engine SHALL process and emit at most N samples (images) from the GT JSONL, maintaining alignment between read GT records and emitted outputs.

#### Scenario: Limited run
- **WHEN** `--limit 5` is passed
- **THEN** only the first 5 samples are processed and written to `pred.jsonl`, and subsequent samples are ignored.

### Requirement: Run-level counters and summary
The engine SHALL aggregate counters (e.g., invalid_json, invalid_geometry, invalid_coord, size_mismatch, empty_pred, mode_gt_mismatch) and emit a summary artifact (`pred.summary.json`) alongside `pred.jsonl`, containing at least the counters map, mode, total samples read, total samples emitted, and distinct error codes seen.

#### Scenario: Summary emitted
- **WHEN** inference completes
- **THEN** a summary file containing counters is written next to `pred.jsonl`.

### Requirement: Pixel-ready downstream consumption
The emitted `pred.jsonl` SHALL already contain pixel-space `gt` and `pred` geometries so that downstream visualization and evaluation can load it without further normalization or scaling.

#### Scenario: Direct evaluator load
- **WHEN** the evaluator or visualizer reads the produced `pred.jsonl`
- **THEN** it can use the pixel-space geometries directly (no denorm or mode inference needed) to render or compute metrics.
