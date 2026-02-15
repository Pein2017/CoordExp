# inference-engine Specification

## Purpose
Define the unified inference entrypoint contract (`scripts/run_infer.py`) including the YAML schema, artifact outputs, and backend selection behavior.

## Requirements
### Requirement: Unified inference CLI
The system SHALL provide a single inference entrypoint that is primarily configured via YAML under `configs/` (with a minimal CLI wrapper that accepts `--config`).

The entrypoint MUST:
- log the resolved configuration (including resolved mode, resolved generation settings, and resolved output paths),
- write canonical inference artifacts under a deterministic run directory,
- remain compatible with the Qwen3-VL chat template behavior.

Entrypoint:
- The unified inference entrypoint SHALL be `scripts/run_infer.py`.

Minimum YAML schema (normative; exact key names MAY differ if documented consistently):
- The YAML MUST provide either:
  - `artifacts.run_dir` (string), OR
  - `run.name` (string) and `run.output_dir` (string) to construct a canonical run directory.
- `infer.gt_jsonl` (string) as dataset input,
- `infer.model_checkpoint` (string) (+ optional adapters, if supported),
- `infer.backend.type` (`hf` or `vllm`),
- `infer.mode` (`coord` | `text` | `auto`) and `infer.pred_coord_mode` (`auto` | `norm1000` | `pixel`),
- `infer.generation` settings (temperature, top_p, max_new_tokens, repetition_penalty, seed),
- `infer.device` and `infer.limit`.

Output defaults:
- Canonical run directory: `output/infer/<run_name>/` unless explicitly overridden.
- Canonical artifacts (paths relative to run directory unless user overrides):
  - `gt_vs_pred.jsonl`
  - `summary.json`

Single-file configuration:
- The YAML configuration SHALL be treated as a single file (no config inheritance such as `extends`/`inherit`).
- The YAML configuration SHALL NOT require variable interpolation; derived paths (e.g., `run_dir`) are computed by the runner when not explicitly provided.

Transition support:
- The entrypoint MAY continue to accept legacy CLI flags during a transition period.
- If both `--config` and legacy CLI flags are provided, legacy CLI flags SHALL override YAML values, and the resolved configuration SHALL be logged.

#### Scenario: Run inference via YAML config
- **WHEN** a user runs the inference entrypoint with `--config configs/infer/<exp>.yaml`
- **THEN** the run executes and writes `gt_vs_pred.jsonl` and `summary.json` under the resolved run directory.

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
Each output line in `gt_vs_pred.jsonl` SHALL contain:
- `gt` and `pred` arrays of objects with fields:
  - `type` (`bbox_2d` or `poly`),
  - `points` (absolute pixel coordinates),
  - `desc` (label string),
  - `score` fixed at 1.0.
- top-level `width` and `height` keys (keys SHALL be present; ints when available, null allowed on error records), `image`, `mode`, `coord_mode` (used as a downstream hint; may be null on error records), `raw_output`, and an `errors` list (empty when none).

Legacy mixed-format fields (e.g., raw norm `predictions`/dual schemas) SHALL NOT be emitted by the unified pipeline.

#### Scenario: Successful sample output
- **WHEN** a sample is processed without errors
- **THEN** the JSONL line includes `gt` and `pred` arrays with pixel `points`, `desc`, `type`, `score:1.0`, along with `width`, `height`, `image`, `mode`, `coord_mode`, and an empty `errors` list.

#### Scenario: Error sample output
- **WHEN** a sample fails validation (e.g., missing height)
- **THEN** the JSONL line contains an `errors` list describing the issue, `width`/`height` keys are present (values may be null), `pred` is empty, and processing continues for subsequent samples.

### Requirement: Polygon preservation and evaluation
Polygons (`poly`) SHALL be preserved in outputs and evaluated via COCO-style polygon segmentation (mask IoU) derived from the vertex list (single ring, clamped, non-degenerate). Bounding boxes MAY be derived for ancillary needs but SHALL NOT replace the polygon geometry in the output schema.

#### Scenario: Polygon prediction
- **WHEN** a prediction is a `poly` with valid vertices
- **THEN** the output keeps the polygon vertex list in pixels, and downstream evaluation uses COCO segmentation/mask IoU derived from those vertices.

### Requirement: Deterministic generation
When a seed is provided, the engine SHALL produce deterministic results for a given backend and resolved generation configuration by seeding torch (and CUDA) in a way compatible with Qwen3-VL model implementations.

The engine SHALL NOT rely on passing a `generator` kwarg into `model.generate()` unless the model is known to accept it; instead, it SHALL seed globally to avoid remote-code incompatibilities.

vLLM determinism scope:
- For `backend.type: vllm`, exact repeatability is NOT required.
- The engine SHOULD pass a seed to vLLM when supported and SHALL record a determinism note in the run summary (e.g., `determinism: best_effort`).

#### Scenario: Reproducible HF runs
- **GIVEN** the same backend (`hf`), the same resolved config (including seed), and the same execution environment
- **WHEN** inference is executed twice with identical inputs and `seed=1234`
- **THEN** the produced `gt_vs_pred.jsonl` files are byte-identical.

#### Scenario: vLLM run records best-effort determinism
- **GIVEN** `backend.type: vllm` and a configured `seed`
- **WHEN** inference is executed
- **THEN** `summary.json` records a determinism note that does not claim byte-identical repeatability (e.g., `determinism: best_effort`).

### Requirement: Limit handling
If `infer.limit N` is set, the engine SHALL process and emit at most N samples (images) from the GT JSONL, maintaining alignment between read GT records and emitted output lines in `gt_vs_pred.jsonl`.

#### Scenario: Limited run
- **WHEN** `infer.limit=5` is configured
- **THEN** only the first 5 samples are processed and written to `gt_vs_pred.jsonl`, and subsequent samples are ignored.

### Requirement: Run-level counters and summary
The engine SHALL aggregate run-level counters (e.g., invalid_json, invalid_geometry, invalid_coord, size_mismatch, empty_pred, mode_gt_mismatch) and emit a summary artifact (`summary.json`) alongside `gt_vs_pred.jsonl`.

The summary SHALL include at least:
- a counters map,
- mode,
- mode resolution reason when `infer.mode=auto` (e.g., `coord_tokens_found`),
- total samples read,
- total samples emitted,
- distinct error codes seen,
- backend metadata (backend type, model checkpoint, resolved generation config, seed).
- a determinism note (e.g., `determinism: strict` for HF runs, `determinism: best_effort` for vLLM runs).

#### Scenario: Summary emitted
- **WHEN** inference completes
- **THEN** a `summary.json` file containing counters and resolved configuration metadata is written under the run directory.

### Requirement: Pixel-ready downstream consumption
The emitted `gt_vs_pred.jsonl` SHALL contain pixel-space `gt` and `pred` geometries (`coord_mode: "pixel"`) so that downstream visualization and evaluation can load it without further normalization or scaling.

#### Scenario: Direct evaluator load
- **WHEN** the evaluator or visualizer reads the produced `gt_vs_pred.jsonl`
- **THEN** it can use the pixel-space geometries directly (no denorm or mode inference needed) to render or compute metrics.

### Requirement: Mode auto-detection (infer.mode=auto)
If `infer.mode` is set to `auto`, the engine SHALL resolve the effective mode deterministically from the GT JSONL.

Auto-detect algorithm (normative):
- Scan the first N records from `infer.gt_jsonl` (default N=128).
- Ignore records that are invalid JSON, have no objects, or do not have valid integer `width` and `height`.
- For each scanned record, inspect GT geometries from `objects` (or `gt` as a fallback):
  - If any geometry contains coord tokens (`"<|coord_...|>"`), resolve mode to `coord` with reason `coord_tokens_found`.
  - Else if any numeric coordinate value exceeds `max(width, height)`, resolve mode to `coord` with reason `points_exceed_image`.
- If no record triggers coord mode, resolve mode to `text` with reason `within_image_bounds`.
- If zero valid records were scanned, resolve mode to `text` with reason `no_valid_records`.

The engine SHALL log the resolved mode and reason and SHALL include them in `summary.json`.

#### Scenario: Auto mode resolves to coord due to coord tokens
- **GIVEN** `infer.mode=auto` and a GT record whose geometry contains `"<|coord_10|>"` tokens
- **WHEN** inference starts
- **THEN** the engine resolves `mode=coord` and logs a reason indicating coord tokens were found.

### Requirement: Generation backends
The inference engine SHALL support:
- `hf` backend (Transformers generation) as the default.
- `vllm` backend as an alternative backend.

All backends MUST emit a `pred_text` (raw generation string) that is subsequently parsed and standardized through shared utilities.

Contract requirements (backend-agnostic):
- The backend implementation MUST NOT change the downstream contract: it MUST write the same `gt_vs_pred.jsonl` schema (canonical keys `gt` and `pred`) and the same artifact layout as the `hf` backend for the same pipeline configuration.
- If a backend is unavailable (e.g., not installed) or incompatible with the resolved model or generation configuration, it MUST fail fast with a clear error that describes how to switch back to `hf` (or how to adjust the config).

#### Scenario: Switch backend without changing downstream stages
- **GIVEN** the same input dataset and resolved generation configuration
- **WHEN** the user switches backend from `hf` to `vllm` in YAML
- **THEN** the pipeline produces the same prediction JSONL schema and artifact layout, enabling evaluation/visualization without schema translation.
