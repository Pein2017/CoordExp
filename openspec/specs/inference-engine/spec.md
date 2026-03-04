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

#### Scenario: Run inference with required flags
- **WHEN** a user runs the inference CLI with `--gt_jsonl`, `--model_checkpoint`, `--mode text`, output path, and generation flags
- **THEN** the run succeeds without prompting for missing mode/config files and processes samples up to the optional `--limit`.


### Requirement: Coord-mode scaling and validation
In `coord` mode, the engine SHALL standardize both GT and predictions to pixel coordinates using per-sample `width`/`height`.

Mode behavior:
- GT MAY be represented as norm1000 (`<|coord_k|>` / 0..999 ints) or as pixel coordinates; the scaler SHALL auto-detect GT representation per geometry and convert to pixels.
- Predictions default to norm1000 interpretation in `coord` mode unless `infer.pred_coord_mode` explicitly overrides prediction interpretation.
- Operator-controlled input violations (e.g., missing/invalid `width` or `height`, unreadable image, invalid GT geometry) MUST fail fast during preflight before generation starts.
- Prediction parsing/validation failures on otherwise valid input records (e.g., invalid predicted coord ranges) SHALL be emitted as sample-scoped errors and continue processing subsequent records.

#### Scenario: Coord mode with valid tokens
- **WHEN** GT and predictions contain coord tokens within 0-999 and width/height are present
- **THEN** the engine denormalizes both GT and preds to pixel-space and emits a valid output line.

#### Scenario: Coord mode with out-of-range token
- **WHEN** a prediction contains a coord value outside 0-999
- **THEN** the sample is skipped for predictions, an error is recorded for that sample, and processing continues.

### Requirement: Text-mode scaling and validation
In `text` mode, the engine SHALL treat GT coordinates as absolute pixels (no scaling), and SHALL auto-detect prediction coordinates as pixel or norm1000 (unless overridden by `infer.pred_coord_mode`) before converting to pixels.

Mode behavior:
- GT containing coord tokens in `text` mode SHALL be rejected with `mode_gt_mismatch`.
- Predictions expressed in norm1000 SHALL be denormalized to pixels; predictions already in pixel-space SHALL be accepted (with clamp/round normalization).
- Operator-controlled input violations are handled by preflight fail-fast; prediction-side malformed geometry SHALL be emitted as sample-scoped errors.

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
  - `desc` (label string).
- top-level `width` and `height` keys (keys SHALL be present; ints when available, null allowed on error records), `image`, `mode`, `coord_mode` (used as a downstream hint; may be null on error records), `raw_output_json` (object or null), `raw_special_tokens` (list[string]), `raw_ends_with_im_end` (bool), an `errors` list (canonical error codes), and `error_entries` (structured error objects containing at least `code`, `message`, `stage`).

Legacy mixed-format fields (e.g., raw norm `predictions`/dual schemas) SHALL NOT be emitted by the unified pipeline.

#### Scenario: Successful sample output
- **WHEN** a sample is processed without errors
- **THEN** the JSONL line includes `gt` and `pred` arrays with pixel `points`, `desc`, `type`, along with `width`, `height`, `image`, `mode`, `coord_mode`, `raw_output_json`, and empty `errors`/`error_entries` lists.

#### Scenario: Error sample output
- **WHEN** prediction output for a valid input sample fails validation (e.g., out-of-range predicted coord token)
- **THEN** the JSONL line contains `errors`/`error_entries`, `pred` is empty, and processing continues for subsequent samples.

#### Scenario: Line in raw output is dropped
- **GIVEN** a generated JSON object that includes a `line` geometry
- **WHEN** the inference engine parses and validates predictions
- **THEN** the `line` object is excluded from `pred`, and an error/counter reflects invalid geometry.


### Requirement: Polygon preservation and evaluation
Polygons (`poly`) SHALL be preserved in outputs and evaluated via COCO-style polygon segmentation (mask IoU) derived from the vertex list (single ring, clamped, non-degenerate). Bounding boxes MAY be derived for ancillary needs but SHALL NOT replace the polygon geometry in the output schema.

#### Scenario: Polygon prediction
- **WHEN** a prediction is a `poly` with valid vertices
- **THEN** the output keeps the polygon vertex list in pixels, and downstream evaluation uses COCO segmentation/mask IoU derived from those vertices.

### Requirement: Preflight input validation is fail-fast
Before any model loading/generation work, inference SHALL run a strict preflight over operator-controlled inputs.

Preflight contract:
- GT JSONL records MUST be valid JSON objects (malformed/non-object records fail fast with file:line diagnostics).
- `width` and `height` MUST be present and positive integers.
- `images` MUST contain exactly one resolvable/readable image path per record.
- GT geometry MUST pass mode-aware validation for the resolved mode.
- If preflight finds any violations, inference MUST terminate before writing partial prediction artifacts.

#### Scenario: Invalid operator input fails before generation
- **GIVEN** an input JSONL containing an invalid `width` value
- **WHEN** inference starts
- **THEN** preflight fails fast with actionable diagnostics and no generation loop is executed.

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

#### Scenario: Reproducible runs
- **WHEN** inference is executed twice with identical inputs and `--seed 1234`
- **THEN** the produced `pred.jsonl` files are byte-identical (ordering preserved, floating points only from deterministic scaling/rounding).


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
- Malformed JSON, non-object records, or invalid/missing `width`/`height` MUST fail fast with file:line diagnostics.
- Records with no GT objects MAY be skipped during scanning.
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


### Requirement: `line` geometry is rejected from standardized predictions
Line geometries MAY appear in model output payloads, but they SHALL be treated as unsupported geometry for the standardized artifact contract.

#### Scenario: Line present
- **WHEN** a `line` object appears in predictions
- **THEN** it is not included in `pred`, and it contributes to invalid-geometry diagnostics/counters.


### Requirement: HF attention backend selection is resilient across environments
Inference-engine SHALL support resilient HF attention backend selection.
If a preferred backend is unavailable in the runtime environment, engine initialization MUST fall back to a supported backend with explicit diagnostics while preserving output contract semantics.
The selected backend (including fallback choice when applied) MUST be recorded in run artifacts via exact `summary.json` fields:
- `backend.attn_implementation_requested`
- `backend.attn_implementation_selected`

`resolved_config.json` MAY mirror these values, but `summary.json` fields are the required compatibility surface for this contract.

#### Scenario: Missing preferred attention backend falls back with warning
- **GIVEN** HF backend inference configuration prefers an unavailable attention backend
- **WHEN** model initialization runs
- **THEN** the engine selects a supported fallback backend
- **AND** emits explicit diagnostics without changing output artifact schema.

#### Scenario: Selected attention backend is captured in run artifacts
- **GIVEN** inference runs under either preferred or fallback attention backend
- **WHEN** artifacts are persisted
- **THEN** `summary.json.backend.attn_implementation_requested` and `summary.json.backend.attn_implementation_selected` are present
- **AND** operators can determine from artifacts whether fallback occurred by comparing requested vs selected values.


### Requirement: Backend runtime is selected through an explicit backend contract
Inference-engine SHALL use an explicit backend runtime contract to isolate backend-specific generation details from artifact standardization.
All backend runtimes MUST produce standardized prediction payloads consumed by shared post-processing.

#### Scenario: Backend runtime swap preserves standardized output payload
- **GIVEN** equivalent inputs and generation settings
- **WHEN** backend runtime selection changes via config
- **THEN** standardized output payload fields remain compatible with shared post-processing and artifact writers.


### Requirement: Inference error reporting remains structured and sample-scoped
Inference-engine SHALL preserve structured, per-sample error reporting in output artifacts and summary counters.

Normative behavior:
- Per-sample error metadata MUST use stable error codes and stage identifiers.
- Run-level summaries MUST include machine-readable aggregate counters by error class/code.
- Logs alone MUST NOT be the only error signal when structured artifacts are emitted.

#### Scenario: Sample-level generation failure is reflected in structured errors
- **GIVEN** generation fails for one sample
- **WHEN** inference continues for the batch/run
- **THEN** the failed sample includes a structured error entry
- **AND** summary counters include the failure classification.

### Requirement: Operator-controlled input violations fail fast in inference/eval
Operator-controlled inference/eval input violations MUST terminate the run and MUST NOT be silently skipped.

Normative behavior:
- Input contract checks SHOULD run in a preflight phase before generation/evaluation work.
- Violations (schema/JSONL/image/size/geometry contract failures) MUST terminate non-zero.
- Implementations MAY aggregate a bounded set of actionable diagnostics before raising.

#### Scenario: Missing required metadata terminates inference
- **GIVEN** inference/eval input records
- **WHEN** a required field such as `width`/`height` is missing or invalid
- **THEN** inference terminates non-zero with actionable diagnostics
- **AND** processing does not continue by silently skipping that sample.

### Requirement: Model-output parse/validation failures are continue-but-observable
Prediction parse/validation failures caused by model-generated output MAY continue per sample, but MUST remain observable.

Normative behavior:
- Parse/validation failures on already-produced `pred_text` MUST emit structured sample errors and increment run counters.
- Invalid predicted objects MAY be dropped for that sample; subsequent samples MAY continue.
- Continue-and-skip under this rule is limited to explicit model-output consumer behavior and does not apply to operator input contracts.

#### Scenario: Invalid prediction text yields sample-scoped error and continue
- **GIVEN** generation produced `pred_text` for a sample
- **WHEN** parsing/validation of that `pred_text` fails
- **THEN** the sample record includes structured error metadata and error counters increment
- **AND** subsequent samples continue.

### Requirement: Unexpected internal exceptions and unrecoverable generation failures terminate the run
Unexpected internal exceptions during inference/eval, including generation failures that prevent usable `pred_text`, MUST terminate the run non-zero.

Normative behavior:
- Internal/runtime failures MUST NOT be converted into silent success outputs.
- Unexpected exceptions MAY be annotated with diagnostics before re-raise, but the run MUST terminate.

#### Scenario: Backend generation failure terminates inference
- **WHEN** the generation backend fails for a sample before producing usable prediction text
- **THEN** inference terminates non-zero
- **AND** the failure is not converted into empty predictions with continued execution.
