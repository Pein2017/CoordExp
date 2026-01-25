## MODIFIED Requirements

### Requirement: Ingestion and validation
For the unified pipeline workflow, the evaluator SHALL treat the pipeline artifact `gt_vs_pred.jsonl` (containing embedded `gt` and `pred` per sample) as the primary evaluation input.

Coordinate handling:
- If a record contains `coord_mode: "pixel"`, the evaluator SHALL interpret `gt` and `pred` `points` as pixel-space coordinates and SHALL NOT denormalize again.
- If a record contains `coord_mode: "norm1000"`, the evaluator SHALL denormalize using per-record `width` and `height` and then clamp/round.
- Records missing `width` or `height` (or with null width/height) SHALL be skipped (counted) because denormalization and validation are undefined.

#### Scenario: Pixel-ready artifact is evaluated without rescaling
- **GIVEN** a `gt_vs_pred.jsonl` record with `coord_mode: "pixel"` and valid `width`/`height`
- **WHEN** the evaluator ingests the record
- **THEN** it evaluates `gt` and `pred` using the provided pixel coordinates without denormalization.

### Requirement: CLI, configuration, and outputs (YAML-first)
The evaluator SHALL support a YAML config template under `configs/eval/` and SHOULD accept `--config` to run evaluation reproducibly.

If both CLI flags and YAML are provided, CLI flags SHALL override YAML values, and the evaluator SHALL log the resolved configuration.

#### Scenario: Evaluate via YAML config
- **GIVEN** `configs/eval/detection.yaml` and a prediction JSONL artifact
- **WHEN** the user runs evaluator with `--config configs/eval/detection.yaml`
- **THEN** it produces `metrics.json`, `per_class.csv`, `per_image.json` under the configured output directory.

### Requirement: Inference JSONL as the only prediction+GT input
The evaluator SHALL treat the inference output JSONL (containing `gt` and `pred` per sample, pixel-ready geometries) as the only supported input format for evaluation in the unified pipeline.

The evaluator MAY retain internal helpers for other ingestion modes for backward compatibility, but the public CLI/pipeline integration SHALL NOT expose a separate-GT mode (`gt_jsonl` separate from predictions) as part of this change.

#### Scenario: Evaluate directly from inference dump
- **WHEN** the user evaluates the `gt_vs_pred.jsonl` produced by the inference engine
- **THEN** the evaluator uses the embedded pixel-space `gt` and `pred` geometries without re-scaling and without requiring a separate GT path.


## ADDED Requirements

### Requirement: Pipeline integration
The evaluator SHALL be callable as a stage from the unified inference pipeline runner, using the same resolved run directory and artifact conventions.

#### Scenario: Pipeline stage calls evaluator
- **GIVEN** a unified pipeline run directory containing `gt_vs_pred.jsonl`
- **WHEN** the pipeline runner executes the eval stage
- **THEN** evaluation outputs are written under the run directory (or a deterministic subdirectory) without requiring additional user inputs.
