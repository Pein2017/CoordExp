# inference-pipeline Specification

## Purpose
Define the staged inference pipeline contract (infer -> eval/viz) and the stable JSONL artifact interface between stages.
## Requirements
### Requirement: Inference runner emits robust JSONL
The unified pipeline inference stage SHALL emit a single JSONL artifact (the "pipeline artifact") that is the stable interface between stages.

Default artifact naming:
- The default pipeline artifact file name SHALL be `gt_vs_pred.jsonl` (unless explicitly overridden by config).

Per-record schema:
- Each JSONL line SHALL be a JSON object that includes at least:
  - `image` (string; relative path preferred),
  - `width` (int or null) and `height` (int or null) (keys SHALL be present; null is allowed on error records),
  - `mode` (string; `coord` or `text`),
  - `gt` (list[object]) and `pred` (list[object]),
  - `coord_mode` (string or null) set to `"pixel"` when `gt`/`pred` `points` are already pixel-space,
  - `raw_output` (string; verbatim model text, or empty when inference is skipped),
  - `errors` (list[string]; empty when none).
- `gt` and `pred` objects SHALL use the canonical object schema:
  - `type`: one of `bbox_2d` or `poly`,
  - `points`: flat list of pixel coordinates (ints), length 4 for `bbox_2d`, length ≥ 6 and even for `poly`,
  - `desc`: string (may be empty),
  - `score`: number fixed at 1.0.
- The inference stage SHALL NOT emit legacy top-level keys like `preds`.

Geometry constraints:
- Supported geometry types are limited to `bbox_2d` and `poly`.
- Any `line` geometry present in model output SHALL be treated as invalid geometry and SHALL NOT appear in `pred` (but SHALL remain visible in `raw_output` for debugging).

Compatibility guidance (non-normative):
- Downstream tools MAY accept legacy keys (e.g., `preds`, `predictions`) during a transition window, but `gt`/`pred` are canonical.

#### Scenario: Line in generation does not appear in pred
- **GIVEN** a sample whose generated text contains a `line` object
- **WHEN** the inference stage writes `gt_vs_pred.jsonl`
- **THEN** it excludes the `line` object from `pred` while retaining the verbatim `raw_output`.

#### Scenario: Downstream tools consume pixel-ready geometry
- **GIVEN** a pipeline artifact record with `coord_mode: "pixel"`
- **WHEN** evaluation or visualization consumes that record
- **THEN** it uses `gt` and `pred` `points` as pixel coordinates without denormalization.

#### Scenario: Malformed generation still yields JSON
- GIVEN a sample whose generated text is truncated mid-JSON
- WHEN the inference runner processes it
- THEN it writes a valid JSON line with `preds: []`, retains `raw_output`, sets `error`, `coord_mode: "norm1000"`, and the job continues.

#### Scenario: Coord mode recorded for downstream scaling
- GIVEN a model that outputs 0–999 coord tokens
- WHEN the inference runner writes the JSONL
- THEN it sets `coord_mode: "norm1000"` and includes `width`/`height`, enabling downstream tools to convert to pixels without guessing.

#### Scenario: Line in generation does not appear in preds
- GIVEN a sample whose generated text contains a `line` object
- WHEN the inference runner processes it
- THEN it excludes the `line` object from `preds` while retaining the verbatim `raw_output` for debugging.

### Requirement: Staged pipeline (inference → eval and/or viz)
The system SHALL provide a unified pipeline runner that can execute stages:
- `infer`: run generation and write the pipeline artifact (`gt_vs_pred.jsonl`) and a run summary (`summary.json`),
- `eval`: compute metrics from the pipeline artifact and write outputs under `eval/`,
- `vis`: render qualitative overlays from the pipeline artifact and write outputs under `vis/`.

Evaluation and visualization MUST be runnable without invoking the model, consuming `gt_vs_pred.jsonl` directly.

The runner SHALL be YAML-driven (single config) and SHALL support toggling stages on/off.

Entrypoint:
- The unified pipeline runner SHALL be invokable via `scripts/run_infer.py --config <yaml>`.

Minimum YAML schema (normative; exact key names MAY differ if documented consistently):
- The YAML MAY define stage toggles under `stages`:
  - `stages.infer` (bool), `stages.eval` (bool), `stages.vis` (bool).
  - If `stages` is omitted, the runner SHALL default to `infer=true`, `eval=false`, `vis=false`.
  - If `stages` is provided, it MUST specify all three keys (`infer`, `eval`, `vis`).
- The YAML MUST define (or deterministically resolve) a run directory using one of:
  - `artifacts.run_dir` (string), OR
  - `run.output_dir` (string) + `run.name` (string), where `run_dir = <run.output_dir>/<run.name>`.
  - `artifacts.gt_vs_pred_jsonl` (string), where `run_dir` defaults to the parent directory of `gt_vs_pred.jsonl` when `artifacts.run_dir` and `run.*` are not provided.

Stage-only requirements:
- If `stages.infer=true`, the YAML MUST include an `infer` section with all required inference inputs (dataset, checkpoint, backend, generation config) as defined by the inference entrypoint requirements.
- If `stages.infer=false` and (`stages.eval=true` OR `stages.vis=true`), the YAML MUST resolve an existing pipeline artifact path (`gt_vs_pred.jsonl`) via the precedence rules below and MUST fail fast if the file does not exist.

Precedence rules:
- CLI flags override YAML values.
- Within YAML, explicit artifact paths override derived defaults:
  - `artifacts.run_dir` (if provided) overrides the derived `<run.output_dir>/<run.name>`.
  - `artifacts.gt_vs_pred_jsonl` (if provided) overrides `<artifacts.run_dir>/gt_vs_pred.jsonl` and `<run_dir>/gt_vs_pred.jsonl`.
  - `artifacts.summary_json` (if provided) overrides `<artifacts.run_dir>/summary.json` and `<run_dir>/summary.json`.
  - `eval.output_dir` (if provided) overrides `<run_dir>/eval`.
  - `vis.output_dir` (if provided) overrides `<run_dir>/vis`.

Default artifact layout:
- Canonical run directory: `output/infer/<run_name>/` (unless overridden).
- Canonical paths (relative to the run directory):
  - `gt_vs_pred.jsonl`,
  - `summary.json`,
  - `eval/` (metrics + reports),
  - `vis/` (overlays).

#### Scenario: One run performs infer+eval+vis
- **GIVEN** a YAML config with `stages.infer=true`, `stages.eval=true`, `stages.vis=true`
- **WHEN** the user runs the pipeline runner
- **THEN** it produces `gt_vs_pred.jsonl`, `summary.json`, `eval/*`, and `vis/*` under the resolved run directory.

#### Scenario: Eval run does not load the model
- **GIVEN** an existing `gt_vs_pred.jsonl`
- **WHEN** the user runs the pipeline runner with `stages.infer=false` and `stages.eval=true`
- **THEN** evaluation completes without loading the model.

#### Scenario: Vis run does not load the model
- **GIVEN** an existing `gt_vs_pred.jsonl`
- **WHEN** the user runs the pipeline runner with `stages.infer=false` and `stages.vis=true`
- **THEN** visualization completes without loading the model.

#### Scenario: One inference feed drives both eval and viz
- GIVEN an inference run that produces `predictions.jsonl`
- WHEN the user runs the detection evaluator and the visualization tool against that file
- THEN both complete without invoking the model, using the same parsed geometry and metadata, enabling apples-to-apples comparison across checkpoints.

### Requirement: Pipeline config resolution is separated from stage execution
Inference-pipeline SHALL separate config parsing/resolution (including artifact path precedence and overrides) from stage execution logic.
Resolved stage and artifact decisions MUST be materialized before stage execution begins.

#### Scenario: Stage execution consumes pre-resolved artifact paths
- **GIVEN** a pipeline config with explicit artifact overrides
- **WHEN** the pipeline runs
- **THEN** stage handlers consume pre-resolved artifact paths
- **AND** stage handlers do not recompute precedence rules.

### Requirement: `resolved_config.json` is the canonical resolved manifest
Inference-pipeline SHALL persist resolved run metadata in `resolved_config.json` in the run directory.
The file SHALL include:
- active stage toggles,
- resolved artifact paths,
- a redacted config snapshot,
- `schema_version` as an integer major version (initial major `1` for this contract).

Within a given `schema_version` major, key evolution MUST be additive-only (existing stable keys must not be renamed/removed).
Consumers MUST explicitly reject unsupported major schema versions.
The pipeline SHALL NOT introduce a second parallel manifest artifact for the same contract.

#### Scenario: `resolved_config.json` captures resolved stage/artifact contract
- **WHEN** a pipeline run starts with infer/eval/vis toggles
- **THEN** the run directory includes `resolved_config.json` recording stage toggles and resolved artifact paths
- **AND** downstream stages/tools can consume this file as the single resolved-config source of truth.

#### Scenario: Unsupported manifest major version is rejected explicitly
- **GIVEN** a consumer reads `resolved_config.json` with an unsupported `schema_version` major
- **WHEN** contract validation runs
- **THEN** the consumer rejects the manifest with explicit version-mismatch diagnostics
- **AND** does not silently proceed with partial assumptions.

### Requirement: Relative image-root fallback behavior is explicit and shared across eval/vis
Inference-pipeline SHALL keep compatibility fallback behavior for relative image path resolution while making fallback activation explicit in logs.
The resolved root behavior SHALL be shared consistently for evaluation and visualization consumers so they do not diverge in relative-path handling.
For overlapping active deltas, detailed helper semantics are authoritative in `2026-02-11-src-ambiguity-cleanup`; this change MUST remain consistent with that contract.
Root resolution precedence SHALL be explicit and deterministic: `ROOT_IMAGE_DIR` env override > config root (`run.root_image_dir`) > `infer.gt_jsonl` parent directory > none.
The resolved image-root decision MUST be recorded in `resolved_config.json` with at least `root_image_dir` and `root_image_dir_source` (`env`, `config`, `gt_parent`, or `none`).
This requirement MUST preserve existing downstream contract compatibility for eval/vis flows.

#### Scenario: Eval/vis run with relative paths uses one explicit root-resolution contract
- **GIVEN** an eval+vis run and relative image paths in artifact records
- **WHEN** root fallback is required for path resolution
- **THEN** fallback activation is explicit
- **AND** both evaluation and visualization resolve image paths under the same root-resolution contract
- **AND** `resolved_config.json` records both the chosen `root_image_dir` and its `root_image_dir_source`
- **AND** stage outputs preserve unchanged artifact semantics.

### Requirement: `resolved_config.json` stable keys are explicit and `cfg` snapshot is opaque
Within `schema_version` major `1`, the stable top-level contract key set SHALL be:
- `schema_version`,
- `config_path`,
- `root_image_dir`,
- `root_image_dir_source`,
- `stages`,
- `artifacts`.

The redacted `cfg` snapshot MAY be persisted for diagnostics, but SHALL be treated as opaque/non-contract by downstream consumers.

#### Scenario: Downstream readers validate only stable manifest keys
- **GIVEN** a `resolved_config.json` containing both stable keys and a large redacted `cfg` snapshot
- **WHEN** a downstream consumer validates compatibility
- **THEN** compatibility decisions are based on stable keys and schema version
- **AND** changes inside `cfg` do not break the manifest contract.

### Requirement: Image-path resolution logic is shared across infer stages
Pipeline stages that resolve image paths for reading (inference engine, visualization) SHALL reuse shared image-path resolution helpers.

The implementation MUST preserve each stage’s intended strictness:
- inference SHOULD resolve paths best-effort (environment/config root + JSONL-relative fallback),
- visualization SHOULD resolve paths strictly (returning no-path / skipping when the image cannot be found).

Root image-dir precedence SHALL be explicit and deterministic:
- `ROOT_IMAGE_DIR` (env override) >
- `run.root_image_dir` (config root) >
- `infer.gt_jsonl` parent (`gt_parent`) >
- `none` (no resolved root).

The chosen root decision SHALL be recorded in `resolved_config.json` using:
- `root_image_dir`
- `root_image_dir_source` in `{env, config, gt_parent, none}`.

#### Scenario: ROOT_IMAGE_DIR is respected when resolving relative paths
- **GIVEN** a relative image path `images/0001.jpg`
- **AND** the environment variable `ROOT_IMAGE_DIR` points to a dataset root
- **WHEN** the infer stage resolves the image path for loading
- **THEN** it resolves under `ROOT_IMAGE_DIR` rather than the JSONL directory.

#### Scenario: Visualization skips missing images without crashing
- **GIVEN** a record whose `image` path cannot be found under any resolution rule
- **WHEN** the visualization stage runs
- **THEN** it skips that record/image deterministically without crashing.

#### Scenario: Infer/eval/vis consume one resolved root decision
- **GIVEN** a pipeline run with relative image paths
- **WHEN** root resolution is computed once for the run
- **THEN** infer/eval/vis use the same resolved root decision
- **AND** `resolved_config.json` preserves both root path and root source for reproducibility.

### Requirement: Pipeline evaluation is score-aware and rejects fixed-score toggles
Pipeline evaluation SHALL be score-aware for COCO metrics by default (it MUST NOT provide a fixed-score mode).

The pipeline YAML MUST NOT include `eval.use_pred_score`. If this key is present, the pipeline MUST fail fast with an actionable error instructing the user to remove it.

When the pipeline runs COCO-style detection evaluation, it MUST consume the scored artifact (`gt_vs_pred_scored.jsonl`) rather than the base inference artifact (`gt_vs_pred.jsonl`).

The scored artifact path MUST be resolved from `artifacts.gt_vs_pred_scored_jsonl`.

If `artifacts.gt_vs_pred_scored_jsonl` is not configured (or the file does not exist), the pipeline MUST fail fast with an actionable error instructing the user to run the confidence post-op first.

#### Scenario: Deprecated score toggle is rejected
- **WHEN** a pipeline config includes `eval.use_pred_score`
- **THEN** the pipeline terminates with a clear error explaining that fixed-score evaluation is unsupported and the key must be removed.

### Requirement: Pipeline artifact contract for confidence workflow
For score-aware COCO workflows, the pipeline/post-op artifact contract SHALL include:
- `artifacts.pred_token_trace_jsonl`: canonical token-trace sidecar path (explicit value or deterministic default `<run_dir>/pred_token_trace.jsonl`).
- Trace records keyed by `line_idx`, with 1:1 `generated_token_text` and `token_logprobs` arrays (full generated output, no filtering).

Inference-only or f1ish-only evaluation runs MAY proceed without running confidence post-op. COCO workflows MUST produce/consume these artifacts in order: `gt_vs_pred.jsonl` + `pred_token_trace.jsonl` -> confidence post-op -> `gt_vs_pred_scored.jsonl`.

#### Scenario: Missing scored artifact fails fast
- **GIVEN** a pipeline config that requests COCO detection evaluation
- **WHEN** `gt_vs_pred_scored.jsonl` is not available
- **THEN** the pipeline terminates with a clear error instructing the user to produce `gt_vs_pred_scored.jsonl` via the confidence post-op before evaluating.

#### Scenario: f1ish-only evaluation does not require a scored artifact
- **GIVEN** a pipeline config that requests only f1ish-style (non-COCO) evaluation
- **WHEN** `gt_vs_pred_scored.jsonl` is not available
- **THEN** the pipeline continues evaluation using the base inference artifact (`gt_vs_pred.jsonl`).

