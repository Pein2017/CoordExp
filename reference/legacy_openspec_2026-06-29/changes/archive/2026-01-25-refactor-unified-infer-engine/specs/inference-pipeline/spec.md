## MODIFIED Requirements

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
