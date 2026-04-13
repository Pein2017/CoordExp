## MODIFIED Requirements

### Requirement: Ingestion and validation
For the unified pipeline workflow, the evaluator SHALL preserve the existing
active evaluation input contract:
- raw/non-COCO evaluation paths may consume `gt_vs_pred.jsonl`,
- score-aware COCO evaluation paths must continue to consume
  `gt_vs_pred_scored.jsonl`.

Normative duplicate-control behavior:
- duplicate-control is an offline post-op that runs only on valid bbox
  predictions after the normal ingestion/validation path,
- malformed or invalid geometry records remain owned by the existing evaluator
  validation contract and MUST NOT be reclassified as duplicate-control actions,
- when evaluation consumes artifacts produced under
  `infer.bbox_format=center_log_size`, the evaluator MUST narrow to the raw
  standardized artifact family:
  - `gt_vs_pred.jsonl`
  - optional `gt_vs_pred_guarded.jsonl`
  - score-aware COCO/scored artifact families are unsupported and MUST fail fast
- when duplicate-control is enabled, the evaluator MUST preserve the raw
  validated prediction view and build a guarded prediction view separately,
- the guarded companion artifact MUST match the active evaluation input family:
  - `gt_vs_pred_guarded.jsonl` for raw/non-COCO evaluation paths,
  - `gt_vs_pred_scored_guarded.jsonl` for score-aware COCO evaluation paths,
- polygon and malformed-geometry handling remain unchanged by this change
  unless those objects are already valid inputs to the duplicate-control policy.

#### Scenario: Duplicate-control does not swallow malformed geometry failures
- **GIVEN** an evaluation run with duplicate-control enabled
- **AND** a prediction object is malformed or has invalid geometry
- **WHEN** the evaluator ingests the artifact
- **THEN** the object is handled by the existing validation/diagnostic path
- **AND** it is not counted as a duplicate-control suppression event.

### Requirement: Metrics and diagnostics
In addition to existing evaluation metrics, the evaluator SHALL support a raw
and guarded reporting contract when duplicate-control is enabled.

Normative behavior:
- the evaluator MUST always score the raw validated prediction view and persist
  the normal raw outputs,
- when duplicate-control is enabled, the evaluator MUST also:
  - apply the canonical offline duplicate-control policy to the raw validated
    bbox predictions of the active evaluation input family,
  - score the guarded prediction view separately,
  - emit `metrics_guarded.json`,
  - emit `duplicate_guard_report.json`,
  - and, where the evaluator normally emits derived summary artifacts, emit
    guarded variants using a deterministic `_guarded` suffix,
- the duplicate-control report MUST include at minimum:
  - total predictions inspected by the guard,
  - total predictions suppressed,
  - total guarded records affected,
  - and a stable breakdown of suppression reasons,
- raw metrics remain the primary research/debug headline,
- guarded metrics are the explicit post-op safety/deploy report,
- both raw and guarded outputs are required when duplicate-control is enabled.

Recommended guarded artifact suffix contract:
- `metrics_guarded.json`
- `per_class_guarded.csv` when `per_class.csv` is emitted
- `per_image_guarded.json` when `per_image.json` is emitted
- `matches_guarded.jsonl` when `matches.jsonl` is emitted
- `matches@{iou_thr}_guarded.jsonl` when `matches@{iou_thr}.jsonl` is emitted
- `gt_vs_pred_guarded.jsonl` for raw/non-COCO evaluation paths
- `gt_vs_pred_scored_guarded.jsonl` for score-aware COCO evaluation paths

#### Scenario: Duplicate-control produces raw and guarded metric families
- **GIVEN** an evaluation run with duplicate-control enabled
- **WHEN** evaluation completes
- **THEN** the output directory contains both `metrics.json` and
  `metrics_guarded.json`
- **AND** the duplicate-control report records the suppression counts and
  reasons
- **AND** raw metrics remain available for direct comparison to guarded metrics.

#### Scenario: Disabled duplicate-control emits only the normal raw outputs
- **GIVEN** an evaluation run with duplicate-control disabled
- **WHEN** evaluation completes
- **THEN** the evaluator emits only the normal raw output family
- **AND** no guarded metrics or duplicate-control report are written.

### Requirement: CLI, configuration, and outputs
The evaluator SHALL continue to support YAML-driven reproducible evaluation and
CLI overrides without adding new CLI flags for duplicate-control.

Normative behavior:
- duplicate-control enablement is configured through YAML only,
- the standalone evaluator config accepted by
  `scripts/evaluate_detection.py` permits the minimal typed surface:
  - `duplicate_control.enabled`
- when evaluation is launched through the unified infer pipeline, the pipeline
  may expose a parallel wrapper surface under:
  - `eval.duplicate_control.enabled`
- duplicate-control configuration MUST fail fast on unknown keys,
- guarded artifact-path keys are resolved outputs for this change rather than
  new authored config keys,
- when duplicate-control is enabled, the evaluator MUST log the resolved
  duplicate-control enablement, active eval input family, and guarded artifact
  paths as part of the resolved configuration.

#### Scenario: Duplicate-control is enabled through YAML without new CLI flags
- **GIVEN** a standalone evaluator config with `duplicate_control.enabled=true`
- **WHEN** the evaluator runs through the normal CLI entrypoint
- **THEN** duplicate-control is enabled
- **AND** the evaluator requires no new CLI flags to produce raw and guarded
  outputs.

#### Scenario: Standalone evaluator writes guarded companions under out_dir
- **GIVEN** a direct evaluator run with `--pred_jsonl` and `--out_dir`
- **AND** `duplicate_control.enabled=true`
- **WHEN** evaluation completes
- **THEN** guarded companion artifacts are written under `out_dir`
- **AND** their names follow the same deterministic `_guarded` suffix contract
  as the pipeline-driven path.

### Requirement: Evaluation artifact and metric schema parity is preserved during refactor
Detection-evaluator SHALL preserve the existing raw artifact family while adding
deterministic guarded companions when duplicate-control is enabled.

Normative behavior:
- existing raw artifact names and metric names remain valid and unchanged,
- guarded artifact names use a deterministic `_guarded` suffix or the explicit
  guarded artifact names defined by the pipeline contract,
- guarded outputs MUST be schema-compatible with their raw counterparts so
  downstream comparison does not require bespoke parsing logic,
- duplicate-control MUST reuse the shared duplicate-control relation from the
  repo-owned runtime helper rather than maintaining a separate evaluator-only
  duplicate detector,
- score-aware COCO guarded outputs MUST preserve score provenance and therefore
  operate on `gt_vs_pred_scored.jsonl` -> `gt_vs_pred_scored_guarded.jsonl`
  rather than silently falling back to the unscored raw artifact,
- because offline evaluation has no explorer views, the evaluator MUST use a
  documented offline conservative subset of the shared policy.

#### Scenario: Guarded artifacts remain schema-compatible with raw outputs
- **GIVEN** an evaluation run with duplicate-control enabled
- **WHEN** both raw and guarded per-image artifacts are produced
- **THEN** the guarded artifacts use the same schema as the raw artifacts
- **AND** downstream tools can compare them without a second parsing contract.
