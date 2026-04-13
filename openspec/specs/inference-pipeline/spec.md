## MODIFIED Requirements

### Requirement: Staged pipeline (inference -> eval and/or viz)
The system SHALL provide a unified pipeline runner that can execute stages:
- `infer`: run generation and write the raw pipeline artifact
  (`gt_vs_pred.jsonl`) and a run summary (`summary.json`),
- `eval`: compute evaluation outputs from the raw pipeline artifact and, when
  duplicate-control is enabled, also emit guarded duplicate-control outputs
  under the same resolved run directory,
- `vis`: render qualitative overlays from the raw pipeline artifact and write
  outputs under `vis/`.

Evaluation and visualization MUST remain runnable without invoking the model.

Normative behavior:
- the raw inference artifact remains `gt_vs_pred.jsonl`,
- the eval stage MUST preserve the existing score-aware input contract:
  - non-COCO / raw evaluation paths may consume `gt_vs_pred.jsonl`,
  - score-aware COCO evaluation paths must continue to consume
    `gt_vs_pred_scored.jsonl`,
- when a run resolves `infer.bbox_format=center_log_size`, the pipeline MUST
  narrow to the raw standardized artifact family:
  - raw/non-COCO evaluation may consume `gt_vs_pred.jsonl`
  - raw duplicate-control may emit `gt_vs_pred_guarded.jsonl`
  - score-aware COCO evaluation and scored guarded artifacts MUST fail fast
- when `eval.duplicate_control.enabled=false`, eval behavior remains the normal
  raw-only path,
- when `eval.duplicate_control.enabled=true`, the eval stage MUST:
  - score the normal resolved raw/scored evaluation input exactly as the base
    evaluation contract requires,
  - apply the canonical offline duplicate-control guard to the valid bbox
    prediction view used by that evaluation path,
  - emit a guarded companion artifact for the active input family:
    - `gt_vs_pred_guarded.jsonl` when the eval path uses `gt_vs_pred.jsonl`,
    - `gt_vs_pred_scored_guarded.jsonl` when the eval path uses
      `gt_vs_pred_scored.jsonl`,
  - emit guarded metrics alongside raw metrics,
  - emit a duplicate-control report that explains how many predictions were
    suppressed and why,
- malformed-output handling MUST remain separate from duplicate-control:
  parse / invalid geometry failures continue to be handled by the existing
  inference/eval validation path rather than by the duplicate guard,
- no new CLI flags are introduced; duplicate-control is configured only through
  YAML.

Resolved artifact contract when duplicate-control is enabled:
- `artifacts.gt_vs_pred_jsonl`: raw artifact path
- `artifacts.gt_vs_pred_guarded_jsonl`: guarded raw companion path when the
  active eval path uses `gt_vs_pred.jsonl`
- `artifacts.gt_vs_pred_scored_jsonl`: scored artifact path for score-aware
  COCO evaluation when required by the base contract
- `artifacts.gt_vs_pred_scored_guarded_jsonl`: guarded scored companion path
  when the active eval path uses `gt_vs_pred_scored.jsonl`
- `artifacts.metrics_json`: raw metrics path
- `artifacts.metrics_guarded_json`: guarded metrics path
- `artifacts.duplicate_guard_report_json`: duplicate-control report path

These guarded artifact-path keys are resolved manifest outputs for this change,
not new required authored YAML keys. Unless a future schema explicitly adds
typed overrides, deterministic defaults relative to the run directory SHALL be:
- `gt_vs_pred.jsonl`
- `gt_vs_pred_guarded.jsonl`
- `gt_vs_pred_scored.jsonl`
- `gt_vs_pred_scored_guarded.jsonl`
- `eval/metrics.json`
- `eval/metrics_guarded.json`
- `eval/duplicate_guard_report.json`

#### Scenario: Eval stage emits raw and guarded outputs without loading the model
- **GIVEN** an existing resolved evaluation input artifact
- **AND** `stages.infer=false`
- **AND** `stages.eval=true`
- **AND** `eval.duplicate_control.enabled=true`
- **WHEN** the user runs the pipeline runner
- **THEN** evaluation completes without loading the model
- **AND** the run emits both raw and guarded metrics
- **AND** the run emits the guarded companion artifact for the active eval path
- **AND** the run emits a duplicate-control report.

#### Scenario: Disabled duplicate-control keeps the raw-only eval contract
- **GIVEN** an existing `gt_vs_pred.jsonl`
- **AND** `eval.duplicate_control.enabled=false`
- **WHEN** the eval stage runs
- **THEN** it emits only the normal raw evaluation outputs
- **AND** it does not emit guarded duplicate-control artifacts.

#### Scenario: `center_log_size` rejects scored evaluation paths
- **GIVEN** an inference/eval run with `infer.bbox_format=center_log_size`
- **WHEN** the pipeline is asked for score-aware COCO evaluation or
  `gt_vs_pred_scored*.jsonl` artifacts
- **THEN** it fails fast with guidance that V1 supports canonical raw
  standardized artifacts and raw guarded duplicate-control only.

### Requirement: `resolved_config.json` is the canonical resolved manifest
Inference-pipeline SHALL persist resolved run metadata in `resolved_config.json`
in the run directory.

Normative behavior:
- when duplicate-control is enabled for eval, the resolved manifest MUST record:
  - the guarded artifact path for the active eval input family:
    - `artifacts.gt_vs_pred_guarded_jsonl` and/or
      `artifacts.gt_vs_pred_scored_guarded_jsonl`
  - `artifacts.metrics_guarded_json`
  - `artifacts.duplicate_guard_report_json`
  - `eval.duplicate_control.enabled`
- the resolved manifest MAY record a small duplicate-control config snapshot for
  diagnostics, but only stable top-level manifest keys remain contract-bearing,
- duplicate-control enablement MUST be auditable from the manifest without
  opening other run artifacts.

#### Scenario: Resolved manifest captures duplicate-control artifact paths
- **GIVEN** a pipeline run with `eval.duplicate_control.enabled=true`
- **WHEN** the run initializes its artifact contract
- **THEN** `resolved_config.json` records the guarded artifact paths and
  duplicate-control enablement
- **AND** downstream readers can determine that both raw and guarded eval
  outputs are expected.

### Requirement: Pipeline evaluation is score-aware and rejects fixed-score toggles
Pipeline evaluation SHALL remain score-aware for COCO metrics by default.

Normative duplicate-control behavior for score-aware COCO workflows:
- duplicate-control post-op is applied only after the evaluator has resolved a
  valid scored artifact or score-ready raw artifact according to the normal
  evaluation contract,
- duplicate-control MUST NOT invent or rewrite malformed score provenance,
- when guarded COCO outputs are produced, they MUST preserve stable score fields
  for all retained predictions,
- suppressed predictions are removed from the guarded artifact rather than
  being zeroed or score-demoted in place,
- score-aware guarded COCO workflows MUST use
  `gt_vs_pred_scored_guarded.jsonl` as the guarded prediction artifact.

#### Scenario: Guarded COCO artifact preserves retained prediction scores
- **GIVEN** a score-aware evaluation run with duplicate-control enabled
- **WHEN** the eval stage emits `gt_vs_pred_scored_guarded.jsonl`
- **THEN** retained predictions keep their existing score values
- **AND** suppressed predictions are absent from the guarded artifact
- **AND** the duplicate-control report explains the suppression counts.
