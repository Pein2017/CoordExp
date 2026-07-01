## Purpose
Define the staged inference pipeline contract for generation, evaluation,
visualization, resolved manifests, and score-aware evaluation behavior.
## Requirements
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
- when a run resolves `infer.bbox_format=cxcy_logw_logh`, the pipeline MUST:
  - keep the canonical standardized artifact family authoritative,
  - reject confidence post-op,
  - allow raw/non-COCO evaluation to consume `gt_vs_pred.jsonl`,
  - allow score-aware COCO/LVIS evaluation to consume
    `gt_vs_pred_scored.jsonl` after it is materialized from the canonical raw
    artifact via a deterministic constant-score compatibility policy,
  - allow duplicate-control to emit `gt_vs_pred_guarded.jsonl` or
    `gt_vs_pred_scored_guarded.jsonl` according to the active eval input family,
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

#### Scenario: `cxcy_logw_logh` materializes a constant-score scored artifact for official eval
- **GIVEN** an inference/eval run with `infer.bbox_format=cxcy_logw_logh`
- **WHEN** the pipeline is asked for score-aware COCO/LVIS evaluation
- **THEN** it materializes `gt_vs_pred_scored.jsonl` from canonical standardized
  predictions using deterministic constant-score provenance
- **AND** it keeps confidence post-op disabled for that run.

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

### Requirement: Pipeline metric inputs are strict and metric-bearing only

Inference-pipeline official evaluation inputs SHALL consume only strict,
metric-bearing prediction artifacts produced or validated by the shared
inference runtime.

Normative behavior:

- `gt_vs_pred.jsonl` remains the canonical raw strict standardized artifact;
- `gt_vs_pred_scored.jsonl` remains the canonical score-bearing companion for
  score-aware COCO/LVIS evaluation;
- diagnostic salvage predictions MUST NOT be inserted into raw, scored, or
  guarded official eval inputs;
- parser policy and `metric_bearing` status MUST be auditable from artifacts or
  run summaries;
- official eval inputs MUST resolve prompt, decode, model, parser, and score
  provenance from run metadata or a colocated sidecar before they are treated as
  comparable;
- strict parser error rows and counters MAY appear in canonical artifacts, but
  repaired predictions from diagnostic salvage may not.

#### Scenario: Diagnostic salvage cannot improve official metrics

- **GIVEN** a malformed model output that strict parsing rejects
- **AND** diagnostic salvage can recover a plausible object
- **WHEN** the eval stage materializes official metric inputs
- **THEN** the recovered object is absent from `gt_vs_pred.jsonl`,
  `gt_vs_pred_scored.jsonl`, and guarded companions
- **AND** any salvage output is stored only in non-metric diagnostics.

### Requirement: Raw and scored artifacts keep separate score policy provenance

Inference-pipeline SHALL preserve raw/scored artifact separation and SHALL make
score provenance explicit.

Normative behavior:

- `gt_vs_pred.jsonl` MUST remain raw/unscored and record or resolve
  `score_policy: none`;
- `gt_vs_pred_scored.jsonl` MUST carry score fields and
  `score_policy_fingerprint`;
- `gt_vs_pred_guarded.jsonl` remains raw/guarded;
- `gt_vs_pred_scored_guarded.jsonl` remains scored/guarded and MUST preserve
  score provenance for retained predictions;
- re-scoring a raw artifact MUST write a new scored artifact and MUST NOT
  mutate the raw artifact.

#### Scenario: Re-scoring creates a new scored artifact

- **GIVEN** an existing raw `gt_vs_pred.jsonl`
- **WHEN** a score policy is applied for score-aware eval
- **THEN** the pipeline writes `gt_vs_pred_scored.jsonl` with
  `score_policy_fingerprint`
- **AND** the original raw artifact remains unmodified and unscored.

### Requirement: Pipeline comparisons require prompt/decode/model provenance

Inference-pipeline comparison, report, and official evaluation paths SHALL
reject comparable artifacts whose required prompt, decode, or model identity
fingerprints are missing.

Normative behavior:

- historical artifacts without fingerprints MAY be opened for inspection as
  `comparable: false`;
- official eval, strict comparison, and report-generation paths MUST fail fast
  for missing required fingerprints unless an explicit migration tool has
  reconstructed and stamped them;
- the required missing-provenance diagnostic code is `missing_provenance`;
- a migration/stamping tool MUST either reconstruct exact fingerprints from old
  metadata or mark/report the artifact as `comparable: false` with a reason;
- diagnostics MUST distinguish missing provenance from model parse failures.

#### Scenario: Historical artifact is readable but not comparable

- **GIVEN** an old `gt_vs_pred.jsonl` without prompt/decode/model fingerprints
- **WHEN** a user opens it through an inspection tool
- **THEN** it may load as historical, non-comparable data
- **WHEN** the same artifact is used for strict comparison or official eval
- **THEN** loading fails with a `missing_provenance` diagnostic.

### Requirement: Provenance carriers preserve JSONL schema compatibility

Inference-pipeline SHALL resolve provenance from run-level metadata or
colocated sidecars without requiring incompatible per-line JSONL schema changes.

Normative behavior:

- `resolved_config.json` and `summary.json` are the required run-level
  provenance carriers for new runs;
- copied standalone JSONL artifacts require a colocated provenance sidecar to
  remain comparable;
- missing or moved provenance sidecars MUST fail comparable paths with
  `missing_provenance`;
- `gt_vs_pred.jsonl` and `gt_vs_pred_scored.jsonl` line schemas MUST remain
  compatible with the stable inference-engine output schema.

#### Scenario: Moved JSONL without provenance is inspection-only

- **GIVEN** a copied `gt_vs_pred_scored.jsonl` without its run metadata or
  sidecar
- **WHEN** an inspection tool loads it
- **THEN** it may load as `comparable: false`
- **WHEN** official eval or comparison loads it
- **THEN** loading fails with `missing_provenance`.
