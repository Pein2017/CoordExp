## ADDED Requirements

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
