## ADDED Requirements

### Requirement: Score provenance and metric-bearing parser policy are single-owned

Inference and evaluation pipelines SHALL use one authoritative implementation
for score provenance, score policy fingerprinting, raw/scored comparability,
parser policy, and metric-bearing status.

Normative behavior:

- pipelines MAY decide when to materialize constant-score, confidence-score, or
  train-time eval artifacts, but MUST NOT each define their own sidecar schema
  or score-policy fingerprint semantics;
- raw artifacts MUST remain unscored and scored artifacts MUST carry score
  provenance through the shared writer/validator;
- metric-bearing artifacts MUST record or reference parser policy and
  `metric_bearing` status;
- diagnostic salvage parsing MUST NOT improve official metric inputs;
- train-time Stage-2 eval artifacts and offline inference artifacts MUST use
  compatible provenance semantics even when their output directories differ.
- official/comparable evaluation MUST reject diagnostic or salvage parser
  outputs and MUST fail with a missing/divergent provenance reason before
  metrics are trusted;
- raw F1-ish or other provenance-free eval paths MAY exist only as
  `inspection` / `non_comparable` outputs and MUST report that status.

#### Scenario: Comparable scored artifact loads through one provenance path

- **GIVEN** a scored artifact produced by offline inference or Stage-2
  train-time eval materialization
- **WHEN** official eval loads it as a comparable score-bearing artifact
- **THEN** the same provenance validator checks raw-artifact identity,
  score-policy fingerprint, parser policy, and metric-bearing status
- **AND** missing or divergent provenance fails before metrics are trusted.

#### Scenario: Provenance-free raw eval is inspection-only

- **GIVEN** an eval path consumes a raw artifact without comparable provenance
- **WHEN** it computes an F1-ish or diagnostic summary
- **THEN** the output declares `inspection` or `non_comparable` status
- **AND** it is not reported as official/comparable evaluation.

### Requirement: Prediction order is preserved independently from prompt object ordering

Canonical inference output SHALL preserve parsed prediction order in emitted
`pred` arrays and SHALL NOT sort predictions merely because prompt or GT object
ordering used a sorted policy.

Normative behavior:

- prompt/input object ordering MAY affect prompt construction or teacher-forced
  target order;
- parsed prediction order MUST remain the order emitted by the model after
  strict parsing and standardization;
- any optional sorted diagnostic view MUST be separate from canonical
  metric-bearing `gt_vs_pred*.jsonl` outputs.

#### Scenario: Sorted prompt policy does not sort emitted predictions

- **GIVEN** a run uses a sorted prompt or object-ordering policy
- **AND** the model emits multiple valid predictions in non-sorted order
- **WHEN** canonical `gt_vs_pred.jsonl` rows are written
- **THEN** each row's `pred` array preserves parsed emission order.
