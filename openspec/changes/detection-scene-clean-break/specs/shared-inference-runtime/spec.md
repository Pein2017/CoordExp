## ADDED Requirements

### Requirement: DetectionScene integrates with shared inference ownership without duplicating it

The DetectionScene clean-break architecture SHALL integrate with shared
inference runtime ownership rather than creating a second prompt, decode,
parser, trace, or provenance system.

Normative behavior:

- prompt rendering, visual input preparation, backend decode, backend lifecycle,
  trace conversion, parser policy, checkpoint identity, and prompt/decode/model
  provenance MUST remain shared inference/runtime responsibilities;
- callers MUST NOT construct independent semantic prompt text, independently
  normalize images, or independently convert backend traces when a shared
  runtime policy exists;
- strict metric-bearing parsing MUST remain separate from diagnostic salvage;
- diagnostic salvage records MUST NOT feed official eval artifacts, confidence
  post-op, COCO/LVIS/mAP, comparable reports, or Stage-2 metric-bearing rollout
  eval;
- scene-derived metric-bearing views MUST carry or resolve prompt, decode,
  model, parser, score-policy, and metric-bearing status provenance.

#### Scenario: DetectionScene does not become a second parser/provenance path

- **GIVEN** model output from offline inference or Stage-2 rollout generation
- **WHEN** it is parsed into detection predictions
- **THEN** parser policy, salvage eligibility, and prompt/decode/model
  provenance are resolved by the shared inference/runtime boundary
- **AND** any `DetectionScene`-derived or `DecodedDetectionResult` view used for
  official metrics is strict and metric-bearing.

### Requirement: Clean-break Stage-2 may retire rollout_matching as a public online namespace

This OpenSpec change SHALL act as the future compatibility-sensitive contract
that permits the clean-break architecture to retire `rollout_matching.*` as a
public online rollout/decode/eval namespace after the archive checkpoint.

Normative behavior:

- offline inference may continue to author `infer.*`;
- new canonical Stage-2 rollout/decode/backend/eval policy SHOULD be authored
  under the Stage-2 rollout-correction surface or an approved successor
  namespace;
- public `rollout_matching.*` MAY remain during migration only as a classified
  temporary migration handle;
- after migration, remaining `rollout_matching` mentions in active surfaces MUST
  be classified as archive history, rejection/absence test, private
  implementation detail, or deletion candidate;
- retiring `rollout_matching.*` MUST NOT reopen `rollout_matching.pipeline` or
  retired rollout-matching trainer semantics.

#### Scenario: New online rollout policy no longer depends on rollout_matching authoring

- **GIVEN** a canonical Stage-2 rollout-correction config after migration
- **WHEN** rollout prompt, backend, decode, parser, and eval policy are resolved
- **THEN** they are resolved from the Stage-2 rollout-correction policy surface
- **AND** `rollout_matching.*` is not required as a public authored namespace.
