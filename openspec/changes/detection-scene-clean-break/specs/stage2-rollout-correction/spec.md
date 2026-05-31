## ADDED Requirements

### Requirement: Stage-2 rollout correction consumes DetectionScene and owns correction projections

Clean-break Stage-2 rollout correction SHALL consume `DetectionScene` plus
parsed rollout predictions and SHALL own assignment and correction-event
construction as explicit projections.

Normative behavior:

- Stage-2 correction target construction MUST consume `DetectionScene` and
  `RolloutPrediction` values;
- `RolloutPrediction` values MUST be derived through the shared
  inference/runtime decode and parser boundary, including parser policy,
  invalid/drop metadata, prompt/decode/model provenance, and metric-bearing
  status when the rollout participates in training metrics or official eval;
- Stage-2 MUST NOT parse raw backend text, backend-native traces, or diagnostic
  salvage outputs into metric-bearing rollout predictions through a private
  parser path;
- GT/pred matching MUST produce `DetectionAssignment` values;
- residual correction target construction MUST produce `CorrectionEvent` values;
- the final training projection MUST produce `DetectionSupervisionView` values;
- Stage-2 target construction MUST NOT consume Stage-1 rendered text or Stage-1
  token sidecars as semantic authority.
- the DetectionScene transition MUST NOT reintroduce Stage-2 A/B, two-channel,
  scheduler/channel knobs, `rollout_matching.pipeline`, or live rollout tokens
  as positive labels.

#### Scenario: Correction target construction has explicit semantic inputs

- **GIVEN** a `DetectionScene` and parsed `RolloutPrediction` values
- **WHEN** Stage-2 rollout-correction targets are built
- **THEN** matching produces `DetectionAssignment`
- **AND** residual supervision is represented as `CorrectionEvent` values before
  token-level supervision is built.

#### Scenario: RolloutPrediction is not a private Stage-2 parser result

- **GIVEN** online rollout generation returns raw model text and backend trace
  metadata
- **WHEN** Stage-2 prepares rollout-correction supervision
- **THEN** shared inference/runtime parsing produces the strict decoded output
  and provenance needed by `RolloutPrediction`
- **AND** Stage-2 uses `RolloutPrediction` only for assignment and correction
  projections, not as a second parser/provenance system.

### Requirement: Rollout matching is not a target public namespace in the clean-break stack

The clean-break Stage-2 architecture SHALL consolidate rollout, decode, backend,
and eval policy under the Stage-2 rollout-correction concept instead of the
historical public `rollout_matching.*` namespace.

Normative behavior:

- new canonical Stage-2 configs SHOULD live under `configs/stage2/rollout_correction/`;
- new public Stage-2 authored policy SHOULD be owned under
  `stage2_rollout_correction` or an approved successor namespace;
- public `rollout_matching.*` MUST NOT be the target namespace for new canonical
  Stage-2 runtime, backend, decode, or eval policy;
- remaining `rollout_matching` mentions MUST be classified as archive history,
  migration handle, rejection/absence test, or private implementation detail
  scheduled for rename/delete.

#### Scenario: New Stage-2 config does not require rollout_matching namespace

- **GIVEN** a new canonical Stage-2 rollout-correction config after the
  clean-break migration
- **WHEN** authored rollout prompt, backend, decode, and eval policy is loaded
- **THEN** the policy is resolved from the Stage-2 rollout-correction surface
- **AND** no public `rollout_matching.*` key is required.
