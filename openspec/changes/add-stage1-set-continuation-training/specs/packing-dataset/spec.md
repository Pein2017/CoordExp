## MODIFIED Requirements

### Requirement: Stage-1 dataset-level packing is static-only
For ordinary Stage-1 dataset-level packing (non-rollout, one-sequence SFT
trainer variants), the system SHALL enforce the existing static-only policy.

The `stage1_set_continuation` trainer variant is a v1 exception to Stage-1
dataset-level packing support.

Additional normative behavior:
- when `custom.trainer_variant: stage1_set_continuation` and
  `training.packing=true`, setup MUST fail fast before static pack-plan dataset
  construction,
- the failure MUST apply whether `training.packing_mode` is omitted, `static`,
  or `dynamic`,
- the failure MUST apply before train or eval packing can collapse multiple raw
  records into one sample,
- `training.eval_packing` does not make set-continuation packing eligible in
  v1.

#### Scenario: Set-continuation rejects training packing before pack-plan build
- **GIVEN** `custom.trainer_variant: stage1_set_continuation`
- **AND** `training.packing=true`
- **WHEN** packing policy is validated
- **THEN** setup fails with actionable guidance that v1 set-continuation
  repeated-forward MP requires unpacked per-sample batches.

#### Scenario: Ordinary Stage-1 static packing remains eligible
- **GIVEN** ordinary Stage-1 SFT without `stage1_set_continuation`
- **AND** `training.packing=true`
- **AND** `training.packing_mode=static`
- **WHEN** packing policy is validated
- **THEN** existing static packing behavior remains unchanged.
