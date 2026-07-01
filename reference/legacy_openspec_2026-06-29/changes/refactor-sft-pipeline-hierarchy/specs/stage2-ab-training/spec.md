## MODIFIED Requirements

### Requirement: Active Stage-2 is rollout correction only

Current Stage-2 configs SHALL use `pipeline.id: stage2_rollout_correction`,
top-level `stage2_rollout_correction`, and exactly one enabled
`residual_set_correction` objective with
`application.preset: rollout_self_prefix`.

`custom.trainer_variant: stage2_rollout_correction` is historical selector
vocabulary after this migration and MUST NOT be accepted as the active selector
outside explicit rejection/migration tests.
Migrated active Stage-2 artifacts MUST record selector identity through
`pipeline.id`, not through a compatibility `trainer_variant` field.

#### Scenario: Active Stage-2 config uses rollout correction

- **GIVEN** a current Stage-2 training config
- **WHEN** config loading resolves the training pipeline
- **THEN** it uses `pipeline.id: stage2_rollout_correction`
- **AND** it contains exactly one enabled `residual_set_correction` objective.
