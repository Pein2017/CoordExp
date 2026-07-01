## Status
Retired. The active Stage-2 contract is
[`stage2-rollout-correction`](../stage2-rollout-correction/spec.md).

## Purpose
This file is a compatibility pointer for the removed Stage-2 AB/two-channel
contract. It is not a source of active requirements.

## Requirements
### Requirement: Removed AB/two-channel public contract fails fast
Active training configs MUST NOT accept the retired Stage-2 AB/two-channel
public names.

Removed authoring surfaces include:

- `custom.trainer_variant: stage2_two_channel`
- `custom.trainer_variant: stage2_ab_training`
- top-level `stage2_ab`
- `stage2_ab.schedule.b_ratio`
- `stage2_ab.pipeline.objective[*].channels`
- Channel-A/Channel-B public config handles and metrics

#### Scenario: Old trainer variant is rejected
- **GIVEN** a config sets `custom.trainer_variant: stage2_two_channel`
- **WHEN** config loading runs
- **THEN** loading fails fast
- **AND** the error directs the author to `stage2_rollout_correction`.

#### Scenario: Old namespace is rejected
- **GIVEN** a config contains top-level `stage2_ab`
- **WHEN** config loading runs
- **THEN** loading fails fast
- **AND** the error directs the author to top-level `stage2_rollout_correction`.

### Requirement: Active Stage-2 is rollout correction only
Current configs MUST use `custom.trainer_variant: stage2_rollout_correction`,
top-level `stage2_rollout_correction`, and exactly one enabled
`residual_set_correction` objective with
`application.preset: rollout_self_prefix`.

#### Scenario: Active Stage-2 config uses rollout correction
- **GIVEN** a current Stage-2 training config
- **WHEN** config loading resolves the trainer surface
- **THEN** it uses `stage2_rollout_correction`
- **AND** it contains exactly one enabled `residual_set_correction` objective.
