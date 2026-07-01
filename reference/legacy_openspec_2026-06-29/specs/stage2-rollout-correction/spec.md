## Purpose
Define the active Stage-2 rollout-correction training contract: a rollout prefix
followed by GT/residual correction, with no public or runtime distinction between
old A/B training branches.

## Requirements
### Requirement: Stage-2 rollout-correction is the only active Stage-2 trainer surface
Training configs MUST select Stage-2 through:

- `custom.trainer_variant: stage2_rollout_correction`
- top-level `stage2_rollout_correction`
- `stage2_rollout_correction.pipeline`

The system MUST fail fast for removed trainer variants and config namespaces:

- `custom.trainer_variant: stage2_two_channel`
- `custom.trainer_variant: stage2_ab_training`
- top-level `stage2_ab`

`rollout_matching.*` remains a private migration/runtime handle for rollout runtime/backend/decode/eval settings until the schema-owned replacement seam is complete.
`rollout_matching.pipeline` remains removed and MUST NOT own Stage-2 objectives.
Train-time rollout prompt variants MUST be authored with
`rollout_matching.prompt_variant`; eval-step rollout prompt variants MUST be
authored separately with `rollout_matching.eval_prompt_variant` when a run needs
to pin a prompt surface for same-surface comparison.

#### Scenario: Removed trainer variant fails fast
- **GIVEN** a config with `custom.trainer_variant: stage2_two_channel`
- **WHEN** config loading resolves the runtime plan
- **THEN** loading fails before trainer construction
- **AND** the error points to `stage2_rollout_correction`.

#### Scenario: Removed top-level namespace fails fast
- **GIVEN** a config with top-level `stage2_ab`
- **WHEN** config loading parses the training config
- **THEN** loading fails before trainer construction
- **AND** the error points to `stage2_rollout_correction`.

#### Scenario: Prompt variants stay in rollout-matching namespace
- **GIVEN** a Stage-2 rollout-correction config with
  `rollout_matching.prompt_variant: coco_80`
- **AND** `rollout_matching.eval_prompt_variant: coco_80`
- **WHEN** config loading parses the rollout-matching namespace
- **THEN** both prompt variant keys validate against the shared prompt registry
- **AND** no `infer.prompt_variant` key is required for training rollouts.

### Requirement: Unified Stage-2 objective is residual-set correction only
`stage2_rollout_correction.pipeline.objective[]` MUST contain exactly one
enabled objective module:

- `name: residual_set_correction`
- `application.preset: rollout_self_prefix`

The pipeline MUST reject:

- `token_ce`
- `hard_sft`
- `stage2_trie_ce`
- geometry auxiliaries
- any `channels` field
- non-empty legacy diagnostics

#### Scenario: Residual-set correction loads
- **GIVEN** `stage2_rollout_correction.pipeline.objective[]` contains one enabled `residual_set_correction`
- **AND** its `application.preset` is `rollout_self_prefix`
- **WHEN** config loading parses the training config
- **THEN** the config loads
- **AND** the manifest family is `stage2_rollout_correction`.

#### Scenario: Clean-prefix objective fails fast
- **GIVEN** a config with `stage2_rollout_correction.pipeline.objective[].name: token_ce`
- **WHEN** config loading parses the training config
- **THEN** loading fails
- **AND** the error states that `token_ce` was removed from unified Stage-2.

### Requirement: Stage-2 has no scheduler or per-channel namespace
Unified Stage-2 MUST NOT expose scheduling, branch, or per-channel authoring
knobs. The config MUST reject:

- `stage2_rollout_correction.schedule`
- `stage2_rollout_correction.b_ratio`
- `stage2_rollout_correction.channel_b`
- `stage2_rollout_correction.pipeline.objective[].channels`
- `_stage2_ab_channel`

#### Scenario: Scheduler key fails fast
- **GIVEN** a config with `stage2_rollout_correction.schedule`
- **WHEN** config loading parses the training config
- **THEN** loading fails
- **AND** the error states that scheduling was removed.

### Requirement: Metrics and manifests use rollout-correction names
Active Stage-2 training MUST emit current metrics under:

- `stage2_rollout_correction/...`

Active manifests MUST use:

- `extra.variant: stage2_rollout_correction`
- manifest family `stage2_rollout_correction`

Active training MUST NOT emit new metrics under removed branch namespaces such as
`stage2/channel_a`, `stage2/channel_b`, `stage2_ab/b_ratio_realized`, or
`stage2_ab/channel_b/...`.

#### Scenario: Active Stage-2 emits rollout-correction identity
- **GIVEN** a Stage-2 rollout-correction training run
- **WHEN** metrics and manifests are emitted
- **THEN** metric keys use the `stage2_rollout_correction/...` namespace
- **AND** manifests identify the `stage2_rollout_correction` variant and family.
