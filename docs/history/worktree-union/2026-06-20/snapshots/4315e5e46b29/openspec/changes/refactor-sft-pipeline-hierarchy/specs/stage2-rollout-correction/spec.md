## MODIFIED Requirements

### Requirement: Stage-2 rollout-correction is the only active Stage-2 trainer surface

Training configs MUST select active Stage-2 rollout correction through:

- top-level `pipeline.id: stage2_rollout_correction`;
- top-level `stage2_rollout_correction`;
- `stage2_rollout_correction.pipeline`.

The system MUST fail fast for removed or legacy trainer selectors and config
namespaces:

- `custom.trainer_variant: stage2_rollout_correction`
- `custom.trainer_variant: stage2_two_channel`
- `custom.trainer_variant: stage2_ab_training`
- top-level `stage2_ab`
- `stage2_rollout_correction.schedule`
- `stage2_rollout_correction.b_ratio`
- `stage2_rollout_correction.channel_b`
- `stage2_rollout_correction.correction.pseudo_positive`
- legacy assignment modes such as `legacy_hungarian_mask_iou`

`rollout_matching.*` remains a runtime/backend/decode/eval settings namespace
until a separate schema-owned replacement seam is specified.
`rollout_matching.pipeline` remains removed and MUST NOT own Stage-2 objectives.
Train-time rollout prompt variants MUST remain authored with
`rollout_matching.prompt_variant`; eval-step rollout prompt variants MUST remain
authored with `rollout_matching.eval_prompt_variant` when a run needs to pin a
prompt surface for same-surface comparison.

Stage-2 selector provenance MUST be recorded with `pipeline.id:
stage2_rollout_correction`. Migrated active Stage-2 manifests and policy
provenance MUST NOT emit `trainer_variant` as a compatibility selector field.

#### Scenario: Pipeline id selects Stage-2 rollout correction

- **GIVEN** a config with `pipeline.id: stage2_rollout_correction`
- **AND** top-level `stage2_rollout_correction.pipeline` is provided
- **WHEN** config loading resolves the runtime plan
- **THEN** Stage-2 rollout correction is selected
- **AND** `stage2_rollout_correction.pipeline.objective[]` remains the internal
  correction objective namespace.

#### Scenario: Former custom trainer variant fails fast

- **GIVEN** a config with `custom.trainer_variant:
  stage2_rollout_correction`
- **WHEN** config loading resolves the runtime plan
- **THEN** loading fails before trainer construction
- **AND** the error points to `pipeline.id: stage2_rollout_correction`.

#### Scenario: Stage-2 provenance records pipeline id without trainer variant

- **GIVEN** a migrated Stage-2 rollout-correction run with
  `pipeline.id: stage2_rollout_correction`
- **WHEN** run manifests and policy provenance are materialized
- **THEN** selector identity is recorded as `pipeline.id:
  stage2_rollout_correction`
- **AND** no compatibility `trainer_variant` selector field is emitted.

#### Scenario: Removed trainer variant fails fast

- **GIVEN** a config with `custom.trainer_variant: stage2_two_channel`
- **WHEN** config loading resolves the runtime plan
- **THEN** loading fails before trainer construction
- **AND** the error points to `pipeline.id: stage2_rollout_correction`.

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

#### Scenario: Stage-2 requires rollout runtime settings

- **GIVEN** a config with `pipeline.id: stage2_rollout_correction`
- **AND** top-level `stage2_rollout_correction.pipeline` is provided
- **AND** top-level `rollout_matching` is absent
- **WHEN** config loading resolves the runtime plan
- **THEN** loading fails before trainer construction
- **AND** the error names the missing `rollout_matching` runtime namespace.

### Requirement: Unified Stage-2 objective is residual-set correction only

`stage2_rollout_correction.pipeline.objective[]` MUST contain exactly one
enabled objective module:

- `name: residual_set_correction`
- `application.preset: rollout_self_prefix`

The top-level `pipeline.id` selector MUST NOT move Stage-2 objective modules
into Stage-1 `objective.id` authoring.

The pipeline MUST reject:

- `token_ce`
- `hard_sft`
- `stage2_trie_ce`
- geometry auxiliaries
- any `channels` field
- non-empty legacy diagnostics

#### Scenario: Residual-set correction loads under top-level pipeline id

- **GIVEN** `pipeline.id: stage2_rollout_correction`
- **AND** `stage2_rollout_correction.pipeline.objective[]` contains one enabled
  `residual_set_correction`
- **AND** its `application.preset` is `rollout_self_prefix`
- **WHEN** config loading parses the training config
- **THEN** the config loads
- **AND** the manifest family is `stage2_rollout_correction`.

#### Scenario: Clean-prefix objective still fails fast

- **GIVEN** a config with
  `stage2_rollout_correction.pipeline.objective[].name: token_ce`
- **WHEN** config loading parses the training config
- **THEN** loading fails
- **AND** the error states that `token_ce` was removed from unified Stage-2.
