# stage2-ab-training Delta

## MODIFIED Requirements

### Requirement: Stage-2 AB objective weights are pipeline-only (no flat objective knobs)

The system SHALL determine Stage-2 objective behavior for
`objective.id: teacher_forcing` from the shared teacher-forcing objective config
and Stage-2 adapter provenance, not from the old decoded-box auxiliary module
pipeline.

Normative behavior:

- Stage-2 teacher-forcing MUST emit `TeacherForcingTargetIR` through a Stage-2
  adapter and route shared objective math through the existing semantic runner
  under `src/training/objectives` and the shared helpers under
  `src/training/teacher_forcing`;
- flat objective knobs remain rejected;
- old Stage-2 module names such as `bbox_geo`, `bbox_size_aux`, `coord_reg`, and
  `token_ce` MUST NOT be active core teacher-forcing objective modules;
- old geometry, coordinate-regression, W1, soft-CE, and gate weights MUST fail
  with migration guidance when authored for active teacher-forcing runs;
- Stage-2 rollout, matching, triage, and Channel-A/Channel-B provenance MAY
  remain Stage-2 adapter metadata.

#### Scenario: Old Stage-2 module pipeline is rejected for teacher forcing

- **WHEN** a Stage-2 config uses `objective.id: teacher_forcing`
- **AND** declares `stage2_ab.pipeline.objective[name=coord_reg]`
- **THEN** config validation fails fast
- **AND** the error points to the shared teacher-forcing module taxonomy.

### Requirement: Stage-2 AB pipeline specs are explicit and complete (no implicit defaults)

The system SHALL keep Stage-2 teacher-forcing adapter configuration explicit
without requiring the removed decoded-box objective modules.

Normative behavior:

- Stage-2 adapter configuration MUST declare how Channel-A/Channel-B provenance,
  rollout subsets, FN/recovered-FN metadata, and object ordering map into
  `TeacherForcingTargetIR`;
- any Stage-2 objective pipeline retained for historical configs MUST be marked
  legacy and MUST NOT be used by new `objective.id: teacher_forcing` runs;
- active teacher-forcing configs MUST NOT require `bbox_geo`, `bbox_size_aux`,
  `coord_reg`, `soft_ce`, `w1`, `coord_gate`, or `text_gate` module specs.

#### Scenario: Stage-2 adapter emits shared target IR

- **WHEN** Stage-2 Channel-B builds a teacher-forcing batch
- **THEN** rollout and matching metadata are represented as target IR provenance
- **AND** shared objective modules consume `TeacherForcingTargetIR`.

### Requirement: Stage-2 AB module configs are strict and canonical (no aliases)

The system SHALL reject old Stage-2 module configs and aliases for active
teacher-forcing runs rather than silently translating them into the new
objective.

Normative behavior:

- `stage2_ab.pipeline.objective[name=bbox_geo]`,
  `stage2_ab.pipeline.objective[name=bbox_size_aux]`, and
  `stage2_ab.pipeline.objective[name=coord_reg]` MUST be rejected for active
  teacher-forcing runs;
- old `coord_reg.config` keys including `soft_ce_weight`, `w1_weight`,
  `coord_gate_weight`, `text_gate_weight`, `temperature`, `target_sigma`, and
  `target_truncate` MUST be rejected for active teacher-forcing runs;
- coordinate-neighborhood behavior, if later enabled, MUST be expressed through
  valid-set or valid-neighborhood marginal semantics, not through old soft CE,
  W1, or gate aliases.

#### Scenario: Old coord_reg soft-CE key fails fast

- **WHEN** an active teacher-forcing Stage-2 config declares
  `stage2_ab.pipeline.objective[name=coord_reg].config.soft_ce_weight`
- **THEN** config validation fails fast with migration guidance.

### Requirement: Stage-2 AB objective application is explicit and non-redundant

The system SHALL represent Stage-2 objective application as adapter provenance
and loss tags in `TeacherForcingTargetIR`.

Normative behavior:

- Channel-A and Channel-B routing MUST NOT use old `bbox_geo`,
  `bbox_size_aux`, or `coord_reg` application presets for active
  teacher-forcing runs;
- Stage-specific adapters MAY assign `loss_tags`, atom weights, and provenance
  metadata;
- shared modules MUST NOT rediscover Stage-2 routing semantics from old module
  names.

#### Scenario: Stage-2 provenance replaces old application presets

- **WHEN** a Stage-2 adapter emits a Channel-A atom
- **THEN** Channel-A provenance is carried in the target IR
- **AND** no old bbox/coord application preset is required.

### Requirement: Stage-2 AB remains compatible with ms-swift and Transformers (no upstream patches)

The system SHALL preserve upstream compatibility for Stage-2 teacher forcing
while respecting explicit target-IR atom-position invariants.

Normative behavior:

- the trainer MUST preserve raw sample fields required for Channel-B rollout and
  target-IR construction;
- the trainer MUST strip `teacher_forcing_target_ir` and other runner-owned
  sidecars before every model forward;
- the trainer MUST preserve legitimate upstream attention and flash-attention
  kwargs when stripping runner-owned sidecars;
- the trainer MUST NOT use logits slicing, including `logits_to_keep`, for the
  teacher-forcing objective;
- when `objective.id: teacher_forcing` is active in v1, Stage-2 MUST fail fast
  for packed or padding-free forwards unless an exact atom-position mapping is
  implemented for packed segments and flash-attention metadata.

#### Scenario: Stage-2 packing fails fast until atom mapping exists

- **GIVEN** `objective.id: teacher_forcing`
- **AND** `custom.trainer_variant: stage2_two_channel`
- **AND** `training.packing: true`
- **WHEN** runtime support validation runs before model forward
- **THEN** validation fails fast unless an exact packed atom-position mapping is
  implemented and enabled.

### Requirement: Stage-2 two-channel training supports a config-declared objective and diagnostics pipeline

The system SHALL use the shared teacher-forcing objective as the active
Stage-2 two-channel objective pipeline for new runs.

Normative behavior:

- new Stage-2 teacher-forcing configs MUST use `objective.id: teacher_forcing`;
- active objective modules MUST be the shared teacher-forcing modules:
  `token_type_mass`, `conditional_valid_set_likelihood`,
  `within_valid_coverage`, and optional `continuation_margin`;
- old canonical ordering `token_ce`, `bbox_geo`, `bbox_size_aux`, `coord_reg`
  MUST NOT be used for active teacher-forcing runs;
- diagnostics MAY remain Stage-2-specific when emitted under the new metric
  namespace and clearly labeled as diagnostics.

#### Scenario: Active Stage-2 module order uses shared teacher-forcing modules

- **WHEN** a new Stage-2 teacher-forcing config resolves its objective modules
- **THEN** the active modules are the shared teacher-forcing modules
- **AND** old decoded-box and coord-reg modules are absent.

### Requirement: Stage-2 Two-Channel module names are stable and discoverable

The system SHALL expose the shared teacher-forcing modules through Stage-2
module discovery for active new runs.

Normative behavior:

- active new runs MUST discover teacher-forcing modules from the shared objective
  registry;
- old Stage-2 module names may remain discoverable only in legacy/historical
  contexts or migration error messages;
- documentation MUST distinguish historical Stage-2 module names from active
  teacher-forcing module names.

#### Scenario: Module discovery returns shared modules for new runs

- **WHEN** module discovery runs for `objective.id: teacher_forcing`
- **THEN** it returns the shared teacher-forcing module names.

### Requirement: Stage-2 Two-Channel module configs are strict and typed

The system SHALL reject stale old Stage-2 module configs during strict config
validation for active teacher-forcing runs.

Normative behavior:

- unknown teacher-forcing module config keys MUST fail fast;
- stale old keys under `bbox_geo`, `bbox_size_aux`, and `coord_reg` MUST fail
  fast when active teacher-forcing is selected;
- `objective.modules.within_valid_coverage.coverage_strength` is the only active
  within-valid coverage scalar in the new objective.

#### Scenario: Old W1 key is rejected

- **WHEN** an active teacher-forcing Stage-2 config declares
  `stage2_ab.pipeline.objective[name=coord_reg].config.w1_weight`
- **THEN** config validation fails fast with migration guidance.

### Requirement: Stage-2 Two-Channel adheres to the unified loss registry contract

The system SHALL make Stage-2 two-channel training consume the unified
teacher-forcing registry contract for active new runs.

Normative behavior:

- active Stage-2 objective modules MUST use the shared module names and
  `TeacherForcingTargetIR`;
- Stage-2-specific rollout and triage metadata MUST remain provenance or
  diagnostics, not separate decoded-box objective modules;
- old geometry, coord-regression, soft-CE, W1, and gate modules MUST NOT be
  required by the active core objective.

#### Scenario: Unified registry contract owns active Stage-2 objective modules

- **WHEN** Stage-2 resolves active objective modules for teacher forcing
- **THEN** the unified registry provides the module identities.

## REMOVED Requirements

### Requirement: Stage-2 AB supports text_gate via coord_reg module config

`text_gate` through `coord_reg` is removed from active teacher-forcing Stage-2
objective configs. Token-role stability is represented by `token_type_mass`.

### Requirement: Stage-2 AB objective includes coord soft-CE and W1 terms on supervised bbox slots

Coord soft-CE and W1 terms through `coord_reg` are removed from active
teacher-forcing Stage-2 objective configs.

### Requirement: Canonical Stage-2 base and prod leaves declare CIoU/coord-CE/soft-CE/W1 weights explicitly

Canonical Stage-2 teacher-forcing configs no longer declare CIoU, coord-CE,
soft-CE, or W1 weights as active objective weights.

### Requirement: Stage-2 AB can add matched decoded-box size auxiliaries through `bbox_size_aux`

`bbox_size_aux` is removed from active teacher-forcing Stage-2 objective configs.
