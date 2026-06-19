## ADDED Requirements

### Requirement: Public Stage-1 CE identity is standard_ce

The public Standard SFT cross-entropy objective identity SHALL be
`standard_ce`.

Normative behavior:

- `standard_ce` denotes ordinary assistant-label cross entropy for Standard
  SFT;
- internal implementation modules MAY be named `token_ce`;
- metrics MAY continue to report token-level CE components when useful;
- user-authored active Standard SFT configs MUST use `objective.id:
  standard_ce`;
- public docs MUST NOT require users to infer Standard SFT from the internal
  `token_ce` name.

#### Scenario: Standard CE maps to token-level CE implementation

- **GIVEN** a config with `objective.id: standard_ce`
- **WHEN** runtime objective modules are built
- **THEN** the ordinary token-level CE implementation is used
- **AND** resolved config records the public objective id as `standard_ce`.

#### Scenario: Public token_ce is not required for Standard SFT

- **GIVEN** an active config with `pipeline.id: stage1_standard_sft`
- **AND** `objective.id: standard_ce`
- **WHEN** the training runtime is resolved
- **THEN** no public `objective.id: token_ce` authoring is required.

### Requirement: Public research teacher-forcing identity is research_teacher_forcing

The public fine-grained research teacher-forcing objective identity SHALL be
`research_teacher_forcing`.

Normative behavior:

- `research_teacher_forcing` owns role/value/span tracing, valid sets, branch
  state, force/weight policies, and exact label/logit position metadata;
- `teacher_forcing` is not the desired long-term public id;
- active migrated configs MUST reject `teacher_forcing` rather than accepting it
  as an alias;
- internal weighted terms under `research_teacher_forcing` MAY reuse shared
  token-role vocabulary:
  - `struct`
  - `desc`
  - `coord`
  - `eos`;
- Stage-2 residual-set correction remains a separate rollout/self-trajectory
  objective namespace.

#### Scenario: Research teacher forcing resolves shared roles

- **GIVEN** `objective.id: research_teacher_forcing`
- **WHEN** target atoms are built
- **THEN** supervised atoms use shared token roles `struct`, `desc`, `coord`,
  and `eos`
- **AND** role/value/span metadata is available to the objective terms.

#### Scenario: Legacy teacher_forcing id fails fast

- **GIVEN** an active migrated config with `objective.id: teacher_forcing`
- **WHEN** config loading resolves objective identity
- **THEN** loading fails before objective construction
- **AND** the error directs the author to `objective.id:
  research_teacher_forcing`.

### Requirement: Stage-2 residual-set correction remains separate from Stage-1 objectives

Stage-2 rollout correction SHALL keep `residual_set_correction` under
`stage2_rollout_correction.pipeline.objective[]`.

Normative behavior:

- Stage-2 MUST NOT author `objective.id: standard_ce` as its active correction
  objective;
- Stage-2 MUST NOT author `objective.id: research_teacher_forcing` as its active
  correction objective;
- Stage-2 rollout prefix remains roll-in context, not positive labels;
- Stage-2 loss component names remain owned by residual-set correction.

#### Scenario: Stage-2 objective does not move to Stage-1 namespace

- **GIVEN** `pipeline.id: stage2_rollout_correction`
- **WHEN** config loading parses objectives
- **THEN** it reads
  `stage2_rollout_correction.pipeline.objective[].name:
  residual_set_correction`
- **AND** it does not require a top-level `objective.id`.
