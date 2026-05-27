# teacher-forcing-unified-loss-registry Delta

## MODIFIED Requirements

### Requirement: Canonical contexts, token types, and loss component names are shared

The live teacher-forcing registry SHALL expose the unified objective module
taxonomy and token roles used by the new teacher-forcing objective.

Normative behavior:

- canonical active modules MUST include `token_type_mass`,
  `conditional_valid_set_likelihood`, `within_valid_coverage`, and optional
  `continuation_margin`;
- token roles MUST be `SCHEMA`, `TEXT`, `COORD`, and `STOP`;
- old live component names such as `struct_ce`, `desc_ce`, `coord_reg`, `geo`,
  and `loss_duplicate_burst_unlikelihood` MUST NOT be required by the new
  teacher-forcing objective;
- duplicate-control, geometry, or coordinate-soft-CE historical diagnostics MAY
  remain outside the active teacher-forcing core when explicitly labeled as
  legacy or ablation.

#### Scenario: Active module registry contains typed valid-set modules

- **WHEN** the live teacher-forcing module registry is enumerated for
  `objective.id: teacher_forcing`
- **THEN** the active core module names are the typed valid-set modules
- **AND** old auxiliary-loss module names are absent from the core registry.

### Requirement: Loss component names and contexts are canonical and shared

The shared active teacher-forcing registry SHALL use the new module taxonomy for
`objective.id: teacher_forcing`.

Normative behavior:

- active objective components MUST be named `token_type_mass`,
  `conditional_valid_set_likelihood`, `within_valid_coverage`, and optional
  `continuation_margin`;
- active components MUST consume `TeacherForcingTargetIR` atoms rather than
  Stage-2-only objective atoms;
- old component names such as `struct_ce`, `desc_ce`, `geo`, `coord_reg`,
  `bbox_size_aux`, and `loss_duplicate_burst_unlikelihood` MUST NOT be required
  or emitted as core teacher-forcing components;
- historical Stage-2 analysis may preserve old terms only when explicitly
  labeled as legacy or ablation output.

#### Scenario: Core registry rejects old auxiliary component names

- **WHEN** `objective.id: teacher_forcing` resolves its active module registry
- **THEN** `coord_reg`, `geo`, `bbox_size_aux`, and
  `loss_duplicate_burst_unlikelihood` are absent from the active core modules.

### Requirement: Canonical loss component names and contexts are canonical and shared

The later duplicate registry requirement SHALL resolve to the same shared module
taxonomy as the primary canonical registry requirement.

Normative behavior:

- active core modules MUST be the same `teacher_forcing` module names across
  Stage-1 and Stage-2 adapters;
- Stage-specific adapters MAY add provenance and context metadata to
  `TeacherForcingTargetIR`;
- Stage-specific adapters MUST NOT introduce alternate active names for the core
  loss modules.

#### Scenario: Stage adapters share objective module names

- **WHEN** Stage-1 and Stage-2 adapters request active teacher-forcing modules
- **THEN** both resolve the same core module names.

### Requirement: Canonical loss scalars are mean-like and scale-invariant

Active teacher-forcing component scalars SHALL be mean-like values for the new
module taxonomy.

Normative behavior:

- `token_type_mass`, `conditional_valid_set_likelihood`,
  `within_valid_coverage`, and optional `continuation_margin` MUST aggregate as
  mean-like scalars over their contributing `SupervisionAtom` records;
- any sum/count values needed to form means MUST remain internal-only or be
  emitted under explicit counter-like names;
- old scalar names such as `struct_ce`, `desc_ce`, `geo`, and `coord_reg` MUST
  NOT be active core scalar names for `objective.id: teacher_forcing`.

#### Scenario: Loss scalars do not scale with atom count

- **WHEN** two forwards contain different numbers of supervised atoms but
  identical per-atom distributions
- **THEN** active teacher-forcing loss scalars are comparable as means.

### Requirement: Canonical loss scalars are mean-like (scale-invariant)

The duplicate scale-invariant scalar requirement SHALL use the same mean-like
aggregation contract as the primary active teacher-forcing scalar requirement.

Normative behavior:

- active module scalars MUST aggregate over contributing `SupervisionAtom`
  records rather than packing length;
- old decoded-box or coord-regularizer scalar names MUST NOT be emitted as core
  teacher-forcing objective losses.

#### Scenario: Packed sequences do not define active scalar names

- **WHEN** active teacher-forcing scalars are logged
- **THEN** they use the new module names
- **AND** no old decoded-box scalar is required for scale invariance.

### Requirement: Token-type partition is explicit and deterministic

The active teacher-forcing objective SHALL use the shared token-role vocabulary.

Normative token roles:

- `SCHEMA`: active schema/control tokens, including object and box markers.
- `TEXT`: free-text description tokens.
- `COORD`: coord-vocabulary tokens `<|coord_k|>`.
- `STOP`: stop token `<|im_end|>`.

Normative behavior:

- role assignment MUST be deterministic for a tokenizer, token id, and template
  policy;
- `STOP` MUST NOT be double-counted as `SCHEMA`;
- old role names `struct`, `desc`, and `eos` MAY appear only in historical
  registry text or migration notes, not as active target IR role names.

#### Scenario: STOP token is counted exactly once

- **WHEN** token-role masks are built for a sequence containing `<|im_end|>`
- **THEN** the stop token receives role `STOP`
- **AND** it does not receive role `SCHEMA`.

### Requirement: Channel-B rollout context is explicit, triage-aware, and EOS-enforced

Stage-2 Channel-B SHALL adapt rollout context into `TeacherForcingTargetIR`
without reviving old decoded-box auxiliary losses.

Normative behavior:

- rollout subsets such as matched, pseudo-positive, shielded, duplicate, FN, and
  recovered-FN objects MAY be represented as target-builder provenance metadata;
- duplicate objects MUST NOT create positive teacher-forcing target atoms;
- retained context objects MAY influence prefix/context construction but MUST
  not create geometry or coord-regression losses in the active core;
- FN and recovered-FN objects MAY create positive teacher-forcing atoms through
  the shared target IR;
- EOS or stop supervision MUST be represented through `STOP` atoms and/or
  continuation diagnostics.

#### Scenario: Duplicate-certified rollout objects do not create positive atoms

- **WHEN** a rollout object is classified as duplicate
- **THEN** it does not contribute positive `SupervisionAtom` records
- **AND** any duplicate information remains diagnostic metadata.

### Requirement: Channel-B rollout context is FP-neutral and EOS-enforced

The later Channel-B rollout-context requirement SHALL use the same shared target
IR adaptation contract.

Normative behavior:

- anchor-edited clean-prefix semantics MAY remain target-builder provenance;
- pseudo-positive and shielded anchor distinctions MAY affect which
  teacher-forcing atoms are emitted;
- old positive `geo`, `coord_reg`, or bbox auxiliary supervision surfaces MUST
  NOT be part of the active core objective.

#### Scenario: Pseudo-positive metadata does not revive geometry losses

- **WHEN** pseudo-positive rollout metadata is present
- **THEN** active objective modules still consume only the shared
  teacher-forcing atoms and module taxonomy.

### Requirement: Stage-2 Channel-A uses GT context only

Stage-2 Channel-A SHALL adapt GT teacher-forcing context into the shared target
IR.

Normative behavior:

- Channel-A MUST use GT context for emitted `TeacherForcingTargetIR` atoms;
- Channel-A MUST NOT introduce a separate `self_context` registry context;
- Channel-A MUST NOT require decoded-box geometry or coord-regularizer modules
  in the active teacher-forcing core.

#### Scenario: Channel-A emits GT-context target IR

- **WHEN** Stage-2 Channel-A constructs objective inputs
- **THEN** emitted atoms use GT-context provenance
- **AND** no `self_context` context is constructed.

### Requirement: Recovered FN weighting is part of rollout-context supervision

Recovered-FN metadata SHALL be represented as target-builder provenance and atom
weights in the shared teacher-forcing target IR.

Normative behavior:

- recovered FN objects MUST be identifiable in rollout-context metadata when the
  Stage-2 adapter supports them;
- positive CE/valid-set atom weights derived from recovered FN objects MAY use
  the configured recovered-FN weight;
- recovered-FN weighting MUST NOT require `geo`, `coord_reg`, or bbox auxiliary
  modules in the active teacher-forcing core.

#### Scenario: Recovered FN weight changes only recovered atoms

- **WHEN** a rollout-context sample contains ordinary FN and recovered-FN objects
- **THEN** only atoms derived from the recovered-FN subset use the recovered-FN
  weight.

## REMOVED Requirements

### Requirement: Stage-2 objective-atom projection is module-owned and deterministic

Old Stage-2 module-owned objective-atom projection is removed from the active
teacher-forcing core. Stage adapters emit `TeacherForcingTargetIR` directly, and
shared modules consume explicit `SupervisionAtom` records.

### Requirement: Gate terms are logit-derived and require no new heads

Old gate-term requirements are removed from the active teacher-forcing core.
Token-type exclusivity is represented by the `token_type_mass` module over
full-vocabulary probabilities.

### Requirement: Gate terms respect context-specific masking (FP-neutral and desc-disabled spans)

Old context-specific gate masking is removed from the active teacher-forcing
core. Stage-specific target builders express supervised applicability through
`SupervisionAtom.loss_tags` and target IR metadata.

### Requirement: Gate terms respect context-specific masking and pseudo-positive coord-positive participation

Old pseudo-positive gate participation rules are removed from the active
teacher-forcing core. Any future pseudo-positive behavior must be expressed as a
target-builder policy and explicit IR metadata.

### Requirement: Geometry decode follows the fixed expectation path

The decoded-box geometry path is removed from active teacher-forcing objective
math for this refactor.

### Requirement: Geometry loss (`geo`) uses canonicalized boxes and a stable decomposition

The `geo` loss component is removed from the active teacher-forcing objective.

### Requirement: Coord regularizer loss (`coord_reg`) is explicit and strictly configured

The `coord_reg` component is removed from the active teacher-forcing objective.

### Requirement: `bbox_size_aux` is a separate optional decoded-box loss component

The `bbox_size_aux` component is removed from the active teacher-forcing
objective.
