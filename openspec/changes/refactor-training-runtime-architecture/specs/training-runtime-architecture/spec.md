# training-runtime-architecture Specification

## Purpose

Define the spec-first architecture for CoordExp training runtime refactors so
future implementation can proceed from OpenSpec to code while preserving
training behavior, artifact contracts, and reproducibility.

## ADDED Requirements

### Requirement: Training variant setup policy SHALL have one import-safe source of truth

The training runtime architecture SHALL expose one lightweight setup plan for
trainer variant policy before trainer construction.

Normative behavior:

- the setup plan MUST be import-safe and MUST NOT import torch, transformers,
  ms-swift, vLLM, trainer classes, datasets, or config loaders;
- the setup plan MUST identify metadata retention, dataset static-packing
  ownership, post-rollout packing ownership, collator family, mixin
  eligibility, and required pipeline namespace;
- removed trainer variants MUST fail fast with replacement guidance;
- trainer-facing runtime-profile helpers MUST derive from the setup plan rather
  than maintaining a second policy registry.

#### Scenario: Runtime profile and setup plan cannot drift

- **WHEN** a Stage-2 trainer variant is inspected through a trainer-facing
  runtime profile
- **THEN** the profile derives explicit-pipeline requirement, rollout
  ownership, collator family, and mixin exclusion from the same setup plan used
  by bootstrap code.

### Requirement: Stage-1 setup ownership SHALL preserve current objective semantics

The training runtime architecture SHALL preserve ordinary Stage-1 SFT and
Stage-1 set-continuation setup behavior.

Normative behavior:

- ordinary Stage-1 SFT MUST remain eligible for dataset static packing and
  ordinary one-sequence mixins;
- `stage1_set_continuation` MUST preserve raw sample metadata;
- `stage1_set_continuation` MUST reject dataset train/eval packing;
- `stage1_set_continuation` MUST select its metadata-preserving collator;
- `stage1_set_continuation` MUST exclude ordinary one-sequence Stage-1 mixins;
- Stage-1 set-continuation branch scoring MUST stay on the current repeated
  independent forward / `smart_batched_exact` production runtime unless a
  dedicated parity and performance gate promotes an alternate branch execution
  path;
- candidate scoring, branch construction, loss math, and token-gate semantics
  MUST NOT move without dedicated parity coverage.

#### Scenario: Stage-1 set-continuation remains branch-owned

- **WHEN** setup resolves `custom.trainer_variant: stage1_set_continuation`
- **THEN** dataset packing is not selected
- **AND** raw metadata survives collation
- **AND** branch execution remains independent-branch scoring rather than
  dataset-level static packing
- **AND** branch-local objective code remains behaviorally unchanged.

### Requirement: Stage-2 setup ownership SHALL preserve rollout and pipeline contracts

The training runtime architecture SHALL preserve Stage-2 two-channel and
rollout-aligned setup behavior.

Normative behavior:

- Stage-2 variants MUST preserve raw sample metadata;
- Stage-2 variants MUST disable dataset static packing;
- `training.packing=true` MUST continue to mean trainer-owned dynamic
  post-rollout packing for Stage-2;
- Stage-2 variants MUST select identity collation;
- Stage-2 variants MUST exclude ordinary Stage-1 mixins;
- `stage2_two_channel` MUST require `stage2_ab.pipeline`;
- `stage2_rollout_aligned` MUST require `rollout_matching.pipeline`;
- Stage-2 rollout runtime settings MUST remain under top-level
  `rollout_matching`;
- target construction, rollout matching, duplicate control, pseudo-positive
  logic, teacher-forcing objectives, metric keys, and artifact names MUST NOT
  change in setup-only slices.

#### Scenario: Stage-2 setup requires authored pipeline namespace

- **WHEN** setup resolves `custom.trainer_variant: stage2_two_channel`
- **THEN** the architecture requires authored `stage2_ab.pipeline`
- **AND** rejects fallback defaults that would hide missing pipeline config.

### Requirement: Implementation SHALL proceed in OpenSpec-governed slices

The refactor SHALL restart from this OpenSpec rather than from the deprecated
dirty code implementations.

Normative behavior:

- implementation branches MUST not directly port code from deprecated
  worktrees without re-deriving the change from this OpenSpec;
- each code slice MUST add focused tests before or with implementation;
- setup-only slices MUST prove existing setup behavior is preserved;
- math-bearing, target-construction, optimizer, geometry, metric, or artifact
  changes MUST have explicit parity tests or a separate behavior-changing
  OpenSpec delta;
- docs MUST be updated in the same slice when user-facing routing, defaults,
  artifact names, or stable workflows change.

#### Scenario: Deprecated code is not the implementation source

- **GIVEN** an old worktree contains dirty implementation files
- **WHEN** the refactor resumes
- **THEN** the implementation starts from this OpenSpec and current `main`
- **AND** old dirty files are treated as deprecated research input, not patch
  sources.

### Requirement: Future recipe and run orchestration MUST cite canonical evidence

Future agent-friendly orchestration SHALL introduce mission, recipe, run,
stage-result, summary, and note records only as explicit repo-visible artifacts.

Normative behavior:

- strategic research work SHOULD compile intent into a mission before
  expensive execution;
- recipes SHOULD declare stage identities, dependencies, required inputs, and
  produced outputs;
- each run SHOULD have one canonical run record;
- each stage attempt SHOULD have one stage-result record;
- summaries and reviews MUST be derived from canonical run/stage/artifact
  evidence;
- smoke and scale SHOULD be explicit profiles with preflight and scale gates;
- hidden agent-only persistence MUST NOT be introduced;
- notes or memory records, if introduced, MUST link to authoritative evidence
  and MUST NOT replace run truth.

#### Scenario: A future smoke-to-scale workflow is evidence-gated

- **GIVEN** a future mission requests scale execution
- **WHEN** the runtime evaluates the request
- **THEN** it checks preflight and prior smoke evidence
- **AND** records the approval rationale through explicit repo-visible
  mission/run evidence rather than hidden chat state.
