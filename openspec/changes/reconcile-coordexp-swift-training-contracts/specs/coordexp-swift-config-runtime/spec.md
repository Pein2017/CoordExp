## ADDED Requirements

### Requirement: Exact Resume Configuration Is Strict And Opt-In

Training configuration SHALL expose a strict `resume` object whose supported
modes are `disabled` and `exact_same_world_size`. The compatibility default
MUST be `disabled`. `resume.checkpoint_dir` MUST be absent or null when disabled
and MUST resolve from the YAML file that authors it when exact continuation is
selected. Unknown modes, unknown fields, or incompatible mode/path
combinations MUST fail during config resolution before cache, model, optimizer,
or accelerator mutation.

#### Scenario: Resume configuration is omitted

- **WHEN** a current training config does not author `resume`
- **THEN** the resolved config MUST record `resume.mode: disabled` and a null
  checkpoint path
- **AND** runtime MUST not publish exact training state.

#### Scenario: Exact resume path is authored

- **WHEN** a config selects `exact_same_world_size` and authors a checkpoint
  path
- **THEN** the path MUST resolve relative to the YAML file that declared it
- **AND** the resolved mode and absolute path MUST be preserved in
  `resolved_config.json`.

#### Scenario: Resume fields are incompatible

- **WHEN** a disabled config supplies a checkpoint path or an exact mode omits
  its checkpoint path
- **THEN** strict validation MUST fail before training-side mutation.

### Requirement: Exact Resume Requires Strict Deterministic Replay Admission

`exact_same_world_size` SHALL require the versioned deterministic-runtime mode
`strict_cuda_replay_v1`. Before Accelerator, model, or CUDA materialization on
every rank, admission MUST verify the launcher-supplied deterministic
environment and establish fail-closed deterministic algorithm settings. The
resolved seed, policy version, dependency/runtime baseline, world size, and
rank/device mapping MUST be included in exact-resume compatibility identity.
The compatibility runtime policy MAY remain available only when exact resume
is disabled.

#### Scenario: Exact resume selects the compatibility runtime policy

- **WHEN** `resume.mode` is `exact_same_world_size` but the deterministic mode
  is not `strict_cuda_replay_v1`
- **THEN** config or runtime admission MUST fail before model or CUDA setup
- **AND** it MUST NOT silently strengthen or weaken the authored policy.

#### Scenario: A launcher prerequisite is absent

- **WHEN** strict replay is selected but a required launcher-provided CUDA,
  CUBLAS, or FlashAttention determinism prerequisite is absent or malformed
- **THEN** admission MUST fail before CUDA initialization
- **AND** the process MUST NOT synthesize a value after launch and claim exact
  compatibility.

#### Scenario: Rank runtime identities disagree

- **WHEN** ranks disagree on seed, policy, dependency baseline, world size, or
  rank/device mapping
- **THEN** all live ranks MUST converge the incompatibility before mutable
  training state is restored.
