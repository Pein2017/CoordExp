# dataset-prompt-variants Spec Delta

This is a delta spec for change `epoch-random-ordering-ablation`.

## MODIFIED Requirements

### Requirement: Variant Registry and Deterministic Resolution
The system SHALL provide a centralized prompt-variant registry for dense detection prompts and SHALL resolve prompt variants by explicit key with deterministic behavior.

Prompt resolution inputs for dense prompts MUST include:
- prompt variant key,
- ordering policy,
- object field order, and
- coord mode.

#### Scenario: Default variant fallback
- **WHEN** no prompt variant is specified in training or inference configuration
- **THEN** the system MUST resolve to the backward-compatible `default` variant

#### Scenario: Unknown variant rejection
- **WHEN** a configuration specifies a prompt variant key that is not registered
- **THEN** configuration resolution MUST fail with an error that includes the unknown key and available variant keys

#### Scenario: Deterministic repeated resolution
- **WHEN** the same variant key is resolved repeatedly with the same prompt inputs (ordering, object field order, and coord mode)
- **THEN** the resolver MUST return byte-identical system and user prompt text across calls
- **AND** `coord mode` refers to the resolver `coord_mode` input (current contract: `coord_tokens`).

#### Scenario: Deterministic cross-surface resolution
- **WHEN** training, trainer-driven rollout/eval prompt rebuilding, and standalone inference resolve the same variant key with equivalent prompt inputs
- **THEN** all of those surfaces MUST produce equivalent policy instructions in system and user prompts

### Requirement: Cross-Surface Prompt Parity
Training, trainer-driven rollout/eval, and standalone inference prompt construction SHALL use the same variant-aware resolver so that a selected variant produces equivalent policy instructions across surfaces.

Normative behavior:
- Training dense prompts MUST resolve variant, object field order, and ordering policy from the active training config.
- Trainer-driven rollout/eval dense prompts MUST reuse the same resolver inputs as the active training config.
- Standalone inference dense prompts MUST resolve variant from `infer.prompt_variant` and ordering policy from `infer.object_ordering`.

#### Scenario: Training variant application
- **WHEN** `custom.extra.prompt_variant` is set in a training config
- **THEN** training prompt resolution MUST apply that variant for dense prompts

#### Scenario: Rollout/eval variant application
- **WHEN** a trainer rebuilds dense prompts for rollout generation or trainer-driven evaluation
- **THEN** it MUST apply the same variant semantics as training for equivalent resolver inputs

#### Scenario: Inference variant application
- **WHEN** `infer.prompt_variant` is set in an inference pipeline config
- **THEN** inference message construction MUST apply the same variant semantics to system and user prompts

#### Scenario: Inference ordering application
- **WHEN** `infer.object_ordering` is set in an inference pipeline config
- **THEN** inference message construction MUST apply the same sorted/random ordering semantics as the shared dense prompt resolver

#### Scenario: Explicit parity guidance
- **WHEN** a checkpoint was trained with a non-default prompt variant or non-default ordering policy
- **THEN** documentation and config examples MUST specify that inference SHOULD use the same variant and ordering for reproducible evaluation

### Requirement: Variant Metadata in Inference Artifacts
Inference artifacts SHALL include the resolved prompt variant and resolved dense ordering policy for reproducibility auditing.

#### Scenario: Resolved config metadata
- **WHEN** an inference pipeline run emits `resolved_config.json`
- **THEN** the file MUST record the resolved prompt variant under inference configuration metadata
- **AND** it MUST record the resolved dense ordering policy used to build generation messages

#### Scenario: Summary metadata
- **WHEN** inference emits summary output
- **THEN** the summary MUST include the resolved prompt variant value used to build generation messages
- **AND** it MUST include the resolved dense ordering policy value
