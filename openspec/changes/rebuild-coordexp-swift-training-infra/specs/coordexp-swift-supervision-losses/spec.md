## ADDED Requirements

### Requirement: TokenSequence Is Canonical Supervision

CoordExp-swift SHALL use `TokenAtom -> TokenSpan -> TokenSequence` as the
canonical token-wise supervision hierarchy. Dense label tensors MAY be
materialized as derived compatibility or debug artifacts, but MUST NOT carry
independent loss semantics absent from `TokenSequence`.

#### Scenario: Dense labels emitted for parity

- **WHEN** dense labels are emitted for a parity check
- **THEN** they MUST be reproducible from the active `TokenSequence`
- **AND** objective modules MUST still consume the validated token records.

### Requirement: TokenAtom Target Semantics

Each `TokenAtom` SHALL represent a supervised target token position, not a
logit row directly. `LossContext` MUST derive the causal
`logits_position = target_position - 1` for standard language-model losses and
MUST reject `target_position == 0` unless a future non-causal objective
explicitly defines that behavior.

#### Scenario: First physical token selected as target

- **WHEN** a token atom has `target_position == 0`
- **THEN** standard causal loss-context construction MUST fail.

### Requirement: Protected Default Losses

V1 training SHALL include both `BaseTokenCE` and `TokenTypeGateLoss` as
protected default losses. Base CE MUST compute full-vocabulary CE over all
supervised atoms. Gate loss MUST enforce token-type legality over the same
supervision surface when a token type is resolved. These protected V1 losses
MUST be treated as the new baseline training objective, not as parity with
archived production coordinate soft-CE, object-balanced, or role-balanced
reduction semantics. Any claim or implementation of those older objective
semantics MUST be introduced by a later explicit auxiliary-loss contract.

#### Scenario: Auxiliary-only loss config attempted

- **WHEN** a training config attempts to omit both protected losses for normal
  supervised training
- **THEN** loss configuration MUST fail unless a future approved research
  override explicitly changes the contract.

#### Scenario: Old production objective parity claimed

- **WHEN** a config, artifact, or report claims parity with archived coordinate
  soft-CE or object/role-balanced objective semantics
- **THEN** validation or review MUST reject the claim unless a later approved
  auxiliary-loss contract implements and verifies those semantics.

### Requirement: FP32 Objective Logits

Loss computation SHALL upcast selected logits to fp32 for objective math even
when model forward uses bf16 or another lower-precision dtype. Metrics MAY
detach tensors and MAY avoid fp32 unless the metric consumes the fp32 selection
for another approved reason.

#### Scenario: BF16 training step

- **WHEN** model logits are bf16 during training
- **THEN** `BaseTokenCE` and `TokenTypeGateLoss` MUST compute their objective
  math on selected fp32 logits.

### Requirement: Token Type Vocabulary Groups

`TokenTypeGateLoss` SHALL use resolved vocabulary groups rather than ad hoc
integer ranges hidden inside a loss function. V1 target token types are closed
to `desc_text`, `schema`, `coordinate`, and `eos`. The resolver MUST exclude
Qwen chat/control, image/video/pad, tool/FIM/repo, think, reserved CoordExp,
and other non-target special tokens from free-text allowance.

#### Scenario: Coordinate token target

- **WHEN** a supervised atom has token type `coordinate`
- **THEN** gate loss MUST treat `<|coord_0|>` through `<|coord_999|>` as the
  allowed target group
- **AND** MUST penalize probability mass assigned outside that group.

#### Scenario: Qwen control token in assistant target

- **WHEN** a supervised assistant target contains a non-approved Qwen control
  token
- **THEN** encoding or supervision validation MUST fail before loss
  computation.

### Requirement: Planned-Step Loss Normalizers

Loss normalizers SHALL be length-invariant over the planned optimizer-step
window. Per-term denominators MUST be computed from the complete planned-step
window across accumulation and ranks, not from pack-local means averaged
afterward. Runtime and loss code MUST avoid backend double scaling.

#### Scenario: Different token counts across micro-steps

- **WHEN** two micro-steps in one planned optimizer step contain different
  numbers of supervised tokens
- **THEN** protected loss normalization MUST divide by the planned-step
  denominator for the selected term
- **AND** MUST NOT average two already-normalized micro-step losses equally.

### Requirement: Loss Bundle Metrics

`LossRunner` SHALL return a `LossBundle` containing total weighted loss,
weighted per-term loss metrics, top-level `acc_top1`, top-level `acc_top5`,
selected-count diagnostics, and finite-status diagnostics. Stored per-term loss
metrics MUST be weighted values.

#### Scenario: Train metric event emitted

- **WHEN** a train metric event is written after a planned step
- **THEN** it MUST include weighted protected-loss metrics and top-level
  `acc_top1` and `acc_top5`
- **AND** top-1/top-5 metric names MUST NOT be nested under a base-CE metric
  namespace.

### Requirement: Non-Finite Loss And Gradient Gates

Training SHALL separate scalar loss finite checks before backward from
gradient/overflow checks after backward. Unsafe non-finite state MUST prevent a
corrupted optimizer update. Recoverable bad examples or warnings MAY be
recorded without changing the planned-step schedule. In distributed execution,
the scalar finite check MUST produce one reduced all-rank decision before any
rank calls backward.

#### Scenario: Non-finite scalar loss

- **WHEN** total loss is NaN or Inf before backward
- **THEN** runtime MUST reduce the unsafe scalar status across all ranks before
  any rank calls backward
- **AND** all ranks MUST skip backward and optimizer update for that planned
  step
- **AND** runtime MUST clear accumulated gradients, record scheduler/update
  status according to planned-step policy, and emit synchronized diagnostics.

#### Scenario: Distributed gradient overflow

- **WHEN** any rank reports unsafe gradient or overflow status
- **THEN** all ranks MUST use the same global skip/update decision for that
  planned step.

### Requirement: Future Auxiliary Loss Seam

The V1 loss abstraction SHALL allow future token-wise, site-wise, coordinate,
hidden-state, and rollout-derived losses to consume `TokenSequence`,
`LossContext`, model outputs, and typed metadata. Hidden-state and rollout
losses MUST remain unimplemented in V1 unless a later approved change promotes
them.

#### Scenario: Hidden-state loss configured in V1

- **WHEN** a V1 config enables a hidden-state-dependent loss before its
  approved implementation exists
- **THEN** loss configuration MUST fail with an explicit unsupported-loss
  diagnostic.
