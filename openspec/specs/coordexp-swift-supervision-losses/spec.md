# coordexp-swift-supervision-losses Specification

## Purpose
TBD - created by archiving change rebuild-coordexp-swift-training-infra. Update Purpose after archive.
## Requirements
### Requirement: Loss Runner Streaming Protocol Is Required At Construction

Training and eval construction SHALL require the paired `LossRunner` to
implement the full streaming protocol — `prepare_planned_step`,
`compute_micro_step`, and `finalize_planned_step` — and MUST fail closed at
construction time, before any planned step begins or any micro-step forward
runs, when any of the three is absent. This check MUST be unconditional: it
applies to every `SupervisedTrainer` construction (with or without a
`forward_input_provider`) and every `ForwardEvalRunner` construction (in
every reduction mode, `replicated` or `disjoint_shard`), not only specific
provider/mode combinations. Whole-planned-step batch loss computation (a
single call that returns a complete `LossBundle` for all contexts at once,
without the prepare/compute-micro-step/finalize sequence) is unsupported and
removed from both the trainer and the eval-forward runner; a loss runner
that implements only a batch-style call and not the streaming trio MUST be
rejected, never silently routed to a batch code path.

#### Scenario: Non-streaming loss runner rejected at trainer construction

- **WHEN** `SupervisedTrainer` is constructed with a `loss_runner` missing
  `prepare_planned_step`, `compute_micro_step`, or `finalize_planned_step`
- **THEN** construction MUST raise a contract error identifying the missing
  streaming protocol before any planned step begins
- **AND** no micro-step forward, loss computation, or gradient computation
  may occur for that trainer instance.

#### Scenario: Non-streaming loss runner rejected at eval construction

- **WHEN** `ForwardEvalRunner` is constructed with a `loss_runner` missing
  any member of the streaming trio
- **THEN** construction MUST raise a contract error identifying the missing
  streaming protocol, regardless of the configured reduction mode
- **AND** no eval forward pass may occur for that runner instance.

#### Scenario: Batch-only loss runner is rejected, not routed to a batch path

- **WHEN** a `loss_runner` implements a whole-planned-step batch `compute`
  call but not `prepare_planned_step`/`compute_micro_step`/`finalize_planned_step`
- **THEN** both trainer and eval construction MUST reject it as non-streaming
- **AND** MUST NOT fall back to computing the planned step's loss in one
  batch call, since no batch code path exists to fall back to.

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
explicitly defines that behavior. In packed training, `LossContext` MUST also
validate that `target_position` and its derived `logits_position` are inside
the same packed segment.

#### Scenario: First physical token selected as target

- **WHEN** a token atom has `target_position == 0`
- **THEN** standard causal loss-context construction MUST fail.

#### Scenario: First token of later segment selected as target

- **WHEN** a token atom targets the first physical token of a later packed
  segment
- **THEN** standard causal loss-context construction MUST fail or the approved
  supervision builder MUST omit that atom before loss construction
- **AND** the loss MUST NOT use logits from the previous packed segment.

### Requirement: Protected Default Losses

Supervised training SHALL include both `BaseTokenCE` and
`TokenTypeGateLoss` as protected losses. Base CE MUST compute full-vocabulary
CE over all supervised atoms, MUST be present in every supported supervised
config, and MUST have weight exactly `1.0`. Gate loss MUST enforce token-type
legality over the same supervision surface. Its canonical enabled weight MUST
be exactly `0.1`; weight `0` MUST be accepted only as an explicitly identified
zero-weight gate ablation. No other protected-loss weights are supported.

Every supervised `TokenAtom` MUST resolve to exactly one closed token type
before protected losses run. `TokenTypeGateLoss` MUST compute group-mass CE
from fp32 selected logits as `logsumexp(all_logits) -
logsumexp(allowed_group_logits)`. In the zero-weight gate ablation, the gate
calculation MUST run without autograd participation and MUST retain its raw,
weighted-zero, count, and finite diagnostics. A finite gate diagnostic MUST NOT
alter the base-CE objective or gradients. A non-finite protected gate
diagnostic MUST remain part of the all-rank pre-backward safety decision and
MUST prevent backward and optimizer update. These protected losses are the
canonical SFT baseline, not parity with archived coordinate soft-CE,
object-balanced, or role-balanced reduction semantics.

#### Scenario: Base CE omitted or reweighted

- **WHEN** a supported supervised config omits base CE or gives it a weight
  other than `1.0`
- **THEN** strict config validation MUST fail before loss construction or
  model mutation.

#### Scenario: Canonical protected baseline configured

- **WHEN** a normal supervised config enables the protected baseline
- **THEN** base CE MUST resolve to weight `1.0`
- **AND** token-type gate MUST resolve to weight `0.1` over all four canonical
  token groups.

#### Scenario: Named zero-weight gate ablation

- **WHEN** a supervised config explicitly identifies the zero-weight gate
  ablation and gives token-type gate weight `0`
- **THEN** the optimized objective and gradients MUST equal the base-CE-only
  objective for the same planned-step inputs
- **AND** gate raw, weighted-zero, count, and finite diagnostics MUST be
  computed without an autograd edge into the optimized objective.

#### Scenario: Zero-weight gate diagnostic is non-finite

- **WHEN** the named zero-weight gate ablation produces a non-finite raw
  protected diagnostic on any rank
- **THEN** all ranks MUST converge one unsafe scalar decision before backward
- **AND** backward and optimizer update MUST be skipped even though the gate
  has no autograd edge and weighted value zero.

#### Scenario: Unidentified zero-weight gate

- **WHEN** a supervised config gives token-type gate weight `0` without the
  explicit gate-ablation identity
- **THEN** strict config validation MUST fail before loss construction.

#### Scenario: Gate loss computed for coordinate target

- **WHEN** a coordinate-token atom is selected for `TokenTypeGateLoss`
- **THEN** the loss MUST compute
  `logsumexp(all selected logits) - logsumexp(<|coord_0|>..<|coord_999|> logits)`
  in fp32
- **AND** the weighted objective value MUST be the raw value multiplied by the
  resolved gate weight.

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
integer ranges hidden inside a loss function. The protected gate configuration
MUST contain exactly `desc_text`, `schema`, `coordinate`, and `eos`, with no
missing, duplicate, or additional group. The resolver MUST exclude Qwen
chat/control, image/video/pad, tool/FIM/repo, think, reserved CoordExp, and
other non-target special tokens from free-text allowance.

#### Scenario: Coordinate token target

- **WHEN** a supervised atom has token type `coordinate`
- **THEN** gate loss MUST treat `<|coord_0|>` through `<|coord_999|>` as the
  allowed target group
- **AND** MUST penalize probability mass assigned outside that group when the
  gate is enabled.

#### Scenario: Incomplete or reordered gate group set

- **WHEN** a supervised config omits, duplicates, adds, or reorders the
  canonical gate groups
- **THEN** strict config validation MUST fail before vocabulary resolution.

#### Scenario: Qwen control token in assistant target

- **WHEN** a supervised assistant target contains a non-approved Qwen control
  token
- **THEN** encoding or supervision validation MUST fail before loss
  computation.

#### Scenario: Atom without resolved token type

- **WHEN** a supervised atom lacks a resolved token type
- **THEN** protected loss setup MUST fail before objective math
- **AND** the diagnostic MUST include the atom position and span provenance.

### Requirement: Planned-Step Loss Normalizers

Computed token-wise losses SHALL be length-invariant over the planned
optimizer-step window. The reducer MUST be `segment_balanced`: compute the mean
loss over eligible atoms within each eligible segment, then the mean over
eligible segments in the complete planned optimizer-step window across
accumulation and ranks. Per-term denominators MUST be computed from the
complete planned-step window, not from rank-local windows or pack-local means
averaged afterward. In distributed training, rank-local denominator
contributions MUST be gathered before backward; differentiable rank-local
contributions MUST be scaled exactly once to compensate the backend's mean
gradient reduction so the effective objective and update equal the same
global planned-step objective at world size one.

Segments with zero eligible atoms MUST be excluded from a computed term's
denominator. An enabled protected or auxiliary term MUST fail if the complete
planned-step window has zero eligible segments. A disabled optional auxiliary
term MUST construct no denominator and therefore MUST NOT trigger an empty
denominator failure. The global semantic raw and weighted values used for
telemetry MUST remain distinct from any rank-local backward compensation.

#### Scenario: Different token counts across micro-steps

- **WHEN** two micro-steps in one planned optimizer step contain different
  numbers of supervised tokens
- **THEN** each computed loss MUST divide by its planned-step
  `segment_balanced` denominator
- **AND** MUST NOT average two already-normalized micro-step losses equally.

#### Scenario: Unequal rank-local segment counts

- **WHEN** distributed ranks contribute unequal numbers of eligible segments
  to one planned optimizer step
- **THEN** all ranks MUST use the same all-rank `planned_step_global`
  denominator for that term
- **AND** the resulting gradient and optimizer update MUST match the equivalent
  world-size-one planned-step calculation within declared numerical tolerance.

#### Scenario: Backend compensation is not telemetry

- **WHEN** distributed backward requires a world-size compensation factor
- **THEN** the factor MUST affect the differentiable rank-local contribution
  exactly once
- **AND** raw and weighted semantic telemetry MUST NOT be multiplied by that
  backend compensation factor.

#### Scenario: Disabled auxiliary has no denominator

- **WHEN** an optional auxiliary loss has weight `0`
- **THEN** no local or global denominator for that loss may be constructed
- **AND** no zero-eligible failure for that loss may affect the planned step.

#### Scenario: Segment-balanced differs from token-balanced

- **WHEN** two segments in one planned step have unequal eligible atom counts
- **THEN** protected token-wise losses MUST weight the two segment means
  equally under `segment_balanced`
- **AND** a single global mean over all eligible atoms MAY be emitted only as a
  diagnostic metric, not as the protected objective.

### Requirement: Loss Bundle Metrics

The completed planned-step loss result SHALL contain the total weighted loss,
explicit raw and weighted values for every computed term, top-level
`acc_top1`, top-level `acc_top5`, matching selected-count diagnostics, and
finite-status diagnostics. Raw values MUST express the globally normalized
term before configured weighting; weighted values MUST equal raw values times
their configured weight; total loss MUST equal the sum of weighted objective
terms. A zero-weight protected gate ablation remains a computed diagnostic term
and MUST expose raw, weighted-zero, count, and finite fields. A zero-weight
optional auxiliary term is not computed and MUST expose none of those fields.

After all-rank reduction, the complete scalar mapping for one planned step
SHALL be written together in that step's wide `train` row in `logging.jsonl`.
Cross-rank reduction of `acc_top1` and `acc_top5` MUST be derived from exact
rank-local sufficient statistics — integer correct counts and rank-local
supervised-atom counts — summed across ranks before the ratio is formed. It
MUST NOT average rank-local ratios, weight them by an already-global count, or
reconstruct integer counts from rounded ratios. At world size one the reduced
value MUST equal the rank-local value. Non-finite computed fields MUST retain
their names with JSON `null` and MUST be listed in `non_finite_fields`.

#### Scenario: Train metric event emitted

- **WHEN** a train step completes with enabled base CE and token-type gate
- **THEN** its single logging row MUST include unambiguous raw and weighted
  fields for both terms and their matching counts and finite fields
- **AND** it MUST include top-level `acc_top1` and `acc_top5`.

#### Scenario: Zero-weight gate metric event emitted

- **WHEN** a train or eval step uses the named zero-weight gate ablation
- **THEN** the gate raw diagnostic MUST remain visible and the gate weighted
  value MUST be zero
- **AND** the gate fields MUST be detached from autograd and distinguishable
  from optimized objective terms.

#### Scenario: Disabled optional metric fields omitted

- **WHEN** an optional auxiliary term resolves to weight `0`
- **THEN** its raw, weighted, denominator, count, and finite fields MUST all be
  absent from the completed-step result.

#### Scenario: Unequal rank-local atom counts

- **WHEN** distributed ranks contribute unequal supervised-atom counts to one
  planned step
- **THEN** reduced `acc_top1` and `acc_top5` MUST equal the pooled ratio of
  summed integer correct counts over summed atom counts
- **AND** MUST NOT be the plain mean of per-rank accuracies.

### Requirement: Non-Finite Loss And Gradient Gates

Training SHALL separate scalar loss finite checks before backward from
gradient/overflow checks after backward. Unsafe non-finite state MUST prevent a
corrupted optimizer update. Recoverable bad examples or warnings MAY be
reported without changing the planned-step schedule. In distributed execution,
the scalar finite check MUST produce one reduced all-rank decision before any
rank calls backward. The pre-backward decision MUST consider the raw finite
status of every computed protected term — including a protected diagnostic
whose weighted contribution is zero, such as the named zero-weight gate
ablation — not only the total optimized loss. The resulting update and finite
status MUST be represented once in the rank-zero train logging row for that
planned step; normal training MUST NOT write duplicate rank-local gate
receipts.

#### Scenario: Non-finite scalar loss

- **WHEN** total loss is NaN or Inf before backward
- **THEN** runtime MUST reduce the unsafe scalar status across all ranks before
  any rank calls backward
- **AND** all ranks MUST skip backward and optimizer update for that planned
  step
- **AND** runtime MUST clear accumulated gradients and maintain planned-step
  scheduler/update policy
- **AND** rank zero MUST log the synchronized unsafe/update status once.
- **AND** non-finite scalar fields MUST be represented as JSON `null` and named
  in `non_finite_fields`.

#### Scenario: Non-finite protected diagnostic with finite total loss

- **WHEN** a computed protected term's raw value is NaN or Inf on any rank
  while the total optimized loss remains finite (for example the zero-weight
  gate ablation, whose weighted contribution is exactly zero)
- **THEN** the all-rank pre-backward decision MUST classify the planned step
  unsafe
- **AND** all ranks MUST skip backward and optimizer update for that planned
  step
- **AND** the term's non-finite fields MUST be represented as JSON `null` and
  named in `non_finite_fields`.

#### Scenario: Distributed gradient overflow

- **WHEN** any rank reports unsafe gradient or overflow status
- **THEN** all ranks MUST use the same global skip/update decision for that
  planned step
- **AND** rank zero MUST log that global decision once.

#### Scenario: Zero eligible protected atoms on one rank

- **WHEN** a protected loss has zero eligible atoms on one rank
- **THEN** runtime MUST include the rank-local eligible count in the same
  planned-step all-rank denominator/finite decision before any rank raises or
  calls backward
- **AND** the global decision MUST avoid distributed deadlock.

### Requirement: Future Auxiliary Loss Seam

The supervised loss surface SHALL expose a closed typed set of implemented
protected and auxiliary losses with a declared role, normalization policy, and
zero-weight policy for each term. The supported public config MUST select only
that closed set and MUST NOT accept Python import paths, arbitrary callables,
or a dynamic loss registry. Coordinate Gaussian/RPS SHALL be a typed optional
auxiliary rather than a protected loss. Hidden-state, rollout-derived, and RL
loss composition remain unsupported unless a later approved change promotes
them.

#### Scenario: Coordinate Gaussian/RPS enabled

- **WHEN** a supervised config enables coordinate Gaussian/RPS with positive
  weight under the auxiliary surface
- **THEN** it MUST participate in fp32, planned-step `segment_balanced`
  objective computation and expose raw and weighted telemetry.

#### Scenario: Coordinate Gaussian/RPS disabled

- **WHEN** coordinate Gaussian/RPS is absent or has weight `0`
- **THEN** it MUST be omitted from loss computation, denominator construction,
  metrics, and finite checks.

#### Scenario: Arbitrary loss implementation configured

- **WHEN** public config attempts to select a loss by import path, callable,
  or unrecognized registry name
- **THEN** strict config validation MUST fail before loss construction.

#### Scenario: Hidden-state loss configured in V1

- **WHEN** a V1 config enables a hidden-state-dependent loss before its
  approved implementation exists
- **THEN** loss configuration MUST fail with an explicit unsupported-loss
  diagnostic.

#### Scenario: Rollout-derived loss configured

- **WHEN** a supervised config enables a rollout-derived or RL-composition
  loss before a later approved contract exists
- **THEN** loss configuration MUST fail with an explicit unsupported-loss
  diagnostic.

