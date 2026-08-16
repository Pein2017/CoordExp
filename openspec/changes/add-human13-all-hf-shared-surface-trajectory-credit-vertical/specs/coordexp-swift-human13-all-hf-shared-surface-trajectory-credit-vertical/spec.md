## Purpose

Defines a bounded Human-13 research path that spends extra HF sampling compute
to test K-trajectory credit, greedy compilation, and Source-owner preservation
through one real private update before considering scalable rollout engines.

## ADDED Requirements

### Requirement: Shared HF training surface

The probe SHALL use one live HF model object for stochastic sampling and
gradient-bearing replay within a proposal.  The checkpoint payload, adapter
representation, selected-token embedding delta, BF16 dtype, FlashAttention-2
backend, model mode, tokenizer, prompt, image inputs, and repetition-penalty
processor order MUST remain unchanged until the update has been proposed.
The probe MUST NOT use vLLM evidence in its score-function objective.

#### Scenario: Surface identity is admitted

- **WHEN** a one-image proposal is prepared
- **THEN** one receipt binds the model object, parameter state, adapter,
  embedding delta, dtype, attention backend, tokenizer, prompt, image, model
  mode, and processor order used by both sampling and replay

#### Scenario: Surface identity changes

- **WHEN** any bound surface field or model parameter changes between sampling
  and gradient replay
- **THEN** the proposal fails before backward or owner analysis

### Requirement: Compute-expensive K16 HF acquisition

The probe SHALL acquire exactly sixteen stochastic trajectories for each
selected image as four sequential logical groups of four.  Sampling SHALL use
HF no-cache full-history causal forwards, temperature `0.4`, `top_p=1.0`, no
top-k truncation, `max_new_tokens=512`, Qwen `<|im_end|>` as the only stop, and
the exact sign-aware repetition-penalty transform before temperature.  The
initial vertical SHALL train at repetition penalty `1.0` with frozen seeds
`35001..35016` on image `1584`.  This seed group is disjoint from every
declared predecessor qualification and matrix group.

#### Scenario: Complete one-image acquisition

- **WHEN** all four image-1584 groups complete under the frozen policy
- **THEN** the acquisition contains exactly sixteen request identities,
  histories, chosen tokens, generation-time processed log probabilities, stop
  reasons, and per-step active-batch shape receipts

#### Scenario: KV or prefix reuse is attempted

- **WHEN** the experiment-local sampler enables `use_cache`, past-key-value
  reuse, or cross-request prefix reuse
- **THEN** acquisition fails before publication

### Requirement: Shared-surface replay admission

The probe SHALL replay each completed four-trajectory group through the same
HF model object by reconstructing every recorded sampler step as one no-cache
teacher-forced BF16/FA2 forward with gradients enabled and model parameters
unchanged.  Each replay forward SHALL use the sampler's exact active request
membership, causal history length, and selected causal logit positions; it SHALL
not right-pad completed histories or group steps merely because membership is
unchanged.  Position-selective logits and bounded activation checkpointing (or
an equivalent resource-bounded mechanism) SHALL preserve the live graph for the
single later proposal backward without retaining full-sequence vocabulary
logits.  Replay SHALL reconstruct the sampled processed policy at every chosen token.  Admission requires exact history and
chosen-token identity, finite values, maximum absolute processed-logprob error
no greater than `0.02` nats, and group-mean absolute error no greater than
`0.002` nats.

#### Scenario: Shared-surface replay passes

- **WHEN** all sixteen trajectories satisfy lineage, token, finiteness, maximum
  error, and mean error requirements
- **THEN** the already-materialized replay logits may feed the one-update loss
  without another policy-changing forward

#### Scenario: Shared-surface replay fails

- **WHEN** a history, chosen token, processor transform, surface identity, or
  numeric tolerance fails
- **THEN** the probe records a typed parity failure, performs zero optimizer
  steps, and does not reinterpret or widen the tolerance

### Requirement: Complete one-update objective

After admission, the first vertical SHALL perform exactly one fresh-AdamW
proposal using the predecessor's frozen K-trajectory first-hit credit,
duplicate/unmatched/premature-STOP costs, sparse Source-boundary greedy
compiler, and owner-wise preservation projection.  The learning rate SHALL be
`3e-6`, betas `(0.9,0.999)`, epsilon `1e-8`, weight decay zero, compiler
coefficient `1.0`, and compiler temperature `kappa=1`.  Legacy K-miss owners
M SHALL remain neutral and masked from direct positive credit.

#### Scenario: One complete proposal is formed

- **WHEN** parity passes and every trajectory/compiler/preservation input is
  admitted
- **THEN** exactly one actual AdamW delta is reconstructed, projected against
  the frozen Source-owner witnesses, applied privately, and receipted with
  component numerators, denominator, gradient norm, unprojected/projected
  delta norms, and active constraints

#### Scenario: Objective component is unavailable

- **WHEN** trajectory credit, a required compiler site, preservation witness,
  AdamW reconstruction, or projected-apply certification is unavailable or
  non-finite
- **THEN** the proposal fails closed rather than silently dropping that
  component or substituting CE

### Requirement: Dual-RP clean-greedy decision

The private proposal SHALL be evaluated under original-prompt clean HF greedy
decode at repetition penalties `1.0` and `1.10` using the same canonical
parser, chronological class-agnostic `IoU>0.95` duplicate exclusion, and
cardinality-first one-to-one owner matcher as Source.  Each surface SHALL
report H gained, G lost, incidental M gained, total unique-owner delta,
duplicate rows, unmatched rows, malformed rows, stop reason, row count, and
token count.

#### Scenario: One-image continuation gate passes

- **WHEN** the RP-1.0 audit gains at least one trusted H owner, loses zero
  RP-1.0 Source G owners, has positive unique-owner delta, both RP audits lose
  zero of their own Source G owners, and neither audit adds a duplicate,
  malformed output, or cap termination
- **THEN** one conditional full-panel shared-surface update may be proposed in
  the same change without changing the objective or learning rate

#### Scenario: One-image update is informative but unsafe or null

- **WHEN** the private proposal completes but the continuation gate does not
  pass
- **THEN** the result is published as bounded one-image algorithm evidence and
  the 13-image continuation remains unexecuted

### Requirement: Exact rollback and no promotion

Every proposal SHALL begin from a complete Source training-state transaction.
After both audits, or after any post-backward failure, model parameters,
optimizer, scheduler, counters, CPU/CUDA RNG, gradients, and temporary proposal
bytes MUST be restored or durably marked as rollback failure.  No proposal
checkpoint SHALL be promoted.

#### Scenario: Proposal audit completes

- **WHEN** both RP audits finish
- **THEN** Source is restored exactly and a reproduction decode confirms the
  Source owner, token, parser, and stop surfaces before the run closes

#### Scenario: Rollback cannot be certified

- **WHEN** any restored state digest or Source reproduction differs
- **THEN** the run terminates with a durable rollback-failure receipt and no
  later proposal or full-panel continuation executes

### Requirement: Conditional width expansion

The 13-image continuation SHALL remain absent until the one-image continuation
gate passes.  If authorized by that gate, it SHALL use the same K16 policy,
objective, learning rate, one-update limit, dual-RP audit, and rollback
semantics over the sealed Human-13 panel.  It SHALL remain an overfit-only
same-panel probe.

#### Scenario: One-image gate has not passed

- **WHEN** no admitted one-image receipt satisfies every continuation condition
- **THEN** the full-panel command refuses model or GPU execution

#### Scenario: Full-panel update completes

- **WHEN** an admitted one-image receipt authorizes the 13-image continuation
- **THEN** one shared-surface K16-per-image update is evaluated and reported
  without a validation, generalization, deployment, or scalable-throughput
  claim
