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

### Requirement: Independent policy baselines and diagnostic cross-surface evidence

The BF16/FlashAttention-2 training surface SHALL remain the sole authority for
sampling, replay, trajectory credit, greedy compilation, preservation, and the
update token/path.  It SHALL construct a BF16-native canonical Source
projection, compiler Source boundary, WitnessMeasurement/Jacobians, and
post-apply margin probe from its own free-running outputs.  The fp32/SDPA
surface SHALL remain a separate owner-level clean-greedy audit surface, with
durable Source baselines frozen at repetition penalties `1.0` and `1.10`.

The single admission constructor/loader/checker/publisher SHALL enforce strict
identity only for model/checkpoint/adapter/embedding/tokenizer/prompt/image/
manifest and declared processor policy.  It SHALL parse each surface with the
frozen canonical parser and apply the cardinality-first one-to-one owner
matcher independently on each surface.  BF16 admission SHALL require every
protected manifest G owner needed for preservation to be present in the BF16
Source baseline, while cross-surface token/row/coordinate/owner differences
SHALL be retained as `diagnostic_only` evidence and SHALL NOT gate the BF16
proposal.  The divergence receipt SHALL preserve token positions, roles,
bins/deltas, boxes, owners, IoUs, owner sets/maps, membership, protected-G
sets, and full-payload hashes where available.

This diagnostic path SHALL NOT relax any sampler-to-replay history, token,
shape, or processed-log-probability parity requirement on BF16/FA2.  A BF16
Source-covered H owner SHALL be baseline context rather than an H gain target;
trajectory credit MAY account for its first hits, but continuation arithmetic
MUST subtract those owners from the BF16 training target.

#### Scenario: BF16 and fp32 baselines diverge diagnostically

- **WHEN** BF16 contains the same protected G owners as the manifest plus an H
  owner that is absent from fp32 Source, and both surfaces have valid canonical
  matcher receipts and strict identity fields
- **THEN** admission succeeds, the divergence receipt is marked
  `diagnostic_only`, and the extra H is not counted as a BF16 post-update gain

#### Scenario: Missing protected BF16 G fails before K16

- **WHEN** the BF16-native Source baseline misses a protected manifest G owner
- **THEN** the run emits a typed non-admission receipt before sampling, replay,
  backward, or private proposal creation

#### Scenario: fp32 Source identity drifts

- **WHEN** fp32 Source and proposal/audit inputs differ in checkpoint, adapter,
  tokenizer, prompt, image, manifest, or declared processor identity
- **THEN** the paired audit fails closed even if BF16 admission is otherwise
  valid

#### Scenario: BF16-native witness and compiler inputs are used

- **WHEN** the BF16 Source projection is admitted
- **THEN** selected witness tokens are greedy on BF16, the compiler boundary and
  preservation bank are bound to the same BF16 model/session, and compiler
  remaining-owner state excludes H owners already present in BF16 Source

#### Scenario: Cross-surface token differences remain evidence

- **WHEN** the old `615`/`616` coordinate difference or larger/non-coordinate
  differences appear between surfaces
- **THEN** the exact token/row/owner/bbox evidence is published as
  `diagnostic_only`; no coordinate tolerance is applied and no BF16 internal
  parity rule is widened

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

### Requirement: Post-prepare AdamW runtime ownership admission

Before the first K16 sampling call, the live BF16 training boundary SHALL
publish a content-addressed ownership receipt for exactly one
`AcceleratedOptimizer` wrapper around exactly one `torch.optim.AdamW` base.
The receipt SHALL bind the scheduler to that same base, parameter order and
object identity, frozen group hyperparameters, empty wrapper/base state,
world-one BF16 sync-neutral accelerator semantics, zero runtime and scheduler
counters, and CUDA RNG capture capability.  Nested or foreign wrappers,
subclasses, foreign schedulers, stale state/gradients/counters, and device or
dtype drift SHALL fail closed.  The runtime wrapper SHALL remain the execution
handle, while proposal capture and `TrainingStateTransaction` SHALL bind the
exact base AdamW without scheduler or runtime-counter advancement.

#### Scenario: Accelerate wrapper and base are admitted

- **WHEN** a real post-prepare world-one BF16 runtime has an empty fresh base
  AdamW and a scheduler bound to that base
- **THEN** the receipt is content-addressed, the transaction binds the base,
  and the runtime wrapper remains available only as the execution handle

#### Scenario: Deterministic ownership drift occurs before acquisition

- **WHEN** the wrapper/base type, scheduler identity, parameter order, state,
  hyperparameters, counters, RNG capability, device, dtype, scaler, or sync
  semantics drift before the first sample group
- **THEN** the entry emits a typed admission failure with zero K16 sampling,
  replay, backward, or optimizer actions

#### Scenario: Ownership is revalidated after K16

- **WHEN** a deterministic runtime field drifts after the four K16 groups but
  before objective materialization
- **THEN** the proposal fails before backward and no objective component is
  silently substituted or dropped

### Requirement: Pre-acquisition ownership receipt is phase-bound

The production entry MUST validate the exact post-prepare Accelerate/AdamW
ownership, frozen cosine-with-warmup scheduler semantics, and live optimizer
group hyperparameters before the first sample. The backend MUST expose the
pre-acquisition hook and return a content-addressed ownership receipt. The
service MUST persist that receipt digest in an immutable pre-acquisition phase
before K16 acquisition.

#### Scenario: Missing hook or receipt fails before acquisition

- **WHEN** a production backend omits the pre-acquisition hook or returns a
  receipt without a 64-character content hash
- **THEN** the entry emits a typed admission failure, records no sample/replay
  call, and does not mask the primary ownership error with persistence errors

#### Scenario: Scheduler and group semantics are exact

- **WHEN** the scheduler is a same-base non-cosine scheduler, or a live
  optimizer group changes `betas`/`eps` while defaults remain unchanged
- **THEN** ownership admission rejects before acquisition and the receipt hash
  cannot remain valid
