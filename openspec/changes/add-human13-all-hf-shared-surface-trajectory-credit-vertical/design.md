## Context

See [proposal.md](proposal.md) for motivation and the
[predecessor result](../../../research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-14-human13-k-trajectory-rp-crossover-screen/results.md)
for the closed cross-engine route.  The reusable implementation already owns
the trajectory ledger, sparse compiler, AdamW reconstruction/preservation,
private checkpoint audit, rollback transaction, matcher, and Human-13 Source
assembly.  The missing seam is a score-function acquisition/replay owner that
uses the training model itself rather than a separately materialized vLLM
policy.

The selected panel is deliberately overfit-only.  Accuracy and reaching one
real algorithm update take precedence over rollout throughput.  The design
must still avoid a physically impossible per-token backward path: stochastic
sampling may repeat prefixes stepwise, while gradient replay reconstructs the
recorded active batch and causal history length for each sampler step, retaining
one graph for the later single proposal backward.

## Goals / Non-Goals

**Goals:**

- Reach one complete trajectory-credit + compiler + preservation proposal on
  image 1584 instead of spending another unit only on cross-engine admission.
- Bind sampling and gradient replay to the same live HF parameter object and
  training-feasible BF16/FA2 surface.
- Preserve exact K16 owner accounting, global loss normalization, dual-RP
  clean-greedy measurement, and exact rollback.
- Make a positive protected one-image result the sole authority for expanding
  to one 13-image update.
- Keep every new owner experiment-local and removable.

**Non-Goals:**

- Proving a mathematically exact unbiased policy gradient.
- Matching vLLM throughput, using vLLM data, or solving production rollout
  scaling in this change.
- A nested A/B/C matrix, LR ray, adaptive multi-update controller, accepted
  checkpoint, K-miss supervision, validation, or generalization claim.
- Changing generic training, HF inference, vLLM, checkpoint, or evaluator
  contracts.

## Decisions

### 1. Use one live BF16/FA2 model object for acquisition and backward

An experiment-local `HFSharedSurfaceSession` will own the loaded Source Qwen,
language-only DoRA adapter, frozen selected-token embedding delta, processor,
and exact trainable parameter layout.  The model remains in `eval()` mode for
both no-grad sampling and grad-enabled replay; `eval()` does not disable
autograd and avoids a train/eval policy change.  BF16 and FlashAttention-2 are
used because the same surface must remain capable of backward and AdamW on one
80-GB-class GPU.

The session exposes only typed operations:

```text
sample_group(image, seeds[4], policy) -> SampledHFGroup
replay_group(sampled_group) -> GradientReplayGroup
source_boundary(image, rp) -> CompilerBoundary
close() -> SharedSurfaceCloseReceipt
```

Every output binds one immutable `SharedSurfaceIdentity`; consumers never pass
an arbitrary model or tokenizer beside it.

**Alternative rejected:** all-HF fp32/SDPA.  It aligns with the historical
audit surface but leaves insufficient margin for training-state, optimizer,
gradient, and replay tensors and is unnecessary for the algorithm question.

**Alternative rejected:** reuse the vLLM collector and call the estimator
off-policy.  That is a legitimate future unit, but it does not answer whether
the algorithm works once the avoidable cross-engine mismatch is removed.

### 2. Spend sampling compute, but retain one bounded replay graph

Sampling runs four trajectories together, one token step at a time, with full
prompt/image/history forwards and `use_cache=False`.  Stopped rows leave the
active batch; the receipt records every active request, padding/mask shape,
history hash, raw chosen logit, processed chosen log probability, sampled
token, and RNG state transition.  Four such groups yield K16.

After trajectories finish, replay reconstructs every recorded sampler step for
the group: the same active request IDs, the same causal history length, and one
selected causal logit position per active request.  Each is a no-cache,
grad-enabled BF16/FA2 forward with the identical RP-then-temperature policy
transform.  Non-reentrant activation checkpointing retains only bounded inputs
for the later single proposal backward; position-selective logits avoid keeping
the full vocabulary sequence.  This exact step alignment is required because
the production FA2 surface is batch-shape and sequence-length dependent;
grouping steps by active membership or padding completed histories changes
accepted log-probabilities.  The observed image-1584 K16 run used 463 sampling
and 463 replay forwards, not four replay forwards.

The unchanged `0.02` maximum and `0.002` mean gate is retained as a semantic
admission limit, not as a claim of bitwise equality.  The receipt also reports
the full error distribution so a passing but noisy surface remains visible.

**Alternative rejected:** cached `generate()` plus full-sequence replay.  It
is faster but changes both cache semantics and physical attention shape, the
exact ambiguity this successor is intended to remove first.

**Alternative rejected:** grad-enabled stepwise replay with backward at every
generated token.  It best matches shape but multiplies full multimodal
backward cost by generated-token count; the implementation instead records all
checkpointed step graphs and performs the existing single proposal backward.

### 3. Keep the original sampled policy and owner ledger

The initial vertical uses image 1584, seeds `35001..35016`, K16, logical batch
four, training RP 1.0, temperature 0.4, top-p 1.0, no top-k, 512-token cap, and
Qwen `<|im_end|>` as the only stop.  The seed group is disjoint from every
predecessor qualification and matrix group.  RP is applied once to the full exact
prompt-plus-generated history before temperature.

The existing pure `TrajectoryCreditLedger` remains the only scientific label
owner.  It retains trusted first-hit credit, convex co-occurrence utility,
duplicate/repeat/unmatched/malformed costs, one-sided premature-STOP credit,
RLOO baselines, legacy-M masking, and the global `N*K` denominator.  The new
session supplies admitted processed log probabilities; it does not reimplement
matching or credit.

### 4. Run the complete C algorithm first

The first behavioral proposal is not another large ablation.  It composes:

```text
trajectory score-function numerator
  + 1.0 * sparse greedy-compiler numerator
  -> fresh AdamW actual delta at lr=3e-6
  -> existing owner-wise preservation projection
  -> one private applied proposal
```

Compiler `kappa=1`, margin `1e-4`, AdamW betas `(0.9,0.999)`, epsilon `1e-8`,
and zero weight decay remain unchanged.  No LR ray precedes the first result;
gradient and delta norms are recorded rather than used to adapt the dose.  A
missing component fails closed—there is no CE or unpreserved fallback hidden
inside the same arm.

This prioritizes the user's actual algorithm over attributing every component
before any full algorithm update exists.  If the complete arm produces a
positive protected result, a later change may compare trajectory-only and
compiler-only additions.

### 5. Separate the training surface from the behavioral audit surface

Sampling, replay, compiler gradient, preservation Jacobians, and AdamW run on
GPU 0 through the shared BF16/FA2 session.  Source and proposal clean-greedy
audits use the established HF fp32/SDPA batch-one evaluator on GPU 1 when two
cards are available.  This audit model is not score-function evidence and does
not need numerical parity with the training surface; it owns the stable owner
behavior readout.

The runtime writes a private adapter checkpoint only for evaluation.  It does
not publish or promote it.  If a second GPU is unavailable, execution stays
pending rather than silently co-residing both models on one nearly full card;
a separately reviewed sequential unload/reload adapter may be added later.

### 6. Make one-image behavior—not another infrastructure matrix—the gate

For each audit RP `r`, define `G_r` as the Source clean-greedy matched-owner
set.  Define `H` before the update from trusted owners hit by the new RP-1.0
K16 acquisition but absent from `G_1.0`.  Proposal readout reports:

```text
H_gain_r = |proposal_r intersect H minus G_r|
G_loss_r = |G_r minus proposal_r|
net_unique_r = |proposal_r| - |G_r|
```

It also reports M gains, duplicates, unmatched rows, malformed rows, stop/cap,
row count, and token count.  The update itself is a valid completed experiment
even when these outcomes are null or negative.

Only the following conjunction authorizes the 13-image continuation:

- RP 1.0 gains at least one H owner and has positive unique-owner delta;
- both audit RPs lose zero members of their own G sets; and
- neither audit increases high-IoU duplicates, emits malformed output, or hits
  the token cap.

Failure closes the full-panel branch but does not erase the one-update result.
This prevents parity from again becoming the only observed scientific event.

### 7. Preserve transactional and artifact lineage

The existing `TrainingStateTransaction` begins immediately before backward
and binds trainable parameters, optimizer, scheduler, counters, gradients,
CPU/CUDA RNG, and Source digest.  The runtime reuses the predecessor's private
checkpoint/audit/rollback lifecycle.  A run publishes immutable acquisition,
replay-parity, objective, projection, audit, rollback, resource, and terminal
receipts.  Failure receipts retain the last completed phase without promoting
proposal bytes.

The one-image entry defaults to dry-run and requires explicit model/GPU
authority.  The full-panel entry additionally requires the exact passing
one-image terminal hash.  There are no adaptive retries.

## Risks / Trade-offs

- **[No-cache HF sampling is slow]** → Limit the first execution to one image,
  K16, four groups of four; record image/prompt/token forwards and wall time.
- **[Stepwise sampling and vectorized replay still differ in physical shape]**
  → Keep one model object and kernel family, compare every chosen-token
  processed log probability, and stop before backward on the frozen gate.
- **[BF16/FA2 is nondeterministic at tiny scale]** → Treat the tolerance as
  numeric admission rather than bitwise identity and retain full error
  evidence; never tune it after observation.
- **[One complete C arm may hide which component helped]** → Record all three
  component numerators and preservation correction, but defer ablation until
  the complete algorithm shows positive behavior.
- **[Preservation witnesses can pass while owners are lost]** → Require
  zero G loss on both clean-greedy RP audits for width expansion.
- **[Two live GPUs may be temporarily unavailable]** → Wait for two cards;
  do not weaken the audit surface or add an unreviewed one-card lifecycle.
- **[Same-image success is memorization]** → Label it overfit-only and require
  a new decision before wider images, validation, or scalable rollout work.

## Migration Plan

1. Add pure typed shared-surface identities, sampling/replay plans, and CPU
   admission tests; keep all CLI paths dry-run.
2. Add the experiment-local HF sampler/replay owner and integrate it with the
   existing ledger/compiler/preservation runtime through injected fakes.
3. Add the one-image production-shaped entry, private dual-GPU audit lifecycle,
   immutable receipts, and a zero-action dry run.
4. Run one no-update image-1584 parity vertical.  If admitted, continue in the
   same reserved root to exactly one private complete update and dual-RP audit.
5. Publish the bounded one-image result and rollback evidence.  Run the
   13-image continuation only if the exact continuation gate passes.

Implementation rollback is deletion of successor-only scripts, configs,
tests, and active-change artifacts.  Runtime rollback is the existing complete
training-state transaction plus removal of private checkpoint bytes; Source
and predecessor artifacts remain immutable.
