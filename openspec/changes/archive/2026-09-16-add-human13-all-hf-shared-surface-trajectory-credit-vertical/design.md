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

### 5. Separate the training policy from the behavioral audit surface

Sampling, replay, compiler gradient, preservation Jacobians, and AdamW run on
GPU 0 through the shared BF16/FA2 session.  The BF16 session freezes a
canonical Source decode independently at each RP, derives its owner rows and
compiler Source boundary, freezes BF16-native WitnessMeasurement/Jacobians,
and uses the same BF16 surface for the post-apply margin probe.  Source and
proposal clean-greedy audits use the established HF fp32/SDPA batch-one
evaluator on GPU 1 when two cards are available.  The fp32 Source baseline is
frozen before the update at RP 1.0 and RP 1.10, and proposal outcomes are
computed only as Source-versus-proposal changes within that fp32 surface.
BF16/FA2 is the sole authority for the scientific objective and update path;
the audit model is not score-function evidence and owns only a stable
owner-level behavioral readout.  Its output need not be numerically or
token-identical to the training surface.

#### Independent baselines and diagnostic cross-surface divergence

The admission choke point has one constructor/loader/checker/publisher path
for two independent canonical baselines.  It parses each output with the
frozen canonical parser and applies the cardinality-first one-to-one owner
matcher on each surface separately.  BF16 admission requires every frozen
manifest G owner needed for preservation to be present in the BF16 Source
baseline and requires the BF16-native witness/compiler inputs to be valid.
The fp32/SDPA Source baseline must be durable at both audit repetition
penalties before any private update.  Cross-surface differences are published
as a `diagnostic_only` divergence receipt; they never gate the BF16 proposal.

The divergence receipt retains the old coordinate-alias evidence format where
available (token positions, coordinate roles, bins/deltas, boxes, owners, IoUs
and disposition), plus complete token counts, owner sets/maps, membership,
protected-G sets, and hashes of the full prediction payloads.  It explicitly
records that the surfaces have different policies.  A difference such as the
observed BF16 extra H owner is therefore scientifically visible without being
misclassified as a quantization alias.  Strict identity checks still fail
closed for model/checkpoint/adapter/embedding/tokenizer/prompt/image/manifest
or declared processor-policy drift.  This diagnostic path never widens
BF16/FA2 sampler-to-replay history, chosen-token, shape, or processed-
log-probability parity.

The runtime writes a private adapter checkpoint only for evaluation.  It does
not publish or promote it.  If a second GPU is unavailable, execution stays
pending rather than silently co-residing both models on one nearly full card;
a separately reviewed sequential unload/reload adapter may be added later.

### 6. Make one-image behavior—not another infrastructure matrix—the gate

For each audit RP `r`, define `G_r` as the fp32/SDPA Source clean-greedy
matched-owner set.  Define the BF16 training target as the manifest H set
minus owners already present in the BF16-native Source baseline.  A BF16
Source-covered H owner is trusted baseline context, not an H gain target.
Trajectory credit may still account for its first hits, but the continuation
gate does not claim a gain for it.  Proposal readout reports:

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

### 8. Admit the post-prepare AdamW ownership boundary before acquisition

The live BF16 assembly has two intentionally distinct optimizer handles after
`Accelerator.prepare`: the runtime keeps one exact
`accelerate.optimizer.AcceleratedOptimizer` execution wrapper, while proposal
capture and `TrainingStateTransaction` bind the wrapper's one exact inner
`torch.optim.AdamW`.  A content-addressed ownership receipt records the wrapper
and base object identities, scheduler-to-base identity, parameter order and
objects, frozen group hyperparameters, empty optimizer state, world-one BF16
and sync-neutral accelerator semantics, zero runtime/scheduler counters, and
CUDA RNG capture capability.  Nested/foreign wrappers, AdamW subclasses,
foreign schedulers, stale state/gradients/counters, and device/dtype drift fail
closed; no generic unwrapping is allowed.

The production entry validates this receipt, and the BF16 witness/probe owner,
after Source freeze but before the first K16 sample.  The same receipt is
revalidated after acquisition and before objective materialization/backward.
Private projected apply continues to use the exact base AdamW without
advancing the runtime wrapper or scheduler.  This is a production admission
correction for the previously observed Accelerate representation blocker, not
algorithm evidence and does not change the objective, learning rate, K16, or
dual-RP gate.

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
4. Run one no-update image-1584 preflight.  Freeze the BF16-native Source
   projection/witness/compiler boundary and the fp32/SDPA Source baselines;
   publish any cross-surface divergence as diagnostic-only evidence while
   retaining strict BF16/FA2 replay parity.  If these independent admissions
   pass, continue in the same reserved root to exactly one private complete
   update and dual-RP audit.
5. Publish the bounded one-image result and rollback evidence.  Run the
   13-image continuation only if the exact continuation gate passes.

Implementation rollback is deletion of successor-only scripts, configs,
tests, and active-change artifacts.  Runtime rollback is the existing complete
training-state transaction plus removal of private checkpoint bytes; Source
and predecessor artifacts remain immutable.

The ownership receipt is phase-bound: it also admits the exact frozen
cosine-with-warmup `LambdaLR` semantics and checks live param-group `betas` and
`eps` independently of optimizer defaults. The production backend must return
this content-addressed receipt from the pre-acquisition hook; the service
persists its digest before the first K16 sample and fails closed when the hook
or receipt is absent.

### 9. Canonical CUDA identity and truthful physical-boundary attempts

The post-prepare ownership receipt also carries one strict logical CUDA
identity. An indexless Accelerate `cuda` device may resolve to the current
logical index only under world-one, process/local-process zero,
`DistributedType.NO`, available CUDA, one explicitly indexed trainable device,
and a non-ambiguous `CUDA_VISIBLE_DEVICES` mapping. Explicit index conflicts,
multi-device trainables, unavailable/current-index drift, and ambiguous
visibility fail closed; generic string aliases are not accepted. The same
content-addressed identity is rebuilt after K16.

Physical model boundaries publish separate append-only `ActionAttemptReceipt`
records for training open, audit open, and audit evaluator/loader attempts.
Attempted, completed/admitted, and failed counts are distinct from admitted
session counters. A loader failure therefore records attempt=1, completed=0,
failed=1, bounded redacted stdout/stderr provenance, exception type/message
hash, and no close claim for a nonexistent handle. Proposal evaluators are
included in the same evaluator-attempt boundary. Terminal publication binds
the complete attempt tuple to the service phase ledger; a phase write failure
after a backend handle is returned closes that unpublished handle before the
caller receives an error. Terminal schema dispatch keeps historical v1
objects serializing with their original schema and content hash without
inferring new counters; new v2 terminals include the typed attempt list. This
is production admission/
telemetry evidence only and does not alter the scientific objective, surface
ownership, K16, learning rate, or outcome gate.

### 10. Attribute retained replay graphs at creation and adapter admission

One experiment-local typed helper now owns autograd-leaf traversal for the live
HF replay creator, the CUDA adapter, and the already-existing Task-3 CPU graph
check.  Its receipts are content-addressed and value-free: every reachable
leaf carries object id, concrete Tensor/Parameter type, exact registered model
name/status, requires-grad, device/dtype/shape, live model object id, and one
bounded input role.  Unknown leaves are evidence and rejection causes; they
are never filtered from the graph to make admission pass.

The HF session attests non-model forward inputs as non-trainable before the
first sample group and again at each forward boundary.  Replay creation binds
each retained log-probability tensor to a graph receipt.  The CUDA adapter
requires the exact receipt/tensor/model tuple and revalidates it across the
realized-margin probe, distinguishing foreign Parameter, unregistered
trainable input, detached/no-grad, wrong-model, stale/rebuilt-graph, and
receipt-mismatch failures.  The pre-acquisition phase persists the graph-input
receipt; K16/private-update phases bind replay receipt hashes or bounded typed
failure details, and terminal phase-ledger binding carries them without adding
fields to historical terminal schemas.

Foreign-leaf disposition and total count are computed over the complete
reachable leaf set before artifact detail is truncated.  The receipt retains
only the first bounded details, plus the full count and an explicit truncation
flag, so a late foreign Parameter cannot be hidden behind earlier input leaves.

One optional injected sentinel runs only on CPU after Source admission and
before acquisition.  Its count is separate from sample/replay/model-forward
counters.  Every model parameter and buffer must be on CPU, the callback
receives a constrained context without the live session, must return an exact
`admitted_model_graph` receipt, and is bracketed by model/session action
snapshots that reject attempted sampling, replay, forward, or mutation.  This
round exercises that sentinel only through production-shaped injection; it
authorizes no CUDA/model/K16 attempt and changes no objective, learning rate,
surface, gate, or Task-3 CPU contract.  A failed graph-evidence journal write
is attached as a note to the original typed adapter error and never replaces
that primary rejection.

## Closeout disposition (2026-08-24)

This design is retired rather than repaired further. The official live route
repeatedly reached Source audit and K16 acquisition/replay, but its immutable
attempts stopped before backward/update; for example,
`one-image-scientific-fresh-primary-5f3df478-v1` ended on typed replay graph
ownership admission with zero backward and optimizer steps. Later CPU graph
attribution work is diagnostic infrastructure evidence only and was not
followed by a new scientific attempt.

The executed K16 boundary is not erased by retirement. In
`one-image-scientific-fresh-primary-6aa30b7-v1`, the durable acquisition/replay
receipt records 463 sampling plus 463 replay forwards and binds compiler ledger
SHA-256
`55c7794cf64c7f5be909471fc1eb119f39ab05738622b10d87c07bc353f99072`.
Four canonical Source baselines were also durably admitted. This completes
Task 3.2's BF16-native compiler materialization contract, but the compiler
objective, backward, and update were not admitted. Task 4.3 remains incomplete
because no proposal audit or gained/lost output completed its compound
contract.

Tasks 4.3, 4.6, and 5.2--5.4 therefore remain unchecked. Task 5.5 is complete
only as a reviewed documentary closeout of this partially executed retired
route; it does not supply the missing scientific execution. Partial
subtasks 4.6a and 4.6b do not complete 4.6. The later standalone N=1, N=4,
and N=13 probes are separate simplified experiments; they establish that a
trajectory-credit plus current-owner preservation update can run, but they do
not test this design's complete compiler/projection algorithm. The N=13
[bounded result](../../../memories/notes/2026-08-24-human13-n13-k4-k8-factorial-result.md)
shows unstable owner churn and no dual-RP case for K8 over K4 under its exact
conditions. It supersedes further infrastructure repair as the program route
without converting this unexecuted design into algorithm evidence.
