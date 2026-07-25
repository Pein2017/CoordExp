---
title: Prefix-Local and On-Policy Final Owner-Set Training
description: Tests several non-exclusive set-aligned objectives for increasing one-completion clean-greedy unique physical-owner coverage without treating owner exchange as success.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized
unit_id: 2026-07-24-prefix-local-and-on-policy-owner-set-training
topic: qwen3-vl-dense-enumeration
status: transfer_evaluation_ready_for_user_discussion
evidence_status: directional_heldout_net_gain_not_robustly_established
updated: 2026-07-25
---

# Prefix-Local and On-Policy Final Owner-Set Training

## Authorization and Stop Boundary

The user authorized this unit for autonomous implementation, real-model
checks, short training smokes, and isolated unattended long training for every
arm that crosses the promotion gate below. Compute cost and equal floating-
point-operation budgets are not method-selection criteria.

The active task remains unchanged:

```text
image plus the original `list all objects` prompt
  -> one autoregressive completion in the existing row schema
  -> clean-greedy net gain in final unique physical owners
```

The unit stops when eligible long training is reliably running, when a
conclusion-bearing scientific or environment blocker prevents launch, or after
the launched round completes and its comparative evaluation is ready for user
discussion. It does not automatically promote a 256-image recipe, a final
architecture, or a successor research direction.

The materialization, implementation, short-smoke, completed long-training,
milestone, matched final-horizon, transfer, and runtime-correction receipts are
recorded in [results.md](results.md). The launched round, its 64-image
comparative evaluation, and the separately approved transition-step-36
development/held-out transfer gate are complete and ready for user discussion.
The held-out result is directionally positive but small and uncertain; full-
pool evaluation, architecture promotion, and a successor direction remain
outside the completed boundary.

## Question

Can supervision that directly distinguishes uncovered-owner progress from
confirmed non-progress, and/or optimizes the final trusted owner set on fresh
model rollouts, produce a clean-greedy net owner gain rather than the owner
exchange observed under positive complete-row imitation?

## Success Semantics

Three outcomes remain separate:

1. **Local actuation**: a selected owner or exact-prefix margin moves. This is a
   manipulation check, not success.
2. **Capability improvement**: the original one-prompt, one-completion clean-
   greedy policy gains net trusted unique owners over frozen Source or its
   matched control. Retention or safety defects may remain and must be reported.
3. **Usable improvement**: capability improvement also preserves ordinary
   Source owners, geometry, precision, row validity, and natural stopping.

An owner appearing while another ordinary owner disappears is owner exchange,
not capability improvement. Longer output, fewer stop tokens, lower training
loss, better exact-prefix scores, or better mean Average Precision alone do not
satisfy the gate.

## Evidence Inherited Without Reopening It

- The completed root-state admission census found only eight images satisfying
  its all-positive natural-alias strict composite, 76 images with at least one
  admissible strict edge, and 709 images with at least two eligible candidates.
  That result stops the exact strict design but does not make its composite a
  universal local-event predicate.
- Existing route and breadth screens show strong selected-owner uptake with
  final-set owner exchange and no held-out breadth advantage at fixed dose.
- Existing exact-prefix experiments show that complete rows and their order
  alter successor scores, while exact-prefix actuation can fail to transfer to
  free rollout.

The authoritative inherited result is
[the admission census](../2026-07-23-trajectory-owner-set-admission-census/results.md).
The stopped exact-design salvage branch remains owned by
[its unit](../2026-07-23-trajectory-owner-set-adjudication-salvage-gate/unit.md)
and must not be resumed for this experiment.

## Data and Split Boundary

Begin from the exact 2,004 Source-eligible training images and 34,068 root-state
Source-plus-sampled candidates materialized by the admission census at:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-23-trajectory-owner-set-admission-census/production-v1/
```

Recover row tokens, exact model-produced prefixes, owner reviews, and route
identity from the receipt-bound Source and sampled artifacts named by that
census. Development and held-out images never supply gradients. Targeted
train-only visual adjudication is allowed only for mechanically selected rows
whose resolution can create a conclusion-bearing event; unresolved rows remain
neutral until resolved.

The strict-edge images form a strong set-expansion stratum, and the eight
natural-alias images form a strong order-evidence subset. Neither is a universal
event-admission condition. Report results for these strata separately from the
broader row-local event pool.

## Row-Local Event Contract

One event binds:

- image identity and canonical original prompt;
- exact model-produced prefix tokens and their route provenance;
- every trusted physical owner covered before the action boundary;
- one trusted owner not covered by that prefix;
- one or more complete candidate-row aliases for that target owner;
- zero or more confirmed covered-owner duplicate rows;
- the natural stop action when it is premature under trusted evidence;
- zero or more parser-confirmed invalid complete actions;
- other valid uncovered-owner candidates and unresolved candidates as logged
  watch rows with no negative label; and
- Source checkpoint, tokenizer, row serialization, decode policy, and review
  provenance.

Admit an event only when ownership is resolved through the action boundary and
the target row is a trusted first occurrence relative to that prefix. A future
unknown or geometry-untrusted row does not censor an earlier complete event.
An unresolved row inside the prefix does censor the event because the covered
set is then unknown.

Do not require a matched remaining row or token budget. Used depth, remaining
context, row length, and stop availability are logged explanatory variables.
The runtime keeps a finite safety cap, but the cap is not a method-selection or
event-admission criterion.

Deduplicate exact `(image, prefix-token hash, target physical owner)` events.
Keep distinct target-row aliases inside one owner group, with a fixed alias cap
or explicit owner-normalized weights so discovery multiplicity cannot silently
become owner weight.

## Complete-Action Score

For a complete row or stop action `r` at image `x` and exact prefix `p`, use the
summed autoregressive log likelihood

```text
s(r | x, p) = sum_k log P(r_k | x, p, r_<k)
```

Do not replace this probability semantics with per-token mean likelihood.
Record row length and expose length sensitivity as a diagnostic. All finite
candidate aggregates below are audited approximations, not claims to enumerate
the full owner-level probability marginal.

## Compared Training Paradigms

### Local owner-conditioned candidate loss

For one uncovered target owner `o`, let `A_o` be its audited candidate-row
aliases and let `H` contain only confirmed covered-owner duplicates, premature
stop, and parser-confirmed invalid actions. Other valid uncovered owners and
unresolved candidates are absent from the labeled denominator.

```text
Q_o = sum_(r in A_o) w_r exp(s(r | x, p))
Q_H = sum_(h in H)   w_h exp(s(h | x, p))
L_local = -log(Q_o / (Q_o + Q_H))
```

This objective supplies no explicit negative label to another valid owner or
an unresolved row. Shared tokens, vocabulary softmax normalization, and shared
parameters can still move them indirectly; the zero-update influence check
must measure that effect before training.

### Exact-prefix pairwise loss

For one useful target-owner row `r+` and one confirmed harmful action `r-`
under exactly the same image and prefix:

```text
L_pair = softplus(s(r- | x, p) - s(r+ | x, p))
```

This is the singleton case of the local candidate loss and must use the same
complete-action scoring implementation. Pairwise training is an independent
local baseline. Non-aligned whole-trajectory ranking is not promoted by this
unit because different prefixes do not cancel and may suppress valid owner
exchanges for unrelated reasons.

The existing first-divergence transition objective remains a separate control.
It compares the first different token and then imitates the positive
continuation, avoiding the strong row-length effect of comparing a complete
row with one stop token. It must not be renamed as the complete-action
pairwise objective; both are measured on matched events where possible.

### Genuine on-policy final-set utility

Generate complete trajectories from the current trainable policy with the
original prompt and output contract. The primary estimator uses rollouts sampled
from the same parameters whose token log probabilities enter the update; refresh
rollouts after every accepted update or use an explicitly validated importance
ratio. Frozen-pool weighted cross-entropy is not represented as this estimator.

For an on-policy trajectory `tau`, define an observed coverage utility from its
trusted final physical-owner cardinality. A balanced variant may also penalize trusted
Source-owner loss, confirmed duplicates, parser-invalid rows, and geometry
failure. Unknown rows receive neither positive nor negative owner utility.

```text
L_on_policy = -(U(tau) - b) * sum_t log P(y_t | x, y_<t)
```

Use a frozen self-critical or leave-one-out baseline `b` that does not consume
held-out labels. Keep every sampled on-policy trajectory in the primary
estimator. Unknown rows contribute zero to the observed utility, but their
tokens can still inherit the trajectory advantage because they may affect
later state and stopping. Dropping unresolved trajectories would instead
optimize a resolved-only conditional objective and must not be called genuine
final-set expected utility. Any span-masked partial-credit variant must be
named heuristic and reported separately.

Generation and differentiable rescoring must bind the same trainable-weight
and adapter fingerprint, with no optimizer step between them. The primary arm
samples with temperature `1`, top-p `1`, and repetition penalty `1`; otherwise
the differentiated log probability must be that of the transformed sampling
policy rather than the raw model probability.

### Local plus on-policy hybrid

The hybrid is eligible only after both component implementations pass their
individual numerical and gradient checks:

```text
L_hybrid = L_on_policy + lambda_local * L_local
```

Source-preservation exposure is kept explicit rather than silently folded into
the name of either component.

## Controls

- frozen Source evaluation;
- ordinary complete-row cross-entropy on the same trusted positive rows and
  prefixes used by the local objectives;
- Source-preservation-only training using the existing trusted Source anchors;
- exact-prefix pairwise training as a local baseline; and
- for on-policy training, frozen-policy and self-critical return baselines.

Controls match the causal question inside a paradigm. They need not consume the
same floating-point operations, rollout count, or wall time across paradigms.
Report actual exposure, gradient-bearing tokens, updates, and rollout count
without using compute cost as an early rejection criterion.

## Zero-Update Gradient Gate

Before optimizer updates, compute real shared-parameter gradients on a bounded
panel and predict the first-order score change

```text
delta s_j = -eta * dot(gradient(s_j), gradient(L))
```

for the target owner, other valid uncovered owners, trusted Source owners,
confirmed duplicates, premature stop, invalid actions, and unresolved watch
rows. Compare local candidate, exact-prefix pairwise, and ordinary row cross-
entropy on the same event when possible.

Proceed only when target-versus-harm direction is correct and the candidate
objective does not show systematic Source/non-target collapse comparable to the
target gain. Report sensitivity to row length, alias count, target category,
prefix depth, and candidate inventory. A bad real-model gradient is a semantic
stop, not a reason to hide the event behind more training.

## Short Smoke and Long-Training Promotion

Every mathematically and mechanically valid arm receives a short real smoke.
The smoke must use the original one-prompt, one-completion clean-greedy policy
for its decision-bearing evaluation.

An arm is promising enough for isolated unattended long training when:

1. event supply is not confined to one image or only row zero;
2. the zero-update gradient gate passes;
3. a short smoke shows a coherent clean-greedy net trusted-owner gain over
   frozen Source or the appropriate matched control;
4. the gain is not explained only by longer output; and
5. duplicates, parser-invalid rows, or geometry failures do not show a burst
   large enough to invalidate the apparent gain.

Statistical significance is not required from a smoke. The individual image
and owner identities must nevertheless support one coherent interpretation.
Launch every arm that crosses this gate; do not select one because it is
cheaper. Use isolated output roots, retain useful dose checkpoints, verify that
graphics-processing units are idle before launch, and do not interfere with
unrelated jobs.

Before comparing grouped local and singleton pairwise arms, freeze exact-token
deduplication, the candidate inventory, and alias weights. The primary grouped
arm normalizes weights within each semantic owner or harmful-action group and
then normalizes events within image, so discovering more aliases does not
silently increase one owner's training weight. This aggregate is a weighted
candidate energy over the audited inventory, not a lower bound on the model's
true owner probability mass.

## Evaluation

Primary evaluation uses deterministic clean greedy decoding with the original
prompt, one autoregressive completion, canonical row schema, and repetition
penalty `1.0`. Report per image and aggregate:

- gained, retained, and lost trusted physical-owner identities;
- final trusted unique-owner count and net change;
- confirmed duplicates and duplication bursts;
- natural stop, premature stop, safety-cap stop, and output row count;
- parser-invalid and malformed rows;
- unresolved predictions without converting them to hallucinations;
- phrase, category, box geometry, and owner-binding failures; and
- exact-prefix manipulation checks as secondary evidence only.

Keep training images, non-gradient training images, development images, held-
out images, the eight natural-alias cases, and the broader strict-edge stratum
separate in every conclusion-bearing table.

## Artifact and Runtime Boundary

The experiment root is:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-24-prefix-local-and-on-policy-owner-set-training/
```

Reuse current StateBank, scoring, training, inference, checkpoint, and owner-
ledger infrastructure. Add only missing task-scoped seams. Long-running jobs use
isolated immutable inputs and separate output roots. Preserve exact model,
adapter, special-token delta, tokenizer, prompt, serialization, decode, split,
source-code, and command identity in receipts.

## Non-Goals and Held Branches

- The external one-row-at-a-time prompt, textual covered-set prompt, and
  controller-assisted generation branch remains `pending`.
- Do not move earlier assistant rows into the input prompt in this unit.
- Do not add an external detector, internal object slots, or a coverage memory.
- Do not resume the 248-additional-admission salvage sample.
- Do not treat arbitrary preservation or background images as if they supply
  the same set-expansion signal.
- Do not promote a final architecture or a full-dataset recipe from this round.

## Required Closeout

Record the materialized event supply, implementation and numerical receipts,
zero-update influence matrices, short-smoke outcomes, every long-run launch or
blocker, and the final clean-greedy gained/retained/lost ledger. Update the
experiment index, compass, and project memory only at the corresponding
meaningful boundaries.
