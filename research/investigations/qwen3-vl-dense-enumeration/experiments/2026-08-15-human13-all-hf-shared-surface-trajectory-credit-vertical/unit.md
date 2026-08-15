---
title: Human-13 All-HF Shared-Surface K-Trajectory Credit Vertical
description: A compute-heavy one-image successor that finally tests one complete K-trajectory-credit, greedy-compiler, and preservation update on one trainable HF surface.
type: investigation
role: research_unit
authority: non_normative_research
implementation_status: planned
unit_id: 2026-08-15-human13-all-hf-shared-surface-trajectory-credit-vertical
topic: qwen3-vl-dense-enumeration
status: planned
evidence_status: none
updated: 2026-08-15
---

# One-sentence question

If K16 sampling and gradient replay share one trainable HF BF16/FA2 model
surface, can one complete trajectory-credit + sparse greedy-compiler +
preservation update add at least one K-retrievable owner to image-1584 clean
greedy without losing any Source-visible owner under either RP audit?

## Authority and predecessor boundary

This unit owns the scientific question, cohort, algorithm, outcome, stop rule,
and claim boundary.  The implementation contract is the active OpenSpec change
[`add-human13-all-hf-shared-surface-trajectory-credit-vertical`](../../../../../openspec/changes/add-human13-all-hf-shared-surface-trajectory-credit-vertical/).
The Superpowers design and plan are execution aids and do not override either
owner.

The predecessor
[`2026-08-14-human13-k-trajectory-rp-crossover-screen`](../2026-08-14-human13-k-trajectory-rp-crossover-screen/results.md)
is immutable verified negative evidence for a different execution design:
vLLM fp32/TRITON sampling versus HF fp32/SDPA replay failed its frozen
exact-policy gate at RP 1.0.  It executed no update and says nothing about the
behavioral value of trajectory credit, the compiler, or preservation.  This
unit neither widens that gate nor resumes its unexecuted matrix.

The user has authorized this documentation and planning successor.  Model/GPU
execution and any material implementation begin only through a later explicit
apply/execution instruction.

## Current evidence and strongest alternative

The Human-13 substrate contains 392 trusted owners.  Historical Source clean
greedy covers `G=173`; historical K16 support adds `H=73`; `M=146` remained
K-miss under that recipe.  Earlier single-suffix and local on-policy proposals
often gained H while losing G, proving both that the language-only DoRA surface
can move recall and that owner exchange is a first-order failure mode.

The new algorithm has never been trained.  Its last unit stopped before
backward.  The strongest alternative explanation is therefore mechanical:
multi-trajectory credit may be useful, but the cross-engine numerical policy
made the score-function evidence inadmissible before the algorithm could be
tested.

The strongest algorithmic alternative is an explicitly approximate/off-policy
vLLM estimator.  That route may be more scalable, but it must expose its bias
and is deferred until this shared-HF vertical shows whether the complete
algorithm has any positive same-panel signal worth scaling.

## Frozen cohort and identities

- Base behavior: S step-2444 four-coordinate `geo_sorted_xy` Source with the
  sealed language-only rank-16 DoRA warm start and frozen selected-token
  embedding delta used by the predecessor.
- Manifest: the sealed Human-13 panel SHA-256
  `a8f88716c1227054ab29dc698f89462c9369c47c8d6415de3783c0937f60a6fb`.
- Initial case: image `1584` only.
- Training acquisition: K=16, four sequential logical groups of four, disjoint
  seeds `35001..35016`, RP `1.0`, temperature `0.4`, top-p `1.0`, no top-k,
  `max_new_tokens=512`.
- Stop: Qwen `<|im_end|>` only; cap termination remains visible harm.
- Audit: original-prompt clean HF greedy, batch one, under RP `1.0` and
  `1.10`, with the canonical parser, duplicate rule, and one-to-one matcher.
- K-miss owners M remain neutral and are not supervised.

Nothing in this unit may substitute the historical vLLM trajectories as
score-function evidence.  They remain support/provenance context only.

## Shared HF surface

Sampling and gradient replay use one live HF model object with:

- the exact same checkpoint, DoRA representation, selected-token embedding
  delta, tokenizer/processor, image tensor, prompt, and parameter state;
- BF16 compute and FlashAttention-2;
- model `eval()` mode for both sampling and grad-enabled replay;
- `use_cache=False` for sampling and replay; and
- repetition penalty applied to the full exact prompt-plus-generated history
  before temperature.

Sampling is deliberately inefficient: each active group of four repeats the
full image/prompt/history forward at every generated token.  It records every
request/history/token/RNG/shape and the generation-time processed chosen-token
log probability.

Backward remains feasible by replaying each completed four-trajectory group in
one padded, no-cache teacher-forced forward and gathering every sampled causal
position.  This retains one model object, dtype, attention implementation,
processor, and parameter state while vectorizing the gradient path.  The
stepwise-versus-vectorized physical-shape difference is visible and owned by
the parity gate; it is not called bitwise equality.

Admission requires exact request/history/chosen-token lineage, finite values,
maximum absolute processed-logprob error `<=0.02` nats, and K16 mean absolute
error `<=0.002` nats.  These are inherited semantic tolerances, not a target to
tune.  Failure produces zero optimizer steps.  It is an implementation HOLD
for this shared-surface seam, not evidence that the algorithm is bad.

## Complete algorithm

This unit tests the full C algorithm first instead of another large ablation.

### K-trajectory credit

All sixteen trajectories contribute through the predecessor's sealed pure
ledger:

- each trusted owner receives positive credit only on its first matched row;
- chronological class-agnostic pred-pred `IoU>0.95` duplicates, trusted
  repeats, invalid rows, non-M unmatched rows, and malformed equivalents carry
  their fixed costs;
- premature natural STOP and cap termination carry remaining trusted mass;
- STOP never receives positive exhaustiveness credit;
- legacy-M rows are neutral and masked from direct score-function pressure;
- row return-to-go uses K16 leave-one-out baselines; and
- the logical image/trajectory denominator is applied once across group
  boundaries.

The score-function numerator is therefore an owner-aware fusion of K
trajectories, not a best-suffix CE target and not a tied union reward.

### Sparse greedy compiler

At the sealed Source premature-STOP boundary, the compiler uses the frozen
metric-valid alias bank to compare normalized uncovered-valid mass with the
realized STOP/bad child.  It keeps `kappa=1`, margin `1e-4`, coefficient `1.0`,
RP-processed logits without temperature, and absent-site zero semantics.
Final free clean greedy—not the teacher-forced compiler margin—owns transfer.

### Proposal preservation

The complete trajectory-plus-compiler gradient feeds one fresh AdamW proposal:

- learning rate `3e-6`;
- betas `(0.9,0.999)`;
- epsilon `1e-8`;
- zero weight decay; and
- exactly one optimizer update.

The actual AdamW parameter delta is reconstructed and projected through the
predecessor's owner-wise Source witness constraints before private application.
All components are mandatory.  A missing compiler, witness, projection, or
finite delta stops; there is no CE, unprojected, or lower-dose fallback inside
the same run.  Gradient and delta norms are recorded but do not adapt LR before
the first behavioral result.

## Primary outcome and continuation gate

For audit RP `r`, let `G_r` be the Source clean-greedy matched-owner set.  Let
`H` be trusted owners hit by the new RP-1.0 K16 acquisition but absent from
`G_1.0`.  Report, by exact owner ID:

```text
H_gain_r     = |proposal_r intersect H minus G_r|
G_loss_r     = |G_r minus proposal_r|
net_unique_r = |proposal_r| - |G_r|.
```

Also report incidental M gains, duplicate rows, unmatched rows, malformed
rows, stop/cap, total rows, and generated tokens.  Gains and losses are never
collapsed into a single number without the two owner-ID lists.

The one-image experiment is complete once one admitted private update, both
audits, and exact rollback finish, even when the behavioral result is null or
negative.  It authorizes one 13-image shared-surface update only if all are
true:

1. RP 1.0 gains at least one H owner;
2. RP 1.0 has positive unique-owner delta;
3. RP 1.0 and RP 1.10 each lose zero members of their own G sets; and
4. neither audit increases high-IoU duplicates, emits malformed output, or hits
   the token cap.

This gate is intentionally stronger than positive net alone: owner exchange
cannot finance a claimed success.  Failure blocks width expansion but remains
decision-bearing evidence about the complete algorithm.

## Execution order

1. CPU-only contracts and dry run: shared identity, K16 groups, RP transform,
   replay mapping, objective normalization, one update, audit, artifact, and
   rollback receipts.
2. Production-shaped no-update vertical on image 1584: one shared BF16/FA2
   sampler/replay model on GPU 0 and the standard fp32/SDPA audit role reserved
   on a distinct GPU.
3. If parity passes unchanged, continue to one complete private update and
   dual-RP clean-greedy audit, then restore Source exactly.
4. Independently verify lineage, component/delta evidence, gained/lost owner
   arithmetic, private-byte cleanup, and rollback.
5. Run one 13-image update only if the exact one-image continuation gate passes.

No adaptive retry, LR ray, multiple update, accepted checkpoint, or nested
matrix belongs to this unit.

## Resource contract

The initial live vertical requires two distinct suitable GPUs:

- GPU 0: one BF16/FA2 Source training model, sampler/replay tensors, fresh
  AdamW state, gradients, and preservation work;
- GPU 1: the established HF fp32/SDPA batch-one Source/proposal clean-greedy
  audit lifecycle.

The dry run estimates prompt/image/token forwards, replay groups, backward
count, wall time, CUDA allocation/reservation, host RSS, and artifact bytes.
The run waits when two cards are unavailable; it does not silently co-reside
both models or downgrade the audit surface.  Eight cards are not required for
the one-image vertical.

## Stop rules

Stop before update on any surface identity, request/history/token, RP order,
finiteness, or `0.02/0.002` parity failure.  Do not change tolerance.

Stop before private apply on a missing/non-finite objective component, failed
AdamW reconstruction, uncertified preservation solve, or non-finite state.

Stop after the one-image result when the continuation conjunction fails.  Do
not search more seeds, tune LR from owner outcomes, or run the full panel.

Stop durably on any rollback mismatch; no later action is valid.

## Claim boundary

A passing one-image result would establish only that one compute-heavy
same-panel all-HF update can fuse sampled owner evidence and transfer at least
one owner into protected clean greedy.  It would not establish robustness,
population benefit, scalable rollout efficiency, production readiness, full
set coverage, K-miss learning, or superiority to CE/RL baselines.

A null or negative result would reject this exact one-update construction on
image 1584; it would not prove multi-trajectory credit impossible.  A parity
failure would be infrastructure/surface evidence only.
