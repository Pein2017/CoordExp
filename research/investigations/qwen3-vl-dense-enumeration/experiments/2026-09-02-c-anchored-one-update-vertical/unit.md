---
title: C-Anchored One-Update Behavior Vertical
description: One real shared AdamW update on eleven C-prefix positive rows, followed by unmerged cold natural readback on the minimal twelve-image panel.
status: complete; mechanically valid; SCIENTIFIC_STOP_C_ANCHORED_ONE_UPDATE_FINITE; zero updates
---

# C-Anchored One-Update Behavior Vertical

Executed outcome: [results](results.md).

## Question and contrast

> Does the finite first AdamW update from the exact C model preserve the three
> surviving behavior-linked incumbents and produce an observable natural-policy
> signal on the eleven evidence-linked missed owners?

The paired contrast is the frozen C natural decode versus `C + one update`.
Both sides use the same universal base, selected-token embedding delta, prompt,
decoder, and one shared unmerged language-DoRA adapter.  The update side changes
only the 588 existing DoRA tensors.  It adds no output residual, per-image
payload, routing, merge, or external memory.

The strongest alternative is that the identity-metric gain direction was
misleading: AdamW's coordinate-wise preconditioning may threaten an incumbent,
or a safe same-prefix likelihood improvement may remain below an
autoregressive branch boundary and cause no natural owner uptake.

## Frozen support and update

Reuse, without reselection, the immutable predecessor plan
`2026-09-02-c-anchored-small-dual-feasibility/plan-v1.json` (SHA-256
`f13ca3d2744ddfa5ae0db5748999043c356621995856d26f512f0bb42e956743`).
It contains eleven complete-row, one-per-image gain events and three exact C
incumbent rows.  The gain loss is the equal-image mean of the existing
field-balanced positive-path loss.  There is no EOS, legality, KL, negative-row,
or preservation auxiliary.

Apply exactly one fresh AdamW step:

- learning rate `1e-5`, betas `(0.9, 0.999)`, epsilon `1e-8`;
- weight decay `0`, global gradient-norm cap `1.0`;
- no warmup and no effective scheduler operation before the sole step;
- deterministic FP32 HF graph at the exact C prefixes;
- optimizer state starts empty and only the 588 DoRA tensors enter it.

Before mutating parameters, materialize the actual first-step AdamW proposal
after clipping and measure in FP64:

\[
q^T\Delta\theta,\qquad a_i^T\Delta\theta\quad(i=1,2,3),
\]

where `q` is the negative gain-loss gradient and each `a_i` is the registered
incumbent-margin gradient.  If any value is nonpositive, stop before the step;
do not rescue it with a projection, another optimizer, learning rate, or dose.
After the step, recompute the realized delta, eleven losses, and three exact
decision margins.  The finite local gate requires lower mean gain loss and all
three margins positive and strictly increased.

Save only a standard PEFT adapter payload.  The base and embedding delta remain
external and immutable.  A merged checkpoint is forbidden.

## Natural behavior panel

Cold-load the saved adapter in a new `src.infer` process and decode the twelve
unique images formed by the eleven gain images plus incumbent-only image
`101636`.  Use the frozen C policy: HF FP32, natural greedy, repetition penalty
`1`, temperature `0`, and `max_new_tokens=3084` with natural `im_end`.
The cold-process manifest must prove value equality between all 588 saved and
materialized adapter tensors (with zero dtype casts in this FP32 arm), not just
matching keys or adapter status.  The reducer binds the predecessor plan, exact
C baseline, panel JSONL, and leaf inference-config hashes.

The primary readout uses the existing category-consistent global one-to-one
matcher at IoU50:

- target uptake: how many of the eleven frozen missed owner references become
  matched;
- incumbent retention: how many of the three protected owner references remain
  matched;
- total annotated-owner gain/loss on the same twelve images.

IoU60/80, image-macro coverage, prediction count, invalid/dropped rows,
duplicates, natural ordering violations, and termination are monitors.  An
unmatched prediction is unknown, not a negative label.

## Decision and stop

- Any identity, prefix, trainable-surface, optimizer-state, adapter-only save,
  unmerged cold-load, panel, or decode-policy mismatch is mechanical invalidity.
- A nonpositive proposed or realized gain/constraint dot product, nondecreasing
  finite gain loss, or nonpositive/decreased protected margin is
  `SCIENTIFIC_STOP_C_ANCHORED_ONE_UPDATE_FINITE`.
- If cold decode retains `3/3` protected owners, has no invalid/capped row,
  gains at least one target owner, and does not reduce total panel IoU50 owner
  coverage, the result is `GO_C_ANCHORED_DDP_MULTISTEP`.
- If the finite gate and retention pass but no immediate target flip (or a
  compensating panel loss) occurs, the result is
  `HOLD_DDP_C_ANCHORED_BOUNDED_MULTISTEP`: only a separately frozen small
  multi-step bridge is authorized.
- A protected-owner loss or invalid/capped decode is
  `SCIENTIFIC_STOP_C_ANCHORED_ONE_UPDATE_BEHAVIOR`.

The CPU AdamW self-check plus the already completed C graph/VJP predecessor are
the mechanics gate; a duplicate GPU sentinel is deliberately omitted.  The
authoritative arm runs once from the unchanged C anchor.  Only a pre-update
mechanical failure may be repaired without changing the frozen semantics;
there is no scientific retry or hyperparameter search.

## Claim boundary

Success would show finite shared-DoRA uptake on this selected training/effect
panel, not held-out generalization, COCO-scale quality, missing-label precision,
DDP equivalence, or production readiness.  DDP and broader data are downstream
only after this behavior bridge gives the corresponding authorization.
