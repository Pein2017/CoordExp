---
title: C-Anchored AdamW Backtracking
description: Choose the largest power-of-two dose of the same first AdamW direction that gives Armijo gain and exact all-token incumbent preservation, then cold-decode the selected adapter.
status: complete; see results.md
---

# C-Anchored AdamW Backtracking

## Frozen question

> Does a non-negligible dose of the already-positive eleven-image AdamW
> direction decrease the shared gain objective while preserving every token of
> the three C incumbent rows, and does that accepted unmerged adapter change
> natural owner coverage?

The predecessor full-dose result is immutable: all eleven target losses
decrease, but one protected `handbag` description token flips toward `umbrella`.
This unit changes only the dose-selection rule.  It neither rescues nor relabels
that result.

## Identities and direction

Reuse the same universal base, C adapter, selected-token embedding delta,
eleven complete-row events, three incumbent rows, FP32 graph, field-balanced
equal-image objective, and fresh AdamW definition.  Bind the exact predecessor
receipt (SHA-256
`44ce1c454601e73d5e7427fc35f497ae64bc7c96811defd53ccefa3287be40fc`)
and require the recomputed full-dose proposal hash
`12628cf7269c3e51b32b746ae1708b97d091b52f9b7dcb97baf7aa9bf9039a76`.

The base optimizer has learning rate `1e-5`, betas `(0.9,0.999)`, epsilon
`1e-8`, weight decay `0`, and global gradient-norm cap `1.0`.  No target,
prefix, loss, optimizer family, or parameter surface is changed.

## Deterministic backtracking

Full dose `alpha=1` is not rerun; the bound predecessor already rejects it.
From the unchanged C anchor, try in order

\[
\alpha\in\{1/2,1/4,1/8,1/16,1/32\}.
\]

Each candidate is one actual fresh `torch.optim.AdamW` step with effective
learning rate `alpha * 1e-5`, followed by exact finite forwards.  Reject and
restore C before trying the next alpha.  Accept the first candidate satisfying
both:

\[
L(\theta+\alpha d)\le L(\theta)-10^{-4}\alpha q^Td,
\]

and minimum selected-versus-new-best-other margin `>1e-3` at every token
position of every protected complete row.  The first passing alpha is the
largest passing registered dose; this is an optimizer rule, not a
best-result sweep.  Persist only that state.  If all five fail, stop without an
adapter.

### Local existence statement

If the finite set of protected margins and the gain objective are continuous
and differentiable near C, all anchor margins exceed `1e-3`, and `q^Td>0`,
then sufficiently small positive alpha preserves the margins and satisfies an
Armijo condition.  Therefore ideal halving terminates locally.  The finite grid
tests the stronger practical question: whether it terminates before dose
`1/32`.  This is not a global convergence, curvature, or generalization
theorem.

## Cold behavior gate

Save only the accepted PEFT DoRA payload.  A fresh `src.infer` process must
prove all 588 saved/materialized tensor values equal with zero FP32 casts,
`merged_adapters=[]`, and the frozen embedding delta and greedy policy.  Decode
the same twelve images through inode-identical hardlinks and use
category-consistent global one-to-one IoU50.

Report target uptake among the eleven selected missed owners, protected
retention `3/3`, and total annotated-owner gains/losses.  IoU60/80, image macro,
prediction/duplicate/drop/invalid counts, natural ordering violations, and EOS
are monitors.  Unmatched predictions are unknown.

## Decision and stop

- Identity, predecessor, proposal-hash, restoration, optimizer, trainable
  surface, save/readback, baseline, config, panel, or decode mismatch is
  mechanical invalidity.
- No passing alpha through `1/32` is
  `SCIENTIFIC_STOP_C_ANCHORED_ADAMW_BACKTRACKING`.
- A protected-owner loss or invalid/capped cold decode is
  `SCIENTIFIC_STOP_C_ANCHORED_BACKTRACKED_BEHAVIOR`.
- Target uptake `>=1`, protected retention `3/3`, and nondecreasing total panel
  IoU50 coverage gives `GO_C_ANCHORED_DDP_MULTISTEP`.
- A safe finite adapter without that immediate behavior gives
  `HOLD_DDP_C_ANCHORED_BOUNDED_MULTISTEP`.

One authoritative line search is run; there is no GPU sentinel, projection,
objective change, extra alpha, or optimizer comparison.

## Claim boundary

Success would establish a safe nonzero first shared step and selected-panel
cold behavior only.  It would not establish held-out generalization, DDP
equivalence, COCO-scale benefit, missing-label precision, or production
readiness.
