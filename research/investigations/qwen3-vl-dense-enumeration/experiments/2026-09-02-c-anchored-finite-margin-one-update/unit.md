---
title: C-Anchored Finite-Margin One-Update Successor
description: Apply the same first AdamW proposal after finite margin-budget admission, then check every protected row token and cold natural behavior.
status: complete; mechanically valid; SCIENTIFIC_STOP_C_ANCHORED_FINITE_MARGIN_ONE_UPDATE; adapter discarded
---

# C-Anchored Finite-Margin One-Update Successor

Executed outcome: [results](results.md).

## Frozen question

> Does the exact first AdamW step improve the eleven-image C-prefix objective
> while keeping every token of all three protected C rows on the positive side,
> and does its unmerged cold decode yield target uptake without protected-owner
> loss?

The predecessor [one-update unit](../2026-09-02-c-anchored-one-update-vertical/results.md)
validly stopped because two directional margin changes were negative.  It also
showed that the corresponding affine margins remain `0.086378` and `0.036314`.
This successor changes only the preservation admission rule.  It does not
reinterpret the predecessor as passed.

## Fixed model, support, and optimizer

Reuse the exact predecessor plan (SHA-256
`f13ca3d2744ddfa5ae0db5748999043c356621995856d26f512f0bb42e956743`):
eleven one-per-image complete-row positives and the same three C incumbents.
Use the same universal base, unmerged C adapter, frozen selected-token embedding
delta, FP32 graph, equal-image field-balanced positive loss, and one fresh
AdamW step:

- learning rate `1e-5`, betas `(0.9, 0.999)`, epsilon `1e-8`;
- weight decay `0`, global gradient-norm cap `1.0`;
- exactly the existing 588 language-DoRA tensors are trainable;
- no auxiliary loss, projection, optimizer switch, scheduler dose, merge, or
  hyperparameter search.

## Finite-margin admission and exact check

For proposal `d`, admit the step only if `q^T d > 0` and every registered
first-order affine margin satisfies

\[
m_i + a_i^T d > \varepsilon_m,\qquad \varepsilon_m=10^{-3}.
\]

The buffer is an FP32 fail-closed tolerance, not a curvature theorem.  If the
proposal is admitted, apply exactly that AdamW step once.  Then freshly score:

1. the eleven complete-row losses, requiring a lower equal-image mean;
2. all token positions of each protected complete row, each against its new
   highest competing vocabulary token, requiring the minimum margin `>1e-3`.

Checking only the three original competitors is insufficient because the
highest competitor can change after the update.  Failure stops before any
adapter is retained.  Projection becomes a candidate only after such a failure;
it is not part of this unit.

## Cold natural behavior

On finite success, write only a standard PEFT DoRA payload.  In a new
`src.infer` process, require exact saved-to-materialized equality for all 588
adapter tensors, zero FP32 dtype casts, `merged_adapters=[]`, the frozen
embedding delta, and the exact greedy policy.  Decode the same 12-image panel:
the eleven gain images plus incumbent-only image `101636`.

At category-consistent global one-to-one IoU50, report the eleven target owner
uptakes, `3/3` protected retention, and all annotated-owner gains/losses.
IoU60/80, macro coverage, duplicates, invalid/dropped output, natural ordering
violations, and EOS/cap are monitors.  Unmatched predictions remain unknown.

## Decision and stop

- Identity, plan, trainable-surface, proposal/realized-step, adapter-only save,
  cold-load value, config, baseline, or panel mismatch is mechanical invalidity.
- Failed affine admission, nondecreasing gain loss, or any post-step all-token
  margin `<=1e-3` is
  `SCIENTIFIC_STOP_C_ANCHORED_FINITE_MARGIN_ONE_UPDATE`.
- A protected-owner loss or invalid/capped cold decode is
  `SCIENTIFIC_STOP_C_ANCHORED_FINITE_MARGIN_BEHAVIOR`.
- Target uptake `>=1`, protected retention `3/3`, and nondecreasing total panel
  IoU50 owner coverage gives `GO_C_ANCHORED_DDP_MULTISTEP`.
- Finite success and retention without that behavioral signal gives
  `HOLD_DDP_C_ANCHORED_BOUNDED_MULTISTEP` and permits only a separately frozen
  small multi-step bridge.

Run one authoritative arm from C.  There is no GPU sentinel, retry, projection,
learning-rate change, or optimizer comparison in this unit.

## Claim boundary

Success is selected training/effect-panel one-step feasibility only.  It does
not establish held-out generalization, COCO-scale improvement, DDP equivalence,
missing-label precision, or production readiness.
