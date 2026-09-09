---
title: C-Anchored Finite-Margin One-Update Results
description: The full AdamW step improves all eleven gain losses but flips an unregistered handbag description token toward umbrella, so the updated adapter is discarded.
status: complete; mechanically valid; SCIENTIFIC_STOP_C_ANCHORED_FINITE_MARGIN_ONE_UPDATE; adapter discarded
---

# C-Anchored Finite-Margin One-Update Results

## Verdict

`SCIENTIFIC_STOP_C_ANCHORED_FINITE_MARGIN_ONE_UPDATE` is mechanically valid.
The exact full AdamW step passed the three registered affine-margin checks and
improved every one of the eleven gain-event losses.  Fresh all-token readback
then found a different protected token whose margin crossed below zero.  The
updated in-memory model was discarded: no adapter, merged checkpoint, or
natural decode was produced.

This is positive multi-image optimization evidence plus a concrete
preservation failure.  It is not a natural-policy gain or a saved-model result.

## Immutable evidence

- authoritative receipt:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-02-c-anchored-finite-margin-one-update/authoritative-v1/receipt.json`,
  SHA-256
  `44ce1c454601e73d5e7427fc35f497ae64bc7c96811defd53ccefa3287be40fc`;
- log:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-02-c-anchored-finite-margin-one-update/authoritative-v1.log`,
  SHA-256
  `54c50b3cac7220ef3ee058eaf539d3408d98d2fd6c0ba4cf13352c9deabb602e`;
- runner commit: `4f6c6d62cb7ff629d6881ff44a579d7ed05a8d31`;
- predecessor plan SHA-256:
  `f13ca3d2744ddfa5ae0db5748999043c356621995856d26f512f0bb42e956743`.

The run retained the exact universal base, C adapter, step-2444 embedding
delta, FP32 graph, 588-tensor DoRA surface, eleven targets, three incumbents,
and first-step AdamW proposal from the predecessor.

## Shared gain result

The proposal reproduced `q^T d = +0.08304185`, norm `0.04236874`, and the same
three affine margins (`0.061677`, `0.086378`, `0.036314`).  The analytic and
realized parameter deltas agree within `3.73e-9` in L-infinity norm.

After the step, the equal-image gain loss fell from `1.731361` to `1.650908`
(`-0.080453`, or `-4.65%`).  All eleven individual losses decreased; their
changes range from `-0.03254` to `-0.15298`.  Thus the selected multi-image
objective has a real finite descent direction under shared DoRA and AdamW.

## Preservation failure

The three originally registered weakest decisions remained positive, and
their exact post-step margins closely followed the affine prediction:

| protected owner | original anchor | exact post-step |
|---|---:|---:|
| `101636:coco_ann:1760455` | 0.029783 | 0.061558 |
| `347671:coco_ann:1799145` | 0.086874 | 0.086630 |
| `359310:coco_ann:1172698` | 0.038918 | 0.036274 |

However, the last owner's full-row minimum moved to description-token offset
1 and became `-0.141983`.  At C, that position's selected-versus-best-other
margin was `+0.067036`.  The protected row describes `handbag` (tokens `hand`,
`bag`); the new winning competitor token decodes as `umb`.  The same image's
selected gain owner is an `umbrella`.

The observed token identities and crossing are exact.  Their interpretation
as intra-image handbag-to-umbrella substitution is a strong mechanism
hypothesis, not yet an isolated causal claim.  Regardless of cause, protecting
only the initially weakest token is falsified: a non-active token can become
the first finite-step failure.

The other two protected rows remain fully positive; the affected handbag row's
other nine token margins remain positive.  The failure is localized, not a
general parse/EOS or all-row collapse.

## Runtime and claim boundary

The run executed 28 forwards, 11 backwards, 3 VJPs, and one disposable
optimizer step in `34.54 s`, peaking at `26.56 GB` CUDA allocation.  The stop
was scientific, after exact finite evaluation.  Since no adapter was retained,
the cold value-equality and natural panel gates were correctly not run.

This establishes selected-panel shared-objective descent and one exact
category-token preservation failure.  It does not establish natural uptake,
held-out transfer, optimizer-state scalability, or COCO/DDP quality.

## Next discriminator

Projection and an objective change remain premature.  The smallest successor
is an anchored backtracking line search along the same first AdamW proposal:

1. deterministically try decreasing powers of two;
2. require Armijo decrease of the eleven-event mean loss;
3. require every token margin of all three protected rows to exceed `1e-3`;
4. accept the largest passing dose, save only its unmerged DoRA, and run the
   frozen twelve-image natural panel.

This is not an arbitrary learning-rate sweep: it is a transactional optimizer
rule with one monotone acceptance predicate.  Under local smoothness, strict
positive anchor margins and `q^T d>0` imply that sufficiently small dose exists
which preserves the finite anchors and decreases the objective.  The live
probe tests whether the useful dose is non-negligible.  Only if backtracking
requires a negligible dose or still loses an owner does projection or a
preservation term become decision-relevant.  DDP/COCO scaling remains held.
