---
title: Mixed Old-Prefix and Refreshed-Prefix Coordinate Correction
description: An equal-budget staged comparison that tests whether coordinate correction transfers better when its exact prefixes are refreshed from the treated model's current rollout distribution.
type: investigation
role: research-unit
authority: non_normative_research
unit_id: 2026-07-21-mixed-old-and-refreshed-prefix-coordinate-correction
topic: qwen3-vl-dense-enumeration
status: complete
updated: 2026-07-21
---

# Mixed Old-Prefix and Refreshed-Prefix Coordinate Correction

## Motivation

The 256-image coordinate-boundary screen showed a clean separation:

- the first-wrong-coordinate objective improved the requested float32 margin
  at frozen Source-checkpoint prefixes; but
- neither tested learning rate improved ordinary clean rollout.

This leaves two main explanations. The correction may be trained on stale
prefix states that the treated model no longer visits, or the language-tower
parameter update may be too entangled to preserve a local boundary correction
through a complete self-generated row. This unit tests the first explanation
before introducing a more local model intervention.

## Primary Hypothesis

If prefix-state distribution mismatch is the main blocker, then correction on
fresh prefixes produced by the current treated checkpoint should improve clean
self-rollout more than an equal-budget repeat on old Source prefixes.

## Compared Paths

Both paths begin from the lower-learning-rate Treated-v1 checkpoint produced by
the preceding screen.

```text
Treated-v1
  -> one common epoch on the old Source-prefix bank
  -> Shared Intermediate checkpoint

Shared Intermediate
  -> Old-Prefix Repeat Control on paired old events
  -> Control checkpoint

Shared Intermediate
  -> Refreshed-Prefix Treatment on paired newly visited events
  -> Refresh checkpoint
```

The common first epoch ensures that the final Refreshed-Prefix Treatment has
seen both old and refreshed trajectory states. The branch comparison isolates
the value of refreshed states: both branches start from the identical shared
checkpoint and consume the same number of paired images, events, optimizer
steps, learning rate, loss, and token-type stabilization.

## Data and Identity Contract

- The old bank remains bound to the Source checkpoint that generated its exact
  prefixes.
- Off-policy replay from Treated-v1 is explicit and records both the old
  trajectory-source identity and current training-warm-start identity.
- The shared intermediate checkpoint runs ordinary clean inference on the same
  256-image source cohort with repetition penalty 1.0.
- The refreshed bank is built from those actual shared-intermediate outputs and
  is bound to that checkpoint.
- Branch comparison uses only the intersection of image identities that yield
  one trusted old coordinate event and one trusted refreshed coordinate event.
- The old and refreshed branch banks contain exactly the same image identities
  and one event per image. No ambiguous, duplicate, unmatched, border-
  truncated, or otherwise untrusted owner receives gradient.
- All model-facing coordinates remain normalized integers in `[0, 999]`.

## Training Contract

- freeze vision tower and multimodal aligner;
- train only language-tower Weight-Decomposed Low-Rank Adaptation parameters;
- use first-wrong-coordinate preference plus the selected-site token-type gate;
- no canonical supervised-fine-tuning replay, Kullback-Leibler divergence,
  Gaussian coordinate target, external detector, new model head, or inference-
  time controller;
- use learning rate `3e-6`, gradient clipping 1.0, and one epoch per stage;
- use all eight available data-parallel ranks when the paired cohort permits a
  meaningful optimizer step; otherwise preserve equal optimizer budgets with
  the smallest valid rank count.

## Primary Measurements

1. Exact-prefix float32 paired replay on each branch's own training events.
2. Clean greedy rollout on the frozen 256-image input cohort.
3. Mean Average Precision, Average Precision at intersection over union 0.75,
   mean recall, prediction count, duplicate indicators, parser-invalid rows,
   dropped predictions, and truncation.
4. For each paired physical owner: selected-coordinate error, category
   retention at the intended row, best same-category full-box intersection over
   union, center error, and size error.

## Interpretation

- **Refresh beats Control in clean rollout:** supports prefix-state distribution
  mismatch and staged offline trajectory refresh as a useful treatment.
- **Both improve only exact-prefix margins:** stale prefixes are not the main
  blocker; the correction is too local or parameter updates are too entangled.
- **Refresh is less stable than Control:** current self-rollout states pollute
  the correction set; require stronger admission, old/new interleaving, or a
  more local parameter path before scaling.
- **Both improve clean rollout similarly:** additional optimization matters,
  but refreshed states are not specifically responsible.

No result from this unit alone authorizes 1,024-image or full-dataset training.
Promotion requires a clean-rollout gain without a material rise in invalid,
duplicate, or truncated output.

## Result

The equal-budget branch comparison is complete on an exact 192-image paired
cohort. Refreshing the complete correction-event package produced a matched-
package interaction:

- on old prefixes, Old-Prefix Repeat Control exceeded Refreshed-Prefix
  Treatment by `0.02637` mean coordinate margin;
- on refreshed prefixes, Refreshed-Prefix Treatment exceeded Old-Prefix Repeat
  Control by `0.02566`;
- the paired difference of differences was `0.05203`, with a bootstrap 95%
  interval of `[0.03273, 0.07498]`.

The full 192-image comparison does not isolate prefix state because the old and
refreshed banks can also select different owners, coordinate targets, and
candidate rows. In the 35-image subset that holds owner, coordinate axis,
acceptable coordinate set, and candidate row fixed while changing the prefix,
the interaction falls to `0.00261`, with bootstrap 95% interval
`[-0.000001, 0.00755]`. Prefix-state sensitivity therefore remains unresolved.

The refreshed event package does not establish a sufficient rollout treatment.
On the same 256-image clean-rollout cohort, refresh improved Average
Precision at intersection over union 0.75 by `0.00922` over the shared
intermediate and by `0.01045` over the old-prefix control, but the overall Mean
Average Precision gain over the shared intermediate was only `0.00070`.
Moreover, dropped predictions increased from `8` to `16`, and intended-row
category retention for the 192 refreshed-bank owners fell from `192` to `175`.
Best same-category owner intersection over union did not improve over the
shared intermediate.

The primary prefix-state hypothesis is therefore **not adjudicated**. The
broader refreshed-event treatment is slightly better than an equal-budget old-
event repeat, but the evidence cannot assign that difference specifically to
prefix refresh. This exact treatment is not promoted to 1,024 images or the
full dataset. The next treatment must protect row ownership and complete-box
behavior, not merely strengthen one selected coordinate boundary.

See [results.md](results.md) for the full comparison and artifact handles.

## Output Root

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-21-mixed-old-and-refreshed-prefix-coordinate-correction/`
