---
title: Repeated First-Differing-Slot Full-Coordinate-Logit Panel
description: Three-case panel separating broad repeat-stable batch-shape shifts from local near-tie coordinate jitter at the exact first differing generated slot.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized_separately
unit_id: 2026-07-15-repeated-first-differing-slot-full-coordinate-logit-panel
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-07-15
---

# Repeated First-Differing-Slot Full-Coordinate-Logit Panel

## Question

Do the three one-to-two-pixel Brain Floating Point 16-bit (`bfloat16`)
batch-shape divergences from the preceding selected-transition screen arise
from broad, repeat-stable changes in the full coordinate distribution, or from
local near ties inside one physical-object coordinate neighborhood?

This is the final discriminator before either entering language-layer
localization or closing the current numerical branch. It does not estimate
dataset prevalence, detection quality, or the cause of dense low recall.

## Motivation and Competing Explanations

The predecessor image-`7574` partial-row recipient showed a broad white-bowl to
orange-bowl shift under `bfloat16` physical batch shape:

- centered coordinate-logit root-mean-square difference: `1.166623`;
- centered root-mean-square difference outside the two object windows:
  `1.139275`; and
- Jensen-Shannon divergence: `0.117722`.

The later six-case screen found exact first-action batch differences on images
`8629`, `13659`, and `17714`, but all three retained the same description and
physical instance and changed only one or two source-image pixels.

Two explanations remain:

1. **Broad repeat-stable distribution shift**: physical batch shape changes a
   substantial part of the coordinate distribution, while the selected token
   happens to remain near the same object. This would justify locating where
   the downstream model amplifies the batch-dependent state.
2. **Local near-tie jitter**: the distribution remains materially the same and
   only neighboring coordinate bins exchange rank. This would make layer
   localization scientifically disproportionate and close the numeric branch.

Same-layout nondeterminism is a third possibility and is measured rather than
silently attributed to physical batch shape.

## Frozen Cases and Recipient Construction

The source of record is the conclusion-owning `bfloat16` receipt from the
[Selected-Transition Batch-Precision Prevalence
Screen](../2026-07-15-selected-transition-batch-precision-prevalence-screen/results.md).

| Common Objects in Context image identifier | Zero-based first differing generated-token index | Source single-recipient object | Frozen local-window center |
|---:|---:|---|---:|
| `8629` | `6` | pizza | source single-recipient coordinate bin at index `6` |
| `13659` | `4` | person | source single-recipient coordinate bin at index `4` |
| `17714` | `4` | cup | source single-recipient coordinate bin at index `4` |

For each case, the runner reconstructs the exact source prompt plus first
complete source row. It then derives the maximal generated-token prefix shared
by the single-recipient and homogeneous-four-copy source executions. The
recipient ends immediately before the first differing coordinate token.

The frozen **Local Coordinate Neighborhood** is the inclusive coordinate-bin
window of radius `16` around the source single-recipient selected bin. The
window is derived from the immutable predecessor receipt before any new logits
are observed.

## Execution Arms

The two physical layouts are:

- **Single Recipient**: one fixed recipient in a physical batch of one;
- **Homogeneous Four-Copy Recipient**: four byte-identical recipients in one
  physical batch.

Each layout is executed twice in one process. Every homogeneous copy must have
the same complete coordinate-logit vector within its execution.

Two recipient paths are retained:

1. **Natural Cached Replay**: generate the frozen common prefix through the
   ordinary autoregressive cache and capture the raw logits at the first
   differing slot. This path owns reproduction of the original `bfloat16`
   phenomenon.
2. **Direct Full-Prefix Scoring**: append the exact common prefix and recompute
   the next-token logits in one forward pass. This path freezes the recipient
   token state and owns the exact-recipient precision comparison.

Natural cached replay is required to reach the frozen recipient under
`bfloat16`. Under Institute of Electrical and Electronics Engineers 754
32-bit floating point (`float32`), an earlier token may legitimately differ;
the direct full-prefix path remains the matched precision control.

## Preserved Evidence

For every case, dtype, path, layout, repeat, and batch position, preserve:

- all `1,000` raw coordinate logits serialized as `float32` values;
- all `1,000` coordinate-conditional log probabilities;
- a Secure Hash Algorithm 256-bit (`SHA-256`) digest of the raw vector;
- the top `20` coordinate bins and logits;
- the top-one minus top-two raw-logit margin;
- conditional probability mass in the frozen local neighborhood;
- the selected full-vocabulary token and coordinate bin;
- prompt and common-prefix hashes; and
- input tensor shapes and attention-mask row sums.

For each single-versus-homogeneous comparison, compute:

- centered maximum absolute coordinate-logit difference;
- centered root-mean-square coordinate-logit difference;
- centered root-mean-square difference outside the frozen local neighborhood;
- Jensen-Shannon divergence between coordinate-conditional distributions;
- local-neighborhood probability-mass and log-mass changes; and
- whether both selected coordinate bins remain in the frozen neighborhood.

The same metrics are computed between the two repeats of each layout to form a
same-layout noise floor.

## Decision Rule

The verified image-`7574` broad shift is the empirical scale anchor rather than
an assumed universal constant.

### Promote language-layer localization

Promote exactly one post-vision layer-onset unit only if at least one case:

1. reproduces its original `bfloat16` selected-token split on Natural Cached
   Replay in both repeats;
2. has between-layout shift at least ten times its same-layout repeat noise;
3. reaches at least one quarter of the image-`7574` anchor on at least two of:
   centered root-mean-square difference, outside-neighborhood centered
   root-mean-square difference, and Jensen-Shannon divergence; and
4. either moves the selected coordinate outside the frozen neighborhood or
   changes local-neighborhood conditional mass by at least `0.05`.

Matched direct `float32` must attenuate the between-layout shift by at least one
order of magnitude before precision is named as the conditioning factor.

### Close the numeric branch

Close the current numeric branch if all three cases:

1. keep both batch layouts inside the frozen neighborhood;
2. stay below one tenth of the image-`7574` anchor on all three distribution
   metrics;
3. change frozen-neighborhood conditional mass by at most `0.02`; and
4. have no larger same-layout repeat instability.

Return to the dense-enumeration mechanism queue without a layer sweep.

### Inconclusive

Any intermediate pattern remains bounded and does not authorize training,
architecture promotion, or an open-ended layer scan. Select at most one new
discriminator from the observed failure surface.

## Minimal Implementation Route

Use one experiment-local runner. Reuse the current model assembly, source-row
extraction, prompt reconstruction, batched tensor collation, natural cached
logit capture, direct full-prefix scoring, and coordinate-token constants.

Do not modify the shared inference backend, model forward pass, training
configuration, decoding policy, or artifact schema. Do not add a general probe
framework before this panel demonstrates a reusable seam.

The first real smoke is image `8629` under `bfloat16` with two repeats. The
remaining two cases and `float32` control run only after source split
reproduction, homogeneous-copy equality, and complete-vector serialization
pass.

## Stop Rule

Stop after the three-case `bfloat16` panel and matched three-case `float32`
direct control are classified. Do not start a layer sweep, Graphics Processing
Unit (`GPU`) training, an architecture prototype, or a production-precision
benchmark inside this unit.

## Artifact Handle

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-15-repeated-first-differing-slot-full-coordinate-logit-panel/<run-id>/
```
