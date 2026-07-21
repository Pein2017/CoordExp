---
title: 256-Image Coordinate-Boundary Training Screen
description: A one-epoch training screen that tests whether first-wrong-coordinate preference can improve geometry and rollout behavior at a meaningful image scale.
type: investigation
role: research-unit
authority: non_normative_research
unit_id: 2026-07-21-256-image-coordinate-boundary-training-screen
topic: qwen3-vl-dense-enumeration
status: completed
updated: 2026-07-21
---

# 256-Image Coordinate-Boundary Training Screen

## Why This Successor Unit Exists

The preceding eight-training-event, two-optimizer-step Smoke B established that
the implementation runs and that the coordinate objective can move its frozen
exact-prefix margin.  It did not provide enough images, optimizer steps, or
ordinary rollouts to decide whether the treatment has useful end-to-end value.
The user therefore explicitly authorized this larger screen.  The old result
remains valid at its original scope; it is no longer a scale blocker.

## Primary Question

After one complete pass over 256 distinct source-rollout images, does direct
first-wrong-coordinate preference improve geometry or detection behavior over
the source checkpoint?

## Compared Models

1. **Source checkpoint:** evaluation only.
2. **Coordinate-boundary preference plus token-type gate:** learning rate
   `1e-5`.
3. **Lower-learning-rate diagnostic:** the same treatment at learning rate
   `3e-6`, added after the first run showed that fixed-prefix learning and
   free rollout moved differently.

The token-type gate is a language stabilizer inside the coordinate treatment;
it is not an independent scientific control.  The trained model uses the
frozen StateBank, random seed, one epoch, and global effective batch size 32.
With eight data-parallel ranks, this is four planned event presentations per
rank per optimizer step and eight optimizer steps for 256 training events.

The treatment result may support the efficacy of direct coordinate-boundary
correction.  This screen does not claim that the token-type stabilizer has zero
causal contribution; that decomposition is outside the current question.

## StateBank Admission

- exactly 256 training images and 16 held-out evaluation images;
- at most one event per physical image;
- source-checkpoint greedy rollout with repetition penalty 1.0;
- source prediction `coord_bins`, reference boxes, coordinate tolerances, and
  coordinate-loss targets all use normalized integers in `[0,999]`;
- absolute pixel boxes are permitted only for rendering and visual review and
  never participate in owner matching or coordinate-loss supervision;
- current training annotation provides a positive physical owner only;
- prediction and owner category must agree;
- best owner intersection over union must be at least 0.35;
- best-minus-second-best same-category intersection-over-union margin must be
  at least 0.15;
- boxes truncated at the normalized image border and reference extents below
  20 coordinate bins are excluded;
- all coordinates before the selected boundary must lie within eight bins of
  the reference; the selected boundary is the first coordinate outside that
  tolerance;
- unmatched predictions, missing annotations, ambiguous ownership, duplicate
  ownership, and malformed rows receive no gradient;
- the twelve human-refined blind images remain excluded from mining, training,
  and model selection.

The automatic filter is a conservative training-data rule, not a claim that
the training annotations are perfect.  A crop-enlarged sample audit must check
representative admitted cases before training starts.

## Model and Optimization Contract

- source: geometry-sorted description-first pure-cross-entropy plus token-type
  gate Weight-Decomposed Low-Rank Adaptation checkpoint at step 4,887;
- freeze the vision tower and multimodal aligner;
- train only the language-tower Weight-Decomposed Low-Rank Adaptation payload
  and the already-declared special-token embedding rows;
- no canonical supervised-fine-tuning replay, Kullback-Leibler divergence,
  Gaussian coordinate loss, external detector, object slot, ledger, or
  inference-time controller;
- AdamW, learning rate `1e-5`, gradient clipping 1.0, one epoch;
- ordinary rollout evaluation uses repetition penalty 1.0;
- use float32 evaluation for conclusion-critical exact-prefix scoring when
  numerical precision can change the comparison.

## Decision Rule

The evidence is read in this order:

1. Full train-256 float32 paired replay tests whether the treatment signal was
   learned at the exact source-rollout prefixes used for training.
2. Clean rollout on the same 256 images tests whether that learning survives
   the model's own prefixes and improves complete rows and detection behavior.
3. The 16 held-out StateBank events are an early transfer observation only.
   They are too small to be a hard promotion gate.

If the coordinate arm only improves the exact-prefix diagnostic but not
ordinary rollout relative to the source checkpoint, the objective remains
mechanistically real but is rejected as the next scale-training treatment.

## Executed Evidence

All model-facing boxes and coordinate targets in this unit are normalized
integer `x1, y1, x2, y2` values in `[0,999]`. Absolute pixels were used only by
rendering code. The frozen StateBank identifier is
`562c00ca8920fb37f17d0329692053b32e4c2d2f214596548b3add3db3df3b38`.

The 256 training boundaries cover all four coordinate positions:

| Selected first-wrong coordinate | Events |
| --- | ---: |
| `x1` | 86 |
| `y1` | 69 |
| `x2` | 46 |
| `y2` | 55 |

The 16 held-out events contain 6 `x1`, 4 `y1`, 3 `x2`, and 3 `y2`
boundaries. An earlier working note that described all events as `x1` was
incorrect.

### Exact-prefix float32 paired replay

| Model | Mean target margin | Mean coordinate loss | Margin improved |
| --- | ---: | ---: | ---: |
| Source | 0.4575 | 1.2450 | - |
| Treatment, learning rate `1e-5` | 0.8424 | 0.9952 | 242 / 256 |
| Treatment, learning rate `3e-6` | 0.5696 | 1.1654 | 241 / 256 |

Relative to the source, the mean margin change was `+0.3849` at `1e-5`
(95% bootstrap interval approximately `[+0.3171, +0.4623]`) and `+0.1121` at
`3e-6` (approximately `[+0.0945, +0.1307]`). Every coordinate-position group
and every prefix-depth group improved at both learning rates. Therefore the
coordinate treatment is implemented correctly and can teach the requested
local preference on the 256 training events.

### Clean rollout on the same 256 images

All three runs use greedy decoding, repetition penalty 1.0, and the same input
rows.

| Model | Mean Average Precision | Average Precision at IoU 0.75 | Mean Recall | Predictions | Truncated rows | Dropped predictions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Source | 0.3880 | 0.4115 | 0.4528 | 2,978 | 3 | 4 |
| Treatment, learning rate `1e-5` | 0.3870 | 0.4178 | 0.4474 | 3,161 | 9 | 25 |
| Treatment, learning rate `3e-6` | 0.3852 | 0.4079 | 0.4494 | 3,057 | 6 | 11 |

The higher learning rate raised Average Precision at IoU 0.75 by 0.0063, but
this did not constitute a net improvement: Mean Average Precision and recall
fell, output length grew, and malformed tail behavior increased. Reducing the
learning rate limited those disturbances but did not improve the detection
metrics.

For the physical owner associated with each training event, the best
same-category box intersection over union averaged 0.7753 for the source,
0.7686 at `1e-5`, and 0.7681 at `3e-6`. At the StateBank row index, 247 source
rows still named the intended category, compared with 220 at `1e-5` and 228 at
`3e-6`. Among the 225 events where both the source and `3e-6` treatment kept
the intended category at that row, the selected coordinate's absolute error
improved by 2.09 normalized bins on average, but full-box intersection over
union fell by 0.0177. This is direct evidence that a boundary can improve
locally while row identity or the other box boundaries become worse.

The 16 held-out events are retained only as an early transfer observation. The
`1e-5` treatment improved 13 of 16 local margins but included one large
long-prefix regression. The `3e-6` treatment improved 14 of 16, with mean and
median margin changes of `+0.0058` and `+0.0421`. These small, mixed panels do
not establish generalization and are not used to override the train-256 clean
rollout result.

## Decision and Next Constraint

This screen does **not** promote either treatment to 1,024 images or the full
dataset. It establishes a narrower result:

> Direct coordinate-boundary supervision can reliably change the intended
> coordinate preference at fixed source-rollout prefixes, but ordinary
> language-tower DoRA updates do not reliably preserve that correction through
> self-generated prefixes and the rest of the object row.

A further learning-rate sweep is not justified. The next treatment must address
at least one of the two unresolved causes rather than repeating this screen:

1. train on prefixes produced by the treated model, so the supervised boundary
   matches the states encountered during rollout; or
2. confine the correction to a more local coordinate-decision path, so that
   changing one boundary does not also alter object identity, other boundaries,
   or row continuation.

The smallest discriminating successor is to compare one treated-model-prefix
refresh against an equally sized repeat on the original frozen prefixes. If
only the refreshed-prefix arm transfers into clean rollout, prefix mismatch is
the dominant problem. If both fail while exact-prefix margins improve, the
global language-parameter update is too entangled for this treatment and a
more local intervention is required.

## Artifact Handles

- Research output root:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-21-256-image-coordinate-boundary-training-screen`
- Source float32 train replay: `train-256-coordinate-source-fp32.json`
- `1e-5` float32 train replay: `train-256-coordinate-treated-fp32.json`
- `3e-6` float32 train replay:
  `train-256-coordinate-treated-learning-rate-3e-6-fp32.json`
- Clean-rollout runs and detection metrics: `train-256-clean-rollouts/`
