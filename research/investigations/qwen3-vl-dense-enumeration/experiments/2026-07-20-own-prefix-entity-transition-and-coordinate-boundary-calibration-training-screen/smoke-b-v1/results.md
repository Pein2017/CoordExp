---
title: Smoke B First-Wrong-Coordinate Calibration Results
description: A reviewed 8-train-event and 5-evaluation-event smoke finds transferable exact-prefix coordinate-margin movement but no stable free-rollout advantage over the learning-rate-matched token-type-gate control.
type: investigation
role: experiment-result
authority: non_normative_research
architecture_promotion_status: not_promoted
training_promotion_status: not_promoted
evidence_status: smoke_b_complete_bounded_negative
topic: qwen3-vl-dense-enumeration
status: complete
updated: 2026-07-20
---

# Smoke B First-Wrong-Coordinate Calibration Results

## Verdict

Stop before the 256-image screen.

The first-wrong-coordinate objective is a real and locally transferable
training signal. After two optimizer steps on eight reviewed training events,
both coordinate-loss arms move the accepted-coordinate-versus-wrong-token
margin in the intended direction on four of five image-held-out events and
outperform their learning-rate-matched selected-site token-type-gate-only
controls on the same four events.

That local effect does not produce a stable whole-rollout advantage. On the
three images that own the five held-out events, neither coordinate arm improves
matched-box Intersection over Union relative to its matched gate-only control.
Both coordinate arms produce many more unmatched predictions and duplicate
candidate pairs. Across all eight ordinary-rollout images, the gate-only arm at
learning rate `1e-5` has the strongest aggregate detection result and much
healthier duplication behavior than the coordinate-loss arm at the same
learning rate.

The correct bounded conclusion is therefore:

> A first-wrong-coordinate preference can reshape the intended exact-prefix
> coordinate decision, but this local correction is not sufficient to improve
> the model's multi-row greedy trajectory reliably.

This closes the current loss-only coordinate arm as a scale candidate. It does
not reject coordinate calibration in general, and it does not establish that
an explicit object-state carrier is necessary.

## Frozen Evidence Scope

- Source model: geometry-sorted pure-cross-entropy Qwen3 Vision-Language
  (`Qwen3-VL`) 2B checkpoint at step `4,887`.
- State bank: 13 independently reviewed coordinate events, split by image into
  8 training events and 5 evaluation events.
- Evaluation events: two chair boundaries on image `19432`, one person
  boundary on image `2299`, and two person boundaries on image `9590`.
- Training: one pass over the eight training events, effective batch size 4,
  two optimizer steps, only the language-model tower trainable through
  Weight-Decomposed Low-Rank Adaptation (`DoRA`).
- Arms: coordinate preference plus selected-site token-type gate at learning
  rates `1e-5` and `2e-5`, and a matched gate-only control at each learning
  rate.
- Exact-prefix evaluation: one model per invocation, full-model 32-bit
  floating point, Scaled Dot-Product Attention (`SDPA`), no backward pass.
- Ordinary evaluation: deterministic greedy decoding, repetition penalty
  `1.0`, batch size 4, eight images, and the same 512-token limit for every
  arm.
- The twelve human-refined dense images remained blind and were not used for
  training, event selection, learning-rate selection, smoke evaluation, or
  model selection.

The frozen state-bank root is:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-20-own-prefix-entity-transition-and-coordinate-boundary-calibration-training-screen/
  smoke-b-v1/coordinate-state-bank-v4
```

Its bank identifier is
`f15e3c8d9555a57138627db7028e710edbf65638b7325354577ff71b93d88a47`.

## Exact-Prefix Coordinate Result

Higher coordinate margin is better. The margin is the logit mass of the
reviewed acceptable coordinate-token set minus the actual wrong coordinate
token at the frozen causal site.

| Model | Evaluation events | Mean coordinate margin | Mean coordinate loss | Legal coordinate-token mass |
| --- | ---: | ---: | ---: | ---: |
| Source checkpoint | 5 | -3.66484 | 4.93122 | 0.998773 |
| Coordinate plus gate, `1e-5` | 5 | -3.61522 | 4.89007 | 0.998757 |
| Gate only, `1e-5` | 5 | -3.77220 | 5.03560 | 0.998928 |
| Coordinate plus gate, `2e-5` | 5 | -3.55577 | 4.84281 | 0.998745 |
| Gate only, `2e-5` | 5 | -3.84039 | 5.11666 | 0.999078 |

Paired effects are small but directionally coherent:

| Coordinate arm | Mean margin change versus source | Mean margin change versus matched gate-only | Events better than matched gate-only |
| --- | ---: | ---: | ---: |
| `1e-5` | +0.04963 | +0.15699 | 4 / 5 |
| `2e-5` | +0.10907 | +0.28461 | 4 / 5 |

One chair top-boundary event worsens relative to source at both learning rates,
and the second chair event improves relative to source but loses to gate-only.
The result is therefore transfer, not uniform correction. The five events also
come from only three images, so these numbers are a mechanism smoke rather than
a variance estimate.

The exact evaluation artifacts are under
[heldout-coordinate-evaluation](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-20-own-prefix-entity-transition-and-coordinate-boundary-calibration-training-screen/smoke-b-v1/heldout-coordinate-evaluation/).

## Ordinary Greedy Rollout

The eight-image Common Objects in Context bounding-box metrics are diagnostic,
not benchmark estimates.

| Model | mean Average Precision | Average Precision at IoU 0.75 | mean Recall | Predictions | Dropped predictions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Source checkpoint | 0.29995 | 0.30575 | 0.33916 | 214 | 1 |
| Coordinate plus gate, `1e-5` | 0.31334 | 0.31580 | 0.35969 | 218 | 1 |
| Gate only, `1e-5` | **0.32803** | **0.34995** | **0.37078** | 185 | 0 |
| Coordinate plus gate, `2e-5` | 0.31574 | 0.33109 | 0.35436 | 195 | 0 |
| Gate only, `2e-5` | 0.31788 | 0.32761 | 0.35934 | 192 | 1 |

The shared visualization matcher gives the following all-eight-image totals at
Intersection over Union `0.5`. Duplicate candidate pairs are a review aid and
not a count of unique duplicated physical objects.

| Model | Matched predictions | Missed annotations | Unmatched predictions | Duplicate candidate pairs | Mean IoU over matched pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Source checkpoint | 113 | 107 | 101 | 303 | 0.79448 |
| Coordinate plus gate, `1e-5` | 123 | 97 | 95 | 163 | 0.79303 |
| Gate only, `1e-5` | **124** | **96** | **61** | **37** | 0.80487 |
| Coordinate plus gate, `2e-5` | 120 | 100 | 75 | 138 | 0.80392 |
| Gate only, `2e-5` | 116 | 104 | 76 | 215 | **0.80760** |

The gate-only `1e-5` arm explains more of the aggregate ordinary-rollout gain
than the coordinate objective. The coordinate objective does not improve mean
matched-box geometry relative to its matched control at either learning rate.

## Held-Out-Image Rollout Check

Restricting the ordinary rollout to images `2299`, `9590`, and `19432`, which
own all five exact-prefix evaluation events, removes apparent gains coming from
training images or unrelated images.

| Model | Matched predictions | Unmatched predictions | Duplicate candidate pairs | Mean IoU over matched pairs |
| --- | ---: | ---: | ---: | ---: |
| Source checkpoint | 43 | 39 | 51 | 0.77800 |
| Coordinate plus gate, `1e-5` | 43 | 39 | 62 | 0.77974 |
| Gate only, `1e-5` | 42 | 22 | 9 | 0.78321 |
| Coordinate plus gate, `2e-5` | 40 | 42 | 115 | 0.78255 |
| Gate only, `2e-5` | 37 | 18 | 5 | 0.80863 |

The `1e-5` coordinate arm exactly matches the source total for matched and
unmatched predictions on these held-out images while increasing duplicate
candidate pairs. The `2e-5` coordinate arm loses three source matches and adds
three unmatched predictions while more than doubling the source duplicate
candidate-pair count.

To remove the easier-match selection difference between variants, a second
diagnostic retains only `(image, ground-truth index)` entries matched by both
members of each pair. For every retained entry, boundary error is the mean
absolute `x1`, `y1`, `x2`, and `y2` error in normalized 0-to-999 coordinate
bins. Each model's Intersection over Union is taken from its own canonical
match to that same ground-truth entry.

| Pair | Common matched entries | Mean boundary error: left model to coordinate model | Mean IoU: left model to coordinate model |
| --- | ---: | ---: | ---: |
| Source versus coordinate `1e-5` | 41 | 9.750 to **10.207**, worse | 0.7793 to **0.7744**, worse |
| Gate-only `1e-5` versus coordinate `1e-5` | 40 | 9.244 to **9.400**, worse | 0.7852 to **0.7848**, flat or worse |
| Source versus coordinate `2e-5` | 38 | 8.651 to 8.612, 0.5% better | 0.7901 to 0.7918, +0.0017 |
| Gate-only `2e-5` versus coordinate `2e-5` | 31 | 9.056 to **9.621**, worse | 0.8164 to **0.8021**, worse |

These effects are far below the later pilot threshold of a 10% boundary-error
reduction or a `0.03` Intersection-over-Union gain. This is a matched-survivor
diagnostic rather than a complete entity metric: it excludes every missed and
unmatched entity, each comparison can retain a different subset, and it does
not adjudicate unlabeled real objects.

The instability is image-specific rather than a simple rollout-length effect:

- on image `9590`, coordinate plus gate produces 59 duplicate candidate pairs
  at `1e-5` versus 6 for gate-only, and 112 at `2e-5` versus 2 for gate-only;
- on image `18380`, the `2e-5` relation reverses: coordinate plus gate produces
  2 duplicate candidate pairs versus 183 for gate-only; and
- on image `19432`, all four trained arms remain close in row count, while the
  coordinate arms do not remove the visibly mixed or over-wide chair extents.

This is probability redistribution between rollout basins, not a stable
owner-conditioned geometry correction.

The complete eight-image visual manifests are under
[all-eight review](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-20-own-prefix-entity-transition-and-coordinate-boundary-calibration-training-screen/smoke-b-v1/review/all-eight/).
The direct `1e-5` coordinate-versus-gate comparison is under
[coordinate-1e-5-vs-gate-only-1e-5](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-20-own-prefix-entity-transition-and-coordinate-boundary-calibration-training-screen/smoke-b-v1/review/coordinate-1e-5-vs-gate-only-1e-5/).

## Interpretation

1. **The objective is learnable.** The held-out exact-prefix margin movement
   cannot be attributed to the token-type gate alone.
2. **The target is too local for the desired behavior.** Improving the first
   wrong coordinate under a frozen prefix does not guarantee the same row is
   reached, the same physical owner remains selected, or later coordinates and
   later rows stay coherent during free rollout.
3. **The type gate is a major confounder and useful control.** It improves
   aggregate rollout more than the coordinate term at `1e-5`, even though its
   held-out coordinate margin moves in the wrong direction.
4. **Local margin and global quality are distinct endpoints.** The experiment
   provides a direct example of an offline causal-site improvement that does
   not establish an end-to-end treatment.
5. **The result does not diagnose missing capacity.** It neither proves nor
   disproves that Qwen3-VL can represent full object extent or covered-object
   state. It only shows that this two-step local loss does not control those
   behaviors reliably.

## Decision and Next Boundary

- Do not launch replicated seeds, the 256-image screen, the 1,024-image stage,
  or full-size training for this first-wrong-coordinate objective.
- Retain the implementation and frozen evaluator as reusable research tools.
- Keep entity-transition and joint arms deferred because no visually trusted
  same-prefix complete-row rescue was found.
- If a later unit revisits loss-only treatment, follow the preregistered branch:
  compare one-row alternatives under the same natural prefix using a
  downstream owner-and-geometry value, rather than increasing this local
  coordinate loss.
- A compact covered-object state or visual write-back remains a competing
  hypothesis, not the automatic next implementation.

Claims not supported by this smoke include general coordinate improvement,
better instance binding, improved covered-set tracking, improved dense-image
recall, and a need for a new inference architecture.
