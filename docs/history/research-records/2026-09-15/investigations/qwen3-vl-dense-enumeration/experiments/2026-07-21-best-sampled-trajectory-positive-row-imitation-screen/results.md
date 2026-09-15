---
title: Best Sampled Trajectory Positive Row Imitation Results
description: One-epoch evidence showing that single-route positive imitation shifts greedy output toward the selected-route owner set but loses other owners, leaving aggregate coverage flat or worse.
type: investigation
role: results
authority: non_normative_research
unit_id: 2026-07-21-best-sampled-trajectory-positive-row-imitation-screen
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: executed_bounded_route_level_shift_without_net_coverage_gain
architecture_promotion_status: not_promoted
updated: 2026-07-21
---

# Best Sampled Trajectory Positive Row Imitation Results

## Verdict

**Do not scale this exact treatment unchanged to 1,024 images. Preserve the
route-conditioned credit hypothesis and test it next with an explicit
preservation constraint.**

Across the 256-image training cohort, every evaluated checkpoint emits fewer
predictions and matches fewer unique annotated owners than the frozen source.
The twelve human-refined images show the same aggregate owner-coverage
direction. Intermediate checkpoints slightly improve official mean Average
Precision; the joint metric and visual-review pattern is consistent with
removing some low-quality or repeated predictions and retaining somewhat
tighter high-overlap boxes, not with finding more objects overall.

Selected intermediate checkpoints therefore exhibit a shorter-output
precision-versus-coverage trade-off in this run. Whether that pattern can be
used deliberately as a precision or repetition regularizer is an untested
follow-up hypothesis. It is not the intended coverage treatment.

This is not a null result. On the 118 images that contributed training events,
the treatment makes more of the physical owners added by the selected sampled
routes appear in a new clean greedy rollout. At step 15, 109 of 238 such owners
are matched, compared with 93 for a separate Source rollout. Relative to that
Source rollout, step 15 gains 31 and loses 15 owners in this fixed route-added
set. At the same time it loses an almost equal number of owners that were
already present on the ordinary selected-route path. Total annotated-owner
coverage on the 118 training images is therefore unchanged, while coverage on
the 138 images that did not contribute events declines.

This is a route-level shift, not demonstrated direct owner-wise imitation.
Only 118 of the 238 route-added owners are direct positive-row event targets.
Matches on those direct targets change only from 61 for Source to 62, 63, and
63 at steps 10, 15, and 16. Matches on the other route-added owners change from
32 to 45, 46, and 47 and account for most of the net shift. The treatment
therefore changes behavior toward the owner distribution represented by the
selected routes; it does not establish that each directly supervised owner is
independently memorized or recovered.

The most precise interpretation is:

> Within this executed screen, positive complete-row supervision redirects
> greedy behavior toward the owner set represented by the selected sampled
> routes. It does not meaningfully expand selected-route coverage or aggregate
> annotated-owner coverage because other stable owners are not preserved and
> behavior outside the admitted training images regresses.

## Executed Scope

- Source: geometry-sorted, description-first, pure-cross-entropy checkpoint at
  step 4,887.
- Training bank: 512 exact-prefix positive-row events from 118 images.
- Bank identity:
  `c496a3653f6f46539d6bce2e110729c7ee77668fab751a5957bd60fe7829fde3`.
- Optimization: language-tower Weight-Decomposed Low-Rank Adaptation only,
  learning rate `1e-5`, gradient clipping at `1.0`, one epoch, eight Graphics
  Processing Units, sixteen optimizer steps.
- Saved milestones: steps 5, 10, 15, and final step 16.
- All 64 microsteps were consumed; every optimizer update was applied; final
  finite status was `finite`.
- Clean greedy evaluation: train-256 at steps 10, 15, and 16; twelve independent
  human-refined images at steps 5, 10, 15, and 16.

The train-256 owner analysis uses one-to-one, same-category, maximum-cardinality
matching at Intersection over Union at least `0.50`. It is an annotated-owner
coverage measure, not an exhaustive physical-entity census. Unmatched
predictions are never automatically called hallucinations.
The twelve-image panel is exploratory and has
`benchmark_eligible: false`; it is not a benchmark estimate.

## Official Detection Metrics

### Train-256

| Checkpoint | Mean Average Precision | Average Precision at 0.50 | Average Precision at 0.75 | Mean Recall | Predictions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Source | 0.3880 | 0.5719 | 0.4115 | 0.4528 | 2,978 |
| Step 10 | 0.3909 | 0.5664 | 0.4206 | 0.4528 | 2,734 |
| Step 15 | 0.3913 | 0.5695 | 0.4211 | 0.4528 | 2,689 |
| Step 16 | 0.3897 | 0.5651 | 0.4207 | 0.4507 | 2,711 |

Step 15 has the best mean Average Precision, but its prediction count falls by
289 and its mean Recall is effectively unchanged. The Average Precision at
`0.75` increase together with the Average Precision at `0.50` decrease is
consistent with retaining cleaner high-overlap detections while losing some
broader object coverage; this screen does not causally decompose that change.

### Twelve human-refined images

| Checkpoint | Mean Average Precision | Average Precision at 0.50 | Average Precision at 0.75 | Mean Recall | Predictions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Source | 0.3757 | 0.5265 | 0.3749 | 0.4095 | 276 |
| Step 5 | 0.3736 | 0.5244 | 0.3688 | 0.4193 | 223 |
| Step 10 | 0.3902 | 0.5325 | 0.3929 | 0.4273 | 208 |
| Step 15 | 0.3824 | 0.5267 | 0.3958 | 0.4248 | 214 |
| Step 16 | 0.3316 | 0.4581 | 0.3379 | 0.3707 | 230 |

The development panel is small, but it exposes an important instability:
step 10 is the best milestone, while one final optimizer step from step 15 to
step 16 produces a large greedy-behavior regression. The final checkpoint
cannot be assumed to be the best checkpoint even in a one-epoch low-learning-
rate treatment.

## Unique Annotated-Owner Coverage

### Train-256

| Checkpoint | Matched owners / 3,001 | Coverage | Change from Source | Conservative duplicate candidates | Prediction change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Source | 1,677 | 0.5588 | - | 188 | - |
| Step 10 | 1,661 | 0.5535 | -16 owners | 136 | -244 |
| Step 15 | 1,658 | 0.5525 | -19 owners | 155 | -289 |
| Step 16 | 1,649 | 0.5495 | -28 owners | 133 | -267 |

### Twelve human-refined images

| Checkpoint | Matched owners / 346 | Coverage | Change from Source | Conservative duplicate candidates | Prediction change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Source | 139 | 0.4017 | - | 14 | - |
| Step 5 | 134 | 0.3873 | -5 owners | 24 | -53 |
| Step 10 | 132 | 0.3815 | -7 owners | 4 | -68 |
| Step 15 | 135 | 0.3902 | -4 owners | 6 | -62 |
| Step 16 | 125 | 0.3613 | -14 owners | 47 | -46 |

The duplicate statistic is deliberately named a candidate count. It counts
extra predictions that can be attributed to exactly one same-category
annotated owner at Intersection over Union at least `0.30`; dense multi-owner
overlaps are reported separately. It is useful for paired screening but is not
a human-confirmed duplicate count.

## Transfer Toward Owners Added by the Selected Sampled Routes

For the 118 StateBank images, the chosen sampled routes contain 238 annotated
owners absent from the frozen sixteen-sample route-analysis greedy artifact and
740 ordinary owners also present in that artifact. The table below intersects
those fixed owner sets with each later clean greedy evaluation.

| Checkpoint | Route-added owners matched / 238 | Gained / lost versus Source | Ordinary route owners matched / 740 | Gained / lost versus Source | All selected-route owners matched / 978 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Source | 93 | - | 712 | - | 805 |
| Step 10 | 107 | +26 / -12 | 700 | +16 / -28 | 807 |
| Step 15 | 109 | +31 / -15 | 697 | +16 / -31 | 806 |
| Step 16 | 110 | +28 / -11 | 697 | +18 / -33 | 807 |

The route-added count rises at every checkpoint. The ordinary-owner count
falls by a similar amount, so the union of owners represented by the selected
routes changes by only `+2`, `+1`, and `+2`. This pattern supports a behavioral
shift toward the selected route family and shows that the current update does
not preserve alternative useful behavior in the same evaluation.

The direct-target split prevents a stronger interpretation:

| Owner subset | Owners | Source matched | Step 10 | Step 15 | Step 16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct positive-row event targets | 118 | 61 | 62 | 63 | 63 |
| Other owners in the fixed route-added set | 120 | 32 | 45 | 46 | 47 |

Most of the measured change is on owners that share the selected route family
but were not themselves direct positive-row targets. This may reflect a route-
level redistribution of prefix-conditioned choices. It is not evidence for an
isolated owner-by-owner credit-assignment mechanism.

A deterministic 20,000-replicate paired bootstrap, resampling the 118 images
as clusters, gives 95-percent intervals of `[+2, +27]`, `[+2, +30]`, and
`[+5, +30]` for the total route-added-owner change at steps 10, 15, and 16.
The corresponding ordinary-owner intervals are `[-27, +4]`, `[-30, 0]`, and
`[-30, +1]`. This supports a repeatable shift toward the fixed route-added set
across images in this one screen; it does not remove the separate-execution
Source caveat below, identify direct owner-wise imitation, or establish
population-level generalization.

The same split is visible without restricting owners to the selected routes:

| Scope | Source | Step 10 | Step 15 | Step 16 |
| --- | ---: | ---: | ---: | ---: |
| 118 images with StateBank events, matched / 1,761 | 907 | 903 | 907 | 903 |
| 138 images without StateBank events, matched / 1,240 | 770 | 758 | 751 | 746 |

Step 15 is the clearest diagnostic milestone: it has the largest net shift
toward the fixed route-added set (`+16`), the best train-256 mean Average
Precision, and no net annotated-owner loss within the 118 admitted images. It
still loses 19 owners over all 256 images because the non-admitted partition
regresses. Step 16 has
one more net targeted owner than step 15 but is worse on aggregate transfer and
on the human-refined panel. None of these checkpoints is promoted as a final
model.

The phrase "route-added" has a strict provenance boundary. It means absent
from the frozen sixteen-sample route-analysis greedy artifact, not absent from
every possible Source execution. The later clean Source rollout already
matches 93 of the 238 owners. The two Source artifacts use the same model artifacts,
image paths, prompt token identities, generation policy, and stable decode
settings, but only 18 of 256 parsed outputs are exactly identical across the
two separate Compute Unified Device Architecture (`CUDA`) executions.
Therefore only treatment-minus-clean-Source changes are called gains.

## Geometry on Commonly Matched Owners

Geometry is compared only for the same annotated owner when both Source and a
treatment checkpoint match it at Intersection over Union at least `0.50`.
This avoids claiming that aggregate geometry improved merely because difficult
owners disappeared.

| Panel | Checkpoint | Common owners | Mean Intersection-over-Union change | Mean center-error change | Mean size-error change |
| --- | --- | ---: | ---: | ---: | ---: |
| Train-256 | Step 10 | 1,570 | +0.00184 | -0.286 pixels | -0.665 pixels |
| Train-256 | Step 15 | 1,566 | +0.00114 | -0.167 pixels | -0.326 pixels |
| Train-256 | Step 16 | 1,568 | +0.00175 | -0.190 pixels | -0.545 pixels |
| Human-refined 12 | Step 10 | 125 | -0.00168 | -0.231 pixels | +0.271 pixels |
| Human-refined 12 | Step 15 | 124 | -0.00299 | -0.036 pixels | +0.240 pixels |
| Human-refined 12 | Step 16 | 117 | +0.00138 | -0.390 pixels | +0.102 pixels |

The train-256 changes are small but directionally consistent with slightly
tighter retained detections. They do not compensate for lost owner coverage,
and they do not establish general localization improvement on the human-
refined panel.

## Visual Review

The Source-versus-step-10 visual panel agrees with the paired statistics.
Step 10 often removes repeated or weak rows and sometimes tightens boxes, but
it usually finds the same or fewer physical owners. Representative inspected
images include 1,584, 2,685, 6,040, 10,707, 13,923, 14,038, 14,439, and 16,228.
Image 14,038 shows a large cleanup at unchanged matched-owner count; images
16,228 and 14,439 lose owners.

This review does not convert unmatched predictions into hallucinations. Common
Objects in Context annotations can still omit real entities, even in the
human-refined subset.

## What This Screen Establishes

The training objective says:

```text
Under this exact sampled prefix, increase the likelihood of this one chosen
complete row from this one chosen route.
```

It does not say:

```text
Across all valid routes, maximize the final set of distinct physical owners.
```

One winning trajectory is therefore not a set-level target. Equal total weight
per image prevents long routes from dominating, but it neither preserves the
Source route nor turns route imitation into a direct objective over the final
owner set.

The observed signature is exactly this mismatch:

1. ordinary output becomes shorter;
2. conservative duplicate candidates usually decline;
3. high-overlap geometry and mean Average Precision sometimes improve;
4. owners in the fixed route-added set enter greedy more often, mainly through
   owners that were not direct positive-row event targets;
5. ordinary owners are lost at a similar rate, so total owner coverage is flat
   on admitted images and declines overall;
6. one additional step can move greedy decoding into a substantially worse
   trajectory basin.

This screen does not isolate why the executed configuration produces that
signature. Suppression of alternative continuations is one compatible
explanation, but off-policy prefix mismatch, deterministic route selection,
the token-type gate, and optimization instability remain competing
explanations. The evidence rules out promotion of this executed configuration;
the route-level shift also rules out the stronger claim that positive route
imitation simply fails to affect greedy behavior. This screen still bundles
route selection, exact-prefix off-policy imitation, row weighting, token-type
gating, and optimization, so it does not isolate which component causes the
offsetting preservation loss.

## Decision and Next Discriminator

Cancel the 1,024-image replication of this exact objective. Keep the
implementation, bank, milestones, and transfer receipt as evidence. The next
useful step is another 256-image treatment screen, not the identical larger
run.

The strongest next treatment hypothesis is that sampled-route credit must be
paired with preservation of already stable owners. To distinguish the two
changes instead of bundling them again, the next screen should use the same
Source checkpoint, 256-image cohort, optimizer budget, and total per-image
credit for these matched arms:

1. frozen Source evaluation;
2. the current single-route treatment, rerun only if an exactly matched budget
   cannot be reconstructed from this screen;
3. single-route positive rows plus a Source-route preservation anchor;
4. multi-route positive rows plus the same Source-route preservation anchor.

Within those arms, the data policy should:

1. include several safely verified high-coverage trajectories in the
   multi-route arm rather than one canonical sampled route;
2. retain a Source-route anchor for owners already recovered reliably;
3. normalize credit within each image so adding routes does not increase that
   image's total training weight;
4. reward distinct verified owner discovery and neutralize annotation-unknown
   predictions rather than treating them as negatives;
5. penalize only confirmed repetition and malformed output;
6. compare trajectories under the same image, prompt, decode budget, and source
   checkpoint;
7. report route-added-owner gain, ordinary-owner retention, total owner
   coverage, non-admitted-image transfer, repetition candidates, malformed
   rows, and geometry separately.

The minimum version is still supervised learning. The matched arms determine
whether preservation, route diversity, or their combination is responsible
for any gain. A later group-relative trajectory objective remains an option if
this simpler test cannot retain stable owners. Neither version requires object
slots, an external detector, or an inference-time controller.

## Evidence Handles

- Training run:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-21-best-sampled-trajectory-positive-row-imitation-screen/train-512-events-v1/runs/qwen3_vl_2b_positive_path_imitation_512_events_one_epoch_learning_rate_1e-5/`
- StateBank:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-21-best-sampled-trajectory-positive-row-imitation-screen/state-bank-v1/`
- Train-256 clean rollouts:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-21-best-sampled-trajectory-positive-row-imitation-screen/train-256-clean-rollouts/`
- Train-256 Source metric and prediction artifact:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-21-256-image-coordinate-boundary-training-screen/train-256-clean-rollouts/qwen3-vl-2b-step4887-coordinate-boundary-train-256-source-hf/`
- Twelve-image clean rollouts:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-21-best-sampled-trajectory-positive-row-imitation-screen/human-refined-12-clean-rollouts/`
- Completed twelve-image Source and final directories:
  `qwen3-vl-2b-positive-path-imitation-source-human-refined-12-hf-v2/` and
  `qwen3-vl-2b-positive-path-imitation-step16-human-refined-12-hf-v2/` under
  the preceding root. The same names without `-v2` are failed partial launch
  directories and are not metric-bearing evidence.
- Paired owner and geometry receipts:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-21-best-sampled-trajectory-positive-row-imitation-screen/paired-owner-coverage/`
- Selected-route added-owner transfer receipt:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-21-best-sampled-trajectory-positive-row-imitation-screen/paired-owner-coverage/selected-route-added-owner-transfer.json`
- Visual comparison:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-21-best-sampled-trajectory-positive-row-imitation-screen/visual-review/source-vs-step10-human-refined-12/`
