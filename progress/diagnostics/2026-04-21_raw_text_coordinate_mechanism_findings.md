---
title: Raw-Text Coordinate Mechanism Findings
date: 2026-04-21
status: mechanism-summary
owner: codex
depends_on:
  - output/analysis/raw-text-coordinate-mechanism-good-basin-20260421/summary.json
  - output/analysis/raw-text-coordinate-mechanism-preburst-margin-model-native-20260421/summary.json
  - output/analysis/raw-text-coordinate-mechanism-preburst-margin-pretty-inline-20260421/summary.json
  - output/analysis/raw-text-coordinate-mechanism-fn-suppression-model-native-20260421/summary.json
---

# Raw-Text Coordinate Mechanism Findings

## Why This Note Exists

The previous raw-text continuity probe was intentionally reset and replaced by a
new mechanism-first study. This note records the current state of that rebuilt
study under the approved scope:

- only the two raw-text checkpoint objects
- no `coord_token` comparisons
- explicit separation of good-basin, bad-basin, pre-burst, and FN-suppression
  mechanisms

This is the decision-facing historical record under `progress/diagnostics`.

## Scope And Active Comparison

The study keeps one fixed two-object comparison throughout:

1. `base_only`
   - `/data/CoordExp/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp`
2. `base_plus_adapter`
   - base:
     `/data/CoordExp/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp`
   - adapter:
     `/data/CoordExp/output/stage1_2b/coco_bbox_max60-coco80-desc_first-1024-lvis_proxy-raw_text_xyxy-pure_ce/epoch_4-raw_text_xyxy-pure_ce-coco80-desc_first-1024-lvis_proxy-from-base-2B/v1-20260417-084341/checkpoint-552`

All findings below are about raw-text `norm1000_text` behavior only.

## Core Artifact Bundle

### Confirmatory / continuity side

- good-basin pilot:
  `/data/CoordExp/output/analysis/raw-text-coordinate-mechanism-good-basin-20260421`
- duplicate-burst onset, model-native:
  `/data/CoordExp/output/analysis/raw-text-coordinate-mechanism-duplicate-burst-model-native-20260421`
- duplicate-burst onset, pretty-inline:
  `/data/CoordExp/output/analysis/raw-text-coordinate-mechanism-duplicate-burst-pretty-inline-20260421`

### New causal mechanism side

- pre-burst margin, model-native:
  `/data/CoordExp/output/analysis/raw-text-coordinate-mechanism-preburst-margin-model-native-20260421`
- pre-burst margin, pretty-inline:
  `/data/CoordExp/output/analysis/raw-text-coordinate-mechanism-preburst-margin-pretty-inline-20260421`
- pre-burst surface comparison:
  `/data/CoordExp/output/analysis/raw-text-coordinate-mechanism-preburst-surface-comparison-20260421.json`

### FN mechanism side

- FN suppression:
  `/data/CoordExp/output/analysis/raw-text-coordinate-mechanism-fn-suppression-model-native-20260421`
- FN visual review gallery:
  `/data/CoordExp/output/analysis/raw-text-coordinate-mechanism-fn-suppression-model-native-20260421/review_gallery/review.html`

## Main Findings

### 1. Raw-text continuity is real, but raw summed score is length-biased

The rebuilt good-basin lane confirmed that local coordinate continuity is not a
fake artifact of single-token logits. However, naive summed logprob is strongly
biased toward shorter numeric realizations.

After controlling for token length, the local GT basin remains strong:

- base `same_gt_token_mass_at_4_mean = 0.8774`
- adapter `same_gt_token_mass_at_4_mean = 0.8907`
- base `same_gt_token_best_is_gt_rate = 0.8281`
- adapter `same_gt_token_best_is_gt_rate = 0.8203`

So the honest read is:

- continuity exists
- but mechanism claims should prefer length-aware or matched-token analyses

### 2. Duplicate-burst onset bad basins are real

The onset probe already showed that when duplicate burst is triggered, the local
mass concentrates around the emitted wrong object rather than the GT object.

This remains true after the rerun with explicit `previous` anchor metrics:

- onset-object bad basin is strong
- previous-anchor attraction is measurable
- surface sensitivity is modest on base and very small on the adapter aggregate

That was important, but it still left open whether the harmful attraction
already exists *before* the duplicate object is emitted.

### 3. Pre-burst anchor-collapse is now directly supported

The new pre-burst probe scores pairwise margins before the duplicate object is
emitted:

`margin = score(gt_next) - score(exact_duplicate)`

under a pre-onset prefix, then recomputes that margin after prefix-geometry
interventions.

#### Base-only

On `model_native`, base-only remains in a bad pre-burst basin on average:

- baseline `mean_margin_mean_logprob = -0.2443`
- `drop_source = -0.2961`
- `source_x1y1_from_gt_next = -0.2874`

Interpretation:

- the base model does not recover the GT candidate on average under these
  hard duplicate-burst cases
- simply changing the previous object’s `x1/y1` is not enough to rescue it in
  aggregate

#### Base plus raw-text adapter

The adapter behaves differently:

- baseline `mean_margin_mean_logprob = -0.0061`
- `source_x1y1_from_gt_next = 0.0719`
- `mean_delta_from_baseline_mean_logprob = +0.0780`
- `positive_margin_mean_rate = 0.625`

Interpretation:

- the adapter is much closer to the GT-vs-duplicate boundary before onset
- moving only the previous object’s `x1/y1` often shifts the current choice
  toward GT

This is strong evidence that the harmful local basin is not just “the model
liking repeated objects” in a generic semantic sense. Prefix geometry itself is
causally involved.

### 4. The adapter’s pre-burst result is largely surface-stable

The pre-burst comparison was run on both:

- `model_native`
- `pretty_inline`

The adapter aggregate is effectively unchanged between the two surfaces, while
base-only shifts modestly.

That matters because it argues against the new pre-burst finding being only a
serialization accident. The strongest causal signal survived the surface swap.

### 5. FN behavior is not purely “no perception”; stop pressure is real

The new FN probe takes selected raw-text oracle-recovered FN cases and compares:

- `eos_now`
- versus `continue_with_gt`

under teacher-forced changed-span scoring on the same baseline raw-text prefix.

Across the selected 5 cases:

- both models always prefer `eos_now` on *joint* continuation score
- but continuation is often better on *mean per-token* score

Aggregate result:

- base-only:
  - `positive_continue_minus_eos_sum_rate = 0.0`
  - `positive_continue_minus_eos_mean_rate = 0.6`
  - `stop_pressure_rate = 0.6`
- adapter:
  - `positive_continue_minus_eos_sum_rate = 0.0`
  - `positive_continue_minus_eos_mean_rate = 0.4`
  - `stop_pressure_rate = 0.4`

Interpretation:

- these cases are not well described as pure perceptual failure
- in several cases, the object continuation is not tokenwise implausible
- but it still loses badly because `EOS now` is extremely short and therefore
  wins on total sequence probability

This is the cleanest current evidence that EOS-biased suppression is part of the
raw-text FN mechanism story.

## Current FN Review Surface

The selected FN cases now have a user-facing HTML review bundle:

- [review.html](/data/CoordExp/output/analysis/raw-text-coordinate-mechanism-fn-suppression-model-native-20260421/review_gallery/review.html)

Each card shows:

- baseline miss on the left
- best recovered oracle hit on the right
- GT category and image id
- oracle run label and IoU
- base-only and adapter stop-pressure readouts

The current heuristic split is:

- `both_models_stop_pressure`
  - `fn:25:3:person`
  - `fn:56:8:bowl`
- `mixed_stop_pressure`
  - `fn:57:6:umbrella`
- `weak_stop_pressure_signal`
  - `fn:10:1:train`
  - `fn:5:9:bicycle`

This review step is still important because some raw-text “FN” cases can be
visually ambiguous or annotation-sensitive. The gallery is the intended path for
human confirmation before hardening the final mechanism claim.

## Current Decision-Level Reading

### What is strongly supported

- raw-text coordinate continuity is real
- duplicate-burst bad basins are real
- pre-burst attraction depends in part on prefix geometry, not only object text
- EOS-vs-continue suppression is a real component of at least some raw-text FN
  cases

### What is not supported

- the idea that `coord_token` is needed merely to create locality or continuity
- the idea that all raw-text FN are simple perceptual misses

### What remains partially open

- how much of the adapter’s improvement comes from genuinely better GT
  attraction versus weaker duplicate collapse
- how much of the selected FN set will survive manual review as clean
  annotation-grounded suppression cases rather than ambiguous/unlabeled scenes

## Practical Next Step

Before locking the final five-question report, the most useful remaining action
is:

- review the 5 FN cards in the HTML gallery
- keep the clean labeled-GT suppression cases
- downgrade visually ambiguous cases before making the strongest final FN claim

That is the shortest path from “good mechanism evidence” to a clean
research-grade final conclusion.
