---
title: 2B FN-Factor Artifact Bundle Summary
date: 2026-03-17
status: complete
topics: [stage2, 2b, fn-analysis, artifact-summary, review-queue]
tags: [2b, diagnostics, rollout, fn, hard-subset, artifact-bundle]
summary: Operator summary of the completed 2B FN-factor artifact bundle after the repaired Hard-16 prefix rerun. This note explains what each artifact is for, which conclusions are stable, and how to read the review queues without confusing them for aggregate result tables.
---

# 2B FN-Factor Artifact Bundle Summary (2026-03-17)

This note compresses the current 2B FN-factor artifact bundle into one operator-facing readout.

Primary artifacts:

- refreshed Hard-16 full-factor report:
  `output/analysis/rollout-fn-factor-2b-hard16-full-20260317/report.md`
- Hard-32 extension report:
  `output/analysis/rollout-fn-factor-2b-hard32-extension-20260317/report.md`
- combined result note:
  `progress/diagnostics/stage2_2b_fn_factor_results_2026-03-17.md`
- Hard-16 review queue:
  `output/analysis/rollout-fn-factor-2b-hard16-full-20260317/review_queue.jsonl`
- Hard-32 review queue:
  `output/analysis/rollout-fn-factor-2b-hard32-extension-20260317/review_queue.jsonl`

Fixed checkpoint pair:

- original:
  `output/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate_merged-1332`
- A-only:
  `output/stage2_ab/2b_1024/a_only_iter1/merged_ckpt-900`

Fixed datasets:

- train:
  `public_data/coco/rescale_32_1024_bbox_max60/train.coord.jsonl`
- val:
  `public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl`

## What Each Artifact Is For

### Refreshed Hard-16 full-factor report

Use the refreshed Hard-16 report as the primary causal artifact.

This is the only layer in the bundle that now combines:

- deterministic image-only greedy rollout
- union-of-`K` sampling
- repaired prefix-order intervention
- broken / switched prefix stress cells
- extended-length controls

This is the authoritative source for:

- the deterministic `A-only > original` result
- the repaired prefix-sensitive recovery result
- the continued null result for length extension

### Hard-32 extension report

Use the Hard-32 report as the scale-up / replication artifact for the stable image-only conclusions.

It covers:

- deterministic image-only greedy rollout
- union-of-`K` sampling
- extended-length controls

It does **not** contain the repaired prefix rerun. So treat it as the true `32`-image extension for baseline/sampling/length only, not for the latest prefix read.

### Combined result note

Use the combined result note for the current human-readable interpretation:

- `progress/diagnostics/stage2_2b_fn_factor_results_2026-03-17.md`

That is now the best single file to read if you want the up-to-date argument rather than the raw tables.

### Review queues

Use the review queues as qualitative audit entrypoints.

They are curated object-level triage lists pointing back to baseline sample directories and overlay folders. They are **not** full aggregate result tables.

## Stable Conclusions Across the Bundle

### 1) A-only is better in deterministic greedy rollout

This remains the most stable result.

Hard-16 deterministic hits:

- train:
  A-only `252 / 838` vs original `225 / 838`
- val:
  A-only `208 / 651` vs original `185 / 651`

Hard-32 deterministic hits:

- train:
  A-only `563 / 1667` vs original `514 / 1667`
- val:
  A-only `449 / 1210` vs original `402 / 1210`

So `A-only` is genuinely better at greedy rollout on both splits and both subset sizes.

### 2) Many FN are decode-selection misses rather than proven incapacity

Union-of-`K` still recovers a substantial additional block of GT objects.

Hard-16 image-only recoverable coverage
(`deterministic_hit + decode_selection_miss + length_bias_miss`):

- train:
  A-only `357` vs original `333`
- val:
  A-only `305` vs original `323`

Hard-32 image-only recoverable coverage:

- train:
  A-only `800` vs original `761`
- val:
  A-only `618` vs original `625`

So the pre-existing split asymmetry survives:

- on train, `A-only` stays better even after sampling
- on val, original still keeps a slightly larger pool of sampling-recoverable misses

### 3) Repaired prefix continuation is now a real recovery channel

This is the key change relative to the earlier same-day read.

The refreshed Hard-16 recovery summaries now report:

- train / A-only:
  `prefix_sensitive_miss = 30`
- train / original:
  `34`
- val / A-only:
  `15`
- val / original:
  `15`

So the repaired prefix intervention now recovers a real continuation-sensitive block of GT objects.

### 4) Prefix state matters, but train-order is not the clear winner

Across the `94` true `prefix_sensitive_miss` rows in the refreshed Hard-16 recovery tables:

- train-order recovered `60`
- random-order recovered `60`
- self-prefix recovered `40`

and the overlap structure is mixed:

- both train-order and random-order:
  `39`
- self-prefix only:
  `26`
- all three:
  `13`
- train-order only:
  `8`
- random-order only:
  `7`

So the repaired bundle supports:

- prefix state matters

but it does **not** strongly support:

- “training serialization order is the dominant causal prior”

### 5) Longer rollout length still does not explain the FN

Across both reports:

- `length_bias_miss = 0` for every checkpoint × split table

So the bundle still does not support the simple EOS / max-length explanation.

## Hard-16 vs Hard-32 Relationship

The two reports are complementary.

- Hard-16 is the full experiment surface with the repaired prefix result.
- Hard-32 is the true extension check for deterministic baseline, sampling, and length.

One operational caveat matters:

- the refreshed Hard-16 rerun used `hard16_size = 16` and `hard32_size = 16`

So the `Hard-32` rows inside that refreshed `report.md` are bookkeeping duplicates of the 16-image cohort, not an independent extension.

For real `32`-image numbers, use:

- `output/analysis/rollout-fn-factor-2b-hard32-extension-20260317/report.md`

## Review Queue Read

Each review queue has `96` curated rows with the same structural balance:

- `32` `deterministic_hit`
- `32` `decode_selection_miss`
- `32` `persistent_unrecovered`

and each queue is evenly split across:

- `train × original`
- `train × A-only`
- `val × original`
- `val × A-only`

with `8` rows per status bucket for each split/checkpoint pair.

What the queues are good for:

- jumping into representative deterministic hits, decode-selection misses, and persistent unrecovered objects
- checking whether a “miss” looks like real incapacity, annotation mismatch, or evaluator-policy disagreement
- locating overlays and baseline sample directories quickly

What stands out:

- both queues are dominated by the same hardest recurring scenes
- the strongest recurring val scene is:
  `images/val2017/000000303566.jpg`
- the strongest recurring train scene is:
  `images/train2017/000000022484.jpg`
- annotation mismatch candidates are rare:
  `7 / 96`
- semantic confusion candidates are `0 / 96`

One important limitation:

- the review queues do **not** contain a dedicated `prefix_sensitive_miss` status bucket

Instead, they expose repaired prefix signal via:

- `prefix_recovered_by_cells`

So use the recovery summaries for authoritative prefix counts, and the review queues for examples.

## Practical Reading Order

If you want the shortest path from headline conclusion to artifact inspection:

1. `progress/diagnostics/stage2_2b_fn_factor_results_2026-03-17.md`
2. refreshed Hard-16 `report.md`
3. Hard-32 extension `report.md`
4. Hard-16 `recovery/*.summary.json`
5. the two `review_queue.jsonl` files

If you want the repair-specific follow-up and the still-pending random-order training ablation state, then continue to:

- `progress/diagnostics/stage2_2b_prefix_random_order_followup_2026-03-17.md`
