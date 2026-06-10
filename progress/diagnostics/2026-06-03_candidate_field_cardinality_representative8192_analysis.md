---
title: Candidate Field Cardinality Representative8192 Analysis
date: 2026-06-03
status: active-diagnostic
owner: codex
depends_on:
  - outputs/analysis/autoreg_object_rollout/ckpt3664_val200/candidate_field_cardinality_tomography_representative8192/summary.json
  - outputs/analysis/autoreg_object_rollout/ckpt3664_val200/candidate_field_cardinality_tomography_representative8192/x1_candidate_field_rows.jsonl
  - outputs/analysis/autoreg_object_rollout/ckpt3664_val200/candidate_field_cardinality_tomography_representative8192/phase_a_case_taxonomy_rows.jsonl
  - outputs/analysis/autoreg_object_rollout/ckpt3664_val200/candidate_field_cardinality_tomography_representative8192/analysis_representative8192_summary.json
  - outputs/analysis/autoreg_object_rollout/ckpt3664_val200/candidate_field_cardinality_tomography_representative8192/analysis_representative8192.md
  - outputs/analysis/autoreg_object_rollout/ckpt3664_purece/candidate_field_cardinality_tomography_representative8192/summary.json
  - outputs/analysis/autoreg_object_rollout/ckpt3664_purece/candidate_field_cardinality_tomography_representative8192/x1_candidate_field_rows.jsonl
  - outputs/analysis/autoreg_object_rollout/candidate_field_cardinality_comparisons/et_rmp_ce_vs_purece_representative8192/phase_a2_dual_checkpoint_analysis/phase_a2_summary.json
  - outputs/analysis/autoreg_object_rollout/candidate_field_cardinality_comparisons/et_rmp_ce_vs_purece_representative8192/phase_a2_dual_checkpoint_analysis/phase_a2_report.md
  - outputs/analysis/autoreg_object_rollout/candidate_field_cardinality_comparisons/et_rmp_ce_vs_purece_representative8192/phase_a2_dual_checkpoint_analysis/unmatched_review/unmatched_peak_summary.json
  - outputs/analysis/autoreg_object_rollout/candidate_field_cardinality_comparisons/et_rmp_ce_vs_purece_representative8192/phase_a2_dual_checkpoint_analysis/unmatched_review/gallery/index.md
---

# Candidate Field Cardinality Representative8192 Analysis

This note records the first post-run analysis of the representative
candidate-field cardinality tomography sample.  The scope is diagnostic:
checkpoint-3664, `representative8192` sampled GPU cases, x1 posterior probe at
the desc-first pre-x1 state.  It is not a full validation result and it does
not by itself establish attention-head causality.

## Artifact Scope

Checkpoint:

`outputs/stage1_2b/recursive_detection_ce_latest/compact_full_et_rmp_ce_support2_bsz16_4epoch_tokenrows_v2/compact-full-et-rmp-ce-support2-bsz16-4epoch-tokenrows-v2/v0-20260504-071356/checkpoint-3664`

Artifact root:

`outputs/analysis/autoreg_object_rollout/ckpt3664_val200/candidate_field_cardinality_tomography_representative8192`

Generated analysis artifacts:

- `analysis_representative8192_summary.json`
- `analysis_representative8192.md`
- `analysis_plots/coverage_fraction_by_pool.png`
- `analysis_plots/same_desc_count_cardinality.png`
- `analysis_plots/target_rank_a1_vs_non_a1.png`
- `analysis_plots/top_desc_a1_count_rate.png`

Run status:

- `summary.json`: `validation_status=ok`
- `manifest.json`: `validation_status=ok`
- `x1_candidate_field_rows.jsonl`: 8192 rows
- `phase_a_case_taxonomy_rows.jsonl`: 8192 rows

Pipeline headline eligibility:

`ineligible_missing_controls`

The missing controls are:

- `competitor_x1_control`
- `gt_x1_jitter`
- `mass_floor_sensitivity`
- `merge_radius_sensitivity`
- `wrong_desc_same_image`
- `wrong_image_same_desc`

Present/pass controls and slices:

- `same_desc_count_1_control=pass`
- `same_desc_count_2_control=pass`
- `p_cond_vs_coord_vocab_mass=pass`
- `x1_projection_collision_slice=present`

## Taxonomy Semantics

`A1_cardinality_collapse` is assigned when the x1 probe is valid, the coordinate
channel has enough probability mass, there is no unresolved x1 projection
collision, and the merged candidate modes do not cover all annotated same-desc
instances.

Operationally, the current taxonomy uses:

- `candidate_modes_cover_target = covered_count >= same_desc_count`
- `projection_collision_unresolved -> unassigned_or_inconclusive`
- `candidate_modes_cover_target=false -> A1_cardinality_collapse`

Thus A1 should be read as an x1 candidate-field coverage diagnosis, not as a
direct proof that an attention component caused the failure.

## Headline Counts

| Metric | Value |
| --- | ---: |
| Valid rows | 8192 |
| A1 cardinality-collapse rows | 2690 |
| A1 rate | 32.837% |
| Unassigned or inconclusive rows | 5502 |

The strongest observed pattern is not low coordinate probability mass.  The
median coordinate-vocabulary mass remains high in both A1 and non-A1 rows:

| Bucket | Rows | Median coord mass | Median target x1 rank | Median p_gt_cond | Mean coverage fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| `A1_cardinality_collapse` | 2690 | 0.9747 | 96.5 | 0.0012 | 0.3377 |
| `unassigned_or_inconclusive` | 5502 | 0.9853 | 7.0 | 0.0452 | 0.9117 |

This says the coordinate channel is generally active, but in A1 cases the GT
instance's x1 becomes much lower-ranked and the merged x1 candidate modes cover
only a small fraction of annotated same-desc instances.

## Pool-Role Breakdown

| Pool role | Rows | A1 rows | A1 rate | Mean same-desc GT count | Mean merged peaks | Mean coverage fraction | Median target rank | Median p_gt_cond | Collision rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `headline_crowded` | 1996 | 1206 | 0.604 | 6.230 | 2.517 | 0.351 | 151 | 0.0011 | 0.374 |
| `same_desc_count_1_control` | 4891 | 384 | 0.079 | 1.000 | 1.341 | 0.921 | 6 | 0.0494 | 0.000 |
| `same_desc_count_2_control` | 1305 | 1100 | 0.843 | 2.000 | 1.739 | 0.550 | 24 | 0.0072 | 0.044 |

The single-instance sanity surface is mostly healthy.  The two-instance surface
already shows a strong coverage drop: on average it covers about 1.10 of 2
annotated instances, with median target rank 24 rather than 6.

## Same-Desc Count Breakdown

| Same-desc count | Rows | A1 rows | A1 rate | Mean merged peaks | Mean coverage fraction | Mean peak deficit | Median target rank | Collision rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4891 | 384 | 0.079 | 1.341 | 0.921 | -0.341 | 6 | 0.000 |
| 2 | 1305 | 1100 | 0.843 | 1.739 | 0.550 | 0.261 | 24 | 0.044 |
| 3 | 606 | 491 | 0.810 | 2.101 | 0.438 | 0.899 | 75.5 | 0.147 |
| 4-5 | 541 | 407 | 0.752 | 2.303 | 0.361 | 2.100 | 138 | 0.222 |
| 6+ | 849 | 308 | 0.363 | 2.951 | 0.282 | 6.750 | 233 | 0.634 |

The A1 rate drops for the `6+` bucket because many highly crowded rows are
removed from A1 by `projection_collision_unresolved`.  The coverage fraction
still decreases as same-desc cardinality grows.

## Projection Collision Slice

| x1 projection collision | Rows | A1 rows | A1 rate | Mean same-desc GT count | Mean merged peaks | Mean coverage fraction | Median target rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| false | 7387 | 2690 | 0.364 | 1.828 | 1.598 | 0.759 | 9 |
| true | 805 | 0 | 0.000 | 7.990 | 2.540 | 0.396 | 135 |

The current A1 evidence is not caused by projection-collision rows.  Collision
rows are explicitly kept out of A1 by the taxonomy and should be analyzed as a
separate ambiguity slice.

## Top Descriptions By A1 Count

| Desc | Rows | A1 rows | A1 rate | Mean same-desc GT count | Mean merged peaks | Mean coverage fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| person | 1449 | 628 | 0.433 | 4.204 | 1.858 | 0.577 |
| car | 274 | 139 | 0.507 | 3.653 | 2.069 | 0.565 |
| chair | 343 | 137 | 0.399 | 2.907 | 2.047 | 0.635 |
| cup | 205 | 86 | 0.420 | 2.185 | 1.698 | 0.702 |
| bottle | 212 | 75 | 0.354 | 2.415 | 1.840 | 0.751 |
| dining table | 284 | 71 | 0.250 | 1.423 | 1.356 | 0.792 |
| bench | 178 | 70 | 0.393 | 1.584 | 2.107 | 0.707 |
| handbag | 168 | 66 | 0.393 | 1.899 | 2.494 | 0.687 |
| bowl | 176 | 57 | 0.324 | 1.903 | 1.841 | 0.741 |
| truck | 146 | 57 | 0.390 | 1.760 | 1.616 | 0.738 |
| traffic light | 106 | 50 | 0.472 | 3.283 | 1.849 | 0.603 |

`person` dominates by count, but the signal is not person-only.  The same
pattern appears for vehicles, chairs, small objects, and tableware.

## Evidence Boundaries

Supported by this run:

- At the desc-first pre-x1 state, the model often produces an x1 candidate
  field that under-covers annotated same-desc instances.
- The under-coverage appears already at same-desc count 2, not only in extreme
  crowded scenes.
- A1 rows are characterized by low target x1 rank and low `p_gt_cond`, while
  coordinate-vocabulary mass remains high.
- Projection-collision rows are not the source of the A1 count because they are
  excluded from A1 by taxonomy.

Not established by this run:

- Whether the bottleneck is caused by a specific attention head or layer.
- Whether missing candidate modes are due to visual encoder salience,
  decoder-side compression, training trajectory bias, or prompt-prefix state.
- Whether the same pattern holds at full indexed scale.
- Whether changing the training objective would improve recall without
  increasing harmful duplication.

## Immediate Follow-Up Controls

Before promoting this into a stronger mechanism claim, run the missing controls
on the same representative8192 surface:

1. `merge_radius_sensitivity`
2. `mass_floor_sensitivity`
3. `gt_x1_jitter`
4. `competitor_x1_control`
5. `wrong_desc_same_image`
6. `wrong_image_same_desc`

The highest-priority analysis slice is non-collision A1 rows with
`same_desc_gt_count_annotated >= 2`, especially `person`, `car`, `chair`,
`traffic light`, `cup`, and `bottle`.

## Pure-CE Contrast Plan

The next approved comparison is an apples-to-apples `representative8192` probe
against the compact-full Stage-1 2B pure-CE/random-SFT checkpoint:

```text
/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_random_sft_bsz1_accum16_4epoch_tokenrows_v4_chatfix_max12k/compact-full-random-sft-bsz1-accum16-4epoch-tokenrows-v4-chatfix-max12k/v0-20260506-021259/checkpoint-3664
```

This checkpoint shares the same compact-full template, coord-token `xyxy`
surface, token rows, random-permutation object ordering, and COCO
`rescale_32_1024_bbox_max60` train/val JSONLs as the ET-RMP-CE reference.  Its
resolved objective is `sft` / `random_order_sft` with trie weights set to zero,
so it is the best available local pure-CE contrast for candidate-field shape.

Pure-CE artifact root:

```text
/data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_purece/candidate_field_cardinality_tomography_representative8192
```

Comparison question:

Does pure CE expose fewer, equal, or more desc-conditioned x1 modes than
ET-RMP-CE at the pre-x1 state?

Required matched settings:

- `sampling.max_cases = 8192`
- `sampling.num_shards = 8`
- `sampling.seed = 3664`
- `peak.absolute_mass_floor = 0.002`
- `peak.relative_floor = 0.10`
- `peak.primary_merge_radius = 24`
- `peak.gt_x1_neighborhood_radius = 24`
- `peak.raw_topk_k = 32`

Primary contrast metrics:

- `multi_peak_row_rate`
- `mean_peak_count`
- `median_peak_count`
- `valid_peak_share`
- `valid_peak_mass_share`
- `coverage_fraction`
- `A1_cardinality_collapse` rate
- target x1 rank
- `p_gt_cond`

The contrast remains diagnostic evidence.  It should not be read as a fully
isolated objective ablation because effective batch shape, normalization, and
state weighting differ between the two runs.

Launch status:

- Config:
  `configs/analysis/candidate_field_cardinality_tomography/ckpt3664_purece_representative8192.yaml`
- Tmux session:
  `candidate_field_cardinality_purece_representative8192_ckpt3664`
- Log root:
  `logs/candidate_field_cardinality_purece_representative8192_ckpt3664`
- `probe_plan_summary.json`: `gpu_probe_planned_cases=8192`,
  `num_shards=8`, `sampling_policy_id=stratified_round_robin_v1`
- At launch, 8 shard processes were alive and GPUs 0-7 were each assigned one
  shard.  This is an analysis probe, not production training.

Completion status:

- Final status: complete.
- Manifest validation: `validation_status=ok`.
- `x1_candidate_field_rows.jsonl`: 8192 rows.
- `phase_a_case_taxonomy_rows.jsonl`: 8192 rows.
- `summary.json`: present.
- `report.md`: present.

Direct comparison artifact:

```text
/data/CoordExp/outputs/analysis/autoreg_object_rollout/candidate_field_cardinality_comparisons/et_rmp_ce_vs_purece_representative8192
```

Initial matched-run comparison:

| Metric | ET-RMP-CE | pure CE | pure - ET |
| --- | ---: | ---: | ---: |
| A1 rate | 0.3284 | 0.2612 | -0.0671 |
| multi-peak row rate | 0.2870 | 0.4561 | +0.1691 |
| mean peak count | 1.6909 | 2.7091 | +1.0182 |
| mean coverage fraction | 0.7232 | 0.7981 | +0.0749 |
| median target x1 rank | 10 | 13 | +3 |
| median `p_gt_cond` | 0.0286 | 0.0176 | -0.0111 |
| valid peak share | 0.6714 | 0.5897 | -0.0817 |
| valid peak mass share | 0.9046 | 0.8912 | -0.0135 |

The sampled cases match exactly between the two runs.  The pure-CE checkpoint
exposes a broader x1 candidate field than ET-RMP-CE on this surface: more
multi-peak rows, more peaks, lower A1, and higher annotated coverage.  However,
some of that extra breadth is unmatched under the annotated COCO universe, as
shown by lower valid-peak share and slightly lower valid-peak mass share.  This
is diagnostic evidence, not yet a headline causal claim.

## Phase A2 Dual-Checkpoint Analysis

Phase A2 deepened the comparison using the exact matched `representative8192`
rows from ET-RMP-CE and pure CE.

Generated script:

```text
scripts/analysis/candidate_field_cardinality_tomography/phase_a2_dual_checkpoint_analysis.py
```

Generated artifact root:

```text
outputs/analysis/autoreg_object_rollout/candidate_field_cardinality_comparisons/et_rmp_ce_vs_purece_representative8192/phase_a2_dual_checkpoint_analysis
```

Generated artifacts:

- `phase_a2_summary.json`
- `phase_a2_report.md`
- `plots/paired_a1_transitions.png`
- `plots/coverage_delta_by_count.png`
- `plots/valid_vs_unmatched_peak_delta.png`
- `plots/sensitivity_grid_a1_rate.png`
- `plots/plot_summary.json`

Verification:

- matched rows: 8192
- sample exact order: `true`
- valid x1 radius: 24
- script compile: passed
- report and summary: readable

Headline paired metrics:

| Metric | ET-RMP-CE | pure CE | pure - ET |
| --- | ---: | ---: | ---: |
| A1 rate | 0.3284 | 0.2612 | -0.0671 |
| multi-peak row rate | 0.2870 | 0.4561 | +0.1691 |
| mean peak count | 1.6909 | 2.7091 | +1.0182 |
| mean valid peak count | 1.1353 | 1.5975 | +0.4623 |
| mean unmatched peak count | 0.5557 | 1.1116 | +0.5559 |
| total valid peak share | 0.6714 | 0.5897 | -0.0817 |
| mean coverage fraction | 0.7232 | 0.7981 | +0.0749 |
| median target rank | 10 | 13 | +3 |
| median `p_gt_cond` | 0.0286 | 0.0176 | -0.0111 |

Paired A1 transition matrix:

| Transition | Rows |
| --- | ---: |
| both A1 | 2035 |
| ET-RMP-CE only A1 | 655 |
| pure CE only A1 | 105 |
| neither A1 | 5397 |

`ET-RMP-CE only A1` means ET-RMP-CE under-covered the row by the A1 rule while
pure CE did not.  `pure CE only A1` means the reverse.

Pure-minus-ET deltas by same-desc count:

| Same-desc bucket | Rows | Delta peaks | Delta valid peaks | Delta unmatched peaks | Delta coverage | Pure higher coverage rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| count1 | 4891 | +0.3173 | +0.0472 | +0.2701 | +0.0027 | 0.0170 |
| count2 | 1305 | +1.7165 | +0.6820 | +1.0345 | +0.1655 | 0.3441 |
| count3 | 606 | +2.1155 | +0.9356 | +1.1799 | +0.1843 | 0.4538 |
| count4_5 | 541 | +2.6007 | +1.3420 | +1.2588 | +0.2197 | 0.5730 |
| count6_plus | 849 | +2.1908 | +1.6172 | +0.5736 | +0.1816 | 0.5760 |

`Delta valid peaks` is peak-side validity: multiple peaks can fall near the
same annotated GT x1.  `Delta coverage` is the instance-side signal and is the
safer recall-proxy field.

Top-32 approximate sensitivity:

- sensitivity grid size: 36 policy cells
- pure CE had lower approximate A1 in 36/36 cells
- pure CE had higher approximate multi-peak rate in 36/36 cells
- pure CE had higher approximate mean peak count in 36/36 cells
- pure CE had higher approximate mean coverage in 36/36 cells

The sensitivity rows are labeled `top32_approx` because they recompute peaks
from stored top-32 coordinate bins rather than rerunning the model or using the
full coordinate posterior.

Descs with largest pure-CE valid-peak gains among descs with at least 50 rows:

| Desc | Rows | Mean same-desc count | Delta valid peaks | Delta unmatched peaks | Delta coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| traffic light | 106 | 3.2830 | +1.1509 | +1.3774 | +0.1838 |
| banana | 60 | 4.8000 | +0.9500 | +0.6167 | +0.1569 |
| person | 1449 | 4.2036 | +0.9096 | +0.6950 | +0.1344 |
| giraffe | 73 | 1.8630 | +0.8630 | +1.1918 | +0.1703 |
| book | 115 | 4.6609 | +0.7913 | +0.6522 | +0.1282 |

Evidence-backed interpretations:

1. The strict local-singleton explanation is weakened.  At `pre_x1`, pure CE is
   not more single-peaked; it is broader than ET-RMP-CE on the same rows.
2. Broader candidate fields are not automatically clean recall.  Pure CE adds
   both annotated-valid peaks and COCO-unmatched peaks.
3. Unmatched peaks remain an ambiguity bucket, not automatic hallucinations,
   because COCO annotation is incomplete.
4. Candidate compression remains in crowded rows.  Pure CE improves coverage,
   but it still does not behave like an exhaustive same-desc instance ledger in
   high-cardinality rows.
5. ET-RMP-CE appears to compact the desc-conditioned pre-x1 candidate field
   relative to pure CE on this diagnostic surface.  This may coexist with
   better final rollout behavior, so candidate-field breadth and final decode
   stability must remain separate claims.
6. This is not a pure objective ablation.  Normalization, state weighting, and
   effective batch shape also differ between the two checkpoints.

## Unmatched Peak Manual Review Sidecar

The Phase A2 `unmatched` bucket was materialized as a manual-review sidecar so
it can be separated later into likely COCO-unlabeled objects, hallucinations,
duplication, ambiguous cases, or other categories.

Generated script:

```text
scripts/analysis/candidate_field_cardinality_tomography/unmatched_peak_review_gallery.py
```

Generated artifact root:

```text
outputs/analysis/autoreg_object_rollout/candidate_field_cardinality_comparisons/et_rmp_ce_vs_purece_representative8192/phase_a2_dual_checkpoint_analysis/unmatched_review
```

Generated artifacts:

- `unmatched_peak_rows.jsonl`
- `unmatched_peak_summary.json`
- `unmatched_peak_review.md`
- `manual_review_template.csv`
- `gallery/gallery_rows.jsonl`
- `gallery/index.md`
- `gallery/images/*.jpg`

Review fields:

- `review_status`
- `manual_label`
- `manual_notes`

Suggested manual labels:

- `unlabeled_object`
- `hallucination`
- `duplication`
- `ambiguous`
- `other`

Visualization layout and legend:

- upper image panel: pure-CE x1 candidate field on the original image
- lower image panel: ET-RMP-CE x1 candidate field on the same original image
- both image panels share the same annotated same-desc GT boxes
- bottom panel: shared 0..1000 x1 coordinate ruler with colored triangle
  markers
- right panel: metadata, legend, and exact peak coordinate table
- green boxes and bottom-edge markers: annotated same-desc GT boxes / GT x1
  edges
- red vertical bands/markers: pure-CE unmatched x1 peaks
- orange vertical lines/markers: pure-CE annotated-valid x1 peaks
- blue vertical lines/markers: ET-RMP-CE unmatched x1 peaks
- cyan vertical lines/markers: ET-RMP-CE annotated-valid x1 peaks

Important visual boundary: these probes only locate the candidate `x1`
left-boundary coordinate.  They do not contain `y1/x2/y2`, so the visualized
signal is a vertical candidate stripe rather than a full predicted box.

Summary:

| Metric | Value |
| --- | ---: |
| review rows | 3554 |
| rows with pure unmatched | 3462 |
| rows with pure new unmatched | 2576 |
| total pure unmatched peaks | 9106 |
| total pure new unmatched peaks | 5711 |
| total ET unmatched peaks | 4552 |
| rendered gallery images | 128 |

Top descs by unmatched review rows:

| Desc | Rows |
| --- | ---: |
| person | 717 |
| chair | 189 |
| car | 145 |
| handbag | 118 |
| bottle | 102 |
| cup | 99 |
| bench | 86 |
| dining table | 85 |
| bowl | 84 |
| backpack | 83 |
| traffic light | 69 |
| book | 69 |
| truck | 69 |

`unmatched` remains annotation-relative: it means no annotated same-desc GT x1
lies within the configured radius.  It is not a hallucination claim until the
manual labels are filled.
