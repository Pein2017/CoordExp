# Phase 4 Basin Candidate Value Projection Summary

Date: 2026-06-11

Scope: run the promoted basin-split candidates through the layer 17 head 1
value-basin projection probe, then join those rows with the candidate
route/content and Q/K summaries. This tests whether a promoted basin's
attention-weighted value contribution is aligned with the current coordinate
slot.

GPU scope: 4 A100 GPUs, devices `0,1,2,3`.

## Artifact

Promotion root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_basin_candidate_promotions_layer17_head1_shortlist
```

Per-candidate value projection outputs:

```text
promoted_cases/<candidate_id>/value_basin_projection_layer17_head1/
  value_basin_projection_rows.jsonl
  phase4_value_basin_projection_summary.json
```

Aggregate output:

```text
candidate_value_basin_projection_summary/
  candidate_value_basin_projection_summary_rows.jsonl
  candidate_value_basin_projection_summary.json
  candidate_value_basin_projection_summary_report.md
```

Coverage:

```text
candidate_count = 10
value_basin_row_count = 30
patch components = duplicate_basin, visual_near_ring, visual_far_background
missing_candidates = []
```

Every promoted candidate produced exactly three value-basin projection rows.

## Joined Duplicate-Basin Readout

| candidate | rec | row | kind | route p rec | qk attn | qk score | value target delta | value r4 margin | value rank |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| `record33_target25_non_target_desc_sub_envelope` | 33 | 25 | `non_target_desc_sub_envelope` | 0.00022 | 0.00011 | 0.92191 | -0.00003 | -0.00003 | 736 |
| `record33_target25_original_duplicate_basin` | 33 | 25 | `original_duplicate_basin` | 0.03324 | 0.57199 | 3.19067 | 0.53303 | 0.53787 | 103 |
| `record33_target25_target_desc_sub_envelope` | 33 | 25 | `target_desc_sub_envelope` | -0.00488 | 0.53955 | 3.72939 | 0.50520 | 0.50979 | 103 |
| `record50_target9_component_row_9_person` | 50 | 9 | `component_row_9_person` | 0.00002 | 0.86539 | 4.96628 | -0.13835 | -0.13983 | 472 |
| `record50_target9_original_duplicate_basin` | 50 | 9 | `original_duplicate_basin` | 0.00784 | 0.12291 | -0.37891 | -0.02742 | -0.02772 | 406 |
| `record114_target4_component_row_6_person` | 114 | 4 | `component_row_6_person` | 0.01380 | 0.93253 | 2.48314 | -0.36625 | -0.36991 | 435 |
| `record114_target4_original_duplicate_basin` | 114 | 4 | `original_duplicate_basin` | 0.02113 | 0.84757 | 2.41282 | -0.35870 | -0.36228 | 433 |
| `record114_target4_target_row_bbox` | 114 | 4 | `target_row_bbox` | 0.00225 | -0.01953 | 3.77618 | 0.04432 | 0.04473 | 482 |
| `record114_target7_component_row_7_handbag` | 114 | 7 | `component_row_7_handbag` | -0.03128 | -0.30703 | -1.90779 | 0.21489 | 0.21675 | 375 |
| `record114_target7_original_duplicate_basin` | 114 | 7 | `original_duplicate_basin` | 0.00267 | 0.45275 | 5.15198 | -0.14487 | -0.14652 | 402 |

## Mechanistic Read

The value projection probe confirms that the layer-17/head-1 candidate story is
not reducible to one scalar.

The strongest clean alignment case is still record 33:

```text
record33 row25 original_duplicate_basin:
  route/content recovery = 0.03324
  Q/K attention delta = 0.57199
  value target delta = 0.53303
  value radius-4 margin = 0.53787
```

But the disagreement cases are more mechanistically important:

- `record33_target25_target_desc_sub_envelope` has strong Q/K and strong
  value-basin projection, but negative route/content recovery. This suggests
  that widening the target-desc region may introduce competing attention/value
  structure even when the aggregate value delta points toward the coordinate.
- `record50_target9_component_row_9_person` has very strong Q/K attention
  delta (`0.86539`) but negative value target delta (`-0.13835`) and nearly
  zero route recovery. This is the cleanest "routing without useful coordinate
  content" case.
- `record114_target4_original_duplicate_basin` and
  `record114_target4_component_row_6_person` have strong Q/K attention and
  positive route/content recovery, but their value projection is negative for
  the clock target. The positive downstream recovery there is therefore not a
  simple coordinate-target value projection story; it may involve full
  residual/logit effects outside this duplicate-basin projection scalar.
- `record114_target4_target_row_bbox` has the only positive value projection
  among the clock candidates, but Q/K attention delta is negative and
  route/content recovery is weak. This is "useful content not routed."
- `record114_target7_component_row_7_handbag` has positive value projection for
  the handbag target, but negative Q/K attention and strongly negative
  route/content recovery. This is the sharpest example where target-local
  content exists but the intervention damages the active route.

The emerging picture is a three-gate mechanism:

```text
1. Q/K routing must place attention mass on a candidate basin.
2. The selected basin's value contribution must be coordinate-aligned.
3. The full route/content intervention must integrate that contribution without
   destructive competition from surrounding source buckets or residual context.
```

This is why target-local boxes are not automatically better, and why original
route-selected basins can remain more causally useful even when they are
semantically uncomfortable. The head is not simply "looking at the correct
object"; it is routing through a basin whose value content and surrounding
competition jointly determine the coordinate slot.

## Next Probe

The highest-yield next step is to turn the three-gate readout into a compact
case taxonomy:

- `aligned_route_and_value`: record 33 original basin;
- `route_without_value`: record 50 target-local person;
- `value_without_route`: record 114 clock target bbox and handbag target-local
  component;
- `route_with_negative_target_projection`: record 114 person basins.

For those cases, the next causal probe should compare component competition
inside `duplicate_basin` versus `visual_near_ring`, because several harmful
cases have value signal in one component while route/content recovery is
dominated by another.

## Verification

GPU run:

```text
per-candidate value_basin_projection_row_count = 3
candidate_count = 10
aggregate value_basin_row_count = 30
missing_candidates = []
```

The value projection outputs reuse the already verified promoted-candidate Q/K
prep rows:

```text
promoted_cases/<candidate_id>/qk_token_windows.jsonl
promoted_cases/<candidate_id>/region_rows.jsonl
```

The run used the existing checked-in value-basin projection runner:

```text
scripts/analysis/run_autoregressive_duplication_phase4_value_basin_projection_shard.py
```
