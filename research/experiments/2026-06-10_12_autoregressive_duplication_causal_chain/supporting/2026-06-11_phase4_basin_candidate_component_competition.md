# Phase 4 Basin Candidate Component Competition

Date: 2026-06-11

Scope: rerun the promoted candidate route/content probe with three source
components instead of only `duplicate_basin`:

```text
duplicate_basin
visual_near_ring
visual_far_background
```

This tests whether the three-gate disagreement cases are better explained by
neighboring visual buckets or background competition.

GPU scope: 4 A100 GPUs, devices `0,1,2,3`.

## Artifact

Promotion root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_basin_candidate_promotions_layer17_head1_shortlist
```

Per-candidate outputs:

```text
promoted_cases/<candidate_id>/route_content_probe_layer17_head1_components/
  paired_route_content_probe_rows.jsonl
  phase4_paired_route_content_probe_summary.json
```

Aggregate output:

```text
candidate_route_content_component_competition_summary/
  candidate_route_content_component_competition_rows.jsonl
  candidate_route_content_component_competition_summary.json
  candidate_route_content_component_competition_report.md
```

Coverage:

```text
candidate_count = 10
row_count = 90
components = duplicate_basin, visual_near_ring, visual_far_background
effect kinds = total_delta, route_delta_masked_values, value_delta_control_route
missing_candidates = []
```

Every candidate produced 9 rows.

## Total-Delta Competition

| candidate | class | duplicate p rec | near-ring p rec | far-bg p rec | best |
|---|---|---:|---:|---:|---|
| `record33_target25_non_target_desc_sub_envelope` | `mixed_or_weak` | 0.00022 | -0.00007 | 0.00016 | `duplicate_basin` |
| `record33_target25_original_duplicate_basin` | `aligned_route_and_value` | 0.03324 | 0.00331 | 0.00410 | `duplicate_basin` |
| `record33_target25_target_desc_sub_envelope` | `routed_positive_value_but_downstream_harm` | -0.00488 | -0.01488 | -0.00188 | `visual_far_background` |
| `record50_target9_component_row_9_person` | `route_without_value` | 0.00002 | 0.00072 | 0.00004 | `visual_near_ring` |
| `record50_target9_original_duplicate_basin` | `route_with_negative_target_projection` | 0.00784 | 0.00661 | -0.01574 | `duplicate_basin` |
| `record114_target4_component_row_6_person` | `route_with_negative_target_projection` | 0.01380 | -0.01224 | -0.00016 | `duplicate_basin` |
| `record114_target4_original_duplicate_basin` | `route_with_negative_target_projection` | 0.02113 | -0.00726 | -0.00083 | `duplicate_basin` |
| `record114_target4_target_row_bbox` | `value_without_route` | 0.00225 | 0.00047 | 0.00039 | `duplicate_basin` |
| `record114_target7_component_row_7_handbag` | `value_without_route` | -0.03128 | 0.00594 | -0.00002 | `visual_near_ring` |
| `record114_target7_original_duplicate_basin` | `route_without_value` | 0.00267 | -0.00011 | 0.00013 | `duplicate_basin` |

## Read

The component competition probe sharpens the three-gate story:

- The clean aligned case remains duplicate-basin dominated:
  `record33_target25_original_duplicate_basin` has duplicate-basin recovery
  `0.03324`, far above near-ring `0.00331` and far-background `0.00410`.
- The record 114 clock person-basin cases are also duplicate-basin dominated,
  despite negative target-coordinate value projection. This supports the
  earlier caveat: their positive route/content recovery is not explained by a
  simple target-coordinate value projection scalar, but it is still tied to the
  promoted person basin rather than near-ring or background.
- The target-local handbag case is the clearest component-competition reversal:
  `record114_target7_component_row_7_handbag` is strongly harmful when patched
  as duplicate basin (`-0.03128`), but near-ring is positive (`0.00594`). The
  target-local box appears to have useful surrounding context while the exact
  promoted region damages the active route.
- The record 50 target-local person case is still weak overall, but near-ring
  is the best component (`0.00072`) while duplicate-basin is effectively zero.
- The record 33 target-desc envelope remains harmful across all three
  components, with near-ring worst. Widening the target-desc envelope seems to
  introduce destructive local competition rather than merely adding useful
  target value.

This argues against a single "correct object box" intervention. The relevant
causal unit is sometimes the tight route-selected basin, sometimes the
surrounding local visual context, and sometimes neither component is helpful.

## Next Probe

The next attractive path is a finer spatial split for the two component
competition reversals:

- `record114_target7_component_row_7_handbag`: split the near-ring into
  inside-adjacent, outer ring, and overlap-with-person subregions.
- `record50_target9_component_row_9_person`: split target-local person context
  from surrounding person/surfboard overlap.

That would test whether the helpful component is true object context or an
overlap/occlusion scaffold outside the promoted bbox.

## Verification

Run coverage:

```text
per-candidate probe_row_count = 9
candidate_count = 10
aggregate row_count = 90
missing_candidates = []
```

The run used the existing checked-in paired route/content runner:

```text
scripts/analysis/run_autoregressive_duplication_phase4_paired_route_content_probe.py
```
