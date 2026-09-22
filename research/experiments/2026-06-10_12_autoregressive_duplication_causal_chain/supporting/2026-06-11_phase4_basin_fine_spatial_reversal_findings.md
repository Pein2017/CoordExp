# Phase 4 Basin Fine-Spatial Reversal Findings

Date: 2026-06-11

Scope: two promoted basin candidates selected from the layer-17/head-1 component-competition panel. This is a mechanistic localization probe, not a validation-scale estimate.

## Inputs

- Source promotion root: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_basin_candidate_promotions_layer17_head1_shortlist`
- Fine-spatial promotion root: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_basin_fine_spatial_promotions_layer17_head1_reversals`
- Fine-spatial route/content summary: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_basin_fine_spatial_promotions_layer17_head1_reversals/fine_spatial_route_content_summary`
- Parent component comparison: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_basin_candidate_promotions_layer17_head1_shortlist/candidate_route_content_component_competition_summary/candidate_route_content_component_competition_rows.jsonl`

Selected source candidates:

- `record114_target7_component_row_7_handbag`
- `record50_target9_component_row_9_person`

Each source candidate was split into six norm1000 fine regions: `expanded_1p25`, `expanded_1p50`, `top_band`, `bottom_band`, `left_band`, and `right_band`. Each fine region was promoted as `region_kind=duplicate_basin` and re-run through the paired route/content probe at layer 17, head 1 with `effect_kinds=total_delta,route_delta_masked_values,value_delta_control_route`.

## Artifact Checks

- Fine candidates materialized: 12.
- Probe rows reduced: 36.
- GPU execution: all 12 one-row probes completed successfully on GPUs `0,1,2,3`; launcher failure count was 0.
- Contract audit: every fine candidate has exactly one `paired_manifest_rows.jsonl`, one `qk_token_windows.jsonl`, and one `region_rows.jsonl`; all boxes are valid clamped norm1000 boxes; all promoted regions preserve `region_kind=duplicate_basin`.

## Main Read

The fine-spatial panel argues against a simple "the whole local basin is uniformly harmful" explanation. The parent candidate aggregate can be strongly harmful, while margin-scale subregions around the same candidate split into positive and negative directions. This makes the basin look more like a structured coordinate-slot attraction field with directional or boundary-sensitive pockets than a homogeneous visual-region value source.

For `record114_target7_component_row_7_handbag`, the parent duplicate-basin `total_delta` was strongly negative:

- Parent `duplicate_basin`: `prob_recovery_from_masked=-0.031281640`
- Parent `visual_near_ring`: `prob_recovery_from_masked=0.005942170`
- Parent `visual_far_background`: `prob_recovery_from_masked=-0.000018306`

Fine split for `total_delta`:

| fine_spatial_kind | prob_recovery | r4_recovery | token_count | bbox |
| --- | ---: | ---: | ---: | --- |
| `right_band` | -0.002841210 | -0.019279581 | 9 | `[753, 320, 812, 748]` |
| `bottom_band` | -0.000376381 | -0.002417527 | 6 | `[519, 748, 753, 855]` |
| `expanded_1p25` | 0.001330170 | 0.009283943 | 65 | `[460, 213, 812, 855]` |
| `expanded_1p50` | 0.002482514 | 0.015894322 | 119 | `[402, 106, 870, 962]` |
| `left_band` | 0.002094205 | -0.000339314 | 9 | `[460, 320, 519, 748]` |
| `top_band` | 0.002080211 | 0.001892559 | 6 | `[519, 213, 753, 320]` |

Interpretation: the harmful signal does not simply live in the expansion halo. The right-side band remains harmful, but the larger expansion boxes flip positive. This looks like spatial directionality or boundary interaction rather than a monotonic "more nearby visual tokens = more duplicate attraction" rule.

For `record50_target9_component_row_9_person`, the parent duplicate-basin effect was near zero:

- Parent `duplicate_basin`: `prob_recovery_from_masked=0.000017685`
- Parent `visual_near_ring`: `prob_recovery_from_masked=0.000717731`
- Parent `visual_far_background`: `prob_recovery_from_masked=0.000039322`

Fine split for `total_delta`:

| fine_spatial_kind | prob_recovery | r4_recovery | token_count | bbox |
| --- | ---: | ---: | ---: | --- |
| `expanded_1p25` | -0.002293150 | -0.014335102 | 45 | `[548, 114, 813, 796]` |
| `bottom_band` | -0.000299192 | 0.000066034 | 3 | `[592, 682, 769, 796]` |
| `right_band` | -0.000230092 | -0.000974707 | 6 | `[769, 228, 813, 682]` |
| `top_band` | -0.000230852 | 0.000435874 | 6 | `[592, 114, 769, 228]` |
| `left_band` | 0.000776855 | 0.001187578 | 6 | `[548, 228, 592, 682]` |
| `expanded_1p50` | 0.007959660 | 0.052730737 | 84 | `[504, 1, 858, 909]` |

Interpretation: this is the opposite of a weak parent basin being uninformative. The wider `expanded_1p50` region has a large positive recovery, while the smaller expansion is negative. That suggests a thresholded/contextual interaction with surrounding visual tokens, not a pure local object-box value.

## Mechanistic Consequence

This result makes the "coordinate slot basin/attraction" hypothesis more specific:

1. The attraction field is not just the object box, and not just an undifferentiated near ring.
2. Spatial side bands can have opposite signs within the same case.
3. Wider context can flip effect sign, so fine-grained visual locality should be analyzed together with route/value decomposition and QK source-token routing, not only as bbox inclusion/exclusion.
4. The parent harmful `duplicate_basin` aggregate can be hiding a mix of constructive and destructive subregions. This is especially important for interpreting apparent destructive value basins.

## Recommended Next Step

Prioritize a fine-spatial QK/source-token follow-up on the sign-changing regions:

- `record114_target7_component_row_7_handbag__fine_right_band` as the clearest negative directional pocket.
- `record114_target7_component_row_7_handbag__fine_expanded_1p50` as a positive wider-context contrast.
- `record50_target9_component_row_9_person__fine_expanded_1p25` versus `record50_target9_component_row_9_person__fine_expanded_1p50` as the strongest scale sign flip.

The goal is to determine whether the sign flips are primarily route-origin changes, value-content changes, or downstream output projection interactions. This should be treated as a promising path worth deeper dynamic exploration because it can materially change the final picture: duplicate bursts may arise from structured coordinate-token attraction fields rather than simple missing visual evidence or duplicated object-box content.
