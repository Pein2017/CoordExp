# Phase 4 Basin Fine-Spatial Value/QK/Route Sign-Flip Join

Date: 2026-06-11

Scope: targeted value-basin projection follow-up for the four fine-spatial sign-flip candidates already probed with route/content and QK route-origin. This is a mechanistic panel, not validation-scale evidence.

## Inputs

- Fine-spatial promotion root: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_basin_fine_spatial_promotions_layer17_head1_reversals`
- Route/content summary: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_basin_fine_spatial_promotions_layer17_head1_reversals/fine_spatial_route_content_summary`
- QK sign-flip summary: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_basin_fine_spatial_promotions_layer17_head1_reversals/fine_spatial_qk_signflip_summary`
- Value/QK/route join summary: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_basin_fine_spatial_promotions_layer17_head1_reversals/fine_spatial_value_qk_route_signflip_summary`
- GPU scope: devices `0,1,2,3`; four value projection probes completed with failure count 0.

Candidates:

- `record114_target7_component_row_7_handbag__fine_right_band`
- `record114_target7_component_row_7_handbag__fine_expanded_1p50`
- `record50_target9_component_row_9_person__fine_expanded_1p25`
- `record50_target9_component_row_9_person__fine_expanded_1p50`

Each candidate was probed at layer 17, head 1 with source components `duplicate_basin`, `visual_near_ring`, and `visual_far_background`.

## Duplicate-Basin Join

| candidate | route prob | qk attn | qk score | value centered | value r4 margin | value rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `record114_target7_component_row_7_handbag__fine_right_band` | -0.002841210 | -0.084757 | -0.518173 | 0.054250 | 0.054820 | 629 |
| `record114_target7_component_row_7_handbag__fine_expanded_1p50` | 0.002482514 | 0.235476 | 4.996083 | -0.031845 | -0.032385 | 425 |
| `record50_target9_component_row_9_person__fine_expanded_1p25` | -0.002293150 | 0.817075 | 7.635741 | -0.137174 | -0.138649 | 471 |
| `record50_target9_component_row_9_person__fine_expanded_1p50` | 0.007959660 | 0.273189 | 5.664747 | -0.152698 | -0.154265 | 536 |

## Mechanistic Read

The value projection does not collapse the fine-spatial story into a simple QK-plus-value two-gate rule. It confirms one disagreement and creates another.

For `record50_target9_component_row_9_person__fine_expanded_1p25`, the diagnosis is clean:

```text
route/content total recovery = -0.002293150
duplicate-basin QK attention delta = 0.817075
duplicate-basin QK score delta = 7.635741
duplicate-basin value centered delta = -0.137174
duplicate-basin radius-4 margin = -0.138649
```

This is the clearest fine-spatial `route_without_value` pocket. The model strongly routes to the promoted visual basin, but the selected basin's value projection points away from the active coordinate target. That supports the idea that false duplicate attraction can survive visual availability because the language/coordinate slot is selecting or amplifying the wrong value content.

For `record50_target9_component_row_9_person__fine_expanded_1p50`, the simple value explanation fails:

```text
route/content total recovery = 0.007959660
duplicate-basin QK attention delta = 0.273189
duplicate-basin QK score delta = 5.664747
duplicate-basin value centered delta = -0.152698
duplicate-basin radius-4 margin = -0.154265
```

The larger region is downstream-positive even though the duplicate-basin value projection scalar is still target-negative. Therefore the positive effect likely comes from a fuller route/content interaction: source bucket competition, non-duplicate-basin components, output projection nonlinearity, or residual context that is not summarized by the duplicate-basin target-centered value scalar.

For `record114_target7_component_row_7_handbag`, the two fine regions also resist a simple value rule:

- The negative `right_band` has negative QK route terms but positive duplicate-basin value projection.
- The positive `expanded_1p50` has positive QK route terms but negative duplicate-basin value projection.

This is a strong warning that `coord_output_target_centered_delta` is useful but not sufficient. It is a local value-basin scalar, not the full causal route/content effect.

## Updated Gate Picture

The current best mechanism is a three-gate-plus-competition picture:

1. QK route gate: does the head route to this visual/spatial basin?
2. Value projection gate: does the selected component's value vector locally point toward the coordinate target?
3. Integration gate: does the full route/content intervention survive competition with surrounding source buckets and residual/output context?

The fine-spatial panel shows all three can disagree:

- `record50 expanded_1p25`: QK positive, value negative, downstream negative.
- `record50 expanded_1p50`: QK positive, duplicate-basin value negative, downstream positive.
- `record114 right_band`: QK negative, duplicate-basin value positive, downstream negative.
- `record114 expanded_1p50`: QK positive, duplicate-basin value negative, downstream positive.

So the core mechanism is not "visual object perceived or not", and not even "the head attends to the right spatial basin or not". It is a structured coordinate-slot attraction system where region boundary changes which keys, values, and competing source buckets are integrated into the coordinate-token output basin.

## Recommended Next Step

The next attractive path is a component-competition probe for these four fine regions, mirroring the earlier parent component competition but staying inside each fine-spatial case:

- compare `duplicate_basin`, `visual_near_ring`, `visual_far_background`, and `non_region_complement` route/content effects;
- include `route_delta_masked_values` and `value_delta_control_route` separately;
- prioritize `record50_target9_component_row_9_person__fine_expanded_1p50`, where positive downstream recovery is not explained by duplicate-basin value projection.

This should tell whether the positive wide-region effect comes from near-ring/far-background rescue, from non-region complement, or from a route/value integration term not visible in the projection scalar.
