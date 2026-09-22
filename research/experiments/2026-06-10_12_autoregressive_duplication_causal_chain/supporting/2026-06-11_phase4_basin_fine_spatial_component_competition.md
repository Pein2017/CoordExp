# Phase 4 Basin Fine-Spatial Component Competition

Date: 2026-06-11

Scope: targeted route/content component competition for the four fine-spatial sign-flip candidates. This follows the fine-spatial QK and value-projection joins and tests whether downstream sign flips are explained by `duplicate_basin`, nearby visual ring, far visual background, or non-region complement source buckets.

## Inputs

- Fine-spatial promotion root: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_basin_fine_spatial_promotions_layer17_head1_reversals`
- Component competition summary: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_basin_fine_spatial_promotions_layer17_head1_reversals/fine_spatial_component_competition_summary`
- Prior value/QK/route join: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_basin_fine_spatial_promotions_layer17_head1_reversals/fine_spatial_value_qk_route_signflip_summary`
- GPU scope: devices `0,1,2,3`; four component-competition probes completed with failure count 0.

Candidates:

- `record114_target7_component_row_7_handbag__fine_right_band`
- `record114_target7_component_row_7_handbag__fine_expanded_1p50`
- `record50_target9_component_row_9_person__fine_expanded_1p25`
- `record50_target9_component_row_9_person__fine_expanded_1p50`

Components:

- `duplicate_basin`
- `visual_near_ring`
- `visual_far_background`
- `non_region_complement`

Effects:

- `total_delta`
- `route_delta_masked_values`
- `value_delta_control_route`

## Total-Delta Extrema

| candidate | min component | min prob recovery | max component | max prob recovery |
| --- | --- | ---: | --- | ---: |
| `record114_target7_component_row_7_handbag__fine_right_band` | `non_region_complement` | -0.004784987 | `visual_far_background` | -0.002239372 |
| `record114_target7_component_row_7_handbag__fine_expanded_1p50` | `visual_near_ring` | -0.000303240 | `duplicate_basin` | 0.002482514 |
| `record50_target9_component_row_9_person__fine_expanded_1p25` | `duplicate_basin` | -0.002293150 | `visual_near_ring` | 0.001142452 |
| `record50_target9_component_row_9_person__fine_expanded_1p50` | `visual_near_ring` | 0.000047573 | `duplicate_basin` | 0.007959660 |

## Mechanistic Read

This component competition explains the record-50 scale flip better than the standalone value projection did.

For `record50_target9_component_row_9_person__fine_expanded_1p25`, the fine region is downstream-negative because the duplicate-basin bucket itself is harmful:

```text
duplicate_basin total recovery = -0.002293150
duplicate_basin route_delta recovery = -0.000985343
duplicate_basin value_delta recovery = -0.002044578
duplicate_basin prior QK attention delta = 0.817075
duplicate_basin prior value centered delta = -0.137174
```

But the competing source buckets are not harmful:

```text
visual_near_ring total recovery = 0.001142452
non_region_complement total recovery = 0.000896362
visual_far_background total recovery = 0.000225276
```

So the small expansion is a clean `route-to-bad-duplicate-basin` pocket: QK selects the basin strongly, the basin's value projection is target-negative, and the route/content effect is negative. Nearby context would actually help, but it is not the dominant duplicate-basin bucket.

For `record50_target9_component_row_9_person__fine_expanded_1p50`, the duplicate-basin bucket becomes the dominant positive component:

```text
duplicate_basin total recovery = 0.007959660
duplicate_basin route_delta recovery = 0.000579406
duplicate_basin value_delta recovery = 0.003132182
duplicate_basin prior QK attention delta = 0.273189
duplicate_basin prior value centered delta = -0.152698
```

The positive wide-region effect is therefore not rescued by near-ring or far-background:

```text
visual_near_ring total recovery = 0.000047573
visual_far_background total recovery = 0.000252433
non_region_complement total recovery = 0.000293298
```

This resolves the previous apparent contradiction. The standalone value projection scalar said the wide duplicate-basin value was target-negative, but the route/content `value_delta_control_route` for the same component is positive. The likely reason is that the projection scalar is a local readout of the attention-weighted value contribution against the coordinate basin, while the route/content intervention measures the integrated effect after routing and output-context interactions.

For `record114_target7_component_row_7_handbag__fine_right_band`, all total-delta components remain negative:

```text
duplicate_basin total recovery = -0.002841210
visual_near_ring total recovery = -0.003453457
visual_far_background total recovery = -0.002239372
non_region_complement total recovery = -0.004784987
```

Yet the value-delta terms for non-duplicate visual/background components are weakly positive. This says the negative right-band pocket is largely a route/integration harm, not a lack of target-local value content.

For `record114_target7_component_row_7_handbag__fine_expanded_1p50`, duplicate basin is again the only clearly useful total and value component:

```text
duplicate_basin total recovery = 0.002482514
duplicate_basin value_delta recovery = 0.004073853
visual_near_ring total recovery = -0.000303240
visual_far_background total recovery = 0.000208655
non_region_complement total recovery = -0.000068866
```

## Updated Picture

The fine-spatial mechanism now looks like this:

1. QK can select a basin strongly.
2. The standalone value projection can be target-positive or target-negative, but it does not fully determine the routed downstream effect.
3. The route/content component intervention reveals whether the selected component actually helps after the model's output-context integration.
4. Region boundary changes can flip the duplicate-basin component itself from harmful to helpful, even for the same source object and same layer/head.

This is stronger than the previous "three gates" framing. The gates are not independent scalar filters. They are coupled:

```text
region boundary -> source-token bucket composition -> QK route -> value/source projection -> output-context integration -> coordinate-slot basin
```

The most important practical consequence is that duplicate bursts cannot be diagnosed by asking only whether the model perceives a visual object, or whether a head attends to an object-shaped region. The same semantic object at a slightly different spatial scale can switch from duplicate-attraction harm to coordinate repair.

## Recommended Next Step

The next promising path is token-level composition inside the duplicate-basin bucket for the record-50 scale pair:

- `record50_target9_component_row_9_person__fine_expanded_1p25`
- `record50_target9_component_row_9_person__fine_expanded_1p50`

Compare which visual tokens enter the duplicate-basin bucket at each scale, their QK scores, and their per-token value/source projection. The central question is whether the wide expansion adds a small set of high-leverage tokens that turn the integrated duplicate-basin route/content effect positive despite the aggregate standalone value projection remaining target-negative.
