# Phase 4 Record-114 Handbag Mask-Geometry 2x2 Control

Date: 2026-06-11

Scope: fixed-source/fixed-mask 2x2 control for the record-114 handbag fine-spatial pair. This tests whether the negative `right_band` and positive `expanded_1p50` effects are driven by source-token bucket composition or by masked-counterfactual geometry.

## Inputs

- Fine-spatial root: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_basin_fine_spatial_promotions_layer17_head1_reversals`
- Fixed right-band source control root: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_basin_fine_spatial_promotions_layer17_head1_reversals/mask_geometry_control_record114_fixed_right_source`
- Fixed expanded source control root: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_basin_fine_spatial_promotions_layer17_head1_reversals/mask_geometry_control_record114_fixed_expanded_source`
- 2x2 summary: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_basin_fine_spatial_promotions_layer17_head1_reversals/mask_geometry_control_record114_handbag_2x2/summary`
- GPU scope: devices `0,1,2,3`; all four route/content probes completed with failure count 0.

Geometry:

```text
right-band source/mask bbox = [753, 320, 812, 748]
expanded source/mask bbox   = [402, 106, 870, 962]
```

The control crosses:

```text
source bucket: right-band 9 tokens vs expanded 119 tokens
mask geometry: right-band mask vs expanded mask
```

## Total-Delta 2x2

| source | mask | source tokens | total recovery | r4 recovery |
| --- | --- | ---: | ---: | ---: |
| right source | right mask | 9 | -0.002841210 | -0.019279581 |
| right source | expanded mask | 9 | 0.002879262 | 0.019931549 |
| expanded source | right mask | 119 | -0.005681694 | -0.037137538 |
| expanded source | expanded mask | 119 | 0.002482514 | 0.015894322 |

## Route And Value Terms

Route-only term:

| source | mask | route prob recovery | route r4 recovery |
| --- | --- | ---: | ---: |
| right source | right mask | -0.001389866 | -0.009519778 |
| right source | expanded mask | 0.000157512 | 0.000815565 |
| expanded source | right mask | -0.004986773 | -0.033391569 |
| expanded source | expanded mask | -0.000209776 | -0.000842040 |

Value-on-control-route term:

| source | mask | value prob recovery | value r4 recovery |
| --- | --- | ---: | ---: |
| right source | right mask | -0.001848305 | -0.010879502 |
| right source | expanded mask | 0.002038716 | 0.014352294 |
| expanded source | right mask | -0.001443710 | -0.009868205 |
| expanded source | expanded mask | 0.004073853 | 0.025943758 |

## Mechanistic Read

This control shows that the handbag sign flip is also mask-geometry dominated.

Holding the small harmful source bucket fixed and changing only the mask geometry flips the sign:

```text
right source + right mask    = -0.002841210
right source + expanded mask =  0.002879262
```

Holding the large helpful source bucket fixed and changing only the mask geometry also flips the sign, in the opposite direction:

```text
expanded source + expanded mask =  0.002482514
expanded source + right mask    = -0.005681694
```

So the source bucket alone does not determine the route/content effect. The same 9-token right-band source can be harmful or helpful, and the same 119-token expanded source can be helpful or harmful, depending on the masked counterfactual.

The source bucket still matters as an amplitude/modulation term:

- under the expanded mask, right source and expanded source are both positive, but right source is slightly stronger on total recovery (`0.002879262` vs `0.002482514`);
- under the right mask, expanded source is more harmful than right source (`-0.005681694` vs `-0.002841210`).

But the sign is controlled by mask geometry.

## Relation To Record 50

This mirrors and strengthens the record-50 finding:

- record 50: fixed 45-token small source becomes positive under wide mask;
- record 114: fixed 9-token right-band source becomes positive under expanded mask, while fixed 119-token expanded source becomes negative under right mask.

The general mechanism now looks robust across both cases:

```text
duplicate-basin effect is primarily a source-vs-masked-counterfactual relation,
not a static property of the source visual tokens.
```

This also reframes the earlier QK/value/source-bucket findings. QK route, value projection, and source composition remain meaningful, but their downstream sign is evaluated relative to the visual counterfactual induced by the mask. Changing the mask boundary can flip the same source bucket from coordinate harm to coordinate repair.

## Updated Mechanism

The current best description is:

```text
region boundary -> masked visual counterfactual -> route/value delta over source bucket -> coordinate-slot basin
```

not:

```text
region boundary -> selected source tokens -> coordinate-slot basin
```

This is a deeper mechanism than surface object duplication. The model's coordinate slot appears to be governed by a counterfactual visual-state comparison. Duplication bursts may arise when the active prefix makes the coordinate slot compare against a malformed or overly local visual counterfactual, causing the route/content delta to point toward a duplicated coordinate basin.

## Recommended Next Step

At this point the next high-value probe is to move from individual cases to a small panel:

- apply fixed-source/mask-geometry controls to 4-8 additional fine-spatial candidates from the basin shortlist;
- include one likely route-origin case, one likely value-only case, and one positive original basin;
- classify each as `mask_geometry_dominated`, `source_bucket_dominated`, or `mixed`.

If mask geometry dominance recurs, it should become a central pillar of the final mechanism diagnosis and of any training objective proposal.
