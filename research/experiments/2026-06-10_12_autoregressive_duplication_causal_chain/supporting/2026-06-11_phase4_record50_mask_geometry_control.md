# Phase 4 Record-50 Mask-Geometry Control

Date: 2026-06-11

Scope: causal control for the record-50 fine-spatial scale pair. This holds the duplicate-basin source bucket fixed to the 45-token `expanded_1p25` region and varies only the visual mask geometry used to create the masked counterfactual.

## Inputs

- Source case: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_basin_fine_spatial_promotions_layer17_head1_reversals/promoted_cases/record50_target9_component_row_9_person__fine_expanded_1p25`
- Control root: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_basin_fine_spatial_promotions_layer17_head1_reversals/mask_geometry_control_record50_fixed_small_source`
- Summary: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_basin_fine_spatial_promotions_layer17_head1_reversals/mask_geometry_control_record50_fixed_small_source/mask_geometry_control_summary`
- GPU scope: devices `0,1`; both route/content probes completed with failure count 0.

Control construction:

```text
source bbox fixed = [548, 114, 813, 796]
small mask bbox  = [548, 114, 813, 796]
wide mask bbox   = [504, 1, 858, 909]
```

The route/content source membership remains the 45-token small duplicate-basin bucket in both control conditions. Only `mask_bbox_norm1000_xyxy` changes.

## Main Table

| condition | source tokens | mask bbox | total recovery | r4 recovery |
| --- | ---: | --- | ---: | ---: |
| original small source + small mask | 45 | original small | -0.002293150 | -0.014335102 |
| original wide source + wide mask | 84 | original wide | 0.007959660 | 0.052730737 |
| fixed small source + small mask | 45 | `[548, 114, 813, 796]` | -0.002293150 | -0.014335102 |
| fixed small source + wide mask | 45 | `[504, 1, 858, 909]` | 0.003718783 | 0.021861142 |

Route-only term:

| condition | route-delta prob recovery | route-delta r4 recovery |
| --- | ---: | ---: |
| fixed small source + small mask | -0.000985343 | -0.006189745 |
| fixed small source + wide mask | 0.000338877 | 0.002103440 |

Value-on-control-route term:

| condition | value-delta prob recovery | value-delta r4 recovery |
| --- | ---: | ---: |
| fixed small source + small mask | -0.002044578 | -0.012090029 |
| fixed small source + wide mask | 0.003454595 | 0.020911302 |

## Mechanistic Read

This control cleanly separates source-token composition from masked-counterfactual geometry.

The sign flip does **not** require the 39 added wide-region source tokens:

```text
fixed 45-token source + small mask = -0.002293150
fixed 45-token source + wide mask  =  0.003718783
```

The fixed-source wide-mask condition recovers roughly half of the original wide-source positive total effect:

```text
original wide source + wide mask total recovery = 0.007959660
fixed small source + wide mask total recovery   = 0.003718783
```

So source composition still adds amplitude, but mask geometry alone is sufficient to cross zero.

This also agrees with the per-token scoring probe:

- the 39 added tokens had nearly zero control-side attention/value;
- the wide mask changed the control-vs-masked deltas on the shared 45 tokens;
- now, when those 45 tokens are held fixed as the patch source, changing only the mask region still flips the sign positive.

The strongest implication is:

```text
the positive wide-region duplicate-basin effect is primarily a masked-counterfactual geometry effect, not a direct added-token evidence effect.
```

## Updated Mechanism

The record-50 scale pair now decomposes as:

1. The 45-token object-centered source bucket is harmful under its own small mask.
2. The same 45-token source bucket becomes helpful when contrasted against the wider masked image.
3. The 39 added tokens are not direct control-side evidence carriers, but they define the wider mask geometry and may add extra positive amplitude when included as source tokens too.

This shifts the duplicate-basin picture from a static visual-token bucket to a counterfactual relation:

```text
coordinate-slot effect = source bucket contribution relative to the masked visual counterfactual
```

The "basin" is therefore not just what the head attends to in the control image. It is what the route/content intervention says the head recovers when a particular visual counterfactual is removed.

## Consequence For Duplication Mechanism

This is now a strong warning against interpreting duplicate bursts as simple object perception or object-box grounding failures. The same visible object-centered tokens can be harmful or helpful depending on the surrounding mask geometry used to define the counterfactual state.

For the final picture, the coordinate-slot attraction basin should be modeled as:

```text
visual boundary -> masked counterfactual state -> QK/value delta over shared source tokens -> coordinate-token basin
```

not merely:

```text
visual object tokens -> coordinate-token basin
```

## Recommended Next Step

Run the same fixed-source/mask-geometry control on `record114_target7_component_row_7_handbag`:

- fixed source: `right_band` or `expanded_1p50`, depending on whether we want the negative pocket or the positive contrast;
- masks: right-band bbox versus expanded-1.50 bbox;
- goal: test whether the handbag sign flip is also mask-geometry dominated or whether it is truly a QK/source-bucket route-origin flip.

If record 114 behaves differently, the mechanism taxonomy becomes richer:

- record 50: counterfactual-geometry dominated;
- record 114: route-origin/source-bucket dominated.
