# Phase 4 Record-50 Fine-Spatial Token Membership

Date: 2026-06-11

Scope: deterministic visual-token membership diff for the record-50 fine-spatial scale pair. This does not yet include per-token QK/value scores; it identifies which visual grid cells enter the positive wide duplicate-basin region.

## Inputs

- Small harmful case: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_basin_fine_spatial_promotions_layer17_head1_reversals/promoted_cases/record50_target9_component_row_9_person__fine_expanded_1p25`
- Wide helpful case: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_basin_fine_spatial_promotions_layer17_head1_reversals/promoted_cases/record50_target9_component_row_9_person__fine_expanded_1p50`
- Token membership output: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_basin_fine_spatial_promotions_layer17_head1_reversals/fine_spatial_token_membership_record50_scale_pair`
- Image: `/data/CoordExp/public_data/coco/raw/images/val2017/000000005193.jpg`
- Image size: `640x425`

The helper inferred the visual grid by matching the existing route/content duplicate-basin token counts for both boxes and selecting the grid whose aspect ratio best matches the image:

```text
grid_h = 13
grid_w = 20
grid_token_count = 260
image_aspect = 1.505882
grid_aspect = 1.538462
candidate_grid_count = 7
```

## Membership Counts

| region | bbox norm1000 | duplicate-basin tokens | route/content total recovery |
| --- | --- | ---: | ---: |
| `expanded_1p25` | `[548, 114, 813, 796]` | 45 | -0.002293150 |
| `expanded_1p50` | `[504, 1, 858, 909]` | 84 | 0.007959660 |

The wide positive region preserves all 45 small-region tokens and adds 39 visual tokens.

Added-token distribution relative to the small harmful box:

| relative position | added tokens |
| --- | ---: |
| `above` | 5 |
| `above_left` | 1 |
| `above_right` | 1 |
| `below` | 10 |
| `below_left` | 2 |
| `below_right` | 2 |
| `left` | 9 |
| `right` | 9 |

Grouped read:

```text
side halo = 18 tokens
lower band + lower corners = 14 tokens
upper band + upper corners = 7 tokens
```

## Added Token Geometry

The added side halo is a one-token column on each side of the harmful box:

- left column: grid col 10, rows 1 through 9, x center `525.0`
- right column: grid col 16, rows 1 through 9, x center `825.0`

The added lower band is two rows below the harmful box:

- rows 10 through 11
- cols 11 through 15 for direct below
- cols 10 and 16 for lower corners
- y centers `807.7` and `884.6`

The added upper band is one row above the harmful box:

- row 0
- cols 11 through 15 for direct above
- cols 10 and 16 for upper corners
- y center `38.5`

## Mechanistic Read

This explains the scale flip at the membership level:

```text
45-token object-centered duplicate basin: harmful
84-token wider duplicate basin: helpful
delta: +39 context tokens
```

The added tokens are not a random diffuse background sample. They form a structured halo around the harmful duplicate basin, especially the lower band and left/right side columns. Since the component-competition probe showed that the wide region's positive downstream effect is carried by `duplicate_basin` itself rather than by `visual_near_ring` or `visual_far_background`, these 39 added tokens are now the prime suspects for flipping the integrated duplicate-basin route/content effect.

This sharpens the current mechanism:

```text
region boundary changes duplicate-basin token composition,
which changes integrated route/content sign,
even when standalone duplicate-basin value projection remains target-negative.
```

So the important unit is not the semantic object box alone. It is the model's visual-token bucket induced by the coordinate-region boundary.

## Caveat

This is a deterministic membership reconstruction, not a per-token causal score. It tells which cells enter the bucket, but not yet which of those cells carry the QK/value leverage.

## Recommended Next Probe

Run a per-token duplicate-basin scoring probe for this record-50 scale pair:

- compute per-token QK score delta and attention mass delta for the 45 shared tokens and 39 added tokens;
- compute per-token output-projected value contribution to the target coordinate basin;
- rank added tokens by contribution and test whether the lower band or side halo dominates the positive wide-region route/content effect.

If a small subset of the 39 added tokens explains the sign flip, the final picture becomes much more concrete: duplicate bursts are sensitive to token-bucket boundary composition, not just object perception or object-level routing.
