# Phase 4 Record-50 Per-Token Score Scale Pair

Date: 2026-06-11

Scope: per-token QK, attention, and coordinate-value scoring for the record-50 fine-spatial scale pair. This follows the deterministic token-membership diff and asks whether the 39 tokens newly admitted by `expanded_1p50` directly carry the positive duplicate-basin route/content effect.

## Inputs

- Small harmful case: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_basin_fine_spatial_promotions_layer17_head1_reversals/promoted_cases/record50_target9_component_row_9_person__fine_expanded_1p25`
- Wide helpful case: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_basin_fine_spatial_promotions_layer17_head1_reversals/promoted_cases/record50_target9_component_row_9_person__fine_expanded_1p50`
- Membership artifact: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_basin_fine_spatial_promotions_layer17_head1_reversals/fine_spatial_token_membership_record50_scale_pair`
- Per-token score summary: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_basin_fine_spatial_promotions_layer17_head1_reversals/fine_spatial_token_scoring_record50_scale_pair_summary`
- GPU scope: devices `0,1`; both token-scoring jobs completed with failure count 0.

The scorer emits one row per duplicate-basin visual token with:

- visual grid position and membership label;
- control and masked attention;
- control and masked QK score;
- output-projected per-token value contribution to the target coordinate basin;
- control-minus-masked per-token value contribution.

## Row Counts

| case | rows | token role |
| --- | ---: | --- |
| `small_expanded_1p25` | 45 | all shared/object-centered tokens |
| `wide_expanded_1p50` | 84 | 45 shared tokens plus 39 added halo/context tokens |

## Group Summary

| group | n | control attn sum | attn delta sum | qk mean | qk delta mean | control value sum | value delta sum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `small_expanded_1p25` | 45 | 0.880601081 | 0.817075462 | 15.600162 | 7.635741 | -0.139905 | -0.137249 |
| `wide_expanded_1p50` | 84 | 0.880992437 | 0.273188478 | 13.367498 | 5.664748 | -0.139890 | -0.152830 |
| `wide shared_duplicate_basin` | 45 | 0.880601081 | 0.605553760 | 15.600162 | 8.336837 | -0.139905 | -0.131142 |
| `wide_only_added` | 39 | 0.000391356 | -0.332365283 | 10.791348 | 2.581567 | 0.000015 | -0.021688 |

## Mechanistic Read

The 39 added tokens do **not** directly carry the positive route/content effect through control-side attention or control-side value.

Evidence:

```text
wide_only_added control attention sum = 0.000391356
wide_only_added control value centered sum = 0.000015
```

That is essentially negligible compared with the 45 shared tokens:

```text
wide shared control attention sum = 0.880601081
wide shared control value centered sum = -0.139905
```

So the wide-region positive route/content effect is not because the model simply attends to useful added halo tokens in the control image. The added tokens are mostly silent on the control side.

The important change is in the counterfactual route/value delta. The 45 shared tokens have identical control-side attention and value in the small and wide cases, but their control-minus-masked value delta changes:

```text
small shared value delta sum = -0.137249
wide shared value delta sum = -0.131142
```

That means the wider intervention changes the masked counterfactual for the same shared object-centered tokens, making the shared-token delta less harmful by about `0.0061`.

The added tokens themselves have negative control-minus-masked attention and value delta:

```text
wide_only_added attention delta sum = -0.332365
wide_only_added value delta sum = -0.021688
```

This is the opposite of the naive "added tokens rescue the coordinate by adding positive value" story. The added tokens are not the direct positive carrier; they reshape the intervention geometry and masked baseline.

## Top Added Tokens

The highest-attention added token is a left-side token:

```text
relative = left
grid = (4, 10)
control attention = 0.000193596
QK score = 18.220901
control value centered = 0.000041
```

Even this strongest added token is tiny in absolute attention and value contribution. The next strongest attention tokens are top and lower-band context cells, but all remain small.

## Updated Picture

The record-50 scale flip is now more subtle:

1. The 45 object-centered shared tokens dominate control attention and value in both small and wide regions.
2. The 39 added halo/context tokens are barely used directly in the control pass.
3. Enlarging the region changes the masked counterfactual and the integrated route/content intervention, including the deltas assigned to the shared tokens.
4. The positive wide-region `duplicate_basin` route/content effect is therefore an intervention-geometry effect, not a direct added-token value effect.

This matters for the final mechanism picture. The duplicate basin is not only a set of visual tokens with fixed evidence. It is an intervention-defined coordinate-slot counterfactual: changing the spatial boundary changes what "masking the basin" means, and that can flip the measured route/content sign even when the control-side token evidence is almost unchanged.

## Consequence

The current strongest hypothesis is:

```text
duplicate bursts are sensitive to visual-token bucket boundary because the boundary changes the counterfactual masked state and route/content integration, not merely because it adds useful visible tokens.
```

This makes the coordinate-slot basin story more mechanistic: a coordinate prediction can be attracted by or repaired by a basin depending on how the model's route is contrasted against the masked visual counterfactual. The origin is not just perception, and not even just control-side attention. It is a coupled control-vs-masked route/value state.

## Recommended Next Step

Run a mask-geometry control on the record-50 pair:

- keep the same 45-token object-centered source bucket for patching;
- vary only the visual mask region used to create the masked counterfactual (`expanded_1p25` vs `expanded_1p50`);
- measure whether the route/content sign flips when source tokens are held fixed.

That would separate "source bucket composition" from "masked counterfactual geometry". The per-token evidence now suggests the latter may be the dominant cause of the wide-region sign flip.
