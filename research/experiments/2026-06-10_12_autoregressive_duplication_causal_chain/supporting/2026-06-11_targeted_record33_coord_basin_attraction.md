# Targeted Record 33 Coordinate-Basin Attraction

Date: 2026-06-11

## Scope

This note links the layer-17/head-1 route-origin picture for the main `none_latest_ckpt32`
record-33 `post_y1/pre_x2` duplication burst to the coordinate-token basin behavior of
the trained `<|coord_*|>` embeddings.

The specific question was whether the constructive/destructive components found in the
projected-direction analysis move coordinate probability mass into different coordinate
slot basins, especially old/duplicate bins versus target/control bins.

## Artifact Handles

- Target panel:
  `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel`
- Projected contribution patch run:
  `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/projected_contribution_coord_basin_layer17_head1`
- Coordinate-basin synthesis:
  `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/targeted_coord_basin_attraction_synthesis_layer17_head1`
- Final report:
  `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/targeted_coord_basin_attraction_synthesis_layer17_head1/targeted_coord_basin_attraction_synthesis_report.md`
- Summary JSON:
  `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/targeted_coord_basin_attraction_synthesis_layer17_head1/targeted_coord_basin_attraction_synthesis_summary.json`
- Source state-patch rows:
  `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/key_value_component_expansion_layer17_head1/attention_score_bias_patch_rows.jsonl`

The projected-contribution patch run completed successfully, but those rows do not carry
the coordinate top-bin fields needed for basin inspection. The final coordinate-basin
synthesis therefore uses the richer state-patch rows from the key+value component
expansion artifact.

## Main Component Basin Shifts

Scope: `none_latest_ckpt32`, record `33`, phase `post_y1/pre_x2`,
patch direction `key_value_state_control`, layer `17`, head `1`.

| component | projected target fraction | prob recovery | rank recovery | expected-bin recovery | top1-distance recovery | patched top1 distance | dominant patched top1 bins | expected toward/away | top1 closer/farther |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| duplicate_basin | 15.9077 | 0.0100274 | 16.25 | -8.49191 | -8.00 | 10.9167 | 178x3, 170x3, 218x2, 246x1, 237x1 | 10/2 | 7/0 |
| visual_near_ring | -9.09364 | 0.0024917 | 6.25 | -3.75074 | -7.25 | 11.6667 | 178x5, 218x2, 170x2, 258x1, 237x1 | 11/1 | 6/0 |
| visual_far_background | 0.358801 | 0.0001975 | 0.75 | -0.21404 | -4.00 | 14.9167 | 178x7, 218x2, 254x1, 237x1, 170x1 | 8/4 | 3/0 |
| non_region_complement | -8.81520 | 0.0032253 | 9.3333 | -4.38181 | -7.25 | 11.6667 | 178x5, 218x2, 170x2, 258x1, 237x1 | 12/0 | 6/0 |
| whole_head | 7.09246 | 0.0130371 | 18.75 | -14.2571 | -13.50 | 5.4167 | 158x4, 165x3, 218x2, 258x1, 237x1 | 12/0 | 9/0 |

Sign convention: negative expected-bin or top1-distance recovery means the patched
distribution moved closer to the target bin than the masked distribution did.

## Reading

The anti-target projected components are not arbitrary far-bin attractors. Even
`visual_near_ring` and `non_region_complement`, which had negative projected target
fractions in the projected-direction cancellation synthesis, still move the expected
coordinate bin and often the top-1 coordinate bin toward the target compared with the
masked state. Their motion is weaker, however, and they usually leave the patched top-1
mass in an intermediate old/duplicate-ish basin dominated by bins like `178`, `185`,
`218`, and nearby values.

The `duplicate_basin` component gives the cleanest positive target-logit direction. It
moves high old bins such as `[198, 185, 178, 193, ...]` toward a nearer bridge basin such
as `[178, 170, 174, 165, ...]`, but it does not usually snap all the way into the
control basin.

The `whole_head` patch relocates the coordinate basin most strongly. Its dominant patched
bins shift toward target/control-adjacent bins such as `158`, `165`, `170`, and `151`,
and it has the best mean top1-distance recovery and expected-bin recovery in this panel.
This explains why the whole-head repair can look better under basin-distance metrics even
though its projected target fraction is lower than the isolated `duplicate_basin`
component: broader context helps basin relocation, while cancellation reduces the clean
target-logit-aligned repair direction.

## Representative Rows

- Row 27, target `170`, generated `<|coord_355|>`, next `<|coord_170|>`:
  - masked top bins: `[198, 185, 178, 193, 186, 184, 191, 187]`
  - `duplicate_basin` patched: `[178, 170, 174, 165, 171, 175, 180, 176]`
  - control: `[158, 165, 159, 163, 162, 157, 170, 156]`
  - prob recovery: `0.0193146`
  - expected-bin recovery: `-12.1666`

- Row 27 under `visual_near_ring`:
  - patched top bins: `[178, 185, 184, 186, 198, 182, 170, 180]`
  - prob recovery: `0.0058614`
  - expected-bin recovery: `-3.18079`
  - interpretation: the patch weakly bends the distribution toward the target, but most
    top-bin mass remains in the old/duplicate basin.

- Row 25, target `165`, generated `<|coord_341|>`, next `<|coord_165|>`,
  under `whole_head`:
  - masked top bins: `[178, 185, 198, 184, 186, 182, 187, 191]`
  - patched top bins: `[165, 170, 158, 159, 163, 157, 162, 171]`
  - control top bins: `[158, 150, 151, 159, 153, 157, 156, 152]`
  - prob recovery: `0.0276502`
  - expected-bin recovery: `-20.4647`
  - interpretation: whole-head context performs the clearest local basin relocation in
    this representative case.

## Mechanism Update

The current picture is no longer just "positive duplicate basin versus negative
background." It is more structured:

1. The duplicate-basin route is the most target-logit-aligned component.
2. Near-ring and non-region components can be anti-target in the projected direction
   while still weakly shaping the coordinate distribution toward a nearby basin.
3. The destructive effect is therefore likely a competition between closely related
   coordinate basins, not a generic visual-noise or far-background effect.
4. Whole-head context improves coordinate-basin relocation but also mixes in
   cancellation directions, so it can outperform on basin movement while underperforming
   the duplicate-basin component on clean projected target contribution.

This connects the earlier key-side route origin to the user's coordinate-slot basin
concern: the failure looks like attraction to an old/intermediate coordinate basin after
the route has locked onto the duplicated region, rather than a simple inability to put
mass on coordinate tokens.

## Suggested Next Deterministic Step

Define explicit old/duplicate, bridge, target, and control coordinate-bin bands per row
from the masked, patched, and control top-bin sets. Then report component-wise mass flow
between those bands, instead of only top-1 and expected-bin movement. This should make it
possible to distinguish:

- target-logit repair,
- local basin relocation,
- old-bin persistence,
- and cancellation from nearby but wrong coordinate attractors.

This can be done post-hoc from the existing `masked_coord_top_bins`,
`patched_coord_top_bins`, `control_coord_top_bins`, and target-bin fields for the same
panel before launching another GPU run.

## Verification

- GPU run completed for projected-contribution patch with
  `projected_contribution_patch_row_count=410`.
- Coordinate-basin synthesis materialized a report and summary JSON with five component
  summaries for the main case.
- No tracked code was changed for this note.
