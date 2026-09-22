# Directional Control-Anchor Split

Date: 2026-06-11

## Scope

This note records the follow-up to the auxiliary-control undershoot finding. The
coordinate band-flow reducer now splits `control_anchor` into row-local directional
sub-bands:

- `control_anchor_below_target`
- `control_anchor_above_target`
- `control_anchor_equal_target`

This makes the `control_anchor` label less ambiguous: movement into the control-state
basin can now be separated into lower-coordinate undershoot and upper-coordinate
overshoot.

## Implementation

Updated:

- `src/analysis/autoregressive_duplication_mechanism/phase4_coord_basin_band_flow.py`
- `tests/analysis/autoregressive_duplication_mechanism/test_phase4_coord_basin_band_flow.py`

The reducer now emits:

- directional control-anchor bin lists per row;
- top-k masses for each directional control sub-band;
- patched/control/masked deltas for those masses;
- top-1 control-anchor direction labels and group counts;
- report columns for lower-control and upper-control deltas.

Regenerated artifact:

`/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/coord_basin_band_flow_layer17_head1`

## Main Auxiliary Result

Scope: `aux_latest_ckpt32`, record `33`, phase `post_y1/pre_x2`.

| component | control delta | lower-control delta | upper-control delta | exact delta | wide delta | patched control direction |
|---|---:|---:|---:|---:|---:|---|
| duplicate_basin | 0.036213 | 0.047930 | -0.011718 | -0.004539 | -0.044645 | below 8, above 1, not-control 3 |
| visual_near_ring | -0.005973 | -0.005289 | -0.000684 | 0.003408 | -0.005096 | below 3, above 4, not-control 5 |
| visual_far_background | -0.001911 | -0.001560 | -0.000351 | 0.003431 | 0.000864 | below 6, above 2, not-control 4 |
| non_region_complement | 0.001457 | 0.003949 | -0.002492 | 0.000563 | -0.004592 | below 5, above 2, not-control 5 |
| whole_head | 0.033430 | 0.049362 | -0.015932 | -0.007962 | -0.056979 | below 8, above 1, not-control 3 |

The auxiliary undershoot mechanism is now explicit. For the two strongest route/context
components:

- `duplicate_basin` gains lower-control mass while losing upper-control, exact-target,
  and wide-radius target-near mass.
- `whole_head` does the same, and more strongly: lower-control `+0.049362`,
  upper-control `-0.015932`, exact `-0.007962`, wide `-0.056979`.

So the earlier "control-like but negative wide-radius" behavior is not a generic
control-anchor ambiguity. It is specifically lower-control attraction.

## Row-Level Check

For `aux_latest_ckpt32`, record 33, `post_y1/pre_x2`, `whole_head` patched top-1
control-anchor rows:

| row | target | masked top1 | patched top1 | control top1 | direction | wide delta | lower-control delta | upper-control delta |
|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 17 | 951 | 999 | 999 | 999 | above | 0.010142 | -0.002719 | 0.011669 |
| 18 | 202 | 198 | 198 | 198 | below | -0.010332 | -0.001247 | -0.010305 |
| 19 | 484 | 480 | 480 | 480 | below | -0.001972 | -0.002238 | 0.000000 |
| 20 | 165 | 165 | 158 | 151 | below | -0.079195 | 0.192730 | 0.000000 |
| 21 | 170 | 170 | 158 | 158 | below | -0.183812 | 0.216171 | 0.000000 |
| 22 | 159 | 158 | 151 | 150 | below | 0.001156 | 0.137197 | 0.000000 |
| 23 | 163 | 170 | 158 | 158 | below | -0.101951 | 0.116587 | -0.033650 |
| 24 | 157 | 170 | 151 | 158 | below | -0.020229 | 0.076580 | -0.022669 |
| 28 | 156 | 151 | 151 | 150 | below | -0.078994 | 0.089200 | -0.051258 |

Rows 20 and 21 remain the cleanest examples: the masked state is exact target, while
patching shifts top-1 down to the lower control basin and removes target-near mass.

## Contrast With `none_latest_ckpt32`

For `none_latest_ckpt32`, record 33, `post_y1/pre_x2`, `whole_head` has:

- control delta: `0.086852`
- lower-control delta: `0.075453`
- upper-control delta: `0.011399`
- exact delta: `0.016706`
- wide delta: `0.116555`
- patched control direction: below `4`, above `6`, not-control `2`

This is very different from the auxiliary checkpoint. `none_latest_ckpt32` also has a
large lower-control component, but it does not lose exact/wide target-near mass; the
whole-head patch moves the broader coordinate basin toward the target neighborhood.

For `aux_latest_ckpt32`, lower-control movement competes against exact/wide mass. For
`none_latest_ckpt32`, lower/upper control movement appears as part of a broader
target-near relocation.

## Mechanism Update

The directional split sharpens the auxiliary story:

1. The auxiliary-loss checkpoint is not simply failing to move into a control basin.
2. It often moves too decisively into a lower control basin.
3. That lower-control attraction can remove mass from an already-correct or
   target-near masked state.
4. This makes the auxiliary mechanics distinct from the `none_latest_ckpt32`
   whole-head relocation mechanism.

This is directly relevant to the coordinate-slot basin concern: the added coordinate
embeddings may preserve local geometry, but the auxiliary checkpoint can create a
directional basin bias within that local geometry. The issue is not "coord tokens versus
non-coord tokens"; it is a directional attractor inside the coordinate-token manifold.

## Next Deterministic Step

Use the directional control split across all target-panel checkpoints to decide whether
lower-control attraction is unique to `aux_latest_ckpt32` or also present in parent
checkpoints under different phases. A compact post-hoc comparison should report, for
each checkpoint/case/component:

- lower-control delta;
- upper-control delta;
- exact/wide delta;
- patched top1 direction counts;
- whether lower-control gain co-occurs with positive or negative wide-radius movement.

No GPU is needed for that comparison.

## Verification

- Focused direct test harness passed for
  `tests/analysis/autoregressive_duplication_mechanism/test_phase4_coord_basin_band_flow.py`.
- `python -m py_compile` passed for the updated reducer and CLI.
- Regenerated the 410-row band-flow artifact with the directional control fields.
- Inspected the regenerated Markdown report and row-level JSONL for
  `aux_latest_ckpt32`, record `33`, `post_y1/pre_x2`.
