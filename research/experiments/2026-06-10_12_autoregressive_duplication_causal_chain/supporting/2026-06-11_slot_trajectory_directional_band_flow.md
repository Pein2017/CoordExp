# Slot Trajectory Directional Band-Flow

Date: 2026-06-11

## Scope

This note records a post-hoc slot and trajectory enrichment for the directional
coordinate band-flow reducer. The immediate question was whether the auxiliary
checkpoint's lower-control undershoot is tied to the x2-after-y1 coordinate slot or is a
generic coordinate effect.

Updated reducer:

`src/analysis/autoregressive_duplication_mechanism/phase4_coord_basin_band_flow.py`

Updated test:

`tests/analysis/autoregressive_duplication_mechanism/test_phase4_coord_basin_band_flow.py`

Regenerated artifact:

`/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/coord_basin_band_flow_layer17_head1`

## Added Fields

The reducer now emits:

- `coord_slot`: conservative phase-based slot inference.
  - `box_start/pre_x1` -> `x1`
  - `post_y1/pre_x2` -> `x2`
  - other known slot phases are wired but absent from the current targeted panel.
- `generated_coord_bin`: parsed from `generated_token_text` when it is a coord token.
- `generated_minus_target_bin`
- `masked_top1_minus_target_bin`
- `patched_top1_minus_target_bin`
- `control_top1_minus_target_bin`
- `patched_top1_abs_error_delta_from_masked`
- `control_top1_abs_error_delta_from_masked`

Important caveat: for `box_start/pre_x1`, the generated token is usually
`<|box_start|>`, not a coordinate token, so `generated_coord_bin` and
`generated_minus_target_bin` are intentionally absent. The x1 and x2 phases are therefore
not symmetric for generated-coordinate trajectory analysis.

## Whole-Head Record-33 Comparison

Scope: `aux_latest_ckpt32` and `none_latest_ckpt32`, record `33`,
component `whole_head`.

| checkpoint | phase | slot | generated-target | masked top1-target | patched top1-target | patched abs-error delta | exact delta | wide delta | lower-control delta | upper-control delta |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| aux_latest_ckpt32 | box_start/pre_x1 | x1 | n/a | 37.1667 | -18.8333 | 41.3333 | 0.000902 | 0.000419 | -0.000410 | -0.000630 |
| aux_latest_ckpt32 | post_y1/pre_x2 | x2 | 62.0000 | 2.2500 | -4.6667 | 3.5833 | -0.007962 | -0.056979 | 0.049362 | -0.015932 |
| none_latest_ckpt32 | box_start/pre_x1 | x1 | n/a | 54.2500 | 48.6667 | -3.5833 | 0.000000 | -0.000131 | -0.000492 | -0.003266 |
| none_latest_ckpt32 | post_y1/pre_x2 | x2 | 215.3333 | 15.7500 | 1.9167 | -13.5000 | 0.016706 | 0.116555 | 0.075453 | 0.011399 |

The auxiliary x2 phase has the undershoot signature:

- generated coordinate is far above target on average (`+62`);
- masked top1 is near target (`+2.25`);
- patched top1 moves below target (`-4.67`);
- patched absolute error worsens by `+3.58`;
- exact and wide target-near mass decrease;
- lower-control mass increases while upper-control mass decreases.

For `none_latest_ckpt32` x2, the trajectory is different:

- generated coordinate is also far above target (`+215.33`);
- masked top1 is above target (`+15.75`);
- patched top1 moves close to target (`+1.92`);
- patched absolute error improves by `-13.5`;
- exact and wide target-near mass increase strongly;
- lower and upper control deltas are both positive.

So both checkpoints may start from an above-target generated x2 trajectory, but the
patching direction differs. `none_latest_ckpt32` uses whole-head context to move toward
the target neighborhood. `aux_latest_ckpt32` overshoots downward into a lower control
basin.

## Auxiliary X2 Row-Level Whole-Head Trace

Scope: `aux_latest_ckpt32`, record `33`, phase `post_y1/pre_x2`,
component `whole_head`.

| row | generated-target | masked top1-target | patched top1-target | control top1-target | patched abs-error delta | wide delta | patched control direction |
|---:|---:|---:|---:|---:|---:|---:|---|
| 17 | -941 | 48 | 48 | 48 | 0 | 0.010142 | above_target |
| 18 | 293 | -4 | -4 | -4 | 0 | -0.010332 | below_target |
| 19 | -258 | -4 | -4 | -4 | 0 | -0.001972 | below_target |
| 20 | 174 | 0 | -7 | -14 | 7 | -0.079195 | below_target |
| 21 | 173 | 0 | -12 | -12 | 12 | -0.183812 | below_target |
| 22 | 182 | -1 | -8 | -9 | 7 | 0.001156 | below_target |
| 23 | 191 | 7 | -5 | -5 | -2 | -0.101951 | below_target |
| 24 | 193 | 13 | -6 | 1 | -7 | -0.020229 | below_target |
| 25 | 180 | -13 | -20 | -8 | 7 | -0.058580 | not_control_anchor |
| 26 | 185 | -4 | -11 | 3 | 7 | -0.104870 | not_control_anchor |
| 27 | 177 | -10 | -22 | -2 | 12 | -0.055108 | not_control_anchor |
| 28 | 195 | -5 | -5 | -6 | 0 | -0.078994 | below_target |

Rows 20 and 21 remain the cleanest damage cases: the masked top1 is exact target, but
the patch moves below target and removes wide target-near mass.

Rows 23 and 24 show that lower-control movement can improve absolute top1 distance while
still reducing wide target-near mass. That means absolute top1 distance alone is not a
sufficient criterion for recovery; the distributional mass movement matters.

## Mechanism Update

The auxiliary undershoot is concentrated in the x2-after-y1 window of this target panel.
The current evidence supports this local trajectory:

1. the generated x2 token is usually above the next target x2;
2. the masked readout is often already near or exactly at the target;
3. patching toward the control/whole-head state moves top1 below target;
4. the move can reduce exact/wide target-near mass even when top1 distance sometimes
   improves.

This suggests the auxiliary checkpoint is not simply failing to perceive the object or
failing to enter coordinate-token mode. It is entering a directional coordinate basin
inside the coordinate manifold, specifically a lower-control x2 attractor after y1.

## Next Deterministic Step

The next cheap post-hoc split is to separate x2 rows by generated-minus-target sign and
masked-top1 status:

- generated above target vs below target;
- masked exact/near target vs old/control/other;
- patch improves top1 distance but loses mass vs patch damages both top1 distance and
  mass.

That should distinguish "corrective downward movement" from "overcorrection into lower
control attractor" without another GPU run.

## Verification

- Focused direct test harness passed for
  `tests/analysis/autoregressive_duplication_mechanism/test_phase4_coord_basin_band_flow.py`.
- `python -m py_compile` passed for the reducer and CLI.
- Regenerated the 410-row band-flow artifact with slot and trajectory fields.
- Extracted whole-head summaries for `aux_latest_ckpt32` and `none_latest_ckpt32`, record
  `33`, phases `box_start/pre_x1` and `post_y1/pre_x2`.
- Inspected auxiliary x2 row-level traces from the regenerated enriched JSONL.
