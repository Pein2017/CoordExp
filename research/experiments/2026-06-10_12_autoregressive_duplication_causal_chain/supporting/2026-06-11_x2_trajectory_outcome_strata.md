# X2 Trajectory Outcome Strata

Date: 2026-06-11

## Scope

This note uses the enriched slot/trajectory band-flow rows to separate corrective
downward movement from overcorrection into a lower-control attractor. It focuses on
`coord_slot=x2`, component `whole_head`, because this has been the most diagnostic route
and coordinate-basin site in the target panel.

Source rows:

`/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/coord_basin_band_flow_layer17_head1/coord_basin_band_flow_rows.jsonl`

Outcome taxonomy:

- `improve_top1_gain_mass`: patched top1 absolute error improves and wide-radius target
  mass increases.
- `improve_top1_lose_mass`: patched top1 absolute error improves but wide-radius target
  mass decreases.
- `damage_top1_gain_mass`: patched top1 absolute error is equal/worse but wide-radius
  target mass increases.
- `damage_top1_lose_mass`: patched top1 absolute error is equal/worse and wide-radius
  target mass decreases.

## Whole-Head X2 Summary

| case | generated sign | masked status | outcome counts | mean generated-target | mean masked-target | mean patched-target | mean abs-error delta | wide delta |
|---|---|---|---|---:|---:|---:|---:|---:|
| aligner_parent_ckpt1824 #54 | below 11 | exact 1, near 6, far 4 | gain+damage 7, lose+damage 3, gain+improve 1 | -63.09 | -13.27 | -14.00 | 0.73 | 0.00647 |
| aux_latest_ckpt32 #33 | above 10, below 2 | exact 2, near 6, far 4 | lose+damage 8, lose+improve 2, gain+damage 2 | 62.00 | 2.25 | -4.67 | 3.58 | -0.05698 |
| no_aligner_parent_ckpt3668 #36 | above 11 | near 7, far 4 | gain+damage 4, lose+damage 3, gain+improve 2, lose+improve 2 | 460.45 | 6.27 | 3.45 | -1.55 | 0.01912 |
| no_aligner_parent_ckpt3668 #48 | above 10, below 2 | exact 2, near 7, far 3 | gain+damage 9, gain+improve 2, lose+damage 1 | 299.42 | -0.33 | -0.83 | -0.50 | 0.04748 |
| none_latest_ckpt32 #33 | above 12 | far 12 | gain+improve 9, gain+damage 2, lose+damage 1 | 215.33 | 15.75 | 1.92 | -13.50 | 0.11656 |

The auxiliary checkpoint is the outlier: most x2 whole-head rows both damage/equal the
top1 absolute error and lose wide-radius target-near mass. `none_latest_ckpt32` is the
opposite: most rows improve top1 distance and gain wide-radius target-near mass.

## Auxiliary vs None Row Trace

### `aux_latest_ckpt32`, record 33, x2, `whole_head`

| row | gen-target | masked-target | patched-target | abs-error delta | wide delta | direction | outcome |
|---:|---:|---:|---:|---:|---:|---|---|
| 17 | -941 | 48 | 48 | 0 | 0.010142 | above | damage+mass |
| 18 | 293 | -4 | -4 | 0 | -0.010332 | below | damage-mass |
| 19 | -258 | -4 | -4 | 0 | -0.001972 | below | damage-mass |
| 20 | 174 | 0 | -7 | 7 | -0.079195 | below | damage-mass |
| 21 | 173 | 0 | -12 | 12 | -0.183812 | below | damage-mass |
| 22 | 182 | -1 | -8 | 7 | 0.001156 | below | damage+mass |
| 23 | 191 | 7 | -5 | -2 | -0.101951 | below | improve-mass |
| 24 | 193 | 13 | -6 | -7 | -0.020229 | below | improve-mass |
| 25 | 180 | -13 | -20 | 7 | -0.058580 | not-control | damage-mass |
| 26 | 185 | -4 | -11 | 7 | -0.104870 | not-control | damage-mass |
| 27 | 177 | -10 | -22 | 12 | -0.055108 | not-control | damage-mass |
| 28 | 195 | -5 | -5 | 0 | -0.078994 | below | damage-mass |

Rows 20 and 21 are the strongest overcorrection evidence:

- generated x2 is far above target;
- masked top1 is exact target;
- patched top1 moves below target;
- exact/wide mass drops.

Rows 23 and 24 are the mixed form:

- top1 absolute error improves;
- but wide-radius mass still drops;
- the move is therefore not a clean distributional repair.

### `none_latest_ckpt32`, record 33, x2, `whole_head`

| row | gen-target | masked-target | patched-target | abs-error delta | wide delta | direction | outcome |
|---:|---:|---:|---:|---:|---:|---|---|
| 20 | 274 | 10 | 10 | 0 | 0.000488 | above | damage+mass |
| 21 | 281 | 9 | 9 | 0 | 0.027653 | above | damage+mass |
| 22 | 254 | -19 | -7 | -12 | 0.022422 | below | improve+mass |
| 23 | 180 | 39 | 6 | -33 | 0.161999 | not-control | improve+mass |
| 24 | 247 | 10 | 10 | 0 | -0.002089 | above | damage-mass |
| 25 | 176 | 13 | 0 | -13 | 0.258553 | not-control | improve+mass |
| 26 | 197 | 21 | 1 | -20 | 0.247208 | above | improve+mass |
| 27 | 185 | 28 | -5 | -23 | 0.131250 | below | improve+mass |
| 28 | 207 | 27 | 7 | -20 | 0.214596 | above | improve+mass |
| 29 | 188 | 16 | -4 | -12 | 0.128390 | below | improve+mass |
| 30 | 194 | 15 | -5 | -10 | 0.076160 | below | improve+mass |
| 31 | 201 | 20 | 1 | -19 | 0.132035 | above | improve+mass |

Here the same high-generated-x2 condition mostly leads to proper correction: patched
top1 moves toward the target and wide-radius mass rises.

## Mechanism Update

The current evidence distinguishes two kinds of downward movement:

1. **Corrective downward movement**: generated and masked top1 are above or far from
   target; patch lowers top1 toward target and increases target-near mass. This is the
   dominant `none_latest_ckpt32` x2 whole-head pattern.
2. **Overcorrection into lower-control attractor**: generated x2 is above target, masked
   top1 is already exact or near target, and patch lowers top1 below target while
   decreasing target-near mass. This is the dominant `aux_latest_ckpt32` x2 whole-head
   pattern.

This makes the auxiliary checkpoint more troubling than a simple "too much downward
correction" story. The damage depends on the masked readout already being target-near.
The auxiliary control state seems to impose a lower x2 basin even when the masked state
has the correct coordinate available.

## Consequence for Future Probes

If we return to GPU causal probes, the highest-yield auxiliary cases are not arbitrary
duplicates. They should target rows matching:

- `coord_slot=x2`;
- generated-minus-target positive;
- masked top1 exact or near target;
- patched/control top1 below target;
- wide-radius mass decreases.

Rows 20 and 21 of `aux_latest_ckpt32`, record 33, are the cleanest current examples.

## Verification

- Read the regenerated 410-row enriched band-flow JSONL.
- Stratified `coord_slot=x2`, component `whole_head`, across target-panel cases.
- Computed generated-sign, masked-status, and outcome-count summaries.
- Inspected row-level traces for `aux_latest_ckpt32` and `none_latest_ckpt32`, record
  `33`.
- No code or generated artifacts were changed for this note.
