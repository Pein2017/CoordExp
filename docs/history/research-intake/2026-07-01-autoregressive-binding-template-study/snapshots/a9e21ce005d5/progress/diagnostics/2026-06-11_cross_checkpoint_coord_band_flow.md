# Cross-Checkpoint Coordinate Band-Flow Contrast

Date: 2026-06-11

## Scope

This note uses the coordinate band-flow reducer to compare the targeted panel across
checkpoints, without launching new GPU probes. The goal is to separate the main
`none_latest_ckpt32` mechanism from effects that are already present in the parent SFT
checkpoints or changed by the auxiliary-loss checkpoint.

Source artifact:

`/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/coord_basin_band_flow_layer17_head1/phase4_coord_basin_band_flow_summary.json`

Band-flow report:

`/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/coord_basin_band_flow_layer17_head1/phase4_coord_basin_band_flow_report.md`

The mass fields are top-k lower-bound diagnostics, not full-softmax mass totals.

## None vs Auxiliary Checkpoint, Record 33

### `none_latest_ckpt32`, record 33, `post_y1/pre_x2`

| component | old-anchor delta | exact-target delta | wide-radius delta | control-anchor delta | bridge-anchor delta | patched top1 bands |
|---|---:|---:|---:|---:|---:|---|
| duplicate_basin | -0.104725 | 0.014802 | 0.085868 | 0.035091 | 0.071641 | control 6, old 6 |
| visual_near_ring | -0.056627 | 0.005290 | 0.016751 | 0.001668 | 0.028476 | control 6, old 6 |
| visual_far_background | -0.023742 | -0.000116 | 0.005720 | 0.002588 | 0.005972 | control 4, old 8 |
| non_region_complement | -0.066200 | 0.005404 | 0.019508 | 0.001281 | 0.038347 | control 5, old 7 |
| whole_head | -0.136830 | 0.016706 | 0.116555 | 0.086852 | 0.056502 | bridge 1, control 10, exact 1 |

### `aux_latest_ckpt32`, record 33, `post_y1/pre_x2`

| component | old-anchor delta | exact-target delta | wide-radius delta | control-anchor delta | bridge-anchor delta | patched top1 bands |
|---|---:|---:|---:|---:|---:|---|
| duplicate_basin | -0.055965 | -0.004539 | -0.044645 | 0.036213 | 0.018976 | control 9, old 3 |
| visual_near_ring | -0.026952 | 0.003408 | -0.005096 | -0.005973 | 0.015391 | control 7, old 4, exact 1 |
| visual_far_background | -0.009184 | 0.003431 | 0.000864 | -0.001911 | 0.003906 | control 8, old 3, exact 1 |
| non_region_complement | -0.030251 | 0.000563 | -0.004592 | 0.001457 | 0.010676 | control 7, old 3, exact 2 |
| whole_head | -0.056438 | -0.007962 | -0.056979 | 0.033430 | 0.020160 | control 9, old 3 |

The auxiliary-loss checkpoint does not simply amplify the `none_latest_ckpt32` repair
pattern. Its patched top-1 bands are already more control-like, but the `post_y1/pre_x2`
wide-radius and exact-target deltas are negative for `duplicate_basin` and `whole_head`.
That means the patch moves top-1 identity toward control anchors while reducing observed
top-k mass near the exact next coordinate. This is a different failure/repair geometry
from `none_latest_ckpt32`, where `whole_head` increases old-anchor removal, exact-target
mass, wide-radius mass, and control-anchor mass together.

## Phase Contrast Within Record 33

For both `none_latest_ckpt32` and `aux_latest_ckpt32`, `box_start/pre_x1` is weak compared
with `post_y1/pre_x2`.

At `box_start/pre_x1`:

- `none_latest_ckpt32` has no exact-target gain for any component, tiny wide-radius
  movement, and top-1 bands remain `control 4, old 8`.
- `aux_latest_ckpt32` has tiny exact-target gains but top-1 remains mostly old
  (`old 8-9` rows depending on component).

This supports keeping `post_y1/pre_x2` as the main deterministic coordinate-basin window
for record 33. The x2 slot after y1 is where route/basin mechanics separate clearly.

## Parent Checkpoint Contrast

### `aligner_parent_ckpt1824`, record 54, `post_y1/pre_x2`

The aligner-tuned parent has small deltas and is already mostly control-like:

- `duplicate_basin`: old delta `-0.018043`, exact delta `0.002686`, control delta
  `0.020991`, patched top1 `control 10, exact 1`.
- `whole_head`: old delta `-0.013579`, exact delta `0.002656`, control delta
  `0.018467`, patched top1 `control 9, old 1, exact 1`.

Here `whole_head` is not better than `duplicate_basin`: it has slightly less old-anchor
removal and slightly less control-anchor gain. That suggests the main `none_latest_ckpt32`
whole-head basin relocation story is not a universal property of the parent SFT state.

### `no_aligner_parent_ckpt3668`, records 36 and 48, `post_y1/pre_x2`

For both no-aligner parent cases, `whole_head` improves control-like relocation over
`duplicate_basin`.

Record 36:

- `duplicate_basin`: old `-0.061186`, wide `-0.013674`, control `0.012033`,
  patched top1 `control 10, other 1`.
- `whole_head`: old `-0.084518`, wide `0.019123`, control `0.044896`,
  patched top1 `control 10, exact 1`.

Record 48:

- `duplicate_basin`: old `-0.028823`, wide `0.010681`, control `0.013795`,
  patched top1 `control 9, old 2, exact 1`.
- `whole_head`: old `-0.088847`, wide `0.047483`, control `0.063179`,
  patched top1 `control 9, old 1, exact 2`.

So the whole-head relocation advantage is present in the no-aligner parent and the
`none_latest_ckpt32` checkpoint, but not in the aligner-tuned parent or the auxiliary-loss
checkpoint record-33 slice.

## Mechanism Update

The cross-checkpoint picture is not "same mechanism, different strength." It is at least
two regimes:

1. **Basin-relocation regime**: `none_latest_ckpt32` record 33 and no-aligner parent
   records 36/48. Whole-head context removes more old-anchor mass and adds more
   control-anchor or wide-radius mass than duplicate-basin alone.
2. **Already-control-like or altered regime**: aligner parent record 54 and
   `aux_latest_ckpt32` record 33. The patched top-1 distribution is more control-like
   already, and whole-head does not consistently add target-near mass beyond
   duplicate-basin.

This matters for the auxiliary-loss checkpoint: it may not repair duplication by the
same target-near basin escape seen in `none_latest_ckpt32`. It may instead reshape the
coordinate basin so that top-1 identity moves toward control anchors while exact/wide
target mass behaves differently. That deserves separate mechanistic treatment before
claiming the auxiliary loss shares the parent SFT mechanism.

## Next Deterministic Step

Use the enriched band-flow rows to inspect the auxiliary-loss row-level cases where
`whole_head` and `duplicate_basin` have negative wide-radius deltas despite control-like
top-1 movement. The immediate post-hoc target is:

- list rows where patched top1 is `control_anchor` but wide-radius delta is negative;
- compare generated coordinate, target coordinate, old/control/bridge bins;
- determine whether the "control anchor" is semantically correct but offset from exact
  target, or whether it is a different nearby attractor.

This is still data analysis from existing rows and does not require a GPU probe.

## Verification

- Read current summary JSON and generated report from the committed band-flow reducer.
- Extracted `post_y1/pre_x2` and `box_start/pre_x1` groups from the existing
  410-row artifact.
- No new code or generated artifacts were required for this note.
