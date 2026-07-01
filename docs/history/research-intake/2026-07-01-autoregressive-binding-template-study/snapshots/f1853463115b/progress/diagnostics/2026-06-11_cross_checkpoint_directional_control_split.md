# Cross-Checkpoint Directional Control Split

Date: 2026-06-11

## Scope

This note uses the directional `control_anchor` split across the full target panel. The
goal is to decide whether the lower-control attraction found in `aux_latest_ckpt32` is
unique, or whether it appears in other checkpoints with different consequences.

Source artifact:

`/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/coord_basin_band_flow_layer17_head1/phase4_coord_basin_band_flow_summary.json`

Rows artifact:

`/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/coord_basin_band_flow_layer17_head1/coord_basin_band_flow_rows.jsonl`

## Duplicate-Basin Component

Scope: `post_y1/pre_x2`.

| case | lower-control delta | upper-control delta | exact delta | wide delta | old delta | label |
|---|---:|---:|---:|---:|---:|---|
| aligner_parent_ckpt1824 #54 | 0.018121 | 0.002870 | 0.002686 | 0.006253 | -0.018043 | lower+ wide+ |
| aux_latest_ckpt32 #33 | 0.047930 | -0.011718 | -0.004539 | -0.044645 | -0.055965 | lower+ wide- |
| no_aligner_parent_ckpt3668 #36 | 0.006906 | 0.005127 | -0.003750 | -0.013674 | -0.061186 | lower+ wide- |
| no_aligner_parent_ckpt3668 #48 | 0.005793 | 0.008002 | 0.003091 | 0.010681 | -0.028823 | lower+ wide+ |
| none_latest_ckpt32 #33 | 0.020743 | 0.014349 | 0.014802 | 0.085868 | -0.104725 | lower+ wide+ |

The auxiliary checkpoint is not the only lower-control case, but it is the clearest
duplicate-basin case where lower-control gain co-occurs with both exact-target and
wide-radius loss. The no-aligner parent record 36 has a weaker related pattern, but its
upper-control delta is still positive, unlike the auxiliary checkpoint.

## Whole-Head Component

Scope: `post_y1/pre_x2`.

| case | lower-control delta | upper-control delta | exact delta | wide delta | old delta | label |
|---|---:|---:|---:|---:|---:|---|
| aligner_parent_ckpt1824 #54 | 0.015843 | 0.002624 | 0.002656 | 0.006472 | -0.013579 | lower+ wide+ |
| aux_latest_ckpt32 #33 | 0.049362 | -0.015932 | -0.007962 | -0.056979 | -0.056438 | lower+ wide- |
| no_aligner_parent_ckpt3668 #36 | 0.035525 | 0.009371 | -0.004105 | 0.019123 | -0.084518 | lower+ wide+ |
| no_aligner_parent_ckpt3668 #48 | 0.034930 | 0.028249 | 0.006199 | 0.047483 | -0.088847 | lower+ wide+ |
| none_latest_ckpt32 #33 | 0.075453 | 0.011399 | 0.016706 | 0.116555 | -0.136830 | lower+ wide+ |

This is the sharper contrast. Every whole-head case has positive lower-control gain, but
only `aux_latest_ckpt32` turns that gain into a negative wide-radius movement. In the
other cases, lower-control mass is part of broader coordinate-basin relocation; in the
auxiliary case it competes against the exact/wide target neighborhood.

## Whole-Head Top-1 Direction Conditioning

For whole-head patched top-1 rows that land in the control-anchor band:

| case | control-top1 rows | below rows | above rows | below wide mean | above wide mean | below exact mean | above exact mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| aligner_parent_ckpt1824 #54 | 9 | 8 | 1 | 0.003872 | 0.020576 | 0.003244 | 0.000423 |
| aux_latest_ckpt32 #33 | 9 | 8 | 1 | -0.059416 | 0.010142 | -0.011866 | -0.000619 |
| no_aligner_parent_ckpt3668 #36 | 10 | 4 | 6 | 0.049097 | 0.003006 | 0.010405 | -0.014766 |
| no_aligner_parent_ckpt3668 #48 | 9 | 1 | 8 | 0.000000 | 0.058707 | 0.000000 | 0.009660 |
| none_latest_ckpt32 #33 | 10 | 4 | 6 | 0.089556 | 0.103315 | 0.009029 | 0.015599 |

This row-conditioned view confirms that the auxiliary checkpoint is the outlier. Its
below-target whole-head control rows have negative wide and exact means, while the
below-target rows in the other cases are zero or positive on wide-radius movement.

## Mechanism Update

Lower-control attraction is not unique to `aux_latest_ckpt32`. The difference is the
sign of the accompanying target-neighborhood movement.

The target panel now separates into three directional regimes:

1. **Balanced relocation**: `none_latest_ckpt32` and no-aligner parent record 48.
   Lower-control and upper-control gains co-occur with positive exact/wide target
   movement.
2. **Weak already-control-like relocation**: aligner parent record 54. Lower-control
   gain is present but small, and exact/wide gains are small positive.
3. **Directional undershoot**: `aux_latest_ckpt32`. Lower-control gain is strong, but
   upper-control, exact-target, and wide-radius mass are negative for the route/context
   components that matter most.

No-aligner parent record 36 is intermediate: duplicate-basin is lower+wide-, but
whole-head restores wide-radius positivity. In `aux_latest_ckpt32`, whole-head does not
restore it; it strengthens the lower-control undershoot.

This makes the auxiliary checkpoint mechanistically distinct from both parent SFT
families. The auxiliary loss appears to create or expose a directional coordinate-basin
bias where the control state can undershoot the next coordinate, rather than merely
reducing duplication or improving target binding.

## Next Deterministic Step

The post-hoc evidence now points to a concrete follow-up: inspect whether the auxiliary
undershoot aligns with coordinate slot type or local coordinate trajectory. The cheapest
next step is to enrich band-flow rows with slot identity and generated-to-target delta:

- slot: x1/y1/x2/y2 inferred from phase and token position;
- generated coordinate minus target coordinate;
- patched/control top1 minus target coordinate;
- whether undershoot is stronger for x2 after y1 than for pre-x1.

This remains a data-analysis step over existing rows and does not require a GPU probe.

## Verification

- Read the regenerated 410-row directional band-flow summary JSON.
- Read the regenerated directional band-flow rows JSONL.
- Compared `duplicate_basin` and `whole_head` across all `post_y1/pre_x2` target-panel
  cases.
- Conditioned whole-head rows on patched top-1 control-anchor direction.
- No code or generated artifact changes were required for this note.
