# Targeted Record 33 Coordinate Band-Flow

Date: 2026-06-11

## Scope

This note records a post-hoc coordinate-band analysis for the targeted layer-17/head-1
record-33 panel. It follows the coordinate-basin attraction note by splitting observed
top-k coordinate probability mass into row-local bands:

- `target_exact`: the exact next coordinate bin.
- `wide-radius`: top-k bins within radius 8 of the target bin.
- `old_anchor`: masked-state top-k bins away from the target and not in the control set.
- `control_anchor`: control-state top-k bins away from the target and not in the old set.
- `bridge_anchor`: patched-state nearby bins that are not exact target, old, or control.

All masses are lower-bound diagnostics from available top-k coordinate bins and
probabilities, not full-softmax mass totals.

## Implementation

Added a reusable reducer:

- Source module:
  `src/analysis/autoregressive_duplication_mechanism/phase4_coord_basin_band_flow.py`
- CLI:
  `scripts/analysis/run_autoregressive_duplication_phase4_coord_basin_band_flow.py`
- Unit test:
  `tests/analysis/autoregressive_duplication_mechanism/test_phase4_coord_basin_band_flow.py`

The reducer reads Phase-4 state-patch rows with `masked_coord_top_bins`,
`patched_coord_top_bins`, `control_coord_top_bins`, and matching top-k probabilities,
then writes enriched rows plus a grouped Markdown/JSON summary.

## Artifact Handles

- Source rows:
  `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/key_value_component_expansion_layer17_head1/attention_score_bias_patch_rows.jsonl`
- Output directory:
  `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/coord_basin_band_flow_layer17_head1`
- Summary JSON:
  `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/coord_basin_band_flow_layer17_head1/phase4_coord_basin_band_flow_summary.json`
- Report:
  `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/coord_basin_band_flow_layer17_head1/phase4_coord_basin_band_flow_report.md`
- Enriched rows:
  `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/coord_basin_band_flow_layer17_head1/coord_basin_band_flow_rows.jsonl`

Run command:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/analysis/run_autoregressive_duplication_phase4_coord_basin_band_flow.py \
  --rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/key_value_component_expansion_layer17_head1/attention_score_bias_patch_rows.jsonl \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/coord_basin_band_flow_layer17_head1
```

## Main Slice Result

Scope: `none_latest_ckpt32`, record `33`, phase `post_y1/pre_x2`,
patch direction `masked_to_control`, patch kind `attention_key_value_state_patch`.

| component | old-anchor delta | exact-target delta | wide-radius delta | control-anchor delta | bridge-anchor delta | patched top1 bands |
|---|---:|---:|---:|---:|---:|---|
| duplicate_basin | -0.104725 | 0.0148022 | 0.0858679 | 0.0350912 | 0.0716406 | control 6, old 6 |
| visual_near_ring | -0.0566266 | 0.0052902 | 0.0167510 | 0.0016677 | 0.0284761 | control 6, old 6 |
| visual_far_background | -0.0237422 | -0.0001157 | 0.0057196 | 0.0025878 | 0.0059724 | control 4, old 8 |
| non_region_complement | -0.0661996 | 0.0054043 | 0.0195081 | 0.0012813 | 0.0383474 | control 5, old 7 |
| whole_head | -0.136830 | 0.0167059 | 0.116555 | 0.0868522 | 0.0565025 | bridge 1, control 10, exact 1 |

## Interpretation

This resolves the previous "basin relocation versus clean target direction" ambiguity.

The `duplicate_basin` patch strongly removes old-anchor top-k mass and adds target/bridge
mass, but its top-1 bin still splits evenly between old and control anchors. This matches
the earlier read that the duplicate route is target-logit-aligned but does not always
complete the coordinate-basin escape by itself.

The `whole_head` patch is the strongest basin relocation operator: it has the largest
old-anchor removal, largest wide-radius gain, and by far the largest control-anchor gain.
Its patched top-1 band is control-like in 10 of 12 rows, exact target in 1 row, and bridge
in 1 row. This explains why whole-head patching looked better under coordinate-distance
metrics even though projected target-direction analysis showed cancellation.

The anti-target projected components remain structured. `visual_near_ring` and
`non_region_complement` reduce old-anchor mass and increase bridge/near-target mass, but
barely add control-anchor mass. They appear to weaken the old basin without fully
selecting the control basin. `visual_far_background` is close to inert for this main
slice: small old removal, near-zero exact-target gain, and no top-1 band shift beyond the
masked distribution.

The current mechanism picture is therefore:

1. masked state: old-anchor top-1 dominates 8 of 12 rows;
2. duplicate-basin route: old mass is reduced and target/bridge mass rises, but top-1
   often remains old;
3. whole-head context: basin relocation completes into the control-anchor neighborhood;
4. cancellation: broader context helps basin relocation while mixing directions that are
   not as cleanly target-logit-aligned as the duplicate-basin component.

## Verification

- Focused direct test harness passed for
  `tests/analysis/autoregressive_duplication_mechanism/test_phase4_coord_basin_band_flow.py`.
- `python -m py_compile` passed for the new source module and CLI.
- The materializer completed on the target panel with `row_count=410` and
  `group_count=35`.
- The local pytest wrapper still prints the known compact `Pytest: No tests collected`
  message in this worktree; the tee log for the focused run showed the expected test
  execution path, and the direct harness was used as the narrow verification signal.
