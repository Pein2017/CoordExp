# Phase 4 Parent Panel Mask Geometry Controls

Date: 2026-06-11

## Scope

This note records the small parent-candidate mask-geometry control panel for the
layer 17 head 1 route/content path. It extends the earlier single-case geometry
controls into three 2x2 parent-level comparisons: fixed source candidate crossed
with alternate duplicate-basin mask geometry.

This is still a targeted mechanistic probe, not a global eval. The point is to
separate source identity/content from the spatial envelope selected by the visual
mask in candidate windows that were already promoted by earlier Phase 4 evidence.

## Inputs

Panel root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_basin_candidate_promotions_layer17_head1_shortlist/mask_geometry_control_parent_panel
```

Summary artifacts:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_basin_candidate_promotions_layer17_head1_shortlist/mask_geometry_control_parent_panel/parent_panel_summary/parent_panel_mask_geometry_summary.json
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_basin_candidate_promotions_layer17_head1_shortlist/mask_geometry_control_parent_panel/parent_panel_summary/parent_panel_mask_geometry_rows.jsonl
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_basin_candidate_promotions_layer17_head1_shortlist/mask_geometry_control_parent_panel/parent_panel_summary/parent_panel_mask_geometry_report.md
```

Run scope:

- Checkpoint: `none_latest_ckpt32`.
- Probe site: layer 17 head 1, self-attention output route/content projected additive patch.
- Execution: 12 one-row control cases launched over GPUs 0, 1, 2, and 3.
- Launcher result: 12 completed, 0 failed.
- Aggregated rows: 36 effect rows, covering `total_delta`, `route_delta_masked_values`, and `value_delta_control_route`.
- Original component references: 18 rows from the candidate route/content component competition summary.

## Total-Delta Sign Table

`total prob recovery` is `patched_target_prob - masked_target_prob` for the
coordinate target token in the selected rollout prefix. Positive means the patch
increases target-coordinate probability relative to the masked counterfactual.

| Pair | Classification | Fixed source | Mask geometry | Total prob recovery | r4 recovery | EAE recovery |
|---|---|---|---|---:|---:|---:|
| `record114_target4_target_bbox_vs_original` | `all_same_sign` | `record114_target4_original_duplicate_basin` | `original_mask` | +0.021131447 | +0.135981 | -26.545422 |
| `record114_target4_target_bbox_vs_original` | `all_same_sign` | `record114_target4_original_duplicate_basin` | `target_bbox_mask` | +0.003476221 | +0.010286 | -0.962471 |
| `record114_target4_target_bbox_vs_original` | `all_same_sign` | `record114_target4_target_row_bbox` | `original_mask` | +0.021982901 | +0.144898 | -26.985128 |
| `record114_target4_target_bbox_vs_original` | `all_same_sign` | `record114_target4_target_row_bbox` | `target_bbox_mask` | +0.002253607 | +0.007964 | -0.713257 |
| `record33_original_vs_target_desc` | `mask_geometry_dominated` | `record33_target25_original_duplicate_basin` | `original_mask` | +0.033239748 | +0.207377 | -13.279889 |
| `record33_original_vs_target_desc` | `mask_geometry_dominated` | `record33_target25_original_duplicate_basin` | `target_desc_mask` | -0.004879313 | -0.021305 | -3.005801 |
| `record33_original_vs_target_desc` | `mask_geometry_dominated` | `record33_target25_target_desc_sub_envelope` | `original_mask` | +0.033239748 | +0.207377 | -13.279889 |
| `record33_original_vs_target_desc` | `mask_geometry_dominated` | `record33_target25_target_desc_sub_envelope` | `target_desc_mask` | -0.004879313 | -0.021305 | -3.005801 |
| `record50_component_vs_original` | `mixed_interaction` | `record50_target9_component_row_9_person` | `component_mask` | +0.000017685 | +0.000133 | -29.348259 |
| `record50_component_vs_original` | `mixed_interaction` | `record50_target9_component_row_9_person` | `original_mask` | -0.008609720 | -0.055489 | +15.948261 |
| `record50_component_vs_original` | `mixed_interaction` | `record50_target9_original_duplicate_basin` | `component_mask` | +0.000000288 | +0.000006 | -6.087234 |
| `record50_component_vs_original` | `mixed_interaction` | `record50_target9_original_duplicate_basin` | `original_mask` | +0.007843459 | +0.057853 | -7.055685 |

## Pair Readouts

### `record33_original_vs_target_desc`

This is the cleanest mask-geometry result in the parent panel. Holding the
source fixed does not change the total-delta values at all, while switching the
mask from `original_mask` to `target_desc_mask` flips the sign from
`+0.033239748` to `-0.004879313`.

Interpretation: for this wine-glass burst window, the duplicate-basin effect is
not explained by the source candidate identity that supplied the cached route or
value. The spatial envelope selected by the mask is sufficient to decide whether
the patch is constructive or harmful for the coordinate target.

### `record114_target4_target_bbox_vs_original`

All four total deltas are positive, so this pair is not a sign-flip case. It is
still strongly mask-amplitude controlled: switching from `target_bbox_mask` to
`original_mask` adds about `+0.0177` to `+0.0197` probability recovery for both
fixed sources, while changing the source under a fixed mask only moves the
result by about `-0.0009` to `+0.0012`.

Interpretation: for this handbag window, both geometries preserve constructive
direction, but the broad original duplicate-basin mask supplies most of the
effective route/content recovery. The earlier single-case 2x2 finding therefore
extends from one source to both parent sources as an amplitude effect.

### `record50_component_vs_original`

This pair is a mixed interaction rather than a pure mask-geometry flip. The
`component_mask` cases are essentially null positive for both fixed sources
(`+0.000017685` and `+0.000000288`). The `original_mask` bifurcates by source:
it is harmful with the component-row source (`-0.008609720`) but constructive
with the original duplicate-basin source (`+0.007843459`).

Interpretation: this person window is not adequately described by mask geometry
alone. The original duplicate basin appears to require both the original source
and original mask to become constructive; transplanting the original mask onto
the component-row source produces downstream harm. This is a candidate for a
deeper value/source compatibility probe rather than a simple spatial-envelope
story.

## Mechanistic Takeaway

The parent panel supports a stronger version of the current picture:

- In some duplicate-burst windows, the coordinate-slot basin is controlled
  primarily by the visual mask geometry, not by the identity of the source row.
- In other windows, mask geometry controls amplitude while preserving the same
  constructive sign.
- A third regime exists where geometry and source identity interact; this should
  not be collapsed into the pure-geometry story.

The useful refinement is that `duplicate basin` is not a single scalar failure
mode. It looks like a routed coordinate attraction system with at least two
axes: spatial envelope selection and source/value compatibility. The next probes
should preserve that distinction when moving into hidden-state and attention
mechanism analysis.

## Follow-Up

Recommended next deterministic steps:

1. Use the `record33` pair as the canonical mask-geometry sign-control case for
   hidden-state basin probing, because source identity cancels out exactly in
   this panel.
2. Use the `record114` pair as the canonical amplitude-control case, because the
   original mask increases recovery by about `8x` to `10x` without changing sign.
3. Use the `record50` pair as the value/source compatibility case, especially to
   test whether the original duplicate source carries a coordinate-compatible
   value vector that the component-row source lacks.
4. Keep these parent-panel classes separate in later attention analysis instead
   of averaging them into one duplicate-basin aggregate.

Dynamic adjustment remains welcome: if one of these regimes yields a promising
path with more influence over the final mechanistic picture, it is worth diving
deeper and updating the task sequence around that path.
