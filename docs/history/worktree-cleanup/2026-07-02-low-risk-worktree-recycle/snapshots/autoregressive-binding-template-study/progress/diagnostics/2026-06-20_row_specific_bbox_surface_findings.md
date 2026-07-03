---
doc_id: progress.diagnostics.row_specific_bbox_surface_findings
layer: progress
doc_type: diagnostic-findings
status: branch-provenance
domain: research-history
summary: Row-specific current/target/previous bbox coord-token surface readout for the checkpoint-928 residual bridge.
tags: [progress, diagnostics, autoregressive-binding, coord-tokens, bbox-surface, residual, duplicate-onset]
updated: 2026-06-20
branch: codex/autoregressive-binding-template-study
---

# Row-Specific Bbox Surface Findings

## Scope

This note extends the residual token-surface probe with row-specific bbox
coordinate surfaces. The previous fixed-anchor readout showed that the residual
was not explained by generic `coord_0`, `coord_500`, or `coord_999` surfaces.
This run asks whether the residual is instead aligned with coordinates from the
row's own target box, current/open emitted box, or previous closed box.

Input residual rows:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_residual_pca/joint_boundary_head26_10_regions_curr_obj_prectx_v1/trajectory_boundary_head_residual_pca_rows.jsonl
```

Union artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_residual_token_surface_alignment/joint_boundary_head26_10_row_specific_bbox_surfaces_all36_union_v1/row_specific_bbox_token_surface_alignment_union.md
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_residual_token_surface_alignment/joint_boundary_head26_10_row_specific_bbox_surfaces_all36_union_v1/row_specific_bbox_token_surface_alignment_union_summary.json
```

Surface grammar added:

```text
row_<target|current|previous>_bbox_<mean|x1|y1|x2|y2>_minus_<box_end|im_end>
```

Examples:

```text
row_target_bbox_mean_minus_box_end
row_current_bbox_x1_minus_box_end
row_previous_bbox_y2_minus_im_end
```

Row count: `1080`.

Status counts:

```json
{"ok": 840, "skipped": 240}
```

The skipped rows are expected: first-object states do not have a previous
closed box.

## Aggregate Readout

Across all rows, residual alignment to row-specific bbox surfaces remains weak
and often negative:

| source | rows | ok | residual cosine | abs residual cosine | residual projection |
| --- | ---: | ---: | ---: | ---: | ---: |
| `current` | 360 | 360 | `-0.0423` | `0.0750` | `-0.508` |
| `target` | 360 | 360 | `-0.0444` | `0.0772` | `-0.667` |
| `previous` | 360 | 120 | `-0.1265` | `0.1265` | `-1.792` |

By slot:

| slot | ok | residual cosine | abs residual cosine | residual projection |
| --- | ---: | ---: | ---: | ---: |
| `mean` | 168 | `-0.0560` | `0.0772` | `-0.760` |
| `x1` | 168 | `-0.0691` | `0.0800` | `-1.632` |
| `y1` | 168 | `-0.0273` | `0.0893` | `0.968` |
| `x2` | 168 | `-0.0591` | `0.0801` | `-1.072` |
| `y2` | 168 | `-0.0646` | `0.0897` | `-1.300` |

This aggregate result is consistent with the fixed-anchor result: the residual
is not primarily a generic coordinate-token direction.

## Important Split

The critical `failure_default128_rescued / next_step_duplicate_onset` split
behaves differently from the aggregate.

For these rows, current and target bbox surfaces are positively aligned with
the residual:

| source / slot / negative | ok | residual cosine | residual projection | contribution cosine |
| --- | ---: | ---: | ---: | ---: |
| `current / mean / box_end` | 3 | `0.0584` | `3.785` | `0.148` |
| `current / x1 / box_end` | 3 | `0.0456` | `2.977` | `0.134` |
| `current / x2 / box_end` | 3 | `0.1113` | `7.115` | `0.210` |
| `target / mean / box_end` | 3 | `0.0424` | `2.789` | `0.133` |
| `target / x1 / box_end` | 3 | `0.0307` | `2.069` | `0.121` |
| `target / x2 / box_end` | 3 | `0.0735` | `4.708` | `0.172` |

Previous-box surfaces are skipped in this split because these are first-object
states in the selected residual panel.

## Interpretation

This is the first result in the current bridge that argues against a simple
"duplication = previous coordinate basin" story for the critical rescued
duplicate-onset rows.

The better current hypothesis is:

1. The residual factor is not a direct structural boundary token direction.
2. It is also not a generic coordinate-anchor direction.
3. In the critical rescued duplicate-onset split, it is weakly but consistently
   aligned with the row-specific current/target coordinate surfaces, especially
   `x2`.
4. Therefore, at least in these states, the residual may be helping complete a
   plausible object box while the higher-level autoregressive handoff or
   identity state is fragile.

Put differently: a visible duplicate may not start as "the model cannot see the
right coordinates." It may start as an object-handoff/binding problem where a
coordinate completion circuit still points toward a locally plausible box.

This aligns with the user-supplied warning that coordinate slots are stable
under strict schema/type loss: the coordinate basin itself may be healthy enough,
while object identity and continuation context decide whether the emitted span
belongs to the intended object, a duplicate, or an unmatched proposal.

## Boundaries

- Evidence scope is `36` state-region rows from `9` cases.
- This is readout-only, not a behavioral intervention.
- The `previous` source is sparse because many selected failure rows are
  first-object states.
- `current` and `target` boxes are often very close; further analysis should
  explicitly separate exact/easy target-current agreement from real divergence.

## Next Probe

The next high-value step is not another global coord-anchor scan. It should
separate binding from coordinate completion:

- join row-specific surface alignment with current-vs-target bbox deltas and
  IoU;
- select rows where current and target boxes diverge meaningfully;
- rerun continuation intervention/readout on those rows;
- compare whether residual directions preserve target-coordinate support while
  changing object handoff or termination behavior.

If that holds, the mechanistic picture becomes a two-factor failure:
coordinate completion remains locally competent, but the autoregressive object
state chooses the wrong "owner" of that box span.
