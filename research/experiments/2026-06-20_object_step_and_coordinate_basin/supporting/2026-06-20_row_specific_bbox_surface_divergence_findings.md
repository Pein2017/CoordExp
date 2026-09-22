---
doc_id: progress.diagnostics.row_specific_bbox_surface_divergence_findings
layer: progress
doc_type: diagnostic-findings
status: branch-provenance
domain: research-history
summary: Artifact-level selector joining row-specific coordinate-token surface alignment with current/target/previous bbox divergence for the checkpoint-928 residual bridge.
tags: [progress, diagnostics, autoregressive-binding, coord-tokens, bbox-surface, selector, duplicate-onset]
updated: 2026-06-20
branch: codex/autoregressive-binding-template-study
---

# Row-Specific Bbox Surface Divergence Findings

## Scope

This note records the next bridge after the row-specific bbox surface readout.
The previous readout asked whether the residual aligns with target, current,
or previous coordinate-token surfaces. This selector adds geometry: for each
prefix state, it compares the target bbox, current open emitted bbox, and
previous closed bbox, then ranks states where coordinate completion and object
ownership may separate.

Input:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_residual_token_surface_alignment/joint_boundary_head26_10_row_specific_bbox_surfaces_all36_union_v1/row_specific_bbox_token_surface_alignment_union_rows.jsonl
```

Output:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/row_specific_bbox_surface_divergence/joint_boundary_head26_10_row_specific_bbox_surfaces_all36_v2/row_specific_bbox_surface_divergence.md
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/row_specific_bbox_surface_divergence/joint_boundary_head26_10_row_specific_bbox_surfaces_all36_v2/row_specific_bbox_surface_divergence_summary.json
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/row_specific_bbox_surface_divergence/joint_boundary_head26_10_row_specific_bbox_surfaces_all36_v2/row_specific_bbox_surface_divergence_state_rows.jsonl
```

Evidence scope: readout-only, artifact-level selector over the existing
`1080` row-specific surface rows. The selector collapses these to `36`
state-region rows: `12` prefix states crossed with `3` value-source regions,
across `9` cases.

Correction note: an earlier `v1` artifact collapsed all three value-source
regions into one state row and is superseded by `v2`. Do not use `v1` for
owner-hypothesis interpretation.

## Implementation Contract

New stage:

```text
analyze-row-specific-bbox-surface-divergence
```

CLI:

```bash
python scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py \
  --stage analyze-row-specific-bbox-surface-divergence \
  --row-specific-bbox-surface-rows /path/to/row_specific_surface_rows.jsonl \
  --bbox-divergence-top-k 20 \
  --output-root /path/to/output_root
```

Output files:

```text
row_specific_bbox_surface_divergence_state_rows.jsonl
row_specific_bbox_surface_divergence_summary.json
row_specific_bbox_surface_divergence.md
```

The selector uses the same execution-prefix precedence as the surface readout:
`assistant_prefix_text`, then `trajectory_state_prefix_text`, then
`trajectory_prefix_text`. It keeps `value_source_region` in the grouping key,
so projections from `current_object_ref_boundaries`, `current_prefix_all`, and
`pre_prefix_non_image_context` are not overwritten or averaged silently.

## Results

State rows:

```text
row_count: 36
source_row_count: 1080
state_count: 12
case_count: 9
```

Target-current geometry:

```json
{"divergent": 6, "moderate_delta": 24, "strong_divergence": 6}
```

Surface-owner hypotheses:

```json
{
  "current_open_box_owner_bias": 2,
  "least_negative_current_surface": 11,
  "least_negative_previous_surface": 5,
  "least_negative_target_surface": 7,
  "weak_current_open_box_owner_bias": 5,
  "weak_target_box_owner_bias": 6
}
```

Value-source regions:

```json
{
  "current_object_ref_boundaries": 12,
  "current_prefix_all": 12,
  "pre_prefix_non_image_context": 12
}
```

Families:

```json
{"desc_first": 33, "geometry_first": 3}
```

Buckets:

```json
{"clean": 12, "failure_default128_rescued": 12, "failure_joint128_only": 12}
```

Onset labels:

```json
{"neutral": 15, "next_step_duplicate_onset": 6, "next_step_unmatched_onset": 15}
```

Mean target-current IoU:

```text
0.822869
```

Mean target-current max absolute coord delta:

```text
14.6667
```

## High-Value Rows

The top duplicate-onset rows split in a useful way.

1. `desc_first-885-8-0-desc_end`
   - bucket: `failure_default128_rescued`
   - onset: `next_step_duplicate_onset`
   - geometry: `strong_divergence`
   - target-current IoU: `0.477981`
   - max coord delta: `47`
   - owner hypothesis: `current_open_box_owner_bias` in
     `current_prefix_all` and `current_object_ref_boundaries`,
     `weak_current_open_box_owner_bias` in `pre_prefix_non_image_context`
   - target/current residual projection by region:
     - `current_prefix_all`: `4.22251` / `5.44392`
     - `current_object_ref_boundaries`: `2.80277` / `3.89028`
     - `pre_prefix_non_image_context`: `1.34255` / `2.02159`

2. `desc_first-2685-33-11-desc_end`
   - bucket: `clean`
   - onset: `next_step_duplicate_onset`
   - geometry: `strong_divergence`
   - target-current IoU: `0.299754`
   - max coord delta: `29`
   - owner hypothesis is region-split and all projections are negative:
     - `current_prefix_all`: `least_negative_current_surface`,
       target/current/previous `-2.82169` / `-2.69145` / `-2.71499`
     - `pre_prefix_non_image_context`: `least_negative_target_surface`,
       target/current/previous `-0.263925` / `-0.355574` / `-0.448263`
     - `current_object_ref_boundaries`: `least_negative_previous_surface`,
       target/current/previous `-0.296348` / `-0.281719` / `-0.274025`

This is the main new selection value. Duplicate onset is not one homogeneous
coordinate phenomenon: at least in this selected panel, one duplicate-onset
state has positive current-open-box owner evidence across all three regions,
while another strong-divergence duplicate-onset state has only relative
least-negative winners and splits across current, target, and previous by
region.

The selector also surfaces previous-basin controls:

```text
desc_first-2685-33-11-desc_end / current_object_ref_boundaries
desc_first-139-0-11-desc_end / selected regions
```

These are not dominant duplicate explanations yet; they are controls for
separating previous-basin reuse from current/target owner competition.

## Interpretation

The selector strengthens the current two-factor picture:

1. Coordinate completion is often close to the target/current geometry, but not
   exact enough to ignore.
2. Strong duplicate-onset rows split by both state and value-source region:
   `desc_first-885` has positive current-open-box owner evidence, while
   `desc_first-2685` has only least-negative relative winners and changes owner
   by region.
3. Previous-box reuse exists as a candidate, but its strongest appearance here
   is a least-negative regional control rather than a universal duplicate
   mechanism.
4. The next hidden-state or intervention probe should not average regions
   together. It should preserve `value_source_region` and contrast the top
   strong-divergence duplicate-onset rows directly.

## Next Probe

Use the selector output as the input panel for the next expensive analysis.

Primary contrast:

```text
desc_first-885-8-0-desc_end / current_prefix_all
desc_first-885-8-0-desc_end / current_object_ref_boundaries
vs
desc_first-2685-33-11-desc_end / current_prefix_all
desc_first-2685-33-11-desc_end / pre_prefix_non_image_context
desc_first-2685-33-11-desc_end / current_object_ref_boundaries
```

Recommended test:

- run hidden-state/readout or residual-PC continuation on both states;
- preserve the row-specific bbox divergence metadata and `value_source_region`
  in the output rows;
- check whether the positive current-open-box-biased duplicate state differs
  from the all-negative region-split duplicate state in object-handoff,
  stop/continue, or post-box-boundary logits;
- include previous-basin rows as controls, not as the default duplicate
  explanation.

If this contrast holds under intervention, the mechanism picture becomes more
precise: duplication is not merely coordinate attraction, but a competition
between local coordinate completion and the ownership/handoff state that decides
which object span those coordinates belong to.
