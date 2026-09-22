# Value Region Projection/Residual Factorial Probe

Date: 2026-06-20
Worktree: `/data/CoordExp/.worktrees/autoregressive-binding-template-study`

## Purpose

The previous structural residual decomposition showed that harmful value-region
full-vector subtraction could trigger a catastrophic coordinate basin, while
projection-only and residual-only subtraction did not cleanly reproduce the
same outlier. This note records the implementation of a direct factorial probe:
separate suppression scales for the structural-logit-gradient-span projection
and the off-span residual of each selected attention-head value contribution.

This is meant to test the threshold-interaction hypothesis, not to establish a
population-level effect.

## Implementation

Added a new value-region patch mode:

```text
direction_span_projection_residual_grid_subtract
```

The mode computes:

```text
patch = -(projection_scale * projected + residual_scale * residual)
projected = projection of region contribution into structural_logit_gradient_span
residual = contribution - projected
```

Key behavior:

- Requires `head_direction_bases=structural_logit_gradient_span`.
- Adds CLI axes:
  - `--value-region-projection-scales`
  - `--value-region-residual-scales`
- If either axis is omitted, `--value-region-scales` is used as that axis.
- Emits explicit row metadata:
  - `value_region_projection_scale`
  - `value_region_residual_scale`
  - `value_region_scale_pair`
- Keeps `value_region_intervention_scale=None` for grid rows, instead of
  pretending that a two-axis intervention has a single scalar scale.
- Includes projection/residual scales in continuation IDs and paired post-hoc
  fallback identities.
- Preserves random residual controls as a separate mode only; the new grid mode
  does not fan out random controls.

Changed tracked files:

- `src/analysis/autoregressive_binding_template_ablation/realized_behavior_bridge.py`
- `scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py`
- `tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py`

## Review

Subagent spec review found no spec-compliance issues.

Subagent code-quality review found no critical or important issues. Two minor
follow-ups were handled immediately:

- Added direct row-generation coverage for Cartesian grid fanout and fallback
  axis behavior.
- Added a `scale=None` guard for the residual-PC private patch helper path.

One minor downstream follow-up remains intentionally outside this commit:
coordinate-copy/top-tie post-hoc summaries are still mostly scalar-scale
oriented. The grid rows preserve pair metadata, and the main continuation
summary includes pair counts, but downstream filtering/grouping can be made more
pair-aware later if we use those analyzers heavily on grid artifacts.

## Verification

All commands were run in:

```text
/data/CoordExp/.worktrees/autoregressive-binding-template-study
```

Commands:

```text
python -m py_compile src/analysis/autoregressive_binding_template_ablation/realized_behavior_bridge.py scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py
python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -q
python -m pytest tests/analysis/test_value_contribution_head_group_tomography.py -q
git diff --check
```

Results:

- Bridge test file: `350 passed`
- Value contribution head-group tomography: `8 passed`
- `py_compile`: passed
- `git diff --check`: passed

## Bounded GPU Smoke

Selection artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_panel_selection/v3_source36_obj14_x2y2_factorial_smoke
```

Run artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_continuation/v4_source36_obj14_x2y2_harmful_combined_span_factorial_p012_r012_regions2_steps6_v1
```

Generated summary:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_factorial_summary/v1_source36_obj14_x2y2_harmful_combined_span_factorial
```

Scope:

- `source_line_idx=36`
- `image_id=3255`
- `object_idx=14`
- slots: `x2,y2`
- harmful combined heads: `20:14,20:3,27:14,27:15`
- value source regions:
  - `current_partial_coords`
  - `current_partial_box_span`
- projection scales: `0.0,0.5,1.0,1.5,2.0`
- residual scales: `0.0,0.5,1.0,1.5,2.0`
- continuation steps: `6`

Run result:

- `row_count=362`
- `source_row_count=2`
- `continuation_count=100`
- `first_step_count=100`

## First Mechanistic Signal

The targeted outlier reproduced the prior full-vector catastrophe only when the
structural-span projection and off-span residual were both high.

Key first-step observations:

- `y2` target was `996`; baseline was `<|coord_999|>`.
- Residual-only at `residual=2.0` kept `y2` at `<|coord_999|>`.
- Projection-only at `projection=2.0` kept `y2` at `<|coord_999|>`.
- `projection=1.5,residual=2.0` changed `y2` to `<|coord_234|>`.
- `projection=2.0,residual=2.0` changed `y2` to `<|coord_234|>`.
- The catastrophic `y2` error at `<|coord_234|>` was `abs_error=762`.
- The effect appeared for both tested source regions:
  - `current_partial_coords`
  - `current_partial_box_span`

`x2` behaved differently:

- `x2` target and baseline were both `456`.
- Residual scale `>=0.5` usually shifted `456 -> 444/445`, a local error of
  about `11-12`.
- `x2` did not enter the catastrophic far-coordinate basin in this smoke.

This supports a sharper local hypothesis:

```text
Some catastrophic coordinate basin switches are not carried by the structural
projection or residual component alone. They require enough projection-aligned
boundary/object-role pressure and enough off-span value-state damage at the
same time.
```

## Interpretation Boundary

This is a tiny targeted smoke, not a population result. The value is that it
turns a vague "full vector is catastrophic" observation into a testable
mechanistic interaction:

```text
projection-only: local or no effect
residual-only: local or no effect
projection + residual above threshold: far-coordinate basin jump
```

The next useful step is to run the same factorial grid over the full strict
20-state object-step panel, then split by slot, source region, and harmful vs
repair head groups.
