---
doc_id: progress.diagnostics.residual_token_surface_alignment_findings
layer: progress
doc_type: diagnostic-findings
status: branch-provenance
domain: research-history
summary: Local head-value residual alignment against boundary and coord-token logit-gradient surfaces for the checkpoint-928 binding-template study.
tags: [progress, diagnostics, autoregressive-binding, token-surface, coord-tokens, residual, attention-head]
updated: 2026-06-20
branch: codex/autoregressive-binding-template-study
---

# Residual Token-Surface Alignment Findings

## Scope

This note records the first model-backed token-surface readout for the
trajectory-boundary residual bridge.

Code stage:

```text
trajectory-boundary-head-residual-token-surface-alignment
```

Input residual rows:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_residual_pca/joint_boundary_head26_10_regions_curr_obj_prectx_v1/trajectory_boundary_head_residual_pca_rows.jsonl
```

Checkpoint/config:

```text
configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml
```

Head and source regions:

- head: `layer=26`, `head=10`
- state-region rows: `36`
- cases: `9`
- value regions: `current_object_ref_boundaries`, `current_prefix_all`,
  `pre_prefix_non_image_context`

The readout stays in attention-head value / `self_attn.o_proj` input space.
It compares residual and full contribution vectors to local token logit-gradient
directions. It does not compare against raw token embeddings.

## Artifact Roots

Smoke:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_residual_token_surface_alignment/smoke_head26_10_two_surfaces_max1_v1
```

Scaled shards:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_residual_token_surface_alignment/joint_boundary_head26_10_structural_surfaces_all36_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_residual_token_surface_alignment/joint_boundary_head26_10_coord_mean_surfaces_all36_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_residual_token_surface_alignment/joint_boundary_head26_10_coord_anchor_0_500_999_surfaces_all36_v1
```

Union:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_residual_token_surface_alignment/joint_boundary_head26_10_token_surface_union_structural_coordmean_anchor_all36_v1/trajectory_boundary_head_residual_token_surface_alignment_union.md
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_residual_token_surface_alignment/joint_boundary_head26_10_token_surface_union_structural_coordmean_anchor_all36_v1/trajectory_boundary_head_residual_token_surface_alignment_union_summary.json
```

Union row count: `432`, all `ok`.

## Main Finding

The behaviorally active residual is not explained by a simple local structural
token surface.

Structural token-gradient readout is essentially zero for the residual:

| surface | rows | mean residual cosine | mean abs residual cosine | mean contribution cosine |
| --- | ---: | ---: | ---: | ---: |
| `box_end_minus_object_ref_start` | 36 | `2.43e-07` | `8.65e-07` | `-0.208` |
| `box_end_minus_object_ref_end` | 36 | `3.51e-09` | `4.86e-07` | `-0.312` |
| `box_end_minus_im_end` | 36 | `-3.33e-07` | `1.51e-06` | `-0.0385` |
| `object_ref_boundaries_minus_im_end` | 36 | `-3.32e-07` | `8.78e-07` | `0.220` |

This is expected in part because the residual PCA input subtracts the structural
logit-gradient span, but it is still an important control: the later behavioral
effect of residual subtraction cannot be described as merely removing the
obvious next-token boundary direction.

Coord-token surfaces are present but weak:

| surface | rows | mean residual cosine | mean abs residual cosine | mean contribution cosine |
| --- | ---: | ---: | ---: | ---: |
| `coord_mean_minus_box_end` | 36 | `-0.0470` | `0.0673` | `0.0582` |
| `coord_mean_minus_im_end` | 36 | `-0.0437` | `0.0631` | `0.0280` |
| `coord_0_minus_box_end` | 36 | `-0.0134` | `0.0770` | `0.0850` |
| `coord_500_minus_box_end` | 36 | `-0.0332` | `0.0534` | `0.0646` |
| `coord_999_minus_box_end` | 36 | `-0.0600` | `0.0669` | `0.0528` |

The largest residual coord-surface cosines are mostly clean/control rows rather
than the failure rows where the residual intervention had the strongest
behavioral meaning. For the critical
`failure_default128_rescued / next_step_duplicate_onset` split, residual
alignment remains modest:

| surface | rows | mean residual cosine | mean residual projection |
| --- | ---: | ---: | ---: |
| `coord_0_minus_box_end` | 3 | `0.0456` | `2.98` |
| `coord_mean_minus_box_end` | 3 | `-0.00989` | `-0.690` |
| `coord_999_minus_box_end` | 3 | `-0.0434` | `-2.75` |

## Interpretation

The current best read is that the residual bridge is not a direct
next-token-surface vector. It is more likely a latent routing or state-preparation
factor whose effect becomes visible over continuation, especially near object
handoff and duplicate-onset states.

The full contribution vector still has substantial structural token-surface
components, while the residual after structural-span subtraction does not. This
split is useful:

- the full head contribution carries ordinary boundary/object-ref pressure;
- the residual factor carries something less aligned with immediate
  boundary-token logits;
- the residual can still be behaviorally relevant in continuation, so the
  mechanism may live in autoregressive state preparation rather than the
  current-token readout surface.

Coord-token basin evidence is not absent, but it is weak and sign-structured
rather than a clean direct attraction to `<|coord_0|>`, `<|coord_500|>`, or
`<|coord_999|>`. The coordinate slot basin should remain a live hypothesis, but
the next probe should use row-specific coord anchors or continuation-level
coordinate-copy outcomes rather than only fixed global coord anchors.

## Boundaries

- Evidence scope is a compact `36` state-region residual panel from `9` cases.
- This is a readout-only analysis. It does not replace the earlier behavioral
  residual intervention evidence.
- Residual vectors are already structural-span residuals, so structural-surface
  near-zero alignment is partly by construction.
- Fixed coord anchors `0/500/999` are coarse. They do not test row-specific
  current-box or target-box coordinate basins.

## Next Probe

The next high-value bridge is a row-specific coordinate surface:

- build token-gradient directions for the current emitted bbox coordinates and
  target bbox coordinates;
- compare residual alignment to those row-specific surfaces;
- join with continuation coordinate-copy outcomes;
- test whether duplicate-onset residuals point away from target-box surfaces or
  toward previous/current-box surfaces before visible duplication.

This should distinguish a true coordinate basin mechanism from a more abstract
handoff/routing-state mechanism.
