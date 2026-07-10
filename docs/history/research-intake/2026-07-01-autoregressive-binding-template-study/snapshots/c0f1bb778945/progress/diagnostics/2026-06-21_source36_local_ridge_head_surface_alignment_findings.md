---
doc_id: progress.diagnostics.source36_local_ridge_head_surface_alignment_findings
layer: progress
doc_type: diagnostic-findings
status: active-evidence
evidence_scope: tiny-one-state-source36-y2-local-ridge-token-surface-alignment
domain: autoregressive-binding-template-ablation
updated: 2026-06-21
branch: codex/autoregressive-binding-template-study
---

# Source36 Local-Ridge Head Surface Alignment Findings

## Scope

This note follows:

```text
progress/diagnostics/2026-06-21_adapter_surface_probe_decomposition_findings.md
```

The adapter-surface decomposition split source36/object14/y2 as the clearest
adapter-distorted local ridge case:

```text
target y2: 996
final LM-head top1: 999
base surface local top bins: 998, 997, 999, 996
effective surface local top bin: 999
target adapter-delta rank among requested probes: 6 of 6
```

This follow-up asks which candidate boundary heads write residual/value
components aligned with the local ridge direction:

```text
target-minus-attractor: 996 - 999
target-minus-near-ridge: 996 - 998, 996 - 997
```

All evidence here is readout-only token-surface alignment. It is not a head
intervention and does not prove causality by itself.

## Helper Added

Residual token-surface alignment now accepts direct coordinate-negative bases:

```text
coord_996_minus_coord_999
coord_996_minus_coord_998
coord_996_minus_coord_997
row_target_bbox_y2_minus_coord_999
row_target_bbox_y2_minus_coord_998
row_target_bbox_y2_minus_coord_997
```

The row-specific forms resolve `row_target_bbox_y2` from the row target bbox
and keep `negative_label=coord_<bin>` in the output metadata.

Verification:

```text
python -m py_compile src/analysis/autoregressive_binding_template_ablation/realized_behavior_bridge.py tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py
  exit 0
python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -q
  370 passed
```

## Artifacts

Input state:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/coordinate_threshold_hidden_state_probe/source36_object14_y2_panel_v1/selected_rows.jsonl
```

Focused two-head residual PCA:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_residual_pca/source36_object14_y2_heads21_2_26_10_regions_current_obj_prectx_v1
```

Focused two-head local-ridge token-surface alignment:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_residual_token_surface_alignment/source36_object14_y2_local_ridge_heads21_2_26_10_v1
```

Default six-head residual PCA:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_residual_pca/source36_object14_y2_default_heads_regions_current_obj_prectx_v1
```

Default six-head local-ridge token-surface alignment:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_residual_token_surface_alignment/source36_object14_y2_local_ridge_default_heads_v1
```

The default-head alignment wrote:

```text
row_count: 84
surface_basis_count: 7
eligible_residual_state_count: 12
residual_token_surface_drop_counts: {}
model_perturbation_ran: false
training_ran: false
readout_only: true
```

## Main Table

Sorted by residual projection for `coord_996_minus_coord_999`.
Negative means the row points toward the 999 attractor side relative to target
996.

| layer:head | source region | residual projection | residual cosine | contribution projection | contribution cosine |
| --- | --- | ---: | ---: | ---: | ---: |
| 26:10 | pre_prefix_non_image_context | -5.0583 | -0.2327 | -4.7085 | -0.2088 |
| 18:0 | pre_prefix_non_image_context | -1.0641 | -0.1721 | -1.2905 | -0.2030 |
| 21:3 | pre_prefix_non_image_context | -0.6948 | -0.0828 | -0.4871 | -0.0573 |
| 19:8 | pre_prefix_non_image_context | -0.1605 | -0.0569 | -0.1314 | -0.0465 |
| 26:10 | current_object_ref_boundaries | -0.0542 | -0.2113 | -0.0320 | -0.1062 |
| 18:0 | current_object_ref_boundaries | -0.0157 | -0.0375 | -0.0335 | -0.0792 |
| 19:8 | current_object_ref_boundaries | 0.0016 | 0.0045 | 0.0207 | 0.0549 |
| 19:9 | current_object_ref_boundaries | 0.0052 | 0.0852 | 0.0060 | 0.0954 |
| 21:3 | current_object_ref_boundaries | 0.0067 | 0.0091 | 0.0605 | 0.0792 |
| 21:2 | current_object_ref_boundaries | 0.1053 | 0.0413 | 0.0298 | 0.0113 |
| 19:9 | pre_prefix_non_image_context | 0.1256 | 0.0344 | 0.0535 | 0.0145 |
| 21:2 | pre_prefix_non_image_context | 0.7803 | 0.0712 | 0.8066 | 0.0720 |

## Neighbor Checks

For `coord_996_minus_coord_998`:

```text
26:10 pre_prefix_non_image_context:
  residual_projection = -1.2671
  contribution_projection = +0.3379

21:2 pre_prefix_non_image_context:
  residual_projection = +1.2544
  contribution_projection = +1.2876
```

For `coord_996_minus_coord_997`:

```text
26:10 pre_prefix_non_image_context:
  residual_projection = +1.1814
  contribution_projection = +2.8624

21:2 pre_prefix_non_image_context:
  residual_projection = +1.1830
  contribution_projection = +1.4809
```

## Mechanistic Reading

The strongest local-ridge readout signal is not attached to the current object
boundary tokens. It is aligned with `pre_prefix_non_image_context`.

```text
26:10 pre-prefix:
  Strongly writes toward 999 over target 996.
  Also writes toward 998 over target 996 in residual projection.
  But it writes toward target 996 over 997.

21:2 pre-prefix:
  Writes toward target 996 over 999, 998, and 997.
  This looks like a countervailing correction head rather than the collapse
  source for this state.
```

This is a sharper hypothesis than "the model likes nearby coord tokens":

```text
The source36 999 basin has a strong history-aligned readout component.
The local attractor candidate is more aligned with a pre-prefix value/residual
route than with the current object-ref boundary span.
```

That is consistent with the surface decomposition:

```text
Hidden/readout already tilts hard to 999.
The token_embeddings_adapter then amplifies 999 even more.
```

In other words, the readout already contains a pre-prefix-aligned component
toward the 999 basin. Causal origin still needs intervention, but the adapter
may be amplifying that local-ridge component, especially the one visible at
head 26:10.

## Next Probe

The next causal test should intervene on the suspected route, not just read it:

```text
source36/object14/y2
head: 26:10
source region: pre_prefix_non_image_context
score surfaces:
  p996
  p999
  p998
  p997
  coord_996_minus_coord_999
  coord_996_minus_coord_998
  coord_996_minus_coord_997
```

Best next implementation direction:

```text
Extend value-region intervention/continuation scoring so local coordinate
family scores are first-class outputs, then suppress or project-subtract
26:10 pre-prefix components and test whether 999 loses to 996.
```

Alternative cheap follow-up:

```text
Run the same local-ridge token-surface alignment for source123/object5/y2 and
source145/object20/y2 to see whether their previously separated surface failure
modes have different routing signatures.
```
