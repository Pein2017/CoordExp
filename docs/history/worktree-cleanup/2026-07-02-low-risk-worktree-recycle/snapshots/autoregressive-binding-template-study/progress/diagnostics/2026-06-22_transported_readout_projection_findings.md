# Transported Readout Projection Findings

Date: 2026-06-22

Worktree: `/data/CoordExp/.worktrees/autoregressive-binding-template-study`

## Question

The hidden-patch continuation probe showed that first-step layer-input patches can
repair the next descriptor and produce a syntactically valid object span, while
still emitting a wrong spatial instance. The next concern was whether the earlier
descriptor-projection analysis was biased by using the static output embedding
direction:

```text
E_target - E_competitor
```

This pass adds and smokes a transported readout-gradient direction at the actual
decoder component site:

```text
q_l = grad_{component_l}(z_target - z_competitor)
```

The intent is to distinguish state components that are causally aligned with the
model's live readout surface from components that only look aligned under a
static embedding basis.

## Implementation

Added component projection bases:

```text
transported_desc_target_minus_boundary_variant_top
orthogonal_to_transported_desc_target_minus_boundary_variant_top
```

For descriptor-target post-box paired rows, the direction is computed by a
backward pass through the active prefix forward. Model parameters are frozen for
the diagnostic pass; only the captured layer-input/component tensor is cloned
with `requires_grad=True`.

Relevant code:

```text
src/analysis/autoregressive_binding_template_ablation/realized_behavior_bridge.py
tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py
```

Test coverage:

```text
python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -k 'normalize_patch_component_projection_bases or transported_readout_projection_rows'
python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -k 'component_projection or transported_readout or projection_basis or projection_bases or causal_activation_patch'
python -m py_compile src/analysis/autoregressive_binding_template_ablation/realized_behavior_bridge.py scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py
```

Results:

```text
2 passed
72 passed
py_compile passed
```

## Tiny GPU Smoke Scope

Input selected rows:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/post_box_counterfactual_boundary_selection/v1_desc_first_strict_selected_descriptor_flips_v1/post_box_counterfactual_boundary_selected_rows.jsonl
```

Pair config:

```text
configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml
```

Common settings:

```text
stage=trajectory-hidden-causal-activation-patch-continuation
max_rows=2
case_count=1
state_row_count=2
row_count=18 per run
patch_component_sites=layer_input
patch_direction_bases=paired_post_box_baseline_minus_current
patch_strengths=1.0
patch_continuation_application_modes=first_step
target_next_kinds=desc
max_new_tokens=32
```

Artifacts:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_continuation_probe/post_box_layer_input_transport_static_m4_m1_smoke_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_continuation_probe/post_box_layer_input_transport_static_m8_m1_smoke_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_continuation_probe/post_box_layer_input_transport_static_m16_m1_smoke_v1
```

This is a tiny smoke over one backpack descriptor-basin flip case, not a
validation run.

## Result

Baseline flip continuation:

```text
desc=person
bbox=[269,105,323,252]
target_bbox_iou=0
```

Depth comparison for the flip row:

| source -> target | full layer_input delta | transported projection | transported orthogonal | static projection | static orthogonal |
| --- | --- | --- | --- | --- | --- |
| -4 -> -1 | backpack, IoU 0 | backpack, IoU 0 | person, IoU 0 | person, IoU 0 | backpack, IoU 0 |
| -8 -> -1 | backpack, IoU 0 | backpack, IoU 0 | person, IoU 0 | person, IoU 0 | backpack, IoU 0 |
| -16 -> -1 | person, IoU 0 | person, IoU 0 | person, IoU 0 | person, IoU 0 | person, IoU 0 |

Projection magnitudes:

| source -> target | transported projection norm | transported cosine | static projection norm | static cosine |
| --- | ---: | ---: | ---: | ---: |
| -4 -> -1 | 34.618 | 0.208 | 29.762 | 0.178 |
| -8 -> -1 | 18.855 | 0.206 | 5.221 | 0.057 |
| -16 -> -1 | 0.445 | 0.050 | 0.349 | 0.039 |

The transported gradient basis is late-forming. It is large enough at `-4/-8`
to isolate descriptor repair, but it is nearly absent at `-16`. The static
descriptor embedding basis is not equivalent: at `-4/-8`, static projection
alone does not repair the descriptor, while its orthogonal residual does.

Among the flip-row full component and projection rows, no patch produced
target-overlapping geometry. The repaired descriptor still lands in the wrong
spatial basin, often around boxes such as:

```text
[874,103,903,255]
[862,103,903,260]
```

This is far from the selected target box:

```text
[420,165,491,329]
```

Artifact nuance: the `-8 -> -1` run has a small nonzero overall mean target IoU
because a baseline-role interpolation row emitted `[420,165,491,320]`
(`IoU=0.945`). That is useful as a separate clean-state interpolation sanity
check, but it is not evidence that the flip-row component/projection rescue
recovered the selected target geometry.

## Mechanistic Update

The current best interpretation is now more specific:

1. A late transported readout axis can carry enough information to repair the
   descriptor token and descriptor span.
2. Static output-embedding projection can misclassify where that causal
   descriptor-readout information lives; the static orthogonal residual can
   contain live readout-aligned information after nonlinear transport.
3. Descriptor repair is insufficient for instance binding. The generated box can
   move into a different same-class or wrong-object spatial basin even when the
   descriptor is repaired and syntax is valid.
4. The useful descriptor-readout axis appears after mid-late processing. The
   `-16 -> -1` failure suggests the origin of the wrong/right object routing is
   earlier than the final readout axis, but not itself linearly available as that
   axis at `-16`.

## Revised Next Directions

Promote:

1. **Path-averaged transported gradients.** The single-point transported
   gradient is useful, but the paired hidden deltas are large. Compute
   `mean_tau grad(z_target-z_competitor)` along the failed-to-baseline path and
   compare linearized margin prediction with actual margin change.
2. **Descriptor-vs-box binding split probes.** Keep descriptor repair as only
   one outcome column. Add target-box overlap, nearest same-class GT overlap,
   nearest previous-object overlap, and emitted-box basin labels as first-class
   continuation outcomes.
3. **Formation-time map.** The useful transported descriptor axis emerges by
   `-8/-4`, not `-16`, so map layers and token positions where the direction
   becomes causal. This should include descriptor onset, box slots, box close,
   post-box boundary, and next-object onset.
4. **Component mediation at late layers.** Run transported projection over
   `layer_input`, `self_attn`, and `mlp`, then path-patch/clamp attention and MLP
   outputs to decide whether MLP is carrying object-binding state, amplifying a
   wrong basin, or merely translating attention state into readout logits.
5. **False-negative guidance bridge.** For missing objects, test whether a
   transported descriptor/readout axis can be created by language guidance while
   visual candidate features already support the object. This directly separates
   visual non-perception from language-side routing fragility.

Demote:

```text
static descriptor cosine as a semantic information percentage
all_steps continuation as natural boundary unfolding evidence
descriptor-token rescue as sufficient evidence of object rescue
large cross-case low-rank transfer before emitted-basin taxonomy is stable
```

Falsifiers to keep the loop honest:

- If path-averaged transported projections still fail to predict margin movement,
  the relevant object-state transition is not well described by a low-rank local
  readout direction.
- If target-box rescue appears only when descriptor repair is absent, descriptor
  and geometry routes may be competing rather than sequential.
- If component path patching shows no attention/MLP mediator, the apparent
  layer-input carrier may be a residual stream cursor state rather than a
  submodule-specific object representation.
- If false-negative rows cannot produce a transported descriptor axis even under
  strong language guidance, visual-side absence becomes a stronger explanation.

Dynamic adjustment remains welcome: if any one of these probes exposes a
promising route toward the final mechanism picture, it should override the
static roadmap and become the next local center of gravity.
