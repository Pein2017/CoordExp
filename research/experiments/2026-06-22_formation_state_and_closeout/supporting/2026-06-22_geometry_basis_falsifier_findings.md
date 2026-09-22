# Geometry-Basis Falsifier Findings

Date: 2026-06-22

Worktree: `/data/CoordExp/.worktrees/autoregressive-binding-template-study`

## Question

The path-averaged transported descriptor-readout probe showed that descriptor
repair is real but not sufficient: the patched continuation emits `backpack`,
then lands in a wrong same-description spatial basin instead of the selected
target box.

This pass tests the fastest geometry-side falsifier: project the same natural
post-box hidden delta onto the existing target-bbox-vs-completed-box output
embedding direction:

```text
target_bbox_tokens_minus_completed_box_tokens
orthogonal_to_target_bbox_tokens_minus_completed_box_tokens
```

The question is whether the selected target geometry is locally recoverable at
the late post-box `layer_input` state once we use a box-oriented basis instead
of a descriptor-readout basis.

## Continuation Falsifier Scope

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
row_count=17 per run
patch_component_sites=layer_input
patch_direction_bases=paired_post_box_baseline_minus_current
patch_component_projection_bases=[
  target_bbox_tokens_minus_completed_box_tokens,
  orthogonal_to_target_bbox_tokens_minus_completed_box_tokens,
  transported_desc_target_minus_boundary_variant_top
]
patch_strengths=[1.0]
patch_continuation_application_modes=[first_step]
target_next_kinds=[desc]
max_new_tokens=32
```

Artifacts:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_continuation_probe/post_box_layer_input_geometry_basis_m4_m1_smoke_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_continuation_probe/post_box_layer_input_geometry_basis_m8_m1_smoke_v1
```

All requested projection bases realized. No projection basis skipped.

This remains a one-case backpack descriptor-basin flip smoke, not a validation
run.

## Continuation Result

Flip-row target:

```text
target desc: backpack
target bbox: [420,165,491,329]
completed-box tokens used by the flip row: [466,98,520,174]
```

Flip-row behavior:

| source -> target | patch | generated desc | emitted basin | target IoU | nearest same-desc GT IoU | projection norm | cosine |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| -4 -> -1 | baseline no patch | person | unmatched_valid | 0.000 | null | null | null |
| -4 -> -1 | full layer_input delta | backpack | same_desc_gt_basin | 0.000 | 0.440 | null | null |
| -4 -> -1 | target-bbox projection | person | unmatched_valid | 0.000 | null | 0.516 | 0.003 |
| -4 -> -1 | target-bbox orthogonal | backpack | same_desc_gt_basin | 0.000 | 0.440 | 166.820 | 0.003 |
| -4 -> -1 | transported descriptor projection | backpack | same_desc_gt_basin | 0.000 | 0.423 | 34.618 | 0.208 |
| -8 -> -1 | baseline no patch | person | unmatched_valid | 0.000 | null | null | null |
| -8 -> -1 | full layer_input delta | backpack | same_desc_gt_basin | 0.000 | 0.423 | null | null |
| -8 -> -1 | target-bbox projection | person | unmatched_valid | 0.000 | null | 1.413 | -0.015 |
| -8 -> -1 | target-bbox orthogonal | backpack | same_desc_gt_basin | 0.000 | 0.423 | 91.296 | -0.015 |
| -8 -> -1 | transported descriptor projection | backpack | same_desc_gt_basin | 0.000 | 0.580 | 18.855 | 0.206 |

The direct target-bbox projection is tiny and behaviorally inert. It preserves
the failed `person` descriptor and does not move geometry toward the selected
box. The orthogonal remainder keeps the descriptor repair and still lands in
the wrong backpack basin.

The only target-overlap row in the `-8 -> -1` aggregate is again a baseline-role
interpolation sanity row:

```text
role=baseline
patch_label=interpolate_source_to_target_alpha_0p75
desc=backpack
bbox=[420,165,491,320]
target IoU=0.945
```

It is not a flip-row rescue.

## Descriptor-Boundary Probe-Bin Readout

Because the continuation falsifier missed, I ran a non-continuation side-channel
readout at the same post-box descriptor boundary with explicit coordinate probe
bins:

```text
target bins: 420,165,491,329
completed bins: 466,98,520,174
wrong backpack bins: 874,103,903,255,862,260
baseline wrong-person bins: 269,105,323,252
```

Artifacts:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/post_box_layer_input_geometry_probe_bins_m4_m1_smoke_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/post_box_layer_input_geometry_probe_bins_m8_m1_smoke_v1
```

Post-hoc probe-support summaries:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/post_box_layer_input_geometry_probe_bins_m4_m1_smoke_v1/probe_support_analysis
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/post_box_layer_input_geometry_probe_bins_m8_m1_smoke_v1/probe_support_analysis
```

Schema caveat: these are descriptor-target states, so `target_coord_bin` is
null. The existing `analyze-trajectory-causal-patch-probe-support` reducer
therefore reports `target_probe_support_row_count=0` and
`offtarget_probe_support_row_count=0`. That is correct for its schema. The
useful read here is the manual group comparison among requested probe bins.

Flip-row coordinate-subspace group summaries:

| source -> target | patch | next token | target min rank / max prob | completed min rank / max prob | wrong-backpack min rank / max prob | wrong-person min rank / max prob |
| --- | --- | --- | --- | --- | --- | --- |
| -4 -> -1 | baseline no patch | person | 99 / 0.00216 | 80 / 0.00296 | 76 / 0.00315 | 405 / 0.00010 |
| -4 -> -1 | full layer_input delta | back | 93 / 0.00219 | 78 / 0.00281 | 78 / 0.00281 | 388 / 0.00011 |
| -4 -> -1 | target-bbox projection | person | 102 / 0.00212 | 82 / 0.00289 | 77 / 0.00308 | 408 / 0.00010 |
| -4 -> -1 | target-bbox orthogonal | back | 92 / 0.00222 | 77 / 0.00284 | 77 / 0.00284 | 390 / 0.00011 |
| -4 -> -1 | transported descriptor projection | back | 96 / 0.00225 | 77 / 0.00307 | 81 / 0.00271 | 433 / 0.00009 |
| -8 -> -1 | baseline no patch | person | 99 / 0.00216 | 80 / 0.00296 | 76 / 0.00315 | 405 / 0.00010 |
| -8 -> -1 | full layer_input delta | back | 95 / 0.00213 | 80 / 0.00273 | 77 / 0.00291 | 388 / 0.00011 |
| -8 -> -1 | target-bbox projection | person | 106 / 0.00199 | 82 / 0.00290 | 79 / 0.00308 | 408 / 0.00010 |
| -8 -> -1 | target-bbox orthogonal | back | 95 / 0.00216 | 78 / 0.00277 | 78 / 0.00277 | 388 / 0.00011 |
| -8 -> -1 | transported descriptor projection | back | 93 / 0.00227 | 82 / 0.00273 | 78 / 0.00291 | 413 / 0.00010 |

At this immediate descriptor boundary, the selected target bins are not
rank-competitive. Wrong-backpack and completed-box bins are consistently
stronger within the coordinate-only subspace, but none are top-10 and full-vocab
coordinate probabilities are effectively tiny. This side-channel therefore
should not be read as a coordinate-slot destination-basin trace. It only says
the target geometry is not already locally privileged at the descriptor onset
state.

## Mechanistic Update

The simple late target-bbox direction is falsified for this case.

The natural baseline-minus-current hidden delta is almost orthogonal to the
target-bbox-vs-completed-box output embedding direction at both `-4 -> -1` and
`-8 -> -1`. Removing that tiny target-bbox component leaves the descriptor
repair intact. Applying only that component leaves the failure intact.

This sharpens the current split:

1. The late descriptor-readout direction is a real causal descriptor lever.
2. The target-bbox output-embedding direction is not the missing target-instance
   lever at the post-box descriptor boundary.
3. The wrong same-description spatial basin is not explained by a simple
   completed-box-to-target output embedding correction.
4. The target-instance variable likely appears later during coordinate-slot
   generation, or lives in a richer hidden/attention state not aligned with a
   mean coordinate-token output direction.

## Continuation Score-Trace Follow-Up

After the descriptor-boundary side channel proved insufficient, I added a
continuation score trace to the existing continuation helper. When
`--probe-coord-bins` is supplied, the helper now asks generation for per-step
scores and records compact rows only for generated coordinate tokens:

```text
generated_coord_score_trace_schema_version=1
generated_coord_score_probe_bins
generated_coord_score_available_step_count
generated_coord_score_trace_count
generated_coord_score_trace=[
  generation_step_index,
  generated_token_id,
  generated_token_text,
  generated_coord_bin,
  generated_coord_slot_index,
  coord_mass_full_vocab,
  coord_top1_bin,
  coord_top1_prob_full_vocab,
  coord_top1_prob_coord_only,
  coord_probe_bins=[bin, token_id, logit, prob_full_vocab, prob_coord_only, rank_coord_only, top1_distance]
]
```

Code:

```text
src/analysis/autoregressive_binding_template_ablation/realized_behavior_bridge.py
tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py
```

TDD verification:

```text
python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -k 'records_coord_score_trace'
```

Red result before implementation:

```text
0 passed, 1 failed
TypeError: _generate_bridge_prefix_tail_with_decoder_patch() got an unexpected keyword argument 'probe_coord_bins'
```

Green result after implementation:

```text
1 passed
```

Focused verification after wiring:

```text
python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -k 'trajectory_hidden_causal_activation_patch_continuation or records_coord_score_trace or component_projection or projection_basis or projection_bases'
python -m py_compile src/analysis/autoregressive_binding_template_ablation/realized_behavior_bridge.py scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py
```

Results:

```text
20 passed
py_compile passed
```

Score-traced continuation artifacts:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_continuation_probe/post_box_layer_input_geometry_basis_scoretrace_m4_m1_smoke_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_continuation_probe/post_box_layer_input_geometry_basis_scoretrace_m8_m1_smoke_v1
```

Common scope stayed the same as the continuation falsifier, with the added probe
panel:

```text
--probe-coord-bins 420,165,491,329,466,98,520,174,874,103,903,255,862,260,269,105,323,252
```

The score trace gives the missing answer. For descriptor-repairing flip rows,
the generated coordinate slots are already dominated by the wrong backpack
basin:

| source -> target | patch | emitted bbox | slot | emitted coord/top1 | target rank / full-vocab prob | wrong-backpack rank / full-vocab prob | completed rank / full-vocab prob |
| --- | --- | --- | ---: | --- | --- | --- | --- |
| -4 -> -1 | full layer_input delta | [874,103,903,259] | x1 | 874 / 874 | 323 / 0.000632 | 1 / 0.017349 | 505 / 0.000383 |
| -4 -> -1 | full layer_input delta | [874,103,903,259] | y1 | 103 / 103 | 34 / 0.013165 | 1 / 0.045951 | 18 / 0.016904 |
| -4 -> -1 | full layer_input delta | [874,103,903,259] | x2 | 903 / 903 | 963 / 0.000000039 | 1 / 0.073472 | 486 / 0.00000179 |
| -4 -> -1 | full layer_input delta | [874,103,903,259] | y2 | 259 / 259 | 202 / 0.000130 | 1 / 0.046294 | 158 / 0.000401 |
| -4 -> -1 | target-bbox orthogonal | [874,103,903,259] | x1 | 874 / 874 | 324 / 0.000629 | 1 / 0.017263 | 483 / 0.000406 |
| -4 -> -1 | transported descriptor projection | [874,103,903,255] | x1 | 874 / 874 | 325 / 0.000632 | 1 / 0.017339 | 482 / 0.000408 |
| -8 -> -1 | full layer_input delta | [874,103,903,255] | x1 | 874 / 874 | 330 / 0.000616 | 1 / 0.017988 | 487 / 0.000397 |
| -8 -> -1 | transported descriptor projection | [862,103,903,260] | x1 | 862 / 862 | 316 / 0.000637 | 1 / 0.017500 | 495 / 0.000387 |

The target-bbox projection rows still emit wrong-person boxes and their
coordinate slots are dominated by the wrong-person bin, not the target bin:

```text
-4 target-bbox projection: [269,105,323,252], slot ranks for target x/y/x/y = 539,42,310,214
-8 target-bbox projection: [269,104,323,252], slot ranks for target x/y/x/y = 536,44,302,206
```

The score trace therefore rules out a pure downstream decode-selection story for
the descriptor-repairing rows. At the coordinate slots themselves, the logits
already make the wrong same-description basin the coordinate top-1. The selected
target coordinate bins are not merely second-best; at x1/x2/y2 they are often
hundreds of coordinate ranks away.

## Revised Next Directions

Promote immediately after the score trace:

1. **Pre-coordinate slot intervention.** Build prefixes ending at the repaired
   descriptor/box-start/x1 boundary and test whether target-vs-wrong-basin hidden
   deltas can move x1 before the wrong basin becomes top-1. The most important
   state is the first coordinate slot, because x1 already chooses `874`/`862`
   over target `420`.
2. **Target-vs-wrong-same-desc basis.** The current geometry basis compares the
   selected target against the completed box, but the realized wrong basin is a
   different same-description GT. Add or emulate target-vs-emitted-wrong-basin
   directions at coordinate-slot states, especially target `420` versus wrong
   `874/862` at x1.
3. **Slot-local formation map.** Run only the few slot states needed to localize
   basin takeover: descriptor onset, object_ref_end, box_start, pre-x1, and
   post-x1. The score trace already shows the wrong basin is decisive by x1, so
   broader late-layer descriptor sweeps are lower value.
4. **Writer localization after x1 handle exists.** Once a pre-x1 target-vs-wrong
   basis moves x1 or at least changes target rank, localize candidate heads/value
   regions. Avoid isolated head sweeps until the x1 basin handle is real.

Demote for this case:

```text
more descriptor-gradient variants
late target_bbox_tokens_minus_completed_box_tokens as a rescue basis
post-box descriptor-boundary coordinate probe bins as a destination-basin trace
downstream sampling/argmax as the main cause after coordinate logits are formed
mean target IoU over mixed baseline/flip roles
```

Fast falsifier for the next implementation:

If a pre-x1 target-vs-wrong-basin patch cannot raise target `420` above the
wrong `874/862` family, then the wrong instance pointer is probably already
fixed before descriptor repair or distributed through visual/prefix attention.
If it can raise target `420` without destroying the descriptor, the mechanism
has a separable coordinate-slot basin lever and should be traced backward to its
writer.
