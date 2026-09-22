---
doc_id: progress.diagnostics.post_repair_duplicate_coordinate_basin_findings
layer: progress
doc_type: findings
status: active
domain: mechanistic-diagnosis
summary: Fine source-region attention and trajectory-prefix value-region continuation probes after local duplicate-loop repair, showing late current-partial coordinate basin dynamics rather than a single head that restores the correct coordinate.
tags:
  - autoregressive-binding-template-ablation
  - coordinate-basin
  - duplication
  - value-region-continuation
  - attention
updated: 2026-06-20
---

# Post-Repair Duplicate Coordinate Basin Findings

## Scope

This note records the next GPU-backed probe after the value-region continuation
result in which head `26:10` current-row/current-prefix value subtraction
repaired a malformed local `<|box_end|>` decision but then exposed a duplicate
same-object/same-coordinate loop.

The focus here is the repaired duplicate trajectory for:

```text
case_id: desc_first-885-8-0-desc_end
seed/current box: <|coord_0|><|coord_0|><|coord_47|><|coord_22|>
target box:       <|coord_1|><|coord_1|><|coord_94|><|coord_22|>
```

This is a single high-priority case, not a validation slice. Treat it as a
mechanistic bridge and hypothesis generator.

## Implementation Added

Two concise probe upgrades were added before running this pass:

- Fine-grained source regions for completed and partial object spans, including
  object-ref boundaries, descriptor, box start, coordinate tokens, box span, and
  row span.
- `trajectory_prefix_source` support for value-region continuation, so a
  continuation probe can operate on each realized trajectory prefix instead of
  silently replaying the original seed boundary state.
- A post-hoc top-tie analyzer for trajectory value-region continuation rows,
  so coordinate-basin claims preserve exact top-k probability ties instead of
  over-interpreting argmax ordering or top-1-only rows.

Relevant commits:

```text
f4e507d9 add fine trajectory source regions
4427285c support trajectory prefixes in value-region continuation
```

Targeted verification:

```text
python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -q
python -m py_compile \
  src/analysis/autoregressive_binding_template_ablation/realized_behavior_bridge.py \
  scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py
git diff --check
```

The full targeted test file passed with `339 passed`.

## Artifacts

Fine source-region attention:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/post_repair_duplicate_loop_attention/desc_first_885_current_prefix_all_full_vector_scale1_fine_regions_v1
```

Key files:

```text
trajectory_source_region_attention_rows.jsonl
trajectory_source_region_attention_summary.json
trajectory_source_region_attention.md
```

Corrected head `26:10` fine-region continuation grid, using realized trajectory
prefixes:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/post_repair_duplicate_loop_value_region_continuation/desc_first_885_coord_fine_regions_v2_trajectory_prefix
```

Candidate-head continuation grid over the highest current-partial attention
heads:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/post_repair_duplicate_loop_value_region_continuation/desc_first_885_coord_candidate_heads_v1
```

Post-hoc top-tie reanalysis of those continuation rows:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/post_repair_duplicate_loop_top_ties/desc_first_885_fine_current_prefix_all_x2_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/post_repair_duplicate_loop_top_ties/desc_first_885_candidate_current_prefix_all_x2_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/post_repair_duplicate_loop_top_ties/desc_first_885_candidate_current_partial_coords_x2_v1
```

Key files in each root:

```text
trajectory_value_region_continuation_top_tie_rows.jsonl
trajectory_value_region_continuation_top_tie_summary.json
trajectory_value_region_continuation_top_tie.md
```

The earlier root below is retained only as a control, because it did not pass
`--trajectory-prefix-source trajectory` and therefore replayed the seed boundary
state rather than the coordinate trajectory states:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/post_repair_duplicate_loop_value_region_continuation/desc_first_885_coord_fine_regions_v1
```

Do not interpret that `v1` root as coordinate-step causality.

## Attention Readout

The attention pass wrote `14080` rows, of which `11520` were usable `ok` rows.
It covered one case, eleven trajectory states, four layers, all sixteen heads,
and twenty source regions.

Head `26:10`, which repaired the local box-closure problem, is not a direct
coordinate-copy head on the repaired coordinate steps. Its normalized mass on
current partial coordinate tokens remains small, while broad pre-prefix context
dominates:

```text
H26:10 x1 target <|coord_1|>:
  current_partial_row_span 0.030678
  current_partial_box_span 0.019624
  last_completed_coords    0.002937
  pre_prefix_context       0.942812

H26:10 y1 target <|coord_1|>:
  current_partial_coords   0.001182
  current_partial_row_span 0.012080
  last_completed_coords    0.006143
  pre_prefix_context       0.946264

H26:10 x2 target <|coord_94|>:
  current_partial_coords   0.003877
  current_partial_row_span 0.043891
  last_completed_coords    0.010489
  last_completed_row_span  0.145190
  pre_prefix_context       0.810918

H26:10 y2 target <|coord_22|>:
  current_partial_coords   0.004544
  current_partial_row_span 0.019802
  last_completed_coords    0.012617
  pre_prefix_context       0.940177
```

By contrast, late layer `27` has strong current-partial row and coordinate
attention at coordinate slots. Mean normalized attention across all heads:

```text
Layer 27 x1:
  current_partial_box_span 0.208360
  current_partial_row_span 0.228229
  current_prefix_all       0.251453
  pre_prefix_context       0.748547

Layer 27 y1:
  current_partial_coords   0.096096
  current_partial_row_span 0.127550
  current_prefix_all       0.142088
  pre_prefix_context       0.857912

Layer 27 x2:
  current_partial_coords   0.274461
  current_partial_row_span 0.291265
  current_prefix_all       0.312264
  pre_prefix_context       0.687736

Layer 27 y2:
  current_partial_coords   0.195009
  current_partial_row_span 0.205239
  current_prefix_all       0.215452
  pre_prefix_context       0.784548
```

Highest mean attention to `current_partial_coords` over coordinate steps:

```text
L27H15 mean 0.885920
L20H14 mean 0.560304
L27H14 mean 0.559455
L20H03 mean 0.511408
L27H02 mean 0.504428
L27H03 mean 0.429386
L26H09 mean 0.371456
L24H09 mean 0.289156
L24H02 mean 0.206573
```

Interpretation: head `26:10` is better understood as a boundary/control repair
lever for this case. Coordinate recurrence is more visible in late
current-partial row heads, especially layer `27`.

## Fine-Region Causal Grid

The corrected head `26:10` fine-region continuation grid used:

```text
trajectory_prefix_source: trajectory
candidate_heads: 26:10
target_next_kinds: coord
value_region_patch_mode: full_vector_subtract
value_region_scales: 0,1,2
trajectory_steps: 4
```

Regions:

```text
current_partial_coords
current_partial_box_span
current_partial_row_span
last_completed_coords
last_completed_box_span
last_completed_row_span
current_prefix_all
pre_prefix_context
```

Patched first-step exactness by coordinate slot:

```text
x1 target <|coord_1|>:  0/7 at every scale; top stays <|coord_0|>
y1 target <|coord_1|>:  0/8 at every scale; top stays <|coord_0|>
x2 target <|coord_94|>: 0/8 at every scale; top stays <|coord_33|>
y2 target <|coord_22|>: 8/8 at every scale; already correct
```

Target-probability movement was small. The largest positive shifts were
insufficient to flip the top token:

```text
y1: pre_prefix_context scale 2 max delta +0.015432
x2: pre_prefix_context scale 2 max delta +0.000612
y2: remains top <|coord_22|>, with small positive/negative shifts
```

Conclusion for this head: the same `26:10` value component that repairs local
box closure does not contain a hidden correct-coordinate signal for the repaired
coordinate trajectory. It can move coordinate probabilities slightly, but it
does not leave the wrong basin.

## Candidate-Head Causal Grid

The candidate-head pass used the top current-partial attention heads:

```text
candidate_heads:
  27:15,27:14,20:14,20:3,27:2,24:9,24:2,27:3
value_region_scales:
  0,1,2,4
```

Patched first-step exactness by slot and scale:

```text
x1: 0/56 at scales 0,1,2,4
y1: 0/64 at scales 0,1,2,4
x2: 0/64 at scales 0,1,2,4
y2: 64/64 at scales 0 and 1; 62/64 at scale 2; 55/64 at scale 4
```

No tested single head, source region, or scale recovers the missing x2 target
`<|coord_94|>` as the top token.

The important positive result is not target repair, but basin steering. Several
late current-partial heads move x2 between wrong coordinate basins:

```text
L27H02 current_partial_coords:
  scale 1: <|coord_33|> -> <|coord_47|>, target <|coord_94|> prob 0.001518
  scale 2: <|coord_33|> -> <|coord_47|>, target <|coord_94|> prob 0.002003
  scale 4: <|coord_33|> -> <|coord_47|>, target <|coord_94|> prob 0.002636

L27H02 current_partial_row_span:
  scale 4: <|coord_33|> -> <|coord_47|>, target <|coord_94|> prob 0.002881

L20H14 current_partial_coords:
  scale 1: <|coord_33|> -> <|coord_47|>, target <|coord_94|> prob 0.001435
  scale 2: <|coord_33|> -> <|coord_47|>, target <|coord_94|> prob 0.001747
  scale 4: <|coord_33|> -> <|coord_53|>, target <|coord_94|> prob 0.002456

L20H03 current_partial_coords:
  scale 2: <|coord_33|> -> <|coord_0|>, target <|coord_94|> prob 0.003035
  scale 4: <|coord_33|> -> <|coord_0|>, target <|coord_94|> prob 0.003920

L27H15 current_partial/current-prefix regions:
  scale 1 or 2 raises target probability but keeps top <|coord_33|>
  scale 4 moves <|coord_33|> -> <|coord_77|>, target prob about 0.0086-0.0090
```

For y2, whose target is already correct, strong perturbations mostly move to
nearby coordinates:

```text
L20H14 current_partial_coords scale 4: <|coord_22|> -> <|coord_21|>
L20H14 current_prefix_all scale 4:   <|coord_22|> -> <|coord_28|>
L27H14 current_partial_coords scale 4: <|coord_22|> -> <|coord_24|>
L27H14 current_prefix_all scale 2:     <|coord_22|> -> <|coord_21|>
```

This is consistent with coordinate locality being present in the trained
surface, while smoothness and basin geometry are distorted enough that
single-head subtraction does not move the state to the intended distant target.

## Top-Tie Reanalysis

The apparent recorded-continuation versus fresh-readout discrepancy at x2 was
rechecked with the stored top-k probabilities. The natural followup rows do not
show a clean `<|coord_47|>` versus `<|coord_33|>` disagreement. They show a
flat ridge:

```text
top tokens: <|coord_47|>, <|coord_33|>, <|coord_34|>, <|coord_32|>, <|coord_31|>
top probs:  0.037223767489, 0.037223767489, 0.025583496317, ...
```

The top-1 patched-first-step rows that surfaced `<|coord_33|>` stored
insufficient top-k information, so they cannot prove a unique coord33 basin.
The post-hoc summaries below were regenerated with strict exact-tie matching
(`top_tie_tolerance: 0`).

Fine head `26:10`, `current_prefix_all`, x2-only:

```text
row_count: 9
top_tie_tolerance: 0
top_tie_status_counts:
  insufficient_topk 3
  observed_tie      6
top_tie_token_set_counts:
  <|coord_33|>                  3
  <|coord_47|> / <|coord_33|>   6
observed_topk_row_count: 6
observed_tie_row_count: 6
observed_tie_rate: 1.0
trajectory_greedy_token_counts:
  <|coord_33|> 3
  <|coord_47|> 6
```

Candidate current-prefix heads, x2-only:

```text
row_count: 96
top_tie_tolerance: 0
top_tie_status_counts:
  insufficient_topk 32
  observed_tie      63
  observed_unique   1
observed_topk_row_count: 64
observed_tie_row_count: 63
observed_tie_rate: 0.984375
dominant top_tie_token_set: <|coord_47|> / <|coord_33|> in 63 rows
```

Candidate current-partial-coordinate heads, x2-only:

```text
row_count: 64
top_tie_tolerance: 0
top_tie_status_counts:
  insufficient_topk 32
  observed_tie      32
observed_topk_row_count: 32
observed_tie_row_count: 32
observed_tie_rate: 1.0
dominant top_tie_token_set: <|coord_47|> / <|coord_33|> in 32 rows
```

This reanalysis changes the interpretation from a unique coord33 fresh basin
to a flat duplicate/neighbor ridge whose visible token depends on argmax
tie-order and which rows preserved top-k information. Candidate-head
perturbations that move to non-tied tokens such as `<|coord_0|>`,
`<|coord_53|>`, `<|coord_60|>`, `<|coord_77|>`, or `<|coord_391|>` remain
interesting basin-stress events, but `<|coord_33|>` versus `<|coord_47|>` alone
should not be treated as a strong basin switch.

## Working Mechanistic Picture

The current best read is a three-part picture:

1. **Local schema repair and duplicate exposure are separable.** Head `26:10`
   carries off-axis current-row value content that can repair malformed local
   box closure, but it is not the coordinate-copy mechanism behind the exposed
   duplicate loop.
2. **Coordinate emission behaves like basin dynamics with flat tie ridges.**
   The x2 state after local repair is not balanced between current
   `<|coord_47|>` and target `<|coord_94|>`. It sits on an exact
   `<|coord_47|>` / `<|coord_33|>` top-probability ridge, and single-head
   current-partial perturbations steer it to neighboring or duplicate-related
   wrong basins such as `<|coord_53|>`, `<|coord_60|>`, `<|coord_77|>`,
   `<|coord_391|>`, or `<|coord_0|>`.
3. **The duplicate coordinate `<|coord_47|>` is a real member of the ridge, not
   just a text accident.** It appears in the natural duplicate row with equal
   stored top probability to `<|coord_33|>`. The important claim is therefore
   not a unique coord47 attractor, but a local duplicate/neighbor ridge that
   excludes the distant target `<|coord_94|>` under these interventions.

The strongest phrase for the result is therefore:

```text
The repaired duplicate loop is not explained by one boundary head copying the
previous coordinate. It looks like a late current-partial coordinate-basin
system whose local perturbations move among wrong tied or nearby attractors,
while the correct target coordinate remains outside the single-head
intervention basin.
```

## Combined-Head Causal Grid

After the single-head candidate sweeps, I added a `combined_all` candidate-head
group mode for the trajectory value-region continuation probe. This applies all
requested head value patches in one patched-first forward pass instead of
enumerating candidate heads independently.

Implementation surface:

```text
--stage trajectory-boundary-head-value-region-continuation
--candidate-head-group-mode combined_all
```

Smoke artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/post_repair_duplicate_loop_value_region_continuation/desc_first_885_coord_combined_heads_v1/smoke_maxrow1_regions2_scales0_1_steps2
```

Main 4-state artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/post_repair_duplicate_loop_value_region_continuation/desc_first_885_coord_combined_heads_v1/full4_regions2_scales0_1_2_4_steps4_patched_topk_v1
```

Post-hoc x2 top-tie artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/post_repair_duplicate_loop_top_ties/desc_first_885_combined_heads_x2_v1/full4_regions2_scales0_1_2_4_steps4_patched_topk_v1
```

Scope:

```text
selected_state_row_count: 4
source_bucket_counts: failure_default128_rescued 4
candidate_head_group_mode: combined_all
candidate_heads: 27:15, 27:14, 20:14, 20:3, 27:2, 24:9, 24:2, 27:3
value_source_regions: current_partial_coords, current_prefix_all
value_region_scales: 0, 1, 2, 4
trajectory_steps: 4
row_count: 104
```

Patched-first dose response, averaged across the four selected coordinate
states:

```text
current_partial_coords:
  scale 0: mean seed target-logit delta  0.03125  (1 skipped, 3 realized)
  scale 1: mean seed target-logit delta -1.78125  (1 skipped, 3 realized)
  scale 2: mean seed target-logit delta -5.59375  (1 skipped, 3 realized)
  scale 4: mean seed target-logit delta -14.953613 (1 skipped, 3 realized)

current_prefix_all:
  scale 0: mean seed target-logit delta   0.0
  scale 1: mean seed target-logit delta  -2.28125
  scale 2: mean seed target-logit delta  -7.484375
  scale 4: mean seed target-logit delta -20.792969
```

x2-only top-tie reanalysis over the combined-head artifact:

```text
row_count: 18
continuation_step_kind_counts:
  patched_first_step 8
  natural_followup   10
top_tie_status_counts:
  observed_tie    12
  observed_unique 6
dominant observed tie:
  <|coord_47|> / <|coord_33|> in 10 rows
```

The important new distinction is that combined-head subtraction can strongly
dent the x2 target state at the patched-first step, but the natural follow-up
often re-enters the same wrong local ridge unless the patch is pushed into a
destructive scale. For the x2 slot:

- Scale `0` starts from a real exact `<|coord_47|>` / `<|coord_33|>` tie, not a
  top-1 logging artifact.
- Scale `1` moved the target `<|coord_94|>` only modestly:
  `rank 92 -> 86/90` for current-partial/current-prefix patched-first rows,
  while making `<|coord_33|>` the unique immediate top token.
- Scale `2` moved the patched-first greedy token to `<|coord_999|>` /
  `<|coord_0|>` for current-partial and `<|coord_0|>` for current-prefix, with
  target ranks `28/19`. Several natural follow-ups still returned to the
  `<|coord_47|>` / `<|coord_33|>` ridge.
- Scale `4` broke the coordinate basin hard: patched-first greedy tokens
  became off-manifold strings such as `:checked` or `ecs`, with target ranks
  around `54590/59783`. This is a stress test, not a clean correction.

Interpretation update:

```text
The combined late-head value vector is causally connected to the duplicate
coordinate basin, but it does not contain a simple "correct target coordinate"
switch. Moderate subtraction moves the immediate x2 state from the exact
coord47/coord33 ridge into unique coord33 or coord0-like basins while the
autoregressive follow-up can reconstruct the wrong local ridge; destructive
subtraction leaves the coord-token manifold. This supports a distributed
basin/onset mechanism more than a single reusable object-binding head.
```

## Caveats

- Single case only.
- Multi-head value subtraction is now run only for this one 4-state
  `desc_first-885` bridge panel; no residual-stack coordinate-slot patch has
  been run yet.
- Scale `4` can be destructive, especially for `L27H15`; interpret it as
  basin stress, not a clean causal edit. In the combined-head grid it can leave
  the coordinate-token manifold.
- Natural followup rows preserve enough top-k information to show an exact
  `<|coord_47|>` / `<|coord_33|>` tie. The patched-top-k rerun now gives
  top-k for patched-first steps too, but only for this regenerated combined
  panel.
- No training was run.

## Next Probes

Highest-value next steps:

1. Keep top-k tie capture in all coordinate-basin probes, and only interpret
   basin switches when the tie set changes or a non-tied token becomes top.
2. Repeat the combined-head coordinate-basin patch on additional
   high-divergence duplicate cases, and add a non-destructive top-k capture for
   patched-first coordinate rows before making cross-case claims.
3. Build a coordinate-logit and embedding-basin profile for
   `<|coord_0|>`, `<|coord_1|>`, `<|coord_21|>`, `<|coord_22|>`,
   `<|coord_24|>`, `<|coord_28|>`, `<|coord_33|>`, `<|coord_47|>`,
   `<|coord_53|>`, `<|coord_60|>`, `<|coord_77|>`, and `<|coord_94|>`.
4. Patch residual states at coordinate slots, not only head value slices, to
   test whether the correct `<|coord_94|>` target is present elsewhere in the
   stack or absent from this trajectory state.
5. Repeat the same panel on additional high-divergence duplicate cases before
   promoting the coordinate-basin picture beyond this bridge case.
