# Image 2157 Contextual Tail-Binding Findings

Date: 2026-06-26

Scope: narrow same-image, same-prefix tail-binding probes for COCO val image
`2157`, using contextual false-negative guidance rows and an explicit weak
pure-CE same-prefix control. This is mechanistic evidence, not a population
metric.

## Classification Correction

The earlier image-2157 simplex note treated the sorted-denoise `knife` row as a
false-negative guidance candidate. That was too loose. The selected
sorted-denoise `knife` span is a false-positive/state-entry span on an image
with FN pressure:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/selected_position_subset/v1_broad_image_coverage_unpatched/selected_position_rows.jsonl
```

For that row:

```text
model_id: sorted_denoise
generated_object_idx: 13
generated_desc: knife
generated_bbox_bins: [374,663,520,805]
pred_eval_status: false_positive
matched_gt_idx: null
tags include: false_negative_image, duplication_burst_image,
  cross_model_divergent_image, pre_x1_state_entry, false_positive_span
```

The more precise role is: a knife-like state-entry/tail-coupling case, not a
clean missing-object case.

The contextual guidance rows used here are real GT coordinate targets, but not
clean "object absent from all predictions" exemplars. They are best understood
as controlled language-prefix probes over three GT target boxes:

```text
gt4   wine glass  [72,40,178,377]
gt5   wine glass  [79,141,174,362]
gt11  knife       [338,729,639,991]
```

## Tooling

I added a reusable adapter:

```text
/data/CoordExp/.worktrees/autoregressive-binding-template-study/src/analysis/prefix_denoising_surgery_probing/contextual_fn_tail_adapter.py
/data/CoordExp/.worktrees/autoregressive-binding-template-study/tests/analysis/test_prefix_denoising_contextual_fn_tail_adapter.py
```

It converts contextual FN guidance plan rows into the anchor-escape-shaped rows
consumed by `slot_scaffold_readout` and `anchor_escape_tail_binding`.

Adapter output for denoise rows:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/contextual_fn_tail_adapter/v1_2157_gtx1_context_anchor
```

Additional weak pure-CE same-prefix control rows:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/contextual_fn_tail_adapter/v2_2157_gtx1_context_anchor_purece_sameprefix
```

The pure-CE control is deliberately scoped: it reuses the same contextual
assistant prefix and swaps only model/checkpoint/run metadata. It is a
same-prefix control, not a native pure-CE rollout-state sample.

Verification:

```text
pytest -q tests/analysis/test_prefix_denoising_contextual_fn_tail_adapter.py \
  tests/analysis/test_prefix_denoising_slot_scaffold_readout.py \
  tests/analysis/test_prefix_denoising_anchor_escape_tail_binding.py
```

Result: `11 passed`.

## Probe Artifacts

Slot-scaffold hidden readout:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/slot_scaffold_readout/v1_2157_contextual_fn_gtx1_tail_sorted_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/slot_scaffold_readout/v1_2157_contextual_fn_gtx1_tail_random_gpu1
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/slot_scaffold_readout/v1_2157_contextual_fn_gtx1_tail_purece_sameprefix_gpu4
```

All three completed with `error_count=0`, `plan_row_count=12`, and
`row_count=120`.

Forced-tail continuations:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_tail_binding/v1_2157_contextual_fn_gtx1_tail_sorted_gpu2
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_tail_binding/v1_2157_contextual_fn_gtx1_tail_random_gpu3
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_tail_binding/v1_2157_contextual_fn_gtx1_tail_purece_sameprefix_gpu5
```

All three completed with `error_count=0`, `plan_row_count=15`,
`row_count=15`, and all rows parsed as `complete_valid`.

## Hidden Readout

The readout is not a simple "target is internally solved" story. Best-layer
target ranks are often good for downstream x2/y2 once preceding coordinates are
forced, but y1 is fragile and often background-dominated.

Best target ranks per slot:

```text
model   gt   mode                         slot  target  best layer  rank  winner      top peak
sorted  4    force_target_x1_pre_y1       y1        40  27          359   background  121
sorted  4    force_target_x1_y1_pre_x2    x2       178  27            1   target      178
sorted  4    force_target_x1_y1_x2_pre_y2 y2       377  26           37   background  343
sorted  5    force_target_x1_pre_y1       y1       141  25           15   target      134
sorted  5    force_target_x1_y1_pre_x2    x2       174  27            1   target      174
sorted  5    force_target_x1_y1_x2_pre_y2 y2       362  23           18   background  414
sorted  11   force_target_x1_pre_y1       y1       729  27            3   target      728
sorted  11   force_target_x1_y1_pre_x2    x2       639  27            3   target      641
sorted  11   force_target_x1_y1_x2_pre_y2 y2       991  27            6   target      989

random  4    force_target_x1_pre_y1       y1        40  27          142   background  119
random  4    force_target_x1_y1_pre_x2    x2       178  27            1   target      178
random  5    force_target_x1_pre_y1       y1       141  27           37   background  119
random  5    force_target_x1_y1_pre_x2    x2       174  25            3   background  205
random  11   force_target_x1_pre_y1       y1       729  27           40   background  710
random  11   force_target_x1_y1_pre_x2    x2       639   0          197   invalid     731

pureCE  4    force_target_x1_pre_y1       y1        40  24          456   background  123
pureCE  4    force_target_x1_y1_pre_x2    x2       178  26            1   target      178
pureCE  4    force_target_x1_y1_x2_pre_y2 y2       377  25            2   target      380
pureCE  5    force_target_x1_pre_y1       y1       141  24           81   background  195
pureCE  5    force_target_x1_y1_pre_x2    x2       174  26            4   background  158
pureCE  11   force_target_x1_pre_y1       y1       729  26            2   target      728
pureCE  11   force_target_x1_y1_pre_x2    x2       639  26            3   target      645
pureCE  11   force_target_x1_y1_x2_pre_y2 y2       991  26            5   target      999
```

Counts over all layers/rows:

```text
model   target rows  background rows  invalid/low coord rows  boundary rows
sorted  14           58               48                      0
random   5           62               45                      8
pureCE  14           63               43                      0
```

Interpretation: prefix-denoise sorted and pure CE both expose about the same
number of target-classified hidden readout rows in this same-prefix control,
while random-denoise exposes fewer. But all three still show many background or
low-mass rows. The hidden readout is a weak map of a binding corridor, not a
finished object box.

## Continuation Behavior

The forced-tail continuation evidence is stronger and more diagnostic than the
slot readout. Mean target IoU by continuation mode:

```text
model   pre_x1_free  force_x1  force_x1_y1  force_x1_y1_x2  full_box
sorted  0.517        0.811     0.911        0.916           1.000
random  0.084        0.686     0.740        0.937           1.000
pureCE  0.667        0.702     0.905        0.898           1.000
```

Per-target details:

```text
sorted gt4  wine glass: free 0.705 -> x1 0.705 -> x1+y1 0.928
sorted gt5  wine glass: free 0.000 -> x1 0.749 -> x1+y1 0.828
sorted gt11 knife:      free 0.846 -> x1 0.978 -> x1+y1 0.978

random gt4  wine glass: free 0.041 -> x1 0.729 -> x1+y1 0.949
random gt5  wine glass: free 0.057 -> x1 0.780 -> x1+y1 0.848
random gt11 knife:      free 0.155 -> x1 0.549 -> x1+y1 0.422 -> x1+y1+x2 0.985

pureCE gt4  wine glass: free 0.515 -> x1 0.653 -> x1+y1 0.982
pureCE gt5  wine glass: free 0.547 -> x1 0.467 -> x1+y1 0.744 -> x1+y1+x2 0.807
pureCE gt11 knife:      free 0.939 -> x1 0.985 -> x1+y1 0.989
```

The random-denoise failure is especially revealing: free continuation has
near-zero target IoU for the two wine-glass targets and low target IoU for the
knife, but supplying the first coordinate repairs both wine-glass tails above
IoU 0.72; supplying x1+y1+x2 repairs the knife to 0.985. So the model can
complete the object once a sufficient prefix enters the right corridor.

The pure-CE same-prefix control is not worse. It often free-binds better than
random-denoise and sometimes better than sorted-denoise. This argues against
"prefix denoising teaches a new tail-completion skill" as the main mechanism.

## Mechanistic Interpretation

The best current interpretation:

1. Contextual guidance can expose the target object coordinates, but the main
   failure is not always visual absence. It is whether the autoregressive state
   enters and stays inside the correct object-tail corridor.
2. `x1` alone is often enough to rescue wine-glass continuations, but not always
   enough for the knife. The knife sometimes needs `x1+y1+x2` before the tail
   snaps to target, consistent with a delayed or distributed binding signal.
3. Prefix denoising does not look like a universal tail-binding upgrade. In the
   same-prefix control, pure CE can complete the supplied corridor at least as
   well as the prefix-denoise checkpoints.
4. Therefore the likely prefix-denoising effect is upstream: it changes which
   object-start and coordinate-basin corridors the model enters during free
   rollout, rather than simply increasing the model's ability to perceive or
   complete a supplied object.

This fits the image-16228 value-origin result: prefix denoising damped or
recentered coordinate-basin entry relative to pure CE. Here, once the corridor
is supplied, pure CE remains competent.

## Next

The next high-value bridge is image `19432`:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/representative_sample_base_panel/v3_current_deep_panel_terminal_dense/representative_sample_base_panel.jsonl
```

That row is `random_denoise`, primary role `recoverable_hidden_tail_binding`,
with existing tail evidence:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_tail_binding_reduce/v1_strict4_all_modes_iou_quality/anchor_escape_tail_binding_quality_rows.jsonl
```

It shows the cleanest staged repair currently available:

```text
gt8 chair: free 0.000 -> x1 0.632 -> x1+y1 0.996
gt5 chair: free 0.000 -> x1 0.584 -> x1+y1 0.900 -> x1+y1+x2 0.973
```

This is the better substrate for late-overwrite, coordinate-ridge, and
adapter-counterfactual work. Image 2157 should stay as a contextual guidance and
state-entry/tail-corridor example, not as the main clean tail-binding bridge.
