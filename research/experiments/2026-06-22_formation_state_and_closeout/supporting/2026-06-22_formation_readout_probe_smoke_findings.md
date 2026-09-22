---
doc_id: progress.diagnostics.formation_readout_probe_smoke_findings_2026_06_22
layer: progress
doc_type: diagnostic-implementation-note
status: active-branch-evidence
evidence_scope: gpu-smoke-teacher-forced-formation-prefix-readout
domain: autoregressive-binding-template-ablation
updated: 2026-06-22
branch: codex/autoregressive-binding-template-study
---

# Formation Readout Probe Smoke Findings

## Purpose

This note records the first model-backed probe over the newly materialized
formation-position rows.

The probe scores the next-token distribution at each `trajectory_prefix_text`
boundary. It is designed to separate:

```text
descriptor/object identity readiness
schema or structural-token readiness
coordinate-token type readiness
coordinate-basin selection quality
late-layer logit-lens emergence
```

It deliberately starts as a smoke. The results below are not broad validation,
but they confirm that the new formation rows can be consumed by checkpoint-928
with the token_embeddings_adapter surface.

## Implementation

Added:

```text
src/analysis/autoregressive_binding_template_ablation/formation_readout_probe.py
scripts/analysis/run_autoregressive_binding_formation_readout_probe.py
tests/analysis/test_formation_readout_probe.py
```

The runner:

```text
loads formation rows from JSONL
filters by formation_position, shard, and limit
resolves checkpoint-928 through the existing deep-probe token_embeddings_adapter loader
renders partial teacher-forced assistant prefixes
locates trajectory_prefix_text inside the chat-tokenized input
scores the final prefix-token logit row against target_next_token_text
records coord-vocab mass, wrapper-token mass, target rank/prob, coord-bin metrics, and hidden-state logit-lens rows
uses the last occurrence of `trajectory_prefix_text` in the chat-tokenized input, so short first-object prefixes do not accidentally bind to earlier prompt instructions
merges sharded runs into a single readout/hidden-row artifact
```

Image root override used for bbox_len12000 row images:

```text
/data/CoordExp/public_data/coco/rescale_32_1024_bbox
```

## Verification

TDD red:

```text
python -m pytest tests/analysis/test_formation_readout_probe.py -q
```

failed before implementation with:

```text
ModuleNotFoundError: No module named 'src.analysis.autoregressive_binding_template_ablation.formation_readout_probe'
```

Green:

```text
python -m pytest tests/analysis/test_formation_readout_probe.py -q
```

result:

```text
5 passed
```

Compile check:

```text
python -m py_compile src/analysis/autoregressive_binding_template_ablation/formation_readout_probe.py scripts/analysis/run_autoregressive_binding_formation_readout_probe.py tests/analysis/test_formation_readout_probe.py
```

result:

```text
passed
```

## Dry Run

Input:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/panel_formation_row_builder/v2_full_train_val_candidate_bank_teacher_forced_gt/panel_formation_position_rows.jsonl
```

Output:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_readout_probe/v1_full_train_val_teacher_forced_dry_run
```

Selected positions:

```text
descriptor_onset
box_start
pre_x1
next_object_onset
```

Counters:

```text
selected_row_count=384
row_counts_by_split={train: 192, val: 192}
row_counts_by_candidate_regime={
  crowded: 64,
  duplicate_basin_nearby: 64,
  repeated_class: 64,
  simple_control: 64,
  small_object: 64,
  termination_tail: 64
}
```

## GPU Smoke 1: Train Broad Panel

Command scope:

```text
CUDA_VISIBLE_DEVICES=7
input: v2_full_train_val_candidate_bank_teacher_forced_gt
positions: descriptor_onset, box_start, pre_x1, next_object_onset
limit: 4
layers: 0, 12, 24, -1
```

Output:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_readout_probe/v2_full_train_val_teacher_forced_smoke_limit4
```

Counters:

```text
selected_row_count=4
output_row_count=4
hidden_row_count=16
error_count=0
row_counts_by_split={train: 4}
row_counts_by_candidate_regime={repeated_class: 4}
```

Readout:

```text
descriptor_onset bird:
  rank=1 prob=0.9790 top1=bird coord_vocab_mass=5.9e-09

box_start <|box_start|>:
  rank=1 prob=0.9999 top1=<|box_start|> coord_vocab_mass=2.1e-16

pre_x1 <|coord_635|>:
  rank=15 prob=0.0105 coord_vocab_mass=0.9975 coord_rank=16 top1_bin=644
  shape=multimodal_irregular

next_object_onset <|object_ref_start|>:
  rank=1 prob=0.9627 top1=<|object_ref_start|> coord_vocab_mass=2.1e-13
```

Hidden readout snapshot:

```text
descriptor_onset becomes top1 by layer 24 and remains top1 at layer 27.
box_start is rank 8 at layer 24 and top1 by layer 27.
pre_x1 coord target is rank 4 in the layer-24 coord logit lens but rank 16 at final output, with top1_bin=644.
next_object_onset becomes top1 by layer 24 and remains top1 at layer 27.
```

## GPU Smoke 2: Val Rollout-Labeled Panel

Command scope:

```text
CUDA_VISIBLE_DEVICES=7
input: v1_first200_ckpt928_val200_probe_panel_teacher_forced_gt
positions: descriptor_onset, box_start, pre_x1, next_object_onset
limit: 4
layers: 0, 12, 24, -1
```

Output:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_readout_probe/v3_val_panel_teacher_forced_smoke_limit4
```

Counters:

```text
selected_row_count=4
output_row_count=4
hidden_row_count=16
error_count=0
row_counts_by_split={val: 4}
row_counts_by_candidate_regime={repeated_class: 4}
row_counts_by_panel_role={same_image_matched_control: 4}
row_counts_by_rollout_match_status={matched_iou50_sem_ok: 4}
```

Readout:

```text
descriptor_onset person:
  rank=1 prob=0.7663 top1=person coord_vocab_mass=5.9e-09

box_start <|box_start|>:
  rank=1 prob=0.9998 top1=<|box_start|> coord_vocab_mass=3.6e-16

pre_x1 <|coord_218|>:
  rank=73 prob=0.0034 coord_vocab_mass=0.9961 coord_rank=74 top1_bin=394
  shape=multimodal_irregular

next_object_onset <|object_ref_start|>:
  rank=1 prob=0.99999 top1=<|object_ref_start|> coord_vocab_mass=5.3e-18
```

Hidden readout snapshot:

```text
descriptor_onset reaches rank 2 by layer 24 and top1 by layer 27.
box_start is rank 18 at layer 24 and top1 by layer 27.
pre_x1 remains weak at layer 24 and final output: coord logit-lens rank 472 at layer 24, final coord rank 74, top1_bin=394.
next_object_onset becomes top1 by layer 24 and remains top1 at layer 27.
```

## Immediate Read

Both smokes show stable schema/type behavior:

```text
descriptor token: top1 in final output
box_start token: top1 in final output
next_object_onset token: top1 in final output
pre_x1 coordinate token type: coord-vocab mass about 0.996 to 0.998
```

The fragile part is not whether the model knows to emit a coordinate token. The
fragile part is which coordinate basin wins.

Smoke-level interpretation:

```text
train repeated-class example:
  coordinate type is correct, target bin is near the winning basin
  target=635, top1=644

val matched repeated-class example:
  coordinate type is correct, but target bin loses to a distant basin
  target=218, top1=394
```

This supports the current working direction:

```text
descriptor and structural transitions are late-layer stabilized.
coordinate slot states are type-stable but basin-fragile.
false negatives or duplicate-like behavior should be probed as basin selection
and object-pointer/context competition, not as generic schema collapse.
```

## Next

Run the same readout over a sharded broader sample:

```text
full broad selected scope: 384 rows
recommended shards: 8
positions: descriptor_onset, box_start, pre_x1, next_object_onset
group by: split, candidate_regime, rollout_match_status, same_desc_count,
object_idx, objects_remaining_after, target_desc
```

Then compare:

```text
train vs val pre_x1 coord_rank_gt and top1_distance
matched vs unmatched val rows under teacher-forced GT prefix
descriptor/box_start readiness versus pre_x1 basin quality
layer-24 versus final-output rank drift at pre_x1
```

## Alignment Fix And Broad Sharded Run

After the first broad merge, descriptor-onset rows for first-object
simple-control cases appeared to put large mass on `coord_108`. This was an
instrumentation artifact. For first-object descriptor onset, the prefix can be
only:

```text
<|object_ref_start|>
```

The runner originally located the first matching token subsequence in the full
chat input. That could bind to an earlier occurrence in prompt instructions
rather than the assistant prefix. The runner now uses the last matching
subsequence. Regression:

```text
tests/analysis/test_formation_readout_probe.py::test_find_last_subsequence_prefers_assistant_suffix_occurrence
```

The old broad merged artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_readout_probe/v4_full_train_val_teacher_forced_sharded8/merged
```

should be treated as superseded for descriptor-onset interpretation. The
current broad artifact is:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_readout_probe/v6_full_train_val_teacher_forced_sharded8_alignfix/merged
```

Counters:

```text
selected_row_count=384
output_row_count=384
hidden_row_count=1536
error_count=0
row_counts_by_split={train: 192, val: 192}
row_counts_by_candidate_regime={
  crowded: 64,
  duplicate_basin_nearby: 64,
  repeated_class: 64,
  simple_control: 64,
  small_object: 64,
  termination_tail: 64
}
```

Corrected position summary:

```text
descriptor_onset:
  n=96 mean_rank=1.42 top1_rate=0.771 mean_prob=0.678 mean_coord_mass=5.2e-08

box_start:
  n=96 mean_rank=1.10 top1_rate=0.917 mean_prob=0.908 mean_coord_mass=5.1e-06

pre_x1:
  n=96 mean_rank=125.0 top1_rate=0.177 mean_prob=0.080 mean_coord_mass=0.998
  mean_coord_rank=129.1

next_object_onset:
  n=96 mean_rank=1.23 top1_rate=0.771 mean_prob=0.778 mean_coord_mass=1.5e-09
```

The broad result strengthens the core mechanism picture:

```text
The model usually knows the next token type.
It usually knows descriptor and structural transitions.
The hard failure is coordinate-basin selection inside the coordinate slot.
```

Pre-x1 split summary:

```text
train:
  n=48 mean_coord_rank=104.6 median_rank=70 exact_top1=0.188
  mean_top1_distance=138.6 mean_coord_mass=0.9982

val:
  n=48 mean_coord_rank=153.5 median_rank=46 exact_top1=0.167
  mean_top1_distance=116.0 mean_coord_mass=0.9980
```

Train and val are not separated by coordinate-token type readiness; both are
type-stable. Val has worse mean rank, but both splits fail badly in hard
regimes. This points away from a simple "unseen val visual perception" story
and toward context/regime-dependent coordinate-basin competition.

Pre-x1 regime summary:

```text
simple_control:
  n=16 mean_rank=1.8 median_rank=1 exact_top1=0.812 mean_top1_distance=0.8

duplicate_basin_nearby:
  n=16 mean_rank=118.8 median_rank=85 exact_top1=0.062 mean_top1_distance=202.1

crowded:
  n=16 mean_rank=122.7 median_rank=76 exact_top1=0.000 mean_top1_distance=157.7

termination_tail:
  n=16 mean_rank=134.7 median_rank=22 exact_top1=0.188 mean_top1_distance=110.3

small_object:
  n=16 mean_rank=171.4 median_rank=134 exact_top1=0.000 mean_top1_distance=129.4

repeated_class:
  n=16 mean_rank=225.0 median_rank=144 exact_top1=0.000 mean_top1_distance=163.4
```

Layer-drift summary at pre-x1:

```text
layer 24 logit lens:
  n=96 mean_coord_rank=174.3 median_rank=64 exact_top1=0.021

final layer / output:
  n=96 mean_coord_rank=128.7 median_rank=47 exact_top1=0.177
```

The coordinate basin often improves after layer 24 but remains weak. That makes
the next promising causal target a late coordinate-basin selection pathway, not
an early descriptor identity pathway.

Descriptor-onset regime summary after alignment fix:

```text
repeated_class:
  top1_rate=1.000 mean_rank=1.00

simple_control:
  top1_rate=0.875 mean_rank=1.13

duplicate_basin_nearby:
  top1_rate=0.812 mean_rank=1.19

termination_tail:
  top1_rate=0.750 mean_rank=1.31

crowded:
  top1_rate=0.625 mean_rank=1.81

small_object:
  top1_rate=0.562 mean_rank=2.06
```

Descriptor selection has some regime sensitivity, but it is far less severe
than pre-x1 coordinate-basin selection and has negligible coordinate-vocab mass.

## Val Matched-Unmatched Panel

The rollout-labeled val panel was also run with the align-fixed probe:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_readout_probe/v7_val_panel_matched_unmatched_teacher_forced_sharded4_alignfix/merged
```

Counters:

```text
selected_row_count=124
output_row_count=124
hidden_row_count=496
error_count=0
row_counts_by_rollout_match_status={
  matched_iou50_sem_ok: 64,
  relaxed_match_iou30_only_sem_ok: 8,
  unmatched_iou30: 52
}
```

Pre-x1 by rollout status:

```text
matched_iou50_sem_ok:
  n=16 mean_coord_rank=45.4 median_rank=19 exact_top1=0.188
  mean_top1_distance=65.3 mean_coord_mass=0.9987

relaxed_match_iou30_only_sem_ok:
  n=2 mean_coord_rank=129.5 median_rank=253 exact_top1=0.000
  mean_top1_distance=190.5 mean_coord_mass=0.9996

unmatched_iou30:
  n=13 mean_coord_rank=155.7 median_rank=94 exact_top1=0.000
  mean_top1_distance=151.8 mean_coord_mass=0.9978
```

This is the first direct answer to the false-negative guidance question under
the current probe:

```text
The unmatched rows are not failing the coordinate-token type gate.
Teacher-forced language-side guidance is not sufficient to make x1 exact.
The missing-object cases still show much worse coordinate-basin selection than
matched controls under the same GT-prefix protocol.
```

That does not prove visual non-perception. It suggests the more specific
failure mode: visual/object evidence and autoregressive context do not
synchronize strongly enough to select the correct coordinate basin.

Panel-role detail:

```text
small_object_matched_control:
  n=1 mean_rank=1 exact_top1=1.000

duplicate_nearby_matched_control:
  n=3 mean_rank=16 exact_top1=0.333

duplicate_nearby_fn:
  n=2 mean_rank=25 exact_top1=0.000

same_image_matched_control:
  n=4 mean_rank=28.5 exact_top1=0.000

small_object_fn:
  n=3 mean_rank=86 exact_top1=0.000

same_image_unmatched_fn:
  n=4 mean_rank=196.3 exact_top1=0.000

crowded_fn:
  n=3 mean_rank=279 exact_top1=0.000
```

Layer-drift by rollout status at pre-x1:

```text
matched_iou50_sem_ok:
  layer24 mean_coord_rank=116.8 -> final mean_coord_rank=45.8

relaxed_match_iou30_only_sem_ok:
  layer24 mean_coord_rank=374.5 -> final mean_coord_rank=129.5

unmatched_iou30:
  layer24 mean_coord_rank=241.6 -> final mean_coord_rank=154.8
```

The final layers improve coordinate selection for every group, but do not
rescue false negatives to matched-control quality. The next causal probe should
therefore target the late pre-x1 coordinate-basin routing state and ask whether
patching matched-control basin states into unmatched rows rescues x1 without
also damaging descriptor/box structure.

## Broad All-Position Train-Val Readout

The broad train/val candidate bank was rerun over all eight formation positions:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_readout_probe/v8_full_train_val_all_positions_teacher_forced_sharded4_alignfix/merged
```

Scope:

```text
selected_row_count=768
output_row_count=768
hidden_row_count=3072
error_count=0
splits={train: 384, val: 384}
positions={
  descriptor_onset: 96,
  descriptor_end: 96,
  object_ref_end: 96,
  box_start: 96,
  pre_x1: 96,
  post_x1: 96,
  box_close: 96,
  next_object_onset: 96
}
```

Slot-level summary:

```text
descriptor_onset:
  top1=0.771 mean_rank=1.42 mean_prob=0.678 coord_mass=0.000000052

descriptor_end:
  top1=0.844 mean_rank=1.24 mean_prob=0.764 coord_mass=0.000000047

object_ref_end:
  top1=1.000 mean_rank=1.00 mean_prob=1.000 coord_mass=0.0000000008

box_start:
  top1=0.917 mean_rank=1.10 mean_prob=0.908 coord_mass=0.0000051

pre_x1:
  top1=0.177 mean_rank=125.0 mean_prob=0.080 coord_mass=0.998
  mean_coord_rank=129.1 median_coord_rank=46.5 exact_coord=0.177

post_x1:
  top1=0.229 mean_rank=7.62 mean_prob=0.100 coord_mass=0.9999
  mean_coord_rank=8.05 median_coord_rank=5 exact_coord=0.229

box_close:
  top1=0.854 mean_rank=1.24 mean_prob=0.841 coord_mass=0.000000064

next_object_onset:
  top1=0.771 mean_rank=1.23 mean_prob=0.778 coord_mass=0.0000000015
```

The important new separation is `pre_x1` versus `post_x1`. Both are coordinate
slots with nearly all probability mass inside the coord vocabulary, but x1 is a
fragile basin-entry decision while y1 after x1 is much more localized. This
argues against a generic "the model cannot see the object box" explanation.
The failure looks like coordinate-basin onset/cursor selection: once a first
coordinate anchor is committed, the next coordinate is substantially easier.

Train versus val did not form a simple memorization split:

```text
pre_x1 train:
  n=48 mean_coord_rank=104.6 median=58.5 exact_coord=0.188

pre_x1 val:
  n=48 mean_coord_rank=153.5 median=46 exact_coord=0.167

post_x1 train:
  n=48 mean_coord_rank=9.15 median=3 exact_coord=0.333

post_x1 val:
  n=48 mean_coord_rank=6.96 median=5 exact_coord=0.125
```

Some trained rows still have severe pre-x1 failures:

```text
train sports ball small_object:
  image=389205 object=9 x1_gt=453 rank=828 top1=302

train car duplicate_basin_nearby:
  image=99844 object=6 x1_gt=934 rank=486 top1=6

train knife termination_tail:
  image=307238 object=71 x1_gt=114 rank=459 top1=630

train bird repeated_class:
  image=320275 object=34 x1_gt=400 rank=344 top1=676
```

The trained-sequence evidence weakens a pure unseen-generalization story. The
shared mechanism is more likely local ambiguity plus autoregressive cursor
state: repeated classes, small objects, crowded rows, tail positions, and nearby
same-desc basins all create x1 attractor competition even under teacher-forced
GT prefix.

Coordinate logit-lens development also separates x1 onset from y1 continuation:

```text
pre_x1:
  layer0 mean_coord_rank=483.6
  layer12 mean_coord_rank=472.4
  layer24 mean_coord_rank=174.3
  layer27/final mean_coord_rank=128.7

post_x1:
  layer0 mean_coord_rank=391.9
  layer12 mean_coord_rank=569.2
  layer24 mean_coord_rank=21.3
  layer27/final mean_coord_rank=8.1
```

The late layers rescue both coordinate slots, but they rescue `post_x1` far
more completely than `pre_x1`. The next intervention should therefore patch or
mediate the late pre-x1 basin-entry state, not the whole box span. Good donor
controls should include simple-control exact rows and same-image/same-regime
matched rows; hard receivers should include both train and val rows with high
pre-x1 rank so the probe does not collapse into a train-vs-val memorization
comparison.

## Formation Patch Pair Selection

A small pair-selection surface was added for the next causal phase:

```text
src/analysis/autoregressive_binding_template_ablation/formation_replay_patch.py
scripts/analysis/run_autoregressive_binding_formation_replay_patch.py
tests/analysis/test_formation_replay_patch.py
```

It does not run model perturbations. It only materializes hard pre-x1 receiver
rows, baseline/self controls, and donor rows for a future layer-input patch.

Exact-only donor panel:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v1_v8_prex1_hard_train_val_pair_selection

receiver_count=24
pair_count=120
row_counts_by_receiver_split={train: 60, val: 60}
row_counts_by_control_kind={
  baseline_no_patch: 24,
  self_noop: 24,
  simple_control_donor: 71,
  same_split_regime_desc_donor: 1
}
```

This panel is useful as a strict exact-rank donor control, but it is too
dominated by simple-control donors to localize the hard-regime mechanism.

Recommended first causal panel:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v2_v8_prex1_hard_train_val_pair_selection_donor_rank10

receiver_count=24
pair_count=144
row_counts_by_receiver_split={train: 72, val: 72}
row_counts_by_receiver_regime={
  crowded: 24,
  duplicate_basin_nearby: 18,
  repeated_class: 48,
  small_object: 36,
  termination_tail: 18
}
row_counts_by_control_kind={
  baseline_no_patch: 24,
  self_noop: 24,
  simple_control_donor: 84,
  same_split_regime_desc_donor: 7,
  same_regime_desc_donor: 4,
  same_image_regime_desc_donor: 1
}
```

The rank-10 donor panel keeps the same 24 hard receivers but adds usable
same-regime/same-desc donors for repeated-class, termination-tail,
duplicate-nearby, and small-object rows. This is the better launch surface for
the first causal `layer_input` patch because it can separate generic
"inject an easy first-object state" from a more meaningful "inject a comparable
coordinate-basin state" effect.

## Explicit-Token Layer-Input Patch

The first causal patch runner was added on top of the pair-selection surface.
Important implementation constraint:

```text
The formation pre_x1 scoring token is not the last token in the chat-template
input. The template appends <|im_end|> and a newline after the assistant
prefix, so last-token patch helpers would patch the wrong position.
```

A tokenizer/template check on the first v2 pair showed:

```text
prefix_last_token=<|box_start|>
prediction_token_index=105
last_input_token_index=107
prediction_token_is_last=false
template_tail_token_count=2
```

The model-backed patcher therefore locates `trajectory_prefix_text` inside the
full processor input and patches the explicit prediction token index at decoder
layer input. Evidence scope is immediate next-token causal readout for `x1`;
it is not suffix replay and does not yet claim generated-span repair.

Implementation surface:

```text
src/analysis/autoregressive_binding_template_ablation/formation_replay_patch.py
scripts/analysis/run_autoregressive_binding_formation_replay_patch.py
tests/analysis/test_formation_replay_patch.py
```

Simple-control smoke:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v3_explicit_token_layer_input_smoke_limit6

patch_row_count=6
error_count=0
activation_patch_ran_count=5
coord_rank_improved_count=4
coord_rank_worsened_count=0
mean_coord_rank_delta=-403.2
```

The smoke receiver was the hardest train small-object row:

```text
receiver: train sports ball small_object
x1_gt=453 baseline_rank=828 baseline_top1=302
self_noop: patched_rank=828 patched_top1=302
simple_control donors:
  patched_rank=346 patched_top1=0
  patched_rank=234 patched_top1=0
  patched_rank=130 patched_top1=0
  patched_rank=183 patched_top1=0
```

This confirms the explicit-position hook can causally move the x1 distribution
while the self-noop hook is inert. The simple-control donors still mostly steer
toward generic zero/edge-like first-object basins, so they are controls rather
than the main mechanism evidence.

Same-regime donor panel at final layer input:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v4_explicit_token_layer_input_same_regime_donors

patch_row_count=12
error_count=0
coord_rank_improved_count=8
coord_rank_worsened_count=4
mean_baseline_coord_rank_gt=379.25
mean_patched_coord_rank_gt=295.42
mean_coord_rank_delta=-83.83
```

Same-regime donor panel at layer 24 input:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v6_explicit_token_layer24_same_regime_donors

patch_row_count=12
error_count=0
coord_rank_improved_count=9
coord_rank_worsened_count=3
mean_baseline_coord_rank_gt=379.25
mean_patched_coord_rank_gt=279.17
mean_coord_rank_delta=-100.08
```

Representative row-level movements:

```text
train termination_tail knife:
  x1_gt=114 baseline_rank=459
  layer27 donor A -> patched_rank=4, patched_top1=116
  layer24 donor A -> patched_rank=4, patched_top1=116
  donor B worsens/slightly changes: layer27 rank=462, layer24 rank=448

train repeated_class bird:
  x1_gt=400 baseline_rank=344
  layer27 -> patched_rank=4, patched_top1=414
  layer24 -> patched_rank=2, patched_top1=414

val duplicate_basin_nearby sheep, same-image donor:
  x1_gt=109 baseline_rank=292
  layer27 -> patched_rank=44, patched_top1=92
  layer24 -> patched_rank=52, patched_top1=92

val repeated_class bird:
  x1_gt=131 baseline_rank=320
  layer27 -> patched_rank=836, patched_top1=414
  layer24 -> patched_rank=823, patched_top1=414
```

Interpretation:

```text
The causal handle exists: replacing/adding donor layer-input state at the
explicit pre_x1 token can move the receiver's coordinate-basin logits after
downstream computation.

The effect is basin-steering, not generic rescue. Some comparable donors improve
the target rank strongly; some steer toward a donor/neighbor basin and worsen
the receiver. This matches the emerging picture that false negatives and
duplications arise from fragile instance-to-coordinate basin binding, not from
schema/type failure.

Layer 24 is at least as effective as the final layer input on this small panel,
which matches the logit-lens finding that late layers sharpen coordinate basin
state. The next deeper probe should localize which subpath writes or preserves
the basin state between layer 24 and the final logits.
```

## Layer-24 Component Mediation

The explicit-token patcher was extended to support component-output patch sites
at the same pre-x1 token index:

```text
patch_site=layer_input
patch_site=self_attn
patch_site=mlp
```

The component runs used the same 12 same-regime/same-desc donor rows as the
layer-24 layer-input result.

Artifacts:

```text
layer_input:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v6_explicit_token_layer24_same_regime_donors

self_attn:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v9_explicit_token_layer24_self_attn_same_regime_donors

mlp:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v10_explicit_token_layer24_mlp_same_regime_donors
```

Summary:

```text
layer_input:
  patch_row_count=12
  error_count=0
  improved=9
  worsened=3
  mean_coord_rank_delta=-100.08

self_attn:
  patch_row_count=12
  error_count=0
  improved=8
  worsened=4
  mean_coord_rank_delta=-25.67

mlp:
  patch_row_count=12
  error_count=0
  improved=8
  worsened=4
  mean_coord_rank_delta=-40.42
```

Representative paired deltas:

```text
train repeated_class bird, x1_gt=400, baseline_rank=344:
  layer_input delta=-342 -> patched_rank=2,   top1=414
  self_attn   delta=-185 -> patched_rank=159, top1=249
  mlp         delta=-296 -> patched_rank=48,  top1=294

train termination_tail knife, x1_gt=114, baseline_rank=459:
  layer_input delta=-455 -> patched_rank=4,   top1=116
  self_attn   delta=-22  -> patched_rank=437, top1=630
  mlp         delta=-179 -> patched_rank=280, top1=0

val duplicate_basin_nearby sheep, same-image donor, x1_gt=109, baseline_rank=292:
  layer_input delta=-240 -> patched_rank=52,  top1=92
  self_attn   delta=-6   -> patched_rank=286, top1=293
  mlp         delta=-124 -> patched_rank=168, top1=290
```

Interpretation:

```text
Both attention-output and MLP-output deltas can move x1 basin logits at the
explicit pre-x1 token, but neither explains the whole layer-input effect.
MLP output carries more of the average effect than self-attention output on
this panel, while full layer-input patching remains strongest.

This suggests the pre-x1 basin state is already present in the residual stream
entering layer 24 and is transformed/amplified through late-layer subpaths,
especially MLP. Attention still has row-specific effects and cannot be ignored,
but the immediate evidence does not support "attention alone writes the basin."

The next high-value probe is a mediation/ablation variant that patches the
layer-input state while clamping or subtracting component deltas, or a residual
stream path scan across layers 20-27. Suffix replay remains downstream of this:
we should not spend generation budget until the path-level handle is sharper.
```

Panel note:

```text
Do not keep centering the next probes on the original person/backpack anchor.
It remains useful as a visually interpretable case, but the mechanism picture
needs a broader row bank. The next deterministic expansion should select
trained-sequence failures and unseen-val analogs by motif, then compare whether
the same pre-x1 basin weakness and the same layer/component patch handles appear
in both groups. This directly tests whether training exposure eliminates the
failure mode or whether the local autoregressive/coordinate-basin dynamics can
fail even on trained object sequences.
```

## Diversified Train-Val Patch Panel

The replay-patch selector now supports a per-split/per-regime receiver cap, so
the hard pre-x1 panel can be diversified by motif instead of ranking all rows
globally. The first diversified panel uses:

```text
readout source:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_readout_probe/v8_full_train_val_all_positions_teacher_forced_sharded4_alignfix/merged/formation_readout_rows.jsonl

formation_position=pre_x1
hard_min_coord_rank=100
donor_max_coord_rank=10
max_receivers_per_split=12
max_receivers_per_split_regime=2
patch_layer_index=24
```

Pair-selection artifacts:

```text
layer_input:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v11_v8_prex1_diverse_train_val_pair_selection_layer24

self_attn:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v14_v8_prex1_diverse_train_val_pair_selection_layer24_self_attn

mlp:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v15_v8_prex1_diverse_train_val_pair_selection_layer24_mlp
```

The layer-input pair selection produced:

```text
receiver_count=19
pair_count=114
row_counts_by_receiver_split={train: 54, val: 60}
row_counts_by_receiver_regime={
  crowded: 24,
  duplicate_basin_nearby: 24,
  repeated_class: 24,
  small_object: 24,
  termination_tail: 18
}
```

Representative receivers include trained-sequence failures:

```text
train small_object sports ball: rank=828, gt=453, top1=302
train duplicate_basin_nearby car: rank=486, gt=934, top1=6
train termination_tail knife: rank=459, gt=114, top1=630
train repeated_class bird: rank=344, gt=400, top1=676
train crowded chair: rank=276, gt=526, top1=354
```

and unseen-val analogs:

```text
val repeated_class bird: rank=907, gt=147, top1=793
val termination_tail apple: rank=902, gt=884, top1=0
val repeated_class bird: rank=540, gt=281, top1=374
val small_object bird: rank=444, gt=223, top1=563
val crowded cake: rank=384, gt=142, top1=650
```

The key result is that comparable donors repair/steer train and val rows in the
same direction at the immediate pre-x1 logit readout:

```text
layer_input comparable donors:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v12_diverse_train_val_layer24_comparable_donors

patch_row_count=9
error_count=0
coord_rank_improved_count=9
coord_rank_worsened_count=0
mean_coord_rank_delta=-231.00
train mean_coord_rank_delta=-250.25
val mean_coord_rank_delta=-215.60
top1_closer=5
top1_farther=4
mean_top1_distance_delta=-60.78
```

This weakens a pure train/val generalization story. The trained sequences can
have severe local pre-x1 basin failures, and the same kind of comparable donor
state can move them. That points toward local autoregressive coordinate-basin
onset/cursor mechanics rather than visual non-perception or unseen-only
generalization failure.

## Simple-Control Contrast

The same diversified receiver panel was run with only simple-control donors:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v13_diverse_train_val_layer24_simple_control_donors

patch_row_count=67
error_count=0
coord_rank_improved_count=35
coord_rank_worsened_count=32
mean_coord_rank_delta=18.13
train mean_coord_rank_delta=142.62
val mean_coord_rank_delta=-95.69
top1_closer=20
top1_farther=35
mean_top1_distance_delta=44.91
```

The simple-control result is not a generic repair story. In this run, the
patched top-1 coordinate collapsed to the simple-control donor origin basin
(`coord_0`) for every receiver group inspected. Some rows still improve target
rank because the distribution is dragged through coordinate space, but top-1
often moves farther from the target. This means rank-only summaries can
overstate repair; top-1 distance and donor-basin identity must be carried in
all future reducers.

## Diversified Component Mediation

The same 9 comparable donor rows were run through component-output patch sites:

```text
self_attn:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v16_diverse_train_val_layer24_self_attn_comparable_donors

mlp:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v17_diverse_train_val_layer24_mlp_comparable_donors
```

Summary:

```text
layer_input:
  patch_row_count=9
  improved=9
  worsened=0
  mean_coord_rank_delta=-231.00
  top1_closer=5
  top1_farther=4

self_attn:
  patch_row_count=9
  improved=6
  worsened=3
  mean_coord_rank_delta=-32.33
  top1_closer=4
  top1_farther=1

mlp:
  patch_row_count=9
  improved=6
  worsened=3
  mean_coord_rank_delta=-41.56
  top1_closer=7
  top1_farther=2
```

Interpretation:

```text
The diversified panel reproduces the earlier component result: full layer-input
state is much stronger than isolated attention-output or MLP-output patching.
MLP remains slightly stronger than attention on mean rank movement and is better
on top-1 distance in this panel, but neither isolated component explains the
full residual-stream effect.

The current best mechanism picture is that a pre-x1 coordinate-basin state is
already present in the residual stream entering late layers. Component subpaths
can modulate or amplify it, especially MLP, but a true mediation/clamp probe is
needed to separate writer, preserver, and readout-amplifier roles.
```

## Patch-Result Basin Taxonomy

A post-hoc reducer was added for replay-patch rows. It labels each row using
target rank, target-distance movement, donor-basin distance, and origin-basin
collapse:

```text
exact_target_repair
near_target_repair
target_rank_repair
donor_basin_steer
origin_basin_collapse
rank_only_improvement_worse_top1
worsened_escape
rank_worsened
no_change_or_unclear
```

Artifacts:

```text
layer_input comparable:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_basin_taxonomy/v1_v12_layer_input_comparable_donors

layer_input simple-control:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_basin_taxonomy/v2_v13_layer_input_simple_control_donors

self_attn comparable:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_basin_taxonomy/v3_v16_self_attn_comparable_donors

mlp comparable:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_basin_taxonomy/v4_v17_mlp_comparable_donors
```

Summary:

```text
layer_input comparable:
  basin_row_count=9
  labels={
    near_target_repair: 3,
    target_rank_repair: 2,
    donor_basin_steer: 3,
    rank_only_improvement_worse_top1: 1
  }
  rank_improved_top1_farther_count=4
  patched_top1_near_target_count=3
  patched_top1_near_donor_count=5
  mean_coord_rank_delta=-231.00
  mean_top1_target_distance_delta=-60.78

layer_input simple-control:
  basin_row_count=67
  labels={origin_basin_collapse: 67}
  rank_improved_top1_farther_count=14
  patched_top1_near_target_count=0
  patched_top1_near_donor_count=67
  patched_top1_in_origin_basin_count=67
  mean_coord_rank_delta=18.13
  mean_top1_target_distance_delta=44.91

self_attn comparable:
  basin_row_count=9
  labels={
    target_rank_repair: 3,
    no_change_or_unclear: 3,
    donor_basin_steer: 1,
    rank_worsened: 1,
    worsened_escape: 1
  }
  patched_top1_near_target_count=0
  mean_coord_rank_delta=-32.33

mlp comparable:
  basin_row_count=9
  labels={
    target_rank_repair: 3,
    donor_basin_steer: 1,
    origin_basin_collapse: 1,
    rank_only_improvement_worse_top1: 1,
    rank_worsened: 2,
    worsened_escape: 1
  }
  patched_top1_near_target_count=0
  mean_coord_rank_delta=-41.56
```

Interpretation:

```text
Rank improvement is not equivalent to object repair. On the comparable
layer-input run, only 3/9 rows are near-target repair, while 3/9 are donor-basin
steering and 1/9 is rank-only improvement with worse top-1 target distance. The
simple-control run is the decisive negative control: every row is an
origin-basin collapse, even when target rank improves.

This makes the current mechanism picture sharper. The residual stream patch can
inject a coordinate-basin cursor state, but whether that cursor belongs to the
receiver object depends on donor compatibility and local context. The model is
not merely missing visual evidence; it can be pushed into target-like,
neighbor-like, donor-like, or origin-like coordinate basins with the same patch
interface.

The next mediation/clamp probe should be evaluated by taxonomy label, not only
rank delta. A clamp that preserves near-target repair while suppressing
donor/origin steering would be much more mechanism-relevant than one that merely
improves average rank.
```

## Layer-Input Clamp Mediation

The replay-patch surface now supports component clamp sites for `layer_input`
patches. The clamp replaces the selected component output at the explicit
pre-x1 token with the receiver baseline component value after injecting the
donor-vs-receiver layer-input delta. This asks whether the patched residual
state still moves the coordinate basin when late self-attention and/or MLP
component outputs are held to the receiver baseline.

Pair selections used the same v8 diversified hard train/val pre-x1 receiver
panel and the same layer-24 comparable donor controls:

```text
clamp self_attn pairs:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v18_v8_prex1_diverse_train_val_layer24_clamp_self_attn_pairs

clamp mlp pairs:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v19_v8_prex1_diverse_train_val_layer24_clamp_mlp_pairs

clamp self_attn+mlp pairs:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v20_v8_prex1_diverse_train_val_layer24_clamp_self_attn_mlp_pairs

receiver_count=19
pair_count=114 for each clamp selection
row_counts_by_receiver_split={train: 54, val: 60}
row_counts_by_receiver_regime={
  crowded: 24,
  duplicate_basin_nearby: 24,
  repeated_class: 24,
  small_object: 24,
  termination_tail: 18
}
```

Patch artifacts:

```text
layer_input + clamp self_attn:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v21_diverse_train_val_layer24_layer_input_clamp_self_attn_comparable_donors

layer_input + clamp mlp:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v22_diverse_train_val_layer24_layer_input_clamp_mlp_comparable_donors

layer_input + clamp self_attn+mlp:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v23_diverse_train_val_layer24_layer_input_clamp_self_attn_mlp_comparable_donors
```

Immediate x1 summary:

```text
unclamped layer_input comparable, v12:
  patch_row_count=9
  improved=9
  worsened=0
  mean_coord_rank_delta=-231.00
  mean_top1_target_distance_delta=-60.78

layer_input + clamp self_attn, v21:
  patch_row_count=9
  error_count=0
  component_clamp_ran_count=9
  improved=7
  worsened=2
  mean_coord_rank_delta=-215.11

layer_input + clamp mlp, v22:
  patch_row_count=9
  error_count=0
  component_clamp_ran_count=9
  improved=7
  worsened=2
  mean_coord_rank_delta=-138.22

layer_input + clamp self_attn+mlp, v23:
  patch_row_count=9
  error_count=0
  component_clamp_ran_count=9
  improved=6
  worsened=3
  mean_coord_rank_delta=-114.44
```

Basin-taxonomy artifacts:

```text
v21 taxonomy:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_basin_taxonomy/v5_v21_layer_input_clamp_self_attn_comparable_donors

v22 taxonomy:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_basin_taxonomy/v6_v22_layer_input_clamp_mlp_comparable_donors

v23 taxonomy:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_basin_taxonomy/v7_v23_layer_input_clamp_self_attn_mlp_comparable_donors
```

Taxonomy summary:

```text
unclamped layer_input comparable, v12:
  labels={
    near_target_repair: 3,
    target_rank_repair: 2,
    donor_basin_steer: 3,
    rank_only_improvement_worse_top1: 1
  }
  patched_top1_near_target_count=3
  patched_top1_near_donor_count=5

layer_input + clamp self_attn, v21:
  labels={
    near_target_repair: 3,
    target_rank_repair: 1,
    donor_basin_steer: 4,
    rank_only_improvement_worse_top1: 1
  }
  patched_top1_near_target_count=3
  patched_top1_near_donor_count=6
  mean_top1_target_distance_delta=-19.44

layer_input + clamp mlp, v22:
  labels={
    near_target_repair: 2,
    target_rank_repair: 2,
    donor_basin_steer: 5
  }
  patched_top1_near_target_count=2
  patched_top1_near_donor_count=6
  mean_top1_target_distance_delta=40.11

layer_input + clamp self_attn+mlp, v23:
  labels={
    near_target_repair: 1,
    donor_basin_steer: 5,
    origin_basin_collapse: 1,
    rank_worsened: 1,
    worsened_escape: 1
  }
  patched_top1_near_target_count=1
  patched_top1_near_donor_count=6
  patched_top1_in_origin_basin_count=1
  mean_top1_target_distance_delta=74.00
```

Row-level checks:

```text
train repeated_class bird, x1_gt=400:
  unclamped near_target_repair, delta=-342, top1=414
  clamp_self_attn near_target_repair, delta=-330, top1=414
  clamp_mlp near_target_repair, delta=-287, top1=394
  clamp_both donor_basin_steer, delta=-186, top1=785

train repeated_class bird, x1_gt=398:
  unclamped near_target_repair, delta=-193, top1=414
  clamp_self_attn near_target_repair, delta=-156, top1=414
  clamp_mlp target_rank_repair, delta=-77, top1=426
  clamp_both rank_worsened, delta=44, top1=426

train termination_tail knife, x1_gt=114:
  unclamped near_target_repair, delta=-455, top1=116
  clamp_self_attn near_target_repair, delta=-453, top1=111
  clamp_mlp near_target_repair, delta=-429, top1=124
  clamp_both near_target_repair, delta=-429, top1=124
```

Interpretation:

```text
Clamping self-attention barely reduces the layer-input effect and preserves all
3 near-target repairs. On this panel, late self-attention output is therefore
not the dominant necessary mediator for the immediate pre-x1 coordinate-basin
movement.

Clamping MLP reduces the mean rank effect much more, drops near-target repairs
from 3 to 2, and makes top-1 target distance worse on average. MLP looks more
necessary for turning the incoming residual cursor into target-aligned repair,
though it is not a single sufficient writer of the whole effect.

Clamping both weakens the effect further and introduces origin/worsening labels,
but donor-basin steering still persists in 5/9 rows. The important mechanism
clue is that a donor-like or target-like coordinate-basin cursor can already be
present in the layer-24 input residual state before the selected component
outputs are allowed to write. The components shape and amplify that cursor; the
cursor itself is not born solely inside the layer-24 attention or MLP output.
```

Next expansion:

```text
The next panel should not be constrained by the original person/backpack case.
The trained-sequence failures are especially valuable: if a row appears in the
training split and still has a weak or wrong pre-x1 basin under teacher-forced
GT prefix, then the failure is not explained by unseen visual content alone.

The next deterministic pass should over-sample trained rows with severe pre-x1
rank or wrong top-1 basins, pair them with motif-matched val analogs, and ask
whether layer-input patching, component clamps, and later continuation behavior
separate into the same taxonomy labels. This should treat readout evidence,
motif, object position, same-desc repetition, crowding, and spatial scale as
selection handles even when train rollout labels are not available.
```

## Expanded Train-Val Layer-Input Panel

Following the train-row expansion rule above, two larger layer-24 pair panels
were materialized from the same v8 train/val teacher-forced readout rows:

```text
readout source:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_readout_probe/v8_full_train_val_all_positions_teacher_forced_sharded4_alignfix/merged/formation_readout_rows.jsonl

strict donor panel, donor_max_coord_rank=10:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v24_v8_prex1_expanded_train_val_pair_selection_layer24_donor10

loose donor panel, donor_max_coord_rank=25:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v25_v8_prex1_expanded_train_val_pair_selection_layer24_donor25

formation_position=pre_x1
hard_min_coord_rank=100
max_receivers_per_split=30
max_receivers_per_split_regime=6
max_donors_per_receiver=6
patch_layer_index=24
patch_site=layer_input
```

Both selections contain 33 hard receivers and 264 total pair/control rows. The
strict donor panel has 23 comparable-donor intervention rows after filtering to
`same_image_regime_desc_donor`, `same_split_regime_desc_donor`, and
`same_regime_desc_donor`; the loose donor panel has 34 comparable rows.

Layer-input patch artifacts:

```text
strict donor run, donor_max_coord_rank=10:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v27_expanded_train_val_layer24_layer_input_comparable_donors_donor10

strict donor taxonomy:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_basin_taxonomy/v9_v27_expanded_train_val_layer24_layer_input_comparable_donors_donor10

loose donor run, donor_max_coord_rank=25:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v26_expanded_train_val_layer24_layer_input_comparable_donors_donor25

loose donor taxonomy:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_basin_taxonomy/v8_v26_expanded_train_val_layer24_layer_input_comparable_donors_donor25
```

Strict donor result:

```text
patch_row_count=23
error_count=0
coord_rank_improved_count=13
coord_rank_worsened_count=10
mean_coord_rank_delta=41.17
row_counts_by_receiver_split={train: 11, val: 12}
row_counts_by_receiver_regime={
  crowded: 6,
  duplicate_basin_nearby: 3,
  repeated_class: 8,
  small_object: 3,
  termination_tail: 3
}

taxonomy labels={
  donor_basin_steer: 10,
  near_target_repair: 4,
  rank_only_improvement_worse_top1: 2,
  rank_worsened: 1,
  target_rank_repair: 2,
  worsened_escape: 4
}
patched_top1_near_target_count=4
patched_top1_near_donor_count=13
rank_improved_top1_farther_count=6
mean_top1_target_distance_delta=41.61
```

Strict donor split comparison:

```text
train:
  n=11
  mean_coord_rank_delta=13.91
  mean_top1_target_distance_delta=13.00
  labels={
    near_target_repair: 3,
    donor_basin_steer: 5,
    worsened_escape: 2,
    rank_only_improvement_worse_top1: 1
  }

val:
  n=12
  mean_coord_rank_delta=66.17
  mean_top1_target_distance_delta=67.83
  labels={
    target_rank_repair: 2,
    rank_only_improvement_worse_top1: 1,
    donor_basin_steer: 5,
    worsened_escape: 2,
    rank_worsened: 1,
    near_target_repair: 1
  }
```

Loose donor result:

```text
patch_row_count=34
error_count=0
coord_rank_improved_count=17
coord_rank_worsened_count=17
mean_coord_rank_delta=51.38
row_counts_by_receiver_split={train: 16, val: 18}
row_counts_by_receiver_regime={
  crowded: 9,
  duplicate_basin_nearby: 3,
  repeated_class: 16,
  small_object: 3,
  termination_tail: 3
}

taxonomy labels={
  donor_basin_steer: 21,
  near_target_repair: 4,
  rank_only_improvement_worse_top1: 2,
  rank_worsened: 1,
  target_rank_repair: 2,
  worsened_escape: 4
}
patched_top1_near_target_count=4
patched_top1_near_donor_count=24
rank_improved_top1_farther_count=8
mean_top1_target_distance_delta=104.03
```

Loose donor split comparison:

```text
train:
  n=16
  mean_coord_rank_delta=50.94
  labels={
    near_target_repair: 3,
    donor_basin_steer: 10,
    worsened_escape: 2,
    rank_only_improvement_worse_top1: 1
  }

val:
  n=18
  mean_coord_rank_delta=51.78
  labels={
    donor_basin_steer: 11,
    target_rank_repair: 2,
    rank_only_improvement_worse_top1: 1,
    worsened_escape: 2,
    rank_worsened: 1,
    near_target_repair: 1
  }
```

Repair-like strict rows:

```text
train termination_tail knife:
  baseline_rank=459, delta=-455, gt=114, baseline_top1=630,
  patched_top1=116, label=near_target_repair,
  donor=train termination_tail knife, donor_rank=9

train repeated_class bird:
  baseline_rank=344, delta=-342, gt=400, baseline_top1=676,
  patched_top1=414, label=near_target_repair,
  donor=train repeated_class bird, donor_rank=8

train repeated_class bird:
  baseline_rank=227, delta=-193, gt=398, baseline_top1=499,
  patched_top1=414, label=near_target_repair,
  donor=train repeated_class bird, donor_rank=8

val repeated_class bird:
  baseline_rank=907, delta=-81, gt=147, baseline_top1=793,
  patched_top1=414, label=target_rank_repair,
  donor=train repeated_class bird, donor_rank=8

val duplicate_basin_nearby sheep:
  baseline_rank=292, delta=-240, gt=109, baseline_top1=298,
  patched_top1=92, label=target_rank_repair,
  donor=val duplicate_basin_nearby sheep, donor_rank=7

val crowded person:
  baseline_rank=188, delta=-36, gt=593, baseline_top1=557,
  patched_top1=604, label=near_target_repair,
  donor=train crowded person, donor_rank=6
```

Interpretation:

```text
The broader panel does not support a simple train-versus-val split. Train rows
and val rows both show repair, donor-basin steering, and worsening under the
same layer-input intervention. The strict donor panel is slightly less harmful
than the loose donor panel, but donor-basin steering remains the largest single
label even when donor rank is <=10.

This points to donor/receiver compatibility and local basin ownership as deeper
than split membership. The layer-input residual handle is powerful enough to
move trained-sequence failures, which means those failures are not just "model
never saw the visual case." But the state being injected often carries the
donor coordinate basin rather than the receiver object's basin, especially in
crowded, repeated-class, and small-object settings.

The next experiment should not merely add more rows. It should explicitly model
compatibility: same image versus same split, same desc, motif match, donor
coordinate rank, donor-target distance, receiver baseline basin, and whether
the patch lands near target, donor, origin, or escapes. Near-target strict rows
are good continuation candidates; donor-steer rows are better for causal
localization of basin ownership and attention/value routing.
```

## Compatibility Reducer and Expanded Clamp Rerun

A post-hoc compatibility reducer was added on top of basin-taxonomy rows. It
does not load the model. It annotates each patch row with:

```text
donor_receiver_relation
donor_coord_rank_bucket
target_donor_distance_bucket
receiver_baseline_landing_relation
patched_landing_relation
compatibility_key
```

and summarizes basin labels and mean rank/top-1-distance deltas by relation.
This is now the preferred reducer before deciding whether a row should go to
continuation, path localization, or a new donor selection panel.

Compatibility artifacts for the expanded layer-input runs:

```text
strict donor compatibility, v27/v9:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_compatibility/v1_v9_v27_expanded_train_val_layer24_layer_input_comparable_donors_donor10

loose donor compatibility, v26/v8:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_compatibility/v2_v8_v26_expanded_train_val_layer24_layer_input_comparable_donors_donor25
```

Strict donor compatibility summary:

```text
compatibility_row_count=23
row_counts_by_donor_receiver_relation={
  cross_split_regime_desc: 11,
  same_image_regime_desc: 1,
  same_split_regime_desc: 11
}
row_counts_by_donor_coord_rank_bucket={
  top1: 1,
  top5: 5,
  top10: 17
}
row_counts_by_target_donor_distance_bucket={
  exact: 1,
  near: 3,
  local: 1,
  mid: 1,
  far: 17
}
row_counts_by_patched_landing_relation={
  donor_basin: 11,
  target_basin: 2,
  target_and_donor_basin: 2,
  other: 8
}
```

Strict relation breakdown:

```text
cross_split_regime_desc:
  labels={
    donor_basin_steer: 5,
    near_target_repair: 1,
    target_rank_repair: 1,
    rank_only_improvement_worse_top1: 1,
    rank_worsened: 1,
    worsened_escape: 2
  }
  mean_coord_rank_delta=74.00

same_image_regime_desc:
  labels={target_rank_repair: 1}
  mean_coord_rank_delta=-240.00

same_split_regime_desc:
  labels={
    donor_basin_steer: 5,
    near_target_repair: 3,
    rank_only_improvement_worse_top1: 1,
    worsened_escape: 2
  }
  mean_coord_rank_delta=33.91
```

Distance bucket result:

```text
near target-donor distance:
  labels={near_target_repair: 2, target_rank_repair: 1}

far target-donor distance:
  labels={
    donor_basin_steer: 8,
    near_target_repair: 2,
    target_rank_repair: 1,
    rank_only_improvement_worse_top1: 1,
    rank_worsened: 1,
    worsened_escape: 4
  }
```

The distance-bucket result is mechanistically useful: when donor and receiver
target bins are near, the patch is much more likely to be a target-like repair;
when they are far, donor-basin steering dominates. This is not merely donor
quality, because all strict donors are rank <=10.

The strict expanded panel was then rerun with component clamps:

```text
clamp self_attn pairs:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v28_v8_prex1_expanded_train_val_layer24_clamp_self_attn_pairs_donor10

clamp mlp pairs:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v29_v8_prex1_expanded_train_val_layer24_clamp_mlp_pairs_donor10

clamp self_attn+mlp pairs:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v30_v8_prex1_expanded_train_val_layer24_clamp_self_attn_mlp_pairs_donor10

layer_input + clamp self_attn:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v31_expanded_train_val_layer24_layer_input_clamp_self_attn_comparable_donors_donor10

layer_input + clamp mlp:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v32_expanded_train_val_layer24_layer_input_clamp_mlp_comparable_donors_donor10

layer_input + clamp self_attn+mlp:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v33_expanded_train_val_layer24_layer_input_clamp_self_attn_mlp_comparable_donors_donor10
```

Expanded clamp taxonomy and compatibility artifacts:

```text
taxonomy v10 for v31:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_basin_taxonomy/v10_v31_expanded_train_val_layer24_layer_input_clamp_self_attn_donor10

taxonomy v11 for v32:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_basin_taxonomy/v11_v32_expanded_train_val_layer24_layer_input_clamp_mlp_donor10

taxonomy v12 for v33:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_basin_taxonomy/v12_v33_expanded_train_val_layer24_layer_input_clamp_self_attn_mlp_donor10

compatibility v3 for v31:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_compatibility/v3_v10_v31_expanded_train_val_layer24_layer_input_clamp_self_attn_donor10

compatibility v4 for v32:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_compatibility/v4_v11_v32_expanded_train_val_layer24_layer_input_clamp_mlp_donor10

compatibility v5 for v33:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_compatibility/v5_v12_v33_expanded_train_val_layer24_layer_input_clamp_self_attn_mlp_donor10
```

Expanded clamp summary over the same 23 strict rows:

```text
unclamped:
  improved=13
  worsened=10
  mean_coord_rank_delta=41.17
  labels={
    donor_basin_steer: 10,
    near_target_repair: 4,
    target_rank_repair: 2,
    rank_only_improvement_worse_top1: 2,
    rank_worsened: 1,
    worsened_escape: 4
  }
  patched_landing={donor_basin: 11, target_basin: 2,
                   target_and_donor_basin: 2, other: 8}

clamp self_attn:
  improved=11
  worsened=12
  mean_coord_rank_delta=38.70
  labels={
    donor_basin_steer: 11,
    near_target_repair: 3,
    target_rank_repair: 2,
    rank_only_improvement_worse_top1: 2,
    rank_worsened: 1,
    worsened_escape: 4
  }
  patched_landing={donor_basin: 12, target_basin: 2,
                   target_and_donor_basin: 1, other: 8}

clamp mlp:
  improved=11
  worsened=12
  mean_coord_rank_delta=58.65
  labels={
    donor_basin_steer: 12,
    near_target_repair: 3,
    target_rank_repair: 4,
    rank_worsened: 1,
    worsened_escape: 3
  }
  patched_landing={donor_basin: 12, target_basin: 2,
                   target_and_donor_basin: 1, other: 8}

clamp both:
  improved=12
  worsened=11
  mean_coord_rank_delta=45.00
  labels={
    donor_basin_steer: 9,
    near_target_repair: 2,
    target_rank_repair: 4,
    origin_basin_collapse: 1,
    rank_worsened: 3,
    worsened_escape: 4
  }
  patched_landing={donor_basin: 9, target_basin: 1,
                   target_and_donor_basin: 1, origin_basin: 1, other: 11}
```

Relation-level clamp result:

```text
mean_coord_rank_delta_by_relation

unclamped:
  cross_split_regime_desc=74.00
  same_image_regime_desc=-240.00
  same_split_regime_desc=33.91

clamp self_attn:
  cross_split_regime_desc=86.55
  same_image_regime_desc=-250.00
  same_split_regime_desc=17.09

clamp mlp:
  cross_split_regime_desc=135.09
  same_image_regime_desc=-29.00
  same_split_regime_desc=-9.82

clamp both:
  cross_split_regime_desc=110.36
  same_image_regime_desc=-41.00
  same_split_regime_desc=-12.55
```

Key row transitions:

```text
train termination_tail knife, target=114, donor=124:
  unclamped near_target_repair delta=-455 top1=116
  clamp_self_attn near_target_repair delta=-453 top1=111
  clamp_mlp near_target_repair delta=-429 top1=124
  clamp_both near_target_repair delta=-429 top1=124

train repeated_class bird, target=400, donor=788:
  unclamped near_target_repair delta=-342 top1=414
  clamp_self_attn near_target_repair delta=-330 top1=414
  clamp_mlp near_target_repair delta=-287 top1=394
  clamp_both donor_basin_steer delta=-186 top1=785

val duplicate_basin_nearby sheep, target=109, donor=97:
  unclamped target_rank_repair delta=-240 top1=92
  clamp_self_attn target_rank_repair delta=-250 top1=92
  clamp_mlp target_rank_repair delta=-29 top1=282
  clamp_both origin_basin_collapse delta=-41 top1=0

val repeated_class bird, target=147, donor=788:
  unclamped target_rank_repair delta=-81 top1=414
  clamp_self_attn donor_basin_steer delta=-107 top1=785
  clamp_mlp donor_basin_steer delta=-84 top1=788
  clamp_both donor_basin_steer delta=-79 top1=788
```

Interpretation:

```text
The expanded clamp panel partly replicates the earlier finding: clamping
self-attention changes less than clamping MLP, and the layer-input residual
state continues to carry substantial basin information even when components are
clamped. However, the compatibility reducer shows a deeper split: component
clamps do not simply reduce a global repair effect; they redistribute which
basin wins.

Same-split compatible rows improve under MLP or both-component clamp on mean
rank delta, while cross-split compatible rows become much worse. This suggests
the MLP path is not merely "the target repair writer." It helps resolve or
stabilize basin ownership when donor/receiver compatibility is weaker, while
some same-split rows can still ride the incoming residual cursor without it.

Near donor-target coordinate distance remains the strongest simple repair
condition. Far donor-target distance is where donor-basin steering dominates,
even for top-ranked donors. The core mechanism now looks like a residual-stream
coordinate cursor plus a compatibility-dependent ownership resolver. MLP is a
major part of that resolver; self-attention is less necessary at this late
pre-x1 layer, though it can still flip individual rows.
```

Next:

```text
Use the compatibility reducer to select two downstream sets:

1. continuation candidates: strict near-target or target-rank repairs where
   the patched landing is target_basin or target_and_donor_basin;
2. localization candidates: far-distance donor_basin_steer rows where the
   patch lands almost exactly on the donor basin despite same-desc/motif
   compatibility.

The next causal question is whether the ownership resolver is written earlier
than layer 24, or whether late MLP converts an already-ambiguous residual cursor
into a target/donor decision. A layer scan over the compatibility-selected rows
is now higher-value than another broad row expansion.
```

## Compatibility-Selected Layer Scan

The compatibility reducer was used to build a layer scan over two candidate
sets from the strict donor panel:

```text
repair_candidate:
  near-target / target-rank repairs whose patched landing is target_basin or
  target_and_donor_basin

donor_steer_candidate:
  far-distance donor_basin_steer rows whose patched landing is donor_basin
```

The layer-scan pair builder retargets the same source pair rows to multiple
layers while preserving the original compatibility-selection metadata:

```text
layer scan pair selection, layers 12,16,20,22,24,26,27:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_layer_scan/v1_strict_donor10_repair_and_donor_steer_layers12_16_20_22_24_26_27_pairs

layer-input patch run:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_layer_scan/v2_strict_donor10_repair_and_donor_steer_layers12_16_20_22_24_26_27_patch

basin taxonomy:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_layer_scan/v3_v2_basin_taxonomy

compatibility summary:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_layer_scan/v4_v3_compatibility
```

Scope:

```text
source_pair_count=10
layer_scan_pair_count=70
layers={12, 16, 20, 22, 24, 26, 27}
roles={repair_candidate: 28, donor_steer_candidate: 42}
error_count=0
```

Coarse layer result:

```text
layer 12:
  labels={
    no_change_or_unclear: 4,
    rank_worsened: 4,
    target_rank_repair: 1,
    rank_only_improvement_worse_top1: 1
  }
  mean_coord_rank_delta=-1.10

layer 16:
  labels={
    target_rank_repair: 5,
    rank_worsened: 3,
    origin_basin_collapse: 1,
    rank_only_improvement_worse_top1: 1
  }
  mean_coord_rank_delta=-18.80

layer 20:
  labels={
    donor_basin_steer: 6,
    near_target_repair: 3,
    rank_only_improvement_worse_top1: 1
  }
  mean_coord_rank_delta=102.80

layer 22:
  labels={donor_basin_steer: 6, near_target_repair: 4}
  mean_coord_rank_delta=99.00

layer 24:
  labels={donor_basin_steer: 6, near_target_repair: 4}
  mean_coord_rank_delta=101.70
```

Role split:

```text
repair_candidate:
  layer 16 mean_delta=-207.25, labels={target_rank_repair: 4}, landing={other: 4}
  layer 20 mean_delta=-250.25, labels={near_target_repair: 3,
                                       rank_only_improvement_worse_top1: 1}
  layer 22 mean_delta=-260.00, labels={near_target_repair: 4}

donor_steer_candidate:
  layer 16 mean_delta=106.83, mostly rank_worsened/origin/unclear
  layer 20 mean_delta=338.17, labels={donor_basin_steer: 6}
  layer 22 mean_delta=338.33, labels={donor_basin_steer: 6}
```

The coarse scan located the ownership transition between layers 16 and 20, so a
fine scan was run over layers 17-21:

```text
fine layer-scan pair selection:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_layer_scan/v5_strict_donor10_repair_and_donor_steer_layers17_18_19_20_21_pairs

fine layer-input patch run:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_layer_scan/v6_strict_donor10_repair_and_donor_steer_layers17_18_19_20_21_patch

fine basin taxonomy:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_layer_scan/v7_v6_basin_taxonomy

fine compatibility summary:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_layer_scan/v8_v7_compatibility
```

Fine layer result:

```text
layer 17:
  labels={
    near_target_repair: 2,
    target_rank_repair: 4,
    origin_basin_collapse: 1,
    rank_worsened: 2,
    worsened_escape: 1
  }

layer 18:
  labels={
    donor_basin_steer: 7,
    near_target_repair: 1,
    target_rank_repair: 1,
    origin_basin_collapse: 1
  }

layer 19:
  labels={
    donor_basin_steer: 5,
    near_target_repair: 3,
    origin_basin_collapse: 1,
    rank_only_improvement_worse_top1: 1
  }

layer 20:
  labels={
    donor_basin_steer: 6,
    near_target_repair: 3,
    rank_only_improvement_worse_top1: 1
  }

layer 21:
  labels={donor_basin_steer: 6, near_target_repair: 4}
```

Fine role split:

```text
repair_candidate:
  L17 mean_delta=-230.25, labels={near_target_repair: 2, target_rank_repair: 2}
  L18 mean_delta=-261.75, labels={donor_basin_steer: 2,
                                  near_target_repair: 1,
                                  target_rank_repair: 1}
  L19 mean_delta=-270.25, labels={near_target_repair: 3,
                                  rank_only_improvement_worse_top1: 1}
  L21 mean_delta=-262.25, labels={near_target_repair: 4}

donor_steer_candidate:
  L17 mean_delta=160.67, no donor-basin landing except origin/other
  L18 mean_delta=338.83, labels={donor_basin_steer: 5,
                                 origin_basin_collapse: 1}
  L19 mean_delta=331.00, labels={donor_basin_steer: 5,
                                 origin_basin_collapse: 1}
  L20 mean_delta=338.17, labels={donor_basin_steer: 6}
```

Row-level transition examples:

```text
donor-steer train crowded person, target=603, donor=191:
  L17 origin_basin_collapse top1=0
  L18 donor_basin_steer top1=190
  L19-L21 donor_basin_steer top1=184

donor-steer val duplicate_basin_nearby bird, target=782, donor=31:
  L17 worsened_escape top1=394
  L18 donor_basin_steer top1=32
  L19-L21 donor_basin_steer top1=32

repair train repeated_class bird, target=400, donor=788:
  L17 target_rank_repair top1=322
  L18 donor_basin_steer top1=781
  L19-L21 near_target_repair top1=414

repair train termination_tail knife, target=114, donor=124:
  L17-L21 near_target_repair, top1 in {124, 116, 109, 111}

repair val crowded person, target=593, donor=603:
  L17 near_target_repair top1=584
  L18 target_rank_repair top1=565
  L19-L20 rank-only improvement with bad top1
  L21 near_target_repair top1=604
```

Interpretation:

```text
The target/donor ownership decision is not first visible at layer 24. It forms
abruptly in the residual stream around layers 18-20. Layer 16 already carries
rank-repair information but usually not a top-1 coordinate basin. By layer 18,
far-distance donor-steer rows often snap directly to donor basin; by layer 19
or 20, repair rows either recover target basin or reveal that the injected
cursor belongs to the donor.

This is a stronger mechanism claim than "late residual patch works." The
coordinate cursor appears to pass through an ambiguous phase where rank can
improve without stable basin ownership. The decisive transition is basin
ownership, not target rank. The layer-18 donor snap and layer-19 target repair
in repeated-class bird rows are especially revealing: the same donor vector can
first expose donor geometry and then be resolved by downstream computation into
the receiver target basin when compatibility/context supports it.
```

## Component Output Layer Scan

To localize the transition path, the same compatibility-selected layers 17-21
were run with first-order component-output patches at `self_attn` and `mlp`.

Artifacts:

```text
self_attn source pair selection:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_layer_scan/v9_v8_prex1_expanded_train_val_self_attn_source_pairs_donor10

mlp source pair selection:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_layer_scan/v10_v8_prex1_expanded_train_val_mlp_source_pairs_donor10

self_attn layer-scan pairs:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_layer_scan/v11_self_attn_strict_donor10_repair_and_donor_steer_layers17_18_19_20_21_pairs

mlp layer-scan pairs:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_layer_scan/v12_mlp_strict_donor10_repair_and_donor_steer_layers17_18_19_20_21_pairs

self_attn patch run:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_layer_scan/v13_self_attn_strict_donor10_repair_and_donor_steer_layers17_18_19_20_21_patch

mlp patch run:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_layer_scan/v14_mlp_strict_donor10_repair_and_donor_steer_layers17_18_19_20_21_patch

self_attn taxonomy:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_layer_scan/v15_v13_self_attn_basin_taxonomy

mlp taxonomy:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_layer_scan/v16_v14_mlp_basin_taxonomy

self_attn compatibility:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_layer_scan/v17_v15_self_attn_compatibility

mlp compatibility:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_layer_scan/v18_v16_mlp_compatibility
```

Component summary over layers 17-21:

```text
layer_input:
  mean_coord_rank_delta_by_role={
    repair_candidate: -254.95,
    donor_steer_candidate: 300.63
  }
  stable basin landing emerges:
    donor_steer_candidate: donor basin by L18-L20
    repair_candidate: target basin by L19-L21

self_attn output:
  mean_coord_rank_delta_by_role={
    repair_candidate: -97.00,
    donor_steer_candidate: 44.57
  }
  layer 17 can directly land in donor basin:
    self_attn L17 labels include donor_basin_steer: 7
  but later layers mostly produce rank movement, not stable target/donor
  basin ownership.

mlp output:
  mean_coord_rank_delta_by_role={
    repair_candidate: -81.05,
    donor_steer_candidate: 85.17
  }
  MLP output mostly gives target-rank movement or worsening/escape. It rarely
  lands top-1 in donor/target basin at layers 17-20; target-like top-1 landing
  appears mainly in some repair rows at L21.
```

Component interpretation:

```text
The ownership transition is not simply "MLP writes the coordinate basin." The
late residual state at layers 18-20 is stronger than isolated component-output
patches. Self-attention output can inject donor-basin top-1 early at L17,
especially for far-distance donor-steer rows and repeated-class bird repairs,
but that effect is not the same as stable residual ownership. MLP output is
more like a rank/decision shaper and late stabilizer than a standalone basin
writer in this window.

Current best mechanistic picture:

1. A coordinate cursor enters an ambiguous residual-stream phase by roughly
   layer 16: target rank can improve, but top-1 basin ownership is often still
   not target or donor.
2. Around layers 18-20, the residual stream commits to a coordinate-basin
   owner. Far target-donor distance commits to donor basin; compatible repair
   rows commit to target/near-target basin, sometimes after a transient donor
   exposure at layer 18.
3. Self-attention can expose or inject donor spatial anchors early, but the
   full residual ownership state requires downstream integration.
4. MLP participates in resolving/stabilizing ownership, but isolated MLP output
   is weaker than full residual patching and often moves rank without causing
   top-1 basin landing.

The next highest-value experiments should run as two coupled tracks rather
than one narrow follow-up:

1. Transition-window mediation scan: patch layer input at L18/L19 while
   clamping self-attention or MLP in the same layer and the following layer, or
   perform a residual stream patch from L17 to L19 with component clamps. This
   should separate "attention exposes donor anchors" from
   "MLP/residual integration chooses basin ownership."
2. Broader trained-failure versus val-analog row mining: deliberately sample
   more dataset rows and motifs instead of letting the person/backpack or bird
   anchors define the study. The training split is especially valuable: if a
   trained sequence still fails under teacher-forced GT prefix, the failure is
   less plausibly raw visual non-exposure and more plausibly local
   autoregressive coordinate-basin selection, ownership competition, or
   context-cursor fragility. The matching val rows then test whether unseen
   failures use the same transition layer and component path, or whether they
   fail earlier because weaker evidence never reaches the basin-ownership
   resolver.

## Broader Train-Failure Versus Val-Failure Panel

To follow the broader-row direction, a post-hoc failure-panel reducer was added:

```text
src/analysis/autoregressive_binding_template_ablation/formation_failure_panel.py
scripts/analysis/run_autoregressive_binding_formation_failure_panel.py
tests/analysis/test_formation_failure_panel.py
```

The reducer collapses formation readout rows by case, scores pre-x1 coordinate
failure as `coord_rank_gt + coord_top1_distance`, selects trained failure seeds
per regime, and attaches unseen-val analogs. Val analogs now require their own
minimum failure score, so the default panel compares failure-like train rows to
failure-like val rows rather than silently substituting easy controls.

The first pass reused the existing v8 readout pool:

```text
v8 readout:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_readout_probe/v8_full_train_val_all_positions_teacher_forced_sharded4_alignfix/merged/formation_readout_rows.jsonl

failure panel v2:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_failure_panel/v2_v8_train_failure_val_failure_analog_panel

formation rows from v2:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/panel_formation_row_builder/v3_v2_failure_panel_teacher_forced_gt
```

v2 selected 40 object rows: 20 train failure seeds and 20 val failure analogs,
balanced across crowded, duplicate-basin-nearby, repeated-class, small-object,
and termination-tail regimes. The mean pre-x1 failure score was higher for train
than val on this selected panel (`train=488.25`, `val=437.4`), which already
rules out a simple "only unseen val rows fail" explanation.

A receiver-case filter was then added to the replay-patch pair selector so
receivers can be restricted to this failure panel while donors still come from
the broader v8 readout pool:

```text
pair selection:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v34_v8_failure_panel_receivers_layer20_donor10_pairs

layer-20 patch shards, image-root fixed:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v36_v34_failure_panel_layer20_patch_sharded4_imagefix

single OOM retry:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v38_v37_oom_row_retry_patch

merged patch rows:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v39_v36_v38_failure_panel_layer20_patch_merged

basin taxonomy:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_basin_taxonomy/v19_v39_failure_panel_layer20

compatibility:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_compatibility/v6_v19_failure_panel_layer20
```

Operational note: the first sharded patch attempt used the wrong image root and
failed all rows with `FileNotFoundError` for
`.../rescale_32_1024_bbox_len12000/images`. The corrected image root is:

```text
/data/CoordExp/public_data/coco/rescale_32_1024_bbox
```

The corrected sharded run produced 199/200 rows, with one row failing from a
GPU0 OOM. Retrying that single pair on GPU3 succeeded, giving a complete
200-row merged patch artifact.

Layer-20 v2 failure-panel causal result:

```text
merged_patch_row_count=200
mean_coord_rank_delta=124.795
coord_rank_improved_count=33
coord_rank_worsened_count=86
receiver rows: train=100, val=100
```

The basin/compatibility reducer shows this is not a generic repair effect:

```text
patch_basin_label counts:
  baseline_no_patch: 40
  near_target_repair: 4
  target_rank_repair: 1
  donor_basin_steer: 15
  origin_basin_collapse: 94
  no_change_or_unclear: 36
  rank_worsened: 3
  worsened_escape: 6
  rank_only_improvement_worse_top1: 1

simple_control donors:
  90/90 origin_basin_collapse

same_split_regime_desc donors:
  train: n=6, mean_delta=-91.17, labels={near_target_repair:3,
         donor_basin_steer:2, worsened_escape:1}
  val:   n=5, mean_delta=496.0, labels={donor_basin_steer:5}

cross_split_regime_desc donors:
  train: n=4, mean_delta=514.0, labels={donor_basin_steer:4}
  val:   n=10, mean_delta=89.6, labels={worsened_escape:5,
         donor_basin_steer:2, rank_worsened:2,
         rank_only_improvement_worse_top1:1}
```

Interpretation:

```text
Trained-sequence failures are genuine local coordinate-cursor failures, not a
mere absence of training exposure. Generic good-coordinate donors are actively
bad controls for these hard receivers because they collapse to the origin
basin. Compatible donors can repair a small number of train rows, but they more
often steer the receiver into the donor basin, especially across split or in val
same-split compatible rows. This strengthens the ownership-resolver picture:
the model often has enough descriptor/schema and later coordinate evidence, but
the pre-x1 basin owner is selected by a fragile local state that can be captured
by origin, donor, or target depending on donor/receiver compatibility.
```

To broaden beyond the 96-case v8 pool, a larger structure-only bank was built:

```text
per-regime-per-split=24 candidate bank:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/train_val_candidate_bank/v4_bbox_len12000_gt_structure_regimes_per24_imgcap2

formation rows, 288 objects x 8 positions = 2304 rows:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/panel_formation_row_builder/v4_v4_per24_candidate_bank_teacher_forced_gt

4-GPU readout, merged:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_readout_probe/v9_per24_train_val_all_positions_teacher_forced_sharded4/merged

failure panel v3:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_failure_panel/v3_v9_per24_train_failure_val_failure_analog_panel
```

The v9 readout completed with `output_row_count=2304`,
`hidden_row_count=9216`, and `error_count=0`.

Broad v9 pre-x1 distribution:

```text
train pre_x1 rows: 144
  failure score >=25: 82
  severe score >=100: 66
  among failures, post_x1 rank <=10: 63

val pre_x1 rows: 144
  failure score >=25: 96
  severe score >=100: 67
  among failures, post_x1 rank <=10: 66
```

By regime:

```text
train crowded: n=24, fail>=25 20, mean_score 323.3
train duplicate_basin_nearby: n=24, fail>=25 13, mean_score 299.4
train repeated_class: n=24, fail>=25 21, mean_score 379.2
train simple_control: n=24, fail>=25 0, mean_score 1.4
train small_object: n=24, fail>=25 18, mean_score 328.5
train termination_tail: n=24, fail>=25 10, mean_score 174.7

val crowded: n=24, fail>=25 21, mean_score 363.9
val duplicate_basin_nearby: n=24, fail>=25 16, mean_score 150.2
val repeated_class: n=24, fail>=25 22, mean_score 443.7
val simple_control: n=24, fail>=25 2, mean_score 9.8
val small_object: n=24, fail>=25 18, mean_score 336.6
val termination_tail: n=24, fail>=25 17, mean_score 208.9
```

The v3 selected failure panel has 80 object rows: 40 train seeds and 40 val
analogs, with 66 severe pre-x1 failures and 12 near-target rank failures. It
expands descriptors beyond the earlier bird/person-heavy panel, including
backpack, baseball glove, book, bowl, car, cell phone, dining table, fork,
knife, laptop, sheep, skateboard, snowboard, spoon, sports ball, traffic light,
and tv.

Updated next directions:

```text
1. Build a v3 receiver-filtered pair selection over the v9 readout pool, but
   avoid letting simple-control donors dominate. Either cap simple-control
   donors separately or select a compatible-donor-only panel plus a separate
   explicit origin-control panel.
2. Run transition-layer scans on v3 receivers after pair selection, with layers
   17-21 first. The v2 layer-20 run says compatible repair is rare and donor
   capture is common; the layer scan should reveal whether train failures enter
   the donor/origin basin earlier or whether layer 20 is already after the
   decisive ownership choice.
3. Use v9 hidden rows to compare trained failures whose post_x1 recovers versus
   those whose post_x1 stays bad. This is likely the cleanest split between
   x1-onset cursor failure and broader visual/geometry evidence weakness.
4. Treat simple-control origin collapse as a diagnostic phenomenon in its own
   right, not merely a bad control. It may expose a strong origin/default
   coordinate basin that competes with object-local ownership under weak
   compatibility.
```

## V3 Transition-Layer Ownership Split

The v9/v3 receiver-filtered pair selector was materialized at:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v40_v9_v3_failure_panel_receivers_layer20_donor10_pairs
```

It selected 400 rows over 80 receivers from the v3 failure panel:

```text
receiver_count: 80
pair_count: 400
train receiver rows: 200
val receiver rows: 200
control kinds:
  baseline_no_patch: 80
  self_noop: 80
  same_image_regime_desc_donor: 6
  same_split_regime_desc_donor: 67
  same_regime_desc_donor: 46
  simple_control_donor: 121
```

To avoid mixing donor-ownership capture with the simple-control origin basin,
two transition scans were split out deliberately:

```text
compatible donor layer-scan pairs:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_layer_scan/v19_v40_v3_compatible_donors_layers17_18_19_20_21_pairs

compatible donor patch merge:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_layer_scan/v22_v21_compatible_donors_layers17_21_patch_merged

compatible donor basin taxonomy:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_layer_scan/v23_v22_compatible_donors_layers17_21_basin_taxonomy

compatible donor compatibility reducer:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_layer_scan/v24_v23_compatible_donors_layers17_21_compatibility

simple-control origin layer-scan pairs:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_layer_scan/v20_v40_v3_simple_control_origin_layers17_18_19_20_21_pairs

simple-control origin patch merge:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_layer_scan/v26_v25_simple_control_origin_layers17_21_patch_merged

simple-control origin basin taxonomy:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_layer_scan/v27_v26_simple_control_origin_layers17_21_basin_taxonomy

simple-control origin compatibility reducer:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_layer_scan/v28_v27_simple_control_origin_layers17_21_compatibility
```

Compatible-donor transition result:

```text
source pairs: 119
layer-scan rows: 595
error_count_after_merge: 0

all rows, n=119 per layer:
  L17 mean_delta=70.67, donor_basin_steer=3,
      near/exact/target-rank repair=29, origin_basin_collapse=38
  L18 mean_delta=103.04, donor_basin_steer=79,
      near/target-rank repair=15
  L19 mean_delta=126.48, donor_basin_steer=87
  L20 mean_delta=156.18, donor_basin_steer=88,
      exact_target_repair=1, near/target-rank repair=12
  L21 mean_delta=158.02, donor_basin_steer=87,
      exact_target_repair=1, near/target-rank repair=13
```

The transition is not "layer 20 writes the answer" in a simple sense. Layer 17
is mixed: origin, target-ish repair, and other states coexist. At layer 18 the
same compatible donors mostly snap into donor-basin ownership, and that remains
stable through layers 19-21. Same-image compatible donors are the important
exception: the tiny same-image group is consistently reparative, with mean rank
deltas around -182 to -192 and several near/exact target repairs. This proves
that genuine target-basin repair exists, but it requires local compatibility;
rank-only improvement is often a donor-ownership artifact.

Simple-control origin transition result:

```text
source pairs: 121
layer-scan rows: 605
error_count_after_merge: 0

overall basin labels:
  origin_basin_collapse: 601
  donor_basin_steer: 3
  worsened_escape: 1

by layer, n=121 per layer:
  L17 mean_delta=172.25, origin_basin_collapse=118
  L18 mean_delta=188.60, origin_basin_collapse=120
  L19 mean_delta=228.48, origin_basin_collapse=121
  L20 mean_delta=242.39, origin_basin_collapse=121
  L21 mean_delta=245.08, origin_basin_collapse=121
```

Interpretation:

```text
There are at least three coordinate-basin attractor regimes:

1. Same-image/local compatible target repair: rare but real, and visible across
   layers 17-21.
2. Donor-basin capture: broad compatible donors move from mixed/origin/other at
   L17 to donor basin at L18, then stay donor-dominant.
3. Origin/default capture: simple-control donor vectors push almost every hard
   receiver to the origin/donor basin already at L17 and saturate by L19-L21.

This refines the ownership-transition picture. Layer 17 can already expose
origin capture and some real repair, but non-origin donor ownership mostly snaps
at layer 18. Full residual ownership choice is not the same as target rank:
target rank can improve while the emitted owner is still donor or origin.
```

## V9 Recovery-vs-Nonrecovery Hidden Split

To follow the broader-row hint, a post-hoc reducer was added and run over the
full v9 train/val readout plus hidden rows:

```text
script:
scripts/analysis/run_autoregressive_binding_formation_recovery_hidden_compare.py

artifact:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_recovery_hidden_compare/v1_v9_prex1_failure_postx1_recovery_hidden_compare

inputs:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_readout_probe/v9_per24_train_val_all_positions_teacher_forced_sharded4/merged/formation_readout_rows.jsonl
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_readout_probe/v9_per24_train_val_all_positions_teacher_forced_sharded4/merged/formation_readout_hidden_rows.jsonl

case_row_count: 178
hidden_row_count: 1424
layer_summary_row_count: 184
```

Recovery labels among broad pre-x1 failures:

```text
post_x1_recovered: 161
post_x1_nonrecovered: 17

train: 75 recovered, 7 nonrecovered
val:   86 recovered, 10 nonrecovered
```

Regime split:

```text
crowded: 39 recovered, 2 nonrecovered
duplicate_basin_nearby: 27 recovered, 2 nonrecovered
repeated_class: 42 recovered, 1 nonrecovered
simple_control: 2 recovered, 0 nonrecovered
small_object: 31 recovered, 5 nonrecovered
termination_tail: 20 recovered, 7 nonrecovered
```

Layerwise logit-lens summary, all regimes:

```text
pre_x1, final recorded layer L27:
  train recovered:     n=75, mean_rank=207.0, mean_distance=229.7
  train nonrecovered:  n=7,  mean_rank=177.1, mean_distance=268.3
  val recovered:       n=86, mean_rank=197.2, mean_distance=171.6
  val nonrecovered:    n=10, mean_rank=245.0, mean_distance=166.0

post_x1, final recorded layer L27:
  train recovered:     n=75, mean_rank=6.57,  mean_distance=3.41
  train nonrecovered:  n=7,  mean_rank=43.43, mean_distance=33.29
  val recovered:       n=86, mean_rank=7.72,  mean_distance=5.02
  val nonrecovered:    n=10, mean_rank=106.4, mean_distance=55.0
```

The important asymmetry is that most failures are still bad in the pre-x1
hidden/logit-lens state even at the final recorded layer, but immediately after
teacher-forcing the x1 token, most rows snap to a good y1 distribution. This is
true for trained rows and val rows. The trained rows therefore cannot be
explained as "model never saw the sequence"; many trained rows have enough
later coordinate evidence but enter the wrong x1 onset basin.

The residual nonrecovered tail is smaller and different:

```text
nonrecovered object_idx mean: 26.8
recovered object_idx mean: 14.8
nonrecovered bbox_area mean: 8336.4
recovered bbox_area mean: 19483.9

nonrecovered examples concentrate in termination_tail and small_object:
  train termination_tail: 3
  val termination_tail: 4
  train small_object: 2
  val small_object: 3
```

Interpretation:

```text
The broad train/val evidence now supports a two-part mechanism:

1. Common failure mode: x1-onset cursor/basin selection is fragile. The model
   has schema/type stability and usually recovers once x1 is supplied, so the
   visible miss is often not raw visual non-perception.
2. Residual hard tail: some trained and val rows remain bad at post_x1. These
   are enriched for tail/stop positions, tiny objects, and awkward crowded or
   duplicated contexts. They are better candidates for genuine weak visual
   evidence, delayed distributed evidence, or termination/routing conflict.
```

Updated next directions:

```text
1. For the next GPU pass, do not return to a single person/backpack anchor.
   Build a compact but motif-diverse nonrecovered-tail panel, with train and val
   rows from termination_tail, small_object, crowded, and duplicate_basin_nearby.
2. Run continuation on the same-image/local repair cases separately from donor
   capture cases. The key test is whether true target-basin repair changes the
   generated object span, not merely the next-token rank.
3. For nonrecovered-tail rows, compare visual-side intervention and language
   prefix guidance: force/patch x1, post_x1, descriptor, and termination context
   separately to decide whether the failure is perception weakness, coordinate
   cursor weakness, or next-object/stop routing conflict.
4. Mine more train rows if the nonrecovered tail remains sparse. A trained row
   that stays bad after x1 is much more mechanistically valuable than another
   easy val analogy.
```

## Nonrecovered-Tail Causal Seed and Layer Scan

The broad recovery reducer produced a compact hard-tail receiver file:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_recovery_hidden_compare/v1_v9_prex1_failure_postx1_recovery_hidden_compare/nonrecovered_receiver_case_rows.jsonl

row_count: 17
train: 7
val: 10
```

Two rejected operational runs should not be used as evidence:

```text
v41_v9_nonrecovered_tail_receivers_layer18_donor10_pairs:
  rejected because the receiver-case file contained all 178 recovery rows,
  not only the 17 nonrecovered rows.

v43_v42_nonrecovered_tail_layer18_seed_patch:
  rejected because image_root was set to the JSONL-only len12000 root and all
  85 rows failed with FileNotFoundError for .../bbox_len12000/images.
```

The corrected nonrecovered-only pair panel:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v42_v9_nonrecovered_only_receivers_layer18_donor10_pairs

receiver_count: 17
pair_count: 85
control kinds:
  baseline_no_patch: 17
  self_noop: 17
  same_regime_desc_donor: 8
  same_split_regime_desc_donor: 13
  simple_control_donor: 30
receiver regimes:
  crowded: 10
  duplicate_basin_nearby: 10
  repeated_class: 5
  small_object: 25
  termination_tail: 35
```

Corrected layer-18 seed patch:

```text
patch:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v44_v42_nonrecovered_tail_layer18_seed_patch_imagefix

basin:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_basin_taxonomy/v29_v44_nonrecovered_tail_layer18_seed

compatibility:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_compatibility/v29_v29_nonrecovered_tail_layer18_seed

patch_row_count: 85
error_count: 0
mean_coord_rank_delta: +57.64
improved: 21
worsened: 30
```

Layer-18 seed basin labels:

```text
baseline_no_patch: 17
donor_basin_steer: 17
near_target_repair: 2
target_rank_repair: 1
origin_basin_collapse: 33
no_change_or_unclear: 14
worsened_escape: 1
```

By donor/receiver relation:

```text
simple_control:
  origin_basin_collapse: 29
  donor_basin_steer: 1

same_image_regime_desc:
  no_change_or_unclear: 14
  origin_basin_collapse: 3

same_split_regime_desc:
  donor_basin_steer: 11
  near_target_repair: 2

cross_split_regime_desc:
  donor_basin_steer: 5
  origin_basin_collapse: 1
  target_rank_repair: 1
  worsened_escape: 1
```

The focused layer-scan panel was then built from the seed compatibility labels:

```text
pairs:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_layer_scan/v29_v42_nonrecovered_tail_seed_selected_layers17_21_pairs

patch:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_layer_scan/v30_v29_nonrecovered_tail_layers17_21_patch

basin:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_layer_scan/v31_v30_nonrecovered_tail_layers17_21_basin_taxonomy

compatibility:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_layer_scan/v32_v31_nonrecovered_tail_layers17_21_compatibility

source pair count: 10
layer-scan row count: 50
error_count: 0
```

Layer-scan summary:

```text
L17: n=10, mean_delta=+122.5
  labels: target_rank_repair=5, near_target_repair=1,
          origin_basin_collapse=3, worsened_escape=1

L18: n=10, mean_delta=+147.8
  labels: near_target_repair=2, donor_basin_steer=8

L19: n=10, mean_delta=+165.1
  labels: near_target_repair=2, donor_basin_steer=8

L20: n=10, mean_delta=+205.7
  labels: near_target_repair=2, donor_basin_steer=8

L21: n=10, mean_delta=+219.1
  labels: near_target_repair=2, donor_basin_steer=8
```

Relation split:

```text
same_split_regime_desc:
  L17: n=7, mean_delta=+16.7,
       target_rank_repair=5, near_target_repair=1, origin_basin_collapse=1
  L18-L21: n=7 per layer,
       near_target_repair=2 and donor_basin_steer=5 at each layer

cross_split_regime_desc:
  L17: n=3, mean_delta=+369.3,
       origin_basin_collapse=2, worsened_escape=1
  L18-L21: n=3 per layer,
       donor_basin_steer=3 at each layer
```

The only robust near-target repairs in the hard-tail layer scan are train
termination-tail rows where the donor x1 is also near the target x1:

```text
train termination_tail cell phone:
  baseline rank 184, target x1 412, donor x1 406
  L18-L21 patched rank 11, distance 6

train termination_tail knife:
  baseline rank 459, target x1 114, donor x1 124
  L18 patched rank 11, distance 2
  L19-L21 patched rank 5/5/4, distance 5/3/5
```

Interpretation:

```text
The nonrecovered tail is not just the common x1-onset failure replayed on
harder rows. It is more resistant:

1. Generic simple-control donors still expose the origin basin almost
   deterministically.
2. Cross-split compatible donors are actively harmful and become donor capture
   by L18-L21.
3. Same-split compatible donors have a softer L17 window, mostly rank-only
   repair, but from L18 onward only near-target donor cases remain truly
   reparative; the rest again become donor capture.
4. The rows that repair are not evidence for general perception rescue; they
   look like coordinate-neighbor guidance. The hard tail therefore needs a
   different next intervention: force or patch x1/post_x1/termination context
   and inspect whether the object span can be completed, rather than expecting
   generic compatible hidden states to repair the basin.
```
