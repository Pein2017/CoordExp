---
doc_id: progress.diagnostics.formation_span_continuation_findings_2026_06_22
layer: progress
doc_type: diagnostic-implementation-note
status: active-branch-evidence
evidence_scope: gpu-probe-formation-prefix-continuation
domain: autoregressive-binding-template-ablation
updated: 2026-06-22
branch: codex/autoregressive-binding-template-study
---

# Formation Span Continuation Findings

## Purpose

This note records the first span-aware continuation probes for checkpoint-928.
The core question is whether pre-x1 failures are visual non-perception, or
language-side/cursor fragility where the model can complete the object span
once the correct x1 coordinate is supplied.

## Implementation

Added:

```text
src/analysis/autoregressive_binding_template_ablation/formation_span_continuation.py
scripts/analysis/run_autoregressive_binding_formation_span_continuation.py
tests/analysis/test_formation_span_continuation.py
```

The runner builds continuation plan rows from either formation replay
compatibility rows or raw formation readout rows, keeps the final assistant
message open with `continue_final_message=true`, uses the existing
token_embeddings_adapter checkpoint loader, calls the processor with
`do_resize=false`, supports deterministic sharding after mode expansion, and
records taxonomy-compatible continuation rows.

Supported modes:

```text
pre_x1_free
force_target_x1
force_post_x1
force_target_x1_y1
force_target_x1_y1_x2
force_target_full_box
force_patched_x1
force_donor_x1
```

Verification:

```text
python -m pytest tests/analysis/test_formation_span_continuation.py -q
7 passed
```

The TDD loop observed red failures for missing module, missing shard API,
missing `continue_final_message=true`, missing raw-readout-row fallback, and
missing source metadata passthrough.

Post-review hardening added:

```text
generated_bbox_valid now requires a complete object box, including <|box_end|>
generated_bbox_coords_valid records relaxed coordinate geometry separately
--dry-run --run-continuation is rejected to avoid accidental GPU execution
run summaries distinguish executed plan rows from pre-truncation plan rows
```

## Probe A: Compatibility-Selected Nonrecovered Tail

Source:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_layer_scan/v32_v31_nonrecovered_tail_layers17_21_compatibility/formation_replay_patch_compatibility_rows.jsonl
```

Execution:

```text
4 GPU shards on CUDA_VISIBLE_DEVICES=0,1,2,3
modes: pre_x1_free, force_target_x1, force_patched_x1, force_donor_x1
plan rows: 200
continuation rows: 200
error_count: 0
```

Shard roots:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_span_continuation/v3_v32_all50_shard0_gpu0
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_span_continuation/v3_v32_all50_shard1_gpu1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_span_continuation/v3_v32_all50_shard2_gpu2
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_span_continuation/v3_v32_all50_shard3_gpu3
```

Taxonomy:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_span_continuation_taxonomy/v4_v3_v32_all50_sharded_batch_taxonomy
```

Strict IoU summary:

```text
pre_x1_free:      mean IoU 0.1477, IoU>=0.5 count 5/50
force_target_x1:  mean IoU 0.2927, IoU>=0.5 count 10/50
force_patched_x1: mean IoU 0.0350, IoU>=0.5 count 0/50
force_donor_x1:   mean IoU 0.0288, IoU>=0.5 count 0/50
```

Read: target-x1 forcing can improve span completion, but donor/patched x1 from
the layer-scan rows usually does not repair the whole object. The panel has
only 6 unique receiver states, so it is useful for patch/donor comparison but
too narrow for broad dataset conclusions.

## Probe B: Broad Train/Val Pre-X1 Failure Panel

Source join:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_span_continuation_sources/v1_v9_prex1_failure_178_readout_join
```

The source join maps:

```text
v1_v9_prex1_failure_postx1_recovery_hidden_compare/formation_recovery_hidden_case_rows.jsonl
```

back to:

```text
v9_per24_train_val_all_positions_teacher_forced_sharded4/merged/formation_readout_rows.jsonl
```

using `formation_source_state_key` and `formation_position=pre_x1`.

Coverage:

```text
row_count: 178 unique states
train: 82
val: 96
candidate regimes:
  crowded: 41
  duplicate_basin_nearby: 29
  repeated_class: 43
  simple_control: 2
  small_object: 36
  termination_tail: 27
target-desc classes: 29
post_x1_recovered: 161
post_x1_nonrecovered: 17
```

Execution:

```text
4 GPU shards on CUDA_VISIBLE_DEVICES=0,1,2,3
modes: pre_x1_free, force_target_x1
plan rows: 356
continuation rows: 356
error_count: 0
```

Shard roots:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_span_continuation/v5_v9_prex1_failure_178_shard0_gpu0_free_force_target
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_span_continuation/v5_v9_prex1_failure_178_shard1_gpu1_free_force_target
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_span_continuation/v5_v9_prex1_failure_178_shard2_gpu2_free_force_target
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_span_continuation/v5_v9_prex1_failure_178_shard3_gpu3_free_force_target
```

Taxonomy:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_span_continuation_taxonomy/v5_v5_v9_prex1_failure_178_free_force_target_taxonomy
```

## Broad Result

Syntax and closure:

```text
pre_x1_free:
  complete_valid: 162/178
  missing_box_end: 15/178
  complete_invalid: 1/178

force_target_x1:
  complete_valid: 162/178
  missing_box_end: 16/178
  complete_invalid: 0/178
```

Strict target IoU requiring `complete_valid`:

```text
pre_x1_free:
  IoU>=0.30: 26/178
  IoU>=0.50: 18/178
  IoU>=0.75: 6/178

force_target_x1:
  IoU>=0.30: 116/178
  IoU>=0.50: 82/178
  IoU>=0.75: 37/178
```

The v3/v5 continuation artifacts were generated before the stricter
`generated_bbox_valid` field semantics landed. The strict counts in this note
therefore use post-hoc `generated_object_parse_status == complete_valid` plus
IoU thresholds, not the older relaxed raw `generated_bbox_valid` field.

Train versus val under `force_target_x1` with complete-valid IoU>=0.5:

```text
train: 37/82
val: 45/96
```

This is not a simple train-versus-val split.

Regime under `force_target_x1` with complete-valid IoU>=0.5:

```text
crowded: 21/41
duplicate_basin_nearby: 20/29
repeated_class: 22/43
small_object: 1/36
termination_tail: 18/27
simple_control: 0/2, both missing box_end in this source slice
```

Post-x1 recovery relation under `force_target_x1`:

```text
post_x1_recovered:
  n: 161
  complete-valid IoU>=0.30: 111
  complete-valid IoU>=0.50: 80
  complete-valid IoU>=0.75: 37

post_x1_nonrecovered:
  n: 17
  complete-valid IoU>=0.30: 5
  complete-valid IoU>=0.50: 2
  complete-valid IoU>=0.75: 0
```

Free-versus-forced categories over 178 states:

```text
guidance_rescue_strict: 66
guidance_partial_rescue: 33
guidance_overlap_only: 24
already_good__still_good: 20
small_gain: 16
guidance_no_gain_or_worse: 19
```

`guidance_rescue_strict` means free IoU<0.3 and force-target complete-valid
IoU>=0.5. The largest concentrations are crowded, repeated-class,
duplicate-nearby, and termination-tail cases.

Small objects are the clearest exception:

```text
small_object force_target_x1 complete-valid IoU>=0.5: 1/36
```

## Probe C: Staged Target-Coordinate Forcing

Purpose: split `pre_x1` failure into x1-onset, y1/row-anchor,
x2/extent, y2/final-extent, and `<|box_end|>` closure/router failure. This
probe uses the same 178-state broad train/val pre-x1 failure panel as Probe B,
but expands each state into five deterministic continuation modes.

Plan dry run:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_span_continuation/v6_v9_prex1_failure_178_staged_forcing_plan
plan_row_count: 890
```

Execution:

```text
2 GPU shards on CUDA_VISIBLE_DEVICES=3,4
modes: pre_x1_free, force_post_x1, force_target_x1_y1, force_target_x1_y1_x2, force_target_full_box
plan rows: 890
continuation rows: 890
error_count: 0
```

Shard roots:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_span_continuation/v7_v9_prex1_failure_178_staged_shard0_gpu3
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_span_continuation/v7_v9_prex1_failure_178_staged_shard1_gpu4
```

Taxonomy:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_span_continuation_taxonomy/v6_v7_v9_prex1_failure_178_staged_taxonomy
```

Strict target IoU requiring `complete_valid`:

```text
pre_x1_free:
  complete_valid: 162/178
  missing_box_end: 15/178
  IoU>=0.30: 26/178
  IoU>=0.50: 18/178
  IoU>=0.75: 6/178

force_post_x1:
  complete_valid: 162/178
  missing_box_end: 16/178
  IoU>=0.30: 116/178
  IoU>=0.50: 82/178
  IoU>=0.75: 37/178

force_target_x1_y1:
  complete_valid: 162/178
  missing_box_end: 15/178
  complete_invalid: 1/178
  IoU>=0.30: 127/178
  IoU>=0.50: 94/178
  IoU>=0.75: 48/178

force_target_x1_y1_x2:
  complete_valid: 162/178
  missing_box_end: 15/178
  complete_invalid: 1/178
  IoU>=0.30: 153/178
  IoU>=0.50: 137/178
  IoU>=0.75: 95/178

force_target_full_box:
  complete_valid: 164/178
  missing_box_end: 14/178
  IoU>=0.30: 164/178
  IoU>=0.50: 164/178
  IoU>=0.75: 164/178
```

Train versus val with complete-valid IoU>=0.5:

```text
pre_x1_free:          train 7/82,  val 11/96
force_post_x1:        train 37/82, val 45/96
force_target_x1_y1:   train 46/82, val 48/96
force_target_x1_y1_x2: train 66/82, val 71/96
force_target_full_box: train 78/82, val 86/96
```

Regime readout with complete-valid IoU>=0.5:

```text
small_object:
  force_post_x1: 1/36
  force_target_x1_y1: 4/36
  force_target_x1_y1_x2: 20/36
  force_target_full_box: 34/36

crowded:
  force_post_x1: 21/41, invalid 10
  force_target_x1_y1: 22/41, invalid 11
  force_target_x1_y1_x2: 27/41, invalid 11
  force_target_full_box: 31/41, invalid 10

repeated_class:
  force_post_x1: 22/43
  force_target_x1_y1: 27/43
  force_target_x1_y1_x2: 40/43
  force_target_full_box: 43/43

termination_tail:
  force_post_x1: 18/27
  force_target_x1_y1: 21/27
  force_target_x1_y1_x2: 25/27
  force_target_full_box: 27/27
```

Full-box closure tails:

```text
<|box_end|>: 164
<|object_ref_end|>: 9
<|object_ref_start|>: 5
```

Strict IoU>=0.5 patterns over modes ordered as
`pre_x1_free, force_post_x1, force_target_x1_y1, force_target_x1_y1_x2,
force_target_full_box`:

```text
0,1,1,1,1: 61
0,0,0,1,1: 37
0,0,0,0,1: 26
0,0,1,1,1: 18
1,1,1,1,1: 15
0,0,0,0,0: 14
```

Interpretation:

```text
61 states: x1 onset/cursor failure; forcing x1 already rescues and later
           forcing mostly preserves the span.
37 states: x1/y1 are insufficient, but x2 forcing rescues; the failure lives
           in width/extent commitment.
26 states: even x2 is insufficient, but full-box forcing closes; these are
           mainly y2/final-extent failures rather than closure failures.
14 states: full target box still fails closure, usually by emitting
           <|object_ref_end|> or <|object_ref_start|> instead of <|box_end|>.
```

This makes the broad train/val story sharper. The mechanism is not primarily a
train-versus-unseen split: trained rows fail under GT prefix too, and staged
forcing improves train and val similarly. The more useful split is by failure
locus. Small objects are mostly extent/final-coordinate failures, repeated-class
and termination-tail rows are largely coordinate-onset/extent failures with
healthy closure once the box is known, and crowded rows carry a real closure or
next-object-router component.

Next direction: expand beyond the current 178-state readout-selected panel by
mining more train rows and matched val analogs. The priority examples are
trained sequences that still fail under teacher-forced GT prefix, because they
separate visual exposure from local autoregressive coordinate-basin selection.
The next panel should be motif-balanced across object class, object order,
scale, crowding, same-class repetition, duplicate-nearby distance, and tail
position, not anchored on the original person/backpack microscope case.

## Probe D: Per48 Train/Val Expansion

Purpose: test whether the Probe C failure-locus split survives a broader
dataset-row panel, especially on trained sequences. This pass doubles the GT
structure bank from `per_regime_per_split=24` to 48 and then selects all
pre-x1 coordinate-basin failures rather than only the compact failure panel.

Candidate bank:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/train_val_candidate_bank/v5_bbox_len12000_gt_structure_regimes_per48_imgcap2
row_count: 576
train: 288
val: 288
regimes: crowded, duplicate_basin_nearby, repeated_class, simple_control, small_object, termination_tail, 48 each
```

Formation rows:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/panel_formation_row_builder/v5_v5_per48_candidate_bank_teacher_forced_gt
row_count: 4608
positions per object: descriptor_onset, descriptor_end, object_ref_end, box_start, pre_x1, post_x1, box_close, next_object_onset
```

Readout:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_readout_probe/v13_per48_train_val_all_positions_teacher_forced_sharded4_imagefix/merged
selected_row_count: 4608
output_row_count: 4242
hidden_row_count: 16968
error_count: 366
```

The readout is complete for coordinate and schema positions, including all
`pre_x1` and `post_x1` rows. The 366 errors are descriptor-slot rows where
`target_next_token_text` encodes to multiple tokens, e.g. `sheep`,
`cell phone`, and `traffic light`. Treat descriptor-onset/descriptor-end
statistics from this artifact as partial until the readout scorer handles
multi-token descriptor targets.

Pre-x1 failure source selection:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_span_continuation_sources/v2_v13_per48_prex1_failure_328_readout_join
selection_scope: v13_per48_pre_x1_rank_or_distance_ge25
row_count: 328
train: 157
val: 171
post_x1_recovered: 243
post_x1_nonrecovered: 85
```

Regime counts in the 328 pre-x1 failures:

```text
crowded: 74
duplicate_basin_nearby: 55
repeated_class: 79
simple_control: 7
small_object: 77
termination_tail: 36
```

The expanded readout confirms that trained-sequence failures are abundant:

```text
train pre_x1 failures: 157/288
val pre_x1 failures: 171/288
train rough post_x1 recovered: 113/157
val rough post_x1 recovered: 130/171
```

Continuation execution:

```text
4 GPU shards on CUDA_VISIBLE_DEVICES=3,5,7,2
modes: pre_x1_free, force_post_x1, force_target_x1_y1, force_target_x1_y1_x2, force_target_full_box
plan rows: 1640
continuation rows: 1640
error_count: 0
```

Shard roots:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_span_continuation/v9_v13_per48_prex1_failure_328_staged_shard0_gpu3
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_span_continuation/v9_v13_per48_prex1_failure_328_staged_shard1_gpu5
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_span_continuation/v9_v13_per48_prex1_failure_328_staged_shard2_gpu7
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_span_continuation/v9_v13_per48_prex1_failure_328_staged_shard3_gpu2
```

Taxonomy and locus reducer:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_span_continuation_taxonomy/v7_v9_v13_per48_prex1_failure_328_staged_taxonomy
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_failure_locus/v2_v13_per48_prex1_failure_328
```

Staged failure-locus counts with complete-valid IoU>=0.5:

```text
already_successful: 20
x1_onset_rescue: 128
y1_anchor_rescue: 38
x2_extent_rescue: 67
y2_final_extent_rescue: 48
full_box_closure_router_failure: 27
```

Split counts:

```text
train:
  already_successful: 9
  x1_onset_rescue: 59
  y1_anchor_rescue: 21
  x2_extent_rescue: 31
  y2_final_extent_rescue: 25
  full_box_closure_router_failure: 12

val:
  already_successful: 11
  x1_onset_rescue: 69
  y1_anchor_rescue: 17
  x2_extent_rescue: 36
  y2_final_extent_rescue: 23
  full_box_closure_router_failure: 15
```

Regime highlights:

```text
small_object:
  x2_extent_rescue: 29/77
  y2_final_extent_rescue: 31/77
  full_box_closure_router_failure: 3/77

crowded:
  x1_onset_rescue: 36/74
  full_box_closure_router_failure: 17/74

repeated_class:
  x1_onset_rescue: 40/79
  y1_anchor_rescue: 15/79
  x2_extent_rescue: 14/79

termination_tail:
  already_successful: 9/36
  x1_onset_rescue: 18/36
```

Probe D strengthens Probe C rather than overturning it. The main mechanism is
still not a train-versus-val exposure split: train and val have nearly the same
failure-locus mix. The dominant split is local autoregressive formation locus:
x1 onset, y/anchor, x2 extent, y2/final extent, or closure/router. The strongest
new population-level result is that small-object failures are overwhelmingly
extent/final-coordinate failures, while crowded rows carry the most boundary
router failures. This makes `visual non-perception` too broad as an explanation:
many rows, including post_x1-nonrecovered rows, close correctly when the missing
coordinate prefix is supplied.

## Mechanistic Read

The broad staged-forcing result argues against a pure visual non-perception
account for most pre-x1 failures. In 82/178 broad failures, supplying the
correct x1 lets the model emit a complete valid box with IoU>=0.5; forcing
through x2 raises this to 137/178, and forcing the full target box leaves only
14/178 closure/router failures. The train and val rates are similar at each
stage, so the current evidence is not a simple memorization/generalization
split.

The sharper picture is:

```text
1. Many pre-x1 failures are x1-onset/cursor failures, not absence of usable object evidence.
2. Once the x1 basin is supplied, many rows can complete the span, but a large subset still needs x2 or y2/final-extent forcing.
3. Donor or patched x1 from hidden-state probes does not guarantee span repair; it often captures the donor basin.
4. Small objects are dominated by extent/final-coordinate uncertainty rather than pure closure failure.
5. Missing box_end after four supplied coordinates is a separate closure/routing failure, most visible in crowded rows.
6. The 17 post_x1_nonrecovered cases are still not automatically visual-nonperception cases: 16/17 close correctly when the full box is supplied, so many are coordinate-slot failures rather than object absence.
```

Future work should split the problem into:

```text
guidance-rescuable x1 onset failures
small-object / extent and final-coordinate geometry failures
span-closure / next-object routing failures
```

## Next Directions

1. Use the per48 staged-failure-locus reducer as the population scaffold rather
   than returning to a single person/backpack pair. The already-built mechanism
   panel has 89 panel instances / 76 unique receiver cases across train and val:
   small_object_extent 24, crowded_closure_router 17,
   repeated_class_onset_anchor 24, and
   post_x1_nonrecovered_full_box_closable 24.
2. Probe the same object state at staged slots: pre_x1, after forced x1, after
   forced x1+y1, after forced x1+y1+x2, and after forced full box. This is the
   direct test for whether a trained or held-out failure is visual absence,
   language/context guidance fragility, extent-basin instability, or closure
   router failure.
3. For small-object extent rows, combine staged-slot readout with visual crops,
   attention, coordinate logit smoothness, and token_embeddings_adapter delta
   analysis. The target distinction is weak visual evidence versus quantized
   coordinate-smoothness loss versus local coordinate-slot basin collapse.
4. For crowded closure/router rows where full-box forcing emits
   `<|object_ref_end|>` or `<|object_ref_start|>` instead of `<|box_end|>`,
   inspect box_end / next-object onset attention and schema-token margins
   separately from coordinate logits.
5. Re-run hidden-state patching separately on x1-onset rescues, x2/y2 extent
   rescues, and full-box closure/router failures. They are likely different
   mechanisms and should not be collapsed into one average patch effect.

## Probe E: Mechanism-Panel Staged-Slot Readout Rows

This update implements the user's hint that the study should explore broader
dataset/row populations and explicitly compare trained sequences against held-
out analogs, instead of staying constrained by the earlier person/backpack pair.

New deterministic artifacts:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_locus_panel/v1_v13_per48_mechanism_panels
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_readout_rows/v1_v13_mechanism_panels
```

Mechanism-panel counts:

```text
rows: 89
unique_receiver_cases: 76
duplicate_receiver_cases: 13

crowded_closure_router: 17
post_x1_nonrecovered_full_box_closable: 24
repeated_class_onset_anchor: 24
small_object_extent: 24

split balance:
crowded_closure_router: train 9, val 8
post_x1_nonrecovered_full_box_closable: train 12, val 12
repeated_class_onset_anchor: train 12, val 12
small_object_extent: train 12, val 12
```

Staged-slot row counts:

```text
source_row_count: 328
valid_source_case_count: 328
panel_row_count: 89
output_row_count: 445

formation positions:
staged_pre_x1: 89
staged_after_x1: 89
staged_after_x1_y1: 89
staged_after_x1_y1_x2: 89
staged_after_full_box: 89

splits:
train: 225
val: 220

target kinds:
coord: 356
box_end: 89
```

Readout dry-run succeeded for all 445 rows:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_readout_probe/v1_v13_mechanism_panels_dry_run
selected_row_count: 445
```

The first GPU command was launched after a `limit=5` smoke succeeded on the
lowest-memory device. The production 8-GPU prefix-denoising SFT job remained
active, so the full readout was run conservatively as two shards on GPUs 0 and
4 instead of using all devices.

```bash
ROWS=/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_readout_rows/v1_v13_mechanism_panels/staged_slot_readout_rows.jsonl
ROOT=/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_readout_probe/v1_v13_mechanism_panels_shards2
CONFIG=configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml
POSITIONS=staged_pre_x1,staged_after_x1,staged_after_x1_y1,staged_after_x1_y1_x2,staged_after_full_box

CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True \
  python scripts/analysis/run_autoregressive_binding_formation_readout_probe.py \
    --config configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml \
    --formation-rows "$ROWS" \
    --output-root "$ROOT/shard0_gpu0" \
    --family desc_first \
    --image-root /data/CoordExp/public_data/coco/rescale_32_1024_bbox \
    --positions "$POSITIONS" \
    --shard-index 0 \
    --num-shards 2 \
    --layers 0,12,24,-1 \
    --top-k 8

CUDA_VISIBLE_DEVICES=4 PYTORCH_ALLOC_CONF=expandable_segments:True \
  python scripts/analysis/run_autoregressive_binding_formation_readout_probe.py \
    --config "$CONFIG" \
    --formation-rows "$ROWS" \
    --output-root "$ROOT/shard1_gpu4" \
    --family desc_first \
    --image-root /data/CoordExp/public_data/coco/rescale_32_1024_bbox \
    --positions "$POSITIONS" \
    --shard-index 1 \
    --num-shards 2 \
    --layers 0,12,24,-1 \
    --top-k 8
```

Readout artifacts:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_readout_probe/v1_v13_mechanism_panels_gpu0_limit5_smoke
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_readout_probe/v1_v13_mechanism_panels_shards2
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_readout_probe/v1_v13_mechanism_panels_merged_shards2
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_readout_reducer/v1_v13_mechanism_panels_shards2
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_hidden_reducer/v1_v13_mechanism_panels_shards2
```

Merged readout:

```text
selected_row_count: 445
output_row_count: 445
hidden_row_count: 1780
error_count: 0
```

Final-logit staged-slot summary:

```text
staged_pre_x1:
  rank<=10: 1.1%
  top1 distance<=16: 1.1%
  mean coord rank: 401.3
  mean top1 distance: 376.2
  dominant top1 token: <|coord_0|> 25/89

staged_after_x1:
  rank<=10: 65.2%
  top1 distance<=16: 76.4%
  mean coord rank: 20.7
  mean top1 distance: 13.0

staged_after_x1_y1:
  rank<=10: 49.4%
  top1 distance<=16: 87.6%
  mean coord rank: 19.3
  mean top1 distance: 9.4

staged_after_x1_y1_x2:
  rank<=10: 48.3%
  top1 distance<=16: 79.8%
  mean coord rank: 36.4
  mean top1 distance: 15.7

staged_after_full_box:
  box_end top1: 80.9%
  top1 token counts: <|box_end|> 72, <|object_ref_end|> 12, <|object_ref_start|> 5
```

Train/val read:

```text
pre_x1 is similarly bad:
  train rank<=10: 0.0%, mean rank 413.9
  val   rank<=10: 2.3%, mean rank 388.5

after_x1 is similar:
  train rank<=10: 62.2%, distance<=16: 75.6%
  val   rank<=10: 68.2%, distance<=16: 77.3%

y2/final-coordinate after x1+y1+x2 has a held-out weakness:
  train rank<=10: 55.6%, mean rank 15.1, mean distance 8.2
  val   rank<=10: 40.9%, mean rank 58.2, mean distance 23.4

box_end closure is similar overall:
  train box_end top1: 80.0%
  val   box_end top1: 81.8%
```

Panel read:

```text
crowded_closure_router:
  after_x1 rank<=10: 82.4%, distance<=16: 94.1%
  after_full_box box_end top1: 0.0%
  full-box top1 tokens: <|object_ref_end|> 12, <|object_ref_start|> 5

small_object_extent:
  after_x1_y1 x2 rank<=10: 41.7%, distance<=16: 95.8%
  after_x1_y1_x2 y2 rank<=10: 70.8%, distance<=16: 91.7%
  after_full_box box_end top1: 100.0%

repeated_class_onset_anchor:
  after_x1 rank<=10: 70.8%, distance<=16: 83.3%
  after_full_box box_end top1: 100.0%

post_x1_nonrecovered_full_box_closable:
  after_x1 rank<=10: 45.8%, distance<=16: 45.8%
  after_full_box box_end top1: 100.0%
```

Hidden logit-lens summary:

```text
Layer 0 and layer 12:
  coordinate targets are generally not linearly visible; top1 tokens are
  lexical/non-coordinate fragments.
  box_end top1 is 0.0%.

Layer 24:
  staged_after_x1 rank<=10: 42.7%, distance<=16: 74.2%
  staged_after_x1_y1 rank<=10: 23.6%, distance<=16: 62.9%
  staged_after_x1_y1_x2 rank<=10: 28.1%, distance<=16: 61.8%
  staged_after_full_box box_end top1: 75.3%

Final logits:
  staged_after_x1 rank<=10: 64.0%, distance<=16: 76.4%
  staged_after_x1_y1 rank<=10: 49.4%, distance<=16: 87.6%
  staged_after_x1_y1_x2 rank<=10: 47.2%, distance<=16: 79.8%
  staged_after_full_box box_end top1: 80.9%
```

Layer-24 panel split:

```text
crowded_closure_router:
  after_full_box box_end top1: 0.0% at layer 24 and 0.0% at final logits

small_object_extent:
  layer-24 after_x1_y1 x2 distance<=16: 87.5% but rank<=10: 33.3%
  final    after_x1_y1 x2 distance<=16: 95.8% but rank<=10: 41.7%

post_x1_nonrecovered_full_box_closable:
  layer-24 after_full_box box_end top1: 91.7%
  final    after_full_box box_end top1: 100.0%
```

Interpretation target for Probe E:

```text
If train and val rows share the same slot where rank/probability recovers, the
mechanism is likely formation-locus dynamics rather than memorization failure.
If trained rows are easy under staged guidance but val rows are not, the visual
side or exemplar coverage is implicated. If both remain weak even after full
box guidance, the failure is schema/closure routing rather than coordinate
perception. If target rank recovers but generated spans still fail, hidden-state
patching must distinguish local next-token repair from whole-span basin repair.
```

Observed interpretation:

```text
1. The broadest mechanism is not visual non-perception. Many trained and held-
   out failures become locally coordinate-ready after x1 guidance, and 72/72
   non-crowded full-box rows emit <|box_end|> as top1.
2. Pre-x1 failures are genuine onset/basin-selection failures. They are bad on
   trained rows and val rows alike, and remain bad through layer 24 and final
   logits when no coordinate history is supplied.
3. Small-object extent rows often know the local coordinate neighborhood but
   do not put enough rank mass on the exact token. This is the clearest current
   bridge to the user's coordinate-token smoothness concern: locality survives,
   but rank/smoothness within the coordinate basin is fragile.
4. Crowded closure/router rows are not coordinate failures. Their coordinates
   recover under guidance, but after the full target box they choose
   <|object_ref_end|> or <|object_ref_start|> instead of <|box_end|>. This is a
   schema-boundary / next-object-router failure.
5. Hidden logit-lens evidence says the useful coordinate/schema decision is a
   late residual phenomenon: mostly absent at layers 0/12, visible around layer
   24, then sharpened by the final projection and token_embeddings_adapter
   surface.
```

## Probe G: layer-24 to final output-surface bridge

Scope: readout-only join between final staged-slot rows and layer-24 hidden
logit-lens rows over the v13 mechanism panel. This probe asks whether the
token_embeddings_adapter / final output surface mainly destroys a ready hidden
coordinate state, creates/sharpens it, or preserves a deeper router absence.

Artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_surface_bridge/v1_v13_layer24_to_final
```

Inputs:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_readout_probe/v1_v13_mechanism_panels_merged_shards2/formation_readout_rows.jsonl
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_readout_probe/v1_v13_mechanism_panels_merged_shards2/formation_readout_hidden_rows.jsonl
```

Command:

```bash
python scripts/analysis/run_autoregressive_binding_staged_slot_surface_bridge.py \
  --readout-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_readout_probe/v1_v13_mechanism_panels_merged_shards2/formation_readout_rows.jsonl \
  --hidden-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_readout_probe/v1_v13_mechanism_panels_merged_shards2/formation_readout_hidden_rows.jsonl \
  --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_surface_bridge/v1_v13_layer24_to_final \
  --configured-layer 24
```

Transition counts:

```text
bridge rows: 445

coord rank readiness, layer24 -> final:
  hidden_not_ready_final_not_ready: 191
  hidden_not_ready_final_ready:      81
  hidden_ready_final_lost:           19
  hidden_ready_final_ready:          65

coord distance readiness, layer24 -> final:
  hidden_not_ready_final_not_ready: 129
  hidden_not_ready_final_ready:      49
  hidden_ready_final_lost:            9
  hidden_ready_final_ready:         169

after-full-box schema routing, layer24 -> final:
  hidden_box_end_final_box_end:      67
  hidden_router_final_box_end:        5
  hidden_router_final_router:        17
```

Group highlights:

```text
staged_pre_x1:
  rank transition: 88 hidden_not_ready_final_not_ready, 1 hidden_not_ready_final_ready
  mean final-minus-hidden rank delta: -71.33
  mean final-minus-hidden distance delta: +30.52

staged_after_x1:
  rank transition: 28 not_ready->not_ready, 23 not_ready->ready, 3 ready->lost, 35 ready->ready
  mean rank delta: -13.44
  mean distance delta: -2.618

small_object_extent:
  rank delta: -9.823
  distance delta: +15.39
  transition includes 23 not_ready->ready and 10 ready->lost rank cases

crowded_closure_router:
  coord rank delta: -53.57
  distance delta: +5.176
  box_end transition: 17/17 hidden_router_final_router
```

Interpretation:

```text
1. The output surface is not predominantly destroying layer-24 coordinate
   readiness. It often sharpens or creates rank readiness: 81 coordinate rows
   move from hidden-not-ready to final-ready, versus 19 hidden-ready rows that
   are lost at final.
2. Small-object extent remains the cleanest coordinate-token basin example.
   Rank improves through the surface, but mean local distance can worsen. This
   makes the user's locality-versus-smoothness observation sharper: CE can keep
   a local basin while leaving exact-token rank and basin smoothness fragile.
3. Crowded closure/router is not a token_embeddings_adapter-only final flip.
   All 17 after-full-box failures are hidden_router_final_router at layer 24
   and final. The missing state is a deeper schema / next-object-router state,
   not merely a final vocabulary-surface distortion.
4. The next causal step should stay population-first. Compact intervention
   rows are allowed, but only as representatives of train/val locus groups:
   trained-sequence failures, held-out analogs, small-object extent cases, and
   crowded closure/router cases. The original person/backpack microscope case
   should remain an interpretability check, not the sampling prior.
```

## Probe H: population-first causal panel selector

Scope: readout-only selector over Probe G bridge rows. This converts the
surface-bridge evidence into a compact intervention launch panel while keeping
the sampling population broader than the original semantic pair.

Artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_causal_panel/v1_v13_surface_bridge_population_first
```

Command:

```bash
python scripts/analysis/run_autoregressive_binding_staged_slot_causal_panel_selector.py \
  --bridge-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_surface_bridge/v1_v13_layer24_to_final/staged_slot_surface_bridge_rows.jsonl \
  --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_causal_panel/v1_v13_surface_bridge_population_first \
  --max-per-locus-split 6
```

Selector design:

```text
pre_x1_onset_failure:
  staged_pre_x1 / x1 rows where layer24 and final are both coordinate-not-ready.

small_object_rank_locality_tension:
  small-object x2/y2 rows where a local coordinate basin is present but exact
  rank remains weak or the final surface worsens local distance.

crowded_closure_router:
  after-full-box crowded rows where layer24 and final both route away from
  <|box_end|>.

final_y2_extent_failure:
  after-x1-y1-x2 / y2 rows where final rank or distance remains weak, the
  strongest current train/val divergence locus.
```

Observed panel:

```text
bridge rows: 445
candidate rows: 193
selected panel rows: 48

candidate_counts_by_locus:
  pre_x1_onset_failure: 88
  small_object_rank_locality_tension: 42
  crowded_closure_router: 17
  final_y2_extent_failure: 46

candidate_counts_by_locus_split:
  crowded_closure_router::train: 9
  crowded_closure_router::val: 8
  final_y2_extent_failure::train: 20
  final_y2_extent_failure::val: 26
  pre_x1_onset_failure::train: 45
  pre_x1_onset_failure::val: 43
  small_object_rank_locality_tension::train: 21
  small_object_rank_locality_tension::val: 21

selected panel_counts_by_locus:
  pre_x1_onset_failure: 12
  small_object_rank_locality_tension: 12
  crowded_closure_router: 12
  final_y2_extent_failure: 12

selected panel_counts_by_split:
  train: 24
  val: 24
```

Selection read:

```text
The selector confirms that each next causal locus has enough train and val
support to avoid a single-pair microscope. The pre-x1 onset pool contains
nearly equal trained and held-out failures (45 train, 43 val). Small-object
rank/locality tension is exactly split (21 train, 21 val). Crowded closure is
smaller but still bidirectional (9 train, 8 val). Final y2 extent has the
clearest held-out skew (20 train, 26 val), matching the earlier val-y2 warning.

The existing trajectory-hidden-causal-activation-patch runner should not be
used directly on this panel: it expects trajectory readout rows with family,
source_line_idx, stop-reason, and trajectory-state prefix fields. The next GPU
intervention should be a staged-slot-specific patch runner, or an explicit
adapter that reconstructs those fields without changing artifact semantics.
```

## Probe I: staged-slot layer-24 direction-patch smoke

Scope: model-perturbation smoke over the Probe H population-first panel. This
is an immediate next-token causal readout only: the patch is applied at the
selected staged-slot prefix position, logits for the next token are rescored,
and no multi-token continuation is generated yet.

Runner:

```text
src/analysis/autoregressive_binding_template_ablation/staged_slot_direction_patch.py
scripts/analysis/run_autoregressive_binding_staged_slot_direction_patch.py
```

Plan artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_direction_patch/v1_v13_population_first_plan
```

The first plan joins the 48-row selector panel back to the full staged-slot
rows by `state_key`, preserving image path, assistant prefix, target token,
family, source line, and train/val metadata. The first 4 sorted state rows are
crowded closure/router train examples, expanded into 24 planned direction arms.

Failed implementation attempts kept for audit:

```text
v2_crowded_router_layer24_boundary_smoke_g1
  model loaded, but image resolution failed because len12000 image_root has no
  images/ sibling.
v3_crowded_router_layer24_boundary_smoke_g1_imagefix
  image fallback fixed, but scorer helper was missing an injected
  target-token scorer.
```

Both were implementation failures, not model evidence.

Successful crowded closure/router smokes:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_direction_patch/v4_crowded_router_layer24_boundary_smoke_g1_scorefix
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_direction_patch/v5_crowded_router_layer24_boundary_strength_scan_g1
```

Protocol:

```text
checkpoint/config: configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml
visible GPU: CUDA_VISIBLE_DEVICES=1
selection_locus: crowded_closure_router
split: train
state rows: 2
patch site: layer_input
layer: 24
direction: unit(box_end - mean(object_ref_start, object_ref_end)) output-embedding direction
small strengths: 2,4,8
large strengths: 16,32,64,128
```

Crowded small-strength result:

```text
patch rows: 6
error_count: 0
target_rank_improved_rate: 0.0
target_top1_rate: 0.0
box_end_top1_rate: 0.0
mean target logit delta: +0.0833
```

Crowded large-strength result:

```text
patch rows: 8
error_count: 0
target_rank_improved_rate: 0.5
target_top1_rate: 0.375
box_end_top1_rate: 0.375
mean target logit delta: +1.375
mean target rank delta: -0.875
```

Per-case threshold read:

```text
case train image 217268 object 0:
  baseline box_end rank: 3
  strength 64: rank 2, top1 remains <|object_ref_start|>
  strength 128: rank 1, top1 becomes <|box_end|>

case train image 470618 object 0:
  baseline box_end rank: 3
  strength 64: rank 1, top1 becomes <|box_end|>
  strength 128: rank 1, top1 remains <|box_end|>
```

Interpretation:

```text
The crowded closure/router state is locally flippable, but only with a large
layer-24 output-embedding boundary direction. This supports the router-margin
view: <|box_end|> is close enough in final-token competition to be forced, but
the natural hidden state does not independently carry a strong object-boundary
decision. The result does not yet prove continuation repair; a next probe must
patch and continue generation to see whether this is a true boundary-state
repair or only an immediate-token vocabulary flip.
```

Successful pre-x1 coordinate-onset contrast:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_direction_patch/v6_prex1_train_layer24_coord_strength_scan_g1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_direction_patch/v7_prex1_val_layer24_coord_strength_scan_g1
```

Protocol:

```text
selection_locus: pre_x1_onset_failure
splits: train and val, one state each
patch site: layer_input
layer: 24
directions: coord_target_minus_mean and target_minus_top1 output-embedding directions
strengths: 32,64,128
```

Train pre-x1 result:

```text
case train image 350111 object 13, target x1=<|coord_168|>
baseline coord rank: 607
baseline top1: <|coord_782|>, distance 614

coord_target_minus_mean:
  strength 32: rank 405, top1 still <|coord_782|>, distance 614
  strength 64: rank 280, top1 <|coord_793|>, distance 625
  strength 128: rank 98, top1 <|coord_793|>, distance 625

target_minus_top1:
  strength 32: rank 458, top1 still <|coord_782|>, distance 614
  strength 64: rank 347, top1 <|coord_793|>, distance 625
  strength 128: rank 174, top1 <|coord_793|>, distance 625
```

Val pre-x1 result:

```text
case val image 315001 object 8, repeated_class, target x1
baseline coord rank: 957
baseline top1: <|coord_821|>, distance 674

coord_target_minus_mean:
  strength 32: rank 881, top1 still <|coord_821|>, distance 674
  strength 64: rank 721, top1 <|coord_801|>, distance 654
  strength 128: rank 284, top1 <|coord_834|>, distance 687

target_minus_top1:
  strength 32: rank 894, top1 <|coord_825|>, distance 654
  strength 64: rank 770, top1 <|coord_801|>, distance 654
  strength 128: rank 371, top1 <|coord_0|>, distance 147
```

Coordinate-onset interpretation:

```text
Large direct target-coordinate directions improve target rank for train and val
but do not create a local coordinate basin: coord_rank<=10 is 0/12 and
coord_distance<=16 is 0/12 across the two one-state smokes. This makes pre-x1
onset look qualitatively different from crowded box_end routing. It is not a
small missing vocabulary direction; it is a broader coordinate-basin/anchor
selection failure. The val row's strength-128 target_minus_top1 patch moves the
top1 token from a far wrong basin to <|coord_0|>, still not the target basin,
which suggests the direction interacts with global coordinate attractors rather
than simply steering to the desired coordinate.
```

Probe I summary:

```text
crowded closure/router:
  immediate token can be forced with sufficiently large boundary direction.
  next question: does patched <|box_end|> produce a valid continuation boundary
  or only a one-token flip?

pre_x1 coordinate onset:
  direct target-token directions raise rank but fail to create local basin
  readiness. Next question: patch donor state or attention/value evidence from
  a guided post-x1 state, rather than patching only output-embedding direction.
```

Falsified or weakened hypotheses:

```text
visual non-perception as the dominant explanation: weakened.
train memorization versus val generalization as the dominant split: weakened,
  although val y2/final-coordinate rows are worse and need follow-up.
single anecdotal person/backpack-style mechanism: falsified for this panel.
pure coordinate failure for crowded closure rows: falsified.
token_embeddings_adapter surface as the dominant source of crowded closure
  failure: weakened; layer-24 readout is already router-like.
```

Highest-value next causal probes:

```text
1. Population-first layer-24 coordinate-basin intervention for trained
   pre-x1 failures and held-out analogs: add/patch target-local coordinate
   directions, then test whether final rank and generated span repair follow.
2. Small-object extent surface/basin probe: compare local distance-preserving
   hidden evidence with exact-token rank distortion after the
   token_embeddings_adapter / final-head surface, with separate train and val
   rows.
3. Crowded closure/router intervention: patch or steer a
   box_end-minus-object_ref direction and inspect whether the state becomes a
   true object-boundary state or merely flips one next token while continuation
   remains in a next-object basin.
4. Focused val-y2 follow-up, because the only strong train/val divergence in
   this pass is final-coordinate y2 rank/distance after x1+y1+x2 guidance.
```

## Probe J: Staged Direction-Patch Continuation

Purpose: test whether the crowded closure/router boundary direction from Probe
I creates a real object-boundary continuation state, or only flips the
immediate next-token logit to `<|box_end|>`.

Implementation:

```text
src/analysis/autoregressive_binding_template_ablation/staged_slot_direction_patch_continuation.py
scripts/analysis/run_autoregressive_binding_staged_slot_direction_patch_continuation.py
tests/analysis/test_staged_slot_direction_patch_continuation.py
```

The runner reuses the staged-slot direction-patch plan, applies the hidden-state
delta only for the first next-token decision, then appends the argmax token and
continues greedily without further patching. It records baseline and patched
generated token sequences, first-token target status, and whether the second
token matches the expected post-`<|box_end|>` route:
`<|object_ref_start|>` when more objects remain, or `<|im_end|>` at the final
object.

Implementation guard:

```text
Manual greedy continuation truncates sequence-shaped model inputs to the actual
assistant-prefix prediction index before appending generated tokens. This avoids
conditioning continuation steps on chat-template tail tokens that appear after
the open assistant prefix in the processor text.
```

CPU verification:

```text
python -m pytest tests/analysis/test_staged_slot_direction_patch_continuation.py tests/analysis/test_staged_slot_direction_patch.py -q
12 passed
```

GPU smoke:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_direction_patch_continuation/v2_crowded_router_layer24_boundary_firststep_g0_auditfix

checkpoint/config: configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml
source panel: v1_v13_surface_bridge_population_first, crowded_closure_router train rows
staged rows: v1_v13_mechanism_panels
patch site: layer_input
layer: 24
direction: box_end_minus_object_ref_boundaries
strengths: 64,128
continuation_steps: 6
rows: 4
errors: 0
```

Summary:

```text
baseline_first_token_target_rate: 0.0
patched_first_token_target_rate: 0.75
patched_first_token_changed_rate: 0.75
expected_after_target_applicable_count: 4
patched_expected_after_target_evaluable_count: 3
patched_expected_after_target_rate: 1.0 over rows where patched first token was <|box_end|>
```

Audit fields in the corrected v2 rows:

```text
first_step_patch_only: true
continuation_prefix_truncated_to_prediction_token: true
template_tail_token_count: 2
hidden_layer_count: 28
resolved_layer_index: 24
image_path: raw COCO train image path
```

Per-row read:

```text
image 217268, crowded object 0:
  baseline: <|object_ref_start|> book <|object_ref_end|> <|box_start|> <|coord_624|> <|coord_0|>
  strength 64: unchanged
  strength 128: <|box_end|> <|object_ref_start|> book <|object_ref_end|> <|box_start|> <|coord_624|>

image 470618, crowded object 0:
  baseline: <|object_ref_start|> traffic light <|object_ref_end|> <|box_start|> <|coord_608|>
  strength 64: <|box_end|> <|object_ref_start|> traffic light <|object_ref_end|> <|box_start|>
  strength 128: <|box_end|> <|object_ref_start|> traffic light <|object_ref_end|> <|box_start|>
```

Interpretation:

```text
For this tiny train-only crowded-router smoke, the large boundary direction is
not merely a one-token vocabulary flip. When it induces <|box_end|>, the next
unpatched token follows the expected object-boundary route in 3/3 flipped rows.
This strengthens the router-margin account for crowded closure rows: the model
already has a coherent next-object continuation available, but the natural
layer-24 state has insufficient boundary margin and falls into premature
next-object onset.

Scope remains tiny. The result should be replicated on more crowded train rows
and held-out analogs before promotion, and it should not be generalized to
pre_x1 coordinate-onset failures. The coordinate-onset smoke still indicates a
deeper basin/anchor-selection failure that direct output-embedding target
directions do not repair.
```

Updated next direction:

```text
Keep the next probes population-first. Expand row mining over both train and
val instead of returning to the person/backpack microscope: trained sequences
that fail under teacher-forced GT prefix are especially valuable because they
separate visual exposure from local autoregressive coordinate-basin selection.
For each slot-locus family, compare train failures and unseen-val analogs by
motif, object order, scale, repetition, crowding, coordinate distance, and
tail/termination position.
```

## Probe K: Crowded Router Train/Val Panel Replication

Purpose: replicate Probe J beyond the two-row train smoke, using the complete
crowded closure/router slice from the 48-row population-first causal launch
panel.

Artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_direction_patch_continuation/v3_crowded_router_layer24_boundary_train_val_panel_g3
```

Protocol:

```text
source panel: v1_v13_surface_bridge_population_first/staged_slot_causal_panel_rows.jsonl
staged rows: v1_v13_mechanism_panels/staged_slot_readout_rows.jsonl
selection_locus: crowded_closure_router
splits: train and val
states: 12, expanded to 24 patch rows
patch site: layer_input
layer: 24
direction: box_end_minus_object_ref_boundaries
strengths: 64,128
continuation_steps: 6
errors: 0
```

Overall summary:

```text
baseline_first_token_target_rate: 0/24
patched_first_token_target_rate: 15/24
patched_first_token_changed_rate: 15/24
patched_expected_after_target_evaluable_count: 15
patched_expected_after_target_rate: 15/15
```

Strength split:

```text
strength 64:
  patched_first_token_target_rate: 3/12
  expected route after patched <|box_end|>: 3/3

strength 128:
  patched_first_token_target_rate: 12/12
  expected route after patched <|box_end|>: 12/12
```

Train/val split:

```text
train:
  rows: 12
  patched <|box_end|>: 8/12
  expected route after patched <|box_end|>: 8/8

val:
  rows: 12
  patched <|box_end|>: 7/12
  expected route after patched <|box_end|>: 7/7
```

At strength 128, both train and val are fully steerable:

```text
train strength 128: patched <|box_end|> 6/6, expected next route 6/6
val strength 128:   patched <|box_end|> 6/6, expected next route 6/6
```

Interpretation:

```text
The crowded closure/router result survives a train/val launch-panel
replication. In these selected crowded rows, the model's baseline next token is
never <|box_end|>; it falls into <|object_ref_end|> or <|object_ref_start|>.
A sufficiently large layer-24 boundary direction flips all 12 states to
<|box_end|>, and every flipped state then continues unpatched to the expected
next-object boundary. This is stronger than a one-token logit artifact and
supports a coherent but under-margined boundary-router state.

This still does not prove the mechanism for all crowded rows. The next
expansion should mine more train crowded failures and matched held-out analogs
from the broader row bank, then check whether the same strength threshold and
route coherence hold outside the launch panel.
```

## Probe L: All-Crowded Candidate Strength Curve

Purpose: remove the `max_per_locus_split=6` launch-panel cap and measure a
coarse boundary-direction threshold over every crowded closure/router candidate
available in the staged-slot surface bridge.

Expanded selector artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_causal_panel/v2_v13_surface_bridge_all_candidates_per_locus

candidate rows: 193
crowded_closure_router: 17
  train: 9
  val: 8
```

Continuation artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_direction_patch_continuation/v4_crowded_router_all17_strength_curve_g3
```

Reduction artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_direction_patch_continuation_reducer/v1_v4_crowded_router_all17_strength_curve
```

Protocol:

```text
selection_locus: crowded_closure_router
states: 17
rows: 68
splits: train 9, val 8
patch site: layer_input
layer: 24
direction: box_end_minus_object_ref_boundaries
strengths: 32,64,96,128
continuation_steps: 6
errors: 0
```

Strength curve:

```text
strength 32:
  patched <|box_end|>: 1/17
  expected route after patched <|box_end|>: 1/1

strength 64:
  patched <|box_end|>: 6/17
  expected route after patched <|box_end|>: 6/6

strength 96:
  patched <|box_end|>: 15/17
  expected route after patched <|box_end|>: 15/15

strength 128:
  patched <|box_end|>: 17/17
  expected route after patched <|box_end|>: 17/17
```

Per-case minimum flip strength:

```text
32:  1 case
64:  5 cases
96:  9 cases
128: 2 cases
None: 0 cases
```

Split threshold:

```text
train:
  cases: 9
  min strength 64: 4
  min strength 96: 5
  min strength 128: 0

val:
  cases: 8
  min strength 32: 1
  min strength 64: 1
  min strength 96: 4
  min strength 128: 2
```

Interpretation:

```text
This turns the crowded closure/router claim from selected-launch-panel evidence
into all-current-candidate evidence for the staged-slot mechanism panel. Every
known crowded closure/router candidate is steerable by the same layer-24
boundary direction, and every successful first-token boundary intervention
continues coherently to the next-object route without further patching.

The effect is threshold-like rather than a smooth low-strength correction:
32 is almost inert, 64 repairs a minority, 96 repairs most, and 128 repairs all
17. This looks like an under-margined boundary-router attractor: the next-object
continuation program is already ready, but the residual state must be pushed
across a fairly large box_end-versus-boundary-token margin. The split
difference is case-threshold heterogeneity rather than a clean train/val
mechanism split; train has no 128-only cases, while val has two harder 128-only
cases and one easy 32 case.

Next crowded-router expansion should leave the current staged-slot mechanism
panel and mine more train/val crowded boundary failures from the broader
dataset rows. The main checks are whether the same threshold range persists,
whether any successful <|box_end|> intervention fails route coherence, and
whether hard 128-only rows share a visual or ordering motif.
```
