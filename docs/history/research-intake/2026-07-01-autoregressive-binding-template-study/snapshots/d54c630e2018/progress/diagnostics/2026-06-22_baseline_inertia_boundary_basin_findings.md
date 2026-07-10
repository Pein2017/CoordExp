---
doc_id: progress.diagnostics.baseline_inertia_boundary_basin_findings_2026_06_22
date: 2026-06-22
scope: ckpt928 bbox_len12000 token_embeddings_adapter staged-slot baseline-inertia tomography
status: current
---

# Baseline Inertia Boundary-Basin Findings

## Question

The previous train/val destination-family work made `baseline_inertia` the
dominant x1-onset failure family. This pass asks whether that family is a
coordinate-value failure that can be broken by later-slot guided deltas, or
whether the state is already trapped in a different token-mode basin.

The user also asked to avoid constraining the study to the original
person/backpack microscope. This note therefore treats the current scan as a
cohort-level train/val diagnostic and updates the next directions toward more
training-dataset rows and broader val analogs.

## Tooling Change

The guided-delta runner now supports layer/site sweeps in one deterministic
plan and reports full-vocab token-type basin summaries:

```text
src/analysis/autoregressive_binding_template_ablation/staged_slot_guided_delta_patch.py
tests/analysis/test_staged_slot_guided_delta_patch.py
```

New planner/CLI surfaces:

```text
--layer-indices 0,8,16,20,24,-1
--patch-sites layer_input,self_attn,mlp
```

New summary fields include:

```text
planned_counts_by_configured_layer_index
by_configured_layer_index
by_configured_layer_index_patch_site
baseline_receiver_top1_token_counts
patched_receiver_top1_token_counts
baseline_receiver_top1_class_counts
patched_receiver_top1_class_counts
mean_*_coord_vocab_mass
mean_*_wrapper_token_mass
```

Verification:

```bash
python -m pytest tests/analysis/test_staged_slot_guided_delta_patch.py -q
python -m py_compile src/analysis/autoregressive_binding_template_ablation/staged_slot_guided_delta_patch.py scripts/analysis/run_autoregressive_binding_staged_slot_guided_delta_patch.py
```

Result:

```text
16 passed
py_compile passed
```

## Artifacts

Input destination panel:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_destination_panel_selector/v3_v4_train_val_layerinput_destination_uncapped/staged_slot_destination_panel_rows.jsonl
```

Input staged rows:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_readout_rows/v8_v3_per256_train_val_launch_filter_all_slots/staged_slot_readout_rows.jsonl
```

Dry plan:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_guided_delta_patch/v27_v3_baseline_inertia_layer_site_scan_plan
```

Model-backed sharded scan:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_guided_delta_patch/v28_v27_baseline_inertia_layer_site_scan_sharded3
```

Merged scan:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_guided_delta_patch/v28_v27_baseline_inertia_layer_site_scan_sharded3/merged
```

Coordinate-value geometry reducer:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_coord_value_geometry/v7_v10_baseline_inertia_layer_site_scan_geometry
```

Destination reducer over the geometry output:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_destination_panel_selector/v6_v7_baseline_inertia_layer_site_scan_destination_uncapped
```

Post-hoc token-mode atlases from existing readout rows:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_readout_reducer/v3_v28_per256_train_val_token_mode_atlas
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_readout_reducer/v4_v23_destination_family_token_mode_atlas
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_readout_reducer/v5_v20_per128_train_val_token_mode_atlas
```

## Scope And Caveat

The scan planned:

```text
49 unique baseline-inertia receiver states
1029 plan rows
7 configured layer indices: 0, 8, 16, 20, 24, 28, -1
3 patch sites: layer_input, self_attn, mlp
1 delta mode: donor_minus_previous_slot
1 donor position: staged_after_x1_y1
1 scale: 1.0
```

The model-backed result:

```text
882 patch rows
147 errors
```

All sampled errors were the same configuration mistake:

```text
ValueError: decoder layer index 28 out of range for 28 layers
```

Interpret the successful layer scan only for explicit layers
`0,8,16,20,24` plus `-1`. The `-1` index already resolves to the final
decoder layer.

## Main Result

The later-slot guided delta does not break `baseline_inertia`.

Merged guided-delta summary:

```text
target_rank_improved_rate: 0.5589569160997733
target_top1_rate: 0.0
coord_rank_lte_10_rate: 0.0
coord_distance_lte_16_rate: 0.0011337868480725624
donor_target_top1_rate: 0.0
donor_nearer_than_receiver_rate: 1.0
slot_intrusion_rate: 1.0
```

This means target rank can move, but the state almost never enters a correct
or near-correct coordinate basin.

The geometry reducer sharpens this:

```text
input_row_count: 882
invalid_geometry_row_count: 0
label_counts: {'intrusive_transition': 882}
patched_landing_region_counts:
  baseline_near: 866
  other: 14
  donor_slot_near: 1
  receiver_and_donor_near: 1
```

So the patched top-1 coordinate usually stays near the original baseline
basin, not near the receiver target and not usually near the donor slot either.

## Boundary-Basin Discovery

The surprising result is visible only when the full vocabulary surface is
summarized, not only the coordinate-restricted metrics.

Patch-row-level top1 token classes:

```text
baseline_receiver_top1_class_counts: {'coord': 54, 'wrapper': 828}
patched_receiver_top1_class_counts: {'coord': 54, 'text': 1, 'wrapper': 827}
baseline_receiver_top1_token_counts:
  <|box_end|>: 792
  <|object_ref_start|>: 18
  <|object_ref_end|>: 18
  <|coord_611|>: 18
  <|coord_841|>: 18
  <|coord_978|>: 18
patched_receiver_top1_token_counts:
  <|box_end|>: 773
  <|object_ref_end|>: 26
  <|object_ref_start|>: 22
  <|im_end|>: 6
  coord-token escapes: sparse
```

Unique-state denominator, before any patch:

```text
all states:   49
wrapper top1: 46/49
box_end top1: 44/49

train states:   24
wrapper top1:   24/24
box_end top1:   24/24

val states:     25
wrapper top1:   22/25
box_end top1:   20/25
coord top1:      3/25
```

Mean token-group mass before patch:

```text
split  coord_vocab_mass       wrapper_token_mass
train  4.8393822496e-11       0.9998385559
val    0.1199696893           0.8765801954
all    0.0612090252           0.9369516373
```

After patching, train rows remain wrapper-dominated:

```text
train patched wrapper top1: 432/432 patch rows
train patched box_end top1: 428/432 patch rows
train patched coord_vocab_mass mean: 6.2534705557e-08
```

Val rows are less absolute but still mostly wrapper-dominated:

```text
val patched wrapper top1: 395/450 patch rows
val patched box_end top1: 345/450 patch rows
val patched coord top1: 54/450 patch rows
val patched coord_vocab_mass mean: 0.1199667775
```

## Interpretation

`baseline_inertia` should no longer be described only as a coordinate-value
basin that resists donor-slot transition deltas. In this cohort, it often
means the model is not in coordinate-emission mode at all. The full-vocab top
token is usually a wrapper or boundary token, especially `<|box_end|>`.

This matters because the coordinate-restricted surface can still show a
wrong coordinate basin, but the causal bottleneck may be earlier or more
global: a schema/boundary gate deciding "close the box/object" versus "enter
the next coordinate slot." The slot-transition delta can improve target rank
inside the coordinate vocabulary without moving enough probability mass out of
the wrapper-token basin.

## Broad Atlas Correction

The token-mode atlas prevents overgeneralizing this result to every pre-x1
failure. On the broad per256 robust-position readout, ordinary `pre_x1` rows
are already in coordinate mode:

```text
artifact:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_readout_reducer/v3_v28_per256_train_val_token_mode_atlas

position/split        n     coord_mass  wrapper_mass  coord_top1  rank<=10  dist<=16
pre_x1 train          1536  0.9978      0.0           1.0         0.3151    0.4473
pre_x1 val            1536  0.9980      0.0           1.0         0.2617    0.4486
post_x1 train         1536  0.9999      0.0           1.0         0.6400    0.8359
post_x1 val           1536  0.9999      0.0           1.0         0.5794    0.8216
box_close train       1536  0.0         0.9798        0.0         n/a       n/a
box_close val         1536  0.0         0.9524        0.0         n/a       n/a
next_object train     1536  0.0         0.7133        0.0         n/a       n/a
next_object val       1536  0.0         0.7148        0.0         n/a       n/a
```

So the general pre-x1 problem is often a coordinate-mode wrong-basin or
low-rank problem, not a wrapper-mode problem.

The destination-family staged atlas shows that `baseline_inertia` is the
exceptional boundary-mode subset:

```text
artifact:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_readout_reducer/v4_v23_destination_family_token_mode_atlas

staged_pre_x1 family       n   coord_mass  wrapper_mass  coord_top1  box_end_top1  dist<=16
baseline_inertia           61  0.0164      0.9828        0.0164      0.9508        0.0
control_slot_pull          35  0.9714      0.0286        0.9714      0.0286        0.0
donor_slot_capture         39  0.8185      0.1660        0.8205      0.1026        0.0
coordinate_edge_ambiguity  25  0.9968      0.0           1.0         0.0           0.12
off_basin_escape           46  0.9978      0.0           1.0         0.0           0.0
receiver_repair             7  0.9974      0.0           1.0         0.0           0.1429
```

Split-specific `baseline_inertia` remains wrapper-mode in both splits:

```text
train baseline_inertia: n=30, coord_mass=0.0,    wrapper_mass=0.9994, box_end_top1=0.9667
val   baseline_inertia: n=31, coord_mass=0.0322, wrapper_mass=0.9667, box_end_top1=0.9355
```

After x1 is supplied, even `baseline_inertia` mostly re-enters coordinate mode:

```text
staged_after_x1 baseline_inertia:
  coord_mass=0.9939
  wrapper_mass=0.0046
  coord_top1=0.9836
  coord_distance_lte16=0.1311
```

But after the full box, `baseline_inertia` still has a weak boundary state:

```text
staged_after_full_box baseline_inertia:
  wrapper_top1=0.9672
  box_end_top1=0.1148
  object_ref_boundary_top1=0.5738
```

This says `baseline_inertia` is a broad schema/cursor pathology, not just an
x1 value pathology: it prematurely prefers box closure at pre_x1, then often
prefers object boundary tokens over `<|box_end|>` after the full box.

The train/val split is also important. The trained rows are not easier here.
They are more strongly over-closed:

```text
train: coord_vocab_mass ~0, wrapper_token_mass ~1, all top1 <|box_end|>
val:   some coordinate mass survives, but wrapper still dominates
```

This is evidence against a simple unseen-val or visual-nonperception account.
The failure appears on trained sequences under teacher-forced staged prefixes.
The more plausible current account is local autoregressive cursor/gating
fragility: the model can have learned visual coordinate locality, yet the
current prefix state can still route to the wrong token mode at the x1 onset.

## Revised Next Directions

1. Make `baseline_inertia` the primary microscope, but split the whole
   pre-x1 population into:

```text
boundary/wrapper inertia:
  full-vocab top1 is <|box_end|>, <|object_ref_end|>, or another wrapper token

coordinate inertia:
  full-vocab top1 is a coordinate token, but it is near the baseline/wrong basin
```

2. Use the token-mode atlas as a standard preflight before more GPU-heavy
causal scans. The minimum atlas should summarize, by split, regime, position,
and destination family:

```text
top1 token class
top1 token text
coord-vocab mass
wrapper-token mass
coord rank and distance
box_end margin
object_ref_start/end margin
```

3. For trained rows, over-sample failures under teacher-forced GT prefixes.
Do not wait for rollout labels. A trained sequence that still routes to
`<|box_end|>` at pre_x1 is a high-value mechanism case because it separates
dataset exposure from local autoregressive state selection.

4. For val rows, keep two roles separate:

```text
matched held-out analog:
  comparable motif/control for trained failures

rollout-attached false negative:
  unseen free-decode stress slice, asymmetric because train rollout labels are absent
```

5. Add explicit boundary/gating interventions before more donor-state deltas:

```text
box_end logit suppression or margin clamp
coordinate-vocab mass promotion under fixed hidden state
object_ref_end/object_ref_start suppression at pre_x1
layer-wise hidden/logit-lens trace of coord-mass versus wrapper-mass
attention/value localization for the boundary gate
```

6. Keep `donor_slot_capture` as the value-copy contrast and `control_slot_pull`
as the previous-slot anchoring contrast. They are still useful, but this scan
says the largest family first needs token-mode gating diagnosis.

## Current Verdict

The strongest new mechanism hypothesis is:

```text
The pre-x1 failure population has at least two separable surfaces:

1. General coordinate-mode wrong-basin failures, where the model is already
   emitting coordinate tokens but rank/locality is poor.
2. Baseline-inertia boundary-mode failures, where the model has not entered
   coordinate-emission mode and instead strongly prefers wrapper tokens,
   especially <|box_end|>.

The second surface is especially strong on trained baseline-inertia rows.
Later-slot coordinate deltas can move coordinate ranks, but cannot reliably
open the coordinate slot.
```

This makes the next core question sharper:

```text
Where is the coordinate-mode versus boundary-mode gate represented, and why
does a teacher-forced trained prefix sometimes select the boundary mode at
the next x1 slot?
```
