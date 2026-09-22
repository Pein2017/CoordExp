---
doc_id: progress.diagnostics.staged_slot_guided_delta_patch_findings_2026_06_22
layer: progress
doc_type: diagnostic-implementation-note
status: active-branch-evidence
evidence_scope: gpu-probe-staged-slot-guided-delta
domain: autoregressive-binding-template-ablation
updated: 2026-06-22
branch: codex/autoregressive-binding-template-study
---

# Staged-Slot Guided Delta Patch Findings

## Purpose

This note records the first population-scale guided-delta probe for pre-x1
coordinate-onset failures in checkpoint-928. The question is whether the model
can be moved from a failed pre-x1 coordinate basin into the correct object
basin by adding hidden-state information from the same object later in the
teacher-forced span.

The probe deliberately avoids treating the older person/backpack case as the
experiment distribution. It uses the staged-slot mechanism panel built from
trained-sequence failures under teacher-forced GT prefixes and matched held-out
val analogs.

## Implementation

Added:

```text
src/analysis/autoregressive_binding_template_ablation/staged_slot_guided_delta_patch.py
scripts/analysis/run_autoregressive_binding_staged_slot_guided_delta_patch.py
scripts/analysis/run_autoregressive_binding_staged_slot_guided_delta_reduce.py
tests/analysis/test_staged_slot_guided_delta_patch.py
```

The runner builds receiver rows from `staged_pre_x1` coordinate states and
same-panel donor rows from later guided slots:

```text
staged_after_x1
staged_after_x1_y1
staged_after_x1_y1_x2
```

For each planned row it captures the layer-input hidden state at the receiver
and donor prediction token, adds:

```text
patch_scale * (donor_hidden - receiver_hidden)
```

to the receiver state, and scores both the receiver x1 target and the donor
slot target on the patched receiver logits.

The reducer now distinguishes exact donor-token intrusion from broader
donor-basin pull:

```text
donor_target_top1_rate
donor_nearer_than_receiver_rate
slot_intrusion_rate
```

`slot_intrusion_rate` is true when the patched top1 is the donor target or when
the patched coordinate top1 is nearer to the donor slot target than to the
receiver x1 target.

The CLI supports sharded execution through:

```text
--patch-row-offset
--max-patch-rows
```

and sharded reductions through:

```text
scripts/analysis/run_autoregressive_binding_staged_slot_guided_delta_reduce.py
```

Verification:

```text
python -m pytest tests/analysis/test_staged_slot_guided_delta_patch.py -q
7 passed
```

## Tiny Matched Smoke

Train smoke:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_guided_delta_patch/v5_prex1_train2_layer24_afterx1_afterx1y1_scales_slotmetric_g3
```

Val smoke:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_guided_delta_patch/v6_prex1_val2_layer24_afterx1_afterx1y1_scales_slotmetric_g3
```

Both ran on GPU 3 with:

```text
layer_index: 24
donor_positions: staged_after_x1, staged_after_x1_y1
patch_scales: 0.5, 1.0
rows: 8 per split
error_count: 0
```

Tiny train result:

```text
target_rank_improved_rate: 1.000
coord_distance_lte_16_rate: 0.375
donor_nearer_than_receiver_rate: 0.875
slot_intrusion_rate: 0.875
target_top1_rate: 0.000
```

Tiny val result:

```text
target_rank_improved_rate: 0.500
coord_distance_lte_16_rate: 0.000
donor_nearer_than_receiver_rate: 1.000
slot_intrusion_rate: 1.000
target_top1_rate: 0.000
```

Read: later-slot donor states strongly move the coordinate distribution, but
the movement is often into the donor slot basin rather than the intended x1
basin. The old exact-donor-top1 metric undercounted this because the patched
top1 often lands within a few bins of the donor slot rather than exactly on the
donor token.

## Broad Train/Val Guided-Delta Run

Plan-only artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_guided_delta_patch/v7_prex1_all_candidates_afterx1_afterx1y1_scales_plan
```

Broad plan:

```text
receiver states: 88
patch rows: 352
donor_positions: staged_after_x1, staged_after_x1_y1
patch_scales: 0.5, 1.0
```

Four GPU shards:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_guided_delta_patch/v8_prex1_all_candidates_afterx1_afterx1y1_scales_shard0_g3
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_guided_delta_patch/v8_prex1_all_candidates_afterx1_afterx1y1_scales_shard1_g0
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_guided_delta_patch/v8_prex1_all_candidates_afterx1_afterx1y1_scales_shard2_g7
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_guided_delta_patch/v8_prex1_all_candidates_afterx1_afterx1y1_scales_shard3_g1
```

Merged artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_guided_delta_patch/v9_prex1_all_candidates_afterx1_afterx1y1_scales_merged
```

Merged summary:

```text
patch_row_count: 352
patched_state_count: 88
error_count: 0
target_rank_improved_rate: 0.5966
target_top1_rate: 0.0057
coord_rank_lte_10_rate: 0.0142
coord_distance_lte_16_rate: 0.2074
donor_target_top1_rate: 0.0653
donor_nearer_than_receiver_rate: 0.9063
slot_intrusion_rate: 0.9063
mean_receiver_target_rank_delta: -41.9432
mean_receiver_coord_distance_delta: -138.3409
```

Split comparison over patch rows:

```text
train:
  rows: 180
  target_rank_improved_rate: 0.5944
  coord_distance_lte_16_rate: 0.1889
  donor_nearer_than_receiver_rate: 0.9111
  target_top1_rate: 0.0000

val:
  rows: 172
  target_rank_improved_rate: 0.5988
  coord_distance_lte_16_rate: 0.2267
  donor_nearer_than_receiver_rate: 0.9012
  target_top1_rate: 0.0116
```

Regime comparison over patch rows:

```text
small_object:
  rows: 120
  target_rank_improved_rate: 0.7083
  coord_distance_lte_16_rate: 0.4417
  donor_nearer_than_receiver_rate: 0.9500

repeated_class:
  rows: 128
  target_rank_improved_rate: 0.6250
  coord_distance_lte_16_rate: 0.0938
  donor_nearer_than_receiver_rate: 0.8906

crowded:
  rows: 88
  target_rank_improved_rate: 0.4432
  coord_distance_lte_16_rate: 0.0795
  donor_nearer_than_receiver_rate: 0.8977

termination_tail:
  rows: 12
  target_rank_improved_rate: 0.3333
  coord_distance_lte_16_rate: 0.0833
  donor_nearer_than_receiver_rate: 0.8333
```

State-level post-hoc read over 88 states:

```text
any_rank_improved: 74/88
any_near_receiver_d16: 46/88
any_target_top1: 2/88
any_donor_nearer: 88/88
all_donor_nearer: 63/88
any_exact_donor_top1: 15/88
```

Train/val state-level comparison:

```text
train:
  states: 45
  any_near_receiver_d16: 23/45
  any_target_top1: 0/45
  any_donor_nearer: 45/45
  all_donor_nearer: 33/45

val:
  states: 43
  any_near_receiver_d16: 23/43
  any_target_top1: 2/43
  any_donor_nearer: 43/43
  all_donor_nearer: 30/43
```

State-level regime read:

```text
small_object:
  states: 30
  any_near_receiver_d16: 30/30
  any_target_top1: 2/30
  any_donor_nearer: 30/30
  all_donor_nearer: 25/30

repeated_class:
  states: 32
  any_near_receiver_d16: 9/32
  any_target_top1: 0/32
  any_donor_nearer: 32/32
  all_donor_nearer: 21/32

crowded:
  states: 22
  any_near_receiver_d16: 6/22
  any_target_top1: 0/22
  any_donor_nearer: 22/22
  all_donor_nearer: 16/22
```

## Mechanistic Read

The broad run makes the earlier tiny observation real: guided later-slot hidden
deltas are strong coordinate attractors, but they are not clean identity repair
vectors for pre-x1.

The most important distinction is:

```text
the patch often improves receiver target rank
but the patched top1 is usually closer to the donor slot than to the receiver x1
```

This is true for both trained rows and held-out rows. The train/val comparison
does not support a simple "trained sequences are memorized, val sequences are
unseen and therefore fail differently" explanation. The failure appears to be a
shared slot/phase/basin mechanism under teacher-forced evidence.

Small objects are special but not solved. Every small-object state has at least
one patch landing within 16 bins of the receiver x1, yet 25/30 are donor-nearer
under every patch variant. Many small-object boxes have nearby x1/y1/x2
coordinates, so a later-slot donor delta can look like local repair while still
being a slot-phase substitution.

Crowded and repeated-class rows are less locally repaired and remain dominated
by donor-basin pull. This keeps crowded closure/router and coordinate-onset as
separate mechanisms: boundary direction patches can make crowded rows route
correctly after `<|box_end|>`, while guided coordinate deltas mostly show
pre-x1 basin-phase fragility.

## Revised Next Directions

1. **Slot-phase disentanglement.** The next coordinate-onset probe should split
   donor-hidden information into object identity, coordinate value, and slot
   phase. Candidate interventions: donor minus same-slot control, later-slot
   donor minus adjacent-slot donor, and receiver plus donor value after
   projecting out a learned slot-position direction.

2. **Train-failure versus val-analog expansion.** Continue mining training
   rows that fail under teacher-forced GT prefixes and match val analogs by
   regime, object order, object area, slot locus, and remaining-object tail.
   The 88-state result is broad enough to demote the single-pair story, but it
   is still a mechanism-panel subset, not a full dataset characterization.

3. **Small-object smoothness/locality split.** For small objects, separate
   geometry locality from slot substitution. Use pairs where x1/y1/x2 are
   close and pairs where they are far. A true receiver-basin repair should land
   near x1 even when the donor slot is not adjacent.

4. **Positive-case microscopes without promotion bias.** The two exact target
   top1 states, both val small-object cases, are useful microscopes for
   component localization. They should not become the sampling prior. Pair them
   with hard negative train and val states where all patches are donor-nearer.

5. **Attention/value path localization.** Guided deltas are too coarse. The
   next high-value probe is to localize which attention or MLP component writes
   the later-slot coordinate attractor, then test whether that component can be
   redirected toward the receiver slot without importing donor slot phase.

6. **Training remains premature.** A limited micro-training step may be useful
   later, but the current evidence says not to optimize toward generic
   donor-state matching. First identify a state-conditioned lever that improves
   receiver x1 without increasing donor-nearer attraction.

## Subagent Review Synthesis

Two read-only subagent reviews were run after the broad guided-delta artifact.
They converged on the same correction:

```text
stop searching for another undifferentiated rescue vector
identify the object-state variables of the autoregressive object tracker
expand rows population-first and failure-locus-first
```

Concrete accepted adjustments:

- Demote more descriptor-only transport, single wrong-bin coordinate vectors,
  and isolated late component hunts unless they are tied to a new object-state
  variable.
- Treat trained-sequence failures under teacher-forced GT prefixes as a primary
  selector, not a nuisance. The next expansion should start train-first, then
  match held-out analogs.
- Build a broader per96 candidate bank with image caps and descriptor/class
  caps before the next promotion step. Do not let `person` or `bird` dominate
  the launch panel.
- Skip descriptor aggregate claims until multi-token descriptor scoring is
  fixed. Use coordinate/schema positions first: `pre_x1`, `post_x1`,
  `box_close`, and `next_object_onset`.
- Split the next GPU work into separate launch panels for pre-x1 onset and
  crowded closure/router. Small-object extent/final-y2 should be the next
  coordinate-smoothness batch unless the broader readout exposes a stronger
  train/val divergence.
- Add same-image/same-description ownership controls, because donor capture
  versus true repair remains under-separated.

The most valuable new paradigms to preserve in the next round are:

```text
visual candidate ledger
latent object-state automaton
coordinate basin energy landscape
same-image ownership swap
guided future-state backtracking
coverage ledger / object-set memory
router margin hazard model
small-object coordinate smoothness
readout-to-behavior causal calibration
```

Immediate next executable scaffold:

```bash
python scripts/analysis/run_autoregressive_binding_train_val_candidate_bank.py \
  --train-jsonl /data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/train.coord.jsonl \
  --val-jsonl /data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/val.coord.jsonl \
  --per-regime-per-split 96 \
  --max-per-image-per-regime 2 \
  --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/train_val_candidate_bank/v6_bbox_len12000_gt_structure_regimes_per96_imgcap2
```

Then materialize formation rows, run narrow sharded readout only on robust
coordinate/schema positions, and choose separate pre-x1 and crowded-router
launch panels with explicit train/val/locus counts.
