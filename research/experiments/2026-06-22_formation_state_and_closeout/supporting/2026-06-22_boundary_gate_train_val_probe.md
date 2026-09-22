---
doc_id: progress.diagnostics.boundary_gate_train_val_probe_2026_06_22
date: 2026-06-22
scope: ckpt928 bbox_len12000 token_embeddings_adapter train-val staged-slot boundary-gate probe
status: current
---

# Boundary-Gate Train/Val Probe

## Question

The previous destination-family analysis made `baseline_inertia` look like a
dominant `pre_x1` failure family. The user asked to broaden beyond the
person/backpack microscope and to compare trained rows against unseen val rows.

This pass asks a narrower causal question:

```text
Can a completed-box/wrapper-mode baseline-inertia state be pushed back into
coordinate mode by output-embedding directions at layer 24?
```

## Tooling Change

`staged_slot_direction_patch.py` now supports:

```text
--slot-destination-families
--patch-row-offset
coord_mean_minus_box_end
coord_target_minus_box_end
coord_mean_minus_wrapper_mean
```

It also reports token-mode summaries by destination family, split, direction,
and strength:

```text
mean_*_coord_vocab_mass
mean_*_wrapper_token_mass
baseline_top1_class_counts
patched_top1_class_counts
by_slot_destination_family_direction
by_slot_destination_family_strength
```

Verification:

```bash
python -m pytest tests/analysis/test_staged_slot_direction_patch.py -q
python -m py_compile src/analysis/autoregressive_binding_template_ablation/staged_slot_direction_patch.py scripts/analysis/run_autoregressive_binding_staged_slot_direction_patch.py
```

Result:

```text
8 passed
py_compile passed
```

## Artifacts

Readout-only plan:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_direction_patch/v30_boundary_gate_train_val_readout_plan
```

Model-backed layer-24 shards:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_direction_patch/v31_boundary_gate_train_val_layer24_shard0_g0
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_direction_patch/v31_boundary_gate_train_val_layer24_shard1_g2
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_direction_patch/v31_boundary_gate_train_val_layer24_shard2_g3
```

Merged artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_direction_patch/v31_boundary_gate_train_val_layer24_merged
```

Reference token-mode atlases:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_readout_reducer/v3_v28_per256_train_val_token_mode_atlas
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_readout_reducer/v4_v23_destination_family_token_mode_atlas
```

## Scope

The model-backed direction patch covered:

```text
102 unique destination-selected states
1836 patch rows
3 destination families: baseline_inertia, off_basin_escape, donor_slot_capture
2 splits: train, val
3 directions
6 strengths: 1, 2, 4, 8, 16, 32
layer 24
patch_site: layer_input
error_count: 0
```

## Main Result

The layer-24 output-embedding directions did not open the
`baseline_inertia` boundary basin.

Unique-state outcome across all directions and strengths:

```text
baseline_inertia train:
  states: 24
  any patched coord top1: 0
  any patched coord rank <=10: 0
  any patched coord top1 distance <=16: 0

baseline_inertia val:
  states: 25
  any patched coord top1: 3
  any patched coord rank <=10: 0
  any patched coord top1 distance <=16: 0
```

The 3 val coord-top1 states were already coord-top1 before patching. No state
was newly opened into a useful coordinate basin.

At the strongest tested strength (`32`), the patch did move target logits, but
not enough to change token mode:

```text
baseline_inertia train, strength 32:
  coord_target_minus_box_end mean target logit delta: +2.1745
  mean target rank delta: -1984.25
  mean coord mass delta: +7.65e-10
  mean wrapper mass delta: -7.22e-05
  patched coord mass: about 8.13e-10
  patched wrapper mass: about 0.9998

baseline_inertia val, strength 32:
  coord_target_minus_box_end mean target logit delta: +1.9625
  mean target rank delta: -2626.44
  mean coord mass delta: +8.79e-07
  mean wrapper mass delta: -6.43e-04
  patched coord mass: about 0.12
  patched wrapper mass: about 0.8759
```

So this direction is a weak logit nudge, not a basin switch.

## Contract Correction

The key correction is about row semantics.

For representative `baseline_inertia` rows, the `assistant_prefix_text` already
contains the target object's four coordinate tokens and is missing only
`<|box_end|>`. Example tail:

```text
...<|object_ref_start|>suitcase<|object_ref_end|><|box_start|><|coord_819|><|coord_862|><|coord_907|><|coord_912|>
```

The row still carries the receiver x1 token as `target_next_token_text`:

```text
target_next_token_text: <|coord_819|>
formation_suffix_text: <|box_end|>
```

Therefore `<|box_end|>` is locally valid for the realized prefix. The
`baseline_inertia` family should be interpreted as a completed-box/wrapper
routing mode in a staged counterfactual, not as a clean pre-x1 onset state
where the model is simply failing to perceive x1.

This correction matters: the boundary basin is real, but it is a different
mechanistic object from clean x1-onset coordinate selection.

## Atlas Contrast

Broad ordinary `pre_x1` rows from the per256 train/val token-mode atlas are
coordinate-mode, not wrapper-mode:

```text
pre_x1 train:
  rows: 1536
  coord top1 class rate: 1.0
  coord mass: 0.997815
  wrapper mass: 1.77e-09
  coord rank <=10: 0.3151
  coord top1 distance <=16: 0.4473

pre_x1 val:
  rows: 1536
  coord top1 class rate: 1.0
  coord mass: 0.997966
  wrapper mass: 1.32e-09
  coord rank <=10: 0.2617
  coord top1 distance <=16: 0.4486
```

Post-x1 rows are much better in both splits:

```text
post_x1 train:
  coord rank <=10: 0.6400
  coord top1 distance <=16: 0.8359

post_x1 val:
  coord rank <=10: 0.5794
  coord top1 distance <=16: 0.8216
```

Destination-selected staged rows are intentionally harder and mixed:

```text
staged_pre_x1 train:
  rows: 85
  coord top1 class rate: 0.6118
  wrapper top1 class rate: 0.3882
  box_end top1 token rate: 0.3647

staged_pre_x1 val:
  rows: 128
  coord top1 class rate: 0.7266
  wrapper top1 class rate: 0.2734
  box_end top1 token rate: 0.2500

staged_pre_x1 baseline_inertia train:
  rows: 30
  coord top1 class rate: 0.0
  wrapper top1 class rate: 1.0
  box_end top1 token rate: 0.9667

staged_pre_x1 baseline_inertia val:
  rows: 31
  coord top1 class rate: 0.0323
  wrapper top1 class rate: 0.9677
  box_end top1 token rate: 0.9355
```

## Interpretation

This pass weakens an over-simple boundary-gate story:

```text
wrong: baseline_inertia is clean pre_x1 visual failure
better: baseline_inertia is a completed-box/wrapper routing basin that carries
        a receiver-x1 comparison label for staged counterfactual analysis
```

The train/val comparison also remains important:

```text
ordinary clean pre_x1:
  train and val are both coordinate-mode
  train is slightly better by rank<=10

destination-selected staged pre_x1:
  train and val are both fragile
  train baseline_inertia is even more locked into wrapper mode
```

The deepest next question is therefore not whether the model can emit a coord
token under strict schema. It can. The question is which local state decides
whether the next coordinate is the target object's x1, a donor/nearby object's
coordinate, an origin/edge basin, or a completed-box routing token.

## Next Directions

1. Keep `baseline_inertia` as a completed-box boundary/router probe, not as the
   main clean-onset probe.
2. Build or select a clean pre-x1 train-vs-val panel from ordinary scored
   readout rows where the prefix truly ends before x1.
3. For that clean-onset panel, compare target/donor/origin basin ownership at
   the known transition layers 17-21, not only layer 24.
4. For completed-box/router states, probe wrapper-token competition directly:
   `<|box_end|>` versus `<|object_ref_start|>`, `<|object_ref_end|>`,
   `<|im_end|>`, and termination/next-object context.
5. For train data specifically, over-sample rows that fail under clean
   teacher-forced prefixes despite being trained sequences; use these to
   separate visual exposure from local autoregressive basin selection.
