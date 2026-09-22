# Continuation Taxonomy Batch And Contrast Findings

Date: 2026-06-22

Scope:

```text
readout-only post-hoc taxonomy over existing hidden causal patch continuation artifacts
source roots: 20 trajectory_hidden_causal_activation_patch_continuation_probe artifacts
batch rows: 1258
paired contrast rows: 1162 non-baseline rows compared to matching baseline_no_patch rows
cases: 2 post_box_boundary_case_id values
```

Batch taxonomy artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/continuation_taxonomy_batch/hidden_patch_continuations_after_29f50827_v1
```

Paired contrast artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/continuation_taxonomy_contrast/hidden_patch_continuations_after_29f50827_v1
```

Commands:

```bash
python scripts/analysis/run_autoregressive_binding_continuation_taxonomy_batch.py \
  --continuation-rows \
  /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_continuation_probe/*/trajectory_hidden_causal_activation_patch_continuation_rows.jsonl \
  --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/continuation_taxonomy_batch/hidden_patch_continuations_after_29f50827_v1

python scripts/analysis/run_autoregressive_binding_continuation_taxonomy_contrast.py \
  --taxonomy-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/continuation_taxonomy_batch/hidden_patch_continuations_after_29f50827_v1/continuation_taxonomy_batch_rows.jsonl \
  --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/continuation_taxonomy_contrast/hidden_patch_continuations_after_29f50827_v1
```

## Why This Pass Was Needed

The previous pre-x1 continuation note showed that the raw bridge
`generated_*` fields can describe the following object when the prefix already
ends at `<|box_start|>`. The batch pass extends that correction across all
existing hidden-patch continuation artifacts and adds paired baseline contrast
so raw target-overlap counts are not mistaken for causal repair.

The contrast key is intentionally conservative:

```text
continuation_taxonomy_source_artifact
state_key
continuation_application_mode
```

Each non-baseline row is compared against the matching `baseline_no_patch` row
for the same source artifact, state, and application mode.

## Batch Counts

Aggregate taxonomy:

```text
row_count=1258
primary_span_source_counts={
  generated_object_fields: 1158,
  ongoing_box_from_prefix: 100
}
taxonomy_label_counts={
  descriptor_only: 30,
  descriptor_valid_box: 473,
  first_token_only: 137,
  malformed_or_incomplete: 95,
  target_overlap: 116,
  wrong_desc_valid: 375,
  wrong_same_desc_basin: 32
}
generated_object_taxonomy_label_counts={
  descriptor_valid_box: 421,
  malformed_or_incomplete: 262,
  target_overlap: 116,
  wrong_desc_valid: 425,
  wrong_same_desc_basin: 34
}
```

Paired contrast:

```text
row_count=1162
contrast_outcome_counts={
  primary_changed_non_target: 539,
  primary_target_gain: 17,
  primary_target_loss: 61,
  primary_target_preserved: 83,
  primary_unchanged_non_target: 462
}
next_object_contrast_outcome_counts={
  next_changed_non_target: 543,
  next_target_gain: 17,
  next_target_loss: 61,
  next_target_preserved: 83,
  next_unchanged_non_target: 458
}
```

By patch family:

```text
full_output_delta:
  primary_target_gain=2
  primary_target_loss=16
  primary_target_preserved=16

layer_input_delta:
  primary_target_gain=2
  primary_target_loss=0
  primary_target_preserved=32

interpolation:
  primary_target_gain=13
  primary_target_loss=37
  primary_target_preserved=27

self_noop:
  primary_target_gain=0
  primary_target_loss=8
  primary_target_preserved=8
```

## Open-Box Versus Post-Box Regimes

The important split is by `continuation_primary_span_source`.

For `ongoing_box_from_prefix` rows, which are the pre-x1 open-box continuations:

```text
full_output_delta, first_step:
  primary_target_gain=2
  primary_unchanged_non_target=3

layer_input_delta:
  primary_target_gain=0
  primary_unchanged_non_target=10

self_attn_delta:
  primary_unchanged_non_target=10

mlp_delta:
  primary_unchanged_non_target=10

interpolation, all_steps:
  primary_changed_non_target=20

full_output_delta, all_steps:
  primary_changed_non_target=5
```

For the same open-box rows, next-object contrast shows the complementary effect:

```text
layer_input_delta, all_steps:
  next_target_gain=2
  next_changed_non_target=2
  next_unchanged_non_target=1

full_output_delta, first_step:
  next_unchanged_non_target=5
```

This repeats the single-artifact observation after adding the v1/v2 contrast:

```text
full_output_delta first-step can repair the current open coordinate box
layer_input_delta all-steps can redirect the following object route
```

They are not the same mechanism.

## Causal Read

1. Current-coordinate repair is localized to the full output at the immediate
   coordinate decision in this evidence. The only open-box `primary_target_gain`
   rows are the `full_output_delta` first-step rows for the backpack
   nearest-previous-same-description variant. They move the current open box
   from zero target overlap to IoU `0.408577`.

2. Layer-input is not a current-box repair vector in these open-box rows. It
   has zero open-box primary target gains. Its positive signal appears as
   next-object routing under all-steps patching: the current box remains a wrong
   valid basin, while the later generated object can become a target-overlap
   backpack.

3. The post-box/generated-object artifacts are dominated by route and syntax
   effects, not clean box repair. Some layer-input and interpolation rows gain
   target overlap after baseline comparison, but many gains have very small IoU
   such as `0.0044`, `0.0150`, or `0.0212`. Those are route/basin contact
   signals, not full object repair.

4. Persistent intervention is hazardous. `all_steps` interpolation and
   full-output deltas frequently produce target losses or malformed repetition
   tails. The `self_noop` all-steps losses in the snowboard case also warn that
   the continuation machinery itself can perturb fragile trajectories; baseline
   contrast is mandatory for this family of experiments.

5. Isolated late self-attention and MLP component deltas remain weak on this
   continuation taxonomy. In the open-box slice they are entirely
   `primary_unchanged_non_target`; in the broader generated-object slice they
   do not produce target gains in the available artifacts.

## Revised Next Directions

Do not build the mixed full-output/layer-input scheduler yet. It is likely
useful, but the second code review showed that the current generation hook
cannot express a semantic switch such as "full-output during current coordinate
tokens, layer-input after `<|box_end|>`" without a new stateful token-role
scheduler. That scheduler is subtle because hooks fire before next-token
prediction, and `first_step` currently means prefill-last-token patching, not a
semantic first decoded token.

Next high-value implementation route:

1. Build a formation-time layer-input map for the open-box backpack case, but
   ask the sharper question exposed here:

```text
When does the late layer-input carrier become a next-row routing state, and why
does it fail to act as a current-coordinate repair state?
```

2. Use the formation map positions:

```text
descriptor onset
descriptor end
object_ref_end
box_start
pre-x1
post-x1
box close
next-object onset
```

3. Keep the outcome taxonomy span-aware. Every continuation result should report
   both:

```text
current/open-box label
next/generated-object label
```

4. Promote the mixed scheduler only after the formation map shows a clear
   temporal handoff. The scheduler should be tested with a fake decode loop
   before GPU use and should record per-step patch family/counts.

5. Treat "target overlap" as a weak basin-contact signal unless IoU magnitude
   is reported. A future reducer should separate tiny-overlap contact from
   material repair, for example IoU bands `>0`, `>=0.1`, `>=0.5`.
