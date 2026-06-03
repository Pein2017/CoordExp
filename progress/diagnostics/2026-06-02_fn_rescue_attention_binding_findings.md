---
title: FN-Rescue Attention And Instance-Binding Findings
date: 2026-06-02
status: active-reference
owner: codex
depends_on:
  - outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_continuation/summary.json
  - outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_continuation/merge_summary.json
  - outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_continuation/report.md
  - outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_continuation/rescue_generation_rows.jsonl
  - outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_continuation/rescue_attention_region_rows.jsonl
  - outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_continuation/gallery/gallery_summary.json
  - outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_desc_x1_phase2/attention_mining/summary.json
  - outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_desc_x1_phase2/attention_mining/head_layer_rankings.json
  - outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_desc_x1_phase2/x1_logit_lens_smoke/probe_rows.jsonl
  - outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_desc_x1_phase2/intervention_plan/summary.json
  - outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_desc_x1_phase2/intervention/summary.json
  - outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_desc_x1_phase3_causal_binding/summary.json
  - outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_desc_x1_phase3_causal_binding/case_linked/case_mechanism_rows.jsonl
  - outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_desc_x1_phase4_binding_mechanism/summary.json
  - outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_desc_x1_phase4_binding_mechanism/instance_attention_binding/summary.json
  - outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_desc_x1_phase4_binding_mechanism/instance_attention_binding/rows.jsonl
  - outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_desc_x1_phase4_binding_mechanism/desc_x1_probe_linkage/summary.json
  - outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_desc_x1_phase5_logit_binding/summary.json
  - outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_desc_x1_phase5_logit_binding/x1_logit_binding_probe/summary.json
  - outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_desc_x1_phase5_logit_binding/x1_logit_binding_probe/rows.jsonl
  - outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_desc_x1_phase5/postx1_coord_slot_logit_lens/summary.json
  - outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_desc_x1_phase5/postx1_coord_slot_logit_lens/probe_rows.jsonl
  - outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_desc_x1_phase5_logit_binding_coordslot/summary.json
  - outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_desc_x1_phase5_logit_binding_coordslot/coord_slot_logit_binding_probe/summary.json
  - outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_desc_x1_phase5_logit_binding_coordslot/coord_slot_logit_binding_probe/rows.jsonl
---

# FN-Rescue Attention And Instance-Binding Findings

## Why This Note Exists

This note records the first full linked FN-rescue continuation and attention
diagnostic for the compact-full ET-RMP-CE Stage-1 checkpoint.  The purpose was
to test whether false-negative objects are purely visually unavailable, or
whether the autoregressive decoder can recover them when the next row is forced
into a desc-first continuation.

This is a mechanism diagnosis, not a full-validation benchmark and not a
deployable inference policy.  The experiment conditions model continuation on
GT-derived rescue targets and, for `desc_x1`, on a GT-derived or wrong-control
spatial hint.

## Scope

Checkpoint:

`outputs/stage1_2b/recursive_detection_ce_latest/compact_full_et_rmp_ce_support2_bsz16_4epoch_tokenrows_v2/compact-full-et-rmp-ce-support2-bsz16-4epoch-tokenrows-v2/v0-20260504-071356/checkpoint-3664`

Evidence scope:

`val200_attention_atlas_linked_fn_stratified`

Primary artifact root:

`outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_continuation`

Config hash:

`7de938eb49423115514a10f34096af7f0c8d18bcecadcebc265c1babee0314b8`

Merged validation:

- `summary.json`: `validation_status=ok`
- `merge_summary.json`: `validation_status=ok`
- all 8 expected shards merged

Artifact row counts:

| Output | Rows |
| --- | ---: |
| `selected_rescue_cases` | 512 |
| `rescue_rows` | 1536 |
| `rescue_generation_rows` | 575 |
| `rescue_replay_prefix_rows` | 575 |
| `rescue_attention_region_rows` | 4106816 |
| `rescue_decision_context_rows` | 575 |
| `rescue_candidate_region_rows` | 27539 |
| `wrong_control_rows` | 191 |

Selected-case buckets:

| Bucket | Counts |
| --- | --- |
| `prefix_quality` | `clean_prefix=176`, `duplicate_prefix=33`, `empty_prefix=93`, `fp_prefix=60`, `mixed_prefix=150` |
| `binding_bucket` | `no_local_object_diffuse=94`, `other=154`, `same_desc_competitor=219`, `target_low_rank=45` |
| `depth_bucket` | `d0=93`, `d1_3=192`, `d4_7=126`, `d8_plus=101` |
| `object_count_bucket` | `gt_1_5=128`, `gt_6_15=220`, `gt_16_plus=164` |

Important denominator caveat:

- `512` selected cases exist, but only `192` `desc_only` rows, `192`
  `desc_x1` rows, and `191` `desc_x1_wrong_control` rows were attempted.
- `320` rows per tier were skipped due to `prefix_reconstruction_failed`.
- Therefore the generation and attention conclusions below apply to the
  reconstructable-prefix diagnostic surface, not to all selected FN cases.

## Terminology And Bucket Semantics

This section fixes the working vocabulary used by the FN-rescue and desc->x1
binding artifacts.  These are diagnostic terms for this experiment family, not
general detection benchmark labels.

### Target

`target` is the rollout false-negative object selected for rescue.  In the
rescue arms, the prompt prefix is reconstructed from the model's previous
rollout, then a continuation is forced toward this object by supplying its desc
and, for `desc_x1`, its GT-derived `x1` coordinate token.

### Competitor

`competitor` is a same-image candidate region that can compete with the target
for instance binding after the desc has been supplied.  It is not a generic
negative patch or all non-target image content.

The current artifacts use two concrete competitor/source families:

| Source | Meaning |
| --- | --- |
| `same_desc_competitor_gt_object` | Another GT object in the same image with the same desc as the target. |
| `same_desc_rollout_prediction` | A same-desc object row that the model already emitted during rollout. |

The term matters because the template is desc-first.  Once the model is
conditioned on a desc such as `person`, the desc alone may identify a category
but not a specific instance.  Same-desc objects or previous same-desc rollout
predictions are therefore the most direct competitors for the subsequent
coordinate chain.

### Wrong-Control X1

`desc_x1_wrong_control` uses the target desc plus an `x1` value taken from a
same-desc competitor/source instead of the target.  This arm tests whether
`x1` affects instance binding, not whether the model can format coordinates.

Two wrong-control raw values were outside the norm1000 coordinate-token range:

| Raw | Used | Case | Source Family |
| ---: | ---: | --- | --- |
| `1127` | `999` | `row39:self_prefix:depth1:gt1` | `same_desc_rollout_prediction` |
| `1003` | `999` | `row14:self_prefix:depth0:gt0` | `same_desc_rollout_prediction` |

Both are artificial wrong-control values derived from same-desc rollout
predictions, then clamped for decoding.  They are not target GT coordinates.

### Case-Linked Mechanism Buckets

The `*_dependent` labels are intervention-derived buckets.  They describe how a
case behaved under replay or image-region masking; they are not COCO labels and
do not by themselves prove attention-head causality.

| Bucket | Operational Meaning |
| --- | --- |
| `target_dependent` | Masking or perturbing the target region changed the continuation enough to reduce target binding or flip success. |
| `competitor_dependent` | Masking or perturbing a same-desc competitor/source region changed the continuation enough to affect the target-binding outcome. |
| `robust_to_region_masks` | The tested target/competitor/source masks did not materially change the target-binding outcome under the current thresholds. |
| `no_op_replay` | The replay/control path did not create a meaningful output change, so no region-dependence claim is assigned. |
| `invalid_or_uninterpretable` | The generated row could not be parsed or compared reliably after replay/intervention. |

Important boundaries:

- `target_dependent` means target-region evidence is behaviorally relevant in
  the intervention setup.  It does not mean normal rollout will enumerate the
  target object.
- `competitor_dependent` means a same-desc competitor/source region affected
  the continuation.  It does not mean competitor attention is always harmful;
  Phase-3 masks often reduced target binding on average, which shows these
  regions can also carry useful context or overlap with the current routing
  evidence.
- `robust_to_region_masks` and `no_op_replay` should not be read as "no
  mechanism."  They mean this particular replay/mask probe did not expose a
  clean region-level dependence.

## Experiment Arms

Each selected FN case can create up to three rescue continuations:

- `desc_only`: append a partial compact-full row containing the target desc and
  `<|box_start|>`, then let the model generate all four coordinates.
- `desc_x1`: append the target desc plus the target `x1` token, then let the
  model generate `y1/x2/y2`.
- `desc_x1_wrong_control`: append the target desc plus a wrong-control `x1`
  drawn from a same-desc competitor or same-desc rollout prediction, then let
  the model generate `y1/x2/y2`.

Wrong-control source distribution:

| Source | Rows |
| --- | ---: |
| `same_desc_competitor_gt_object` | 118 |
| `same_desc_rollout_prediction` | 73 |

Two wrong-control `x1` values were outside norm1000 token range.  They came
from artificial same-desc rollout-prediction wrong controls, not from target GT
objects:

- `1127 -> 999`, case `row39:self_prefix:depth1:gt1`
- `1003 -> 999`, case `row14:self_prefix:depth0:gt0`

The pipeline records these as `hint_x1_raw`, `hint_x1_used`, and
`hint_x1_clamped=true`.

## Main Result

The headline result is that many rollout false negatives are recoverable under
forced desc-first continuation, and `x1` is a strong instance-binding signal.

| Tier | Attempted | Valid Parse | Target Desc Preserved | IoU30 | IoU50 | IoU75 | Primary Success | Duplicate IoU>0.95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `desc_only` | 192 | 189 | 192 | 105 | 97 | 80 | 95 | 2 |
| `desc_x1` | 192 | 190 | 192 | 165 | 155 | 126 | 155 | 0 |
| `desc_x1_wrong_control` | 191 | 187 | 191 | 62 | 51 | 33 | 49 | 2 |

Rates:

| Tier | IoU50 Rate | Primary Success Rate |
| --- | ---: | ---: |
| `desc_only` | 0.505 | 0.495 |
| `desc_x1` | 0.807 | 0.807 |
| `desc_x1_wrong_control` | 0.267 | 0.257 |

Outcome buckets:

| Tier | Primary Success | Target IoU Below 0.50 | Invalid Parse | Duplicate Rejected |
| --- | ---: | ---: | ---: | ---: |
| `desc_only` | 95 | 92 | 3 | 2 |
| `desc_x1` | 155 | 35 | 2 | 0 |
| `desc_x1_wrong_control` | 49 | 136 | 4 | 2 |

Paired comparisons:

- `192` cases had both `desc_only` and `desc_x1`.
- `61` cases were rescued by `desc_x1` when `desc_only` failed.
- `1` case was rescued by `desc_only` when `desc_x1` failed.
- `94` cases succeeded under both.
- `36` cases failed under both.

Wrong-control comparisons:

- `191` cases had all three arms.
- `49` wrong-control continuations still succeeded.
- `108` cases succeeded under correct `desc_x1` but failed under
  `desc_x1_wrong_control`.
- `3` cases succeeded under wrong control but failed under correct `desc_x1`.

## Mechanism Read

### 1. False negatives are not purely visual invisibility.

`desc_only` recovers about half of attempted FN objects at IoU50.  Since the
model can localize those objects after the target desc is forced into the next
row, at least this subset of FN errors is not explained by the visual encoder
simply failing to encode the object.

The stronger read is that low recall includes a row-selection or continuation
failure: the model often does not choose to enumerate an object during normal
rollout, but can bind and localize it once the next-row desc is supplied.

### 2. `x1` is a strong instance-binding seed.

Correct `desc_x1` raises primary success from `0.495` to `0.807`.  It also
raises IoU75 success from `80/192` to `126/192`, so the effect is not merely a
loose rescue; it improves precise localization.

The paired comparison is asymmetric: `61` cases are rescued only by `desc_x1`,
while only `1` case is rescued only by `desc_only`.

This supports the desc-first decomposition:

1. desc token narrows the problem to a category-conditioned continuation;
2. `x1` gives the decoder a spatial commitment point;
3. remaining coordinates are generated as an instance-localized continuation.

### 3. Wrong `x1` is not harmless.

Wrong-control success drops to `0.257`, far below correct `desc_x1`.  In `108`
trios, the correct `x1` succeeds but wrong-control `x1` fails.

This means the `x1` token is not just a formatting convenience.  It influences
where the model binds the instance.

### 4. Same-desc competitors are a central failure mode.

Binding-bucket rates show the most dramatic rescue effect:

| Tier | `same_desc_competitor` Primary Success |
| --- | ---: |
| `desc_only` | 0.136 |
| `desc_x1` | 0.758 |
| `desc_x1_wrong_control` | 0.045 |

When the target has same-desc competitors, desc alone is often insufficient.
Correct `x1` largely resolves the ambiguity; wrong-control `x1` strongly
misdirects it.

### 5. Crowded scenes remain hard, but `x1` helps substantially.

Object-count bucket rates:

| Tier | `gt_1_5` | `gt_6_15` | `gt_16_plus` |
| --- | ---: | ---: | ---: |
| `desc_only` | 0.776 | 0.265 | 0.038 |
| `desc_x1` | 0.939 | 0.735 | 0.500 |
| `desc_x1_wrong_control` | 0.454 | 0.074 | 0.000 |

The model's recall problem is much worse in crowded images, but correct `x1`
turns many crowded-scene FNs into recoverable continuations.

### 6. Duplication is not the main failure in this diagnostic slice.

Same-desc duplicate IoU>0.95 is rare:

- `desc_only`: `2/192`
- `desc_x1`: `0/192`
- `desc_x1_wrong_control`: `2/191`

The dominant failed outcome is `target_iou_below_0p50`, not duplicate-copy
rejection.  For this FN-rescue diagnostic, instance binding and localization are
more important than duplication burst.

## Attention Findings

Attention rows are diagnostic, not causal proof.  Still, the full linked run
shows a consistent pattern: attention is not a clean target-object spotlight.
Background and context mass are high, and correct versus wrong `x1` changes the
target/competitor balance.

Mean union attention mass by tier and layer group:

| Tier | Layer Group | Far Background | Context Ring | Target GT | Same-Desc Competitor GT | Same-Desc Rollout Pred | Wrong-Control Source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `desc_only` | `layers_00_07` | 0.0353 | 0.0070 | 0.0046 | 0.0077 | 0.0073 | - |
| `desc_only` | `layers_08_15` | 0.1523 | 0.0574 | 0.0390 | 0.0532 | 0.0486 | - |
| `desc_only` | `layers_16_23` | 0.1030 | 0.0505 | 0.0355 | 0.0475 | 0.0337 | - |
| `desc_only` | `layers_24_31` | 0.0112 | 0.0018 | 0.0012 | 0.0016 | 0.0018 | - |
| `desc_x1` | `layers_00_07` | 0.0289 | 0.0049 | 0.0034 | 0.0051 | 0.0048 | - |
| `desc_x1` | `layers_08_15` | 0.1304 | 0.0701 | 0.0480 | 0.0382 | 0.0415 | - |
| `desc_x1` | `layers_16_23` | 0.0996 | 0.0578 | 0.0410 | 0.0219 | 0.0333 | - |
| `desc_x1` | `layers_24_31` | 0.0169 | 0.0023 | 0.0014 | 0.0016 | 0.0024 | - |
| `desc_x1_wrong_control` | `layers_00_07` | 0.0296 | 0.0050 | 0.0035 | 0.0053 | 0.0049 | 0.0030 |
| `desc_x1_wrong_control` | `layers_08_15` | 0.1278 | 0.0442 | 0.0288 | 0.0545 | 0.0409 | 0.0382 |
| `desc_x1_wrong_control` | `layers_16_23` | 0.1019 | 0.0356 | 0.0217 | 0.0406 | 0.0355 | 0.0336 |
| `desc_x1_wrong_control` | `layers_24_31` | 0.0170 | 0.0022 | 0.0014 | 0.0014 | 0.0020 | 0.0012 |

Key reads:

- Mid layers dominate the usable signal.  `layers_08_15` and `layers_16_23`
  carry much larger region mass than early or final layers.
- Correct `x1` increases target GT attention relative to desc-only in mid
  layers:
  - `layers_08_15`: target `0.0390 -> 0.0480`
  - `layers_16_23`: target `0.0355 -> 0.0410`
- Correct `x1` reduces same-desc competitor GT mass:
  - `layers_08_15`: competitor `0.0532 -> 0.0382`
  - `layers_16_23`: competitor `0.0475 -> 0.0219`
- Wrong-control `x1` reverses this pattern:
  - `layers_08_15`: target `0.0288`, competitor `0.0545`,
    wrong source `0.0382`
  - `layers_16_23`: target `0.0217`, competitor `0.0406`,
    wrong source `0.0336`
- Far-background mass remains high across all arms.  Even in successful
  `desc_x1`, `layers_08_15` far-background mass is `0.1304`, much higher than
  target GT mass `0.0480`.

The attention evidence therefore supports a mixed mechanism:

- the model is not simply focusing on a single target object region;
- `x1` changes the target-versus-competitor balance;
- background/context remain part of the continuation state and may be involved
  in objectness, layout, or decoder bookkeeping.

## Phase-2 Desc-X1 Evidence Routing Update

The follow-up super-power plan materialized three analysis-only artifacts under:

`outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_desc_x1_phase2`

No production training was launched.  The work used completed FN-rescue
artifacts plus a sharded hidden-state probe smoke.

### Full head/layer/region mining

The CPU mining stage streamed the full attention artifact:

| Counter | Value |
| --- | ---: |
| `processed_attention_rows` | 4106816 |
| `joined_attention_rows` | 1268288 |
| `summary_row_count` | 7168 |
| `generation_outcome_count` | 575 |
| `skipped_scope_rows` | 2719360 |
| `skipped_region_rows` | 119168 |
| `missing_generation_rows` | 0 |

The strongest target-vs-same-desc-competitor margins concentrate around
`desc_x1/pre_y1` in middle layers:

| Tier | Role | Layer | Head | Target | Competitor | Far Background | Target-Competitor |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `desc_x1` | `pre_y1` | 13 | 6 | 0.6249 | 0.3253 | 0.9636 | 0.2997 |
| `desc_x1` | `pre_y1` | 17 | 12 | 0.6712 | 0.4287 | 0.9702 | 0.2425 |
| `desc_x1` | `pre_y1` | 16 | 8 | 0.6764 | 0.4379 | 0.9408 | 0.2385 |
| `desc_x1` | `pre_y1` | 14 | 15 | 0.4201 | 0.2288 | 0.6259 | 0.1913 |
| `desc_x1` | `pre_y1` | 16 | 13 | 0.4969 | 0.3060 | 0.6844 | 0.1909 |

Wrong-control rows reverse the same region competition in several of these
heads.  For example, `desc_x1_wrong_control/pre_y1` at layer 16 head 8 has
target `0.2788`, competitor `0.6813`, and far-background `0.9122`; layer 17
head 12 has target `0.2689`, competitor `0.6584`, and far-background `0.9605`.

This is the clearest current attention-side evidence that the model is not
merely deciding "continue versus stop"; it is routing a desc-conditioned
continuation through competing same-desc spatial evidence.  Far-background mass
is also extremely high in the same heads, so it remains a candidate sink or
bookkeeping reservoir, but this remains non-causal until intervention.

### X1 logit-lens smoke

The Lane-D hidden-state logit-lens smoke wrote `13824` probe rows over `512`
selected cases.  `desc_end`, `box_start`, and `pre_x1` had available `x1`
logit-lens metrics; other compact roles were intentionally unavailable for this
smoke.

| Role | Layer Group | Layer | N | Mean Target X1 Rank | Median Rank | Top1 | Mean Target-Top1 Margin |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `desc_end` | `middle` | 12 | 512 | 431.09 | 417.0 | 0 | -1.0819 |
| `desc_end` | `late` | 24 | 512 | 434.67 | 406.5 | 1 | -1.8424 |
| `desc_end` | `last` | 27 | 512 | 468.46 | 477.0 | 2 | -7.1028 |
| `box_start` | `middle` | 12 | 512 | 485.65 | 454.5 | 0 | -1.8099 |
| `box_start` | `late` | 24 | 512 | 345.56 | 283.0 | 2 | -2.2761 |
| `box_start` | `last` | 27 | 512 | 226.05 | 146.0 | 0 | -3.3234 |
| `pre_x1` | `middle` | 12 | 512 | 485.65 | 454.5 | 0 | -1.8099 |
| `pre_x1` | `late` | 24 | 512 | 345.56 | 283.0 | 2 | -2.2761 |
| `pre_x1` | `last` | 27 | 512 | 226.05 | 146.0 | 0 | -3.3234 |

This does not support a strong claim that the exact target `x1` is already
cleanly linearly readable at `desc_end`.  The final `box_start/pre_x1` states
move the target `x1` rank substantially upward, but top-1 remains rare.  The
current read is therefore: desc provides category intent; the spatial binding
becomes more expressed near the coordinate boundary; exact coordinate choice is
still weak/noisy under this simple logit-lens probe.

### CPU intervention plan

The causal stage currently materializes a plan, not executed interventions:

| Counter | Value |
| --- | ---: |
| `case_count` | 145 |
| `selected_generation_rows` | 192 |
| `planned_intervention_rows` | 989 |
| `far_background_sink_mask` | 50 |
| `no_op_control` | 192 |
| `same_desc_competitor_mask` | 244 |
| `same_desc_rollout_prediction_mask` | 311 |
| `target_gt_mask` | 50 |
| `wrong_control_source_region_mask` | 142 |

The next executable causal test should begin with no-op parity and target-mask
sanity.  Only if masking target GT reliably reduces target binding should
same-desc competitor and far-background sink suppression be interpreted as
causal evidence.

The first GPU smoke executed exactly that no-op/target-mask sanity check on `4`
cases (`8` continuation rows):

| Intervention | Rows | Exact Tail Match | Invalid Parse | Mean Target IoU Delta | Primary Success Changed |
| --- | ---: | ---: | ---: | ---: | ---: |
| `no_op_control` | 4 | 4 | 0 | 0.0000 | 0 |
| `target_gt_mask` | 4 | 1 | 0 | -0.3163 | 2 |

This is tiny evidence, but it is causally stronger than attention mining:
rerunning the same prompts without image modification reproduced the original
tails exactly, while masking the target GT region changed three of four tails
and flipped two successful continuations to failure.  One large/near-full-image
case remained unchanged, and one case slightly improved, so target masking is
not yet a universal effect.  The result is sufficient to justify broader
competitor-mask and sink-mask tests, but not yet sufficient to train a
background-suppression objective.

## Phase-3 Attention-Guided Causal Binding Smoke

The next independent spec is:

`docs/superpowers/specs/2026-06-02-fn-rescue-attention-guided-causal-binding-design.md`

The implementation scope is the worktree:

`/data/CoordExp/.worktrees/fn-rescue-attention-probes`

The Phase-3 smoke artifact root is:

`outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_desc_x1_phase3_causal_binding_smoke`

This stage sharpens the causal estimand: it tests attention-guided
image-region dependence, not attention-head causality.  Attention rows select
or describe candidate regions; image-region masking tests whether those visual
regions change the autoregressive continuation.

### Target-mask lane

| Intervention | Rows | Cases | Exact Tail Match | Invalid Parse | Mean Target IoU Delta | Primary Success Changed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_op_control` | 4 | 4 | 4 | 0 | 0.0000 | 0 |
| `target_gt_mask` | 4 | 4 | 1 | 0 | -0.3163 | 2 |

The Phase-3 target-mask lane reproduces the Phase-2 tiny target sanity result
in a separate artifact root and now records mask metadata such as
`mask_applied`, `mask_pixel_box_xyxy`, `mask_area_pixels`, `image_size_wh`, and
`intervention_error`.

### Competitor/source lane

| Intervention | Rows | Cases | Exact Tail Match | Invalid Parse | Mean Target IoU Delta | Primary Success Changed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_op_control` | 4 | 4 | 4 | 0 | 0.0000 | 0 |
| `same_desc_competitor_mask` | 11 | 2 | 4 | 0 | -0.1056 | 1 |
| `same_desc_rollout_prediction_mask` | 10 | 2 | 3 | 0 | -0.1477 | 2 |
| `wrong_control_source_region_mask` | 2 | 2 | 0 | 0 | 0.0000 | 0 |

This is a smoke result, not a final mechanism claim.  It confirms that the
new lane selector covers both `desc_x1_success_same_desc_competitor` and
`desc_x1_wrong_control_failure`; earlier global-cap selection would have
starved wrong-control rows.  In this tiny sample, same-desc competitor and
rollout-prediction masks more often hurt than help target IoU, so competitor
attention should not yet be interpreted as a harmful-stealing mechanism.

### Case-linked table

The case-linked stage scans the raw attention artifact and joins intervention
outcomes with coarse case-level `pre_y1/union` attention summaries.

| Counter | Value |
| --- | ---: |
| `row_count` | 35 |
| `case_count` | 6 |
| `processed_attention_rows` | 4106816 |
| `joined_attention_rows` | 11648 |

Mechanism buckets:

| Bucket | Rows |
| --- | ---: |
| `no_op_replay` | 8 |
| `robust_to_region_masks` | 22 |
| `target_dependent` | 2 |
| `competitor_dependent` | 3 |

The current case-linked table uses `aggregation_scope=union`, so it supports
coarse case-level region-kind evidence.  It does not prove that a specific
same-desc competitor instance or attention head caused the change.  Exact
source-specific attention joins require `aggregation_scope=instance` and keys
including `(case_id, rescue_tier, role, region_kind, region_instance_id)`.

## Interpretation

The best-supported current read is:

`fn_recall_is_partly_autoregressive_selection_and_binding_failure`

More explicitly:

- The model often has enough visual information to localize an FN object after
  a desc-conditioned continuation.
- Normal rollout still fails to choose that object as the next row.
- Desc-first behavior appears natural for this architecture: the desc supplies
  category-level intent, and `x1` supplies instance-level commitment.
- Same-desc competitor scenes expose the binding weakness most clearly.
- Wrong `x1` pulls the continuation away from the target, so spatial commitment
  is real and measurable.
- Attention is background/context-heavy rather than purely object-centric, so
  attention mass should be used as a diagnostic surface, not a complete causal
  explanation.

This is not evidence that the model globally perceives all objects early and
merely translates them later.  It is more consistent with a dynamic
row-continuation process where desc and prefix condition the next local binding
decision.

## Limits

- GT leakage remains possible because rescue continuations use GT-derived desc
  and, for `desc_x1`, GT-derived `x1`.
- The evidence scope is linked-stratified, not the full val200 FN universe.
- `prefix_reconstruction_failed` removes 320 selected cases per tier from the
  attempted generation surface.
- Attention mass is not causal attribution.
- Gallery rows are qualitative examples, not metric sources.
- Clamped wrong-control x1 values are recorded, but only `2` generation rows
  were affected; both became invalid parses and do not drive the headline.

## Next Exploration Directions

### A. Causal region intervention on continuation

Run target/wrong-control/competitor visual-region masking or patching during the
same continuation prompts.  The goal is to distinguish "attention moved there"
from "that region causally determined the continuation."

Priority contrasts:

- correct `desc_x1` success cases: mask target GT versus competitor GT;
- wrong-control failures: mask wrong-control source versus target GT;
- same-desc competitor bucket: swap or patch competitor and target regions.

### B. Prefix-reconstruction recovery

The current full result loses 320 selected cases per tier to
`prefix_reconstruction_failed`.  Recovering these cases would extend the probe
to deeper prefixes (`d4_7`, `d8_plus`) and make the continuation read less
biased toward early rows.

Likely route:

- reconstruct assistant prefixes from token trace rather than raw compact text
  when possible;
- preserve a separate `prefix_source=raw_text|token_trace` artifact column;
- rerun FN-rescue to see whether long-prefix attention dilution or binding
  instability appears.

### C. Desc-first training signal design

The result suggests that fighting desc-first may be unnecessary.  A more
architecture-aligned training signal would supervise row selection as:

1. category/objectness continuation intent;
2. instance spatial commitment;
3. coordinate completion.

Candidate training ideas:

- residual-set or coverage-aware next-row sampling;
- desc-conditioned positive continuation examples for omitted objects;
- hard same-desc competitor negatives for spatial commitment;
- lightweight `x1` or coarse-location auxiliary supervision before full bbox.

### D. Wrong-control contrastive objective

Wrong `x1` reliably degrades target binding.  This can become an explicit
training diagnostic or objective:

- positive: target desc plus correct x1 should bind target;
- negative: same desc plus competitor x1 should not score as target;
- metric: target-vs-wrong control delta in IoU50, target attention, and
  competitor attention.

### E. Attention-head and layer specialization mining

The current summary aggregates layer groups.  The 4.1M attention rows can be
mined for heads that separate:

- target GT vs same-desc competitor;
- correct x1 vs wrong-control x1;
- success vs target-IoU-below-0.50.

This is the best next step if the goal is to understand internal mechanism
rather than immediately design a training objective.

## Phase-3 ScriptMaster Smoke Refresh

Scope:

```text
worktree: /data/CoordExp/.worktrees/fn-rescue-attention-probes
artifact_root: /data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_desc_x1_phase3_causal_binding_smoke
session: autoreg_fn_rescue_phase3_smoke_scriptmaster
```

The Phase-3 ScriptMaster ran the smoke workflow as analysis orchestration, not
production training:

1. `target_mask` on GPU `0`;
2. `competitor_source` on GPU `1`, gated by target-mask no-op/effect summary;
3. `sink_triage,case_linked,report` on CPU.

The launcher command file is:

```text
/data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_desc_x1_phase3_causal_binding_smoke/logs/autoreg_fn_rescue_phase3_smoke_scriptmaster_commands.sh
```

Top-level artifacts now include:

```text
target_mask/summary.json
target_mask/intervention_rows.jsonl
target_mask/report.md
competitor_source/summary.json
competitor_source/intervention_rows.jsonl
competitor_source/report.md
sink_triage/summary.json
sink_triage/sink_candidate_rows.jsonl
sink_triage/report.md
case_linked/summary.json
case_linked/case_mechanism_rows.jsonl
case_linked/report.md
summary.json
manifest.json
report.md
```

Updated smoke counts:

| Stage | Rows | Cases | Notes |
| --- | ---: | ---: | --- |
| `target_mask` | 8 | 4 | no-op exact-tail parity `4/4`; target mask mean IoU delta `-0.316272`; success flips `2/4` |
| `competitor_source` | 27 | 4 | no-op exact-tail parity `4/4`; wrong-control rows `2` |
| `sink_triage` | 5000 | 383 | bounded top candidates from `171584` far-background candidates over `4106816` attention rows |
| `case_linked` | 35 | 6 | valid paired rows `35`; joined attention rows `11648` |

Case-linked mechanism buckets after paired no-op gating:

| Bucket | Rows |
| --- | ---: |
| `no_op_replay` | 8 |
| `robust_to_region_masks` | 22 |
| `target_dependent` | 2 |
| `competitor_dependent` | 3 |

Important interpretation boundaries:

- No-op parity is checked at generated-tail text, parsed-box, IoU, and success
  levels.  Token-level replay is not claimed because baseline generated token
  IDs are not persisted in the intervention rows.
- Wrong-control source masks are metric-neutral in this smoke
  (`mean_target_iou_delta=0`, `primary_success_changed=0`), but they are not
  generation-neutral (`exact_tail_match_baseline=0/2`).
- `sink_triage` is candidate-only.  It ranks far-background attention rows but
  does not mask, suppress, or train on background tokens.
- `top_attention_heads_for_case` is diagnostic metadata for linked analysis;
  the image-region intervention does not prove head-level causal mechanisms.
- The first ScriptMaster uses the 8 GPUs as a resource pool but only binds one
  GPU to each decode lane because Phase-3 shard-local merge is not yet
  implemented.

## Phase-3 ScriptMaster Full Run

Scope:

```text
worktree: /data/CoordExp/.worktrees/fn-rescue-attention-probes
artifact_root: /data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_desc_x1_phase3_causal_binding
session: autoreg_fn_rescue_phase3_ckpt3664
```

The full Phase-3 ScriptMaster completed with whole-lane GPU allocation:

1. `target_mask` on GPU `0`;
2. `competitor_source` on GPU `1`;
3. `sink_triage,case_linked,report` on CPU.

The launcher did not shard one intervention lane across multiple GPUs because
Phase-3 shard-local outputs and merge validation are not implemented yet.

Full-run artifact contract check passed for:

```text
target_mask/summary.json
target_mask/intervention_rows.jsonl
target_mask/report.md
competitor_source/summary.json
competitor_source/intervention_rows.jsonl
competitor_source/report.md
sink_triage/summary.json
sink_triage/sink_candidate_rows.jsonl
sink_triage/report.md
case_linked/summary.json
case_linked/case_mechanism_rows.jsonl
case_linked/report.md
summary.json
manifest.json
report.md
```

Full-run counts:

| Stage | Rows | Cases | Notes |
| --- | ---: | ---: | --- |
| `target_mask` | 100 | 50 | no-op exact-tail parity `50/50`; target mask mean IoU delta `-0.380324`; success flips `28/50`; invalid target-mask parses `1/50` |
| `competitor_source` | 200 | 28 | no-op exact-tail parity `32/32`; no-op invalid parses `1/32`; wrong-control rows `16`; wrong-control invalid parses `3/16` |
| `sink_triage` | 5000 | 383 | bounded top candidates from `171584` far-background candidates over `4106816` attention rows |
| `case_linked` | 300 | 66 | valid paired rows `299`; joined attention rows `123200` |

Intervention summaries:

| Lane | Intervention | Rows | Mean Target-IoU Delta | Success Changed | Exact Tail Match | Invalid Parse |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `target_mask` | `no_op_control` | 50 | `0.000000` | 0 | 50 | 0 |
| `target_mask` | `target_gt_mask` | 50 | `-0.380324` | 28 | 2 | 1 |
| `competitor_source` | `no_op_control` | 32 | `0.000000` | 0 | 32 | 1 |
| `competitor_source` | `same_desc_competitor_mask` | 68 | `-0.077918` | 6 | 27 | 0 |
| `competitor_source` | `same_desc_rollout_prediction_mask` | 84 | `-0.118669` | 13 | 35 | 0 |
| `competitor_source` | `wrong_control_source_region_mask` | 16 | `-0.028977` | 0 | 1 | 3 |

Case-linked mechanism buckets after paired no-op gating:

| Bucket | Rows |
| --- | ---: |
| `no_op_replay` | 81 |
| `robust_to_region_masks` | 165 |
| `target_dependent` | 29 |
| `competitor_dependent` | 20 |
| `invalid_or_uninterpretable` | 5 |

Full-run interpretation:

- Target-region evidence is causally relevant in a large fraction of the
  checked desc+x1 cases: masking the GT target flips success for `28/50` cases
  and produces mean target-IoU delta about `-0.38`.
- Same-description competitor/source interventions are not simply helpful
  controls.  In this run they mostly reduce target binding on average, which
  suggests the selected competitor/source masks often remove useful context or
  overlap with the model's current target-routing evidence rather than cleanly
  isolating a competing object.
- Wrong-control source masks remain largely metric-neutral for target success
  (`primary_success_changed=0/16`), but they are not generation-neutral and have
  invalid parse risk (`3/16` invalid).
- Sink/background evidence is still candidate-only.  The top `5000`
  far-background rows identify where to inspect next, not a justification for
  background suppression.
- Case-linked buckets support the working hypothesis that recall loss includes
  autoregressive selection/binding failures, but the current intervention is
  image-region causal evidence, not attention-head causal proof.

## Phase-4 Desc-X1 Binding Mechanism Linkage

Scope:

```text
worktree: /data/CoordExp/.worktrees/fn-rescue-attention-probes
smoke_artifact_root: /data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_desc_x1_phase4_binding_mechanism_smoke
full_artifact_root: /data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_desc_x1_phase4_binding_mechanism
config: configs/analysis/autoreg_fn_rescue_desc_x1_phase4/ckpt3664_val200.yaml
```

Purpose:

Phase-4 links the Phase-3 causal buckets to existing instance-level attention
rows and to the Lane-D desc/x1 probe artifact availability.  It deliberately
does not claim attention-head causality.  The goal is to test whether the
model's desc-first continuation and x1 binding failures are visible as
instance-level attention allocation patterns, and whether the next independent
probe should look inside hidden/logit states rather than only rollout outcomes.

Artifact contract check passed for the full root:

```text
instance_attention_binding/rows.jsonl
instance_attention_binding/summary.json
instance_attention_binding/report.md
desc_x1_probe_linkage/summary.json
desc_x1_probe_linkage/report.md
summary.json
report.md
```

Full-run counts:

| Stage | Rows | Cases | Notes |
| --- | ---: | ---: | --- |
| `instance_attention_binding` | 300 | 66 | processed `4106816` attention rows; joined `407232` instance attention rows |
| `desc_x1_probe_linkage` | 13824 probe rows | 40 joined Phase-3 cases | roles include `desc_end`, `pre_x1`, `post_x1`, `post_y1`; layer groups include `middle`, `late`, `last` |

Bucket-level target-vs-competitor instance attention margins:

| Bucket | Rows | Valid Paired Rows | Mean Target - Competitor | Median | Positive Fraction | Negative Fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `robust_to_region_masks` | 165 | 165 | `0.020610` | `0.021189` | `0.921` | `0.079` |
| `target_dependent` | 29 | 29 | `0.016245` | `0.016566` | `1.000` | `0.000` |
| `no_op_replay` | 81 | 81 | `0.015082` | `0.016341` | `0.815` | `0.185` |
| `invalid_or_uninterpretable` | 5 | 4 | `0.012523` | `0.005386` | `0.800` | `0.200` |
| `competitor_dependent` | 20 | 20 | `0.011744` | `0.015879` | `0.900` | `0.100` |

Top instance-attention heads are dominated by `pre_y1` roles:

| Bucket | Most Frequent Top Heads |
| --- | --- |
| `robust_to_region_masks` | `(layer=16, head=8, pre_y1)` 156; `(17, 12, pre_y1)` 132; `(17, 8, pre_y1)` 117 |
| `target_dependent` | `(16, 8, pre_y1)` 24; `(17, 12, pre_y1)` 22; `(17, 8, pre_y1)` 16 |
| `competitor_dependent` | `(17, 12, pre_y1)` 20; `(16, 8, pre_y1)` 18; `(17, 8, pre_y1)` 17 |

Phase-4 read:

- Instance-level attention is correlated with successful target binding, but it
  is not a sufficient explanation.  `target_dependent` has a clean positive
  target-minus-competitor margin in all rows, yet `competitor_dependent` also
  has mostly positive margins.  Therefore the next mechanism question is not
  simply "did the model attend to the target instance?".
- The more useful hypothesis is that desc-first decoding forms a category or
  desc-conditioned candidate set, and the transition into `x1/y1` has to bind
  one instance from that set.  Attention can already be target-positive while
  the hidden/logit state still routes the continuation to a competitor, an
  existing rollout prediction, or an unstable coordinate basin.
- The most frequent heads are concentrated at `pre_y1`, especially late layers
  16-17.  This suggests the decisive binding signal may sharpen after `x1` has
  been emitted, not only at the desc token.  This is compatible with the earlier
  result that correct `x1` rescues many FNs and wrong `x1` strongly misdirects
  binding.
- Background/sink suppression should not be the first training intervention.
  The current evidence says background attention is real, but Phase-4 shows the
  target can still receive more instance attention than competitors even when
  the case remains competitor-dependent or no-op.  A background-only remedy
  risks treating a symptom while leaving desc->x1 instance choice untouched.
- Lane-D linkage is available but incomplete for this exact Phase-3 surface:
  `13824` probe rows join to `40` of the `66` Phase-3 cases.  That is enough to
  start a hidden/logit linkage analysis, but not enough to present as a full
  explanation unless coverage is expanded or the missing cases are stratified.

Next research direction from Phase-4:

1. Build a hidden/logit binding probe over the existing linked cases: compare
   `desc_end`, `pre_x1`, `post_x1`, and `pre_y1` states for target versus
   same-desc competitor x1 candidates.
2. Split by mechanism bucket and by whether target-minus-competitor attention
   is positive.  The crucial slice is cases where target attention is positive
   but binding still fails or routes to a competitor.
3. Only after this probe should we design a training objective.  The likely
   useful target is not generic attention suppression, but a desc-conditioned
   instance-choice or coordinate-binding supervision signal around the
   desc->x1 transition.

## Phase-5 Desc-X1 Logit Binding Probe

Scope:

```text
worktree: /data/CoordExp/.worktrees/fn-rescue-attention-probes
smoke_artifact_root: /data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_desc_x1_phase5_logit_binding_smoke
full_artifact_root: /data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_desc_x1_phase5_logit_binding
config: configs/analysis/autoreg_fn_rescue_desc_x1_phase5/ckpt3664_val200.yaml
```

Purpose:

Phase-5 separates two things that were easy to conflate:

- `x1_target_rank` and `x1_top_peak_attribution` are Lane-C case-selection
  metadata copied into every probe row.
- `x1_logit_lens_rank`, `x1_logit_lens_top1_bin`, and
  `x1_logit_lens_target_minus_top1` are state-dependent logit-lens fields.

This matters because a mechanism claim about desc->x1 binding must be based on
the `x1_logit_lens_*` fields, not the copied Lane-C ledger fields.

Artifact contract check passed for the full root:

```text
x1_logit_binding_probe/rows.jsonl
x1_logit_binding_probe/summary.json
x1_logit_binding_probe/report.md
summary.json
report.md
```

Full-run counts:

| Quantity | Value |
| --- | ---: |
| Phase-4 rows | 300 |
| Processed Lane-D probe rows | 13824 |
| Joined probe rows | 480 |
| Joined output rows | 2712 |
| Joined cases | 40 |
| Available state-dependent logit rows | 2034 |
| Unique available case-role-layer rows | 360 |

Coverage:

| Role | State-Dependent X1 Logit-Lens Coverage |
| --- | --- |
| `desc_end` | available for `middle`, `late`, `last` |
| `box_start` | available for `middle`, `late`, `last` |
| `pre_x1` | available for `middle`, `late`, `last` |
| `post_x1` | rows exist but `x1_logit_lens_available=false` |
| `pre_y1` | no joined logit-lens rows |

Position-inventory sanity check:

- For case `row0:self_prefix:depth0:gt19`, `box_start` and `pre_x1` both map
  to absolute token index `1387`, token `<|coord_858|>`.  Therefore their
  logit-lens summaries are the same position and should not be interpreted as
  two independent decoding stages.
- `post_x1` maps to the next coordinate token position, but current Lane-D
  config did not enable state-dependent x1 logit lens there.

Bucket-level logit-binding summaries, row-weighted by Phase-4 mechanism rows:

| Bucket | Available Rows | Mean X1 Rank | Median X1 Rank | Mean Target - Top1 Logit | Negative Margin Fraction | Positive Attention But Poor X1 Rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `target_dependent` | 216 | `412.426` | `380.5` | `-2.995` | `1.000` | 216 |
| `competitor_dependent` | 162 | `340.858` | `247.0` | `-2.989` | `1.000` | 144 |
| `robust_to_region_masks` | 1143 | `348.894` | `252.0` | `-2.962` | `1.000` | 1080 |
| `no_op_replay` | 504 | `368.603` | `328.0` | `-2.940` | `1.000` | 432 |
| `invalid_or_uninterpretable` | 9 | `341.889` | `245.0` | `-2.521` | `1.000` | 9 |

Unique case-role-layer role summaries:

| Role/Layer | Unique Rows | Mean X1 Rank | Median X1 Rank | Mean Target - Top1 Logit | Negative Margin Fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| `desc_end::middle` | 40 | `402.075` | `345.0` | `-1.066` | `1.000` |
| `desc_end::late` | 40 | `416.350` | `376.0` | `-1.674` | `1.000` |
| `desc_end::last` | 40 | `437.975` | `426.5` | `-6.310` | `1.000` |
| `box_start::middle` | 40 | `484.600` | `440.0` | `-1.671` | `1.000` |
| `box_start::late` | 40 | `406.750` | `260.5` | `-2.826` | `1.000` |
| `box_start::last` | 40 | `222.150` | `120.5` | `-4.252` | `1.000` |
| `pre_x1::middle` | 40 | `484.600` | `440.0` | `-1.671` | `1.000` |
| `pre_x1::late` | 40 | `406.750` | `260.5` | `-2.826` | `1.000` |
| `pre_x1::last` | 40 | `222.150` | `120.5` | `-4.252` | `1.000` |

Critical-slice result:

- `1881 / 2034 = 0.9248` available row-weighted rows have positive target
  instance attention but poor x1 logit binding under the configured criterion
  (`rank > 100` or negative target-minus-top1 logit margin).
- Every available state-dependent row has negative target-minus-top1 logit
  margin.  In other words, the target x1 coordinate is never the top coordinate
  under this logit-lens probe on the joined Phase-4 surface.

Phase-5 read:

- This directly supports the Phase-4 suspicion: target-positive attention is
  not enough.  The decoder can allocate more instance attention to the target
  while the coordinate-logit state still prefers another x1 bin.
- The current artifact is strongest as an early-coordinate probe.  It covers
  `desc_end` and the `<|coord_x1|>` token position, but does not yet cover the
  decisive `post_x1/pre_y1` transition that Phase-4 attention heads highlighted.
- The next experiment should rerun Lane-D with `x1_logit_lens_roles` including
  `post_x1` and, if position inventory supports it, `pre_y1`.  That will test
  whether the model's binding sharpens after x1 is emitted or remains
  competitor/coordinate-basin dominated.
- Training-wise, these results still point away from generic background
  suppression as the first move.  A better candidate objective is
  desc-conditioned coordinate-choice supervision at the desc->x1 and
  post_x1->y1 boundary, especially for same-desc competitor cases and positive
  target-attention-but-poor-x1-logit cases.

## Phase-5B Coord-Slot Logit Binding Probe

Scope:

```text
worktree: /data/CoordExp/.worktrees/fn-rescue-attention-probes
lane_d_artifact_root: /data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_desc_x1_phase5/postx1_coord_slot_logit_lens
phase5_artifact_root: /data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_desc_x1_phase5_logit_binding_coordslot
lane_d_config: configs/analysis/autoreg_hidden_state_probe/ckpt3664_lane_d_coord_slot_logit_lens_postx1.yaml
phase5_config: configs/analysis/autoreg_fn_rescue_desc_x1_phase5/ckpt3664_val200_coordslot.yaml
tmux_session: autoreg_lane_d_coordslot_ckpt3664
```

Why Phase-5B was needed:

The earlier `post_x1` run still used `x1_logit_lens_*`, which asks whether the
target `x1` bin is linearly preferred even after `x1` has already been emitted.
That is not the right probe for the next-token state.  Phase-5B adds
`coord_slot_logit_lens_*` fields:

- `pre_x1` probes target `x1`;
- `post_x1` probes target `y1`;
- `post_y1` probes target `x2`.

This makes `post_x1` a real proxy for the pre-`y1` decision.

Lane-D extraction:

| Quantity | Value |
| --- | ---: |
| Shards | 8 |
| Selected cases | 512 |
| Position inventory rows | 4608 |
| Probe rows | 13824 |
| Coord-slot available rows | 4608 |
| Covered slots | `x1`, `y1`, `x2` |

Phase-5B linkage counts:

| Quantity | Value |
| --- | ---: |
| Phase-4 rows | 300 |
| Joined cases | 40 |
| Coord-slot joined output rows | 2034 |
| Available coord-slot rows | 2034 |
| Unique available case-role-slot-layer rows | 360 |

Unique role-slot summaries:

| Role/Slot | Unique Rows | Mean Rank | Median Rank | Mean Target - Top1 Logit | Negative Margin Fraction | Positive Attention But Poor Rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pre_x1::x1` | 120 | `371.167` | `267.0` | `-2.917` | `1.000` | 99 |
| `post_x1::y1` | 120 | `181.967` | `63.0` | `-1.304` | `0.942` | 92 |
| `post_y1::x2` | 120 | `235.300` | `30.0` | `-1.265` | `0.883` | 88 |

Bucket-level coord-slot summaries:

| Bucket | Available Rows | Mean Rank | Median Rank | Mean Target - Top1 Logit | Negative Margin Fraction | Positive Attention But Poor Rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `target_dependent` | 216 | `251.694` | `124.0` | `-1.734` | `0.949` | 205 |
| `competitor_dependent` | 162 | `184.549` | `86.0` | `-1.595` | `0.944` | 139 |
| `robust_to_region_masks` | 1143 | `256.232` | `128.0` | `-1.934` | `0.951` | 1027 |
| `no_op_replay` | 504 | `253.006` | `120.0` | `-1.795` | `0.931` | 402 |
| `invalid_or_uninterpretable` | 9 | `340.000` | `200.0` | `-1.678` | `1.000` | 9 |

Critical-slice result:

- `1782 / 2034 = 0.8761` available coord-slot rows still have positive target
  attention but poor coordinate-slot logit binding.
- This is lower than the x1-only probe's `0.9248`, so the state does sharpen
  after coordinates begin, but it remains weak: the target slot is still not top
  in most rows.

Phase-5B read:

- The decoder does improve after `x1`: `post_x1::y1` has much better median
  rank (`63`) than `pre_x1::x1` (`267`).  `post_y1::x2` improves further in
  median rank (`30`).
- However, negative target-minus-top1 margin remains high even for the true
  next-slot probes: `post_x1::y1` has negative margin in `0.942` of unique rows,
  and `post_y1::x2` in `0.883`.
- This supports a more nuanced mechanism: `x1` is a powerful spatial commitment
  token because it moves the hidden state toward the right local coordinate
  manifold, but the model still often has a stronger competing coordinate basin.
- The observed weak surface is not limited to the first coordinate.  The
  measured transition chain is desc->x1, x1->y1, and y1->x2, with each slot
  improving over the previous one while still showing high negative-margin
  fractions.

## Evidence-Backed Conclusions Across Phases

The following conclusions are limited to the checkpoint, `val200` diagnostic
surface, reconstructable prefixes, and linked Phase-3/4/5 case subset recorded
above.  They should not be read as full-val detection metrics.

### 1. Many FN objects are recoverable under forced continuation.

`desc_only` recovers `97/192` attempted rows at IoU50 and `95/192` at the
primary-success criterion.  This establishes that at least part of the recall
loss is not explained by target objects being visually unavailable to the
model under all conditions.

### 2. X1 is a behaviorally strong binding token.

Correct `desc_x1` raises primary success from `95/192` to `155/192`, while
`desc_x1_wrong_control` drops to `49/191`.  In paired trios, `108` cases
succeed under correct `desc_x1` but fail under wrong-control `x1`.

This evidence supports the narrower claim that `x1` materially influences
which instance the coordinate chain binds.

### 3. Same-desc competition is a measured failure surface.

For `same_desc_competitor` cases, primary success is:

| Tier | Primary Success |
| --- | ---: |
| `desc_only` | `0.136` |
| `desc_x1` | `0.758` |
| `desc_x1_wrong_control` | `0.045` |

This makes same-desc competitors a central diagnostic slice for desc-first
instance binding.  It does not imply that every false positive is a duplicate
or that every competitor region is harmful.

### 4. Target-region evidence is causally relevant in the Phase-3 mask lane.

The full Phase-3 target-mask lane has no-op exact-tail parity for `50/50`
rows.  Masking the target GT region changes the continuation strongly enough
to produce mean target-IoU delta `-0.380324` and success flips in `28/50`
cases.

This is image-region causal evidence, not attention-head causal evidence.

### 5. Instance attention is target-positive but not sufficient.

Phase-4 instance-level linkage shows positive target-minus-competitor attention
margins in most rows, including:

| Bucket | Mean Target - Competitor | Positive Fraction |
| --- | ---: | ---: |
| `target_dependent` | `0.016245` | `1.000` |
| `competitor_dependent` | `0.011744` | `0.900` |
| `robust_to_region_masks` | `0.020610` | `0.921` |

Because `competitor_dependent` and `robust_to_region_masks` also often have
positive target attention, "the model attended to the target" is not by itself
a sufficient explanation for correct target binding.

### 6. Coordinate-slot logit binding remains weak after attention is positive.

Phase-5 reports that `1881/2034 = 0.9248` available row-weighted rows have
positive target attention but poor x1 logit binding under the configured
criterion.  Phase-5B uses slot-aware probes and lowers that critical slice to
`1782/2034 = 0.8761`, showing improvement after coordinate generation begins
but not a fully resolved coordinate choice.

Slot-aware unique summaries show the coordinate chain sharpens over time:

| Role/Slot | Median Rank | Negative Margin Fraction |
| --- | ---: | ---: |
| `pre_x1::x1` | `267.0` | `1.000` |
| `post_x1::y1` | `63.0` | `0.942` |
| `post_y1::x2` | `30.0` | `0.883` |

The measured pattern is: x1 moves the state toward a better local coordinate
manifold, but the target slot is still not top-ranked in most linked rows.

### 7. Background attention is real but not yet the primary causal explanation.

The attention tables show high far-background mass, including `0.1304` in
`desc_x1/layers_08_15`, versus target GT mass `0.0480` in the same layer group.
However, the current background/sink evidence is candidate-level or
diagnostic.  No completed stage in this note establishes that suppressing
background patches causally improves FN-rescue recall.

### 8. Duplication is not the dominant failure in this diagnostic slice.

Same-desc duplicate IoU>0.95 is rare in the attempted generation rows:

| Tier | Duplicate IoU>0.95 |
| --- | ---: |
| `desc_only` | `2/192` |
| `desc_x1` | `0/192` |
| `desc_x1_wrong_control` | `2/191` |

The dominant failed outcome is target IoU below `0.50`, not duplicate-copy
rejection.

## Resolved Next Experiment Direction: Candidate-Field Cardinality Tomography

Date: 2026-06-03

Scope:

`none-yet`; this is a grill-me decision record for the next diagnostic phase,
not a measured result.

Decision:

The next high-value diagnostic phase should first test the model's
desc-conditioned candidate field before launching painting, ordering, or
training interventions.  The working phase name is
`candidate_field_cardinality_tomography`.

The phase should combine these probes into one source-of-truth experiment:

| Probe | Question |
| --- | --- |
| `cluster_cardinality` | At `pre_x1`, does the x1 coordinate distribution expose at least as many distinct peaks as same-desc GT instances? |
| `x1_distribution_peak_count` | Do top-k x1 peaks cover the GT x1 neighborhoods for all same-desc instances, including rollout FNs? |
| `instance_basin_attraction` | If each GT same-desc x1 is forced, do subsequent y1/x2/y2 tokens collapse to distinct GT boxes? |
| `residual_row_logprob_audit` | For a fixed prefix, are multiple residual GT rows high-probability alternatives, or are all residual rows below EOS/stop? |

Rationale:

The previous FN-rescue phases already show that many FNs are recoverable under
forced desc-first continuation, that correct x1 strongly affects instance
binding, and that target-positive attention is not sufficient for coordinate
binding.  The unresolved fork is whether same-desc crowded failures come from
a coarse candidate field, weak coordinate-chain readout, or coverage/stop
policy.  Measuring candidate-field cardinality first determines which later
intervention has research meaning.

Consequence:

Painting and prefix-coverage interventions become Phase B.  Sorted/random
ordering and teacher-trajectory compatibility become Phase C.  These phases
should not be interpreted until Phase A establishes whether the model exposes
enough same-desc instance modes under desc-conditioned pre-coordinate states.

Evidence handles:

- Checkpoint:
  `outputs/stage1_2b/recursive_detection_ce_latest/compact_full_et_rmp_ce_support2_bsz16_4epoch_tokenrows_v2/compact-full-et-rmp-ce-support2-bsz16-4epoch-tokenrows-v2/v0-20260504-071356/checkpoint-3664`
- Current FN-rescue root:
  `outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_continuation`
- Current Phase-5B coord-slot root:
  `outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_desc_x1_phase5_logit_binding_coordslot`
