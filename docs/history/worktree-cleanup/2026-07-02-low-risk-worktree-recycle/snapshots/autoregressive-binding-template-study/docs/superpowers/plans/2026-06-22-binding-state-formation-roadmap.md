# Binding-State Formation Roadmap

> **For agentic workers:** Use `superpowers:subagent-driven-development` for
> independent implementation slices. Keep edits scoped, run focused tests, and
> record every model-backed probe with exact artifact roots and scope labels.

**Goal:** Trace where the post-box/pre-x1 object-binding or cursor state forms,
how it becomes a coordinate-basin or next-row routing state, and why descriptor
repair, current-box repair, and next-row routing separate in the current
checkpoint-928 selected cases.

**Current base:** The branch already contains transported-readout projection,
path-averaged projection, emitted-basin labels, pre-x1 row materialization,
coordinate score traces, continuation taxonomy, and batch/contrast reducers.
Do not rebuild those surfaces unless verification finds a concrete bug.

**Main evidence anchor:**

```text
case_id: post_box_boundary_ec16feebe5d087b0
image_id: 12670
source_line_idx: 123
target desc: backpack
target x1: 420
wrong x1 family: 874,862,858,893,426,387,358
```

**Current correction, 2026-06-22 boundary-gate pass:**

`baseline_inertia` is not a clean pre-x1 onset row family. The representative
rows carry a receiver-x1 label, but the realized `assistant_prefix_text` already
contains the target object's four coordinate tokens and is missing only
`<|box_end|>`. Treat it as a completed-box/wrapper routing basin, not as direct
evidence that the model cannot perceive the object x1.

The layer-24 output-embedding direction patch over train/val destination
families confirms this distinction:

```text
artifact:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_direction_patch/v31_boundary_gate_train_val_layer24_merged

scope:
1836 patch rows, error_count=0
baseline_inertia/off_basin_escape/donor_slot_capture
train + val
directions: coord_mean_minus_box_end, coord_target_minus_box_end,
            coord_mean_minus_wrapper_mean
strengths: 1,2,4,8,16,32

baseline_inertia train:
24 states, no patched coord top1, no rank<=10, no dist<=16

baseline_inertia val:
25 states, 3 coord top1 states already present before patch, no rank<=10,
no dist<=16
```

Ordinary scored `pre_x1` rows remain coordinate-mode at broad scale:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_readout_reducer/v3_v28_per256_train_val_token_mode_atlas

pre_x1 train: coord top1 1.0, coord mass 0.9978, rank<=10 0.3151
pre_x1 val:   coord top1 1.0, coord mass 0.9980, rank<=10 0.2617
```

Roadmap consequence: split the next work into two tracks. Clean pre-x1 onset
rows should drive coordinate-basin ownership probes at layers 17-21. Completed
box states should drive wrapper/router probes over `<|box_end|>`,
`<|object_ref_start|>`, `<|object_ref_end|>`, `<|im_end|>`, and next-object or
termination context.

**2026-06-23 ownership-transition update:** A post-hoc reducer over the
clean pre-x1 compatibility scans now separates mutually exclusive ownership
classes from overlapping target/donor/origin flags. Compatible donor rows over
595 train/val patch rows show a soft mixed state at layer 17 and a donor-basin
snap by layer 18. True target-geometry ownership is rare and mostly already
visible at layer 17 or in same-image/near-target cases. Simple-control donors
are not repair controls; they collapse almost deterministically to the origin
basin from layer 17 onward. This reinforces the current split: coordinate-onset
work should localize object/slot/value ownership formation at layers 17-18,
while completed-box work should remain a separate router-boundary track.

New reducer artifacts:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_ownership_transition/v1_v24_clean_prex1_compatible_layers17_21
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_ownership_transition/v2_v28_clean_prex1_simple_control_layers17_21
```

**2026-06-23 component-tomography update:** The first model-backed
component-output pass over the selected ownership panel covers 85 train/val
sequences and 340 self-attention/MLP patch rows at layers 17 and 18. The
strongest localization is layer-17 self-attention: it exposes the compatible
basin identity with high gain, including donor/origin in bad rows and
target-like geometry only when target and donor are already compatible. Layer
18 self-attention sharply reduces donor-like landing and becomes mixed
target/worse/resolver behavior. MLP is more consistent with rank/value shaping
or basin stabilization than first exposure. Next attention/value work should be
role-stratified around layer-17 source heads and layer-18 resolver states,
while the next population pass should over-sample trained-sequence failures and
compare them to held-out analogs by failure locus instead of returning to the
original semantic pair.

Component tomography artifacts:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_ownership_component_pairs/v1_v24_v28_panel_layers17_18_selfattn_mlp
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_ownership_component_patch/v1_component_layers17_18_panel_merged
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_ownership_component_reducer/v1_component_layers17_18_panel_merged
```

**2026-06-23 per256 failure-mode update:** A post-hoc reducer over the per256
train/val launch-filter guided-delta artifacts shows that the broad pre-x1
failure cohort is not one coordinate-basin family. Out of 492 layer-input /
self-attention / MLP guided-delta rows, 348 are coordinate-mode and 144 are
already wrapper-mode at the staged prefix; 138 of those wrapper rows have
`<|box_end|>` as top1. This includes trained rows (`train
premature_boundary_mode=78`), so trained-sequence failures include both wrong
coordinate-basin selection and premature boundary/router-mode switching. The
recommended 118-row panel should now be split before further GPU work:
`strict_clean_local_repair` versus `intrusive_slot_transition` for layer-17
attention/value tomography, `premature_boundary_mode` for boundary/router
probes, and small-object/tail rows for locality, smoothness, or delayed
evidence checks.

Failure-mode reducer artifacts:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_failure_mode_reducer/v1_v24_v25_v26_per256_train_val_failure_modes
progress/diagnostics/2026-06-23_per256_staged_failure_mode_findings.md
```

**2026-06-23 layer/site failure-mode scan update:** A bounded model-backed
follow-up over the 118-row recommended panel scanned layer-17/layer-18
self-attention and MLP sites with the same `donor_minus_previous_slot` handle.
It completed 472 patch rows with 0 errors. The main finding is that
premature-boundary rows are not clean coordinate-basin repair targets:
`strict_clean_local_repair_rate=0.0` and `intrusive_transition_rate=1.0` at
all layer/site combinations, even when layer-18 self-attention produces a large
rank swing (`mean_rank_delta=-5545.64`). By contrast, coordinate-mode
`x1_onset_anchor_failure` rows show the strongest clean/local repair at layer
17 (`mlp=0.2941`, `self_attn=0.2549`), weaker repair at layer 18, and a tail
subset where layer-18 self-attention may act as a delayed-evidence resolver.
The next GPU work should split three ways: boundary/router interventions for
wrapper-mode rows, layer-17 attention/value localization for coordinate-mode
x1 onset rows, and delayed-evidence/locality checks for tail rows.

Layer/site scan artifacts:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_guided_delta_patch/v29_failure_mode_panel_layers17_18_selfattn_mlp_plan
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_guided_delta_patch/v30_failure_mode_panel_layers17_18_selfattn_mlp_shard0of3_gpu0
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_guided_delta_patch/v30_failure_mode_panel_layers17_18_selfattn_mlp_shard1of3_gpu1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_guided_delta_patch/v30_failure_mode_panel_layers17_18_selfattn_mlp_shard2of3_gpu7
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_failure_mode_reducer/v3_v30_layers17_18_selfattn_mlp_layer_aware
```

Progress note:

```text
progress/diagnostics/2026-06-23_clean_prex1_ownership_transition_findings.md
```

**Non-goals for this cycle:**

```text
training intervention
broad low-rank SVD over mixed regimes
checkpoint/template transfer claims
more descriptor-only gradient variants
mixed full-output/layer-input scheduler without a stateful token-role hook
```

## Task 1: Formation-Position Rows

- [x] Build a concise row-builder for meaningful token positions around the
      positive pre-x1 handle.
- [x] Include at least: descriptor onset, descriptor end, `object_ref_end`,
      `box_start`, pre-x1, post-x1, box close, and next-object onset.
- [x] Preserve `post_box_boundary_case_id`, `source_line_idx`, `image_id`,
      `object_idx`, target bbox, completed/emitted boxes, and variant role.
- [x] Make row output deterministic and testable without model loading.
- [x] Write tests for position selection, missing-token handling, invalid-case
      filtering, provenance fields, and suffix-replay invariants.

Acceptance:

```text
formation_position_rows.jsonl
formation_position_summary.json
tests pass for row construction and filtering
```

## Task 2: Suffix-Replay Formation Probe

- [x] Add a formation-specific donor/receiver pair selector for hard pre-x1
      receivers and explicit baseline/self/donor controls. This is a pair
      selection surface only; it does not run model perturbations.
- [x] Add a model-backed explicit-token layer-input patch probe for the pre-x1
      next-token logits. This is causal for the immediate x1 decision but does
      not yet claim suffix replay or generated-span repair.
- [x] Do not patch a stale cached early hidden state without propagation for the
      current claim. The implemented probe patches an explicit token index in a
      fresh receiver forward pass and scores the receiver's next-token logits.
- [x] Start the model-backed intervention on the broader v8 hard train/val
      receiver panel before returning to the old backpack anchor. Keep the
      backpack case as an interpretability check, not the center of gravity.
- [x] Record active-to-baseline donor directions with baseline and self-noop
      controls. Baseline-to-active reverse directions remain pending.
- [x] Score target rank/prob, coord top1 bin, rank deltas, probability deltas,
      patch norm, and explicit token-index/template-tail metadata. Wrong-family
      ranking and parse health remain pending.
- [ ] Promote the immediate x1 patch probe into true suffix replay only after a
      causal handle is stable enough to justify generation.

Acceptance:

```text
formation_replay_patch_rows.jsonl
formation_replay_patch_summary.json
artifact note with tiny/smoke scope
```

Current pair-selection artifacts:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v1_v8_prex1_hard_train_val_pair_selection
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v2_v8_prex1_hard_train_val_pair_selection_donor_rank10
```

Recommended next intervention panel is `v2`: 24 hard receivers balanced across
train and val, 144 pair/control rows, and same-regime/same-desc donors for a
subset of repeated-class, termination-tail, duplicate-nearby, and small-object
cases. The `v1` exact-only panel is a stricter control but is too dominated by
simple-control donors for mechanism localization.

Current explicit-token layer-input patch artifacts:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v3_explicit_token_layer_input_smoke_limit6
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v4_explicit_token_layer_input_same_regime_donors
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v6_explicit_token_layer24_same_regime_donors
```

The first causal result: explicit-token donor layer-input patches move hard
receiver pre-x1 logits without relying on last-token patching. The same-regime
panel improves 8/12 rows at layer 27 and 9/12 at layer 24, but also worsens a
minority, consistent with donor-basin steering rather than generic quality
injection.

## Task 3: Family-Aware Coordinate Basin Reducer

- [x] Add a patch-result basin taxonomy reducer so target-rank improvement is
      separated from true/near target repair, donor-basin steering,
      origin-basin collapse, and rank-only misleading movement.
- [ ] Add a stricter reducer that scores target versus a named wrong coordinate
      family, not only target versus one bin.
- [ ] For the backpack x1 smoke, use target `420` and wrong family
      `874,862,858,893,426,387,358`.
- [ ] Report family substitution, exact-bin recovery, near-target overshoot,
      and target loss relative to paired baseline.
- [ ] Keep row-level outputs so later probes can join against continuation
      taxonomy.

Acceptance:

```text
coordinate_family_basin_rows.jsonl
coordinate_family_basin_summary.json
coordinate_family_basin_summary.md
formation_replay_patch_basin_rows.jsonl
formation_replay_patch_basin_summary.json
focused tests for family ranking and substitution labels
```

Current patch-result taxonomy artifacts:

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

Current taxonomy verdict: layer-input comparable patches are mixed
target-repair and donor-steering, not uniform repair. Simple-control patches
are all labeled `origin_basin_collapse`, confirming that generic easy-coordinate
donors are a misleading repair control. Isolated self-attention and MLP patches
produce weaker target-rank movement and no near-target repair on this panel.

## Task 4: Path-Mediation Probe

- [x] Use the positive late `layer_input` carrier as the handle.
- [x] Compare first-order component-output patch sites:
      `layer_input`, `self_attn`, and `mlp` at the explicit pre-x1 token.
- [x] Compare true mediation/clamp modes:
      `patch layer_input + recompute attention/MLP`,
      `patch layer_input + clamp attention`,
      `patch layer_input + clamp MLP`,
      and `patch layer_input + clamp both`.
- [ ] Treat isolated `self_attn` and `mlp` deltas as controls, not presumed
      origin writers.
- [x] Score the first clamp pass by coordinate-basin taxonomy, not just
      descriptor exactness or average target-rank movement.
- [x] Run compatibility-selected layer scans for repair and donor-steer rows.
      Coarse layers 12/16/20/22/24/26/27 located the ownership transition
      between layers 16 and 20; fine layers 17-21 showed donor-basin snap
      around layer 18 and target/near-target repair around layers 19-21.
- [x] Run component-output layer scans over the same transition rows. Isolated
      self-attention can expose donor anchors early, while isolated MLP more
      often moves rank or stabilizes late repair; neither component alone is as
      strong as full residual layer-input patching.
- [ ] Score span-aware continuation only after the clamp/path handle is stable
      enough to justify generation budget.

Acceptance:

```text
path_mediation_rows.jsonl
path_mediation_summary.json
path_mediation_summary.md
tests for clamp-mode selection and recorded patch counters
```

Current first-order component artifacts:

```text
layer_input:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v6_explicit_token_layer24_same_regime_donors

self_attn:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v9_explicit_token_layer24_self_attn_same_regime_donors

mlp:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v10_explicit_token_layer24_mlp_same_regime_donors
```

Current verdict: both component-output patches move x1 basin logits, MLP more
than self-attention on average, but full layer-input patching remains strongest.
This supports a residual-stream basin state entering late layers and being
transformed/amplified by subpaths, rather than a pure attention-only writer.

Diversified train/val component artifacts:

```text
diverse pair selection:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v11_v8_prex1_diverse_train_val_pair_selection_layer24

layer_input comparable donors:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v12_diverse_train_val_layer24_comparable_donors

simple-control donors:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v13_diverse_train_val_layer24_simple_control_donors

self_attn comparable donors:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v16_diverse_train_val_layer24_self_attn_comparable_donors

mlp comparable donors:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v17_diverse_train_val_layer24_mlp_comparable_donors
```

Diversified verdict: the train/val hard receiver panel reproduces the basin
steering mechanism outside the original anchor. Comparable donors improve rank
for all 9 layer-input rows, across train and val. Simple-control donors are not
a valid repair control by themselves: they split 35/32 improved/worsened and
mostly force the top-1 coordinate to the donor origin basin. Component-output
patches remain much weaker than full layer input on the same 9 rows
(`self_attn` mean rank delta -32.33, `mlp` -41.56, layer input -231.0), with
MLP better on top-1 distance in this panel.

Current clamp-mediation artifacts:

```text
clamp self_attn pair selection:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v18_v8_prex1_diverse_train_val_layer24_clamp_self_attn_pairs

clamp mlp pair selection:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v19_v8_prex1_diverse_train_val_layer24_clamp_mlp_pairs

clamp self_attn+mlp pair selection:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v20_v8_prex1_diverse_train_val_layer24_clamp_self_attn_mlp_pairs

layer_input + clamp self_attn:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v21_diverse_train_val_layer24_layer_input_clamp_self_attn_comparable_donors

layer_input + clamp mlp:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v22_diverse_train_val_layer24_layer_input_clamp_mlp_comparable_donors

layer_input + clamp self_attn+mlp:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch/v23_diverse_train_val_layer24_layer_input_clamp_self_attn_mlp_comparable_donors

taxonomy for v21:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_basin_taxonomy/v5_v21_layer_input_clamp_self_attn_comparable_donors

taxonomy for v22:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_basin_taxonomy/v6_v22_layer_input_clamp_mlp_comparable_donors

taxonomy for v23:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_basin_taxonomy/v7_v23_layer_input_clamp_self_attn_mlp_comparable_donors
```

Clamp verdict: clamping self-attention barely reduces the layer-input effect
(`mean_coord_rank_delta` -231.00 to -215.11) and preserves all 3 near-target
repairs, so attention output is not the dominant necessary mediator on this
panel. Clamping MLP reduces the effect more strongly (-138.22), drops
near-target repair from 3 to 2, and worsens mean top-1 target distance. Clamping
both reduces the effect further (-114.44), leaves only 1 near-target repair,
and introduces origin/worsening labels. Donor-basin steering persists under all
clamps, which means the layer-24 input residual state already carries a basin
cursor before these component outputs; MLP appears more necessary for converting
that cursor into target-aligned repair, while self-attention is less necessary
on this selected panel.

## Task 5: Span-Aware Continuation Gate

- [x] Run continuation only after a formation or mediation patch moves a
      coordinate-family readout enough to justify generation.
- [x] Score current/open-box and next/generated-object spans separately.
- [x] Include IoU bands beyond permissive overlap, including `>=0.3`,
      `>=0.5`, and `>=0.75`; require `complete_valid` for strict success.
- [ ] Keep `first_step` and `all_steps` separate.
- [x] Compare each broad free pre-x1 continuation to a matching forced-target
      x1 continuation on the same state. Patch-specific baseline contrast
      remains pending.
- [x] Add a deterministic sharded formation continuation runner.
- [x] Run a compatibility-selected nonrecovered-tail continuation pass. Result:
      target-x1 forcing improves span IoU more than donor or patched x1, but
      the source has only 6 unique receiver states and should not carry broad
      dataset conclusions.
- [x] Run a broad free-vs-force-target continuation pass over 178 unique v9
      train/val pre-x1 failures. Result: force_target_x1 raises
      complete-valid IoU>=0.5 from 18/178 to 82/178, with similar train/val
      rates and a strong small-object exception.
- [x] Add forced post-x1, forced x1+y1, forced x1+y1+x2, and forced full-box
      closure modes to separate x1 onset, extent completion, final-coordinate,
      and `<|box_end|>` routing failures. Result on the 178-state broad v9
      pre-x1 failure panel: complete-valid IoU>=0.5 rose from 18/178 free to
      82/178 with x1 forced, 94/178 with x1+y1 forced, 137/178 with
      x1+y1+x2 forced, and 164/178 with full target box forced. The remaining
      full-box failures are closure/router failures emitting
      `<|object_ref_end|>` or `<|object_ref_start|>` instead of `<|box_end|>`.

Acceptance:

```text
continuation_taxonomy rows and contrast rows for the new probe
progress note with gains, losses, malformed rows, and wrong-family substitutions
```

## Task 6: Regime Label Bank

- [ ] Label cases before aggregation:
      layer-input steerable, descriptor-only repair, wrong-family substitution,
      local-simplex stubbornness, adapter-amplified cliff, weak-evidence/FN, and
      termination/empty.
- [ ] Only after labels stabilize, test low-rank or cross-case transfer within a
      regime.

Acceptance:

```text
regime_label_rows.jsonl
regime_label_summary.md
explicit gate on whether low-rank/cross-case transfer is now meaningful
```

## Task 7: Train-Vs-Val Case Expansion

- [x] Build a candidate bank from both bbox_len12000 splits:
      `/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/train.coord.jsonl`
      and
      `/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/val.coord.jsonl`.
- [x] Select representative GT-structure and control rows beyond the current
      person/backpack pair, including repeated-class, crowded, small-object,
      termination-tail, nearby same-desc duplicate-basin, and simple-control
      cases. Model failure/control labels remain pending rollout attachment.
- [x] Attach existing val200 rollout labels to the first-200 val candidate
      rows so real false-negative/control rows can be selected before
      hidden-state probes. Matched train rollout/probe labels remain pending.
- [x] Select a compact matched/unmatched probe panel from the labeled val
      candidates, including same-image contrasts and regime-specific
      false-negative/control roles.
- [x] Materialize teacher-forced formation-position rows for the rollout-labeled
      val probe panel. This is the first hidden-state probe panel, not the final
      train-vs-val cohort.
- [x] Materialize broader formation-position rows for train and val cohorts
      before broad GPU suffix-replay probes.
- [x] Compare whether trained-sequence failures and unseen-val failures share
      the same pre-x1 row-state / coordinate-family mechanism at teacher-forced
      readout level. Current v8 evidence weakens a pure memorization/generalization
      split: trained rows can fail severely at pre-x1, while the same rows are
      much better at post-x1.
- [x] Build the next train-vs-val failure panel deliberately, not as a side
      effect of the backpack anchor: choose trained-sequence rows that still
      show weak or wrong pre-x1 coordinate basins, choose unseen-val analogs
      matched by motif, and compare whether the same patch sites repair, steer,
      or worsen both groups.

Acceptance:

```text
train_val_formation_candidate_bank.jsonl
train_val_formation_candidate_bank_summary.md
formation rows for train and val cohorts
explicit note on shared vs split train/val mechanisms
```

## Task 8: Broader Dataset-Row Mechanism Expansion

- [x] Stop treating the current person/backpack case as the default center of
      gravity. Keep it as an interpretable anchor, but sample additional rows
      across desc categories, object counts, same-desc repetition, spatial
      scales, crowded images, and tail positions.
- [x] Expand the row bank before spending more generation budget: include
      multiple motifs where the model should have enough supervision signal
      but still fails locally, especially trained rows with severe pre-x1
      coordinate-rank errors and val rows with matched object/scene structure.
- [x] Include train rows even when rollout labels are unavailable. Teacher-forced
      formation readouts can still ask whether a trained sequence has a strong
      visual-language binding state at the target object.
- [x] Add a model-backed formation-prefix readout probe and smoke it on train
      and val teacher-forced rows.
- [x] Run and merge an align-fixed broad sharded readout over 384 train/val
      formation rows for descriptor_onset, box_start, pre_x1, and
      next_object_onset.
- [x] Compare train and val rows at the same token positions:
      descriptor_onset, descriptor_end, box_start, pre_x1, box_close, and
      next_object_onset.
- [x] Separate three failure explanations at teacher-forced readout level:
      visual non-perception, language-side guidance/cursor fragility, and
      coordinate-basin competition under local context.
      The current evidence does not prove visual non-perception. It shows that
      type/schema and most structural slots are stable, while x1 basin entry is
      weak even under GT prefix and post-x1 is far better once x1 is committed.
      This favors coordinate-basin onset/cursor competition as the next causal
      target.
- [ ] Let the next panel grow dynamically when a path looks unusually
      explanatory. The mission is the final mechanism picture, not strict
      adherence to a preselected panel size.
- [ ] Next expansion should deliberately over-sample trained-sequence failures,
      then compare them to unseen-val analogs by motif. A trained row that still
      fails under teacher-forced GT prefix is especially valuable because it
      separates visual exposure from local autoregressive coordinate-basin
      selection. Do not require rollout labels before using train rows; use
      readout rank, top-1 basin, motif, and object position as sufficient
      selection signals for hidden-state probes.
- [x] Run the first expanded train-failure versus val-analog layer-input panel.
      Strict comparable donors (`donor_max_coord_rank=10`) produced 23
      intervention rows; looser donors (`donor_max_coord_rank=25`) produced 34.
      Both panels showed repair, donor-basin steering, and worsening in both
      train and val rows. This confirms that the current mechanism is not
      explained by unseen-val exposure alone.
- [ ] The next expansion should be compatibility-aware rather than simply
      larger: explicitly summarize same-image/same-split/same-desc controls,
      motif match, donor coordinate rank, donor-target distance, receiver
      baseline basin, and whether the patched top-1 lands near target, donor,
      origin, or escapes.
- [x] Add the compatibility reducer and run it on the expanded strict/loose
      layer-input panels. Result: repair is concentrated in near target-donor
      distance and a small number of compatible cases; far target-donor distance
      is dominated by donor-basin steering even when donors are rank <=10.
- [x] Rerun component clamps on the expanded strict donor panel. Result:
      self-attention clamp is again mild, while MLP/both-component clamps
      redistribute basin ownership rather than globally suppressing all repair.
      Same-split compatible rows improve on mean rank under MLP/both clamps,
      while cross-split compatible rows worsen, pointing to a compatibility-
      dependent ownership resolver rather than a pure writer in one component.
- [ ] Use compatibility-selected rows for the next causal step:
      near-target/target-rank repair rows should go to span-aware continuation;
      far-distance donor-basin steering rows should go to a layer scan for
      where the target/donor ownership decision forms.
- [x] Add the compatibility-selected layer-scan pair builder and run the first
      coarse/fine transition scans over both repair and donor-steer rows.
      Result: the decisive coordinate-basin ownership transition is concentrated
      around layers 18-20, not layer 24.
- [x] Run the matching self-attention and MLP component-output layer scans at
      layers 17-21. Result: self-attention exposes donor anchors, MLP shapes
      and stabilizes rank/ownership, and full residual integration is required
      for stable basin commitment.
- [x] Broaden the next row-mining pass beyond the current selected motifs.
      Deliberately over-sample trained-sequence failures from the training
      dataset, then match them to unseen-val analogs by motif, object order,
      repetition class, scale, crowding, coordinate distance, and tail/stop
      position. Avoid letting person/backpack or bird-only cases become the
      hidden default mechanism. Current v9 evidence covers 288 train/val
      objects and 2304 formation-position rows across crowded, repeated-class,
      duplicate-basin, small-object, simple-control, and termination-tail
      regimes.
- [x] Compare train and val failures as paired mechanism questions, not just
      split-level aggregates: for each motif, ask whether the trained row has
      visual/semantic evidence but still enters the wrong coordinate basin, and
      whether the unseen-val analog fails at the same transition layer, with the
      same component path, or only under weaker local context. Current result:
      trained rows fail at pre-x1 too, and most trained failures recover after
      x1 is supplied, weakening a pure unseen-val or visual-nonperception
      explanation.
- [x] Add a post-hoc formation-failure panel reducer that selects trained
      pre-x1 coordinate-basin failures and matched unseen-val failure analogs
      from model readout rows. The reducer uses readout evidence rather than
      hand-picked person/backpack or bird-only anchors.
- [x] Run the first failure-panel causal patch with receivers restricted to the
      selected train/val failure panel and donors drawn from the broader v8
      pool. Result: layer-20 donor patching is mostly not repair; simple-control
      donors collapse to origin, compatible donors often steer into donor
      basin, and trained failures remain fragile despite being from trained
      sequences.
- [x] Build and run a broader per-regime-per-split=24 train/val readout panel
      over 288 objects and 2304 formation-position rows. Result: train and val
      both contain many pre-x1 failures, and most train failures recover strongly
      by post-x1, supporting x1-onset cursor fragility over raw visual
      non-perception.
- [x] For the next causal pass, use the v9/v3 failure panel but split donor
      roles deliberately: compatible-donor-only repair/steer probes,
      simple-control origin-basin probes, and separate recovery-vs-nonrecovery
      hidden-state comparisons. Result: compatible donors snap from mixed states
      at layer 17 into donor-basin ownership by layer 18, while simple-control
      donors induce near-total origin-basin collapse from layer 17 and saturate
      by layers 19-21.
- [x] Add and run a post-hoc recovery-vs-nonrecovery hidden-state reducer over
      v9. Result: 161/178 broad pre-x1 failures recover by post_x1, including
      75/82 trained failures and 86/96 val failures. The 17 nonrecovered cases
      concentrate in termination_tail and small_object regimes and are better
      candidates for genuinely weak visual evidence, delayed distributed
      evidence, or next-object/stop routing conflict.
- [x] Build the next compact nonrecovered-tail panel across train and val rows,
      with termination_tail, small_object, crowded, and duplicate_basin_nearby
      examples. Do not let the sparse tail disappear inside aggregate recovered
      behavior. Result: 17 nonrecovered receivers yielded an 85-row layer-18
      seed panel. Corrected seed patch completed with 0 errors; generic
      simple-control donors again collapsed to origin, while compatible donors
      mostly donor-captured or produced only rare local repair.
- [x] Run a first nonrecovered-tail transition-layer scan. Result: 10 seed
      source pairs expanded to 50 rows over layers 17-21. Layer 17 is the
      softest same-split window with mostly rank-only repair, but layers 18-21
      are dominated by donor-basin steering except for two train
      termination-tail rows whose donor x1 is already near the target x1.
- [x] Run span-aware continuation separately for same-image/local target repairs
      and donor-basin capture rows. The next causal question is whether a true
      target-basin repair changes the generated object span, not only the
      immediate coordinate rank. Result: target-x1 forcing improves generated
      spans more than donor/patched x1, while donor/patched x1 mostly preserves
      donor or non-target basins. This confirms that next-token rank repair is
      not automatically whole-span repair.
- [x] Broaden span-aware continuation beyond the compatibility rows by joining
      178 unique v9 pre-x1 failure states back to readout prefixes. Result:
      true x1 forcing rescues many train and val failures, but small objects
      and post_x1-nonrecovered cases remain hard.
- [ ] For nonrecovered-tail rows, split visual-side weakness from
      language/context guidance by forcing or patching x1, y1, x2, y2,
      descriptor, and termination context separately. The first staged forcing
      pass shows many post_x1-nonrecovered rows still close when the full box is
      supplied, so visual weakness should not be inferred until coordinate-slot
      and closure-slot interventions are separated.
- [x] Split the next continuation pass into explicit families:
      guidance-rescuable x1 onset failures, small-object/weak-evidence or
      extent failures, and span-closure or next-object routing failures. Result:
      61/178 states are x1-onset rescues, 37/178 require x2 forcing, 26/178
      require full-box forcing, and 14/178 still fail after full-box forcing.
      Small objects are dominated by x2/y2 extent failures; crowded rows retain
      the strongest closure/router component.
- [x] Broaden the staged-continuation panel beyond the current 178 readout-
      selected states. Result: the per48 bbox_len12000 bank covers 576 objects
      and 328 pre-x1 failures, including 157 train and 171 val failures. This
      confirms that trained-sequence failures under teacher-forced GT prefix
      are common, not a person/backpack artifact.
- [x] Add a selector/reducer that groups broader candidates by staged failure
      locus: x1-onset rescue, y1/anchor rescue, x2/extent rescue, y2/final-
      extent rescue, full-box closure/router failure, and no-rescue. Result on
      the 328-state per48 panel: x1_onset_rescue 128, y1_anchor_rescue 38,
      x2_extent_rescue 67, y2_final_extent_rescue 48, full_box_closure_router
      27, already_successful 20. Train and val have similar locus mixtures.
- [ ] Fix or route around descriptor-slot readout for multi-token class names
      before using descriptor_onset/descriptor_end aggregate claims from the
      per48 artifact. Coordinate and schema positions are complete; descriptor
      positions are partial because `sheep`, `cell phone`, `traffic light`, and
      similar targets encode to multiple tokens under the current scorer.
- [x] Build locus-specific follow-up panels from the per48 reducer: small-object
      x2/y2 extent cases, crowded closure/router cases, repeated-class x1/y1
      onset-anchor cases, and post_x1-nonrecovered rows that nevertheless close
      under full-box forcing. Result: 89 panel instances / 76 unique receiver
      cases, balanced across train and val where possible, with four roles:
      small_object_extent, crowded_closure_router, repeated_class_onset_anchor,
      and post_x1_nonrecovered_full_box_closable.
- [x] Build staged-slot readout rows for the mechanism panels. Result: 445
      deterministic readout states, five per panel instance: staged_pre_x1,
      staged_after_x1, staged_after_x1_y1, staged_after_x1_y1_x2, and
      staged_after_full_box. This directly addresses the user's broader-row
      hint: the next probe is a train-vs-val, regime-vs-regime slot-locus
      analysis, not a person/backpack pair study.
- [x] Run the staged-slot GPU readout once GPUs are free, then reduce by split,
      panel role, candidate regime, formation position, target rank/probability,
      coordinate top1 distance, and box_end-vs-router token competition. Result:
      445 rows / 1780 hidden rows, 0 errors, launched as two shards on the
      lowest-memory GPUs while the 8-GPU prefix-denoising SFT job was active.
      The reducer shows pre_x1 is almost always off-basin, x1 guidance often
      restores local coordinate readiness, small-object extent is mostly
      near-local but rank-competitive, and crowded closure/router failures are
      schema-boundary failures even after the full box is supplied.
- [ ] After staged-slot readout, choose hidden-state and attention interventions
      by locus: x1-onset rows get pre_x1 basin-patch tests; small-object extent
      rows get x2/y2 visual-locality and coordinate-smoothness tests; crowded
      closure/router rows get box_end / next-object router tests.
- [x] Reduce staged-slot hidden logit-lens rows by layer. Result: coordinate and
      box_end evidence is mostly absent from layers 0/12, becomes linearly
      visible around layer 24 under guided prefixes, and is sharpened at final
      logits. The crowded closure/router panel remains box_end-negative even at
      layer 24 and final logits.
- [x] Bridge layer-24 hidden logit-lens rows to final staged-slot logits.
      Result: the final output surface often sharpens or creates coordinate
      rank readiness (81 hidden-not-ready -> final-ready rows versus 19
      hidden-ready -> final-lost rows), while all 17 crowded closure/router
      failures are already hidden-router -> final-router after the full target
      box. This weakens a surface-only explanation for closure failures and
      sharpens the coordinate-locality versus exact-rank/smoothness question for
      small objects.
- [ ] Run causal interventions at the layer-24/final transition separately by
      locus and population. Coordinate-basin add/patch probes should use trained
      pre-x1 failures plus held-out analogs, not the old person/backpack pair as
      the sampling prior. Small-object rows should test coordinate-locality
      versus exact-token rank/smoothness. Crowded closure rows should test
      box_end/router steering and continuation, because the surface bridge says
      their missing boundary state is already present at layer 24.
- [ ] Add or run the next population expansion before promoting any causal
      result: include more training-dataset rows that still fail under
      teacher-forced GT prefix, and compare them with held-out val analogs by
      motif, object order, crowding, area, and slot locus. A compact causal
      smoke can be small, but the row selector behind it must remain broader
      than one semantic pair or one visual anecdote.
- [x] Add a population-first causal panel selector over the staged-slot
      layer-24-to-final bridge rows. Result: 193 candidate rows reduced to a
      balanced 48-row launch panel: 12 pre_x1_onset_failure, 12
      small_object_rank_locality_tension, 12 crowded_closure_router, and 12
      final_y2_extent_failure rows, exactly 24 train and 24 val. The selector
      deliberately includes trained-sequence failures under GT-prefix evidence,
      not only held-out failures.
- [ ] Implement the next staged-slot causal intervention runner instead of
      forcing the existing trajectory patch runner onto the staged panel. The
      trajectory runner requires `family`, `source_line_idx`, stop-reason, and
      trajectory-state prefix semantics; using it directly on staged-slot bridge
      rows would blur artifact meaning. The staged runner should consume the
      48-row panel, reconstruct image/prefix/target state from the staged row
      source when needed, and run locus-specific patches:
      pre-x1 coord-basin, small-object coord-locality/rank, crowded
      box_end/router, and final-y2 extent.
- [x] Implement a staged-slot direction-patch runner and smoke it on GPU 1.
      Result: model-loading and image-root fallbacks work for staged-slot rows,
      and layer-24 immediate-token direction patches now emit auditable plan,
      patch-row, summary, and markdown artifacts. This runner is deliberately
      immediate-next-token only; continuation repair remains a separate step.
- [x] Run the first crowded closure/router direction-patch smoke. Result:
      small boundary-direction strengths 2/4/8 do not move rank/top1; large
      strengths 16/32/64/128 flip box_end top1 in 3/8 rows over two train
      crowded states. This supports a router-margin mechanism: box_end can be
      forced by a large boundary direction, but the natural layer-24 state does
      not carry a strong enough boundary decision.
- [x] Run the first pre-x1 coordinate-onset direction-patch contrast on one
      train and one val state. Result: large coord_target_minus_mean and
      target_minus_top1 directions improve rank dramatically, but never reach
      coord_rank<=10 or coord_distance<=16. This separates pre-x1 coordinate
      onset from crowded closure: onset failure is not just a missing
      target-token vocabulary direction; it is a broader coordinate-basin or
      anchor-selection state failure.
- [x] Add staged-slot patched continuation. For crowded closure, test whether
      the large direction-induced <|box_end|> top1 leads to a valid next-object
      boundary/termination continuation or only a one-token flip. For pre-x1
      coordinate onset, prefer donor-state or attention/value patches from
      guided post-x1 states over more output-embedding target-token nudges.
      Result on a tiny train crowded-router smoke: when the layer-24
      box_end-minus-object_ref boundary direction flips the first token to
      `<|box_end|>`, the next unpatched greedy token is the expected
      `<|object_ref_start|>` in 3/3 flipped rows. This strengthens the
      router-margin account for crowded closure rows, while keeping scope
      explicitly tiny and train-only.
- [x] Replicate staged-slot patched continuation on broader train crowded rows
      and matched held-out analogs. Treat the person/backpack anchor as an
      interpretability microscope only; row selection should be population-
      first and should include trained sequences that fail under teacher-forced
      GT prefix. Result on the 12-state crowded closure/router launch panel:
      baseline never emits `<|box_end|>`; strength 128 flips all 12 train/val
      states to `<|box_end|>`; all 12 flipped states then continue unpatched to
      the expected `<|object_ref_start|>` next-object route. Strength 64 flips
      3/12, and all 3 flipped rows route correctly.
- [ ] Mine the next crowded closure/router expansion beyond the 48-row launch
      panel. Over-sample additional training rows that fail at the boundary
      under teacher-forced GT prefixes, then match held-out analogs by object
      order, crowding, class repetition, and remaining-object tail length.
      Test whether the layer-24 boundary direction has the same threshold and
      route coherence outside the selected launch panel. First expansion beyond
      the capped launch panel is complete: all 17 crowded candidates in the
      staged-slot mechanism panel are steerable by the layer-24 boundary
      direction. Minimum flip strengths are 32 for 1 case, 64 for 5 cases, 96
      for 9 cases, and 128 for 2 cases; every flipped row routes correctly to
      `<|object_ref_start|>`. The next expansion must leave the current
      mechanism panel and mine additional train/val crowded rows.
- [x] Build the next coordinate-onset causal probe from guided post-x1 donor
      states, not just target-token output embedding directions. Result: the
      staged-slot guided-delta runner adds donor-hidden minus receiver-hidden
      deltas at layer 24 and scores receiver x1 plus donor-slot attraction.
      Broad sharded run over 88 train/val pre-x1 states and 352 patch rows:
      target rank improves in 0.5966 of rows, but exact target top1 appears in
      only 0.0057 of rows; donor-nearer/slot-intrusion appears in 0.9063 of
      rows. Train and val are similar, so the phenomenon is not explained by a
      simple memorized-training versus unseen-val split. This demotes generic
      donor-state matching and promotes slot-phase disentanglement.
- [ ] Split guided later-slot state into object identity, coordinate value, and
      slot phase before any training intervention. Candidate probes: donor
      minus same-slot controls, later-slot donor minus adjacent-slot donor,
      receiver plus donor value after projecting out a slot-position direction,
      and attention/value component localization for the donor-nearer
      coordinate attractor.
- [ ] Expand coordinate-onset sampling outside the staged mechanism panel:
      mine more training rows that fail under teacher-forced GT prefixes and
      match held-out val analogs by regime, object order, object area, slot
      locus, and remaining-object tail. Keep the person/backpack case and the
      two exact-top1 small-object positives as microscopes only, not as the
      sampling prior.
- [x] Build the next population scaffold as a per96 bbox_len12000 train/val
      candidate bank with image caps and descriptor/class caps before GPU
      promotion. The first readout should skip descriptor aggregate claims until
      multi-token descriptor scoring is fixed; use robust coordinate/schema
      positions first (`pre_x1`, `post_x1`, `box_close`, `next_object_onset`).
      Select separate launch panels for pre-x1 onset and crowded
      closure/router, then run small-object extent/final-y2 as the next
      coordinate-smoothness batch unless the readout shows a stronger train/val
      divergence. Result: the descriptor-capped per96 bank covers 1152
      train/val objects with balanced split/regime counts and no image/desc cap
      violations. The robust-position v17 GPU readout covers 4608 rows with 0
      errors. Pre-x1 remains the broad failure point under GT prefix
      (`target_rank_lte10_rate` 0.3194), while post_x1 recovers locality
      (`coord_distance_lte16_rate` 0.8411). Train is somewhat better than val,
      but trained rows still fail severely at pre_x1, keeping trained-sequence
      failures central to the mechanism search.
- [ ] Build the next train-first failure panel from the v17 per96 readout before
      spending more causal GPU budget. Start with existing tooling at
      `train_seeds_per_regime=16` or `24` and `val_analogs_per_train=2`, inspect
      match quality, then patch analog matching only if the selected analogs are
      too loose. Match quality should be judged by candidate regime, descriptor,
      repeated-desc pressure, object count, remaining-object tail, area,
      coordinate neighborhood, and post_x1 recovery/nonrecovery. First pass
      result: v6 selected 83 train seeds and 166 val analogs from the per96 v17
      readout. All analogs are same-regime and 82/166 are same-regime/same-desc,
      but coordinate-center, object-count/tail, and failure-score deltas remain
      large. Treat v6 as a mining and cohort-splitting panel, not a strict
      paired causal contrast.
- [x] Add a stricter analog selector or launch-cohort filter before paired
      train-vs-val causal claims. The current expanded panel summary now reports
      analog match tiers and paired deltas, making the gap visible. The next
      selector should penalize coordinate-center distance, bbox shape/area
      mismatch, object-count/tail mismatch, and post_x1 recovery/nonrecovery
      mismatch after preserving regime and descriptor when possible. Result:
      the launch filter now writes selected panel rows plus pair-row sidecars.
      v1 is a broad balanced cohort with 40 pairs across the six regimes
      except sparse simple-control coverage; v2 is a same-desc-only cohort with
      29 cleaner identity-binding pairs. Use v1 for regime coverage and router
      smoke tests, v2 for descriptor-controlled binding probes.
- [ ] Split the expanded train-first panel into launch cohorts before
      intervention: pre-x1 coordinate-onset failures, crowded closure/router
      failures, small-object extent/final-coordinate failures, and tail/no-rescue
      rows. Do not merge these into a single donor-patching result because the
      current evidence says they stress different mechanisms. First bridge
      result: v1/v2 launch filters now materialize staged-slot rows. The broad
      v1 cohort readout over 80 train/val cases shows pre_x1 rank<=10 is 0.0
      and distance<=16 is 0.1125, but after target x1 is supplied, distance<=16
      rises to 0.8125 and rank<=10 to 0.675. Full-box closure is 0.95 top1,
      with remaining misses concentrated in crowded boundary states plus one
      simple-control terminal edge case.
- [x] For pre-x1 coordinate-onset rows, run slot-phase disentanglement before
      more generic donor-state deltas. Candidate controls: donor minus same-slot
      control, later-slot donor minus adjacent-slot donor, same-image same-desc
      ownership swaps, and donor value after projecting out a slot-position
      direction. The previous broad guided-delta run showed target-rank
      improvement but high donor-nearer/slot-intrusion, so another unconstrained
      donor-hidden delta would mostly remeasure donor capture. The v19 launch
      readout strengthens this priority: neither layer 24 nor final surface has
      a usable pre_x1 onset state, while post-x1 states are locally ready.
      First population result: the guided-delta runner now supports launch
      panels without `state_key` and compares `raw_donor_minus_receiver` against
      `donor_minus_previous_slot`. The broad v13 cohort covers 80 receiver
      states and 320 patch rows; the same-desc v14 cohort covers 58 receiver
      states and 232 patch rows. Both complete with zero errors. The main
      mechanism result is not repair but disentanglement: previous-slot deltas
      reduce donor capture versus raw donor import (`slot_intrusion_rate` 0.825
      vs 0.975 on v13; 0.802 vs 0.991 on v14) while leaving exact pre-x1 repair
      rare (`target_top1_rate` 0.00625 on v13, 0.0 on v14). Clean rank
      improvements concentrate in `donor_minus_previous_slot /
      staged_after_x1_y1`, especially small-object and duplicate-nearby rows,
      but clean coord-close repair remains only about 3%. Treat this as a weak
      slot-transition handle, not a training-ready correction vector.
- [x] Split previous-slot transition results into clean versus intrusive
      contrast panels before launching attention/value tomography. Result: the
      staged-slot delta contrast selector materializes broad v13 and same-desc
      v14 panels focused on `donor_minus_previous_slot /
      staged_after_x1_y1`. Broad v1 yields 74 contrast rows
      (`clean_transition` 22, `intrusive_transition` 52); same-desc v2 yields
      54 contrast rows (`clean_transition` 19, `intrusive_transition` 35). The
      important refinement is that clean rank-improvement rows are often not
      local coordinate repairs: broad clean rows have mean top-1 coordinate
      distance 284.7, while intrusive rows are closer at 93.2 because they
      import donor basin. Strict clean coord-close rows are only 6 in each
      panel. The next tomography pass should therefore compare
      `clean_transition && slot_phase_clean_coord_close` against
      `intrusive_transition && donor_nearer_than_receiver`, rather than all
      rank-improved clean rows.
- [x] Apply the same previous-slot transition probe to the full v6 train-first
      failure / val-analog panel instead of only the capped 80-row launch
      cohort. Result: 249 receiver states produced 249 model-backed patch rows
      with 0 errors. The expanded panel preserves 83 trained-sequence failures,
      84 same-regime val analogs, and 82 same-regime/same-desc val analogs.
      Target rank improves in 0.7149 of rows, but exact target top1 remains
      0.0080 and slot intrusion remains 0.6386. The uncapped clean/intrusive
      selector yields 68 clean transitions, 159 intrusive transitions, and 23
      strict clean coord-close rows across train and val. This confirms that
      the slot-phase handle generalizes beyond the small launch cohort while
      remaining a diagnostic handle, not a correction vector.
- [x] Add a coordinate-value geometry reducer over the full v6 clean/intrusive
      contrast rows before attention/value tomography. Result: 227 rows reduce
      with 0 invalid geometry rows. The reducer separates patched coordinate
      landings near receiver x1, donor x2, both, baseline, or elsewhere.
      Clean transitions include true receiver-near rows, but 0.1176 are
      coordinate-edge ambiguity candidates where donor x2 is also near receiver
      x1. Intrusive transitions show a stronger donor-slot value capture
      signature: patched top1 lands near donor x2 in 0.5472 of intrusive rows,
      including 0.4444 of wide-object intrusive rows where receiver x1 is far
      from donor x2. This keeps the main mechanism split sharp: small-object
      clean repairs need coordinate-locality/smoothness analysis, while
      donor-far intrusive rows need attention/value or slot-value localization.
- [x] Expand again to a per128 train/val candidate bank and rerun the robust
      formation readout plus previous-slot guided-delta probe. Result: the new
      bank covers 1536 objects, 768 train and 768 val, with 128 per split-
      regime cell and no image/descriptor cap violations. The v20 readout
      covers 6144 robust-position rows with 0 errors and repeats the trained-
      sequence failure result: train pre_x1 target_rank_lte10 is only 0.3542,
      while train post_x1 rises to 0.6849. Small-object pre_x1 remains the
      strongest failure locus, with target_top1 0.0078 in both train and val.
      The v7 failure panel selects 126 trained pre_x1 failure seeds and 252
      val analogs; the v18 guided-delta replication over 378 rows has exact
      target_top1 0.0026, coord_distance_lte16 0.1005, and slot_intrusion
      0.6905. The v7 geometry reducer revises the earlier donor-capture-only
      framing: intrusive rows include donor-slot capture, but also baseline
      inertia and previous/control-slot pull. The next localization panel must
      stratify by landing region.
- [ ] For crowded closure/router rows, replicate the layer-24 boundary-direction
      threshold outside the capped staged mechanism panel. This is the cleanest
      short-term causal handle, but it should not replace the coordinate-onset
      investigation because it is a distinct router-margin mechanism. The v19
      broad launch readout finds 3/4 full-box misses in crowded rows, with
      `<|object_ref_end|>` beating `<|box_end|>`, so the next router pass should
      target these residual crowded boundary states.
- [ ] For small-object and final-extent rows, separate coordinate locality from
      exact-rank/smoothness failure. Bin by target/donor coordinate distance and
      local coordinate mass before interpreting failures as weak visual
      perception.
- [x] Add a post-hoc ownership-transition reducer over clean pre-x1
      compatibility rows. Result: compatible donors over 595 train/val rows are
      mixed at layer 17 but mostly snap into donor-basin ownership by layer 18;
      true target-geometry repair is rare and concentrated in already-compatible
      same-image or near-target cases. Simple-control origin donors collapse to
      origin ownership from layer 17 onward. This is a train/val, multi-regime
      result, not a person/backpack-only observation.
- [x] Use the new ownership-transition rows to choose the next attention/value
      tomography panel. Compare four populations rather than one semantic pair:
      stable target-geometry sequences, layer-18 donor-snap sequences,
      origin-collapse sequences, and worse/escape sequences. Prioritize layers
      17-18 and keep train-sequence failures plus val analogs in every panel.
      Result: the first launch panel spans 85 train/val sequences across five
      roles and five regimes, and the first GPU component-output pass over this
      panel completed with 340 layer-17/layer-18 self-attention/MLP patch rows.
      Layer-17 self-attention is the strongest basin-exposure site, while
      layer-18 self-attention looks like a resolver/redistribution point.
- [ ] Mine another population-first train/val row bank before the next broad
      coordinate-onset causal run. Over-sample trained rows that fail under
      teacher-forced GT prefixes, then match val analogs by regime, descriptor,
      object order, object area, repetition pressure, coordinate neighborhood,
      remaining-object tail, and post_x1 recovery/nonrecovery. Keep the
      person/backpack row as a microscope only.
- [ ] For pre-x1 coordinate onset, avoid another raw donor-state import as the
      main experiment. The previous-slot and ownership-transition evidence says
      raw import mostly remeasures donor capture. Next probes should disentangle
      object identity, coordinate value, and slot phase with donor-minus-
      previous-slot controls, projected value directions, same-image swaps, and
      component/value localization around the layer-17 to layer-18 transition.
- [x] Materialize the first ownership-transition tomography launch panel from
      the compatible-donor and simple-control sequence rows. Result: 240 input
      transition sequences yielded 224 candidates and an 85-row capped panel
      spanning train 44 / val 41, all five major regimes, and five roles:
      `layer18_donor_snap`, `origin_collapse`, `persistent_worse_escape`,
      `rank_only_target_repair`, and `stable_target_geometry`. This is a
      selector-only artifact and should be the next launch handle for
      attention/value tomography, not a result claim by itself.
- [x] Run the first ownership-panel component-output tomography at layers 17
      and 18. Result: 340 GPU-backed self-attention/MLP component patch rows
      over 85 train/val sequences completed with 0 errors. Layer-17
      self-attention is the strongest basin-exposure component
      (`donor_like_rate=0.7882` overall; `0.8889` on layer-18 donor-snap rows;
      `origin_like_rate=0.9333` on origin-collapse rows), while layer-18
      self-attention drops donor-like landing to `0.0941` and behaves more
      like a resolver or redistribution point. MLP mainly shapes rank/value or
      stabilizes origin/donor basins; it is not the primary first-exposure
      path.
- [ ] Expand the next row population by failure locus before the next broad
      causal GPU pass. Mine more training-dataset rows that fail under
      teacher-forced GT prefixes, then match held-out val analogs by locus
      (`pre_x1` onset, small-object extent, crowded closure/router,
      tail/no-rescue), regime, descriptor when available, object order,
      object area, repeated-desc pressure, coordinate neighborhood, and
      post_x1 recovery/nonrecovery. Use existing train/val comparison evidence
      as reference, but keep the new operational probe/tooling concise and
      locus-specific.
- [x] Add a post-hoc staged-slot failure-mode reducer for the per256 launch
      cohort. Result: 492 existing guided-delta rows split into coordinate-mode
      wrong-basin rows (`348`) and wrapper-mode rows (`144`), mostly
      `<|box_end|>` top1 (`138`). This prevents the next causal pass from
      mixing x1 anchor failures with premature boundary/router-mode failures.
      The reducer writes a 118-row recommended panel with explicit roles:
      strict clean local repair, intrusive slot transition, premature boundary
      mode, rank-only clean transition, small-object weak/extent cases, and
      tail delayed/no-rescue cases.
- [x] Run a bounded layer-17/layer-18 self-attention and MLP scan over the
      failure-mode recommended panel. Result: 472 model-backed rows, 0 errors.
      Premature-boundary rows show large rank movement but zero strict clean
      repair at every layer/site, so they should move to router/boundary probes.
      Coordinate-mode x1 onset rows repair best at layer 17, so source/value
      localization should focus there before another pooled layer scan.
- [ ] For the next attention/value tomography, do not collapse all rows into
      one aggregate. Compare layer-17 self-attention source/value patterns for
      stable target-geometry, layer-18 donor-snap, origin-collapse, and
      persistent-worse rows, then inspect layer 18 as a resolver state. The
      main question is how object evidence, coordinate value, and slot phase
      bind or separate before the model emits the object span.
- [ ] Route the next GPU work by locus instead of by a single broad panel:
      boundary/router intervention for wrapper-mode rows; layer-17
      attention/value localization for coordinate-mode x1 onset rows; and
      delayed-evidence/locality checks for tail/no-rescue rows.

Acceptance:

```text
broader train/val teacher-forced formation rows
split-aware readout summary over more categories and positions
explicit decision on whether the next GPU pass stays compact or expands
motif-balanced trained-failure versus val-analog comparison
compatibility-aware donor/receiver reducer or panel summary
expanded strict-panel component-clamp comparison
compatibility-selected continuation and layer-scan candidate sets
span-aware continuation runner and broad free-vs-force-target taxonomy
staged coordinate-forcing continuation taxonomy
per48 staged failure-locus reducer over 328 train/val pre-x1 failures
broader trained-failure versus val-analog transition-layer panel
post-hoc trained-failure versus val-failure selector
broader per24 train/val readout artifact
receiver-filtered pair selector with broad donor pool
v9/v3 compatible-donor and simple-control transition-layer scans
recovery-vs-nonrecovery hidden-state comparison over broad v9 failures
nonrecovered-tail train/val causal panel
nonrecovered-tail layer-18 seed patch and layers17-21 scan
broadened staged-failure-locus candidate bank over more train rows and val analogs
mechanism-panel staged-slot readout row set spanning train and val
GPU staged-slot readout and reducer by slot/locus/split
staged-slot hidden logit-lens reducer by layer/slot/locus
layer-24 to final staged-slot surface bridge
layer-24 locus-specific causal intervention plan
population-first train-failure versus val-analog selector for causal probes
48-row staged-slot causal launch panel
staged-slot direction-patch runner with GPU smoke artifacts
crowded closure/router immediate-token strength scan
pre-x1 train/val coordinate-onset direction-patch contrast
staged-slot patched continuation smoke
population-first replication plan for crowded closure and coordinate onset
crowded closure/router train-val patched continuation replication
all-current-crowded-candidate strength-curve reducer and artifact
staged-slot guided-delta coordinate-onset runner and reducer
all-current pre-x1 guided-delta train-val sharded artifact
launch-panel guided-delta planner for broad/same-desc cohorts without state_key
raw-vs-previous-slot delta-mode causal artifacts v13 and v14
slot-phase disentanglement progress note with train/val and regime breakdowns
clean-vs-intrusive previous-slot transition selector and v1/v2 contrast panels
per96 train-first failure and val-analog population scaffold
expanded train-first failure panel from per96 v17 readout
coordinate-value geometry reducer over full v6 clean/intrusive contrast rows
per128 train/val candidate bank, robust readout, v7 failure panel, guided-delta replication, and destination-aware geometry reducer
strict analog selector or launch-cohort filter with match-quality summary
slot-locus split over expanded train-first panel
coordinate-onset slot-phase disentanglement plan or runner
crowded closure/router expansion outside capped mechanism panel
ownership-transition reducer and clean pre-x1 train/val transition artifacts
population-first train-failure/val-analog tomography panel from transition rows
ownership-transition tomography launch panel over train/val multi-regime roles
ownership-panel component-output tomography over layer-17/layer-18 self-attention and MLP
train-first failure-locus expansion panel for the next causal pass
per256 staged-slot failure-mode reducer and recommended panel
role-stratified layer-17 attention/value tomography plan or artifact
small-object locality versus exact-rank/smoothness reducer
v1 broad launch staged-slot row bridge and v19 GPU readout
v19 staged-slot surface and hidden reducers
```

## Verification Rules

- Run unit tests for every new reducer or row builder.
- For GPU probes, record checkpoint, config, device, selected rows, case ids,
  decode settings, parse counters, output roots, and scope labels.
- Before committing, run:

```bash
git diff --check
python -m pytest <new_or_touched_tests> -q
python -m py_compile <new_or_touched_python_files>
```

## Promotion Rule

Training remains blocked until a replicated, span-aware causal handle exists.
The first plausible training candidates would be localized state-conditioned
pre-x1 basin objectives or small formation-layer adapter/corrector probes. Full
hidden-state MSE and descriptor-gain-only objectives are not justified by the
current evidence.

The next experimental frontier should stay population-first. Single vivid cases
remain useful for interpretation and visualization, but promotion requires that
the same slot-locus signature appears across trained rows and held-out analogs.
Dynamic detours are allowed when a locus-specific path reveals a stronger route
to the final mechanistic picture, especially if it connects coordinate-token
basins, token_embeddings_adapter behavior, and hidden-state routing.

The immediate next causal round should therefore be selected as a cohort, not
as a story: trained rows that fail under the exact trained sequence, held-out
val analogs with the same failure locus, and a small set of interpretable
visual examples. The person/backpack case can still be used as a microscope,
but it should not define the experiment distribution. The raw-vs-previous slot
population runs say the pre-x1 coordinate-onset problem is not solved by
importing later slot states: raw donor import mostly measures donor-basin
capture, and previous-slot transition deltas expose a cleaner local geometry
route only for a minority of rows. The expanded v6 result is now the preferred
population-first source for this question:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_delta_contrast_selector/v4_v16_full_v6_prevslot_afterx1y1_uncapped/staged_slot_delta_contrast_rows.jsonl
```

The next pre-x1 round should purify the `staged_after_x1_y1 -
staged_after_x1` handle using same-slot ownership controls, coordinate-value
projection, or attention/value localization over strict coord-close clean rows
versus donor-nearer intrusive rows. Small-object and duplicate-basin rows now
look especially useful for coordinate-locality versus exact-rank/smoothness
analysis; crowded closure/router rows remain the cleanest separate short-term
router-margin replication target.

The coordinate-value geometry reducer refines that split: do not pool all
strict clean rows as object-binding repairs. Receiver-and-donor-near clean rows
are coordinate-locality/smoothness candidates, while receiver-near and
donor-far clean rows are the better repair controls. Donor-far intrusive rows,
especially wide-object cases whose patched top1 lands near donor x2 but far
from receiver x1, are the most direct slot-value capture candidates for the
next attention/value localization pass.

The per128 v7 expansion adds one more correction: not all intrusive rows are
donor-slot capture. In the larger train-first panel, many intrusive rows land
near the baseline coordinate or previous/control slot. The next coordinate-
onset localization should therefore use a five-way destination taxonomy:
receiver repair, donor-slot value capture, previous/control-slot pull,
baseline inertia, and off-basin escape. This is likely a more faithful route
to the core mechanism than another pooled donor-delta run.

Current per128 destination-family evidence sharpens the next round. The
destination selector and staged hidden readout artifacts are:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_destination_panel_selector/v1_v2_v7_destination_panel_cap6
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_readout_rows/v6_v1_destination_panel_cap6_all_slots
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_readout_probe/v23_v6_destination_staged_slots_hidden_context_sharded3_image_root/merged
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_hidden_reducer/v4_v23_destination_family_hidden_context
```

The v23 readout covers 213 destination-panel cases, 1065 staged slots, and 4260
hidden logit-lens rows with 0 model errors. It confirms the user's broader-row
hint: pre-x1 failures are not limited to the old semantic pair and are not a
simple unseen-val phenomenon. Direct logits over the destination panel show:

```text
staged_pre_x1: coord_rank<=10 0.0094, coord_distance<=16 0.0188, mean rank 445.16
staged_after_x1: coord_rank<=10 0.2723, coord_distance<=16 0.4038, mean rank 176.82
staged_after_x1_y1: coord_rank<=10 0.2066, coord_distance<=16 0.4272, mean rank 204.77
staged_after_x1_y1_x2: coord_rank<=10 0.3380, coord_distance<=16 0.5634, mean rank 106.69
staged_after_full_box: box_end top1 0.5023
```

This is strong evidence against treating all failures as visual non-perception:
many rows become locally readable only after the correct coordinate prefix is
supplied. The main failure is at the coordinate-slot onset or slot-transition
state, not necessarily at object visibility.

The destination families separate into distinct mechanisms:

```text
receiver_repair: only 7 rows, but later slots and box_end are usually readable
donor_slot_capture: post-x1/y1 states are highly readable, but toward donor-slot value
baseline_inertia: pre_x1 and y1 remain poor, while x2 can become readable; box_end is weak
control_slot_pull: poor throughout, consistent with previous/control-slot anchoring
coordinate_edge_ambiguity: mostly small/local objects, high coord locality after x1/y1
off_basin_escape: mixed but often guidable after x1, not a pure perception failure
```

The next implementation round should therefore expand training-dataset rows in
two layers. First, mine more trained rows that fail under teacher-forced GT
prefix and label them with the destination taxonomy, without requiring rollout
labels. Second, match held-out val analogs by motif only after the train
mechanism has been identified. For false negatives, separate visual weakness
from language-side guidance by staged forcing before claiming the model cannot
perceive the object: descriptor/box-start guidance, x1 forcing, x1+y1 forcing,
x1+y1+x2 forcing, full-box forcing, and box_end/router forcing should be
reported as different rescue modes.

Immediate active probes:

```text
1. Destination-family train-row miner over a larger train sample, using
   teacher-forced pre_x1/post_x1/staged-slot readouts as labels.
2. Attention/value localization only on destination-pure cohorts:
   donor_slot_capture vs baseline_inertia vs control_slot_pull vs receiver_repair.
3. Small-object coordinate-locality/smoothness panel from
   coordinate_edge_ambiguity and strict receiver-near rows.
4. FN guidance-versus-perception panel using staged forcing/continuation on
   missing-object rows, with train failures and val analogs reported separately.
```

The first destination-family train-row miner is now complete:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/train_val_candidate_bank/v8_bbox_len12000_regimes_per256_imgcap2_desccap12
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_readout_probe/v25_v8_per256_train_robust_positions_sharded3_image_root/merged
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_failure_panel/v8_v25_per256_train_only_prex1_failure_seeds64
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_destination_panel_selector/v2_v3_per256_train_only_destination_uncapped
```

The per256 train-only readout covers 1536 train objects and 6144 robust
formation positions. It finds that trained rows still fail at pre_x1
(`coord_rank<=10` 0.3151, `coord_distance<=16` 0.4473), while post_x1 is much
better (`coord_rank<=10` 0.6400, `coord_distance<=16` 0.8359). Small-object,
crowded, and repeated-class train rows are the most important coordinate-onset
failure regimes; simple-control rows are mostly not coordinate-onset failures
and instead expose box-close/termination behavior.

The train-only failure panel selected 346 trained-sequence failures, including
328 severe pre_x1 basin failures. The previous-slot guided-delta pass over
these rows produced 346 patch rows with 0 errors but remained mostly a
diagnostic perturbation, not a repair vector:

```text
target_rank_improved_rate: 0.4162
target_top1_rate: 0.0029
coord_distance_lte16_rate: 0.0694
slot_intrusion_rate: 0.6821
mean_receiver_target_rank_delta: +2223.87
```

Destination-family counts over the train-only panel:

```text
baseline_inertia: 108
off_basin_escape: 62
donor_slot_capture: 47
control_slot_pull: 32
coordinate_edge_ambiguity: 12
receiver_repair: 8
```

This updates the mechanism priority: baseline inertia is the largest trained-
sequence destination family, larger than donor-slot capture. The next
attention/value localization should therefore compare destination-pure cohorts
rather than another pooled donor-delta run. Immediate candidates are:

```text
baseline_inertia:
  why top1 remains near the baseline coordinate basin after a slot-transition delta

donor_slot_capture:
  where donor x2 value enters and becomes selected over receiver x1

control_slot_pull:
  whether previous-slot anchoring is written by slot-phase state or local token history

coordinate_edge_ambiguity:
  coordinate-token locality/smoothness, especially small-object rows

receiver_repair:
  a positive microscope only; count is too small for population claims
```

Per256 train/val mirror update:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_readout_probe/v27_v8_per256_val_robust_positions_sharded3_image_root/merged
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_readout_probe/v28_v25_v27_per256_train_val_robust_positions_combined
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_failure_panel/v9_v28_per256_prex1_train64_val1_analog_panel
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_failure_launch_filter/v3_v9_per256_train64_val1_loose_balanced_pairs16
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_guided_delta_patch/v24_v3_per256_train_val_launch_filter_prevslot_afterx1y1_layerinput
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_guided_delta_patch/v25_v3_per256_train_val_launch_filter_prevslot_afterx1y1_selfattn
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_guided_delta_patch/v26_v3_per256_train_val_launch_filter_prevslot_afterx1y1_mlp
```

The v8/per256 candidate bank already covers 1536 train and 1536 val objects
across 79 target descriptions. The new val readout mirrors the train readout
with 6144 val robust-position rows and 0 errors. The combined readout has
12288 rows over 3072 cases.

The matched raw failure panel selected 346 train seeds and 346 val analogs.
The filtered launch panel keeps 82 train/val pairs, with 77/82 same-regime
same-desc matches. It is broad beyond the original hand-picked pair: 37 unique
descriptions in the filtered panel, and only 22 person rows out of 164.

Matched train/val component intervention at layer 24:

```text
site         split  n   improved  d16     intrusion  mean_rank_delta
layer_input train  82  0.4756    0.0854  0.6829     +2218.40
layer_input val    82  0.4390    0.0976  0.6585     +994.93
self_attn   train  82  0.4878    0.0488  0.5976     +337.24
self_attn   val    82  0.5000    0.1098  0.5122     -1347.29
mlp         train  82  0.5610    0.0854  0.5976     -333.07
mlp         val    82  0.5732    0.0610  0.5366     -329.24
```

The matched analog result does not support train-vs-val memorization as the
main mechanism. Train seeds are more severe, but matched val analogs are at
least as responsive under self-attention and MLP output patches. The more
stable axis is destination family and component site.

Matched train/val destination-family selection shows that `baseline_inertia`
is dominant across component sites:

```text
layer_input baseline_inertia: train 24, val 25
self_attn   baseline_inertia: train 35, val 28
mlp         baseline_inertia: train 25, val 26
```

Self-attention almost eliminates donor-slot capture as a final landing family
(`donor_slot_capture`: train 1, val 1), but it increases `baseline_inertia`
and `control_slot_pull`. This suggests self-attention helps suppress direct
donor-slot copying while leaving sticky coordinate basins and previous-slot
anchoring intact. MLP remains a plausible rerouting/write surface but does not
reliably repair x1 coordinate basins.

Revised next implementation priority:

```text
1. Make baseline_inertia the main microscope, but split it into
   boundary/wrapper inertia versus coordinate inertia. The first completed
   layer/site scan shows that many baseline_inertia rows, especially trained
   rows, are not in coordinate-emission mode at all: full-vocab top1 is often
   <|box_end|> and wrapper-token mass can be essentially 1.0.
2. Add a token-mode atlas over broader train/val staged rows before another
   pooled donor-delta run: summarize coord-vocab mass, wrapper-token mass,
   box_end/object_ref margins, full-vocab top1 class, and coordinate-locality
   metrics by split, regime, position, and destination family.
3. Probe the coordinate-mode versus boundary-mode gate directly:
   box_end/object_ref suppression, coordinate-vocab promotion, hidden
   logit-lens mass traces, and attention/value localization for the boundary
   decision.
4. Probe control_slot_pull as previous-slot anchoring:
   patch or ablate the previous coordinate slot and y1-like control state.
5. Keep donor_slot_capture as a smaller reroutable subset:
   component localization and value/head tracing only after baseline inertia.
6. Use small_object and coordinate_edge_ambiguity to separate object binding
   from coordinate quantization/smoothness loss.
7. Use termination_tail for boundary-state and stop/continue competition.
8. Continue reporting train and val separately. Trained rows are not merely
   controls: a trained sequence that still routes to <|box_end|> at pre_x1
   under teacher-forced GT prefix is a high-value mechanism case. Val rollout
   false negatives remain useful, but they are an asymmetric stress slice
   unless matched train rollout labels are available.
```

Completed baseline-inertia layer/site tomography:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_guided_delta_patch/v28_v27_baseline_inertia_layer_site_scan_sharded3/merged
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_coord_value_geometry/v7_v10_baseline_inertia_layer_site_scan_geometry
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_destination_panel_selector/v6_v7_baseline_inertia_layer_site_scan_destination_uncapped
```

Key result: 49 unique baseline-inertia states were scanned across layer/site
patches. Successful rows exclude the invalid explicit layer 28 configuration.
Later-slot `donor_minus_previous_slot` deltas improve target rank in many rows
but do not open the correct coordinate basin (`target_top1_rate` 0.0,
`coord_rank_lte_10_rate` 0.0, `slot_intrusion_rate` 1.0). At the full-vocab
surface, 46/49 baseline states are wrapper-token top1 before patch, 44/49 are
`<|box_end|>` top1, and the train subset is 24/24 `<|box_end|>` top1 with
mean coordinate-vocab mass about `4.84e-11`. This revises the mechanism from
pure coordinate-basin stickiness toward coordinate-mode versus boundary-mode
gating. See:

```text
progress/diagnostics/2026-06-22_baseline_inertia_boundary_basin_findings.md
```

Post-hoc token-mode atlas correction:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_readout_reducer/v3_v28_per256_train_val_token_mode_atlas
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_readout_reducer/v4_v23_destination_family_token_mode_atlas
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_readout_reducer/v5_v20_per128_train_val_token_mode_atlas
```

Do not overgeneralize the boundary-mode result to all pre-x1 failures. The
broad per256 readout has coordinate-mode pre_x1 rows in both splits
(`coord_top1_class_rate` 1.0, coord-vocab mass about 0.998) even though exact
rank/locality is often poor. The boundary-mode signature is destination-family
specific: `baseline_inertia` at `staged_pre_x1` has coord mass 0.0164, wrapper
mass 0.9828, and box_end top1 0.9508, while `control_slot_pull`,
`coordinate_edge_ambiguity`, `off_basin_escape`, and `receiver_repair` are
mostly coordinate-mode. The next taxonomy should therefore separate:

```text
coordinate-mode wrong-basin failures:
  coordinate top1, high coord mass, poor exact rank/locality

boundary-mode pre-x1 failures:
  wrapper top1, high wrapper mass, usually <|box_end|>

post-full-box router failures:
  wrapper top1, but object_ref_start/end beats <|box_end|>
```
