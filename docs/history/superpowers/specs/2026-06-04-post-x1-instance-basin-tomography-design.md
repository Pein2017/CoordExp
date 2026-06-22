# Post-X1 Instance-Basin Tomography Design

## Decision

Run an A3.3 mechanism study that compares three checkpoints by their
post-`x1` instance-basin dynamics, not by final detector accuracy.

The study asks:

```text
Given desc + x1_i for one same-desc GT instance i,
does the model's y1/x2/y2 posterior remain in instance i's basin,
or does it drift to a same-desc competitor, background, boundary extremes,
or invalid geometry?
```

The primary evidence is slot-level posterior / logit readout.  Greedy
continuation is retained only as secondary evidence.

## Checkpoint Roles

Primary clean pair:

- `fullobj_random_pure_ce_ckpt3668`
- `fullobj_sorted_pure_ce_ckpt3668`

Reference anchor:

- `et_rmp_ce_ckpt3664`

The two full-object pure-CE checkpoints form the clean controlled comparison for
`random_permutation` versus `sorted` training order under the no-newline
compact-full template contract.  The ET-RMP-CE checkpoint is a historical
mechanism reference, not a clean accuracy baseline and not a controlled
objective-only comparison.

Rows and summaries must expose `controlled_comparison_group`:

- pure-CE pair: `pure_ce_sorted_vs_random_no_newline`
- ET-RMP-CE reference anchor: `reference_anchor_not_controlled`

Final reports may place all three checkpoints in one table or figure, but
ET-RMP-CE conclusions must carry a `reference_anchor` or
`template_objective_confounded_reference` label.

## Checkpoints

Random-order no-newline pure-CE:

```text
/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_fullobj_random_sft_bsz16_4epoch_tokenrows_v2/compact-full-fullobj-random-sft-bsz16-4epoch-tokenrows-v2/v1-20260601-062428/checkpoint-3668
```

Sorted-order no-newline pure-CE:

```text
/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_fullobj_sorted_sft_bsz16_4epoch_tokenrows_v2/compact-full-fullobj-sorted-sft-bsz16-4epoch-tokenrows-v2/v1-20260601-062429/checkpoint-3668
```

ET-RMP-CE reference anchor:

```text
/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_et_rmp_ce_support2_bsz16_4epoch_tokenrows_v2/compact-full-et-rmp-ce-support2-bsz16-4epoch-tokenrows-v2/v0-20260504-071356/checkpoint-3664
```

## Template Contract

A3.3 must use per-checkpoint template contracts.  Do not use the A3.2 global
`row_separator: none` assumption for every checkpoint.

Primary native-template readout:

```yaml
fullobj_random_pure_ce_ckpt3668:
  template_contract_id: compact_full_no_newline_native_v1
  detection_sequence_format: compact_full
  coordinate_surface: coord_token
  bbox_format: xyxy
  row_separator: none
  contract_provenance: user_reported_training_contract

fullobj_sorted_pure_ce_ckpt3668:
  template_contract_id: compact_full_no_newline_native_v1
  detection_sequence_format: compact_full
  coordinate_surface: coord_token
  bbox_format: xyxy
  row_separator: none
  contract_provenance: user_reported_training_contract

et_rmp_ce_ckpt3664:
  template_contract_id: compact_full_newline_native_v1
  detection_sequence_format: compact_full
  coordinate_surface: coord_token
  bbox_format: xyxy
  row_separator: newline
  contract_provenance: legacy_compact_full_default_inferred
```

The three `resolved_config.json` files record `expected_template=compact_full`,
`coordinate_surface=coord_token`, and `bbox_format=xyxy`, but they do not expose
a direct `row_separator` field.  A3.3 must therefore persist the explicit
template contract used by each readout row and report the provenance above.

Optional cross-template sanity:

- Run ET-RMP-CE on a tiny `row_separator: none` subset.
- Run the two no-newline pure-CE checkpoints on a tiny
  `row_separator: newline` subset.

These sanity rows are diagnostic only.  They must not replace the
native-template primary comparison.

Runtime and artifact rows must validate the full per-checkpoint contract, not
only the `template_contract_id`.  For every runtime-produced row, the
`checkpoint_role` must imply the exact expected:

- `template_contract_id`
- `row_separator`
- `contract_provenance`
- `detection_sequence_format`
- `coordinate_surface`
- `bbox_format`

The runtime must also record prompt/render hashes:

- `system_prompt_sha256`
- `user_prompt_sha256`
- `template_prompt_hash`
- `assistant_prefix_sha256`
- `forced_prompt_sha256`

These hashes are part of the reproducibility contract because
`src/config/prompts.py` defaults compact-full prompts to newline unless
`row_separator` is explicitly supplied.  A3.3 must always build both system/user
instructions and assistant forced-state text from the checkpoint's own template
contract.

## Motivation From A3.2

A3.2 sorted-vs-random no-newline smoke found a surprising separation:

- Boundary readout did not support a pure EOS-only explanation.  Residual
  candidates were favored in `58/64` random rows and `61/64` sorted rows.
- Mean sorted-minus-random `residual_vs_eos_margin` was `+1.0678`.
- FN hint probe rows were dominated by coordinate / instance binding failures:
  `coord_binding_failure=380/768`, `desc_selection_failure=224/768`,
  `rescued_residual_instance=57/768`.
- Slot evidence showed that `x1` can be a useful control signal, but model
  decoded slots still miss often; later slots `y1/x2/y2` remain major failure
  surfaces.
- This contrasts with the older ckpt3664 FN-rescue result where `desc_x1` was a
  strong rescue arm.  A3.3 exists to explain whether the discrepancy comes from
  post-`x1` basin geometry, prefix state, objective/template differences, or a
  mix of them.

This motivation is mechanism evidence, not a final benchmark claim.

## Scope

Proposed idea-wise analysis surface:

```text
src/analysis/post_x1_instance_basin_tomography/
scripts/analysis/post_x1_instance_basin_tomography/
configs/analysis/post_x1_instance_basin_tomography/
```

Proposed artifact root:

```text
/data/CoordExp/outputs/analysis/autoreg_object_rollout/post_x1_instance_basin_tomography/three_ckpt_phase_a3_3
```

Primary data surface:

```text
/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/train.coord.jsonl
/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/val.coord.jsonl
/data/CoordExp/public_data/coco/rescale_32_1024_bbox
```

If ET-RMP-CE compatibility with `len12000` needs a legacy-control lane, record
that separately and do not mix its counts into the primary pure-CE clean pair.

The primary `len12000` JSONLs use `bbox_2d` coord-token strings such as
`<|coord_699|>`.  A3.3 must parse this as `coord_token` surface and must not
silently coerce it as pixel xyxy.  Case rows must persist:

- `source_bbox_field`
- `source_bbox_surface`
- `bbox_coord_token_xyxy`
- optional `bbox_pixel_xyxy`
- `bbox_normalization_policy_id`

Numeric `bbox` / `bbox_xyxy` inputs are accepted only when their surface is
explicitly declared by source metadata such as `bbox_surface` or `coord_mode`.
Nested `points` inputs are a separate pixel-corner surface and must be recorded
as such.

## Lanes

### Lane 1: GT-Controlled Same-Desc Basin Trajectory

Use only same-desc GT instances to define target and competitor basins.

For one image and one repeated desc:

```text
GT instances: i, j, k, ...
target: i
competitors: j, k, ...
```

Construct forced states:

```text
state pre_x1:      prefix + desc + box_start
state post_x1:     prefix + desc + x1_i
state post_y1:     prefix + desc + x1_i + y1_i
state post_x2:     prefix + desc + x1_i + y1_i + x2_i
```

Primary readout:

```text
pre_x1  -> x1 posterior, context only
post_x1 -> y1 posterior, primary
post_y1 -> x2 posterior, primary
post_x2 -> y2 posterior, primary
```

The goal is to determine whether target identity remains stable after the
first coordinate is supplied.

Lane 1 null:

```text
Given desc+x1_i, the y1/x2/y2 posterior remains in instance i's basin.
```

Lane 1 alternative:

```text
The posterior switches to a same-desc competitor, other-desc GT object,
background/outlier, boundary extreme, or invalid/non-coordinate state.
```

### Lane 2: Same-Desc Basin Attraction Matrix

For each same-desc GT instance `i`, force `desc + x1_i` and measure which
instance wins at each subsequent slot.

Expected artifact shape:

```text
rows: forced anchor instance i
cols: winner instance at y1, x2, y2, final box summary
values: target mass, competitor mass, margin, winner bucket
```

This lane reveals whether different `x1_i` values produce distinct instance
basins, or whether multiple anchors collapse toward one salient same-desc
object.

Lane 2 is a cluster-level aggregation of Lane 1 rows, not an independent second
runtime probe.  Its null is that the same-desc attraction matrix is near
diagonal: each forced anchor `x1_i` routes mostly to instance `i`.  Its
alternative is many-to-one collapse, asymmetric off-diagonal attraction, or
dominant sinks.  Lane 2 reports:

- diagonal stay rate;
- off-diagonal rate;
- dominant sink instance id;
- per-slot attraction matrix;
- off-diagonal concentration by desc and checkpoint.

### Lane 3: Prefix-Quality Perturbation

For the same target instance and the same `desc+x1_i`, vary prefix quality.

Prefix modes:

```text
P0 minimal_or_empty_prefix
P1 canonical_sorted_gt_prefix_before_target
P2 clean_non_target_same_desc_prefix
P3 duplicate_same_desc_prefix
P4 wrong_instance_same_desc_prefix
P5 rollout_native_prefix_with_quality_label
```

Good prefix:

- built from GT or high-quality matched predictions;
- no same-desc duplicate;
- no obvious unmatched / FP row;
- does not include the target instance;
- same-desc prefix objects match non-target GT instances.

Bad prefix buckets:

- `duplicate_prefix`
- `wrong_instance_prefix`
- `unmatched_prefix`
- `malformed_prefix`
- `overcovered_prefix`
- `target_leak_prefix`

`target_leak_prefix` must be excluded from recall-rescue conclusions and used
only for ledger / duplicate analysis.

Lane 3 tests whether `x1_i` is an absolute local anchor, or whether prior
autoregressive state can reshape the post-`x1` basin.

Lane 3 uses prefix-quality basin deltas.  Avoid using "FN rescue" terminology
for the primary Lane 3 readout; historical FN-rescue results are motivation
only.  Preferred metric names are:

- `prefix_basin_delta`
- `prefix_recovery_delta`
- `bad_to_good_basin_recovery_rate`
- `prefix_damage_type`

`target_leak_prefix` is excluded from primary basin-recovery denominators and
kept only for duplicate / ledger diagnostics.

## Primary Candidate Semantics

Primary basin labels use same-desc GT instances only:

- `target_instance`
- `same_desc_competitor`
- `other_desc_object`
- `background_or_outlier`
- `boundary_extreme`
- `invalid_or_unsupported`

Rollout emitted boxes, unmatched predictions, and possible unlabeled objects
are secondary annotations only.  They must not define the primary
target/competitor basin labels.

Case universe rows must annotate whether `desc+x1_i` is a unique anchor:

- `anchor_slot`
- `target_x1_coord`
- `same_desc_x1_competitor_values`
- `min_same_desc_x1_separation_bins`
- `x1_anchor_collision_count`
- `x1_anchor_unique_under_r95`
- `anchor_ambiguity_bucket`
- `primary_case_eligible`
- `case_exclusion_reason`

Strict primary denominators should exclude exact or near-collision anchors
where `x1_i` cannot reasonably identify one same-desc instance.  Ambiguous
anchors remain useful as a secondary stress slice.

Rationale:

- GT-controlled basin readout needs a stable coordinate system.
- Unmatched predictions may be unlabeled objects, hallucinations, duplicates,
  or shifted boxes.
- Emitted rollout boxes can be biased or wrong, so they are useful provenance
  but not stable primary basin labels.

## Slot Posterior Metrics

For each state and target slot, record both a full-vocab view and a conditional
coordinate view.

Full-vocab view:

- `coord_vocab_mass`
- `noncoord_top_token_id`
- `noncoord_top_prob`
- `coord_mass_low_flag`

Conditional coord-token view over bins `0..999`:

- `target_slot_mass`
- `best_same_desc_competitor_slot_mass`
- `target_vs_competitor_margin`
- `target_slot_logit`
- `best_same_desc_competitor_slot_logit`
- `target_vs_competitor_logit_margin`
- `top_peak_logit`
- `target_logprob`
- `best_same_desc_competitor_logprob`
- `winner_instance_id`
- `winner_bucket`
- `slot_taxonomy`
- `low_margin_flag`
- `top_peak_value`
- `top_peak_mass`
- `target_rank`
- `target_r95_hit`
- `best_competitor_r95_hit`
- `other_desc_object_r95_hit`
- `boundary_extreme_flag`
- `background_or_outlier_flag`

`coord_vocab_mass` must be computed from full-vocab logits, not from a
conditional 1000-bin coordinate softmax.  The conditional coordinate posterior
is still the primary shape used for target/competitor mass, but low
full-vocab coordinate mass must prevent a row from being counted as a clean
basin stay.

Strict coordinate neighborhoods use the focused R95 axis rule:

```text
R95 = floor(min(8, 0.04 * axis_len))
sigma = R95 / 1.96
```

Use the relevant target axis:

- `x1` and `x2`: bbox width
- `y1` and `y2`: bbox height

If `R95=0`, treat the strict target as one-hot.

Boundary extremes `0/999` are annotations, not always failure buckets.  If
`top_peak_value=0` or `999` lies inside the target or competitor R95
neighborhood, the row keeps its target/competitor identity and sets
`boundary_extreme_flag=true`.  Use `winner_bucket=boundary_extreme` only when
the boundary peak is not explained by target, same-desc competitor, or
other-desc GT neighborhoods.

Low-margin states must be operationally classified.  If target and competitor
mass or logit margins are below the configured threshold, set
`low_margin_flag=true` and use `slot_taxonomy=slot_ambiguous_low_margin`
instead of making a clean stay/switch claim.

## Trajectory Taxonomy

The primary success taxonomy is slot-level and trajectory-level.

Slot-level:

- `slot_target_r95_hit`
- `slot_competitor_r95_hit`
- `slot_target_margin_positive`
- `slot_ambiguous_low_margin`
- `slot_background_or_outlier`
- `slot_boundary_extreme`
- `slot_invalid_or_unsupported`

Trajectory-level:

- `stay_target_all_slots`
- `early_switch`
- `partial_target_then_switch`
- `competitor_consistent`
- `background_drift`
- `boundary_extreme_dominated`
- `ambiguous_tied`
- `invalid_or_unsupported`

Trajectory classification precedence:

1. Any invalid / unsupported / low coord-mass slot -> `invalid_or_unsupported`.
2. All confident target slots -> `stay_target_all_slots`.
3. All confident same competitor id -> `competitor_consistent`.
4. First confident non-target slot at `y1` -> `early_switch`, with
   `switched_to_instance_id` and later-slot consistency fields.
5. Target prefix followed by confident competitor -> `partial_target_then_switch`.
6. Boundary-extreme failure in one or more unexplained slots ->
   `boundary_extreme_dominated`, with a list of affected slots.
7. Other-desc or background drift -> `background_drift` / `other_desc_drift`.
8. Low-margin or mixed non-confident states -> `ambiguous_tied`.

Box-level summaries are secondary:

- `target_iou50`
- `target_iou75`
- `best_same_desc_competitor_iou`
- `matched_instance_id`

IoU50 and IoU75 are useful summaries, but they must not be the only success
definition.

## Prefix Sensitivity Metrics

Lane 3 must report paired deltas for the same image, desc, target instance,
checkpoint, template contract, and forced coordinate state:

- `prefix_basin_delta`
- `prefix_switch_rate`
- `prefix_recovery_delta`
- `bad_to_good_basin_recovery_rate`
- `prefix_damage_type`
- `good_prefix_stay_bad_prefix_switch`
- `good_prefix_switch_bad_prefix_stay`
- `prefix_invariant_stay`
- `prefix_invariant_fail`

The report should identify whether bad prefixes mainly cause:

- same-desc competitor attraction;
- background drift;
- boundary extremes;
- invalid geometry;
- desc/format degradation in the secondary greedy lane.

Prefix rows must be artifact-grade and self-auditing.  They should record
source kind, source artifact / line id when applicable, object lineage, match
policy and IoU for matched predictions, leak / duplicate / malformed flags,
construction policy id for synthetic bad prefixes, denominator eligibility,
and skipped-mode reasons.  `rollout_native_prefix_with_quality_label` may be
skipped in smoke, but full runs must either provide a rollout source artifact
or fail the status gate for missing requested prefix modes.

Prefix artifacts store semantic `prefix_objects` as runtime truth.  Per-checkpoint
rendered strings are derived at runtime from `prefix_objects` plus that
checkpoint's `row_separator`; a checkpoint-agnostic `prefix_text` must not be
used as the primary runtime surface.

`prefix_mode_summary.json` must record:

- `prefix_modes_requested`
- `prefix_modes_materialized`
- `prefix_modes_skipped`
- `rollout_prefix_missing_policy`
- `skipped_mode_reasons`

## Greedy Continuation Lane

Greedy continuation is secondary.

Use deterministic free-text continuation under the same prefix states:

```text
desc + x1_i -> generate y1/x2/y2
desc + x1_i + y1_i -> generate x2/y2
```

Record parse validity and final matched instance, but do not let parser
failures hide posterior evidence.

Sampling guidance:

- Smoke: greedy on a small representative subset.
- Full: greedy on 10-20% of target states.

## Scale

Two-stage run:

```text
Smoke:
  same-desc GT basin cases: 64 images
  target instances: about 256
  checkpoints: 3
  prefix modes: 3-4
  primary: slot posterior readout
  secondary: small greedy continuation subset

Full:
  same-desc GT basin cases: 512-1024 images
  target instances: 2048-4096
  checkpoints: 3
  prefix modes: 5-6
  primary: slot posterior readout
  secondary: greedy continuation on 10-20%
  execution: 8 single-GPU shards
```

Hard-biased sampling should favor:

- same-desc count >= 3;
- object count >= 6;
- crowded repeated classes such as person, book, cup, chair, bottle, traffic
  light, car, motorcycle, and similar repeated COCO categories;
- train and val coverage, with train allowed because train failures are less
  deniable;
- a small easy sanity slice where same-desc count is 1 or 2.

## Required Artifacts

Minimum artifact contract:

- `config_resolved.json`
- `template_contracts.json`
- `case_universe.jsonl`
- `prefix_modes.jsonl`
- `prefix_mode_skipped_rows.jsonl`
- `prefix_modes_summary.json`
- `slot_posterior_shards/shard_0.jsonl` through `shard_7.jsonl`
- `slot_posterior_shard_summaries.jsonl`
- `merge_manifest.json`
- `slot_posterior_rows.jsonl`
- `trajectory_rows.jsonl`
- `basin_attraction_matrix.jsonl`
- `prefix_sensitivity_rows.jsonl`
- `greedy_continuation_rows.jsonl`
- `summary.json`
- `report.md`
- `gallery/`

Every JSON/JSONL artifact must carry schema/provenance fields appropriate to
its surface:

- `artifact_schema_version`
- `row_schema_version`
- `runtime_kind`
- `runtime_id`
- `mock_runtime`
- `dry_run`
- `config_path`
- `config_sha256`
- `code_revision`
- `checkpoint_paths_by_role`
- `checkpoint_fingerprints_by_role`
- `template_contracts_by_role`
- `template_contract_sha256`
- `primary_basin_label_source`
- `source_shard_ids`
- `merge_input_sha256_by_shard`

Recommended plots:

- per-checkpoint slot hardness heatmap;
- same-desc attraction matrix;
- target-vs-competitor margin by slot;
- prefix-quality delta violin or box plot;
- boundary extreme rate by checkpoint/template;
- representative per-image basin trajectory cards.

## Status Gates

Before GPU work:

- all checkpoint paths exist;
- per-checkpoint template contracts are explicit;
- pure-CE no-newline and ET-RMP newline contracts are not silently mixed;
- JSONL image root audit passes;
- same-desc GT case universe has enough hard cases;
- CPU dry-run materializes case and prefix schemas;
- smoke config uses a distinct artifact root.

After smoke:

- all three checkpoint roles have non-empty slot posterior rows;
- every posterior row records checkpoint role and template contract;
- every runtime row records the exact expected full template contract for its checkpoint role;
- every runtime row records prompt/render hashes;
- target/competitor labels are GT-derived;
- `bbox_2d` coord-token surface is parsed and persisted explicitly;
- ambiguous `x1` anchors are excluded or flagged outside the primary denominator;
- full-vocab coordinate mass is present and low-mass rows cannot be counted as clean stays;
- low-margin rows are classified as ambiguous;
- other-desc GT drift is distinct from background drift;
- strict R95 radii are slot-axis-specific;
- boundary extremes `0/999` are separately counted;
- legitimate target/competitor boundary coordinates keep their identity while setting boundary flags;
- shard row counts equal merged posterior row counts;
- downstream trajectory / matrix / prefix-sensitivity rows are non-empty for all expected roles;
- greedy parser failures do not remove posterior rows;
- report labels ET-RMP as reference anchor.

Full launch requires the smoke gates to pass.

## Interpretation Boundaries

A3.3 may support statements such as:

- sorted and random differ in post-`x1` basin stability;
- a checkpoint has broader candidate fields but weaker basin stay rate;
- a checkpoint collapses multiple `x1_i` anchors to one salient same-desc
  instance;
- bad prefixes reshape the post-`x1` basin;
- ET-RMP behaves differently under its native template and objective ecology.

A3.3 must not claim:

- a final detector accuracy ranking among the three checkpoints;
- that ET-RMP is a clean controlled baseline for the no-newline pure-CE pair;
- that attention-head causality has been proven;
- that unmatched predictions are hallucinations without manual or secondary
  evidence;
- that IoU50 alone defines mechanism success.

## Open Implementation Notes

Implementation should start from tests and a small CPU schema dry-run.  Reuse
A3.2 compact row rendering, coord conversion, and JSONL helpers where they are
generic, but keep A3.3 as a separate idea-wise analysis surface.

The first implementation plan should be written only after this design is
reviewed.
