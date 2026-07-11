# Legacy Correctness-Invariant Inventory

## Scope

Wave 1A read-only inventory for OpenSpec task 2.9. No production source code
was implemented and old `src/` was not moved. This inventory is the reviewable
evidence for the checked task 2.9 gate.

The goal is to preserve correctness lessons from legacy `src/` and tests
without preserving the old implementation as active authority.

## P0 Stop Gate

Do not archive old `src/` until this inventory has been reviewed and the Wave 2
archive/skeleton plan can point to it.

## Invariants By Rebuild Wave

### Wave 2: Fixture, Config, Artifact Seed

#### No-Resize Is A Hard Runtime Contract

Legacy evidence:

- `src/detection/dataset.py` passes `do_resize=False` when supported.
- `src/common/qwen_generation.py` centralizes Qwen processor kwargs with
  `do_resize=False`.
- `src/trainers/rollout_matching/preflight.py` rejects missing or true
  `mm_processor_kwargs.do_resize` in Stage-2 server preflight.

V1 action:

- add fixture checks for processor-derived admissible dimensions;
- assert pre-processor failure for invalid no-resize dimensions;
- record `patch_size`, `merge_size`, raw pixels, `image_grid_thw`, and merged
  visual-token count.

#### Artifact Names And Metric Identities Are Stable

Legacy evidence:

- `tests/test_infer_artifact_metadata.py` checks infer metadata and run-relative
  artifact binding.
- `tests/test_metric_events.py` checks metric denominator reduction and alias
  collision behavior.

V1 action:

- explicitly reject padded checkpoint step ids such as `step-000005`;
- assert `checkpoints/checkpoint-final.json`;
- store metric split and metric name separately;
- freeze `acc_top1` and `acc_top5` as top-level metric names.

#### Checkpoint Metadata Proves Payload Completeness

Legacy evidence:

- `tests/test_checkpoint_weight_only_policy.py` distinguishes restartable and
  artifact-only behavior and sidecar completeness.
- `src/trainers/final_checkpoint.py` preserves final checkpoint metadata.

V1 action:

- once special-token embedding mechanism is chosen, test compact sidecar
  metadata and selected-token payload completeness;
- keep V1 checkpoint schema focused on adapter payloads, selected embedding
  deltas, processor identity, config fingerprint, schedule identity, metric
  status, trainable surface, and optimizer update status.
- port payload completeness, identity, metadata, update-status, final-alias, and
  selected-token payload checks; do not port HF restartable checkpoint/resume
  sidecars or exact optimizer/scheduler/scaler/RNG resume unless a later
  contract promotes them.

#### Strict Config And Planned-Step Schedule Are Rebuild Contracts

Legacy evidence:

- `tests/test_training_config_strict_unknown_keys.py` checks unknown-key
  dotted diagnostics, removed/legacy config surface rejection, list replacement
  behavior, and strict authored fields.
- `tests/test_stage2_step_budget_windows.py` contains full-window guard patterns
  that should be mined for planned-step invariants, not ported as Stage-2
  behavior.

V1 action:

- unknown authored fields fail before model, dataset, or artifact mutation;
- inheritance remains one-parent, with list replacement, path-origin tracking,
  and cycle diagnostics;
- legacy cadence aliases such as `save_steps` and `eval_steps` fail in favor of
  canonical schedule fields;
- `resolved_step_schedule.json` records the planned-step clock before training;
- effective batch size divides exactly by world size, with no rounding and no
  public `grad_accum_steps` field;
- deterministic tail-fill behavior avoids dropping final packs or performing
  partial optimizer updates unexpectedly.

### Wave 3: Data, Template, Qwen Encoding

#### Geometry And Object Order Are Preserved

Legacy evidence:

- `tests/test_coord_geometry_invariants.py`
- `tests/test_coord_utils.py`
- `tests/test_bbox_parameterization.py`

V1 action:

- port shape/order invariants, not old alternate bbox parameterizations;
- preserve canonical `x1,y1,x2,y2`;
- preserve object order unless an approved ordering policy changes it;
- reject legacy `sorted`.

#### RawExample And Source Identity Are Canonical

Legacy evidence:

- `tests/test_detection_prompt_input_codec.py` covers multiple-image rejection
  patterns.
- `tests/test_detection_scene_contract.py` covers resolved image references and
  multiple-reference rejection.
- `tests/test_detection_raw_schema_contract.py` covers unknown or empty raw
  schema rejection.
- `tests/test_common_paths.py` covers strict path resolution.
- `tests/test_max_pixels_enforcement.py` covers required integer dimensions.

V1 action:

- preserve `example_id` and source row identity through
  `RawExample -> RenderedExample -> EncodedExample -> PackedSequence`;
- enforce exactly one image for V1 teacher-forced Qwen3-VL training examples;
- reject unknown raw fields, empty raw objects, duplicate example/object ids,
  and invalid or missing image paths;
- normalize current `len12000` source rows into the canonical raw-example shape
  without regenerating training JSONL sources;
- require integer image width/height before Qwen encoding;
- keep this V1-scoped and do not port legacy inference multi-image behavior or
  Stage-2 eval row-shape behavior.

#### Rendered Spans Align Exactly Before Supervision

Legacy evidence:

- `tests/test_detection_template_span_alignment.py`

Important legacy assertions:

- exact token boundaries;
- one chat-template application;
- assistant-only labels;
- explicit `<|im_end|>` supervision;
- no guessed stop marker;
- failure on malformed or missing offsets.

V1 action:

- test crossing spans;
- test uncovered loss-bearing characters;
- test first assistant answer token;
- test `<|im_end|>` supervised and following newline ignored.

#### Token Roles Map To Closed Token Types

Legacy evidence:

- `tests/test_token_span_masks_from_templates.py`

V1 action:

- wrappers become `schema`;
- `<|coord_0|>` through `<|coord_999|>` become `coordinate`;
- `<|im_end|>` becomes `eos`;
- free text becomes `desc_text`;
- Qwen chat/control/image/video/pad/tool/FIM/think/reserved tokens are excluded
  from free-text allowance.

### Wave 4: Packing And Qwen Forward

#### Packed Segment Isolation Is Multi-Layered

Legacy evidence:

- `tests/test_packed_labels_and_coord_targets.py`
- `tests/test_train_batch_contract.py`
- `tests/test_stage2_ab_packing_mask_gradients.py`

Port the invariant that packed segment boundaries control labels, supervision
remapping, position resets, and attention metadata.

Do not port legacy row-count authority blindly. Some legacy tests checked a
3-row Qwen-era shape; current V1 OpenSpec requires installed-version validation
and the current local Qwen3-VL source study proves a 4-row `[text,t,h,w]`
boundary. Implementation must validate the installed row count and meaning, and
fail if upstream changes. Preserve reset semantics, not the old row shape.

#### Packer Policy Is Deterministic And Non-Truncating

Legacy evidence:

- `tests/test_packing_wrapper.py`

V1 action:

- exact-max examples fit;
- overlength examples fail;
- packer commits current pack on overflow;
- no silent truncation;
- deterministic pack-plan receipt;
- do not inherit static cache complexity unless a later approved need appears.

### Wave 5: Supervision, Losses, Metrics

#### Causal Shift Must Not Cross Segment Boundaries

Legacy coverage is indirect through packed labels and batch contracts, while V1
OpenSpec now states the rule explicitly.

V1 action:

- add direct test where the first token of segment 2 would be supervised;
- assert `LossContext` rejects or omits it without using segment 1 final logits.

#### Denominator Semantics Are Explicit And Window-Level

Legacy evidence:

- `tests/test_length_insensitive_loss_normalization.py`
- `tests/test_teacher_forcing_token_ce.py`
- `tests/test_grad_accum_loss_scale_mixin.py`

V1 action:

- port denominator-sanity patterns, not old objective semantics;
- test `segment_balanced` over unequal segment/atom counts;
- test planned-step denominator across accumulation;
- test no backend double scaling.

#### Gate Loss Is Exact Group Mass

Legacy evidence:

- `tests/test_recursive_loss_support_balance.py`

V1 action:

- add exact numeric tests for `TokenTypeGateLoss`:
  `logsumexp(all_logits) - logsumexp(allowed_group_logits)`;
- cover coordinate, schema, desc, and eos groups;
- assert fp32 selected-logit objective math.

### Wave 8 / Wave 10: Vertical Smoke Acceptance

V1 smoke must prove more than unit tests:

- resolved config;
- `run_manifest.json`;
- `resolved_step_schedule.json`;
- Qwen setup receipt;
- pack plan;
- loss plan;
- trainable-surface receipt;
- optimizer-group receipt;
- two scheduled `eval.forward` runs;
- multi-segment packed forward receipt;
- metric events;
- eval summaries;
- checkpoint metadata;
- `checkpoints/checkpoint-final.json`;
- actual artifact files, not only manifest links;
- debug evidence for MRoPE resets, FA2 splits, same-segment causal mapping, and
  `segment_balanced` denominator behavior.

## Recommended New Tests

Data/template/encoding:

- `test_raw_example_rejects_zero_or_multi_image`
- `test_example_id_preserved_through_render_encode_pack`
- `test_current_len12000_row_normalizes_to_canonical_raw_example`
- `test_raw_dimensions_required_before_qwen_encoding`
- `test_geometry_preserves_xyxy_coord_order_and_object_order`
- `test_renderer_rejects_crossing_or_uncovered_loss_spans`
- `test_assistant_suffix_supervises_im_end_but_ignores_newline`
- `test_no_resize_admissible_dimensions_derive_from_processor_patch_merge`
- `test_no_resize_invalid_dimensions_fail_before_processor_forward`

Config/runtime:

- `test_strict_config_rejects_unknown_fields_before_mutation`
- `test_child_list_replaces_parent_list`
- `test_legacy_cadence_aliases_fail`
- `test_effective_batch_division_must_be_exact`
- `test_epoch_tail_fill_does_not_drop_or_partial_update`

Packing/forward:

- `test_pack_plan_commits_on_overflow_and_never_truncates`
- `test_logical_to_physical_supervision_mapping_is_invertible`
- `test_causal_shift_rejects_first_token_of_later_segment`
- `test_mrope_position_ids_reset_at_segment_boundaries_and_row_count_matches_qwen`
- `test_fa2_receipt_contains_cu_seq_lens_max_lengths_and_segment_count`

Supervision/losses:

- `test_base_token_ce_matches_dense_reference_for_selected_atoms`
- `test_token_type_gate_loss_uses_exact_logsumexp_group_mass`
- `test_segment_balanced_differs_from_token_balanced_on_unequal_segments`

Training/artifacts:

- `test_metric_events_store_split_and_name_separately`
- `test_eval_forward_summary_uses_unpadded_planned_step_path`
- `test_checkpoint_final_alias_and_step_dirs_are_unpadded`

Fixture checks:

- source path and row id;
- image checksum;
- raw image dimensions;
- processor `patch_size` and `merge_size`;
- `image_grid_thw`;
- merged visual-token count;
- expected object order;
- expected assistant text;
- expected typed spans;
- expected two-example pack boundaries.

## Findings

- **P0:** Do not archive old `src/` until this inventory is reviewed and
  available to Wave 2.
- **P0:** MRoPE row-count authority must come from installed Qwen source study,
  not legacy tests. Port reset semantics, not stale row shape.
- **P1:** No-resize has strong contract text but needs clean V1 unit coverage
  for processor-derived divisibility and budget checks.
- **P1:** Raw-example identity, single-image validation, source-row
  normalization, strict config, planned-step schedule, and effective-batch
  invariants must be inventoried before old `src/` is archived.
- **P1:** Loss denominator semantics are easy to regress and need direct
  `segment_balanced` tests.
- **P1:** Vertical smoke acceptance must include resolved config, manifest,
  schedule, setup, pack, loss, trainable, optimizer, eval, metric, checkpoint,
  and debug evidence artifacts, not only core outputs.
- **P2:** Artifact/metric compatibility should be narrowed to V1 names and
  paths rather than preserving legacy aliases wholesale.

## Status

Task 2.9 has reviewed source-study evidence and is accepted as the Wave 2
archive prerequisite inventory. This does not authorize moving old `src/`; the
Wave 2 archive/skeleton step still needs its own execution gate.
