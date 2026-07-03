---
doc_id: progress.diagnostics.realized_behavior_bridge_findings
layer: progress
doc_type: diagnostic-findings
status: active-evidence
evidence_scope: val200-16case-prefix-continuation-and-86state-hidden-readout
domain: autoregressive-binding-template-ablation
updated: 2026-06-20
---

# Realized Behavior Bridge Findings

## Scope

This note records Task 5 state for the checkpoint-928 autoregressive binding
template bridge. It now includes deterministic short-continuation evidence over
the 16-case bridge set and readout-only hidden/LM-head-input probes over 86
box-end trajectory states. It does not claim a causal activation patch target,
a training recommendation, or a complete mechanism explanation.

## Resolved Design Decisions

- Stage A remains `realized_behavior_bridge`: deterministic short
  continuations with prefix-side guidance and controls only.
- Stage B `latent_behavior_patch` remains blocked until real next-object
  behavior movement beats controls outside discovery.
- No raw hidden-state activation patching, attention patching, model weight
  updates, or micro-training are in scope for this branch state. The boundary
  patch artifact below is a readout-only final-normed LM-head-input lens.
- The current bridge materialization enriches cases from
  `candidate_step_rows_matched_target_only.jsonl` so coordinate guidance uses
  concrete target `candidate_x1,candidate_y1,candidate_x2,candidate_y2`
  values. Coordinate-bearing target-guidance arms are not considered `ready`
  when target bbox payloads are missing or malformed.
- Promotion must be conservative: behavior rows, target-guided rows, control
  rows, outside-discovery movement, mostly-null controls, parse preservation,
  repeated target-guided effects, and at least one family with three or more
  nontrivial `changed_to_target_gt` cases are all required before latent
  patching can be recommended.

## Artifact Root

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge
```

Current materialized artifacts at that root:

```text
bridge_case_manifest.json
bridge_case_manifest.md
rendered_prefix_interventions.jsonl
rendered_prefix_interventions_summary.json
executable_prefix_interventions.jsonl
executable_prefix_interventions_summary.json
run_manifest.json
shards/shard_000.jsonl
shards/shard_001.jsonl
shards/shard_002.jsonl
shards/shard_003.jsonl
raw_prefix_continuation_rows.jsonl
raw_prefix_continuation_summary.json
prefix_continuation_scored/prefix_continuation_behavior_rows.jsonl
prefix_continuation_scored/prefix_continuation_behavior_summary.json
prefix_continuation_scored/prefix_continuation_case_event_rows.jsonl
prefix_continuation_scored/prefix_continuation_control_rows.jsonl
prefix_continuation_scored/prefix_continuation_promotion_gate_summary.json
prefix_continuation_guidance_matrix/guidance_separability_matrix.json
prefix_continuation_guidance_matrix/guidance_separability_matrix.md
next_token_readout_shard_000/prefix_next_token_readout_rows.jsonl
next_token_readout_shard_001/prefix_next_token_readout_rows.jsonl
next_token_readout_shard_002/prefix_next_token_readout_rows.jsonl
next_token_readout_shard_003/prefix_next_token_readout_rows.jsonl
prefix_next_token_readout_rows.jsonl
prefix_next_token_readout_summary.json
prefix_next_token_readout_analysis/prefix_next_token_readout_behavior_joined_rows.jsonl
prefix_next_token_readout_analysis/prefix_next_token_readout_analysis_summary.json
prefix_next_token_readout_analysis/prefix_next_token_readout_analysis.md
trajectory_readout_shard_000/prefix_greedy_trajectory_readout_rows.jsonl
trajectory_readout_shard_001/prefix_greedy_trajectory_readout_rows.jsonl
trajectory_readout_shard_002/prefix_greedy_trajectory_readout_rows.jsonl
trajectory_readout_shard_003/prefix_greedy_trajectory_readout_rows.jsonl
prefix_greedy_trajectory_readout_rows.jsonl
prefix_greedy_trajectory_readout_summary.json
prefix_greedy_trajectory_analysis.json
prefix_greedy_trajectory_analysis.md
trajectory_hidden_state_probe/selected_smoke_rows.jsonl
trajectory_hidden_state_probe/smoke_layers_last4/trajectory_hidden_state_readout_rows.jsonl
trajectory_hidden_state_probe/smoke_layers_last4/trajectory_hidden_state_readout_summary.json
trajectory_hidden_state_probe/full_coord_boxend_shards/trajectory_hidden_state_readout_rows.jsonl
trajectory_hidden_state_probe/full_coord_boxend_shards/trajectory_hidden_state_analysis.json
trajectory_hidden_state_probe/full_coord_boxend_shards/trajectory_hidden_state_analysis.md
trajectory_hidden_state_probe/full_coord_boxend_shards_normed_lens/trajectory_hidden_state_readout_rows.jsonl
trajectory_hidden_state_probe/full_coord_boxend_shards_normed_lens/trajectory_hidden_state_readout_summary.json
trajectory_hidden_delta_probe/full_boxend_shards_normed_lens/trajectory_hidden_delta_readout_rows.jsonl
trajectory_hidden_delta_probe/full_boxend_shards_normed_lens/trajectory_hidden_delta_readout_summary.json
trajectory_hidden_delta_probe/full_boxend_shards_normed_lens/trajectory_hidden_delta_analysis.md
trajectory_hidden_delta_probe/full_boxend_shards_joint_boundary_projection/trajectory_hidden_delta_readout_rows.jsonl
trajectory_hidden_delta_probe/full_boxend_shards_joint_boundary_projection/trajectory_hidden_delta_readout_summary.json
trajectory_hidden_delta_probe/full_boxend_shards_joint_boundary_projection/trajectory_hidden_delta_analysis.md
trajectory_hidden_delta_probe/full_boxend_shards_joint_boundary_projection/trajectory_hidden_delta_causal_join_rows.jsonl
trajectory_hidden_delta_probe/full_boxend_shards_joint_boundary_projection/trajectory_hidden_delta_causal_join_summary.json
trajectory_hidden_delta_probe/full_boxend_shards_joint_boundary_projection/trajectory_hidden_delta_causal_join_analysis.md
trajectory_boundary_routing_selection/joint_boundary_probe_seed_v1/trajectory_boundary_routing_selected_rows.jsonl
trajectory_boundary_routing_selection/joint_boundary_probe_seed_v1/trajectory_boundary_routing_selection_summary.json
trajectory_boundary_routing_selection/joint_boundary_probe_seed_v1/trajectory_boundary_routing_selection.md
trajectory_boundary_anchor_attention/smoke_1row_last_layer/trajectory_boundary_anchor_attention_rows.jsonl
trajectory_boundary_anchor_attention/smoke_1row_last_layer/trajectory_boundary_anchor_attention_summary.json
trajectory_boundary_anchor_attention/smoke_1row_last_layer/trajectory_boundary_anchor_attention.md
trajectory_boundary_anchor_attention/full_selected_late4_v1/trajectory_boundary_anchor_attention_rows.jsonl
trajectory_boundary_anchor_attention/full_selected_late4_v1/trajectory_boundary_anchor_attention_summary.json
trajectory_boundary_anchor_attention/full_selected_late4_v1/trajectory_boundary_anchor_attention.md
trajectory_boundary_anchor_attention/full_selected_all_layers_v1/trajectory_boundary_anchor_attention_rows.jsonl
trajectory_boundary_anchor_attention/full_selected_all_layers_v1/trajectory_boundary_anchor_attention_summary.json
trajectory_boundary_anchor_attention/full_selected_all_layers_v1/trajectory_boundary_anchor_attention.md
trajectory_hidden_boundary_patch_probe/full_boxend_shards_normed_head/trajectory_hidden_boundary_patch_rows.jsonl
trajectory_hidden_boundary_patch_probe/full_boxend_shards_normed_head/trajectory_hidden_boundary_patch_summary.json
trajectory_hidden_boundary_patch_probe/full_boxend_shards_normed_head/trajectory_hidden_boundary_patch_analysis.md
```

The scored prefix-continuation rows are not official detector evaluation rows.
They are local bridge rows parsed from short assistant-prefix continuations and
matched to canonical GT bboxes by IoU when the pair config is supplied.

## Input Handles

- Phase 3 adjusted input root:
  `/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase3_audit_adjusted`
- Required source rows are the Phase 3 candidate-only and audit-adjusted rows
  named by `docs/superpowers/plans/2026-06-19-realized-behavior-bridge.md`.
- Candidate target enrichment source:
  `/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase3_audit_adjusted/candidate_step_rows_matched_target_only.jsonl`
- Config handle:
  `configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml`

## Case Manifest

`bridge_case_manifest.json` currently contains 16 cases over:

- families: `desc_first`, `geometry_first`
- splits: `discovery`, `reserve`, `val200_remainder`
- selection events: `desc_end`, `pre_box_start`, `pre_desc`, `pre_x1`

This is a manifest and routing artifact, not realized model behavior evidence.
The regenerated manifest records `input_paths.candidate_rows` and the 16
selected cases currently have non-missing `target_desc` and integer
`target_bbox` payloads. The first case is
`desc_first-139-0-0-desc_end`, `clock`, `[699, 284, 722, 336]`.

## Intervention Arms

`rendered_prefix_interventions.jsonl` currently contains 208 planned rows: 16
cases times 13 intervention arms. The rendered dry-run summary counts:

- `target_guided`: 64 rows
- `wrong_control`: 80 rows
- `decode_control`: 32 rows
- `noop`: 16 rows
- `timing_control`: 16 rows

These rows are dry-run prefix intervention specifications. They are not model
continuations. Current guidance payload status counts are:

- `ready`: 48 rows
- `metadata_only`: 80 rows
- `unresolved`: 80 rows

Current artifact integrity check found zero coordinate-bearing target-guided
rows marked `ready` with a null bbox.

## Executable Prefix Materialization

`executable_prefix_interventions.jsonl` materializes the dry-run rows against
the actual val200 token traces and confidence rows from both checkpoint-928
families. This is still pre-GPU prefix materialization, not a model
continuation.

Current materialization summary:

- row count: 208
- case count: 16
- executable prefix count: 96
- `ready`: 18 rows
- `partial_slot_guidance`: 14 rows
- `ready_noop`: 64 rows
- `slot_incompatible`: 16 rows
- `unimplemented_guidance_payload`: 16 rows
- `unresolved_guidance_payload`: 80 rows

Interpretation boundary: descriptor-only target guidance is executable only for
the two `desc_first/pre_desc` windows in this case set. Most descriptor-only
arms are correctly marked `slot_incompatible` because the chosen windows are
coordinate-side or post-description. Coordinate/x1 guidance is the executable
target-guided surface for most selected windows. `target_object_start_seed`
is explicitly non-executable for now because this branch does not yet implement
a nonempty object-start guidance mutation; it must not be interpreted as an
object-start intervention result.

## Smoke And Scaled Execution

`run_manifest.json` currently records `stage: execution-manifest`, 96
executable-prefix rows after filtering from 208 materialized rows, 16 cases, 4
GPU shards, and GPU ids `[0, 1, 2, 3]`. Each shard has 24 runnable rows and
zero non-executable placeholder rows.

The branch now has a guarded `execute-prefix-smoke` backend for short greedy
continuations from executable assistant prefixes. It uses the checkpoint pair
config, validates the `compact_object_box_closed` token_embeddings_adapter
surface, resolves images from the canonical dataset rows, and records raw tails
before a separate scorer materializes behavior rows.

Tiny GPU smoke artifacts:

- desc-first, GPU 0:
  `/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/smoke_desc_first_2rows`
- geometry-first, GPU 1:
  `/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/smoke_geometry_first_2rows`

Both smoke runs used `max_new_tokens=8`, `self_noop,target_coord_seed`, two
rows, and reported `decode_status_counts: {"ok": 2}`. This proves the model
load, chat-template continuation, image path resolution, and token-embeddings
adapter surface are operational for both families.

Full prefix-continuation run:

- shard roots:
  - `prefix_continuation_shard_000`
  - `prefix_continuation_shard_001`
  - `prefix_continuation_shard_002`
  - `prefix_continuation_shard_003`
- devices: GPUs `0,1,2,3`
- shard row counts: 24 each
- merged raw row count: 96
- case count: 16
- decode status counts: `{"ok": 96}`
- families: `{"desc_first": 72, "geometry_first": 24}`
- intervention arms:
  - `self_noop`: 16
  - `target_coord_seed`: 14
  - `target_desc_plus_coord_seed`: 16
  - `target_desc_seed`: 2
  - `post_commit_too_late`: 16
  - `repetition_penalty_on`: 16
  - `repetition_penalty_off`: 16

## Behavior Summary

`score-prefix-continuations` compares each row to its same
`case_id,family,selection_event` `self_noop` baseline. The parser is strict for
the `compact_object_box_closed` template: a completed object span must include
`<|box_end|>`. This intentionally separates coordinate-slot nudges from rows
that did not complete the closed object grammar in the short continuation.

Scored output:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/prefix_continuation_scored/prefix_continuation_behavior_rows.jsonl
```

Summary counts:

- row count: 96
- control rows: 64
- target-guided rows: 32
- `changed_next_object_behavior`: 4
- `changed_to_target_gt`: 2
- `duplicate_to_new_gt`: 0
- `unmatched_to_target_gt`: 0
- `stop_or_invalid_to_valid`: 2
- `valid_to_invalid_regression`: 2
- `changed_emitted_bbox`: 25
- `changed_emitted_desc`: 4
- target bbox IoU improved: 16
- target bbox IoU regressed: 5
- target bbox IoU unchanged: 37
- after target bbox IoU >= 0.50: 43
- after target bbox IoU >= 0.70: 39
- parse-failure kinds:
  - `incomplete_or_missing_object`: 32
  - `invalid_geometry`: 4

Arm-level sketch:

| arm | rows | after parse ok | changed to target GT | after hits target | changed bbox | IoU improved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `self_noop` | 16 | 10 | 0 | 7 | 0 | 0 |
| `post_commit_too_late` | 16 | 10 | 0 | 7 | 0 | 0 |
| `repetition_penalty_on` | 16 | 10 | 0 | 7 | 0 | 0 |
| `repetition_penalty_off` | 16 | 8 | 0 | 6 | 4 | 1 |
| `target_coord_seed` | 14 | 10 | 1 | 7 | 10 | 7 |
| `target_desc_plus_coord_seed` | 16 | 11 | 1 | 8 | 11 | 8 |
| `target_desc_seed` | 2 | 1 | 0 | 1 | 0 | 0 |

Guidance matrix over the scored rows:

- case count: 16
- `coord_only`: 1
- `none`: 10
- `invalid_or_unparseable`: 5
- `control_contaminated`: 0
- `desc_only`: 0
- `desc_and_coord_independent`: 0
- `desc_coord_joint_only`: 0
- `object_start_only`: 0

## Prefix Next-Token Readout

The branch now has a read-only `prefix-next-token-readout` stage for executable
assistant prefixes. It loads the same checkpoint pair config and
`compact_object_box_closed` token-embeddings-adapter surface as the short
continuation run, runs one forward pass at the final prompt token, and records
next-token logits/probabilities without sampling, hidden-state patching,
attention patching, training, or model perturbation.

Execution scope:

- shard roots: `next_token_readout_shard_000` through
  `next_token_readout_shard_003`
- devices: GPUs `0,1,2,3`
- merged readout rows: 96
- case count: 16
- `readout_status_counts`: `{"ok": 96}`
- `target_next_token_observed_count`: 96
- `target_next_kind` counts:
  - `box_start`: 48
  - `coord`: 38
  - `desc`: 8
  - `object_ref_end`: 2

The readout-analysis stage joins all 96 next-token rows to the scored short
continuation rows:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/prefix_next_token_readout_analysis/prefix_next_token_readout_analysis.md
```

Join status: `{"matched": 96}`.

Key aggregate readouts:

| slice | rows | target prob | coord mass | target coord prob | r4 mass | target coord rank | box_end prob | parse ok | changed target | bbox delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 96 | 0.580002 | 0.395605 | 0.060804 | 0.529785 | 7.342 | 0.00000627 | 0.625 | 0.0208 | 0.0141 |
| `self_noop` | 16 | 0.809759 | 0.124762 | 0.056870 | 0.430936 | 3.000 | 0.00000940 | 0.625 | 0.0000 | 0.0000 |
| `target_coord_seed` | 14 | 0.059522 | 0.999761 | 0.059522 | 0.554364 | 8.714 | 0.000000000494 | 0.714 | 0.0714 | 0.0430 |
| `target_desc_plus_coord_seed` | 16 | 0.063892 | 0.999791 | 0.063892 | 0.557701 | 8.313 | 0.000000000432 | 0.688 | 0.0625 | 0.0428 |
| `target_desc_seed` | 2 | 0.999992 | ~0 | n/a | n/a | n/a | 0.000000032 | 0.500 | 0.0000 | 0.0000 |

By local target kind:

- `box_start` rows are locally schema-confident:
  mean target probability `0.962969`, mean box-start rank `1.0`.
- `desc` rows are locally semantic-token confident:
  first descriptor token mean probability `0.643391`, rank `1` in all eight
  rows. The eight rows are `clock` or `person` first-token targets.
- `object_ref_end` rows are locally closure-confident:
  mean target probability `0.999992`, mean object-ref-end rank `1.0`.
- `coord` rows are slot-confident but exact-bin diffuse:
  mean coordinate mass `0.999423`, mean target coordinate probability
  `0.060804`, mean target-coordinate rank `7.342`, and mean radius-4 mass
  `0.529785`.

The two `changed_to_target_gt` rows both occur under coordinate-bearing target
guidance, but neither has a strong first target-coordinate token:

- `desc_first-139-0-9-desc_end / target_coord_seed`:
  target coord probability `0.008947`, rank `16`, radius-4 mass `0.393162`.
- `desc_first-139-0-9-desc_end / target_desc_plus_coord_seed`:
  same first-token readout values.

This is a useful negative result against a simple first-token steering story.
The realized target transition appears to come from a local coordinate-basin
trajectory over the following greedy chain, not from immediate high probability
on the exact next target coordinate token.

## Prefix Greedy Trajectory Readout

The branch now has a second read-only probe,
`prefix-greedy-trajectory-readout`. It repeatedly runs the same one-token
forward readout, appends the greedy decoded token to the assistant prefix, and
recomputes the next local target spec. This records how the model walks through
descriptor, coordinate, box-end, and next-object-start states without sampling,
hidden-state patching, attention patching, training, or model perturbation.

Execution scope:

- shard roots: `trajectory_readout_shard_000` through
  `trajectory_readout_shard_003`
- devices: GPUs `0,1,2,3`
- seed rows: 96 executable prefixes
- trajectory max steps: 6
- merged trajectory-step rows: 556
- case count: 16
- `readout_status_counts`: `{"ok": 556}`
- stop reasons:
  - `continue`: 460
  - `max_steps`: 69
  - `im_end`: 2
  - `schema_break_open_box`: 25
- validity checks after the open-box repair:
  - `malformed_open_box_target_rows`: 0
  - `malformed_open_box_prefix_rows`: 0
  - `post_schema_break_rows`: 0
  - `open_box_invalid_next_rows`: 25
  - `open_box_invalid_next_not_schema_break_rows`: 0

Durable grouped analysis:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/prefix_greedy_trajectory_analysis.md
```

Key trajectory readouts:

| slice | rows | greedy match | target prob | coord prob | coord rank | r4 mass | top dist | box-end prob | greedy box-end | greedy object-start |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `target_next_kind=coord` | 350 | 0.200 | 0.0913 | 0.0913 | 9.40 | 0.502 | 8.26 | ~0 | 0 | 0 |
| `target_next_kind=box_end` | 86 | 0.709 | 0.671 | n/a | n/a | n/a | n/a | 0.671 | 61 | 19 |
| `target_next_kind=box_start` | 64 | 0.938 | 0.908 | n/a | n/a | n/a | n/a | 0.000014 | 0 | 2 |
| `target_next_kind=desc` | 21 | 0.762 | 0.583 | n/a | n/a | n/a | n/a | ~0 | 0 | 0 |
| `target_next_kind=object_ref_end` | 10 | 1.000 | 0.999992 | n/a | n/a | n/a | n/a | ~0 | 0 | 0 |
| `target_next_kind=object_ref_start` | 25 | 0.680 | 0.695 | n/a | n/a | n/a | n/a | 0.000003 | 0 | 17 |

The dominant target-kind path is now explicit:

```text
box_start -> coord -> coord -> coord -> coord -> box_end
```

It appears for 48 seed trajectories. This supports the local schema reading:
the model strongly knows when it is in a coordinate slot, and it often knows
that box closure should come after four coordinate tokens. The failure is not
that `<|box_end|>` is globally unavailable; the failure is that the coordinate
trajectory can land in a basin where the closure gate loses to next-object or
descriptor-boundary tokens. Those events are now represented explicitly as
`schema_break_open_box` terminal rows rather than allowed to create misleading
post-break target states.

Arm-level contrast:

- `self_noop`/decode/timing controls have `coord` greedy-match rate about
  `0.194` and `box_end` greedy-match rate about `0.714`.
- `target_coord_seed` has `coord` greedy-match rate about `0.205`, but
  `box_end` greedy-match rate about `0.714`.
- `target_desc_plus_coord_seed` has `coord` greedy-match rate about `0.220`,
  and `box_end` greedy-match rate about `0.688`.

So coordinate guidance does not solve exact coordinate identity at the next
token. It keeps the model inside coordinate-type space and nudges emitted boxes,
while the later closure boundary remains a separate competition between
`<|box_end|>` and object/descriptor structural tokens. The relevant mechanism is
therefore a coupled coordinate-basin / boundary-state trajectory, not a single
target-coordinate logit or a single missing structural-token logit.

## Trajectory Hidden-State Readout

The branch now has an observational hidden-state readout stage,
`trajectory-hidden-state-readout`. It consumes corrected greedy trajectory rows,
replays the exact assistant prefix for each selected state, requests
`output_hidden_states=True`, and projects selected hidden layers through the
current model head. It records full-vocab target/greedy/schema-token logits,
coord-token mass, coord-only target geometry, and effective output-row
coordinate surface metrics for the `token_embeddings_adapter` checkpoint
surface. It does not patch hidden states, patch attention, sample, train, or
change model weights.

Smoke artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_state_probe/smoke_layers_last4
```

The smoke used 8 manually balanced trajectory states (`coord` interior,
ordinary `box_end`, and `schema_break_open_box`) over layers `-1,-4`, producing
16 layer rows with `readout_status_counts={"ok": 16}`. It caught and fixed one
real runtime issue: intermediate hidden vectors must be cast to the lm-head
weight dtype/device before a bf16 lens projection.

Broader readout artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_state_probe/full_coord_boxend_shards_normed_lens
```

This supersedes the earlier exploratory pre-norm root:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_state_probe/full_coord_boxend_shards
```

Execution scope:

- selected states: all corrected trajectory states with
  `target_next_kind in {"coord", "box_end"}`
- selected state count: 436
- layer rows: 872
- layers: `-1`, `-4`
- cases: 16
- shard roots: `readout_shard_000` through `readout_shard_003`
- devices: GPUs `0,1,2,3`
- `readout_status_counts`: `{"ok": 872}`
- `counts_by_readout_projection_space`:
  `{"lm_head_input_after_final_norm": 872}`

Merged rows and summary:

```text
trajectory_hidden_state_readout_rows.jsonl
trajectory_hidden_state_readout_summary.json
```

Key normed hidden-state readouts:

| slice | layer | states | target-greedy logit | box_end-object_start logit | coord mass | target prob | target rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `box_end / continue` | -1 | 29 | 0.0 | 6.9655 | 0.0 | 0.8925 | 1.0 |
| `box_end / max_steps` | -1 | 32 | 0.0 | 8.8516 | 0.0 | 0.9392 | 1.0 |
| `box_end / schema_break_open_box` | -4 | 25 | 1.5363 | 1.8738 | 0.0005 | 0.0049 | 15.4 |
| `box_end / schema_break_open_box` | -1 | 25 | -2.3150 | -2.1850 | 0.0 | 0.0723 | 3.24 |
| `coord / continue` | -1 | 336 | -0.9196 | -1.9827 | 0.9994 | 0.0917 | 9.60 |
| `box_end / schema_break_open_box` | -1 | 25 | -43.600 | -35.920 | 0.0 | n/a | n/a | n/a | 0.0 | 4.16 |
| `box_end / schema_break_open_box` | -4 | 25 | 25.780 | 35.180 | 0.0 | n/a | n/a | n/a | 0.0 | 802.28 |
| `coord / continue` | -1 | 336 | -23.554 | -53.088 | 0.627 | 54.35 | 54.10 | 58.22 | 0.124 | 667.83 |
| `coord / max_steps` | -1 | 14 | -3.286 | -14.018 | 0.643 | 4.50 | 4.50 | 52.44 | 0.141 | 618.57 |

Interpretation with scope label `hidden_state_readout_val200_436states`:

- Clean `box_end` states and max-step `box_end` states have very large positive
  final-layer `box_end - object_ref_start` margins. The model is not missing a
  global close-box capability at those states.
- `schema_break_open_box` states invert late: at layer `-4`, the same rows still
  have positive `box_end - object_ref_start` margin (`+35.18`), but at the final
  layer the margin is negative (`-35.92`) and greedy chooses a structural
  continuation. This makes the closure failure look like a late boundary-state
  overwrite rather than an absent early representation of `<|box_end|>`.
- Coordinate states show high final-layer coord-type mass (`0.627` for ordinary
  continuing coord rows), but exact coordinate identity remains diffuse:
  ordinary continuing coord rows have mean final-layer coord top1 distance about
  `54` bins and target rank about `668`. The `token_embeddings_adapter` surface
  contributes large positive target-coordinate deltas, but the delta sharpens a
  coordinate basin more than it pins the exact target bin.
- Target-guided coordinate rows move the final-layer coordinate top1 closer on
  this readout (`target_desc_plus_coord_seed` about `43.6` bins vs
  `self_noop`/decode/timing controls about `56.3` bins), but still not enough
  to make exact coordinate identity reliable.

This is a stronger bridge into the next microscope: compare hidden-state
directions between layer `-4` and final layer for matched clean `box_end` and
`schema_break_open_box` states. The candidate mechanism is now a late
boundary-gate rotation/overwrite that can defeat an earlier close-box-favorable
state, plus a separate coordinate-basin attraction that is type-correct but
weakly pinned to exact bins.

## Controls

Executed controls include noop, too-late timing, and repetition-penalty arms.
Wrong-object, shuffled-label, and nearby-wrong-coordinate controls remain
non-executable in this materialization because their guidance payloads are not
resolved yet.

Under the corrected prefix-continuation transition semantics,
`changed_to_target_gt` means that the arm moves from a non-target `self_noop`
baseline to the target after intervention. With that definition:

- control changed-to-target count: 0
- control changed-to-target rate: 0.0
- controls mostly null: true
- missing control family: `wrong_control`

## Promotion Gate

The promotion gate must not recommend latent behavior patching yet.

Current promotion decision:

```text
promote_to_latent_behavior_patch: false
changed_to_target_gt_target_guided_count: 2
changed_to_target_gt_control_count: 0
control_changed_to_target_gt_rate: 0.0
parse_preserved_rate_target_guided: 0.625
primary_blockers:
  - target_guided_parse_preserved_rate_below_threshold
  - target_guided_changed_to_target_gt_below_threshold
  - no_effect_visible_outside_discovery
  - missing_control_families
  - no_family_with_at_least_three_changed_to_target_gt
```

## Interpretation

The branch crossed the first realized-behavior bridge, but the result is not a
promotion-ready latent patch. The stronger observation is narrower and more
mechanistic:

- Coordinate-bearing guidance often nudges the emitted bbox toward the target
  basin without changing the descriptor.
- Only two target-guided rows transition from non-target baseline to target,
  both in the discovery split.
- Controls are mostly null under the transition definition, but parse
  preservation is too weak and wrong-control rows are not yet executable.
- Strict closed-template scoring exposes grammar-closure fragility: several
  continuations produce plausible coordinate slots but do not emit a complete
  closed object span within the short continuation.

This points toward a coordinate-slot attraction / closed-template completion
problem rather than a clean object-identity steering mechanism. The one-step
readout sharpens that into three claims:

- Local schema states are mostly well understood: descriptor starts, object-ref
  closure, and box starts can all be rank-1/high-probability when the prefix is
  at that local grammar boundary.
- Coordinate guidance creates a very strong coordinate-type basin but only a
  weak exact-coordinate target. The mass is spatially nearby rather than pinned
  to the target bin.
- Closed-object behavior fails downstream of local type knowledge. At open
  coordinate prefixes, `<|box_end|>` is essentially unavailable; after four
  coordinate emissions it often becomes high-rank/high-probability, but
  coordinate-guided trajectories can still jump to `<|object_ref_start|>` or
  other continuation modes. Parse failures and bbox movement should therefore
  be studied as coupled coordinate-basin trajectories and boundary-shift
  dynamics, not as a single missing boundary logit at the seeded prefix.

This is a useful bridge to deeper hidden-state and attention analysis,
especially around coordinate basin attraction, object-pointer state movement
through x1/y1/x2/y2, `<|box_end|>` closure after coordinate emission, and the
semantic/geometry split in pre-desc versus desc-end windows.

## Trajectory Hidden Delta Readout

The branch now has a paired-layer hidden-delta readout stage,
`trajectory-hidden-delta-readout`. It replays a greedy trajectory state before
the next token, captures two hidden layers in one forward pass, applies the
model's final language-model norm before LM-head projection, and records only
scalar margins, flip booleans, and hidden-delta direction alignments. No hidden
vectors are written.

Primary artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_delta_probe/full_boxend_shards_normed_lens
```

This supersedes the earlier exploratory pre-norm root:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_delta_probe/full_boxend_shards
```

Merged files:

```text
trajectory_hidden_delta_readout_rows.jsonl
trajectory_hidden_delta_readout_summary.json
trajectory_hidden_delta_analysis.md
```

Scope label: `hidden_delta_normed_lens_val200_boxend_86states`.

Run shape:

- source trajectory rows:
  `/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/prefix_greedy_trajectory_readout_rows.jsonl`
- selected states: `target_next_kind=box_end`
- selected row count: 86
- hidden layer pair: `source=-4`, `target=-1`
- shard count: 8
- readout status: `{"ok": 86}`
- model perturbation: false
- training: false

Key aggregate result:

| stop_reason | rows | greedy token family | source `box_end-object_ref_start` | target `box_end-object_ref_start` | delta | flip rate | hidden delta projection to `object_ref_start-box_end` |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `continue` | 29 | all `<|box_end|>` | 3.4698 | 6.9655 | 3.4957 | 0.0 | -59.0292 |
| `max_steps` | 32 | all `<|box_end|>` | 4.6523 | 8.8516 | 4.1992 | 0.0 | -84.6909 |
| `schema_break_open_box` | 25 | 19 `<|object_ref_start|>`, 6 `<|object_ref_end|>` | 1.8738 | -2.1850 | -4.0588 | 1.0 | 106.959 |

Failure-vs-clean separation:

- failure label: `box_end_schema_break_open_box`
- clean label: `box_end_continue_or_max_steps_greedy_box_end`
- failure rows: 25
- clean rows: 61
- failure pos-to-neg flip rate for
  `box_end - object_ref_start`: 1.0
- clean pos-to-neg flip rate for `box_end - object_ref_start`: 0.0
- failure mean delta for `box_end - object_ref_start`: -4.0588
- clean mean delta for `box_end - object_ref_start`: 3.8648
- failure-clean delta gap: -7.9235
- failure mean hidden-delta unit projection to
  `object_ref_start - box_end`: 106.959
- clean mean hidden-delta unit projection to
  `object_ref_start - box_end`: -72.4911

Interpretation with scope label
`hidden_delta_normed_lens_val200_boxend_86states`:
the failed close-box states are not missing a boundary-favorable intermediate
state. At layer `-4`, even the `schema_break_open_box` rows are still
box-end-favorable (`box_end - object_ref_start = +1.8738`). Between layer `-4`
and final layer `-1`, every failed row rotates/updates into an object-boundary
direction, while every clean row strengthens the box-end direction. This makes
the current best mechanism a late boundary-gate overwrite: the model has a
valid local close-box state, but final-layer dynamics can redirect that state
toward `<|object_ref_start|>` or `<|object_ref_end|>` before the next-token
distribution is read out.

This does not yet prove a causal intervention. It motivated a readout-only
LM-head-input patch lens, recorded next.

## Boundary Patch Readout

Authoritative artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_boundary_patch_probe/full_boxend_shards_normed_head
```

Rows and summaries:

```text
trajectory_hidden_boundary_patch_rows.jsonl
trajectory_hidden_boundary_patch_summary.json
trajectory_hidden_boundary_patch_analysis.md
```

This supersedes the earlier exploratory pre-norm artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_boundary_patch_probe/full_boxend_shards
```

The superseded artifact projected patched raw hidden vectors without applying
the model's final language-model norm first, and its baseline top tokens did
not faithfully reproduce the realized greedy clean box closures. The
authoritative artifact patches the final-normed LM-head input and records
`patch_space: lm_head_input_after_final_norm`.

Run scope:

- selected states: `target_next_kind=box_end`
- selected state count: 86
- hidden layer pair: `source=-4`, `target=-1`
- patch space: `lm_head_input_after_final_norm`
- patch variants per state: baseline, 4 source-interpolation alphas, 9
  `box_end - object_ref_start` direction strengths
- merged row count: 1204
- shard count: 8
- readout status: `{"ok": 1204}`
- model perturbation: false
- training: false

Baseline sanity:

| stop_reason | states | baseline top token |
| --- | ---: | --- |
| `continue` | 29 | 29 `<|box_end|>` |
| `max_steps` | 32 | 32 `<|box_end|>` |
| `schema_break_open_box` | 25 | 19 `<|object_ref_start|>`, 6 `<|object_ref_end|>` |

Direction-patch threshold over the 25 failed box-end states:

| patch | failed states rescued | clean states preserved | failure mean `box_end-object_ref_start` |
| --- | ---: | ---: | ---: |
| baseline | 0 / 25 | 61 / 61 | -2.185 |
| `add_box_end_direction_4` | 2 / 25 | 61 / 61 | 0.600 |
| `add_box_end_direction_8` | 15 / 25 | 61 / 61 | 3.355 |
| `add_box_end_direction_16` | 25 / 25 | 61 / 61 | 9.020 |
| `add_box_end_direction_128` | 25 / 25 | 61 / 61 | 87.600 |

Per-state minimum direction strength:

- strength `4`: 2 failure states
- strength `8`: 13 failure states
- strength `16`: 10 failure states

Source-interpolation control:

- alpha `0.25`: rescues 0 / 25 failures and harms 0 / 61 clean states
- alpha `0.50`: rescues 6 / 25 failures and harms 8 / 61 clean states
- alpha `0.75`: rescues 12 / 25 failures and harms 12 / 61 clean states
- alpha `1.00`: rescues 0 / 25 failures and harms 18 / 61 clean states

Interpretation with scope label
`lm_head_input_boundary_patch_readout_val200_boxend_86states`: the failure
states are close to a simple boundary-logit decision surface in final-normed
LM-head input space. A small positive movement along
`unit(box_end - object_ref_start)` flips all schema-break states by strength
16 while preserving every clean box closure in this selected set. Source-layer
interpolation is weaker and dirtier: it can rescue some failures, but it also
pulls clean states toward object-boundary or unrelated token basins. This
supports the late boundary-gate overwrite explanation more specifically: the
collapse is not a missing visual/semantic object representation at the close
box step; it is a final readout-space boundary competition that can be
redirected with a low-dimensional structural-token direction.

This is still not a causal activation patch. It does not run a patched raw
intermediate state through later transformer blocks. The next causal test must
patch an actual activation or residual component and replay the remaining
model computation, then compare to this readout-space threshold curve.

## Next Branch

Continue with the planned deeper mechanism work before any Stage B causal patch
claim:

1. Convert the readout-space boundary direction into a causal activation test:
   patch or preserve a raw layer `-4` close-box component through later blocks
   for matched `schema_break_open_box` rows, then check whether greedy
   next-token emission switches from object-boundary markers back to
   `<|box_end|>` without simply forcing the LM-head readout.
2. Compare pre/post x1, y1, x2, y2, box-end, and next-object-start hidden states
   for the one clean `coord_only` case and matched `none`/`invalid` cases.
3. Investigate whether target-coordinate guidance sharpens geometry while
   leaving semantic identity unchanged, or whether it also moves object-pointer
   mass in hidden/attention space. The trajectory readout says the next
   microscope should include boundary-state movement, because coordinate-guided
   rows can reshape the later box-end / next-object competition even when bbox
   IoU improves.
4. Resolve or implement wrong-control executable payloads before any promotion
   claim.

Do not implement latent hidden-state patching, attention patching, or
micro-training from this note alone.

## Verification

Commands run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -q -p no:cacheprovider

python scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py \
  --stage score-prefix-continuations \
  --prefix-continuation-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/raw_prefix_continuation_rows.jsonl \
  --pair-config configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml \
  --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/prefix_continuation_scored

python scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py \
  --stage guidance-matrix \
  --scored-behavior-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/prefix_continuation_scored/prefix_continuation_behavior_rows.jsonl \
  --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/prefix_continuation_guidance_matrix

python scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py \
  --stage prefix-next-token-readout \
  --executable-prefix-interventions /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/shards/shard_000.jsonl \
  --pair-config configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml \
  --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/next_token_readout_shard_000 \
  --device cuda:0 \
  --allow-model-load

# Same command shape was run for shard_001/cuda:1, shard_002/cuda:2, and shard_003/cuda:3.

python scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py \
  --stage analyze-prefix-next-token-readout \
  --prefix-next-token-readout-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/prefix_next_token_readout_rows.jsonl \
  --scored-behavior-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/prefix_continuation_scored/prefix_continuation_behavior_rows.jsonl \
  --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/prefix_next_token_readout_analysis

python scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py \
  --stage prefix-greedy-trajectory-readout \
  --executable-prefix-interventions /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/shards/shard_000.jsonl \
  --pair-config configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml \
  --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_readout_shard_000 \
  --device cuda:0 \
  --trajectory-steps 6 \
  --allow-model-load

# Same command shape was run for shard_001/cuda:1, shard_002/cuda:2, and shard_003/cuda:3.

git diff --check

python -m py_compile \
  src/analysis/autoregressive_binding_template_ablation/realized_behavior_bridge.py \
  scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py

python scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py \
  --stage trajectory-hidden-state-readout \
  --prefix-greedy-trajectory-readout-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_state_probe/selected_smoke_rows.jsonl \
  --pair-config configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml \
  --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_state_probe/smoke_layers_last4 \
  --device cuda:0 \
  --hidden-layers=-1,-4 \
  --allow-model-load

python scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py \
  --stage trajectory-hidden-state-readout \
  --prefix-greedy-trajectory-readout-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_state_probe/full_coord_boxend_shards/shard_000.jsonl \
  --pair-config configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml \
  --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_state_probe/full_coord_boxend_shards_normed_lens/readout_shard_000 \
  --device cuda:0 \
  --hidden-layers=-1,-4 \
  --allow-model-load

# Same command shape was run for hidden-state shards 001/cuda:1, 002/cuda:2, and 003/cuda:3.

python scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py \
  --stage trajectory-hidden-delta-readout \
  --prefix-greedy-trajectory-readout-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/prefix_greedy_trajectory_readout_rows.jsonl \
  --pair-config configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml \
  --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_delta_probe/smoke_boxend4 \
  --device cuda:0 \
  --hidden-layers=-4,-1 \
  --target-next-kinds box_end \
  --max-rows 4 \
  --allow-model-load

python scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py \
  --stage trajectory-hidden-delta-readout \
  --prefix-greedy-trajectory-readout-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_delta_probe/full_boxend_shards/shard_000.jsonl \
  --pair-config configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml \
  --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_delta_probe/full_boxend_shards_normed_lens/readout_shard_000 \
  --device cuda:0 \
  --hidden-layers=-4,-1 \
  --target-next-kinds box_end \
  --allow-model-load

# Same command shape was run for hidden-delta shards 001 through 007 with
# CUDA_VISIBLE_DEVICES=1 through CUDA_VISIBLE_DEVICES=7 and --device cuda:0.

python scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py \
  --stage trajectory-hidden-boundary-patch \
  --prefix-greedy-trajectory-readout-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_delta_probe/full_boxend_shards/shard_000.jsonl \
  --pair-config configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml \
  --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_boundary_patch_probe/full_boxend_shards_normed_head/readout_shard_000 \
  --device cuda:0 \
  --hidden-layers=-4,-1 \
  --target-next-kinds box_end \
  --patch-alphas 0.25,0.5,0.75,1.0 \
  --patch-strengths 4,8,16,24,32,48,64,96,128 \
  --allow-model-load

# Same command shape was run for boundary-patch shards 001 through 007 with
# CUDA_VISIBLE_DEVICES=1 through CUDA_VISIBLE_DEVICES=7 and --device cuda:0.
```

Verification result: `138 passed`; prefix scoring row count 96; guidance
matrix case count 16; next-token readout row count 96; readout-analysis join
status `{"matched": 96}`; greedy trajectory seed row count 96; trajectory-step
row count 556; trajectory validity checks
`{"malformed_open_box_target_rows": 0, "malformed_open_box_prefix_rows": 0,
"schema_break_open_box_rows": 25, "post_schema_break_rows": 0,
"open_box_invalid_next_rows": 25,
"open_box_invalid_next_not_schema_break_rows": 0}`; hidden-state smoke row
count 16; broad normed-lens hidden-state readout row count 872 over 436
selected coord/box states with projection-space count
`{"lm_head_input_after_final_norm": 872}`; hidden-delta smoke row count 4;
broad normed-lens hidden-delta readout
row count 86 over all selected `box_end` trajectory states with readout status
`{"ok": 86}`; boundary-patch readout row count 1204 over the same 86
`box_end` states with readout status `{"ok": 1204}` and patch-space count
`{"lm_head_input_after_final_norm": 1204}`.

## 2026-06-19 raw decoder-layer causal activation patch

Implemented and ran a true model-forward activation patch stage,
`trajectory-hidden-causal-activation-patch`, to separate the earlier
LM-head-input readout result from a causal intervention. The stage patches the
raw last-token output of the decoder layer corresponding to hidden state `-4`,
runs the remaining model blocks, and scores the resulting next-token logits.
It records row-level `activation_patch_ran`, `model_perturbation_ran`,
`model_weight_update_ran`, `training_ran`, and `readout_only` flags, plus
direction-patch realization counters so skipped direction patches cannot look
like successful interventions.

Smoke artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/smoke_shard000
```

Full artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/full_boxend_shards_raw_activation
```

Full combined files:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/full_boxend_shards_raw_activation/trajectory_hidden_causal_activation_patch_rows.jsonl
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/full_boxend_shards_raw_activation/trajectory_hidden_causal_activation_patch_summary.json
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/full_boxend_shards_raw_activation/trajectory_hidden_causal_activation_patch_analysis.md
```

Run command shape:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py \
  --stage trajectory-hidden-causal-activation-patch \
  --prefix-greedy-trajectory-readout-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_delta_probe/full_boxend_shards/shard_000.jsonl \
  --pair-config configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml \
  --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/full_boxend_shards_raw_activation/readout_shard_000 \
  --device cuda:0 \
  --hidden-layers=-4,-1 \
  --target-next-kinds box_end \
  --patch-alphas 0.25,0.5,0.75,1.0 \
  --patch-strengths 4,8,16,24,32,48,64,96,128 \
  --allow-model-load

# Same command shape was run for shards 001 through 007 with
# CUDA_VISIBLE_DEVICES=1 through CUDA_VISIBLE_DEVICES=7 and --device cuda:0.
```

Full result:

```text
row_count: 1290
state_row_count: 86
case_count: 16
patch_space: raw_decoder_layer_output
readout_status_counts: {'ok': 1290}
activation_patch_ran: true
model_perturbation_ran: true
model_weight_update_ran: false
training_ran: false
readout_only: false
requested_direction_strength_count: 9
realized_direction_patch_count: 774
direction_patch_status_counts: {'realized': 1290}
```

The hook sanity check is clean: `self_source_noop` matches `baseline_no_patch`
exactly on the aggregate box-end margin and top-token rates. This supports the
decoder-layer index mapping and confirms that replacing the layer `-4` output
with the already-observed source state is effectively a no-op.

Main causal result:

```text
baseline_no_patch:
  schema_break rescue: 0/25
  clean preservation: 61/61
  mean box_end - object_ref_start margin: 5.0073 overall, -2.185 on failures

add_box_end_direction_64:
  schema_break rescue: 4/25
  clean preservation: 61/61

add_box_end_direction_96:
  schema_break rescue: 15/25
  clean preservation: 61/61

add_box_end_direction_128:
  schema_break rescue: 19/25
  clean preservation: 61/61
  mean box_end - object_ref_start margin: 10.6163 overall, 3.255 on failures

interpolate_source_to_target_alpha_0p25:
  schema_break rescue: 6/25
  clean preservation: 49/61

interpolate_source_to_target_alpha_0p5:
  schema_break rescue: 6/25
  clean preservation: 49/61
```

Interpretation:

- The readout-only boundary direction is not merely decorative: adding a raw
  `box_end - object_ref_start` output-embedding direction at the layer `-4`
  decoder output causally flips many hidden schema-break states after running
  through later blocks.
- The causal effect is dose-like and gated. Small strengths (`4` through `48`)
  mostly raise the box-end margin without flipping failures; `64`, `96`, and
  `128` cross visible decision thresholds. Mean margin shift versus
  `self_source_noop` is monotonic from `+0.1628` at strength `4` to `+5.6090`
  at strength `128`.
- Clean states are robust under direction patches: the high-strength direction
  patches preserve all `continue` and `max_steps` clean box-end decisions.
- Midpoint interpolation between final target hidden and source hidden is less
  clean: it can rescue `6/25` failures, but damages `12/61` clean states. This
  suggests the source-to-final hidden difference carries mixed circuitry, while
  the output-embedding boundary direction is a more targeted intervention.
- The remaining `6/25` failures under strength `128` are not simple
  `box_end`-versus-`object_ref_start` margin failures anymore. Their
  `box_end - object_ref_start` margin becomes positive, but top-1 can remain
  `<|object_ref_end|>` or `<|im_end|>`. The next mechanism branch should
  therefore include competing boundary/termination tokens, not only
  object-start opposition.

Comparison to the readout-only boundary patch:

```text
readout-only LM-head-input patch:
  best failure rescue: add_box_end_direction_16 -> 25/25
  clean preservation: 61/61

raw decoder-layer causal patch:
  best failure rescue: add_box_end_direction_128 -> 19/25
  clean preservation: 61/61
```

This gap is itself informative: the boundary direction is legible at the final
LM-head input, and a related raw layer-`-4` intervention survives through later
blocks, but late-layer processing attenuates/gates the signal and allows
`object_ref_end` / `im_end` competitors to remain decisive in the resistant
subset.

## 2026-06-19 competing boundary direction causal patch

Extended `trajectory-hidden-causal-activation-patch` so raw decoder-layer
direction patches can test multiple output-embedding directions in one pass:

```text
box_end_minus_object_ref_start
box_end_minus_object_ref_end
box_end_minus_im_end
box_end_minus_end_of_text
```

The default basis preserves the prior row label
`box_end_minus_object_ref_start_output_embedding` for artifact compatibility,
and rows now also include `direction_patch_basis_key` for the normalized basis
name. Basis summaries are computed only over realized direction-patch rows, so
baseline/noop/interpolation rows do not inflate basis counts.

Full artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/full_boxend_shards_competing_direction_bases
```

Full combined files:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/full_boxend_shards_competing_direction_bases/trajectory_hidden_causal_activation_patch_rows.jsonl
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/full_boxend_shards_competing_direction_bases/trajectory_hidden_causal_activation_patch_summary.json
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/full_boxend_shards_competing_direction_bases/trajectory_hidden_causal_activation_patch_analysis.md
```

Run scope:

```text
source trajectory shards: trajectory_readout_shard_000 through 003
selected_state_row_count: 86
schema-break failure states: 25
clean continue/max-step states: 61
source_hidden_layer_index: -4
target_hidden_layer_index: -1
patch_space: raw_decoder_layer_output
patch_alphas: [0.25, 0.5, 0.75, 1.0]
patch_strengths: [64, 96, 128]
direction_patch_status_counts: {'realized': 1548}
direction_patch_basis_key_counts:
  box_end_minus_object_ref_start: 258
  box_end_minus_object_ref_end: 258
  box_end_minus_im_end: 258
  box_end_minus_end_of_text: 258
```

Failure-rescue / clean-preservation result:

```text
baseline_no_patch:
  schema_break rescue: 0/25
  clean preservation: 61/61

add_box_end_direction_64:
  schema_break rescue: 4/25
  clean preservation: 61/61

add_box_end_direction_96:
  schema_break rescue: 15/25
  clean preservation: 61/61

add_box_end_direction_128:
  schema_break rescue: 19/25
  clean preservation: 61/61

add_box_end_minus_object_ref_end_direction_64:
  schema_break rescue: 6/25
  clean preservation: 61/61

add_box_end_minus_object_ref_end_direction_96:
  schema_break rescue: 12/25
  clean preservation: 61/61

add_box_end_minus_object_ref_end_direction_128:
  schema_break rescue: 19/25
  clean preservation: 61/61

add_box_end_minus_im_end_direction_128:
  schema_break rescue: 2/25
  clean preservation: 61/61

add_box_end_minus_end_of_text_direction_128:
  schema_break rescue: 7/25
  clean preservation: 61/61
```

Key mechanistic update:

- `box_end - object_ref_end` is a real boundary direction, but it does not
  expand the rescued failure set beyond the original `box_end -
  object_ref_start` direction at strength `128`: both rescue the exact same
  `19/25` schema-break states and fail on the exact same `6/25`.
- `box_end - im_end` is mostly inert for this failure family (`2/25` at
  strength `128`), and `box_end - end_of_text` is weak (`7/25` at strength
  `128`). Termination-token opposition is therefore not the primary missing
  mechanism for the current resistant set.
- The `6/25` resistant states are all the same `desc_first-1268-10-0`
  `person` object at `desc_end`, repeated across intervention arms
  (`post_commit_too_late`, `repetition_penalty_off`, `repetition_penalty_on`,
  `self_noop`, `target_coord_seed`, and `target_desc_plus_coord_seed`).
- In those resistant states, adding `box_end - object_ref_start` at strength
  `128` makes `box_end - object_ref_start` positive, but top-1 remains
  `<|object_ref_end|>` with `box_end` at rank 2. Adding `box_end -
  object_ref_end` symmetrically makes `box_end - object_ref_end` positive, but
  top-1 remains `<|object_ref_start|>` with `box_end` at rank 2.

Interpretation:

The resistant failures are no longer best described as a one-dimensional
`box_end` under-activation or as simple competition with a single wrong
boundary token. They look like a local object-boundary basin in which
`object_ref_start` and `object_ref_end` remain mutually substitutable attractors
around the same missing close-box event. The next high-value branch should
therefore test joint/subspace interventions or late-block attention/readout
mechanisms that separate "close current box" from "enter/exit object-ref span",
rather than only adding more one-vs-one output-embedding directions.

## 2026-06-19 joint object-ref-boundary causal patch

Added and tested a joint direction basis:

```text
box_end_minus_object_ref_boundaries
```

Semantics:

```text
embedding(box_end) - mean(embedding(object_ref_start), embedding(object_ref_end))
```

This basis directly tests the two-boundary attractor interpretation from the
competing-basis run: if the remaining six failures are caused by a local
object-ref-boundary basin, then suppressing both object-ref boundary directions
together should rescue them, while the old one-vs-one directions should keep
bouncing between `<|object_ref_start|>` and `<|object_ref_end|>`.

Full artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/full_boxend_shards_joint_boundary_basis
```

Full combined files:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/full_boxend_shards_joint_boundary_basis/trajectory_hidden_causal_activation_patch_rows.jsonl
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/full_boxend_shards_joint_boundary_basis/trajectory_hidden_causal_activation_patch_summary.json
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/full_boxend_shards_joint_boundary_basis/trajectory_hidden_causal_activation_patch_analysis.md
```

Run scope:

```text
source trajectory shards: trajectory_readout_shard_000 through 003
selected_state_row_count: 86
schema-break failure states: 25
clean continue/max-step states: 61
source_hidden_layer_index: -4
target_hidden_layer_index: -1
patch_space: raw_decoder_layer_output
patch_alphas: [0.25, 0.5, 0.75, 1.0]
patch_strengths: [64, 96, 128, 160, 192]
direction_bases:
  box_end_minus_object_ref_start
  box_end_minus_object_ref_boundaries
direction_patch_status_counts: {'realized': 1376}
direction_patch_basis_key_counts:
  box_end_minus_object_ref_start: 430
  box_end_minus_object_ref_boundaries: 430
```

Failure-rescue / clean-preservation result:

```text
baseline_no_patch:
  schema_break rescue: 0/25
  clean preservation: 61/61

add_box_end_direction_64:
  schema_break rescue: 6/25
  clean preservation: 61/61

add_box_end_direction_96:
  schema_break rescue: 15/25
  clean preservation: 61/61

add_box_end_direction_128:
  schema_break rescue: 19/25
  clean preservation: 61/61

add_box_end_direction_160:
  schema_break rescue: 19/25
  clean preservation: 61/61

add_box_end_direction_192:
  schema_break rescue: 19/25
  clean preservation: 61/61

add_box_end_minus_object_ref_boundaries_direction_64:
  schema_break rescue: 12/25
  clean preservation: 61/61

add_box_end_minus_object_ref_boundaries_direction_96:
  schema_break rescue: 19/25
  clean preservation: 61/61

add_box_end_minus_object_ref_boundaries_direction_128:
  schema_break rescue: 25/25
  clean preservation: 61/61

add_box_end_minus_object_ref_boundaries_direction_160:
  schema_break rescue: 25/25
  clean preservation: 61/61

add_box_end_minus_object_ref_boundaries_direction_192:
  schema_break rescue: 25/25
  clean preservation: 61/61
```

Key resistant-state result:

```text
previously resistant states under add_box_end_direction_128: 6/25
previously resistant states rescued by joint boundary direction at 128: 6/6
```

For the six `desc_first-1268-10-0` `person` states:

```text
baseline_no_patch:
  top-1: <|object_ref_start|>
  box_end rank: 3
  box_end - object_ref_start: about -3.75 to -3.875
  box_end - object_ref_end: about -3.625 to -3.75

add_box_end_direction_128:
  top-1: <|object_ref_end|>
  box_end rank: 2
  box_end - object_ref_start: about +1.125 to +1.25
  box_end - object_ref_end: about -1.5 to -1.75

add_box_end_minus_object_ref_boundaries_direction_96:
  top-1: <|object_ref_end|>
  box_end rank: 3
  box_end - object_ref_start: about -0.75 to -0.875
  box_end - object_ref_end: about -0.875 to -1.0

add_box_end_minus_object_ref_boundaries_direction_128:
  top-1: <|box_end|>
  box_end rank: 1
  box_end - object_ref_start: about +0.375
  box_end - object_ref_end: about +0.125 to +0.25
```

Interpretation:

This is the strongest causal result in the bridge sequence so far. The failure
is not simply missing positive evidence for `<|box_end|>`, because increasing
the one-vs-one `box_end - object_ref_start` direction from strength `128` to
`192` still leaves the same `6/25` failures stuck at `<|object_ref_end|>`. It
is also not a pure termination-token basin. Instead, the model appears to have
a local object-ref-boundary subspace that can absorb the missing close-box event:
when only one boundary is suppressed, the other boundary becomes the attractor;
when both object-ref boundaries are suppressed together, all schema-break
failures flip to `<|box_end|>` with no damage to clean box-end states.

Working mechanism statement:

```text
At desc-first box-closing failures, the late decoder state contains enough
recoverable "close a structural span now" information, but the structural span
identity is mis-bound: the state falls into an object-reference boundary basin
instead of the box boundary basin. A joint object-ref-boundary suppression
direction at raw layer -4 restores the intended box-close transition.
```

Next high-value branches:

- Test whether this joint object-ref-boundary subspace is specific to
  schema-break close-box failures or also predicts duplication onset before the
  visible duplicate burst.
- Move from output-embedding directions to learned/observed low-dimensional
  subspaces: e.g. boundary-pair plane, box-boundary plane, and residual
  source-target deltas for resistant versus easily rescued states.
- Inspect late-block attention/logit mediation for the `desc_first-1268-10-0`
  cluster: the key question is which context tokens route the state toward
  object-ref span closure rather than box span closure.

## 2026-06-19 hidden-delta joint-boundary projection bridge

Extended `trajectory-hidden-delta-readout` with a paired object-ref-boundary
projection that mirrors the causal patch basis above:

```text
delta_object_ref_boundaries_minus_box_end =
  hidden_delta dot unit(mean(E(object_ref_start), E(object_ref_end)) - E(box_end))

delta_box_end_minus_object_ref_boundaries =
  hidden_delta dot unit(E(box_end) - mean(E(object_ref_start), E(object_ref_end)))
```

This is an observational late-block hidden-delta readout over the same 86
box-end trajectory states, not a model perturbation. The value is that it tests
whether the natural transition from layer `-4` to layer `-1` already moves
states along the same object-ref-boundary basin direction that the causal patch
later suppresses.

Artifact root:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_delta_probe/full_boxend_shards_joint_boundary_projection
```

Combined files:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_delta_probe/full_boxend_shards_joint_boundary_projection/trajectory_hidden_delta_readout_rows.jsonl
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_delta_probe/full_boxend_shards_joint_boundary_projection/trajectory_hidden_delta_readout_summary.json
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_delta_probe/full_boxend_shards_joint_boundary_projection/trajectory_hidden_delta_analysis.md
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_delta_probe/full_boxend_shards_joint_boundary_projection/trajectory_hidden_delta_causal_join_rows.jsonl
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_delta_probe/full_boxend_shards_joint_boundary_projection/trajectory_hidden_delta_causal_join_summary.json
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_delta_probe/full_boxend_shards_joint_boundary_projection/trajectory_hidden_delta_causal_join_analysis.md
```

The causal join is now regenerated through the first-class read-only CLI stage:

```bash
PYTHONPATH=/data/CoordExp/.worktrees/autoregressive-binding-template-study \
python scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py \
  --stage analyze-trajectory-hidden-delta-causal-join \
  --trajectory-hidden-delta-readout-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_delta_probe/full_boxend_shards_joint_boundary_projection/trajectory_hidden_delta_readout_rows.jsonl \
  --trajectory-hidden-causal-patch-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/full_boxend_shards_joint_boundary_basis/trajectory_hidden_causal_activation_patch_rows.jsonl \
  --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_delta_probe/full_boxend_shards_joint_boundary_projection
```

Run scope:

```text
source trajectory shards: trajectory_readout_shard_000 through 003
devices: GPUs 0,1,2,3 via CUDA_VISIBLE_DEVICES
selected_state_row_count: 86
readout_status_counts: {'ok': 86}
source_hidden_layer_index: -4
target_hidden_layer_index: -1
target_next_kinds: ['box_end']
model_perturbation_ran: false
training_ran: false
```

Failure/clean projection split:

```text
schema-break failure states: 25
clean continue/max-step box-end states: 61

failure mean delta_object_ref_start_minus_box_end_unit_projection:
  106.898
clean mean delta_object_ref_start_minus_box_end_unit_projection:
  -72.599

failure mean delta_object_ref_boundaries_minus_box_end_unit_projection:
  115.297
clean mean delta_object_ref_boundaries_minus_box_end_unit_projection:
  -97.080

failure-clean gap for joint object-ref-boundary projection:
  212.377
```

Stop-reason split:

```text
continue:
  rows: 29
  mean joint object-ref-boundary projection: -86.305
  mean reverse box-end projection: +86.305

max_steps:
  rows: 32
  mean joint object-ref-boundary projection: -106.845
  mean reverse box-end projection: +106.845

schema_break_open_box:
  rows: 25
  mean joint object-ref-boundary projection: +115.297
  mean reverse box-end projection: -115.297
```

Joined against the joint-boundary causal patch artifact:

```text
bucket_counts:
  clean: 61
  failure_default128_rescued: 19
  failure_joint128_only: 6

causal_join_status_counts:
  matched: 86

failure rescue rates:
  add_box_end_direction_128: 19/25 = 0.76
  add_box_end_direction_192: 19/25 = 0.76
  add_box_end_minus_object_ref_boundaries_direction_64: 12/25 = 0.48
  add_box_end_minus_object_ref_boundaries_direction_96: 19/25 = 0.76
  add_box_end_minus_object_ref_boundaries_direction_128: 25/25 = 1.00

clean preservation rates:
  baseline_no_patch: 61/61 = 1.00
  add_box_end_direction_128: 61/61 = 1.00
  add_box_end_direction_192: 61/61 = 1.00
  add_box_end_minus_object_ref_boundaries_direction_64: 61/61 = 1.00
  add_box_end_minus_object_ref_boundaries_direction_96: 61/61 = 1.00
  add_box_end_minus_object_ref_boundaries_direction_128: 61/61 = 1.00

joint_only_default128_object_ref_end_handoff_count:
  6/6
```

Mean hidden-delta projections by causal bucket:

```text
clean:
  joint object-ref-boundary projection: -97.080
  object_ref_start-box_end projection: -72.599
  object_ref_end-box_end projection: -95.568

failure_default128_rescued:
  joint object-ref-boundary projection: +104.396
  object_ref_start-box_end projection: +101.733
  object_ref_end-box_end projection: +81.019

failure_joint128_only:
  joint object-ref-boundary projection: +149.817
  object_ref_start-box_end projection: +123.256
  object_ref_end-box_end projection: +137.171
```

The six joint-only failures are all the same `desc_first-1268-10-0-desc_end`
`person` state across intervention arms. They are not merely low in the
one-vs-one `object_ref_start` direction; they have the strongest joint
object-ref-boundary projection and a high `object_ref_end-box_end` component.
For a representative joint-only state:

```text
baseline_no_patch:
  top-1: <|object_ref_start|>
  box_end rank: 3
  box_end - object_ref_start: -3.75
  box_end - object_ref_end: -3.625

add_box_end_direction_128:
  top-1: <|object_ref_end|>
  box_end rank: 2
  box_end - object_ref_start: +1.25
  box_end - object_ref_end: -1.5

add_box_end_minus_object_ref_boundaries_direction_128:
  top-1: <|box_end|>
  box_end rank: 1
  box_end - object_ref_start: +0.375
  box_end - object_ref_end: +0.25
```

Interpretation update:

The hidden-delta readout makes the causal patch result more mechanistically
credible. The natural late-block update points schema-break states toward the
object-ref-boundary basin and clean states toward the box-end side of the same
axis. The previously resistant six failures are exactly the strongest
joint-boundary cases, and the one-vs-one patch fails because it transfers mass
from `<|object_ref_start|>` to `<|object_ref_end|>` rather than clearing the
whole object-ref-boundary pair. The core object-span failure is therefore best
described as a structural span-type binding error: the model has a generic
"close/open a structural boundary now" event, but in these states the boundary
identity is bound to object-ref span tokens instead of the current box span.

## 2026-06-19 boundary-routing selector for attention follow-up

Added a read-only selector stage to choose deterministic rows for the next
attention/routing probe:

```bash
PYTHONPATH=/data/CoordExp/.worktrees/autoregressive-binding-template-study \
python scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py \
  --stage analyze-trajectory-boundary-routing-selection \
  --trajectory-hidden-delta-causal-join-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_delta_probe/full_boxend_shards_joint_boundary_projection/trajectory_hidden_delta_causal_join_rows.jsonl \
  --selector-max-rows-per-bucket 4 \
  --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_routing_selection/joint_boundary_probe_seed_v1
```

Artifact root:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_routing_selection/joint_boundary_probe_seed_v1
```

Files:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_routing_selection/joint_boundary_probe_seed_v1/trajectory_boundary_routing_selected_rows.jsonl
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_routing_selection/joint_boundary_probe_seed_v1/trajectory_boundary_routing_selection_summary.json
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_routing_selection/joint_boundary_probe_seed_v1/trajectory_boundary_routing_selection.md
```

Selection scope:

```text
source_row_count: 86
selected_row_count: 12
selected_case_count: 9
selected_bucket_counts:
  failure_joint128_only: 4
  failure_default128_rescued: 4
  clean: 4
required probe metadata missing counts:
  source_line_idx: 0
  image_id: 0
  image_path: 0
  object_idx: 0
  target_gt_idx: 0
  target_desc: 0
  target_bbox: 0
  decode_repetition_penalty: 0
  trajectory_state_prefix_text: 0
```

The selector ranks rows by the joint object-ref-boundary projection, but prefers
distinct `case_id`s before filling duplicate arms. This keeps the hard bucket
faithful to the data, where all `failure_joint128_only` rows are the same
`desc_first-1268-10-0-desc_end` `person` state, while making the easy-failure
and clean controls more diverse.

Post-review contract adjustment: the selected JSONL is now directly consumable
as a GPU attention/routing seed artifact. The causal join and selected rows
preserve source/image/object target metadata plus decode and prefix-routing
fields when present, so the next runner should not need to reconstruct these
handles from `case_id`.

Selected attention/routing seed rows:

```text
failure_joint128_only:
  desc_first-1268-10-0-desc_end target_coord_seed person
  desc_first-1268-10-0-desc_end target_desc_plus_coord_seed person
  desc_first-1268-10-0-desc_end post_commit_too_late person
  desc_first-1268-10-0-desc_end repetition_penalty_off person

failure_default128_rescued:
  desc_first-285-1-0-desc_end post_commit_too_late bear
  desc_first-885-8-0-desc_end post_commit_too_late person
  desc_first-139-0-0-desc_end target_coord_seed clock
  desc_first-139-0-0-pre_desc target_desc_plus_coord_seed clock

clean:
  desc_first-2685-33-11-desc_end post_commit_too_late wine glass
  desc_first-139-0-11-desc_end target_coord_seed chair
  desc_first-139-0-9-desc_end target_coord_seed chair
  geometry_first-139-0-1-pre_box_start post_commit_too_late clock
```

Prefix-anchor observation:

All selected states are at a box-closing decision surface with an open box and
four coordinates after the latest `<|box_start|>`. The discriminating factor is
therefore not gross schema position. The next GPU probe should compare how the
last-token state attends to, or is patched from, the current descriptor and
structural anchors:

```text
current object-ref descriptor
current <|object_ref_start|> / <|object_ref_end|>
current <|box_start|>
the four current coord tokens
previous completed <|box_end|> when present
```

## 2026-06-19 boundary-anchor attention probe

Added a read-only model-backed attention stage over the selected routing seed
rows. The stage forces eager attention and aggregates the last prefix-token
query over structural text anchors: current object-ref boundaries, current
descriptor tokens, current box-start, the four current coord tokens, current
box span, and previous completed box-end when present.

Smoke command:

```bash
PYTHONPATH=/data/CoordExp/.worktrees/autoregressive-binding-template-study \
python scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py \
  --stage trajectory-boundary-anchor-attention \
  --trajectory-boundary-routing-selected-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_routing_selection/joint_boundary_probe_seed_v1/trajectory_boundary_routing_selected_rows.jsonl \
  --pair-config configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml \
  --allow-model-load \
  --max-rows 1 \
  --hidden-layers=-1 \
  --device cuda:0 \
  --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_anchor_attention/smoke_1row_last_layer
```

Full selected all-layer command:

```bash
LAYERS=$(python - <<'PY'
print(','.join(str(i) for i in range(28)))
PY
)
PYTHONPATH=/data/CoordExp/.worktrees/autoregressive-binding-template-study \
python scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py \
  --stage trajectory-boundary-anchor-attention \
  --trajectory-boundary-routing-selected-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_routing_selection/joint_boundary_probe_seed_v1/trajectory_boundary_routing_selected_rows.jsonl \
  --pair-config configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml \
  --allow-model-load \
  --hidden-layers="$LAYERS" \
  --device cuda:0 \
  --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_anchor_attention/full_selected_all_layers_v1
```

Artifact roots:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_anchor_attention/smoke_1row_last_layer
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_anchor_attention/full_selected_late4_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_anchor_attention/full_selected_all_layers_v1
```

Scope:

```text
smoke_1row_last_layer:
  selected_state_row_count: 1
  row_count: 176

full_selected_late4_v1:
  selected_state_row_count: 12
  row_count: 8960
  layers: -1,-2,-4,-8

full_selected_all_layers_v1:
  selected_state_row_count: 12
  row_count: 62720
  layers: 0..27
  case_count: 9
```

All-layer bucket-level contrast:

```text
mean normalized attention mass over state/layer/head:

clean:
  object_ref_boundaries: 0.027017
  current_box_span: 0.219989
  current_coords: 0.199800
  object_ref_minus_box_span: -0.192972

failure_default128_rescued:
  object_ref_boundaries: 0.057653
  current_box_span: 0.258497
  current_coords: 0.231250
  object_ref_minus_box_span: -0.200844

failure_joint128_only:
  object_ref_boundaries: 0.060850
  current_box_span: 0.231371
  current_coords: 0.203788
  object_ref_minus_box_span: -0.170522
```

Late-layer bucket-level contrast (`-1,-2,-4,-8`):

```text
clean:
  object_ref_boundaries: 0.024357
  current_box_span: 0.076311
  current_coords: 0.066024
  object_ref_minus_box_span: -0.051954

failure_default128_rescued:
  object_ref_boundaries: 0.066980
  current_box_span: 0.082523
  current_coords: 0.070338
  object_ref_minus_box_span: -0.015543

failure_joint128_only:
  object_ref_boundaries: 0.077047
  current_box_span: 0.085058
  current_coords: 0.072276
  object_ref_minus_box_span: -0.008011
```

Layer localization:

```text
mean object_ref_boundaries - current_box_span by bucket:

clean best layers:
  layer 24: -0.002534
  layer 21: -0.004336
  layer 25: -0.014393

failure_default128_rescued best layers:
  layer 21:  0.064212
  layer 24:  0.019278
  layer 22:  0.017437

failure_joint128_only best layers:
  layer 21:  0.089496
  layer 19:  0.039376
  layer 20:  0.037987
```

Strongest object-ref-minus-box-span heads:

```text
failure_joint128_only:
  layer 21 head 2:  mean +0.559718, min +0.559468, max +0.559967, n=4
  layer 21 head 3:  mean +0.410417, min +0.409536, max +0.411299, n=4
  layer 19 head 9:  mean +0.395498, min +0.387822, max +0.403173, n=4
  layer 19 head 8:  mean +0.394208, min +0.384211, max +0.404205, n=4
  layer 26 head 10: mean +0.364299, min +0.361010, max +0.367588, n=4

failure_default128_rescued:
  layer 18 head 0:  mean +0.435514, min +0.156542, max +0.573929, n=4
  layer 26 head 10: mean +0.345480, min +0.252338, max +0.389202, n=4
  layer 19 head 8:  mean +0.333735, min +0.264659, max +0.402556, n=4
  layer 21 head 2:  mean +0.319249, min +0.205475, max +0.563870, n=4

clean:
  layer 24 head 1:  mean +0.227453, min +0.000736, max +0.362416, n=4
  layer 19 head 3:  mean +0.166785, min +0.030034, max +0.322868, n=4
  layer 24 head 0:  mean +0.129728, min -0.069379, max +0.345476, n=4
```

Interpretation:

The failure states do not simply ignore the current box. Across all layers the
current box/coord span still receives more attention mass than object-ref
boundaries. The sharper signal is a mid/late-layer routing mixture: failures,
especially the joint-only hard failure, carry about 2.2x higher all-layer
object-ref-boundary mass than clean controls, and by layers 19-21 the
object-ref-boundary pair overtakes the current box span in the failure buckets
but not in clean controls. This is a better mechanistic bridge than "attention
went to the wrong place" globally: the model still attends to the current box,
yet a small set of heads appears to route the final box-closing decision through
the object-ref-boundary basin.

Next high-value probe:

Use layers 19-21, especially layer 21 heads 2/3 and layer 19 heads 8/9, as
candidate attention/value-stream intervention sites. The causal question should
not be "remove all object-ref attention"; it should be whether suppressing or
value-patching the object-ref-boundary contribution in these heads increases the
box-end margin without damaging clean controls.

## 2026-06-19 boundary-head causal intervention probe

Added a model-backed causal readout stage:

```text
trajectory-boundary-head-intervention
```

The stage reuses the selected boundary-routing seed rows and applies a
last-token attention-head intervention at `self_attn.o_proj` input: for a
candidate decoder layer/head, it scales that head's concatenated per-head
attention output slice at the recovered final prefix token before projection.
The default intervention is head suppression (`head_scale=0.0`). This is not a
training run and not a generation rollout, but it is a genuine activation
perturbation; realized rows therefore record `model_perturbation_ran=true` and
`readout_only=false`.

Smoke artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_intervention/smoke_1row_head21_2
```

Full selected 12-state, 6-head sharded artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_intervention/full_selected_all_candidate_heads_sharded_v1
```

Source shards:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_intervention/full_selected_head21_2_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_intervention/full_selected_head21_3_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_intervention/full_selected_heads19_8_9_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_intervention/full_selected_heads26_10_18_0_v1
```

Combined artifact scope:

```text
selected_state_count: 12
candidate_heads: 18:0, 19:8, 19:9, 21:2, 21:3, 26:10
row_count: 72
status_counts: {'realized': 72}
attention_implementation: eager
families: desc_first=66 rows, geometry_first=6 rows
trajectory_stop_reason: schema_break_open_box=48, continue=12, max_steps=12
```

Mean suppression effect by head, measured as intervention minus baseline on
`box_end - mean(object_ref_start, object_ref_end)` logit margin:

```text
18:0   +0.713542
19:8   -0.005208
19:9   -0.596354
21:2   -0.776042
21:3   +0.182292
26:10  +3.640625
```

For `schema_break_open_box` rows only:

```text
18:0   +0.937500
19:8    0.000000
19:9   -0.757812
21:2   -1.054688
21:3   +0.093750
26:10  +4.773438
```

Logit-component decomposition by head:

```text
head    box_end_delta  object_ref_start_delta  object_ref_end_delta  object_ref_boundary_mean_delta
18:0       +0.447917              -0.046875            -0.484375                    -0.265625
19:8       -0.083333              -0.140625            -0.015625                    -0.078125
19:9       -0.447917              +0.088542            +0.208333                    +0.148438
21:2       -0.677083              +0.067708            +0.130208                    +0.098958
21:3       -0.114583              +0.036458            -0.630208                    -0.296875
26:10      +0.322917              -2.713542            -3.921875                    -3.317708
```

Interpretation:

The causal panel breaks the tempting "object-ref-attending head is bad" story.
Layer 21 head 2 and layer 19 head 9 were strong object-ref-minus-box-span
attention heads, but suppressing them makes the box-end decision worse. They
appear to carry box-closure support despite attending through object-ref
anchors. In contrast, layer 26 head 10 is a late high-impact carrier of the
next-object/continue-enumeration basin: suppressing it strongly lowers
object-ref boundary logits and improves the `box_end` margin, especially in
open-box failure states. However, this is not a pure rescue head either; several
large positive 26:10 effects route the top token to `<|im_end|>` rather than
`<|box_end|>`. That suggests a late arbitration mechanism between three basins:
close-current-box, begin-next-object, and terminate.

Next high-value probe:

Separate "box closure" from "object-boundary suppression" and "termination
escape" for layer 26 head 10. The promising follow-up is not just more
suppression; it is value patching or directional interpolation that preserves
box-end support while selectively removing the next-object boundary component,
with clean-control damage tracked explicitly.

## 2026-06-19 boundary-head scale sweep

Extended `trajectory-boundary-head-intervention` with `--head-scales` so the
same selected states can be probed under partial head-output scaling, not only
binary suppression. The intervention remains the same hook site:
`self_attn.o_proj` input at the recovered final prefix token. The sweep used
scales `[0.0, 0.25, 0.5, 0.75, 1.0]`, where `1.0` is the no-op control.

Combined artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_intervention/scale_sweep_selected_heads_v1
```

Source shards:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_intervention/scale_sweep_head26_10_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_intervention/scale_sweep_head18_0_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_intervention/scale_sweep_head21_2_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_intervention/scale_sweep_head19_9_v1
```

Combined scope:

```text
selected_state_count: 12
heads: 18:0, 19:9, 21:2, 26:10
head_scales: 0.0, 0.25, 0.5, 0.75, 1.0
row_count: 240
status_counts: {'realized': 240}
attention_implementation: eager
```

Mean intervention-minus-baseline effect on
`box_end - mean(object_ref_start, object_ref_end)`:

```text
all selected states:
head   scale0    scale0.25  scale0.50  scale0.75  scale1.00
18:0   +0.7135    +0.6406    +0.4896    +0.2135    +0.0000
19:9   -0.5964    -0.4661    -0.2578    -0.1380    +0.0000
21:2   -0.7760    -0.5651    -0.3568    -0.1562    +0.0000
26:10  +3.6406    +2.3776    +1.2604    +0.5000    +0.0000

schema_break_open_box rows only:
18:0   +0.9375    +0.8125    +0.6328    +0.2656    +0.0000
19:9   -0.7578    -0.5781    -0.3281    -0.2109    +0.0000
21:2   -1.0547    -0.7578    -0.5078    -0.2500    +0.0000
26:10  +4.7734    +3.0742    +1.5781    +0.5859    +0.0000
```

Top-token behavior for `26:10` across all selected states:

```text
scale0.00: box_end=7, im_end=5, object_ref_boundary=0
scale0.25: box_end=5, im_end=1, object_ref_boundary=6
scale0.50: box_end=4, im_end=1, object_ref_boundary=7
scale0.75: box_end=4, im_end=1, object_ref_boundary=7
scale1.00: box_end=4, im_end=1, object_ref_boundary=7
```

For the eight open-box failure rows specifically:

```text
scale0.00: box_end=3, im_end=5, object_ref_boundary=0
scale0.25: box_end=1, im_end=1, object_ref_boundary=6
scale0.50: box_end=0, im_end=1, object_ref_boundary=7
scale0.75: box_end=0, im_end=1, object_ref_boundary=7
scale1.00: box_end=0, im_end=1, object_ref_boundary=7
```

Interpretation:

The 26:10 effect is strongly graded in logit margin but thresholded in
top-token identity. Partial suppression weakens the object-ref boundary basin
without usually crossing the decision boundary. Full suppression collapses the
object-ref boundary basin enough to flip some rows to `<|box_end|>`, but it also
lets `<|im_end|>` win in five of eight open-box failure rows. That makes 26:10
look less like a box-closure head and more like a late continuation-boundary
controller whose removal exposes an already-present termination basin. In
contrast, 21:2 and 19:9 remain monotonically box-end-supportive under partial
suppression, which reinforces the earlier conclusion that object-ref attention
can carry closure support rather than duplicate pressure.

Next high-value probe:

For 26:10, compare pure suppression with a directional or value-patched
intervention that subtracts only the object-ref-boundary component while adding
or preserving a box-end component. The measurable target should be "increase
box_end over object_ref boundaries without increasing `<|im_end|>` top-token
rate." This is now sharper than asking whether the head is good or bad.

## 2026-06-19 boundary-head empirical direction patch

Added an additive head-slice direction patch to
`trajectory-boundary-head-intervention`. The hook site is still
`self_attn.o_proj` input at the recovered final prefix token, but instead of
only scaling one concatenated head slice, the probe can now build a per-family,
per-head empirical unit direction:

```text
box_end_success_minus_object_ref_failure
  = mean(head_slice | baseline top token is <|box_end|>)
  - mean(head_slice | baseline top token is <|object_ref_start|> or <|object_ref_end|>)
```

The production path is intentionally two-pass and streaming: direction capture
stores only CPU head slices and baseline-top labels, then rows are re-prepared
one at a time for scale and patch emission. This avoids retaining image/model
inputs for a whole family on GPU.

Implementation verification:

```text
python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -q -p no:cacheprovider
192 passed

python -m py_compile src/analysis/autoregressive_binding_template_ablation/realized_behavior_bridge.py scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py
passed

git diff --check
passed
```

Primary `26:10` strength sweep:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_intervention/direction_patch_head26_10_strength_sweep_v1
```

Scope:

```text
family: desc_first
selected_state_count: 11
candidate_head: 26:10
head_scales: 0.0, 1.0
head_patch_strengths: -64, -32, -16, -8, 0, 8, 16, 32, 64, 128
row_count: 132
status_counts: {'realized': 132}
direction_norm: 73.871376
direction_basis_counts: box_end_success=3, object_ref_failure=7
```

Top-token behavior for additive `26:10` direction patch across all 11
`desc_first` states:

```text
strength -64: object_ref_end=6, object_ref_start=2, box_end=3
strength -32: object_ref_end=5, object_ref_start=3, box_end=3
strength -16: object_ref_start=7, object_ref_end=1, box_end=3
strength  -8: object_ref_start=7, object_ref_end=1, box_end=3
strength   0: object_ref_start=6, im_end=1, object_ref_end=1, box_end=3
strength   8: object_ref_start=6, im_end=1, object_ref_end=1, box_end=3
strength  16: object_ref_start=6, im_end=1, object_ref_end=1, box_end=3
strength  32: object_ref_start=6, im_end=1, object_ref_end=1, box_end=3
strength  64: object_ref_start=6, im_end=1, box_end=4
strength 128: im_end=5, box_end=6
```

Threshold sweep for `26:10`:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_intervention/direction_patch_head26_10_threshold_sweep_v1
```

For the eight open-box failure rows:

```text
strength  64: object_ref_start=6, im_end=1, box_end=1, mean_margin=-0.9062
strength  72: object_ref_start=6, im_end=1, box_end=1, mean_margin=-0.4766
strength  80: object_ref_start=6, im_end=1, box_end=1, mean_margin=-0.1172
strength  88: object_ref_start=6, im_end=1, box_end=1, mean_margin=+0.2422
strength  96: object_ref_start=4, im_end=1, box_end=3, mean_margin=+0.7773
strength 104: object_ref_start=4, im_end=1, box_end=3, mean_margin=+1.1914
strength 112: object_ref_start=4, im_end=1, box_end=3, mean_margin=+1.6836
strength 120: im_end=5, box_end=3, mean_margin=+2.1484
strength 128: im_end=5, box_end=3, mean_margin=+2.6289
```

The hard repeated `desc_first-1268-10-0-desc_end` states are especially
diagnostic. Their box-end-minus-object-ref-boundary margin rises monotonically
under the positive 26:10 direction, but top token stays `<|object_ref_start|>`
through strength 112 and then flips to `<|im_end|>` at 120/128. In other words,
even when the box-end margin reaches zero or positive, the top-token winner is
not box closure; termination becomes the exposed basin.

Coarse same-basis comparison across other candidate heads:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_intervention/direction_patch_head18_0_coarse_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_intervention/direction_patch_head19_9_coarse_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_intervention/direction_patch_head21_2_coarse_v1
```

Summary:

```text
18:0  positive direction pushes object_ref_start -> object_ref_end and worsens box_end margin.
19:9  negative direction weakly improves margin but does not change the object-ref loop; positive direction worsens.
21:2  negative direction weakly improves margin but mostly routes object_ref_start -> object_ref_end; positive direction worsens.
26:10 positive direction strongly improves box_end margin and rescues some non-hard failures, but hard duplicate states hit a termination cliff before clean box closure.
```

Interpretation:

The direction patch supports a more specific mechanism than "head 26:10 is bad"
or "object-ref attention causes duplication." Among the tested heads, only
26:10 has a strong positive direction toward the box-end margin. But in hard
duplicate-onset states, changing that direction does not directly convert the
loop to `<|box_end|>`. Instead the state appears to sit in a three-way late
arbitration basin: object-ref continuation, current-box closure, and sequence
termination. The empirical 26:10 direction can reduce object-ref pressure and
raise the box-end margin, but once the duplicate basin is sufficiently weakened,
the termination basin often wins before closure does. This points to a missing
"close current object and continue" commitment signal, not merely excess
object-ref-boundary evidence.

Next high-value probe:

Split the 26:10 direction into components that separately affect
`object_ref_start`, `object_ref_end`, `<|box_end|>`, and `<|im_end|>`. The next
intervention should try a readout-gradient or local logit-linearized direction
that increases `<|box_end|>` over object-ref boundaries while explicitly
penalizing `<|im_end|>`. The concrete question is whether there exists a local
patch direction that closes the hard open boxes without crossing the termination
cliff. If not, the core mechanism is likely not just a single-head value vector
but a later residual-state basin involving stop/continue arbitration.

## 2026-06-19 boundary-head no-terminate logit-gradient patch

Added a second additive direction basis for `trajectory-boundary-head-intervention`:

```text
logit_box_end_minus_object_ref_boundaries_minus_im_end
```

This basis is computed per row, per candidate head, by differentiating the local
scalar objective below with respect to the selected head slice at
`self_attn.o_proj` input:

```text
logit(<|box_end|>)
  - mean(logit(<|object_ref_start|>), logit(<|object_ref_end|>))
  - logit(<|im_end|>)
```

The gradient hook detaches the `o_proj` input before enabling gradients, so the
direction is local to the intervention site and does not retain the upstream
vision/language graph. This directly tests whether a nearby head-slice direction
can close the current object while suppressing both duplicate object-ref
continuation and premature termination.

Implementation verification:

```text
python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -q -p no:cacheprovider
194 passed

python -m py_compile src/analysis/autoregressive_binding_template_ablation/realized_behavior_bridge.py scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py
passed

git diff --check
passed
```

Primary artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_intervention/logit_no_terminate_head26_10_strength_sweep_v1
```

Scope:

```text
family: desc_first
selected_state_count: 11
candidate_head: 26:10
head_scales: 1.0
head_patch_direction_bases: logit_box_end_minus_object_ref_boundaries_minus_im_end
head_patch_strengths: 8, 16, 32, 64, 96, 128, 160, 192, 256
row_count: 110
status_counts: {'realized': 110}
gradient_direction_norm: min=0.064687, max=0.084021, mean=0.074180
gradient_objective_value: min=-24.4375, max=7.40625, mean=-15.238636
```

Top-token behavior:

```text
strength   8: all object_ref_start=6, im_end=1, object_ref_end=1, box_end=3
strength  16: all object_ref_start=6, im_end=1, object_ref_end=1, box_end=3
strength  32: all object_ref_start=6, im_end=1, object_ref_end=1, box_end=3
strength  64: all object_ref_start=6, box_end=5
strength  96: all object_ref_start=4, box_end=7
strength 128: all object_ref_start=2, box_end=9
strength 160: all box_end=11
strength 192: all box_end=11
strength 256: all box_end=11
```

For the eight open-box failure rows:

```text
strength   8: object_ref_start=6, im_end=1, object_ref_end=1, mean_margin=-2.5078
strength  16: object_ref_start=6, im_end=1, object_ref_end=1, mean_margin=-2.1953
strength  32: object_ref_start=6, im_end=1, object_ref_end=1, mean_margin=-1.5859
strength  64: object_ref_start=6, box_end=2, mean_margin=-0.4609
strength  96: object_ref_start=4, box_end=4, mean_margin=+0.7266
strength 128: object_ref_start=2, box_end=6, mean_margin=+1.8359
strength 160: box_end=8, mean_margin=+2.9180
strength 192: box_end=8, mean_margin=+3.9531
strength 256: box_end=8, mean_margin=+5.9883
```

Narrow threshold artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_intervention/logit_no_terminate_head26_10_threshold_sweep_v1
```

The hard repeated `desc_first-1268-10-0-desc_end` states transition cleanly:

```text
strength 128: hard1268 object_ref_start=2, box_end=2, mean_im_end_prob=0.006556
strength 136: hard1268 box_end=4, mean_im_end_prob=0.005632
strength 144: hard1268 box_end=4, mean_im_end_prob=0.005127
strength 152: hard1268 box_end=4, mean_im_end_prob=0.004139
strength 160: hard1268 box_end=4, mean_im_end_prob=0.003666
```

For the two hardest target-guided `1268` variants, the transition happens
between 128 and 136:

```text
128: target_coord_seed / target_desc_plus_coord_seed stay object_ref_start with margin +0.500
136: target_coord_seed / target_desc_plus_coord_seed flip to box_end with margin +0.812
```

Interpretation:

This is the first probe that cleanly separates current-box closure from the
termination cliff. The empirical clean-minus-failure head-slice direction raised
the box-end margin but often exposed `<|im_end|>` before hard duplicate states
closed. The local logit-gradient direction explicitly subtracts `<|im_end|>`;
under that basis, the same hard states flip to `<|box_end|>` while their
`<|im_end|>` probability decreases. Therefore the termination cliff is not an
unavoidable downstream consequence of perturbing 26:10. The head-slice space at
26:10 contains a local direction that represents "close current box, not start a
new object, not stop." The failure mode is more likely a missing or underused
directional component in the natural autoregressive state than an absence of
visual grounding or an inability to score `<|box_end|>`.

Mechanistic update:

Layer 26 head 10 should now be treated as a late boundary arbitration interface,
not a monolithic duplicate/termination head. The same head slice can support at
least three separable movements:

```text
empirical clean-minus-failure direction:
  weakens object-ref continuation but can expose termination.

pure suppression:
  collapses object-ref boundaries and often exposes termination.

local no-terminate logit-gradient direction:
  closes hard open boxes and suppresses termination.
```

This makes the next research target sharper: identify why the natural residual
state does not activate the no-terminate closure component at duplicate onset.
Promising follow-ups are (1) compare the no-terminate direction projection
between clean, rescued, and hard failure states before intervention; (2) trace
which upstream layers/heads write or fail to write that component; and (3) test
whether small prefix/context guidance increases the natural projection onto this
direction before the boundary token is emitted.

## 2026-06-19 boundary-head natural projection readout

Extended additive boundary-head patch rows with natural source-slice projection
metrics:

```text
head_patch_source_slice_norm
head_patch_source_projection
head_patch_source_projection_cosine
```

These are computed from the unpatched selected head slice captured at
`self_attn.o_proj` input during a realized additive patch row. The projection is
the dot product between the natural head slice and the requested unit patch
direction.

Implementation verification:

```text
python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -q -p no:cacheprovider
194 passed

python -m py_compile src/analysis/autoregressive_binding_template_ablation/realized_behavior_bridge.py scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py
passed

git diff --check
passed
```

Projection artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_intervention/logit_no_terminate_head26_10_projection_readout_v1
```

Scope:

```text
family: desc_first
selected_state_count: 11
candidate_head: 26:10
head_scales: 1.0
head_patch_direction_bases: logit_box_end_minus_object_ref_boundaries_minus_im_end
head_patch_strengths: 136
row_count: 22
patch_row_count: 11
```

Projection summary by causal bucket:

```text
all selected:
  projection=-22.919288, cosine=-0.251740, slice_norm=101.345068,
  gradient_norm=0.074180, objective=-15.238636

clean:
  projection=-20.025918, cosine=-0.399338, slice_norm=49.776285,
  gradient_norm=0.076579, objective=+3.895833

failure_default128_rescued:
  projection=-25.212905, cosine=-0.211556, slice_norm=115.574553,
  gradient_norm=0.077654, objective=-20.828125

failure_joint128_only:
  projection=-22.795699, cosine=-0.181227, slice_norm=125.792171,
  gradient_norm=0.068905, objective=-24.000000
```

Hard repeated `desc_first-1268-10-0-desc_end` rows:

```text
post_commit_too_late:
  projection=-22.958822, cosine=-0.183330, slice_norm=125.231895,
  objective=-23.5625, baseline_margin=-3.6875

repetition_penalty_off:
  projection=-22.958822, cosine=-0.183330, slice_norm=125.231895,
  objective=-23.5625, baseline_margin=-3.6875

target_coord_seed:
  projection=-22.632576, cosine=-0.179123, slice_norm=126.352,
  objective=-24.4375, baseline_margin=-4.1875

target_desc_plus_coord_seed:
  projection=-22.632576, cosine=-0.179123, slice_norm=126.352,
  objective=-24.4375, baseline_margin=-4.1875
```

Interpretation:

The projection readout corrects the first intuitive story. Hard duplicate states
do not simply have less natural projection onto the no-terminate closure
direction. All selected states have negative source projection along their local
no-terminate direction, including clean rows. Clean rows nevertheless emit
`<|box_end|>` because the rest of the state makes the no-terminate objective
positive. The clearer failure signature is:

```text
failure rows have much larger 26:10 source-slice norm
and a strongly negative no-terminate objective.
```

For `failure_joint128_only`, source-slice norm is roughly 2.5x clean
(`125.8` vs `49.8`), while the no-terminate objective is about `-24` versus
`+3.9` for clean. The natural 26:10 slice is therefore not merely missing a
small positive closure component; it is participating in a high-magnitude
boundary state whose downstream readout objective is deeply anti-closure. The
successful strength-136 no-terminate patch works by injecting a corrective
local direction into that high-magnitude state.

Mechanistic update:

The next tracing target should shift from "which upstream module writes the
positive no-terminate projection?" to "which upstream module causes the large
26:10 boundary slice norm and negative no-terminate objective at duplicate
onset?" Natural projection alone is not the separating variable. The likely
separating variables are source-slice magnitude, no-terminate objective value,
and the downstream mapping from the 26:10 slice into object-ref/box/termination
logits.

Next high-value probe:

Trace the no-terminate objective and 26:10 source-slice norm backward across
layers and attention heads. A practical next slice is to add a readout-only
`trajectory-boundary-head-direction-readout` that captures, for selected heads
and layers, source-slice norm, no-terminate gradient norm, objective value, and
optional attention anchor mass without applying a patch. That would let us rank
which heads show the same high-magnitude anti-closure signature before doing
more interventions.

## 2026-06-19 boundary-head direction readout implementation and six-head ranking

Implemented the readout-only stage proposed above:

```text
stage: trajectory-boundary-head-direction-readout
default direction basis: logit_box_end_minus_object_ref_boundaries_minus_im_end
row artifact: trajectory_boundary_head_direction_readout_rows.jsonl
summary artifact: trajectory_boundary_head_direction_readout_summary.json
report artifact: trajectory_boundary_head_direction_readout.md
```

The stage captures candidate attention-head slices at the recovered final prefix
token before `self_attn.o_proj`, builds the local no-terminate logit-gradient
direction, and records source norm/projection/cosine without patching, scaling,
decoding, or training.

Implementation checks:

```text
python -m py_compile src/analysis/autoregressive_binding_template_ablation/realized_behavior_bridge.py scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py
passed

python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -q -p no:cacheprovider -k "trajectory_boundary_head_direction_readout or trajectory_boundary_head_intervention or logit_gradient"
12 passed

python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -q -p no:cacheprovider
199 passed

git diff --check
passed
```

Review gates:

```text
subagent spec compliance review: passed
subagent code-quality review: passed after tightening selected-source summary
semantics, readout_only guard semantics, skipped-direction coverage, and CLI
direction-basis clarity.
```

Readout scope:

```text
family: desc_first
selected states: 11
source selected bucket counts:
  clean: 3
  failure_default128_rescued: 4
  failure_joint128_only: 4
candidate heads:
  21:2
  21:3
  19:8
  19:9
  26:10
  18:0
```

Per-head artifacts:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_direction_readout/desc_first_head21_2_logit_no_terminate_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_direction_readout/desc_first_head21_3_logit_no_terminate_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_direction_readout/desc_first_head19_8_logit_no_terminate_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_direction_readout/desc_first_head19_9_logit_no_terminate_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_direction_readout/desc_first_head26_10_logit_no_terminate_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_direction_readout/desc_first_head18_0_logit_no_terminate_v1
```

Combined ranking artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_direction_readout/desc_first_selected_heads_logit_no_terminate_combined_summary_v1.json
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_direction_readout/desc_first_selected_heads_logit_no_terminate_combined_summary_v1.md
```

Ranking by mean source-slice norm:

```text
head   mean grad norm   mean source norm   failure/clean norm   mean projection   mean abs projection   mean cosine
26:10  0.074180         101.345068         2.424515             -22.919288       22.919288             -0.251740
21:2   0.144335         61.515401          1.435566              8.155921        8.155921              0.122795
21:3   0.222969         47.414298          0.920079              2.434237        2.434237              0.051188
19:9   0.195940         34.373767          1.168612              1.859922        3.232135              0.048111
18:0   0.366606         24.602835          0.841777             -2.604273        2.604273             -0.109435
19:8   0.209837         22.875200          1.493555             -1.129735        1.657584             -0.047309
```

Important interpretation detail:

`head_direction_objective_value` is the downstream no-terminate objective at
the same prompt state, so it is identical across candidate heads for the same
state. It measures state severity, not head ranking. Head ranking comes from
gradient norm, source-slice norm, projection, and cosine.

Updated mechanism hypothesis:

Head `26:10` remains the dominant boundary-state site. It has the largest
source-slice norm, largest failure/clean norm ratio, and largest absolute
projection onto the local no-terminate direction. The earlier result that a
large positive no-terminate patch rescues hard duplicate states now looks less
like "26:10 is missing a small closure vector" and more like "duplicate onset
contains an unusually large 26:10 boundary-state slice whose downstream local
readout is in a strongly anti-closure basin." Head `21:2` is a secondary
candidate because its source norm and positive projection are nontrivial, but
it is much weaker than `26:10`.

Next high-value probe:

Trace upstream writers into the large `26:10` source slice. The immediate next
slice should locate which previous layers/heads/features predict the jump from
clean-like `26:10` norms near `50` to duplicate-onset norms around `120-126`,
then test whether suppressing or rotating those upstream contributors reduces
the anti-closure basin without requiring the direct no-terminate patch.

## 2026-06-19 value-contribution decomposition of 26:10 source slice

Implemented a second readout-only stage:

```text
stage: trajectory-boundary-head-value-contribution
default direction basis: logit_box_end_minus_object_ref_boundaries_minus_im_end
row artifact: trajectory_boundary_head_value_contribution_rows.jsonl
summary artifact: trajectory_boundary_head_value_contribution_summary.json
report artifact: trajectory_boundary_head_value_contribution.md
```

This stage captures the target layer's `v_proj` output, the natural
`self_attn.o_proj` input slice, and eager attention weights for a selected
head/query token. It decomposes the head output into attention-weighted value
contributions by source-token region, then projects each regional contribution
onto the local no-terminate direction. Reconstruction error is included as a
sanity check; for the real model runs below it stayed around `0.0015-0.0019`
relative L2.

Implementation checks:

```text
python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -q -p no:cacheprovider -k "trajectory_boundary_head_value_contribution or trajectory_boundary_head_direction_readout or trajectory_boundary_head_intervention or logit_gradient or attention_value_key"
14 passed
```

GPU artifacts:

```text
smoke:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_value_contribution/desc_first_head26_10_smoke1_v1

selected desc_first, image/non-image split:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_value_contribution/desc_first_head26_10_selected_v2_image_split
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_value_contribution/desc_first_head21_2_selected_v2_image_split
```

Combined comparison artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_value_contribution/desc_first_heads26_10_vs21_2_value_contribution_image_split_v1.json
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_value_contribution/desc_first_heads26_10_vs21_2_value_contribution_image_split_v1.md
```

Key 26:10 regional decomposition:

```text
region                         mass      clean mass  fail mass  norm       clean norm  fail norm  projection  clean proj  fail proj
pre_prefix_image_tokens        0.006916  0.013443    0.004468   0.836476   1.637759    0.535994  -0.297683   -0.636747  -0.170534
pre_prefix_non_image_context   0.561965  0.718186    0.503382   36.691394  15.683402   44.569391 -9.845437   -4.842405  -11.721574
current_object_ref_boundaries  0.315446  0.030660    0.422241   57.462297  4.906819    77.170601 -8.755085   -1.810736  -11.359216
current_object_ref_end         0.136527  0.015574    0.181885   29.851407  3.202706    39.844670 -6.767786   -1.253986  -8.835461
current_prefix_all             0.430788  0.268682    0.491578   69.699154  37.238406   81.871934 -12.774646  -14.555042 -12.106998
all_context                    0.999669  1.000312    0.999428   101.353260 49.777692   120.694098 -22.917767 -20.034194 -23.999107
```

Secondary comparison, 21:2:

```text
region                         mass      fail mass  norm       fail norm  projection  fail projection
pre_prefix_image_tokens        0.005625  0.005657   0.319086   0.323390  -0.036209   -0.042206
pre_prefix_non_image_context   0.510774  0.493084   25.186777  28.212338  3.184859    4.634230
current_object_ref_boundaries  0.375377  0.470581   32.506586  40.940298  4.616717    6.224473
current_prefix_all             0.484176  0.502025   39.926645  42.632357  4.997373    6.378909
all_context                    1.000574  1.000767   61.518488  67.068020  8.146024    10.970933
```

Top negative-projection tokens for 26:10 failure rows:

```text
current_object_ref_boundaries:
  <|object_ref_end|>: count=8, summed_projection=-70.683687
  <|object_ref_start|>: count=8, summed_projection=-20.190047

pre_prefix_non_image_context:
  <|im_start|>: count=8, summed_projection=-29.490395
  newline token: count=8, summed_projection=-17.805649
  <|object_ref_start|>: count=14, summed_projection=-10.054285
  <|im_end|>: count=8, summed_projection=-7.975892
```

Mechanistic update:

The large `26:10` anti-closure boundary state is not primarily a visual-token
value contribution in this selected cohort. The `pre_prefix_image_tokens`
region is tiny by attention mass, norm, and projection. The dominant negative
contributors are:

1. Current object-ref boundary tokens, especially `<|object_ref_end|>`.
2. Non-image pre-prefix context/history/template tokens, including chat wrapper
   tokens and repeated object-ref markers.

This shifts the next mechanism question again. The failure does not look like
"the model cannot visually perceive the object" at the 26:10 boundary site.
It looks more like an autoregressive binding/template-history circuit in which
object-ref boundary tokens and pre-prefix non-image context pull the boundary
head into an anti-closure basin. The image stream may still matter upstream for
which object should be emitted, but the immediate duplicate-onset closure
failure at 26:10 is dominated by language-side/boundary-token value
contributions.

Next high-value probe:

Do a causal value-region patch or ablation for 26:10: selectively suppress or
rotate the contribution from `current_object_ref_boundaries` and
`pre_prefix_non_image_context` while leaving image-token values untouched. If
object-ref-boundary suppression rescues `<|box_end|>` without destroying clean
rows, it would strongly support a boundary-token-driven duplicate basin. If
pre-prefix non-image context is the effective lever, the origin is further
upstream in history/template binding rather than the current object span alone.

## 2026-06-19 causal value-region suppression at 26:10

Implemented the causal next-token stage proposed above:

```text
stage: trajectory-boundary-head-value-region-intervention
default source regions:
  current_object_ref_boundaries
  pre_prefix_non_image_context
  pre_prefix_image_tokens
default scale: 1.0
hook site: self_attn.o_proj.forward_pre_hook
patch: -scale * attention-weighted value-region contribution at the final prefix token
```

The stage reuses the value-contribution capture, sums the selected source
region's attention-weighted value contribution for a chosen head, and subtracts
that vector at the same head/query slice before `o_proj`. It records
baseline/intervention next-token scores and logit deltas. It is a causal
next-token intervention, not a full continuation or detector-eval result.

Implementation checks:

```text
python -m py_compile src/analysis/autoregressive_binding_template_ablation/realized_behavior_bridge.py scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py
passed

python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -q -k "value_region_intervention"
8 passed

python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -q -k "trajectory_boundary_head_value_region_intervention or trajectory_boundary_head_value_contribution or trajectory_boundary_head_direction_readout or trajectory_boundary_head_intervention or trajectory_boundary_head_value_region_metrics or attention_value_key"
22 passed

python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -q -p no:cacheprovider
211 passed

GPU CLI validation smoke after validation fixes:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_value_region_intervention/desc_first_head26_10_smoke1_validation_v2
row_count: 3
model_perturbation_ran: true
readout_only: false
```

The focused tests now include:

- exact CPU-only assertion that the patch runner receives
  `-scale * region_contribution`;
- fail-fast validation for misspelled source regions;
- fail-fast validation that this causal stage accepts exactly one projection
  basis, because multiple bases would duplicate identical perturbations;
- mocked writer coverage for rows, summary, report, provenance flags, and
  region/scale grouping.

GPU artifacts:

```text
smoke:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_value_region_intervention/desc_first_head26_10_smoke1_regions_v1

selected desc_first, split regions and scales:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_value_region_intervention/desc_first_head26_10_selected_regions_split_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_value_region_intervention/desc_first_head21_2_selected_regions_split_v1
```

Run scope for the selected artifacts:

```text
family: desc_first
selected states: 11
source bucket counts:
  clean: 3
  failure_default128_rescued: 4
  failure_joint128_only: 4
regions:
  current_object_ref_start
  current_object_ref_end
  current_object_ref_boundaries
  current_descriptor
  current_box_span
  pre_prefix_non_image_context
  pre_prefix_image_tokens
scales: 0.5, 1.0, 2.0
rows per head: 231
```

Core result, 26:10:

```text
overall mean target prob delta: 0.037780872707
overall mean box_end_minus_object_ref_boundaries_logit_delta: 0.774891774892

region / scale                         source projection   prob delta    box-boundary delta
current_object_ref_boundaries / 1.0     -8.755085           0.116953      2.000000
current_object_ref_boundaries / 2.0     -8.755085           0.228222      5.170455
current_object_ref_end / 2.0            -6.767786           0.135837      2.068182
pre_prefix_non_image_context / 2.0      -9.845437           0.088126      1.715909
pre_prefix_image_tokens / 2.0           -0.297683           0.000040     -0.011364
```

Comparison, 21:2:

```text
overall mean target prob delta: -0.004954612729
overall mean box_end_minus_object_ref_boundaries_logit_delta: -0.173160173160

region / scale                         source projection   prob delta    box-boundary delta
current_object_ref_boundaries / 1.0      4.616717          -0.010001     -0.329545
current_object_ref_boundaries / 2.0      4.616717          -0.016137     -0.693182
current_object_ref_end / 2.0             2.260574          -0.010787     -0.377841
pre_prefix_non_image_context / 2.0       3.184859          -0.020708     -0.877841
pre_prefix_image_tokens / 2.0           -0.036209          -0.000505     -0.008523
```

Visible top-token changes were sparse but diagnostic. For 26:10, boundary
subtraction turned one `failure_default128_rescued` row from
`<|object_ref_end|>` to `<|box_end|>` at scale 1.0:

```text
case: desc_first-885-8-0-desc_end
arm: post_commit_too_late
region: current_object_ref_boundaries
scale: 1.0
top token: <|object_ref_end|> -> <|box_end|>
target prob delta: 0.561142027378
box-boundary delta: 2.875
```

At scale 2.0, 26:10 boundary suppression also produced several
`<|object_ref_start|> -> <|im_end|>` transitions in hard rows. This is an
important caveat: the boundary contribution is not a simple bad vector that can
be removed freely. It appears to sit between three local basins:

1. duplicate/object-ref continuation,
2. valid box closure,
3. premature termination.

Mechanistic update:

The causal sign matches the value-decomposition sign. In 26:10, current
object-ref boundary and pre-prefix non-image contributions project negatively
onto the no-terminate box-end direction; subtracting them moves the model
toward `<|box_end|>`. In 21:2, those same regions project positively; subtracting
them moves the model away from box closure. This makes 26:10 a much stronger
candidate for the immediate duplicate-boundary attraction mechanism than 21:2.

The image-token control remains near-null. Suppressing `pre_prefix_image_tokens`
has tiny norm/projection and near-zero causal effect, reinforcing the current
interpretation that this boundary failure is immediate language-side/template
history dynamics rather than visual non-perception at the final boundary head.

Next high-value probe:

Decode a short continuation after moderate 26:10 boundary suppression and after
matched no-image controls, not just next-token logits. The key question is
whether scale around 1.0 can repair local object-span closure without pushing
the sequence into premature `<|im_end|>` or damaging clean rows. If the repair
persists through 2-4 generated tokens, the next round should trace upstream
writers that make `<|object_ref_end|>` such a large negative contributor in
26:10 failure rows.

## 2026-06-19 value-region continuation after first-token patch

Implemented the continuation probe proposed above:

```text
stage: trajectory-boundary-head-value-region-continuation
first step: same causal value-region subtraction as trajectory-boundary-head-value-region-intervention
follow-up steps: natural greedy prefix-next-token readouts from the patched prefix
row flags: patched_first_step rows are perturbation rows; natural_followup rows are readout-only
seed fields: natural rows carry seed_* causal evidence but no unprefixed causal deltas
summary: aggregate first/final rates plus by_continuation_id first-vs-final endpoints
```

This stage is still a short deterministic probe, not a full rollout repair
claim and not official detection evaluation. Its purpose is to test whether a
local boundary repair survives the first emitted token.

Implementation checks:

```text
python -m py_compile src/analysis/autoregressive_binding_template_ablation/realized_behavior_bridge.py scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py
passed

python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -q -k "value_region_continuation or value_region_intervention"
12 passed

python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -q
215 passed
```

Review checks:

```text
subagent spec compliance review after fixes: passed
reviewer-rerun py_compile: passed
reviewer-rerun pytest -k value_region_continuation: 4 passed
```

GPU artifacts:

```text
single-row smoke, 26:10 boundary vs image controls:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_value_region_continuation/desc_first_head26_10_smoke1_boundary_vs_image_v1

selected post_commit_too_late, 3-step continuation, 26:10:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_value_region_continuation/desc_first_postcommit_head26_10_regions_scales012_v3

selected post_commit_too_late, 3-step continuation, 21:2 contrast:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_value_region_continuation/desc_first_postcommit_head21_2_regions_scales012_v3

selected post_commit_too_late, 8-step continuation, 26:10:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_value_region_continuation/desc_first_postcommit_head26_10_regions_scales012_steps8_v2
```

Selected-run scope:

```text
family: desc_first
intervention_arm filter: post_commit_too_late
selected states: 4
regions:
  current_object_ref_boundaries
  pre_prefix_non_image_context
  pre_prefix_image_tokens
scales: 0.0, 1.0, 2.0
heads: 26:10 and 21:2 contrast
```

Three-step result, 26:10:

```text
row_count: 60
continuation_count: 36
first_step_token_counts:
  <|box_end|>: 12
  <|im_end|>: 10
  <|object_ref_end|>: 6
  <|object_ref_start|>: 8
first_step_box_end_rate: 0.333333333333
first_step_im_end_rate: 0.277777777778
final_stop_reason_counts:
  max_steps: 12
  schema_break_open_box: 24
mean_first_step_target_next_token_prob_delta: 0.073374005034
mean_first_step_box_end_minus_object_ref_boundaries_logit_delta: 1.135416666667
```

Region/scale contrast, 26:10:

```text
current_object_ref_boundaries / 1.0:
  first_step_box_end_rate: 0.5
  first_step_im_end_rate: 0.25
  mean_first_step_target_next_token_prob_delta: 0.185138166649
  mean_first_step_box_end_minus_object_ref_boundaries_logit_delta: 2.109375

current_object_ref_boundaries / 2.0:
  first_step_box_end_rate: 0.5
  first_step_im_end_rate: 0.5
  mean_first_step_target_next_token_prob_delta: 0.278615093557
  mean_first_step_box_end_minus_object_ref_boundaries_logit_delta: 5.1953125

pre_prefix_non_image_context / 2.0:
  first_step_box_end_rate: 0.5
  first_step_im_end_rate: 0.25
  mean_first_step_target_next_token_prob_delta: 0.140725215664
  mean_first_step_box_end_minus_object_ref_boundaries_logit_delta: 2.046875

pre_prefix_image_tokens / 2.0:
  first_step_box_end_rate: 0.25
  first_step_im_end_rate: 0.25
  mean_first_step_target_next_token_prob_delta: 0.003186578164
  mean_first_step_box_end_minus_object_ref_boundaries_logit_delta: 0.03125
```

Three-step result, 21:2 contrast:

```text
row_count: 54
continuation_count: 36
first_step_box_end_rate: 0.25
first_step_im_end_rate: 0.222222222222
mean_first_step_target_next_token_prob_delta: -0.011410598533
mean_first_step_box_end_minus_object_ref_boundaries_logit_delta: -0.268229166667

current_object_ref_boundaries / 1.0:
  mean_first_step_target_next_token_prob_delta: -0.018134952174
  mean_first_step_box_end_minus_object_ref_boundaries_logit_delta: -0.3671875

current_object_ref_boundaries / 2.0:
  mean_first_step_target_next_token_prob_delta: -0.030514889222
  mean_first_step_box_end_minus_object_ref_boundaries_logit_delta: -0.765625
```

The strongest observed repair path is the same row identified by the
next-token intervention:

```text
case: desc_first-885-8-0-desc_end
arm: post_commit_too_late
head: 26:10
region: current_object_ref_boundaries
scale: 1.0
first token: <|object_ref_end|> -> <|box_end|>
target prob delta: 0.561142027378
box-boundary delta: 2.875
3-step path:
  <|box_end|> -> <|object_ref_start|> -> person
8-step path:
  <|box_end|> -> <|object_ref_start|> -> person -> <|object_ref_end|> -> <|box_start|> -> <|coord_0|> -> <|coord_0|> -> <|coord_47|>
```

The same case at scale 2.0 follows the same 8-step path with a larger first
step effect:

```text
target prob delta: 0.825568020344
box-boundary delta: 7.46875
8-step path:
  <|box_end|> -> <|object_ref_start|> -> person -> <|object_ref_end|> -> <|box_start|> -> <|coord_0|> -> <|coord_0|> -> <|coord_47|>
```

Mechanistic update:

Moderate suppression of the 26:10 current object-ref-boundary contribution can
do more than improve the next-token logit. In the repaired 885 row, it restores
a syntactically coherent next object start and descriptor. That supports a
local boundary-token basin explanation: the model had enough language-side
state to continue the object template once `<|box_end|>` was forced by a
targeted head/value intervention.

However, the 8-step probe exposes a second basin immediately downstream. The
new span's coordinates collapse toward low coordinate tokens
`<|coord_0|>,<|coord_0|>,<|coord_47|>`. Thus the patch repairs the object-span
boundary transition but does not recover grounded geometry. The failure is not
a single missing visual-perception bit; it separates into at least two coupled
mechanisms:

1. a boundary/history circuit at 26:10 that decides whether to close the prior
   box or loop through object-ref tokens;
2. a subsequent grounding/coordinate basin that can remain corrupted after the
   boundary transition is repaired.

Post-hoc coordinate-copy analyzer update:

```text
12-step focused artifact:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_value_region_continuation_coordinate_copy/desc_first_postcommit_head26_10_objref_scales12_steps12_v1

8-step broad artifact:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_value_region_continuation_coordinate_copy/desc_first_postcommit_head26_10_regions_scales012_steps8_v2
```

The analyzer groups continuation rows by `continuation_id`, extracts the active
seed/open bbox, emitted greedy coord tokens, the first emitted bbox when four
coord tokens are available, and target bbox. It then classifies whether the path
is `exact_seed_copy`, `seed_like`, `target_like`, `mixed`, `incomplete`, or
`no_coords`.

The 12-step focused run changes the interpretation of the strongest repaired
885 path. For both scale 1.0 and scale 2.0:

```text
case: desc_first-885-8-0-desc_end
region: current_object_ref_boundaries
seed_bbox: [0, 0, 47, 22]
target_bbox: [1, 1, 94, 22]
emitted_bbox: [0, 0, 47, 22]
coordinate_copy_class: exact_seed_copy
available_l1_to_seed: 0
available_l1_to_target: 49
```

Thus the boundary intervention repairs span syntax but does not re-ground to the
target object. It closes the old open box and then re-enters a local bbox replay
basin for the next `person` span. The earlier 8-step run was already pointing in
the same direction: the 885 current-object-ref-boundary scale 1.0 and 2.0 rows
emitted `[0, 0, 47]` before the probe stopped, with available-slot L1-to-seed
`0` and L1-to-target `49`. The apparent "coordinate collapse" is more specific
than low-coordinate attraction: in this case it is old-box replay.

The same 12-step focused artifact also marks
`desc_first-2685-33-11-desc_end` as `seed_like`:

```text
seed_bbox: [118, 349, 141, 415]
target_bbox: [123, 341, 170, 410]
emitted_bbox: [118, 351, 134, 415]
available_l1_to_seed: 9
available_l1_to_target: 56
```

This suggests the repaired continuation can preserve the local spatial anchor
even when it is not an exact copy. The current evidence is still tiny
(`row_count=8` in the focused 12-step post-hoc analyzer), but it sharpens the
next mechanism question: after a boundary repair, does the model re-query image
evidence for the next object, or does the active local object state keep the
coordinate decoder inside a spatial replay attractor?

Selected hidden/logit readout on copy-like coordinate states:

```text
state selection:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_value_region_coordinate_copy_state_selection/desc_first_postcommit_head26_10_objref_copylike_coord_states_v1

hidden readout:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_state_probe/coord_copy_selected_copylike_coord_states_v1
```

The selector keeps only coord-slot states from `exact_seed_copy` and `seed_like`
continuations. It produced 16 selected states across four continuations; the
hidden readout over layers `-1,-2,-4,-8` produced 64 readout rows with
`readout_status_counts: {"ok": 64}`.

Final-layer summary at scale 1.0:

| class | case | emitted bbox | target bbox | coord top1 distances | target ranks | mean target prob |
| --- | --- | --- | --- | --- | --- | ---: |
| `seed_like` | `desc_first-2685-33-11-desc_end` | `[118, 351, 134, 415]` | `[123, 341, 170, 410]` | `[5, 10, 38, 5]` | `[6, 42, 55, 11]` | `0.009208` |
| `exact_seed_copy` | `desc_first-885-8-0-desc_end` | `[0, 0, 47, 22]` | `[1, 1, 94, 22]` | `[1, 1, 61, 0]` | `[2, 2, 94, 1]` | `0.050407` |

Both paths allocate almost all final-layer mass to coordinate tokens at coord
slots. The replay failure is therefore not a grammar/type failure at these
states. In the `885` exact-copy path, the first two coordinates are near the
target only because target and seed both start near zero; the discriminating
slot is x2, where the target is 94 but the hidden readout's coord top1 is 33,
the greedy token is 47, and target rank is 94. The y2 slot is exact at 22. This
points to a spatial-anchor replay problem concentrated in specific coordinate
slots, not a uniform inability to represent or emit the target object box.

Counterfactual-prefix smoke:

```text
885 x2 previous-box rewrite rows:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_value_region_coordinate_copy_counterfactual_prefixes/desc_first_885_x2_prevbox_rewrite_v1

885 x2 readout:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_state_probe/coord_copy_885_x2_prevbox_counterfactual_v1

2685 x2 previous/current rewrite rows:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_value_region_coordinate_copy_counterfactual_prefixes/desc_first_2685_x2_prevbox_currentxy_rewrite_v1

2685 x2 readout:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_state_probe/coord_copy_2685_x2_prevbox_currentxy_counterfactual_v1
```

For `885`, rewriting the previous closed bbox before the repaired next object
does not make x2 jump to the target 94:

| variant | target coord | final coord top1 | distance | target rank |
| --- | ---: | ---: | ---: | ---: |
| natural | 94 | 33 | 61 | 94 |
| previous box -> target bbox | 94 | 33 | 61 | 90 |
| previous x2 -> target | 94 | 47 | 47 | 86 |
| previous x2 -> far 777 | 94 | 33 | 61 | 94 |

For `2685`, rewriting either the previous closed box or the current partial
`x1,y1` nudges x2 slightly but still leaves it far below target 170:

| variant | target coord | final coord top1 | distance | target rank |
| --- | ---: | ---: | ---: | ---: |
| natural | 170 | 132 | 38 | 55 |
| current xy -> target | 170 | 139 | 31 | 45 |
| previous box -> target bbox | 170 | 141 | 29 | 50 |
| previous box + current xy -> target | 170 | 141 | 29 | 44 |
| previous x2 -> far 777 | 170 | 141 | 29 | 51 |

This weakens the simplest "the model literally copies the previous coordinate
token from text" explanation. The replay basin is local-history-sensitive, but
not a direct editable text copy of the previous closed box. A more plausible
working hypothesis is that the repaired boundary leaves the decoder in a
spatial-anchor attractor shaped by the same-image/object-cluster trajectory,
where coordinate grammar is stable but grounding fails to refresh the object
pointer strongly enough to move x2 toward the target.

The image-token control remains near-null in both next-token and continuation
probes. The useful bridge lever is current object-ref boundaries, with
pre-prefix non-image context acting as a weaker but real contributor. The
contrast head 21:2 moves in the opposite aggregate direction for
current_object_ref_boundaries, reinforcing 26:10 as the cleaner causal target
for immediate duplicate-boundary attraction.

Next high-value probe:

Trace the repaired-but-ungrounded path. Starting from the 885 repaired
continuation, compare hidden/logit/attention states at the new object's
`<|object_ref_start|>`, descriptor end, `<|box_start|>`, and coordinate slots
against a naturally clean same-class or same-image path. The question is
whether the coordinate collapse is inherited from the patched boundary head
state, from missing visual re-binding after object-start emission, or from an
autoregressive low-coordinate prior triggered by the unnatural repair.

## Source-Region Attention Readout On Copy-Like Coord Slots

New readout stage:

```text
trajectory-source-region-attention
```

Implementation surface:

```text
scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py
src/analysis/autoregressive_binding_template_ablation/realized_behavior_bridge.py
tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py
```

This stage is read-only. It rebuilds the same image/chat prompt for selected
trajectory states, runs eager attention, and records last-token self-attention
mass from reusable source regions. It does not patch activations, decode new
tokens, train, or recompute target probabilities. The key addition over
boundary-anchor attention is reuse of the broader value/source-region inventory,
including `pre_prefix_image_tokens`, `pre_prefix_non_image_context`,
`current_prefix_all`, current object-ref/descriptor/box/coord spans, and
previous box-end regions.

The important implementation detail is `--trajectory-prefix-source trajectory`.
For these coordinate slots, `trajectory_state_prefix_text` can collapse all four
coord positions in a repaired object to the same object-state prefix. The
per-step `trajectory_prefix_text` is needed to inspect the actual coordinate
slot being emitted.

Representative selected-state input:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_source_region_attention/inputs/coord_copy_selected_copylike_scale1_slot_states_v1.jsonl
```

Input summary:

```text
row_count: 8
case_count: 2
trajectory_step_idx_counts: {"5": 1, "6": 2, "7": 2, "8": 2, "9": 1}
coordinate_copy_class_counts: {"exact_seed_copy": 4, "seed_like": 4}
```

Main readout artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_source_region_attention/coord_copy_scale1_slots_last4_selected_regions_trajectory_prefix_v1
```

Command:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py \
  --stage trajectory-source-region-attention \
  --trajectory-boundary-routing-selected-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_source_region_attention/inputs/coord_copy_selected_copylike_scale1_slot_states_v1.jsonl \
  --pair-config configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml \
  --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_source_region_attention/coord_copy_scale1_slots_last4_selected_regions_trajectory_prefix_v1 \
  --device cuda:0 \
  --allow-model-load \
  --hidden-layers=-1,-2,-4,-8 \
  --trajectory-prefix-source trajectory \
  --value-source-regions current_object_ref_boundaries,current_descriptor,current_box_start,current_coords,current_box_span,previous_box_end,all_previous_box_ends,current_prefix_other,current_prefix_all,pre_prefix_image_tokens,pre_prefix_non_image_context,pre_prefix_context \
  --families desc_first \
  --target-next-kinds coord
```

Summary:

```text
row_count: 6144
ok_row_count: 6016
skipped_row_count: 128
source_row_count: 8
case_count: 2
prefix_source: trajectory
status_counts: {"ok": 6016, "skipped": 128}
```

Mean normalized attention by source region. `mean` uses the requested-region
denominator, while `ok mean` excludes rows where the requested source region
was absent from the recovered prefix.

| region | rows | ok rows | skipped | mean | ok mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pre_prefix_context` | `512` | `512` | `0` | `0.826023` | `0.826023` |
| `pre_prefix_non_image_context` | `512` | `512` | `0` | `0.810110` | `0.810110` |
| `current_prefix_all` | `512` | `512` | `0` | `0.173977` | `0.173977` |
| `current_box_span` | `512` | `512` | `0` | `0.109619` | `0.109619` |
| `current_coords` | `512` | `384` | `128` | `0.072077` | `0.096103` |
| `current_box_start` | `512` | `512` | `0` | `0.037542` | `0.037542` |
| `pre_prefix_image_tokens` | `512` | `512` | `0` | `0.015913` | `0.015913` |
| `current_object_ref_boundaries` | `512` | `512` | `0` | `0.011102` | `0.011102` |
| `previous_box_end` | `512` | `512` | `0` | `0.001884` | `0.001884` |
| `current_descriptor` | `512` | `512` | `0` | `0.001519` | `0.001519` |

The mean image-token mass is low at the coordinate slots, but not exactly zero.
Some heads spike to image tokens, with max normalized image mass `0.595302` in
the representative selected-state run. The stronger and more consistent
coordinate-slot routing is the combination of dominant pre-prefix non-image
context plus a smaller set of sharply local current-box/current-coordinate
heads. Strong current-coordinate heads include:

```text
27 / 15 / current_coords: mean 0.865503, max 0.936002
20 / 14 / current_coords: mean 0.585120, max 0.975163
26 / 9  / current_coords: mean 0.582461, max 0.900023
```

Counterfactual-prefix source-region readout:

```text
combined input:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_source_region_attention/inputs/coord_copy_counterfactual_prefix_rows_v1.jsonl

readout:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_source_region_attention/coord_copy_counterfactual_prefix_rows_last4_regions_v1
```

Command:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py \
  --stage trajectory-source-region-attention \
  --trajectory-boundary-routing-selected-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_source_region_attention/inputs/coord_copy_counterfactual_prefix_rows_v1.jsonl \
  --pair-config configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml \
  --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_source_region_attention/coord_copy_counterfactual_prefix_rows_last4_regions_v1 \
  --device cuda:0 \
  --allow-model-load \
  --hidden-layers=-1,-2,-4,-8 \
  --trajectory-prefix-source trajectory \
  --value-source-regions current_object_ref_boundaries,current_descriptor,current_box_start,current_coords,current_box_span,previous_box_end,all_previous_box_ends,current_prefix_other,current_prefix_all,pre_prefix_image_tokens,pre_prefix_non_image_context,pre_prefix_context \
  --families desc_first \
  --target-next-kinds coord
```

Summary:

```text
row_count: 8448
ok_row_count: 8448
skipped_row_count: 0
source_row_count: 11
case_count: 2
prefix_source: trajectory
status_counts: {"ok": 8448}
counterfactual_variant_counts:
  current_xy_to_target: 768
  natural: 1536
  prev_box_and_current_xy_to_target: 768
  prev_box_to_target_bbox: 1536
  prev_box_x2_to_far777: 1536
  prev_box_x2_to_mid333: 768
  prev_box_x2_to_target: 1536
```

Mean normalized attention by source region:

| region | rows | ok rows | skipped | mean | ok mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pre_prefix_context` | `704` | `704` | `0` | `0.841578` | `0.841578` |
| `pre_prefix_non_image_context` | `704` | `704` | `0` | `0.822985` | `0.822985` |
| `current_prefix_all` | `704` | `704` | `0` | `0.158422` | `0.158422` |
| `current_box_span` | `704` | `704` | `0` | `0.125282` | `0.125282` |
| `current_coords` | `704` | `704` | `0` | `0.119679` | `0.119679` |
| `pre_prefix_image_tokens` | `704` | `704` | `0` | `0.018593` | `0.018593` |
| `current_object_ref_boundaries` | `704` | `704` | `0` | `0.005315` | `0.005315` |
| `current_box_start` | `704` | `704` | `0` | `0.005603` | `0.005603` |
| `current_descriptor` | `704` | `704` | `0` | `0.000842` | `0.000842` |

Counterfactual variant means:

| variant | pre-prefix non-image | current prefix | current box span | current coords | image tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `current_xy_to_target` | `0.839852` | `0.144855` | `0.110990` | `0.105890` | `0.015293` |
| `natural` | `0.822348` | `0.159217` | `0.126095` | `0.120641` | `0.018435` |
| `prev_box_and_current_xy_to_target` | `0.841605` | `0.143050` | `0.109372` | `0.104134` | `0.015344` |
| `prev_box_to_target_bbox` | `0.822666` | `0.157982` | `0.125321` | `0.119857` | `0.019353` |
| `prev_box_x2_to_far777` | `0.819440` | `0.161546` | `0.129251` | `0.123042` | `0.019014` |
| `prev_box_x2_to_mid333` | `0.795758` | `0.182019` | `0.147695` | `0.141473` | `0.022223` |
| `prev_box_x2_to_target` | `0.823357` | `0.157615` | `0.124357` | `0.118946` | `0.019028` |

The counterfactual readout does not support a simple literal text-copy story.
Rewriting previous coordinate text does change local routing mildly, especially
`prev_box_x2_to_mid333`, but it does not produce a wholesale shift away from
the dominant non-image context basin or into image tokens. The readout instead
points to a mixed autoregressive state: strong global text/context attention,
plus specialized local coordinate heads that can lock onto the active current
box span.

Mechanistic update:

The coordinate-copy/replay failure is increasingly unlikely to be a pure visual
perception absence at the coordinate emission instant. The model is in a stable
coordinate-token type basin, and selected heads can strongly attend to current
coordinate slots. The failure looks more like stale or mis-bound object state:
after the boundary repair, the decoder has a coherent object-span template, but
the grounding pointer is not refreshed strongly enough to pull the local
coordinate attractor toward the target bbox. Image evidence may still have been
used upstream, but the immediate slot-level readout shows little average
image-token routing compared with non-image context and current local box
history.

Scope boundary:

The counterfactual-prefix rows used here preserve inherited
`target_next_token_*` score fields from the selected state. This source-region
attention stage measures routing over rewritten prefixes; it should not be
interpreted as per-variant target-probability or rank movement unless a
separate score-recompute stage is run on each rewritten prefix.

Row schema note:

Per-head rows keep exact `attention_source_prompt_token_indices` and
`attention_source_token_count`, but only store a bounded
`attention_source_token_texts_preview` by default. This prevents composite
regions such as `pre_prefix_context` and `pre_prefix_image_tokens` from
repeating full token-text arrays once per layer/head/region. Full token text can
be reconstructed from the prompt ids and stored indices when needed.

## Counterfactual Prefix Fresh Score Recompute

New readout stage:

```text
trajectory-prefix-next-token-readout
```

Implementation surface:

```text
scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py
src/analysis/autoregressive_binding_template_ablation/realized_behavior_bridge.py
tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py
```

This stage adapts trajectory-state-shaped rows into the existing
`prefix-next-token-readout` executor. It scores either
`trajectory_state_prefix_text` or per-step `trajectory_prefix_text`, preserving
inherited selected-state score fields under `input_*` while writing fresh scores
to the canonical `target_next_token_*`, coord-neighborhood, and structural
readout fields. It is read-only: no decode, patch, or training.

Fresh-score artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_prefix_next_token_readout/coord_copy_counterfactual_prefix_rows_v1
```

Command:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py \
  --stage trajectory-prefix-next-token-readout \
  --trajectory-boundary-routing-selected-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_source_region_attention/inputs/coord_copy_counterfactual_prefix_rows_v1.jsonl \
  --pair-config configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml \
  --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_prefix_next_token_readout/coord_copy_counterfactual_prefix_rows_v1 \
  --device cuda:0 \
  --allow-model-load \
  --trajectory-prefix-source trajectory \
  --families desc_first \
  --target-next-kinds coord
```

Summary:

```text
row_count: 11
source_row_count: 11
case_count: 2
readout_status_counts: {"ok": 11}
mean_target_next_token_prob: 0.001215890406
mean_target_coord_rank: 68.0
mean_fresh_minus_input_target_next_token_prob: 0.000512225639
mean_fresh_minus_input_target_next_token_rank: -3.272727272727
```

Variant-level fresh score table:

| case | variant | target | fresh prob | fresh rank | coord top1 | distance | radius4 mass |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `885` | `natural` | `coord_94` | `0.001066` | `94` | `33` | `61` | `0.008934` |
| `885` | `prev_box_to_target_bbox` | `coord_94` | `0.001975` | `90` | `33` | `61` | `0.016963` |
| `885` | `prev_box_x2_to_target` | `coord_94` | `0.002159` | `86` | `47` | `47` | `0.017803` |
| `885` | `prev_box_x2_to_far777` | `coord_94` | `0.001505` | `94` | `33` | `61` | `0.014852` |
| `885` | `prev_box_x2_to_mid333` | `coord_94` | `0.002380` | `86` | `47` | `47` | `0.020875` |
| `2685` | `natural` | `coord_170` | `0.000238` | `55` | `132` | `38` | `0.002372` |
| `2685` | `prev_box_to_target_bbox` | `coord_170` | `0.000787` | `50` | `141` | `29` | `0.006842` |
| `2685` | `prev_box_x2_to_target` | `coord_170` | `0.000494` | `53` | `141` | `29` | `0.004590` |
| `2685` | `prev_box_x2_to_far777` | `coord_170` | `0.000694` | `51` | `141` | `29` | `0.005783` |
| `2685` | `current_xy_to_target` | `coord_170` | `0.000789` | `45` | `139` | `31` | `0.006585` |
| `2685` | `prev_box_and_current_xy_to_target` | `coord_170` | `0.001288` | `44` | `141` | `29` | `0.011181` |

Routing-to-score bridge:

The fresh score recompute confirms the earlier warning about inherited score
fields. Rewritten prefixes do move the target token in the right direction, but
the movement remains small compared with the size of the coordinate basin:

- In `885`, rewriting previous x2 to the target improves the target rank from
  `94` to `86` and moves top-1 from `33` to `47`, but the target `94` is still
  far outside the dominant local anchor. The `prev_box_x2_to_mid333` rewrite
  produces a similar rank/top-1 pattern and the strongest current-prefix/current
  coord attention from the source-region run.
- In `2685`, combining previous-box and current-xy target rewrites gives the
  best fresh target rank (`44`) and highest target radius-4 mass (`0.011181`),
  but top-1 remains `141`, still well below target `170`.
- All 11 rows keep coord mass near one. The failure is not coord-token support
  collapse; it is failure to relocate the coordinate distribution's center to
  the intended object.

Mechanistic update:

The simple text-copy hypothesis is weaker after fresh scoring. If the model
were literally copying an editable previous coordinate string, target rewrites
should cause much larger target-rank jumps. Instead, rewrites act like weak
contextual nudges inside a persistent local spatial attractor. The more likely
mechanism is a stale local object/box state whose coordinate decoder is
well-typed and locally attentive but not sufficiently re-bound to the target
visual object. This keeps the model in a same-region coordinate basin even when
language-side prefix text is edited toward the target.

## Prefix Score And Source-Attention Join

New offline analysis stage:

```text
analyze-trajectory-prefix-score-attention-join
```

Implementation surface:

```text
scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py
src/analysis/autoregressive_binding_template_ablation/realized_behavior_bridge.py
tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py
```

Joined artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_prefix_score_attention_join/coord_copy_counterfactual_prefix_rows_v1
```

Command:

```bash
python scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py \
  --stage analyze-trajectory-prefix-score-attention-join \
  --trajectory-prefix-next-token-readout-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_prefix_next_token_readout/coord_copy_counterfactual_prefix_rows_v1/trajectory_prefix_next_token_readout_rows.jsonl \
  --trajectory-source-region-attention-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_source_region_attention/coord_copy_counterfactual_prefix_rows_last4_regions_v1/trajectory_source_region_attention_rows.jsonl \
  --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_prefix_score_attention_join/coord_copy_counterfactual_prefix_rows_v1
```

Summary:

```text
row_count: 11
score_row_count: 11
attention_row_count: 8448
matched_score_row_count: 11
missing_attention_score_row_count: 0
join_status_counts: {"matched": 11}
```

The join uses the score prefix and attention prefix text as part of the key,
along with family, case, source line, intervention arm, trajectory step,
counterfactual variant, target kind, and target token. This protects the
analysis from merging the natural and rewritten prefixes for the same case.

Global attention/score correlations are high but misleading because this
11-row set has only two cases. The joined summary therefore records both global
and case-centered correlations. The case-centered values are the safer
exploratory readout:

| source region | mean attention | within-case corr with target prob | within-case corr with target rank | within-case corr with coord distance | within-case corr with radius4 mass |
| --- | ---: | ---: | ---: | ---: | ---: |
| `all_previous_box_ends` | `0.003927` | `0.787589` | `-0.742651` | `-0.512728` | `0.840840` |
| `previous_box_end` | `0.002289` | `0.713848` | `-0.468810` | `-0.607643` | `0.779979` |
| `current_object_ref_boundaries` | `0.005315` | `0.553155` | `-0.624879` | `-0.148380` | `0.554250` |
| `current_descriptor` | `0.000842` | `-0.549553` | `0.496874` | `0.623870` | `-0.447770` |
| `pre_prefix_non_image_context` | `0.822985` | `-0.294167` | `0.534122` | `-0.036141` | `-0.323510` |
| `current_prefix_all` | `0.158422` | `0.225813` | `-0.520417` | `0.054053` | `0.232549` |
| `current_coords` | `0.119679` | `0.158664` | `-0.436553` | `0.139814` | `0.178924` |
| `current_box_span` | `0.125282` | `0.158097` | `-0.413935` | `0.141591` | `0.182574` |
| `pre_prefix_image_tokens` | `0.018593` | `0.120584` | `0.289765` | `-0.106402` | `0.205361` |

Interpretation:

The global table alone would overstate current-coordinate/current-box-span
attention as a score driver, because those signals partly separate the two
handpicked cases. After case-centering, the strongest positive association with
target probability and radius-4 mass is attention to previous-box-end regions,
even though the absolute mass there is tiny. This is a sharper form of the
spatial-attractor hypothesis: the previous local box is not simply copied as
surface text, but the model's small amount of routing to previous box boundary
state varies with whether a rewrite can nudge the target coordinate upward.

Current-coordinate and current-box-span attention remain important as local
slot machinery, but in this tiny joined set they are not the strongest
within-case predictor of score improvement. That suggests the next scale-up
should not only ask "does the coordinate slot attend to current coords?" It
should separate at least three signals:

1. current coord/box-span routing as local emission machinery;
2. previous box-end routing as replay-anchor pressure;
3. image-token or pre-prefix context routing as possible upstream grounding or
   missing-refresh evidence.

Scope boundary:

The correlation table is `n=11` over two cases and is exploratory. It is strong
enough to choose the next scaling direction, not to claim a universal attention
mechanism. The next run should apply the same joined analysis to a larger set of
copy-like and non-copy-like coordinate states so case-centered statistics are
not dominated by two repaired trajectories.

## Scaled Natural-State Prefix Score/Attention Bridge

New selection stage:

```text
select-trajectory-prefix-score-attention-bridge-states
```

Implementation surface:

```text
scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py
src/analysis/autoregressive_binding_template_ablation/realized_behavior_bridge.py
tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py
```

This stage selects final-layer trajectory coordinate states from the larger
hidden-state readout and writes a reusable manifest for the existing
trajectory-prefix score recompute and source-region attention stages. It
deduplicates state keys, buckets by family / onset / prediction kind /
intervention / target kind, and round-robins by `case_id` inside each bucket so
the first rows are not dominated by the lexicographically earliest case.

The first scaled selector pass (`balanced96_v1`) selected `96` rows but only
covered `7` cases. That was useful as a smoke, but the selector was adjusted to
round-robin by case before the primary scaled run below. The primary run is:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_prefix_score_attention_bridge_selection/full_coord_final_layer_perbucket4_v2
```

Selection command:

```bash
python scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py \
  --stage select-trajectory-prefix-score-attention-bridge-states \
  --trajectory-hidden-state-readout-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_state_probe/full_coord_boxend_shards/trajectory_hidden_state_readout_rows.jsonl \
  --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_prefix_score_attention_bridge_selection/full_coord_final_layer_perbucket4_v2 \
  --target-next-kinds coord \
  --selector-max-rows-per-bucket 4
```

Selection summary:

```text
input_row_count: 872
candidate_row_count: 350
selected_row_count: 142
case_count: 15
target_next_kind_counts: {"coord": 142}
onset_label_counts: {"neutral": 52, "next_step_duplicate_onset": 66, "next_step_unmatched_onset": 24}
prediction_kind_counts: {"duplicate_iou70": 22, "new_gt": 98, "repeated_gt": 22}
```

Model-backed score and attention commands:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py \
  --stage trajectory-prefix-next-token-readout \
  --trajectory-boundary-routing-selected-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_prefix_score_attention_bridge_selection/full_coord_final_layer_perbucket4_v2/trajectory_prefix_score_attention_bridge_selected_rows.jsonl \
  --pair-config configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml \
  --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_prefix_next_token_readout/full_coord_final_layer_perbucket4_v2 \
  --device cuda:0 \
  --allow-model-load \
  --trajectory-prefix-source trajectory \
  --target-next-kinds coord

CUDA_VISIBLE_DEVICES=1 python scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py \
  --stage trajectory-source-region-attention \
  --trajectory-boundary-routing-selected-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_prefix_score_attention_bridge_selection/full_coord_final_layer_perbucket4_v2/trajectory_prefix_score_attention_bridge_selected_rows.jsonl \
  --pair-config configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml \
  --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_source_region_attention/full_coord_final_layer_perbucket4_last4_regions_v2 \
  --device cuda:0 \
  --allow-model-load \
  --hidden-layers=-1,-2,-4,-8 \
  --trajectory-prefix-source trajectory \
  --value-source-regions current_object_ref_boundaries,current_descriptor,current_box_start,current_coords,current_box_span,previous_box_end,all_previous_box_ends,current_prefix_other,current_prefix_all,pre_prefix_image_tokens,pre_prefix_non_image_context,pre_prefix_context \
  --target-next-kinds coord
```

Joined artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_prefix_score_attention_join/full_coord_final_layer_perbucket4_v2
```

Join summary:

```text
row_count: 142
score_row_count: 142
attention_row_count: 109056
matched_score_row_count: 142
missing_attention_score_row_count: 0
case_count: 15
join_status_counts: {"matched": 142}
```

Behavior slices:

| slice | rows | cases | target prob | target rank | coord distance | radius4 mass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `desc_first / neutral / new_gt` | `28` | `6` | `0.098884` | `3.857` | `3.107` | `0.652503` |
| `desc_first / next_step_duplicate_onset / duplicate_iou70` | `22` | `1` | `0.011527` | `30.000` | `12.364` | `0.108513` |
| `desc_first / next_step_duplicate_onset / new_gt` | `22` | `1` | `0.044804` | `23.727` | `12.727` | `0.450440` |
| `desc_first / next_step_duplicate_onset / repeated_gt` | `22` | `1` | `0.056464` | `26.818` | `15.818` | `0.449563` |
| `desc_first / next_step_unmatched_onset / new_gt` | `24` | `2` | `0.126276` | `5.917` | `4.083` | `0.497624` |
| `geometry_first / neutral / new_gt` | `24` | `4` | `0.064907` | `3.667` | `3.667` | `0.510422` |

Case-centered attention/score correlations:

| source region | mean attention | within-case corr target prob | within-case corr target rank | within-case corr coord distance | within-case corr radius4 mass |
| --- | ---: | ---: | ---: | ---: | ---: |
| `current_coords` | `0.083093` | `0.042859` | `0.574282` | `0.638138` | `-0.690695` |
| `current_box_span` | `0.089789` | `-0.047296` | `0.505512` | `0.566487` | `-0.584878` |
| `current_prefix_all` | `0.129831` | `-0.047958` | `0.310949` | `0.271921` | `-0.485476` |
| `previous_box_end` | `0.002215` | `0.119354` | `0.298620` | `0.193417` | `0.058203` |
| `all_previous_box_ends` | `0.004746` | `0.164938` | `0.046156` | `-0.051071` | `0.178613` |
| `current_object_ref_boundaries` | `0.015415` | `-0.037705` | `-0.146312` | `-0.168154` | `-0.224324` |
| `pre_prefix_non_image_context` | `0.852768` | `0.037095` | `-0.357527` | `-0.328358` | `0.481923` |
| `pre_prefix_image_tokens` | `0.017401` | `0.019432` | `0.135775` | `0.153867` | `-0.041340` |

Group mean source-region attention:

| group | rows | cases | prev box end | all prev box ends | current coords | current box span | object-ref boundaries | image tokens | non-image context |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `neutral` | `52` | `10` | `0.002518` | `0.003233` | `0.063453` | `0.075126` | `0.019777` | `0.011081` | `0.888826` |
| `next_step_duplicate_onset` | `66` | `3` | `0.002140` | `0.005074` | `0.097046` | `0.108005` | `0.012302` | `0.017334` | `0.825612` |
| `next_step_unmatched_onset` | `24` | `2` | `0.001810` | `0.006948` | `0.058096` | `0.071463` | `0.016706` | `0.031279` | `0.849321` |

Mechanistic update:

The scaled natural-state bridge changes the interpretation of the earlier
two-case counterfactual result. In the tiny counterfactual set, previous box-end
attention rose with target-coordinate score improvement, suggesting a
replay-anchor pressure. In the broader natural-state set, previous box-end
attention is not the dominant positive score driver. It has tiny absolute mass
and its case-centered correlation with target rank is positive (`0.298620`),
meaning more previous-box-end attention is associated with worse rank in this
natural-state sample.

The strongest scaled signal is different: current coordinate and current
box-span attention are strongly associated with worse target rank, worse
coordinate distance, and lower radius-4 mass. Duplicate-onset rows also have
higher mean current-coordinate/current-box-span attention than neutral rows,
lower object-ref-boundary attention, and lower non-image-context mass. This
looks like local coordinate-slot lock-in: once the trajectory is near a
duplicate onset, the decoder is strongly routed to its active local coordinate
history, but that local machinery is not re-bound to the intended target
object.

The `next_step_unmatched_onset` rows are not the same failure. They keep much
better target rank and distance than duplicate-onset rows, have lower
current-coordinate/current-box-span attention than duplicates, and have the
largest image-token mass among the onset groups (`0.031279`). This supports the
earlier split: unmatched drift is more like unresolved or distributed visual
anchor search, while visible duplicate onset is more like a local coordinate
template re-entry.

Scope boundary:

The scaled selector has `142` rows but only `15` cases, and the duplicate-onset
subgroups still come from very few cases in this val200 bridge artifact. The
result is strong enough to demote a universal previous-box replay explanation,
but not enough to claim a dataset-wide causal mechanism. The next causal test
should target the current-coordinate/current-box-span local lock-in pathway and
object-ref-boundary refresh pathway separately, rather than only suppressing
previous-box-end routing.

## Prefix Lock-In Counterfactual Probe

New offline builder:

```text
build-trajectory-prefix-lockin-counterfactuals
```

Implementation surface:

```text
scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py
src/analysis/autoregressive_binding_template_ablation/realized_behavior_bridge.py
tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py
```

The builder duplicates selected trajectory states into executable prefix
counterfactual rows and a separate all-attempt ledger. It does not load the
model, patch activations, decode, or train. Skipped variants are kept in
`trajectory_prefix_lockin_counterfactual_attempts.jsonl` but are not fed to the
GPU scorer.

Primary full artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_prefix_lockin_counterfactuals/full_coord_perbucket4_v3
```

Builder command:

```bash
python scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py \
  --stage build-trajectory-prefix-lockin-counterfactuals \
  --trajectory-boundary-routing-selected-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_prefix_score_attention_bridge_selection/full_coord_final_layer_perbucket4_v2/trajectory_prefix_score_attention_bridge_selected_rows.jsonl \
  --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_prefix_lockin_counterfactuals/full_coord_perbucket4_v3 \
  --target-next-kinds coord
```

Builder summary:

```text
source_row_count: 142
attempt_row_count: 710
row_count: 464
case_count: 15
variant_counts:
  natural: 142
  current_object_ref_to_target_desc: 118
  current_coords_to_target_so_far: 88
  same_desc_history_coords_to_sentinel: 58
  same_desc_history_coords_to_target_bbox: 58
skipped_attempt_count: 246
```

Model-backed artifacts:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_prefix_next_token_readout/lockin_counterfactual_full_coord_perbucket4_v3
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_source_region_attention/lockin_counterfactual_full_coord_perbucket4_last4_regions_v3
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_prefix_score_attention_join/lockin_counterfactual_full_coord_perbucket4_v3
```

Join summary:

```text
score_row_count: 464
attention_row_count: 356352
matched_score_row_count: 464
missing_attention_score_row_count: 0
```

Paired deltas below are against the `natural` row with the same
`source_state_key`. All `464` joined rows preserve `source_state_key`. Negative
rank and distance deltas are better; positive probability and radius-4 mass
deltas are better.

Overall paired deltas:

| variant | rows | cases | d target prob | d rank | d distance | d radius4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `current_coords_to_target_so_far` | `88` | `15` | `-0.002610` | `-0.534091` | `-0.454545` | `-0.014642` |
| `current_object_ref_to_target_desc` | `118` | `11` | `0.000000` | `0.000000` | `0.000000` | `0.000000` |
| `same_desc_history_coords_to_sentinel` | `58` | `5` | `-0.032700` | `172.396552` | `141.068966` | `-0.262992` |
| `same_desc_history_coords_to_target_bbox` | `58` | `5` | `0.043782` | `18.034483` | `13.724138` | `0.036665` |

Duplicate-onset paired deltas:

| variant | rows | cases | d target prob | d rank | d distance | d radius4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `current_coords_to_target_so_far` | `54` | `3` | `-0.004671` | `-0.814815` | `-0.814815` | `-0.022079` |
| `current_object_ref_to_target_desc` | `66` | `3` | `0.000000` | `0.000000` | `0.000000` | `0.000000` |
| `same_desc_history_coords_to_sentinel` | `44` | `2` | `-0.027365` | `205.363636` | `169.181818` | `-0.215577` |
| `same_desc_history_coords_to_target_bbox` | `44` | `2` | `0.061553` | `-6.636364` | `-2.409091` | `0.100049` |

`duplicate_iou70` slice:

| variant | rows | cases | d target prob | d rank | d distance | d radius4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `same_desc_history_coords_to_target_bbox` | `22` | `1` | `0.044375` | `-14.090909` | `-4.545455` | `0.224702` |
| `same_desc_history_coords_to_sentinel` | `22` | `1` | `-0.000523` | `313.272727` | `112.545455` | `0.003050` |

Mechanistic update:

The cleanest new signal is not that previous same-description boxes are merely
distractors. Making those history boxes OOD with sentinel coordinates damages
target-coordinate scoring badly. But retargeting the same-description history
boxes to the target bbox substantially improves duplicate-onset rows in this
controlled prefix rewrite, especially the `duplicate_iou70` slice. This is
candidate evidence consistent with the decoder using same-description
coordinate history as an autoregressive binding prior. The synthetic
target-shaped history anchor can redirect target-coordinate logits; it does not
yet prove that natural same-description history caused the original wrong
duplicate basin in general.

This is different from a simple current-open-box lock-in explanation. Rewriting
only already-emitted current-box coordinates to the target-so-far has small and
mixed effects: slightly better rank/distance, but worse probability/radius-4
mass. That suggests the open-box local coordinate tokens are part of the basin,
but not the deepest lever in this sample.

Object-ref descriptor refresh is a no-op in the executable `v3` rows because
geometry-first rows are now skipped for this variant and same-description rows
already have the target descriptor. This supports a narrower interpretation:
language-side descriptor refresh is not the lever for these same-class duplicate
cases. The model likely needs an instance-level visual/coordinate anchor, and
the autoregressive history can provide a synthetic anchor that changes the
coordinate logits.

Attention update:

The observed same-description-history-to-target-bbox duplicate-onset improvement
does not require a large final-query source-attention redistribution. In the
duplicate-onset slice its mean current-box/current-coord attention changes are
small, while target probability and rank improve. This points toward token
content and residual/logit basin effects in the prefix history as the next
hypothesis to test, not just visible attention-mass reallocation at the final
query token.

Scope boundary:

This is still a `15`-case / `464`-row prefix-counterfactual bridge. The
duplicate-onset improvement is concentrated in very few cases (`2` cases overall
and `1` case for the `duplicate_iou70` slice), and the key positive variant uses
a synthetic target-shaped history anchor. Treat it as a high-value mechanism
candidate, not as a dataset-wide estimate or proof of the natural causal path.
The next deeper test should localize which layers/heads read same-description
history coordinates and whether the same target-history effect transfers to the
new template/checkpoint pair rather than only this ckpt928 val200 bridge
artifact.

## Same-Description History Value-Route Probe

Implementation update:

```text
trajectory_boundary_anchor_regions_from_token_texts
analyze-trajectory-lockin-value-contribution
```

The source-region extractor now exposes same-description history regions when a
closed prior object has the same descriptor as the current open object:

```text
same_desc_history_object_ref_boundaries
same_desc_history_coords
same_desc_history_box_ends
same_desc_history_box_spans
```

These regions are available to both `trajectory-source-region-attention` and
`trajectory-boundary-head-value-contribution`, so the lock-in counterfactual
effect can be decomposed without broad prefix buckets hiding the literal
same-description history coordinates.

Smoke input:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_prefix_lockin_head_value_inputs/same_desc_history_smoke_v1
```

Smoke model-backed artifacts:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_value_contribution/lockin_same_desc_history_smoke_heads26_10_21_2_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_source_region_attention/lockin_same_desc_history_smoke_last_layer_regions_v1
```

The smoke used six rows from the `desc_first-2685-33-11-desc_end`
`duplicate_iou70` case: natural, sentinel-history, and target-history variants
for two coordinate slots. Fresh baseline scores from the value-contribution
stage confirmed the prefix rewrite effect:

| slot | variant | target prob | target rank | top coord |
| --- | --- | ---: | ---: | --- |
| `x1` | natural | `0.010354` | `12` | `<|coord_118|>` |
| `x1` | same-desc sentinel | `0.000126` | `599` | `<|coord_511|>` |
| `x1` | same-desc target bbox | `0.015009` | `5` | `<|coord_124|>` |
| `y1` | natural | `0.000891` | `47` | `<|coord_349|>` |
| `y1` | same-desc sentinel | `0.000000044` | `787` | `<|coord_501|>` |
| `y1` | same-desc target bbox | `0.149332` | `3` | `<|coord_343|>` |

The smoke attention result did not show a large final-layer attention reroute to
the literal same-description history coordinates. Mean normalized attention to
`same_desc_history_coords` moved only from `0.009526` natural to `0.013228`
under target-history rewrite, while `pre_prefix_non_image_context` remained the
dominant mass (`0.772133` natural, `0.788179` target-history).

Scaled value-contribution run:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_prefix_lockin_head_value_inputs/same_desc_history_full_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_value_contribution/lockin_same_desc_history_full_heads26_10_21_2_v1
```

Execution scope:

```text
source states: 58
input rows: 174 = natural + sentinel + target-history for each source state
cases: 5
value-contribution rows: 7044
candidate heads: 26:10, 21:2
readout-only: true
model perturbation: false
training: false
shard devices: cuda:0,cuda:1,cuda:2,cuda:4
```

Reusable paired-delta analysis artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_value_contribution/lockin_same_desc_history_full_heads26_10_21_2_v1/lockin_same_desc_history_value_contribution_analysis.md
```

Key paired deltas are natural-vs-counterfactual within the same
`source_state_key`, coord slot, head, and source region:

| variant | head / region | pairs | d prob | d rank | d value projection | d attention | d head projection |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| same-desc target bbox | `26:10 / pre_prefix_non_image_context` | `58` | `+0.047090` | `+17.4828` | `+1.94351` | `-0.002075` | `+1.92721` |
| same-desc target bbox | `26:10 / same_desc_history_coords` | `58` | `+0.047090` | `+17.4828` | `-0.035432` | `-0.000807` | `+1.92721` |
| same-desc target bbox | `26:10 / same_desc_history_box_spans` | `58` | `+0.047090` | `+17.4828` | `+0.029747` | `-0.001452` | `+1.92721` |
| same-desc sentinel | `26:10 / pre_prefix_non_image_context` | `58` | `-0.030334` | `+175.793` | `+1.88702` | `+0.028838` | `+3.87033` |
| same-desc sentinel | `26:10 / same_desc_history_coords` | `58` | `-0.030334` | `+175.793` | `+0.007131` | `-0.004257` | `+3.87033` |
| same-desc sentinel | `26:10 / same_desc_history_box_spans` | `58` | `-0.030334` | `+175.793` | `+0.140699` | `-0.006064` | `+3.87033` |

Mechanistic update:

The same-description history rewrite is real as a prefix-level lever: replacing
history coordinates with sentinel values damages coordinate selection badly, and
target-shaped history coordinates raise target probability. However, the
head/value route is not a simple literal read of the rewritten coordinate
tokens. In the scaled head-value decomposition, the largest projection deltas
for the dominant candidate head `26:10` come from the broad
`pre_prefix_non_image_context` bucket, while the literal
`same_desc_history_coords` region has tiny attention and small projection
deltas.

This demotes a naive "the final query attends to previous same-class coordinate
tokens and copies them" explanation. The stronger current hypothesis is that
same-description history coordinates alter a broader prefix-state/residual
basin, possibly via layer-normalized context state, accumulated non-image
prefix channels, or earlier-layer routing that is not visible as final-query
attention to the literal coord tokens. The next high-value test should therefore
compare residual/hidden deltas between natural, sentinel-history, and
target-history prefixes by layer, then patch or project those deltas before the
coordinate-slot LM-head readout.

Scope boundary:

This is still a checkpoint-928 val200 same-description-history subset:
`58` source states from `5` cases. It sharpens the route-localization question
but does not yet prove the mechanism transfers to the new checkpoint/template
pair or to all duplication modes.

## Same-Description History Hidden Delta Probe

Implementation update:

```text
analyze-trajectory-lockin-hidden-delta
```

This post-hoc analyzer pairs hidden-state readout rows by
`source_state_key`, target coordinate slot, and hidden layer. It compares
natural prefixes against the same `same_desc_history_coords_to_sentinel` and
`same_desc_history_coords_to_target_bbox` counterfactual variants used above,
then reports deltas through the LM-head lens and the
`token_embeddings_adapter` surfaces. It does not store hidden vectors, patch
activations, decode, or train.

Input and model-backed readout:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_state_probe/lockin_same_desc_history_full_layers0_4_8_12_16_20_24_28_v1
```

Execution scope:

```text
source states: 58
input rows: 174 = natural + sentinel + target-history for each source state
hidden layers: 0,4,8,12,16,20,24,28
hidden readout rows: 1392
cases: 5
readout-only: true
model perturbation: false
training: false
shard devices: cuda:0,cuda:1,cuda:2,cuda:4
```

Reusable hidden-delta report:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_state_probe/lockin_same_desc_history_full_layers0_4_8_12_16_20_24_28_v1/lockin_same_desc_history_hidden_delta_analysis.md
```

Key layer deltas:

| variant | layer | pairs | d prob | d rank | d coord r4 | d coord dist | d surface score | d surface r4 | d hidden norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| same-desc target bbox | `0` | `58` | `0` | `0` | `0` | `0` | `0` | `0` | `0` |
| same-desc target bbox | `12` | `58` | `+0.000000036` | `-1057.78` | `+0.000031` | `-2.724` | `+0.028015` | `+0.000030` | `+0.129` |
| same-desc target bbox | `20` | `58` | `+0.000025` | `-118.78` | `-0.000288` | `+46.67` | `-1.39444` | `+0.007821` | `-9.433` |
| same-desc target bbox | `24` | `58` | `+0.002202` | `+4510.76` | `+0.015678` | `+15.71` | `+1.07667` | `-0.051532` | `-10.992` |
| same-desc target bbox | `28` | `58` | `+0.043782` | `+18.03` | `+0.036623` | `+13.72` | `-1.00787` | `-0.009019` | `+159.95` |
| same-desc sentinel | `0` | `58` | `0` | `0` | `0` | `0` | `0` | `0` | `0` |
| same-desc sentinel | `20` | `58` | `-0.000133` | `+514.45` | `-0.002362` | `-98.28` | `-4.61086` | `-0.020100` | `-2.493` |
| same-desc sentinel | `24` | `58` | `-0.012448` | `+5069.41` | `-0.148106` | `+133.24` | `-26.1647` | `-0.172718` | `-32.222` |
| same-desc sentinel | `28` | `58` | `-0.032699` | `+172.40` | `-0.263078` | `+141.07` | `-169.422` | `-0.361944` | `+211.588` |

Interpretation:

The sentinel-history failure is not a shallow lexical disturbance. Layers
`0-16` show only tiny LM-head/adapter deltas, layer `20` starts to separate,
and layers `24` and `28` carry the major coordinate-basin collapse. The final
layer reproduces the earlier prefix-score result: target-history increases
target-coordinate probability, while sentinel-history sharply lowers target
probability, rank, radius-4 mass, and adapter-surface score.

The target-history variant is more ambiguous than a simple rescue: final target
probability improves, but average rank and distance are still worse overall in
this full `58`-state slice. This matches the earlier warning that the positive
target-history effect is concentrated in duplicate-onset / duplicate-iou70
subsets rather than being a universal fix.

Mechanistic update:

The combined value-route and hidden-delta probes point to a late residual basin.
Same-description history coordinates perturb the model weakly in early layers,
but the decisive coordinate-slot basin is formed or amplified in late layers,
especially layer `24` into the final layer. Since final-query attention to the
literal history coordinate tokens is small, the next causal probe should patch
late hidden/residual deltas, not only suppress attention to coordinate tokens.

Next recommended probe:

Select paired natural/sentinel/target-history rows from the duplicate-onset
subset and run a bounded causal activation patch at layers `20`, `24`, and `28`:

```text
target-history minus natural delta
natural minus sentinel-history delta
```

Measure whether those directions move `target_next_token_prob`,
`target_coord_radius4_mass`, rank, and greedy coordinate choice before trying
any broader training or decoding experiment.

## Paired Hidden-Delta Causal Activation Patch Smoke

Implementation update:

```text
trajectory-hidden-causal-activation-patch
paired_same_desc_history_target_bbox_minus_natural
paired_natural_minus_same_desc_history_sentinel
```

The causal activation patch stage now supports paired hidden-delta direction
bases. For each selected natural row, it looks up peer rows with the same
`source_state_key` and counterfactual variants:

```text
same_desc_history_coords_to_target_bbox
same_desc_history_coords_to_sentinel
```

and patches the natural execution with either target-history minus natural or
natural minus sentinel-history hidden deltas. The implementation deliberately
keeps these as activation patches only: no model weights are changed, no
training runs, and the stage records `model_perturbation_ran=true` only when a
real activation patch was executed.

Review hardening added in this round:

```text
paired bases default selection to counterfactual_variant=natural
paired bases require explicit --patch-strengths
duplicate source_state_key / counterfactual_variant peers fail fast
patched rows expose target-token and coord-basin readouts
```

The explicit-strength requirement matters because the legacy
`32,64,96,128,160` output-embedding direction strengths are too large for raw
hidden-delta paired bases. The smoke below used small scale factors:

```text
0.25,0.5,1.0
```

Smoke input:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/paired_lockin_smoke_input/selected_rows.jsonl
```

Model-backed smoke artifacts:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/paired_lockin_smoke_layer20_v2
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/paired_lockin_smoke_layer24_v2
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/paired_lockin_smoke_layer28_v2
```

Execution scope:

```text
selected source states: 1
input rows: natural + target-history peer + sentinel-history peer
target next kind: coord
patch rows per run: 9
realized paired direction patches per run: 6
patch strengths: 0.25,0.5,1.0
devices: cuda:0,cuda:1,cuda:2
readout-only: false
model perturbation: true
training: false
```

The smoke intentionally exercised the paired-basis default selection: the CLI
did not pass `--counterfactual-variants`, and each summary records
`counterfactual_variants=["natural"]` with
`counterfactual_variants_defaulted_to_natural_for_paired_bases=true`.

Key smoke deltas relative to the no-patch baseline. The baseline target token
was `<|coord_123|>`, while the unpatched greedy/top coordinate was
`<|coord_118|>` at distance `5`.

| patch | strength | target prob delta | target rank | radius-4 delta | top coord | top distance | target-vs-greedy margin |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `L20->28 target-history - natural` | `0.25` | `+0.001219` | `7` | `+0.015871` | `118` | `5` | `-0.5` |
| `L20->28 target-history - natural` | `1.0` | `+0.001580` | `12` | `+0.049287` | `118` | `5` | `-0.5` |
| `L24->28 target-history - natural` | `1.0` | `+0.002538` | `10` | `+0.043962` | `124` | `1` | `-0.25` |
| `L28->24 target-history - natural` | `1.0` | `+0.001519` | `12` | `+0.035231` | `124` | `1` | `-0.25` |
| `L24->28 natural - sentinel-history` | `1.0` | `+0.009431` | `13` | `+0.125326` | `118` | `5` | `-1.125` |
| `L28->24 natural - sentinel-history` | `1.0` | `+0.002926` | `20` | `+0.072825` | `118` | `5` | `-1.375` |

Interpretation:

This one-state smoke separates a useful local-basin signal from a true target
lock-in signal. Both paired directions can increase radius-4 coordinate mass
around the target, but they do not mean the same thing. The target-history minus
natural direction can move the greedy coordinate from `118` to `124`, one bin
from the target `123`, and improve the target-vs-greedy margin at late patch
sites. The natural minus sentinel-history direction increases target-neighborhood
mass more strongly in some settings, but keeps the top coordinate anchored at
`118` and worsens target rank/margin. That looks more like broad coordinate-basin
stabilization than object-specific target binding.

Mechanistic update:

The promising bridge is now concrete enough to scale: use paired hidden-delta
activation patches as a causal test of the late residual-basin hypothesis. The
next run should move from this one-state smoke to the `58` source-state
duplicate-onset slice and stratify by whether the patch changes top coordinate,
target rank, radius-4 mass, and stop behavior. The key distinction to preserve
is:

```text
local coordinate-neighborhood stabilization
vs.
object-specific target binding / lock-in
```

Scope boundary:

This is a tiny smoke over one selected coordinate-slot state from checkpoint
`928`; it validates the tooling and gives a directional signal, but it is not a
dataset-level mechanistic result. The earlier `v1` smoke did not include the
target-token and coord-basin metrics added here; treat the `v2` artifacts as the
interpretable version.

## Scaled Paired Hidden-Delta Causal Patch

Model-backed scaled artifacts:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/lockin_same_desc_history_full_layer20_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/lockin_same_desc_history_full_layer24_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/lockin_same_desc_history_full_layer28_v1
```

Execution scope:

```text
input rows: 174 = natural + sentinel-history + target-history
selected natural source states: 58
cases: 5
target next kind: coord
patch rows per layer-pair run: 522
realized paired direction patches per run: 348
patch strengths: 0.25,0.5,1.0
layer pairs: 20->28, 24->28, 28->24
devices: cuda:0,cuda:1,cuda:2
readout-only: false
model perturbation: true
training: false
```

Summary deltas below are patch-row values minus each source state's
`baseline_no_patch` row. Negative `d rank` and `d distance` are good; positive
`d prob`, `d radius4`, and `d margin` are good.

| layer pair | direction | strength | d prob | d rank | d radius4 | d distance | d margin | prob up | rank improved | closer | exact |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `20->28` | target-history - natural | `1.0` | `+0.003980` | `+22.17` | `-0.053128` | `+15.28` | `+0.506` | `70.7%` | `41.4%` | `3.4%` | `10.3%` |
| `24->28` | target-history - natural | `0.5` | `+0.048125` | `+16.29` | `-0.012423` | `+8.41` | `+1.145` | `70.7%` | `58.6%` | `20.7%` | `20.7%` |
| `24->28` | target-history - natural | `1.0` | `+0.043953` | `+18.83` | `+0.033963` | `+14.41` | `+2.239` | `70.7%` | `55.2%` | `41.4%` | `20.7%` |
| `28->24` | target-history - natural | `0.25` | `+0.026790` | `-2.78` | `+0.002536` | `-0.03` | `+0.596` | `70.7%` | `48.3%` | `10.3%` | `10.3%` |
| `28->24` | target-history - natural | `0.5` | `+0.052735` | `-0.05` | `+0.002283` | `+10.60` | `+1.161` | `70.7%` | `65.5%` | `20.7%` | `20.7%` |
| `28->24` | target-history - natural | `1.0` | `+0.043788` | `+17.97` | `+0.036620` | `+13.72` | `+2.274` | `70.7%` | `55.2%` | `51.7%` | `20.7%` |
| `24->28` | natural - sentinel-history | `1.0` | `+0.001369` | `-2.50` | `+0.014432` | `-0.69` | `+0.323` | `67.2%` | `31.0%` | `20.7%` | `10.3%` |
| `28->24` | natural - sentinel-history | `0.5` | `+0.002588` | `+1.55` | `+0.020392` | `-0.86` | `+0.006` | `60.3%` | `41.4%` | `10.3%` | `10.3%` |

Strict good-patch rates:

```text
criterion A: target prob up + target rank improves + top coordinate not farther
criterion B: target prob up + top coordinate closer
catastrophic: d distance >= 20 or d rank >= 100
```

| layer pair | direction | strength | A | B | catastrophic |
| --- | --- | ---: | ---: | ---: | ---: |
| `24->28` | target-history - natural | `0.5` | `58.6%` | `20.7%` | `8.6%` |
| `24->28` | target-history - natural | `1.0` | `55.2%` | `41.4%` | `8.6%` |
| `28->24` | target-history - natural | `0.25` | `41.4%` | `10.3%` | `0.0%` |
| `28->24` | target-history - natural | `0.5` | `65.5%` | `20.7%` | `8.6%` |
| `28->24` | target-history - natural | `1.0` | `55.2%` | `51.7%` | `8.6%` |
| `24->28` | natural - sentinel-history | `1.0` | `20.7%` | `20.7%` | `0.0%` |
| `28->24` | natural - sentinel-history | `0.5` | `24.1%` | `10.3%` | `0.0%` |

Interpretation:

The scaled result confirms the smoke but adds an important correction. The
target-history minus natural hidden delta is a real causal handle on the
coordinate-slot basin, strongest at late patch sites (`24->28` and `28->24`).
It consistently raises the target coordinate probability in about `70.7%` of
states and can improve target rank in more than half of states. However, it is
not a clean object-binding vector. It also contains a brittle coordinate-anchor
component: a small cluster of states, mostly from `desc_first-139-0-11-desc_end`
and one `desc_first-139-0-13-desc_end` state, jumps to far competing coordinate
bins and dominates the mean distance/rank damage.

The most revealing examples are paired:

```text
good: desc_first-2685-33-11 y1, target 341
baseline top 349 -> patched top 343/342
target prob about 0.00084 -> 0.13-0.15
rank 47/41 -> 3

bad: desc_first-139-0-11 x1, target 645
baseline top 644 -> patched top 463
target prob about 0.05335 -> 0.00039-0.00048
rank 3 -> 233/234
```

The natural minus sentinel-history direction is much safer but weaker. It often
increases local radius-4 mass and slightly improves distance, but it rarely
creates the target-specific lock-in seen in the best target-history patches.
That makes it more like a coordinate-basin stabilizer than a binding vector.

Mechanistic update:

The late residual basin appears to have at least two separable components:

```text
1. target-specific binding pressure
   raises target probability/rank and sometimes moves the greedy coordinate
   toward the target;

2. anchor/basin drift
   can move the coordinate slot toward a different strong coordinate basin,
   especially when the baseline is already near a high-confidence coordinate.
```

This is deeper than a simple "copy prior same-class coordinates" picture. The
same paired hidden delta can be a rescue vector for one coordinate slot and a
collapse vector for another, even within the same small case family. The next
probe should decompose the target-history hidden delta into target-improving
versus catastrophic subspaces. A concrete next move is to build paired delta
projection/readout by outcome group:

```text
positive group: prob up + rank improves + not farther
catastrophic group: d distance >= 20 or d rank >= 100
```

Then compare which token/source regions and which coordinate slots produce the
shared direction. If the catastrophic group has a coherent projection signature,
we can test a guarded patch such as:

```text
target-history delta minus catastrophic mean direction
```

or use orthogonalized patching to separate object-specific binding from
coordinate-anchor drift before considering any tiny training step.

## Target-History Bifurcation Sweep

Additional target-history-only curve artifacts:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/lockin_same_desc_history_full_layer24_target_curve_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/lockin_same_desc_history_full_layer28_target_curve_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/lockin_same_desc_history_full_layer24_target_threshold_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/lockin_same_desc_history_full_layer28_target_threshold_v1
```

The curve and threshold sweeps used the same `58` natural source states as the
scaled run, but only the `target-history - natural` paired hidden-delta basis.
They cover strengths:

```text
0.10,0.15,0.20,0.25,0.30,0.32,0.34,0.36,0.38,0.40,0.45,0.50
```

High-level threshold result:

| layer pair | strength | d prob | d rank | d distance | A-good | catastrophic |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `24->28` | `0.15` | `+0.011841` | `-2.00` | `+0.03` | `48.3%` | `0.0%` |
| `24->28` | `0.34` | `+0.034635` | `+4.17` | `+2.74` | `56.9%` | `1.7%` |
| `24->28` | `0.36` | `+0.040459` | `+4.59` | `+3.22` | `56.9%` | `8.6%` |
| `24->28` | `0.40` | `+0.042242` | `+6.72` | `+3.50` | `69.0%` | `8.6%` |
| `28->24` | `0.25` | `+0.026790` | `-2.78` | `-0.03` | `41.4%` | `0.0%` |
| `28->24` | `0.32` | `+0.036445` | `-2.60` | `+2.19` | `48.3%` | `1.7%` |
| `28->24` | `0.40` | `+0.044645` | `-1.67` | `+1.21` | `65.5%` | `1.7%` |
| `28->24` | `0.50` | `+0.052735` | `-0.05` | `+10.60` | `65.5%` | `8.6%` |

`A-good` here means target probability increases, target rank improves, and
top coordinate is not farther from the target. `Catastrophic` means
`d distance >= 20` or `d rank >= 100`.

The most important split is not the mean. It is a thresholded basin transition:

```text
24->28:
  desc_first-139-0-11 x1 starts catastrophic at about 0.36
  desc_first-139-0-13 y1 starts catastrophic at about 0.25

28->24:
  desc_first-139-0-13 y1 starts catastrophic at about 0.30
  desc_first-139-0-11 x1 remains near the original basin until about 0.50
```

Independent post-hoc review of the scaled artifacts reached the same core
interpretation: the target-history paired delta is helpful for duplicate or
repeated same-description continuation states, especially y-coordinate slots,
but harmful for `new_gt` unmatched x-slot states. The repeated basin switches
are stable enough to treat as mechanistic signatures rather than noise:

```text
target-history catastrophic signatures:
  645 -> 463/495/556/494
  693 -> 677/685/689
  726 -> 608/607/595/566/581

natural-sentinel late catastrophic signature:
  1 -> 269
```

Targeted dense bifurcation input:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/lockin_same_desc_history_targeted_bifurcation_input
```

Targeted dense bifurcation artifacts:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/lockin_same_desc_history_targeted_bifurcation_layer24_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/lockin_same_desc_history_targeted_bifurcation_layer28_v1
```

Execution scope:

```text
selected source states: 23
input rows: 69 = natural + sentinel-history + target-history
cases: 4
patch strengths: 0.00,0.02,...,0.60
layer pairs: 24->28, 28->24
patch rows per run: 782
realized target-history direction patches per run: 713
model perturbation: true
training: false
```

Key transition paths:

| layer pair | state | top-coordinate path as strength rises | best useful regime |
| --- | --- | --- | --- |
| `24->28` | `new_gt x1 target 645` | `644 -> 630 -> 628 -> 620 -> 495 -> 556 -> 495 -> 494` | only tiny `0.02`; after `0.18` target prob collapses |
| `28->24` | `new_gt x1 target 645` | `644 -> 649 -> 519 -> 463 -> 460 -> 466` | safer until about `0.46`; collapse after `0.50` |
| `24->28` | `new_gt x2 target 693` | `691 -> 693 -> 691 -> 689 -> 691 -> 693 -> 689 -> 685` | exact target appears at `0.12` and `0.26`, but probability declines |
| `28->24` | `new_gt x2 target 693` | `691 -> 693 -> 689 -> 693 -> 689 -> 693 -> 689 -> 685 -> 693` | exact target reappears several times; not monotone |
| `24->28` | `new_gt y1 target 726` | `716 -> 607 -> 608 -> 607 -> 595 -> 607 -> 595 -> 608 -> 566` | no useful positive regime |
| `28->24` | `new_gt y1 target 726` | `716 -> 608 -> 595 -> 607 -> 608 -> 591 -> 595 -> 591 -> 581` | no useful positive regime |
| `24->28` | `duplicate x1 target 123` | `118 -> 124` only at `0.60` | rank/prob best around `0.52-0.56` before top flips |
| `28->24` | `duplicate x1 target 123` | `118 -> 116 -> 118` | low-amplitude rank/prob gain; no target lock-in |
| `24->28` | `duplicate y1 target 341` | top stays `349` through `0.60` | probability/rank improve strongly without top flip |
| `28->24` | `duplicate y1 target 341` | `349 -> 348` near `0.38-0.60` | monotone probability/rank improvement |
| `24/28` | `repeated person y1 target 1` | `0 -> 1` at `0.36-0.38` | clean snap to exact target; prob peaks around `0.50-0.54` |

Mechanistic update:

This looks like a discrete attractor crossing, not a smooth correction vector.
The paired target-history delta pushes the coordinate slot through learned
coordinate basins. In helpful repeated/duplicate cases, the next basin is often
the target or a near-target local basin. In `new_gt` unmatched cases, the same
direction crosses into a far learned anchor. The non-monotone exact-target
reappearances for `x2 target 693` are especially telling: the vector is moving
through a rugged coordinate energy landscape, not simply increasing target
evidence.

The current best hypothesis is therefore:

```text
same-description history creates a late residual coordinate-basin steering
field. Autoregressive duplication happens when this field aligns with a
same-description continuation basin; false/helpful guidance failures happen
when the same field crosses into a competing coordinate anchor before target
binding stabilizes.
```

Post-hoc bifurcation analyzer:

Implemented a reusable read-only stage:

```text
python scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py \
  --stage analyze-trajectory-causal-patch-bifurcation \
  --trajectory-hidden-causal-patch-rows <trajectory_hidden_causal_activation_patch_rows.jsonl> \
  --output-root <bifurcation_analysis>
```

The stage emits:

```text
trajectory_causal_patch_bifurcation_state_rows.jsonl
trajectory_causal_patch_bifurcation_transition_rows.jsonl
trajectory_causal_patch_bifurcation_summary.json
trajectory_causal_patch_bifurcation_analysis.md
```

It is a post-hoc artifact reader only:

```text
model_perturbation_ran: false
training_ran: false
readout_only: true
```

The analysis key is variant-aware for artifacts that contain multiple
counterfactual variants under the same `source_state_key`, and duplicate
direction-patch points fail fast instead of silently reweighting grouped rates.

Regenerated normalized bifurcation outputs:

| artifact | states | points | transitions | catastrophic states | exact-target states |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lockin_same_desc_history_full_layer20_v1/bifurcation_analysis` | 116 | 348 | 73 | 9 | 12 |
| `lockin_same_desc_history_full_layer24_v1/bifurcation_analysis` | 116 | 348 | 93 | 5 | 22 |
| `lockin_same_desc_history_full_layer28_v1/bifurcation_analysis` | 116 | 348 | 86 | 9 | 18 |
| `lockin_same_desc_history_full_layer24_target_threshold_v1/bifurcation_analysis` | 58 | 406 | 70 | 5 | 18 |
| `lockin_same_desc_history_full_layer28_target_threshold_v1/bifurcation_analysis` | 58 | 406 | 63 | 5 | 20 |
| `lockin_same_desc_history_targeted_bifurcation_layer24_v1/bifurcation_analysis` | 23 | 713 | 68 | 5 | 8 |
| `lockin_same_desc_history_targeted_bifurcation_layer28_v1/bifurcation_analysis` | 23 | 713 | 70 | 5 | 8 |

The analyzer confirms the hand inspection above while making transition
thresholds and destination-basin signatures queryable. The next high-value
implementation direction is now:

1. Build destination-basin controls for `645 -> 463/495`, `726 -> 608/595`,
   and `1 -> 269`: measure whether the patch raises destination-basin logit
   before target-basin logit.
2. Try orthogonalized patches:

```text
target-history delta minus new_gt-catastrophic mean direction
target-history delta minus destination-basin direction
```

The goal is to separate the duplicate/repeated binding component from the
new-object anchor drift component before any training intervention.

Destination-basin probe-bin replay:

Added optional coordinate probe-bin readouts to the model-backed causal patch
stage so a replay can score known destination basins before they become top-1:

```text
--probe-coord-bins 645,726,519,463,460,466,620,495,556,494,608,595,591,581,607,566
```

Tiny replay input:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/destination_basin_probe_input/selected_rows.jsonl
```

Scope:

```text
source rows: 6 = 2 natural states + target/sentinel peers
states: desc_first-139-0-11 self_noop x1 target 645
        desc_first-139-0-13 target_coord_seed y1 target 726
patch strengths: 0.00,0.02,...,0.60
layer pairs: 24->28, 28->24
probe coord bins: 16
rows per run: 74
model perturbation: true
training: false
```

Probe artifacts:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/destination_basin_probe_layer24_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/destination_basin_probe_layer28_v1
```

Key evidence from coordinate-conditioned probabilities/ranks:

| layer pair | state | pre-takeover signal | takeover / post-takeover signal |
| --- | --- | --- | --- |
| `24->28` | `x1 target 645` | at `0.24`, target rank is already `25` while destination `620` is rank `23`; at `0.32`, `620` reaches rank `5` before top flips | at `0.36`, `620` becomes rank/top `1`; by `0.42-0.60`, `495/556/494` become rank `1-5` while target rank falls to `150-270` |
| `28->24` | `x1 target 645` | target remains top-family through `0.44`, then at `0.46` destination `519` is rank `3` while target rank is `9` and top only moves near-target to `649` | at `0.50`, `519` is rank/top `1`; at `0.52-0.60`, `519/463/460/466` share or take rank `1` while target rank falls to `29-55` |
| `24->28` | `y1 target 726` | first measured bad basin is already decisive: at `0.24`, `608/607` are rank `1` while target rank is `60` | later scales cycle among `608/607/595/566`; by `0.44-0.60`, `566` becomes rank/top `1` and target rank falls to `188-249` |
| `28->24` | `y1 target 726` | at `0.24`, `608/607/595` are rank `4` while target rank is `35`; at `0.26`, they rise to rank `2-3` | at `0.28`, `608` becomes rank/top `1`; later `608/607/595/591/581` repeatedly share rank `1-2` while target rank falls to `44-198` |

Mechanistic implication:

The wrong destination is not merely a byproduct of the final argmax flip. Its
basin receives measurable coordinate-conditioned probability/rank support before
visible takeover, especially in the `28->24` `x1 target 645` case where `519`
becomes coordinate-conditioned rank `3` while the surface top still looks
near-target (`649`). This strengthens the current picture: same-description
history introduces a coordinate-basin field that first weakens target binding,
then raises one or more destination anchors, then the autoregressive top
coordinate snaps into that basin family. Full-vocab probe probabilities are
also emitted in the artifact and should be used when making absolute probability
mass claims.

Post-hoc destination-basin analyzer:

Implemented a reusable read-only stage:

```text
python scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py \
  --stage analyze-trajectory-causal-patch-destination-basin \
  --trajectory-hidden-causal-patch-rows <trajectory_hidden_causal_activation_patch_rows.jsonl> \
  --output-root <destination_basin_analysis>
```

The stage emits:

```text
trajectory_causal_patch_destination_basin_rows.jsonl
trajectory_causal_patch_destination_basin_summary.json
trajectory_causal_patch_destination_basin_analysis.md
```

It is a post-hoc artifact reader only:

```text
model_perturbation_ran: false
training_ran: false
readout_only: true
```

Real rerun outputs:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/destination_basin_probe_layer24_v1/destination_basin_analysis
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/destination_basin_probe_layer28_v1/destination_basin_analysis
```

Scope:

```text
source artifacts: destination_basin_probe_layer24_v1 and destination_basin_probe_layer28_v1
states per artifact: 2
probe destination rows per artifact: 30
requested coord-probe panel: fixed 16 bins, not exhaustive top-K
threshold tracked here: first destination rank <= 5 before exact destination top1
```

Aggregate destination-basin traces:

| layer pair | rows | exact destination top1 | basin top1 within radius 4 | pre-top rank<=5 destination traces | mean first rank<=5 scale | mean exact top1 scale | mean lead scale |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `24->28` | 30 | 8 | 9 | 9 | 0.328889 | 0.387500 | 0.062500 |
| `28->24` | 30 | 9 | 9 | 8 | 0.386667 | 0.453333 | 0.062222 |

Representative trace rows:

| layer pair | state | target | destination | baseline dest rank | first dest rank<=5 | first exact dest top1 | lead | dest rank at onset | target rank at onset |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `24->28` | `desc_first-139-0-11 x1` | 645 | 494 | 342 | 0.42 | 0.58 | 0.16 | 5 | 150 |
| `24->28` | `desc_first-139-0-13 y1` | 726 | 595 | 35 | 0.24 | 0.36 | 0.12 | 5 | 60 |
| `24->28` | `desc_first-139-0-13 y1` | 726 | 608 | 33 | 0.20 | 0.26 | 0.06 | 4 | 42 |
| `28->24` | `desc_first-139-0-13 y1` | 726 | 591 | 39 | 0.28 | 0.44 | 0.16 | 5 | 44 |
| `28->24` | `desc_first-139-0-13 y1` | 726 | 607 | 36 | 0.24 | 0.36 | 0.12 | 4 | 35 |
| `28->24` | `desc_first-139-0-11 x1` | 645 | 519 | 30 | 0.46 | 0.50 | 0.04 | 3 | 9 |

Mechanistic update:

The new analyzer makes the earlier hand-inspection falsifiable at the artifact
level: in both layer directions, a substantial subset of requested non-target
destination bins becomes rank-competitive before exact visible top1 takeover.
The strongest cases are not local coordinate noise around the target: they are
far destination basins such as `645 -> 494/495/519/620` and
`726 -> 591/595/607/608`, often with the target rank already degraded at the
same onset scale. Because the probe panel is fixed and sparse, this evidence
should be read as "known wrong basins are already supported before takeover",
not "all possible destination basins have been enumerated."

Post-hoc probe-support trace analyzer:

Implemented a reusable read-only stage:

```text
python scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py \
  --stage analyze-trajectory-causal-patch-probe-support \
  --trajectory-hidden-causal-patch-rows <trajectory_hidden_causal_activation_patch_rows.jsonl> \
  --output-root <probe_support_analysis>
```

The stage emits:

```text
trajectory_causal_patch_probe_support_rows.jsonl
trajectory_causal_patch_probe_support_strongest_rows.jsonl
trajectory_causal_patch_probe_support_summary.json
trajectory_causal_patch_probe_support_analysis.md
```

It is a post-hoc artifact reader only:

```text
model_perturbation_ran: false
training_ran: false
readout_only: true
```

Real rerun outputs:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/destination_basin_probe_layer24_v1/probe_support_analysis
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/destination_basin_probe_layer28_v1/probe_support_analysis
```

Scope:

```text
source artifacts: destination_basin_probe_layer24_v1 and destination_basin_probe_layer28_v1
source rows per artifact: 74
probe support rows per artifact: 1184 = 74 rows * 16 requested bins
off-target probe rows per artifact: 1110
requested coord-probe panel: fixed 16 bins, not exhaustive top-K
```

Dense probe-support rates:

| layer pair | source rows | off-target rows | off-target rank<=10 | off-target prob>=0.01 | any off-target rank<=10 | any off-target prob>=0.01 | any off-target r4>=0.05 | any previsible off-target rank<=10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `24->28` | 74 | 1110 | 0.114414 | 0.107207 | 0.500000 | 0.594595 | 0.743243 | 0.486486 |
| `28->24` | 74 | 1110 | 0.115315 | 0.127928 | 0.445946 | 0.540541 | 0.689189 | 0.445946 |

Within the paired target-history direction only:

| layer pair | direction rows | off-target rank<=10 | off-target prob>=0.01 | any off-target rank<=10 | any previsible off-target rank<=10 | support tiers |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `24->28` | 62 | 0.136559 | 0.125806 | 0.596774 | 0.580645 | `{'loose': 68, 'moderate': 144, 'none': 664, 'strong': 116}` |
| `28->24` | 62 | 0.137634 | 0.152688 | 0.532258 | 0.532258 | `{'loose': 46, 'moderate': 137, 'none': 671, 'strong': 138}` |

Representative previsible off-target rows:

| layer pair | state | target | visible top | scale | previsible bin | rank | coord-prob |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `24->28` | `desc_first-139-0-11 x1` | 645 | 495 | 0.42 | 556 | 1 | 0.012419 |
| `24->28` | `desc_first-139-0-11 x1` | 645 | 494 | 0.58 | 495 | 1 | 0.020587 |
| `24->28` | `desc_first-139-0-13 y1` | 726 | 607 | 0.24 | 608 | 1 | 0.011465 |
| `24->28` | `desc_first-139-0-13 y1` | 726 | 595 | 0.36 | 607 | 1 | 0.011099 |
| `28->24` | `desc_first-139-0-11 x1` | 645 | 463 | 0.52 | 466 | 1 | 0.015721 |
| `28->24` | `desc_first-139-0-11 x1` | 645 | 460 | 0.56 | 463 | 1 | 0.016856 |
| `28->24` | `desc_first-139-0-13 y1` | 726 | 591 | 0.44 | 595 | 1 | 0.012894 |
| `28->24` | `desc_first-139-0-13 y1` | 726 | 581 | 0.50 | 591 | 1 | 0.012430 |

Mechanistic update:

The destination-basin signal is not merely "one wrong coordinate takes over".
The dense probe-support rows show broad off-target basin activation across the
requested panel: in the paired target-history direction, over half of source
rows have at least one off-target probe at rank `<=10`, and roughly the same
fraction has a rank-competitive off-target probe that is not the visible top
coordinate. Several examples have a previsible wrong bin at rank `1` while a
different wrong bin is the visible top coordinate. This points to a distributed
coordinate-basin field: the model appears to enter a wrong spatial attractor
region with multiple nearby/related coordinate anchors competing, and the
autoregressive top token is a surface sample from that field rather than the
whole mechanism.

## Paired-Delta Component Analysis: Target-History Is Mostly Not Clean Binding

Added a guarded read-only analyzer:

```bash
python scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py \
  --stage analyze-trajectory-causal-patch-paired-delta-components \
  --trajectory-hidden-causal-patch-rows <trajectory_hidden_causal_activation_patch_rows.jsonl> \
  --output-root <paired_delta_component_analysis>
```

The stage emits:

```text
trajectory_causal_patch_paired_delta_component_rows.jsonl
trajectory_causal_patch_paired_delta_component_summary.json
trajectory_causal_patch_paired_delta_component_analysis.md
```

It is a post-hoc artifact reader only:

```text
model_perturbation_ran: false
training_ran: false
readout_only: true
```

Guardrails:

```text
selected_direction_patch_basis_key: paired_same_desc_history_target_bbox_minus_natural
requires target_next_kind=coord
requires patched_coord_readout_status=ok
requires non-null coordinate target/top/rank/radius4/distance metrics
fails fast when no selected paired target-history direction rows are present
semantic duplicate detection excludes patch_label
```

Real rerun outputs:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/destination_basin_probe_layer24_v1/paired_delta_component_analysis
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/destination_basin_probe_layer28_v1/paired_delta_component_analysis
```

Scope:

```text
source artifacts: destination_basin_probe_layer24_v1 and destination_basin_probe_layer28_v1
selected paired target-history component rows per artifact: 62
state count per artifact: 2
case count per artifact: 2
fixed 16-bin coordinate probe panel still applies to companion probe-support analysis
```

Component outcome rates:

| layer pair | component rows | binding-positive | local-basin stabilized | catastrophic anchor drift | mean d target prob | mean d target rank | mean d target r4 | mean d top1 distance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `24->28` | 62 | 0.016129 | 0.000000 | 0.516129 | -0.023938 | 97.016129 | -0.195983 | 62.193548 |
| `28->24` | 62 | 0.000000 | 0.000000 | 0.370968 | -0.016796 | 40.322581 | -0.133324 | 49.741935 |

Outcome counts:

| layer pair | binding-positive | local-basin stabilized | anchor drift | catastrophic anchor drift | neutral/mixed |
| --- | ---: | ---: | ---: | ---: | ---: |
| `24->28` | 1 | 0 | 9 | 32 | 20 |
| `28->24` | 0 | 0 | 2 | 23 | 37 |

Representative catastrophic examples:

| layer pair | case | slot | target | scale | transition | d rank | d distance | d prob |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| `24->28` | `desc_first-139-0-11-desc_end` | x1 | 645 | 0.58 | `644->494` | 264 | 150 | -0.052729 |
| `24->28` | `desc_first-139-0-13-desc_end` | y1 | 726 | 0.44 | `716->566` | 177 | 150 | -0.014132 |
| `28->24` | `desc_first-139-0-11-desc_end` | x1 | 645 | 0.56 | `644->460` | 43 | 184 | -0.046324 |
| `28->24` | `desc_first-139-0-11-desc_end` | x1 | 645 | 0.52 | `644->463` | 26 | 181 | -0.043815 |

Mechanistic update:

Within this tiny two-state, fixed-probe-panel slice, the paired
target-history hidden delta is not behaving like a clean object-binding
correction. It usually decreases target coordinate probability, worsens target
rank, reduces local target radius-4 mass, and moves the visible top coordinate
farther from the target. The rare binding-positive point in `24->28` is the
exception rather than the rule. Together with the destination-basin and
probe-support analyses above, the current picture is that target-history
patching often injects or amplifies a wrong coordinate-attractor field rather
than isolating the intended object-specific grounding component.

Next implication:

Do not add residualized paired-basis patching as a story-driven primitive yet.
The next high-value run should explicitly include both paired target-history
and paired natural-minus-sentinel directions on the same selected coord states,
then compare raw target-history, sentinel-control, and only then a residual or
projection control if those components show stable geometry.

## Matched Paired-Basis Control: Target-History Selects an Off-Target Anchor Component

Follow-up purpose:

The paired-delta component analysis above intentionally selected only
`paired_same_desc_history_target_bbox_minus_natural`. To test whether the
observed basin collapse was merely a generic coordinate-history perturbation,
I reran the same two selected coord states with both paired bases on the same
strength grid:

```text
paired_same_desc_history_target_bbox_minus_natural
paired_natural_minus_same_desc_history_sentinel
```

Model-backed artifact roots:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/paired_basis_compare_layer24_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/paired_basis_compare_layer28_v1
```

Shared inputs and scope:

```text
selected_rows.jsonl: destination_basin_probe_input/selected_rows.jsonl
case_count: 2
state_row_count: 2
target_next_kinds: coord
counterfactual_variants: natural
patch_strengths: 0.00..0.60 step 0.02
patch_alphas: 0.25,0.5,0.75,1.0
probe_coord_bins: 645,726,519,463,460,466,620,495,556,494,608,595,591,581,607,566
model_perturbation_ran: true
training_ran: false
```

Post-hoc outputs:

```text
<root>/bifurcation_analysis
<root>/destination_basin_analysis
<root>/probe_support_analysis
<root>/paired_delta_component_analysis
```

Compact matched-basis result:

| layer pair | basis | rows | catastrophic anchor drift | first catastrophic scale | A-good rate | mean d target rank | mean d top1 distance | destination-basin top1 | pre-top destination rank<=5 | any off-target rank<=10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `24->28` | target-history | 62 | 0.516 | 0.240 | 0.016 | 97.016 | 62.194 | 0.300 | 0.300 | 0.597 |
| `24->28` | sentinel-control | 62 | 0.000 | NA | 0.339 | 4.290 | 2.452 | 0.000 | 0.000 | 0.000 |
| `28->24` | target-history | 62 | 0.371 | 0.280 | 0.000 | 40.323 | 49.742 | 0.300 | 0.267 | 0.532 |
| `28->24` | sentinel-control | 62 | 0.000 | NA | 0.355 | -0.194 | 0.000 | 0.000 | 0.000 | 0.000 |

Slot and scale shape:

| layer pair | basis | slot | rows | catastrophic | A-good | mean d rank | mean d distance |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `24->28` | target-history | x1 | 31 | 0.419 | 0.032 | 91.806 | 45.065 |
| `24->28` | target-history | y1 | 31 | 0.613 | 0.000 | 102.226 | 79.323 |
| `24->28` | sentinel-control | x1 | 31 | 0.000 | 0.000 | 9.194 | 4.903 |
| `24->28` | sentinel-control | y1 | 31 | 0.000 | 0.677 | -0.613 | 0.000 |
| `28->24` | target-history | x1 | 31 | 0.194 | 0.000 | 9.290 | 33.516 |
| `28->24` | target-history | y1 | 31 | 0.548 | 0.000 | 71.355 | 65.968 |
| `28->24` | sentinel-control | x1 | 31 | 0.000 | 0.032 | 0.226 | 0.000 |
| `28->24` | sentinel-control | y1 | 31 | 0.000 | 0.677 | -0.613 | 0.000 |

| layer pair | basis | scale band | rows | catastrophic | mean d rank | mean d distance |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `24->28` | target-history | `<=0.20` | 22 | 0.000 | 5.909 | 1.273 |
| `24->28` | target-history | `0.22..0.40` | 20 | 0.600 | 80.300 | 59.150 |
| `24->28` | target-history | `>0.40` | 20 | 1.000 | 213.950 | 132.250 |
| `24->28` | sentinel-control | `<=0.20` | 22 | 0.000 | 1.318 | 1.455 |
| `24->28` | sentinel-control | `0.22..0.40` | 20 | 0.000 | 5.250 | 3.000 |
| `24->28` | sentinel-control | `>0.40` | 20 | 0.000 | 6.600 | 3.000 |
| `28->24` | target-history | `<=0.20` | 22 | 0.000 | 2.727 | 0.000 |
| `28->24` | target-history | `0.22..0.40` | 20 | 0.350 | 28.900 | 39.150 |
| `28->24` | target-history | `>0.40` | 20 | 0.800 | 93.100 | 115.050 |
| `28->24` | sentinel-control | `<=0.20` | 22 | 0.000 | 0.091 | 0.000 |
| `28->24` | sentinel-control | `0.22..0.40` | 20 | 0.000 | -0.350 | 0.000 |
| `28->24` | sentinel-control | `>0.40` | 20 | 0.000 | -0.350 | 0.000 |

Interpretation:

The sentinel-control basis is not reproducing the target-history basin
collapse in this slice. It has zero destination-basin top1, zero pre-top
destination rank<=5, zero any-off-target rank<=10 support in the strongest
probe rows, and zero catastrophic anchor drift at both layer settings. The
target-history basis, by contrast, shows scale-thresholded collapse: low
strengths are mostly harmless, mid strengths begin moving into wrong anchor
basins, and high strengths make drift common or near-deterministic.

The phrase `target-specific` should therefore be used carefully. The effect is
specific to the target-history intervention basis relative to the sentinel
control. It is not specific to the intended target coordinate. The current
evidence says that target-history hidden deltas can selectively activate an
off-target destination-basin or anchor component, not a clean target-coordinate
binding direction.

## Source-Layer Sweep and Patch-Surface Caveat

I launched an eight-way layer lattice to test whether the paired-basis effect
was localized to a particular bridge:

```text
16->24, 20->24, 24->16, 24->20, 24->28,
28->24, 16->28, 20->28, 28->16, 28->20
```

The initially attempted `*->32` / `32->*` jobs failed cleanly because this
checkpoint exposes `hidden_layer_count=29`; direct hidden-state indices above
`28` are out of range.

Valid model-backed lattice roots:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/paired_basis_lattice_s16_t24_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/paired_basis_lattice_s20_t24_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/paired_basis_lattice_s24_t16_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/paired_basis_lattice_s24_t20_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/paired_basis_lattice_s20_t28_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/paired_basis_lattice_s28_t20_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/paired_basis_lattice_s16_t28_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/paired_basis_lattice_s28_t16_v1
```

Existing comparison roots reused:

```text
paired_basis_compare_layer24_v1  # 24->28
paired_basis_compare_layer28_v1  # 28->24
```

Reduced source-layer table:

| source layer | basis | rows | catastrophic anchor drift | first catastrophic scale | A-good rate | mean d rank | mean d distance | destination-basin top1 | any off-target rank<=10 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | target-history | 62 | 0.210 | 0.360 | 0.000 | 19.952 | 9.758 | 0.133 | 0.194 |
| 16 | sentinel-control | 62 | 0.000 | NA | 0.129 | 0.855 | 0.000 | 0.000 | 0.000 |
| 20 | target-history | 62 | 0.694 | 0.180 | 0.000 | 126.694 | 62.823 | 0.433 | 0.726 |
| 20 | sentinel-control | 62 | 0.323 | 0.220 | 0.306 | 69.500 | 11.565 | 0.000 | 0.000 |
| 24 | target-history | 62 | 0.516 | 0.240 | 0.016 | 97.016 | 62.194 | 0.300 | 0.597 |
| 24 | sentinel-control | 62 | 0.000 | NA | 0.339 | 4.290 | 2.452 | 0.000 | 0.000 |
| 28 | target-history | 62 | 0.371 | 0.280 | 0.000 | 40.323 | 49.742 | 0.300 | 0.532 |
| 28 | sentinel-control | 62 | 0.000 | NA | 0.355 | -0.194 | 0.000 | 0.000 | 0.000 |

Patch-surface caveat:

For the paired hidden-delta direction bases, the implementation currently uses
`source_hidden_layer_index` for both the hidden-delta extraction and the decoder
layer hook. The `target_hidden_layer_index` affects interpolation patch specs,
but not the paired direction patch specs. This is visible in
`src/analysis/autoregressive_binding_template_ablation/realized_behavior_bridge.py`
inside `_readout_bridge_trajectory_hidden_causal_activation_patch`, where:

```text
source_hidden = hidden_states[source_resolved_index][0, -1]
peer_hidden = peer_hidden_states[peer_source_resolved_index][0, -1]
hidden_delta = peer_hidden - source_hidden  # or source_hidden - peer_hidden
decoder_layer_patch_index = _decoder_layer_index_for_hidden_state(source_layer_index, ...)
patch_hidden = source_hidden + strength * hidden_delta
```

As a result, identical outcomes for `24->16`, `24->20`, and `24->28` should not
be interpreted as evidence that target layer is irrelevant in the model. They
show that, for this specific paired-direction patch surface, the experimental
variable is effectively the source/hook layer.

Mechanistic update:

The off-target basin component is strongest at source layer `20`, substantial
at `24`, weaker but still present at `28`, and weakest at `16`. This suggests
that the harmful target-history component is not merely a final readout artifact
at the last layer. It appears to be especially accessible in the middle-late
decoder stream around layer `20`, then remains patchable downstream. The
sentinel control remains mostly null except for a source-20 rank/distance
degradation that does not create destination-basin top1 or off-target probe
support, so the source-20 control should be treated as a caution rather than
as a matched collapse.

Next implication:

Before expanding to many cases, the analyzer should formalize the
target-history-vs-sentinel contrast and explicitly group by source/hook layer.
After that, the highest-value scaled run is not a broader target-layer lattice;
it is a source-layer sweep across more cases with symmetric basis contrast,
destination-basin onset, and off-target probe support.

## Paired-Basis Contrast Analyzer

Added a guarded read-only analyzer:

```bash
python scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py \
  --stage analyze-trajectory-causal-patch-paired-basis-contrast \
  --trajectory-hidden-causal-patch-rows <trajectory_hidden_causal_activation_patch_rows.jsonl> \
  --output-root <paired_basis_contrast_analysis>
```

The stage emits:

```text
trajectory_causal_patch_paired_basis_contrast_rows.jsonl
trajectory_causal_patch_paired_basis_contrast_summary.json
trajectory_causal_patch_paired_basis_contrast_analysis.md
```

It is a post-hoc artifact reader only:

```text
model_perturbation_ran: false
training_ran: false
readout_only: true
```

Guardrails:

```text
target_basis_key: paired_same_desc_history_target_bbox_minus_natural
sentinel_basis_key: paired_natural_minus_same_desc_history_sentinel
requires target_next_kind=coord
requires patched_coord_readout_status=ok
requires non-null coordinate target/top/rank/radius4/distance metrics
matches semantic points while excluding patch_label/patch_kind/basis
fails fast when no matched target-vs-sentinel contrast rows are present
rejects duplicate semantic points per basis
```

Real outputs on the matched paired-basis roots:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/paired_basis_compare_layer24_v1/paired_basis_contrast_analysis
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/paired_basis_compare_layer28_v1/paired_basis_contrast_analysis
```

Contrast summary:

| layer pair | contrast rows | target worse than sentinel | target catastrophic | sentinel catastrophic | target-specific basin-like | mean d prob target-sentinel | mean d rank target-sentinel | mean d r4 target-sentinel | mean d distance target-sentinel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `24->28` | 62 | 0.968 | 0.516 | 0.000 | 0.661 | -0.014757 | 92.725806 | -0.117787 | 59.741935 |
| `28->24` | 62 | 0.968 | 0.371 | 0.000 | 0.403 | -0.021766 | 40.516129 | -0.158144 | 49.741935 |

Analyzer-applied source-layer sweep:

| layer pair | contrast rows | target worse than sentinel | target catastrophic | sentinel catastrophic | target-specific basin-like | mean d prob | mean d rank | mean d r4 | mean d distance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `16->24` | 62 | 0.968 | 0.210 | 0.000 | 0.210 | -0.017 | 19.097 | -0.148 | 9.758 |
| `16->28` | 62 | 0.968 | 0.210 | 0.000 | 0.210 | -0.017 | 19.097 | -0.148 | 9.758 |
| `20->24` | 62 | 0.903 | 0.694 | 0.323 | 0.435 | -0.005 | 57.194 | -0.040 | 51.258 |
| `20->28` | 62 | 0.903 | 0.694 | 0.323 | 0.435 | -0.005 | 57.194 | -0.040 | 51.258 |
| `24->16` | 62 | 0.968 | 0.516 | 0.000 | 0.661 | -0.015 | 92.726 | -0.118 | 59.742 |
| `24->20` | 62 | 0.968 | 0.516 | 0.000 | 0.661 | -0.015 | 92.726 | -0.118 | 59.742 |
| `24->28` | 62 | 0.968 | 0.516 | 0.000 | 0.661 | -0.015 | 92.726 | -0.118 | 59.742 |
| `28->16` | 62 | 0.968 | 0.371 | 0.000 | 0.403 | -0.022 | 40.516 | -0.158 | 49.742 |
| `28->20` | 62 | 0.968 | 0.371 | 0.000 | 0.403 | -0.022 | 40.516 | -0.158 | 49.742 |
| `28->24` | 62 | 0.968 | 0.371 | 0.000 | 0.403 | -0.022 | 40.516 | -0.158 | 49.742 |

Mechanistic update:

The formal contrast stage sharpens the manual read. On the two original
comparison roots, target-history is worse than sentinel on `60/62` matched
points, and the target-minus-sentinel deltas move in the wrong direction:
target probability and radius mass decrease, while target rank and top1
distance increase. The source-layer sweep separates two notions of severity:
source `20` has the highest raw target-basis catastrophic rate, but also a
non-null sentinel-control degradation; source `24` has the cleanest
target-specific basin-like contrast because sentinel remains null while the
target-history direction strongly worsens rank/distance. This makes source
`24` the best next source/hook layer for clean mechanism tracing, and source
`20` the best stress point for studying how a generic perturbation can become
coordinate-rank destructive without forming the same off-target destination
basin signature.

## Scaled Same-Desc Lock-In Paired-Basis Sweep

The two-state result above was intentionally high-leverage but too narrow. I
scaled the same paired-basis causal activation patch to the existing
same-desc-history full triad set:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_prefix_lockin_head_value_inputs/same_desc_history_full_v1/selected_rows.jsonl
```

Input scope:

```text
row_count: 174
natural source states: 58
cases: 5
counterfactual variants per state:
  natural
  same_desc_history_coords_to_target_bbox
  same_desc_history_coords_to_sentinel
target_next_kind: coord
prediction kinds: duplicate_iou70=22 natural states, repeated_gt=22, new_gt=14
onset labels: next_step_duplicate_onset=44 natural states, next_step_unmatched_onset=12, neutral=2
coord slots: x1=12, y1=20, x2=14, y2=12
```

Model-backed source-layer roots:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/paired_basis_scaled_full_s16_t28_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/paired_basis_scaled_full_s20_t28_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/paired_basis_scaled_full_s24_t28_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/paired_basis_scaled_full_s28_t24_v1
```

Execution scope per root:

```text
state_row_count: 58
case_count: 5
row_count: 2204
realized_direction_patch_count: 1856
patch_strengths: 0.00..0.60 step 0.04
probe_coord_bins: 12 unique target bins from the 58-state input
model_perturbation_ran: true
training_ran: false
```

Post-hoc outputs:

```text
<root>/bifurcation_analysis
<root>/destination_basin_analysis
<root>/probe_support_analysis
<root>/paired_delta_component_analysis
<root>/paired_basis_contrast_analysis
```

Scaled contrast summary:

| source layer | contrast rows | states | cases | target worse than sentinel | target catastrophic | sentinel catastrophic | target-specific basin-like | mean d prob | mean d rank | mean d r4 | mean d distance |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 928 | 58 | 5 | 0.707 | 0.030 | 0.000 | 0.086 | -0.002 | 2.634 | -0.022 | 1.254 |
| 20 | 928 | 58 | 5 | 0.788 | 0.059 | 0.043 | 0.197 | -0.003 | 2.776 | -0.027 | 3.933 |
| 24 | 928 | 58 | 5 | 0.594 | 0.041 | 0.000 | 0.144 | 0.028 | 6.704 | -0.020 | 4.469 |
| 28 | 928 | 58 | 5 | 0.480 | 0.023 | 0.000 | 0.115 | 0.028 | -1.083 | -0.014 | 3.866 |

Raw basis summary:

| source layer | basis | rows | catastrophic | first catastrophic scale | A-good | mean d rank | mean d distance | any off-target rank<=10 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | target-history | 928 | 0.030 | 0.360 | 0.223 | 2.584 | 1.623 | 0.017 |
| 16 | sentinel-control | 928 | 0.000 | NA | 0.133 | -0.050 | 0.369 | 0.017 |
| 20 | target-history | 928 | 0.059 | 0.200 | 0.308 | 10.216 | 4.812 | 0.017 |
| 20 | sentinel-control | 928 | 0.043 | 0.240 | 0.263 | 7.440 | 0.879 | 0.017 |
| 24 | target-history | 928 | 0.041 | 0.240 | 0.481 | 5.394 | 4.197 | 0.017 |
| 24 | sentinel-control | 928 | 0.000 | NA | 0.207 | -1.309 | -0.272 | 0.017 |
| 28 | target-history | 928 | 0.023 | 0.280 | 0.463 | -1.044 | 3.394 | 0.034 |
| 28 | sentinel-control | 928 | 0.000 | NA | 0.203 | 0.039 | -0.472 | 0.017 |

High-level scaled finding:

The extreme two-state basin collapse does not dominate the broader
same-desc-history set. At 58-state scope, target-history is often worse than
sentinel, but the catastrophic and basin-like rates are much smaller. This is
not a refutation of the two-state mechanism; it means the mechanism is
state-conditional. The target-history vector sometimes carries harmful
wrong-anchor pressure, sometimes acts like a useful local binding direction
(`A-good` reaches `0.481` for source `24`), and sometimes is mostly neutral.

The harm is concentrated in `new_gt` and unmatched/neutral states rather than
uniformly in all duplicate/repeated states:

| source | subgroup | n | target-specific basin-like | target worse | mean d rank | mean d distance |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 20 | `prediction_kind=new_gt` | 224 | 0.487 | 0.862 | 6.902 | 12.946 |
| 20 | `onset=next_step_unmatched_onset` | 192 | 0.510 | 0.917 | -2.073 | 8.115 |
| 24 | `prediction_kind=new_gt` | 224 | 0.330 | 0.469 | 30.701 | 16.228 |
| 24 | `onset=next_step_unmatched_onset` | 192 | 0.333 | 0.469 | 27.250 | 12.073 |
| 28 | `prediction_kind=new_gt` | 224 | 0.192 | 0.531 | 7.174 | 13.795 |

Top high-harm states:

| source | case | slot | target | prediction | onset | basin-like | first basin scale | representative target-basis top bins |
| ---: | --- | --- | ---: | --- | --- | ---: | ---: | --- |
| 24 | `desc_first-139-0-11-desc_end` | x1 | 645 | new_gt | unmatched | 0.688 | 0.20 | 628, 556, 620 |
| 24 | `desc_first-139-0-13-desc_end` | y1 | 726 | new_gt | neutral | 0.625 | 0.24 | 566, 608, 595 |
| 24 | `desc_first-139-0-11-desc_end` | x2 | 693 | new_gt | unmatched | 0.625 | 0.16 | 685, 689, 691 |
| 24 | `desc_first-885-8-1-desc_end` | x2 | 94 | repeated_gt | duplicate onset | 0.500 | 0.08 | 33, 47 |
| 20 | `desc_first-139-0-11-desc_end` | x2 | 693 | new_gt | unmatched | 0.750 | 0.12 | 685, 689, 691 |
| 20 | `desc_first-139-0-13-desc_end` | y1 | 726 | new_gt | neutral | 0.688 | 0.20 | 566, 608, 613 |

Mechanistic update:

The scaled run changes the claim from "target-history deltas are generally bad"
to "target-history deltas expose a conditional branch point." In many states,
the same-desc-history vector improves local target readout; in a smaller but
mechanistically rich subset, it pushes the model toward already-plausible wrong
anchors. This fits a basin-selection picture better than a simple missing
perception or generic coordinate-history field picture.

## Adaptive Wrong-Anchor Probe

Because the scaled run used only the 12 target bins as the probe panel, it
could not fully test whether the newly observed wrong top-1 anchors were
rank-visible before they became top-1. I therefore built an adaptive top-state
input from the high-harm signatures above:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/paired_basis_adaptive_basin_topstates_input/selected_rows.jsonl
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/paired_basis_adaptive_basin_topstates_input/manifest.json
```

Adaptive input scope:

```text
selected signatures: 5
source_state_count: 19
row_count: 57
case_count in model-backed run: 3
required variants preserved: natural, same_desc_history_coords_to_target_bbox, same_desc_history_coords_to_sentinel
probe_coord_bins: 34 target plus observed wrong-anchor bins
```

Model-backed roots:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/paired_basis_adaptive_basin_topstates_s20_t28_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/paired_basis_adaptive_basin_topstates_s24_t28_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/paired_basis_adaptive_basin_topstates_s28_t24_v1
```

Execution scope per root:

```text
state_row_count: 19
case_count: 3
row_count: 722
realized_direction_patch_count: 608
patch_strengths: 0.00..0.60 step 0.04
probe_coord_bin_count: 34
model_perturbation_ran: true
training_ran: false
```

Adaptive contrast:

| source | contrast rows | target worse | target catastrophic | sentinel catastrophic | target-specific basin-like | mean d prob | mean d rank | mean d r4 | mean d distance |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 304 | 0.924 | 0.181 | 0.132 | 0.549 | -0.006 | 5.651 | -0.045 | 12.105 |
| 24 | 304 | 0.641 | 0.125 | 0.000 | 0.401 | 0.004 | 23.339 | -0.042 | 14.168 |
| 28 | 304 | 0.668 | 0.069 | 0.000 | 0.286 | 0.002 | 6.270 | -0.063 | 12.191 |

Adaptive destination/probe support:

| source | basis | destination rows | exact destination top1 | basin top1 | pre-top rank<=5 | mean first rank<=5 scale | mean first top1 scale | any off-target rank<=10 | previsible off-target rank<=10 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | target-history | 627 | 0.093 | 0.128 | 0.085 | 0.177 | 0.168 | 1.000 | 0.961 |
| 20 | sentinel-control | 627 | 0.065 | 0.094 | 0.048 | 0.066 | 0.060 | 0.908 | 0.776 |
| 24 | target-history | 627 | 0.073 | 0.088 | 0.070 | 0.112 | 0.161 | 0.974 | 0.944 |
| 24 | sentinel-control | 627 | 0.053 | 0.062 | 0.041 | 0.036 | 0.056 | 1.000 | 0.941 |
| 28 | target-history | 627 | 0.069 | 0.081 | 0.051 | 0.155 | 0.196 | 1.000 | 0.984 |
| 28 | sentinel-control | 627 | 0.046 | 0.049 | 0.035 | 0.012 | 0.052 | 1.000 | 0.941 |

Representative previsible wrong anchors:

| source | case | slot | target | destination | distance | first rank<=5 | pre-top rank<=5 | first top1 | best pre-top rank |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | `desc_first-139-0-11-desc_end` | x2 | 693 | 689 | 4 | 0.00 | 0.00 | 0.16 | 2 |
| 20 | `desc_first-139-0-11-desc_end` | y1 | 524 | 519 | 5 | 0.00 | 0.00 | 0.12 | 2 |
| 20 | `desc_first-139-0-11-desc_end` | y1 | 524 | 520 | 4 | 0.00 | 0.00 | 0.16 | 1 |
| 20 | `desc_first-885-8-1-desc_end` | x2 | 94 | 47 | 47 | 0.00 | 0.00 | NA | 1 |
| 24 | `desc_first-139-0-11-desc_end` | x2 | 693 | 689 | 4 | 0.00 | 0.00 | 0.16 | 1 |
| 24 | `desc_first-139-0-11-desc_end` | y1 | 524 | 520 | 4 | 0.00 | 0.00 | 0.04 | 1 |
| 28 | `desc_first-139-0-11-desc_end` | x2 | 693 | 689 | 4 | 0.00 | 0.00 | 0.44 | 2 |
| 28 | `desc_first-139-0-11-desc_end` | y1 | 524 | 520 | 4 | 0.00 | 0.00 | 0.08 | 1 |

Mechanistic update:

The wrong anchors in the adaptive slice are often already rank-visible at
baseline scale `0.00`; many have rank `<=5` before they become the visible top
coordinate. This means the harmful target-history vector is usually not
inventing a new coordinate from scratch. It is shifting selection inside an
existing coordinate-basin field. The sentinel direction also sees much of the
previsible off-target support once those bins are included, so the decisive
target-vs-sentinel difference is not "wrong anchors exist only under
target-history." The difference is that target-history more often turns those
available anchors into top/basin choices and worsens target-local rank/distance.

This is currently the deepest mechanistic picture from this branch:

```text
visual/coordinate readout contains multiple plausible anchor basins ->
same-desc target-history delta perturbs the selection field ->
in some states the perturbation improves target-local binding ->
in vulnerable states it selects a pre-existing wrong anchor basin ->
autoregressive emission then exposes that selected basin as visible coordinate drift or duplicate-like lock-in
```

Next implication:

The next high-value implementation should move from "does a harmful hidden
delta exist?" to "which upstream signal chooses among pre-existing coordinate
basins?" Concretely, target the high-harm adaptive states with attention/value
or logit-lens decomposition over image regions, current object text, previous
same-description objects, and coordinate-token embeddings. The evidence now
suggests a basin-selection mechanism more than a pure visual non-perception
mechanism.
