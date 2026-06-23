---
doc_id: docs.implementation-map
layer: docs
doc_type: implementation-map
status: canonical
domain: repo
summary: Task-to-file routing guide for common CoordExp changes.
updated: 2026-05-16
---

# Implementation Map

Purpose: route common research and engineering changes to the smallest useful set of files, configs, docs, and tests.
Authority: code-navigation guide for the current repo; for current defaults, defer to `docs/PROJECT_CONTEXT.md` and runbooks; for stable contract semantics, defer to `openspec/specs/`.
Read this after: `docs/SYSTEM_OVERVIEW.md`
Read this before: opening many source files blindly or doing broad repo-wide searches
Primary code handles: `src/sft.py`, `src/training/`, `src/detection/runtime.py`, `src/detection/template.py`, `src/common/detection_sequence.py`, `src/common/detection_compact_rows.py`, `src/bootstrap/`, `src/config/schema.py`, `src/datasets/`, `src/trainers/metrics/`, `src/metrics/events.py`, `src/trainers/stage2_rollout_correction.py`, `src/trainers/rollout_aligned_targets.py`, `src/trainers/rollout_aligned_evaluator.py`, `src/launchers/stage2_vllm_server.py`, `src/infer/pipeline.py`, `src/infer/runtime.py`, `src/infer/backend.py`, `src/infer/backend_sync.py`, `src/infer/backend_vllm_server.py`, `src/infer/constraints.py`, `src/infer/artifacts.py`, `src/eval/detection.py`, `src/eval/detection_orchestrator.py`, `src/eval/detection_records.py`, `src/eval/detection_geometry.py`, `src/eval/detection_coco.py`, `src/eval/detection_lvis.py`, `src/eval/detection_duplicate_guard.py`, `src/eval/detection_f1ish.py`, `src/eval/orchestration.py`, `src/eval/artifacts.py`
Verification: use the targeted test files listed below before running broader suites

## 1. Data Contract, JSONL Rendering, Or Geometry

Open these docs first:
- [`docs/data/README.md`](data/README.md)
- [`docs/data/CONTRACT.md`](data/CONTRACT.md)
- [`docs/data/PREPARATION.md`](data/PREPARATION.md)
- [`docs/data/PACKING.md`](data/PACKING.md)

Open these code files first:
- `src/datasets/dense_caption.py`
- `src/datasets/builders/jsonlines.py`
- `src/datasets/geometry.py`
- `src/detection/template.py`
- `src/common/detection_sequence.py`
- `src/common/detection_compact_rows.py`
- `src/config/schema.py`

Run these tests first:
- `tests/test_common_io_jsonl.py`
- `tests/test_dataset_runtime_contracts.py`
- `tests/test_coord_geometry_invariants.py`
- `tests/test_chat_template_regression.py`
- `tests/test_prompt_variants.py`

## 2. Stage-1 Baseline SFT, Compact Detection, Or Coord-Token Losses

Open these docs first:
- [`docs/training/README.md`](training/README.md)
- [`docs/training/STAGE1_OBJECTIVE.md`](training/STAGE1_OBJECTIVE.md)
- [`docs/data/PACKING.md`](data/PACKING.md)
- [`configs/stage1/detection_teacher_forcing/`](../configs/stage1/detection_teacher_forcing/) for canonical compact Stage-1 research teacher-forcing configs using `pipeline.id: stage1_research_teacher_forcing`
- [`configs/archive/detection_scene_clean_break/stage1/recursive_detection_ce/prod/compact_full_support2.yaml`](../configs/archive/detection_scene_clean_break/stage1/recursive_detection_ce/prod/compact_full_support2.yaml) only as a quarantined historical recursive-detection CE handle, not the current public teacher-forcing route
- [`configs/archive/detection_scene_clean_break/stage1/recursive_detection_ce/ablation/compact_full_prefix_rollin_balance2.yaml`](../configs/archive/detection_scene_clean_break/stage1/recursive_detection_ce/ablation/compact_full_prefix_rollin_balance2.yaml) only as a quarantined historical compact-full prefix-rollin E1 ablation handle

Open these configs first:
- `configs/stage1/sft_base.yaml`
- `configs/_shared/datasets/`
- `configs/_shared/prompts/`
- `configs/stage1/profiles/`
- `configs/stage1/smoke/`
- `configs/stage1/detection_teacher_forcing/`
- `configs/archive/detection_scene_clean_break/stage1/recursive_detection_ce/prod/` only for quarantined historical migration references
- `configs/archive/detection_scene_clean_break/stage1/recursive_detection_ce/ablation/` only for quarantined historical ablation references

Open these code files first:
- `src/training/pipeline_registry.py`
- `src/training/pipelines/stage1_json_ce.py`
- `src/training/pipelines/stage1_compact_trie_ce.py`
- `src/training/objectives/`
- `src/training/supervision/`
- `src/training/templates/compact_full.py`
- `src/sft.py`
- `src/detection/runtime.py`
- `src/detection/template.py`
- `src/common/detection_sequence.py`
- `src/common/detection_compact_rows.py`
- `src/metrics/events.py`
- `src/metrics/dataset_metrics.py`
- `src/trainers/losses/coord_soft_ce_w1.py`
- `src/trainers/metrics/mixins.py`
- `src/trainers/metrics/batch_contract.py`
- `src/trainers/metrics/structural_close.py`
- `src/trainers/metrics/recursive_detection.py`
- `src/trainers/metrics/aggregate_tokens.py`
- `src/trainers/metrics/coord_losses.py`
- `src/data_collators/batch_extras_collator.py`

Stage-1 detection teacher-forcing ownership:
- `src/training/pipeline_registry.py` owns the guarded pipeline registry and
  supported public `pipeline.id` values: `stage1_standard_sft`,
  `stage1_research_teacher_forcing`, and `stage2_rollout_correction`.
- Shadow objective profiles resolve through text/trie teacher-forcing modules;
  geometry regularizers are not part of the active Stage-2 objective surface.
- `src/detection/runtime.py` owns detection runtime support/preflight,
  teacher-forcing runtime policy, prompt/mode/custom shim resolution, and
  `build_detection_training_dataset`.
- `src/detection/objective.py`, `src/detection/rollin.py`, `src/detection/dataset.py`, `src/detection/token_types.py`, and `src/detection/loss.py` own the `prefix_rollin_et_rmp_ce` roll-in state, objectized sparse targets, compact type gates, ordinary teacher-forced `<|im_end|>` CE, and loss-sidecar behavior.
- `src/sft.py` delegates policy and keeps backward-compatible private aliases.
- `src/detection/template.py` owns strict templates; factory-visible strict IDs
  are `stage1_json_pretty`, `compact`, `compact_box_closed`,
  `compact_object_box_closed`, and `compact_object_box_closed_lines`.
- `src/common/detection_sequence.py` is the compatibility facade; malformed helper-format rows return `None`.
- `src/common/detection_compact_rows.py` is the stdlib-only low-level marker/render/split helper.
- `compact_no_desc`, `compact_no_bbox`, and `compact_min` stay compatibility/helper formats.
- `pipeline.id: stage1_research_teacher_forcing` keeps packing/cache fail-fast policy and
  remains config-first under `configs/stage1/detection_teacher_forcing/`.
  Quarantined recursive-detection CE configs under
  `configs/archive/detection_scene_clean_break/stage1/` remain only as archive history,
  ablation anchors, or explicit comparator handles.

Trainer metric ownership:
- `src/trainers/metrics/mixins.py` is a compatibility re-export facade.
- source-level edits should target `batch_contract.py`, `structural_close.py`, `recursive_detection.py`, `aggregate_tokens.py`, `coord_losses.py`, or `bbox_losses.py`.
- metric event flattening and aliasing live in `src/metrics/events.py`.

Run these tests first:
- `tests/test_coord_softce_w1_loss.py`
- `tests/test_coord_soft_ce_w1_collective_guard.py`
- `tests/test_stage1_metric_key_parity.py`
- `tests/test_stage1_registry_masks.py`
- `tests/test_stage1_static_packing_runtime_config.py`
- `tests/test_prefix_rollin_schema.py`
- `tests/test_prefix_rollin_sampler.py`
- `tests/test_prefix_rollin_dataset_alignment.py`
- `tests/test_compact_type_gate.py`
- `tests/test_recursive_detection_ce_loss_adapter.py` for legacy/comparator recursive_detection_ce loss-adapter checks
- `tests/test_recursive_detection_ce_sft_wiring.py` for legacy/comparator recursive_detection_ce wiring checks
- `tests/test_training_pipeline_registry.py`
- `tests/test_objective_profile_resolution.py`
- `tests/test_compact_full_encoding_contract.py`
- `tests/test_compact_span_projector.py`
- `tests/test_training_architecture_golden_thread.py`

## 3. Stage-2 Rollout-Correction Training, Matching, Triage, Or Duplicate Control Diagnostics

Open these docs first:
- [`docs/training/STAGE2_RUNBOOK.md`](training/STAGE2_RUNBOOK.md)
- [`docs/training/METRICS.md`](training/METRICS.md)
- [`openspec/specs/stage2-rollout-correction/spec.md`](../openspec/specs/stage2-rollout-correction/spec.md)
- [`openspec/specs/rollout-matching-sft/spec.md`](../openspec/specs/rollout-matching-sft/spec.md) only for retired-contract rejection checks
- [`openspec/specs/teacher-forcing-unified-loss-registry/spec.md`](../openspec/specs/teacher-forcing-unified-loss-registry/spec.md)
- [`openspec/specs/runtime-architecture-refactor-program/spec.md`](../openspec/specs/runtime-architecture-refactor-program/spec.md)

Historical context only:
- Use `progress/` notes only for historical evidence after checking the current docs above.

Open these configs first:
- `configs/stage2/rollout_correction/base.yaml`
- `configs/_shared/datasets/`
- `configs/_shared/prompts/`
- `configs/stage2/rollout_correction/prod/`
- `configs/stage2/rollout_correction/smoke/`

Key v3 config handles:
- `stage2_rollout_correction.correction.triage_posterior.*`
- `rollout_matching.decoding.*`

Open these code files first:
- `src/training/pipeline_registry.py`
- `src/trainers/stage2_rollout_correction.py`
- `src/training/stage2/assignment.py`
- `src/training/stage2/duplicate_filter.py`
- `src/training/stage2/planners.py`
- `src/training/ordering.py`
- `src/sft.py`
- `src/bootstrap/pipeline_manifest.py`
- `src/bootstrap/trainer_setup.py`
- `src/trainers/stage2_coordination.py`
- `src/trainers/stage2_rollout_correction.py`
- `src/trainers/stage2_rollout_runtime.py` (trainer-owned facade; do not add
  shared prompt, decode, backend, parser, trace, or artifact policy here)
- `src/trainers/rollout_aligned_targets.py`
- `src/trainers/rollout_aligned_evaluator.py`
- `src/launchers/stage2_vllm_server.py`
- `src/infer/backend_vllm_server.py`
- `src/infer/backend_sync.py`
- `src/infer/rollout_dispatch.py`
- `src/trainers/rollout_matching/parsing.py`
- `src/trainers/rollout_matching/matching.py`
- `src/trainers/teacher_forcing/module_registry.py`
- `src/trainers/teacher_forcing/objective_atoms.py`
- `src/trainers/teacher_forcing/modules/token_ce.py`
- `src/trainers/teacher_forcing/modules/stage2_trie_ce.py`

Stage-2 planning direction:
- `src/training/stage2/duplicate_filter.py` filters accepted rollout objects
  before assignment and target realization.
- `src/training/stage2/assignment.py::GreedyIoUAssignment` is the target
  assignment strategy for new shadow planning.
- `src/training/stage2/planners.py::Stage2GreedyIoUShadowPlanner` derives
  false-negative GT insertions after duplicate filtering.
- `src/trainers/rollout_matching/matching.py::greedy_match_iou` is the
  shared rollout matching helper used by Stage-2 runtime code.

Run these tests first:
- `tests/test_stage2_rollout_correction_contract.py`
- `tests/test_stage2_rollout_runtime.py`
- `tests/test_stage2_objective_atoms_projection.py`
- `tests/test_teacher_forcing_token_ce.py`
- `tests/test_stage2_pending_metrics_aggregation.py`
- `tests/test_stage2_rollout_import_boundaries.py`
- `tests/test_training_config_strict_unknown_keys.py`
- `tests/test_stage2_assignment_greedy_iou.py`
- `tests/test_stage2_duplicate_filter.py`
- `tests/test_stage2_supervision_planning_smoke.py`

## 4. Inference, Confidence, And Offline Evaluation

Open these docs first:
- [`docs/eval/README.md`](eval/README.md)
- [`docs/eval/CONTRACT.md`](eval/CONTRACT.md)
- [`docs/eval/WORKFLOW.md`](eval/WORKFLOW.md)
- [`docs/ARTIFACTS.md`](ARTIFACTS.md), which owns the full artifact inventory
- [`openspec/specs/inference-pipeline/spec.md`](../openspec/specs/inference-pipeline/spec.md)
- [`openspec/specs/inference-engine/spec.md`](../openspec/specs/inference-engine/spec.md)
- [`openspec/specs/detection-evaluator/spec.md`](../openspec/specs/detection-evaluator/spec.md)
- [`openspec/specs/runtime-architecture-refactor-program/spec.md`](../openspec/specs/runtime-architecture-refactor-program/spec.md)

Open these configs first:
- `configs/infer/pipeline.yaml`
- `configs/eval/detection.yaml`
- `configs/postop/confidence.yaml`
- `configs/bench/`

Open these code files first:
- `scripts/run_infer.py`
- `src/infer/pipeline.py`
- `src/infer/runtime.py`
- `src/infer/backend.py`
- `src/infer/constraints.py`
- `src/infer/artifacts.py`
- `scripts/postop_confidence.py`
- `scripts/evaluate_detection.py`
- `src/eval/detection.py`
- `src/eval/detection_records.py`
- `src/eval/detection_geometry.py`
- `src/eval/detection_coco.py`
- `src/eval/detection_lvis.py`
- `src/eval/detection_duplicate_guard.py`
- `src/eval/detection_f1ish.py`
- `src/eval/detection_orchestrator.py`
- `src/eval/orchestration.py`
- `src/eval/artifacts.py`

Runtime ownership notes:

- `src/infer/runtime.py` is the shared inference seam for offline inference and
  live-model eval callers. It owns the offline inference owner lifecycle and
  artifact loop; the legacy engine module is not an active surface.
- `src/infer/backend.py` owns decode backend projection and strict trace
  normalization. Trace-bearing HF and vLLM results must normalize into the same
  generated-token IDs, token text, and logprob fields before they can feed
  metric-bearing artifacts.
- OpenAI-compatible vLLM server inference may use
  `infer.generation.trace_logprobs: true` only when the server returns generated
  token IDs and generated-token logprobs; otherwise the shared backend fails
  before emitting comparable artifacts.

Detection eval ownership:
- `src/eval/detection.py` is the import-compatible facade.
- durable source edits should target the decomposed modules listed above.
- `SemanticDescEncoder` facade patch/import compatibility is preserved for existing imports.
- `src/callbacks/detection_eval.py`

Run these tests first:
- `tests/test_unified_infer_pipeline.py`
- `tests/test_detection_eval_output_parity.py`
- `tests/test_detection_eval_ingestion_diagnostics.py`
- `tests/test_confidence_postop.py`
- `tests/test_bbox_confidence.py`

## 5. Logging, Provenance, And Run Manifests

Open these docs first:
- [`docs/ARTIFACTS.md`](ARTIFACTS.md)
- [`docs/training/METRICS.md`](training/METRICS.md)

Freeze/gate reference:
- `docs/ARTIFACTS.md#artifactprovenance-freeze` records the current rank-0
  artifact owner/name map and the Stage-2 eval diagnostic compatibility table.
  Preserve those names or add an explicit migration with tests.

Open these code files first:
- `src/sft.py`
- `src/training/observability/events.py`
- `src/training/observability/service.py`
- `src/training/observability/legacy.py`
- `src/training/observability/contracts.py`
- `src/bootstrap/experiment_manifest.py`
- `src/bootstrap/pipeline_manifest.py`
- `src/bootstrap/run_metadata.py`
- `src/bootstrap/trainer_setup.py`
- `src/metrics/events.py`
- `src/metrics/reporter.py`
- `src/metrics/payload_contract.py`
- `src/trainers/metrics/batch_contract.py`
- `src/trainers/metrics/aggregate_tokens.py`
- `src/trainers/metrics/recursive_detection.py`
- `src/callbacks/`

Metric event contract:
- `src/metrics/events.py` defines `MetricEvent` and `flatten_metric_events`.
- `src/training/observability/events.py` defines `DiagnosticEvent` and the
  bounded diagnostic profiles `off`, `standard`, and `debug`.
- New writers should use clean current `MetricEvent` / `DiagnosticEvent`
  surfaces. Legacy flat metric records are read through tolerant adapters only.
- aggregate coord token metrics publish canonical identities `coord_token_acc/full_vocab/top1` and `coord_token_acc/full_vocab/top5`.
- legacy aliases remain `coord_token_acc` and `coord_token_acc_top5`.
- the reserved-alias collision guard prevents new canonical identities from colliding with reserved legacy flat keys.

Run these tests first:
- `tests/test_experiment_manifest_file.py`
- `tests/test_run_manifest_files.py`
- `tests/test_run_metadata_file.py`
- `tests/test_dependency_provenance.py`
- `tests/test_launcher_metadata_env.py`
- `tests/test_trainer_metrics_payload_contract.py`
- `tests/test_observability_events.py`
- `tests/test_diagnostic_sampling_policy.py`
- `tests/test_artifact_contract_docs.py`

## 6. When To Update Docs And Specs Too

Update docs when you change:
- user-facing config defaults,
- artifact names or log keys,
- current entrypoints,
- recommended run or smoke workflows.

Update OpenSpec when you change:
- stable training or evaluation behavior,
- config contracts,
- loss/pipeline semantics,
- normative metrics semantics.

Practical rule:
- if the change affects `openspec/specs/`, also check `docs/PROJECT_CONTEXT.md`, the relevant runbook, and this page.
