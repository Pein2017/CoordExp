---
doc_id: docs.implementation-map
layer: docs
doc_type: implementation-map
status: canonical
domain: repo
summary: Task-to-file routing guide for common CoordExp changes.
updated: 2026-04-25
---

# Implementation Map

Purpose: route common research and engineering changes to the smallest useful set of files, configs, docs, and tests.
Authority: code-navigation guide for the current repo; for semantics and defaults, defer to `docs/PROJECT_CONTEXT.md`, runbooks, and `openspec/specs/`.
Read this after: `docs/SYSTEM_OVERVIEW.md`
Read this before: opening many source files blindly or doing broad repo-wide searches
Primary code handles: `src/sft.py`, `src/bootstrap/`, `src/config/schema.py`, `src/datasets/`, `src/trainers/stage1_set_continuation/`, `src/trainers/stage2_two_channel.py`, `src/trainers/stage2_two_channel/`, `src/trainers/stage2_rollout_aligned.py`, `src/trainers/rollout_aligned_targets.py`, `src/trainers/rollout_aligned_evaluator.py`, `src/trainers/rollout_runtime/`, `src/launchers/stage2_vllm_server.py`, `src/infer/pipeline.py`, `src/infer/engine.py`, `src/infer/backends.py`, `src/infer/artifacts.py`, `src/eval/detection.py`, `src/eval/orchestration.py`, `src/eval/artifacts.py`
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
- `src/config/schema.py`

Run these tests first:
- `tests/test_common_io_jsonl.py`
- `tests/test_dataset_runtime_contracts.py`
- `tests/test_coord_geometry_invariants.py`
- `tests/test_chat_template_regression.py`
- `tests/test_prompt_variants.py`

## 2. Stage-1 Baseline SFT, Set-Continuation, Or Coord-Token Losses

Open these docs first:
- [`docs/training/README.md`](training/README.md)
- [`docs/training/STAGE1_OBJECTIVE.md`](training/STAGE1_OBJECTIVE.md)
- [`docs/data/PACKING.md`](data/PACKING.md)

Open these configs first:
- `configs/stage1/sft_base.yaml`
- `configs/_shared/datasets/`
- `configs/_shared/prompts/`
- `configs/stage1/profiles/`
- `configs/stage1/set_continuation/`
- `configs/stage1/smoke/`

Open these code files first:
- `src/sft.py`
- `src/metrics/dataset_metrics.py`
- `src/trainers/losses/coord_soft_ce_w1.py`
- `src/trainers/metrics/mixins.py`
- `src/trainers/stage1_set_continuation/`
- `src/data_collators/batch_extras_collator.py`
- `src/data_collators/stage1_set_continuation_collator.py`

Run these tests first:
- `tests/test_stage1_set_continuation_config.py`
- `tests/test_stage1_set_continuation_cache_policy.py`
- `tests/test_stage1_set_continuation_serialization.py`
- `tests/test_stage1_set_continuation_sampler.py`
- `tests/test_stage1_set_continuation_loss.py`
- `tests/test_stage1_set_continuation_collator.py`
- `tests/test_stage1_set_continuation_trainer_smoke.py`
- `tests/test_stage1_set_continuation_benchmark_profiles.py`
- `tests/test_coord_softce_w1_loss.py`
- `tests/test_coord_soft_ce_w1_collective_guard.py`
- `tests/test_stage1_metric_key_parity.py`
- `tests/test_stage1_registry_masks.py`
- `tests/test_stage1_static_packing_runtime_config.py`

## 3. Stage-2 Two-Channel Or Rollout-Aligned Training, Matching, Triage, Or Duplicate UL

Open these docs first:
- [`docs/training/STAGE2_RUNBOOK.md`](training/STAGE2_RUNBOOK.md)
- [`docs/training/METRICS.md`](training/METRICS.md)
- [`openspec/specs/stage2-ab-training/spec.md`](../openspec/specs/stage2-ab-training/spec.md)
- [`openspec/specs/rollout-matching-sft/spec.md`](../openspec/specs/rollout-matching-sft/spec.md)
- [`openspec/specs/teacher-forcing-unified-loss-registry/spec.md`](../openspec/specs/teacher-forcing-unified-loss-registry/spec.md)
- [`openspec/specs/runtime-architecture-refactor-program/spec.md`](../openspec/specs/runtime-architecture-refactor-program/spec.md)

Historical context only:
- [`docs/training/STAGE2_DESIGN.md`](training/STAGE2_DESIGN.md)

Open these configs first:
- `configs/stage2_two_channel/base.yaml`
- `configs/_shared/datasets/`
- `configs/_shared/prompts/`
- `configs/stage2_two_channel/prod/`
- `configs/stage2_two_channel/smoke/`

For `custom.trainer_variant: stage2_rollout_aligned`:
- author `rollout_matching.pipeline.objective[]` and `rollout_matching.pipeline.diagnostics[]`
- do not author `stage2_ab.pipeline.*`

Key v3 config handles:
- `stage2_ab.channel_b.triage_posterior.*`
- `rollout_matching.pipeline.*`
- `rollout_matching.decoding.*`

Open these code files first:
- `src/sft.py`
- `src/bootstrap/pipeline_manifest.py`
- `src/bootstrap/trainer_setup.py`
- `src/trainers/stage2_coordination.py`
- `src/trainers/stage2_two_channel.py`
- `src/trainers/stage2_two_channel/scheduler.py`
- `src/trainers/stage2_two_channel/target_builder.py`
- `src/trainers/stage2_two_channel/objective_runner.py`
- `src/trainers/stage2_two_channel/types.py`
- `src/trainers/stage2_two_channel/rollout_views.py`
- `src/trainers/stage2_two_channel/coordination.py`
- `src/trainers/stage2_two_channel/executors.py`
- `src/trainers/stage2_ab/`
- `src/trainers/stage2_rollout_aligned.py`
- `src/trainers/rollout_aligned_targets.py`
- `src/trainers/rollout_aligned_evaluator.py`
- `src/trainers/rollout_matching_sft.py`
- `src/trainers/rollout_runtime/`
- `src/launchers/stage2_vllm_server.py`
- `src/trainers/rollout_matching/parsing.py`
- `src/trainers/rollout_matching/matching.py`
- `src/trainers/teacher_forcing/module_registry.py`
- `src/trainers/teacher_forcing/objective_atoms.py`
- `src/trainers/teacher_forcing/modules/token_ce.py`
- `src/trainers/teacher_forcing/modules/loss_duplicate_burst_unlikelihood.py`
- `src/trainers/teacher_forcing/modules/bbox_geo.py`
- `src/trainers/teacher_forcing/modules/coord_reg.py`

Run these tests first:
- `tests/test_stage2_ab_config_contract.py`
- `tests/test_stage2_ab_training.py`
- `tests/test_stage2_rollout_aligned.py`
- `tests/test_stage2_two_channel_training.py`
- `tests/test_stage2_objective_atoms_projection.py`
- `tests/test_teacher_forcing_token_ce.py`
- `tests/test_stage2_pending_metrics_aggregation.py`
- `tests/test_stage2_rollout_import_boundaries.py`
- `tests/test_training_config_strict_unknown_keys.py`
- `tests/test_stage2_ab_vllm_server_mode_smoke.py`
- `tests/test_stage2_ab_ddp_phase_monitor_disable.py`
- `tests/test_stage2_ab_disable_average_tokens_across_devices.py`

## 4. Inference, Confidence, And Offline Evaluation

Open these docs first:
- [`docs/eval/README.md`](eval/README.md)
- [`docs/eval/CONTRACT.md`](eval/CONTRACT.md)
- [`docs/eval/WORKFLOW.md`](eval/WORKFLOW.md)
- [`docs/ARTIFACTS.md`](ARTIFACTS.md)
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
- `src/infer/engine.py`
- `src/infer/backends.py`
- `src/infer/artifacts.py`
- `scripts/postop_confidence.py`
- `scripts/evaluate_detection.py`
- `src/eval/detection.py`
- `src/eval/orchestration.py`
- `src/eval/artifacts.py`
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

Open these code files first:
- `src/sft.py`
- `src/bootstrap/experiment_manifest.py`
- `src/bootstrap/pipeline_manifest.py`
- `src/bootstrap/run_metadata.py`
- `src/bootstrap/trainer_setup.py`
- `src/metrics/reporter.py`
- `src/metrics/payload_contract.py`
- `src/callbacks/`

Run these tests first:
- `tests/test_experiment_manifest_file.py`
- `tests/test_run_manifest_files.py`
- `tests/test_run_metadata_file.py`
- `tests/test_dependency_provenance.py`
- `tests/test_launcher_metadata_env.py`
- `tests/test_trainer_metrics_payload_contract.py`

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
