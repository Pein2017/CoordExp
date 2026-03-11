---
doc_id: docs.implementation-map
layer: docs
doc_type: implementation-map
status: canonical
domain: repo
summary: Task-to-file routing guide for common CoordExp changes.
updated: 2026-03-09
---

# Implementation Map

Purpose: route common research and engineering changes to the smallest useful set of files, configs, docs, and tests.
Authority: code-navigation guide for the current repo; for semantics and defaults, defer to `docs/PROJECT_CONTEXT.md`, runbooks, and `openspec/specs/`.
Read this after: `docs/SYSTEM_OVERVIEW.md`
Read this before: opening many source files blindly or doing broad repo-wide searches
Primary code handles: `src/sft.py`, `src/config/schema.py`, `src/datasets/`, `src/trainers/stage2_two_channel.py`, `src/infer/pipeline.py`, `src/eval/detection.py`
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

## 2. Stage-1 Baseline SFT Or Coord-Token Losses

Open these docs first:
- [`docs/training/README.md`](training/README.md)
- [`docs/training/STAGE1_OBJECTIVE.md`](training/STAGE1_OBJECTIVE.md)
- [`docs/data/PACKING.md`](data/PACKING.md)

Open these configs first:
- `configs/stage1/sft_base.yaml`
- `configs/stage1/ablation/`
- `configs/stage1/smoke/`

Historical lineage only:
- `configs/dlora/`

Open these code files first:
- `src/sft.py`
- `src/metrics/dataset_metrics.py`
- `src/trainers/losses/coord_soft_ce_w1.py`
- `src/trainers/metrics/mixins.py`
- `src/data_collators/batch_extras_collator.py`

Run these tests first:
- `tests/test_coord_softce_w1_loss.py`
- `tests/test_coord_soft_ce_w1_collective_guard.py`
- `tests/test_stage1_metric_key_parity.py`
- `tests/test_stage1_registry_masks.py`
- `tests/test_stage1_static_packing_runtime_config.py`

## 3. Stage-2 Two-Channel Training, Matching, Triage, Or Duplicate UL

Open these docs first:
- [`docs/training/STAGE2_DESIGN.md`](training/STAGE2_DESIGN.md)
- [`docs/training/STAGE2_RUNBOOK.md`](training/STAGE2_RUNBOOK.md)
- [`openspec/specs/stage2-ab-training/spec.md`](../openspec/specs/stage2-ab-training/spec.md)
- [`openspec/specs/teacher-forcing-unified-loss-registry/spec.md`](../openspec/specs/teacher-forcing-unified-loss-registry/spec.md)

Open these configs first:
- `configs/stage2_two_channel/base.yaml`
- `configs/stage2_two_channel/prod/`
- `configs/stage2_two_channel/smoke/`

Key v3 config handles:
- `stage2_ab.channel_b.v3_k2.*`
- `rollout_matching.decoding.*`

Open these code files first:
- `src/sft.py`
- `src/trainers/stage2_two_channel.py`
- `src/trainers/rollout_matching/parsing.py`
- `src/trainers/rollout_matching/matching.py`
- `src/trainers/teacher_forcing/module_registry.py`
- `src/trainers/teacher_forcing/objective_atoms.py`
- `src/trainers/teacher_forcing/modules/token_ce.py`
- `src/trainers/teacher_forcing/modules/duplicate_ul.py`
- `src/trainers/teacher_forcing/modules/bbox_geo.py`
- `src/trainers/teacher_forcing/modules/coord_reg.py`

Run these tests first:
- `tests/test_stage2_ab_config_contract.py`
- `tests/test_stage2_ab_training.py`
- `tests/test_stage2_objective_atoms_projection.py`
- `tests/test_teacher_forcing_token_ce.py`
- `tests/test_stage2_pending_metrics_aggregation.py`
- `tests/test_stage2_ab_vllm_server_mode_smoke.py`

## 4. Inference, Confidence, And Offline Evaluation

Open these docs first:
- [`docs/eval/README.md`](eval/README.md)
- [`docs/ARTIFACTS.md`](ARTIFACTS.md)

Open these configs first:
- `configs/infer/pipeline.yaml`
- `configs/eval/detection.yaml`
- `configs/postop/confidence.yaml`
- `configs/bench/`

Open these code files first:
- `scripts/run_infer.py`
- `src/infer/pipeline.py`
- `scripts/postop_confidence.py`
- `scripts/evaluate_detection.py`
- `src/eval/detection.py`
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
- `src/metrics/reporter.py`
- `src/metrics/payload_contract.py`
- `src/callbacks/`

Run these tests first:
- `tests/test_run_manifest_files.py`
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
