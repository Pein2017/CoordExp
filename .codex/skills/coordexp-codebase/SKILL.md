---
name: coordexp-codebase
description: "Use when navigating the CoordExp research codebase, locating current docs/specs/code entrypoints, or changing data, training, Stage-1, Stage-2, inference, evaluation, artifact, or provenance behavior."
---

# CoordExp Codebase Navigation

This skill is an activation layer for daily CoordExp work. Keep it pointer-first: use repo docs for durable truth, then narrow to code symbols.

## Authority Model

Use the current repo authority spine for behavior questions:

1. `docs/PROJECT_CONTEXT.md`
2. `docs/SYSTEM_OVERVIEW.md`
3. `docs/IMPLEMENTATION_MAP.md`
4. relevant domain docs under `docs/`
5. `openspec/specs/` only for stable compatibility-sensitive contracts
6. `openspec/changes/<active-change>/` only when the user explicitly scopes an active change
7. `progress/` only for history, diagnostics, benchmark evidence, or design derivation

Fast routing helpers:

- `docs/AGENT_INDEX.md` for agent-first retrieval
- `docs/catalog.yaml` for machine-readable inventory
- `docs/ARTIFACTS.md` for artifact, manifest, and provenance surfaces

When sources disagree, prefer current checked-in `docs/` for operator behavior. Use OpenSpec to pin stable contract semantics, not as the default planning layer.

## Daily Navigation Loop

1. Open the relevant docs route before searching source.
2. Use `rg` or `rtk grep` to narrow files, config keys, symbols, and tests.
3. For Python code understanding or edits, activate Serena on the exact repo/worktree and use symbol tools after narrowing.
4. Inspect the smallest code surface that can answer the question.
5. Validate only when validation is in scope, using targeted tests or artifacts named by `docs/IMPLEMENTATION_MAP.md`.
6. Update docs when changing user-facing defaults, entrypoints, artifact names, log keys, metrics, or recommended workflows.

## `src/` Map

- `src/sft.py`: YAML-first training entrypoint; resolves config, runtime plan, datasets, packing/cache policy, trainers, callbacks, manifests.
- `src/config/`: config loading, strict schema, prompt variants, rollout matching schema, latest compact detection schema.
- `src/training_runtime/`: trainer-variant policy. Start with `src/training_runtime/plan.py::resolve_training_runtime_plan` for collator, packing-owner, and pipeline-namespace questions.
- `src/datasets/`: dense-caption JSONL path, preprocessors, builders, geometry, augmentation, cache, legacy fusion surfaces.
- `src/detection/`: compact/latest detection sequence stack: dataset, template, objective, tokenization, packing policy.
- `src/data_collators/`: collator families, batch extras, dataset metrics, Stage-1 set-continuation collator.
- `src/trainers/`: Stage-1 set-continuation, Stage-2 two-channel, rollout-aligned, rollout runtime, teacher-forcing modules, losses, metrics.
- `src/infer/`: inference pipeline, engine, backends, checkpoint resolution, artifact summaries, visualization helpers.
- `src/eval/`: detection evaluation, scoring/confidence, proxy views, Oracle-K, evaluation artifacts.
- `src/bootstrap/`: experiment manifests, pipeline manifests, run metadata, trainer setup.
- `src/common/`: shared schemas, paths, prediction parsing, duplicate control, detection sequence, object field ordering.
- `src/analysis/`: research probes and mechanism studies; use for evidence, not first-source current behavior.

## Task Routing Matrix

Data contract, JSONL, geometry, or preprocessing:

- Docs: `docs/data/README.md`, `docs/data/CONTRACT.md`, `docs/data/PREPARATION.md`, `docs/data/PACKING.md`
- Code: `src/datasets/geometry.py`, `src/datasets/dense_caption.py`, `src/datasets/builders/jsonlines.py`, `src/detection/dataset.py`
- Guardrail: preserve image/geometry alignment end-to-end; route bbox math through `src/datasets/geometry.py` unless the task is explicitly in `src/detection/` serialization code.

Stage-1 baseline SFT:

- Docs: `docs/training/README.md`, `docs/training/STAGE1_OBJECTIVE.md`, `docs/data/PACKING.md`
- Configs: `configs/stage1/sft_base.yaml`, `configs/stage1/profiles/`, `configs/stage1/smoke/`
- Code: `src/sft.py`, `src/datasets/dense_caption.py`, `src/trainers/losses/`, `src/trainers/metrics/`, `src/data_collators/`

Stage-1 set-continuation ET-RMP-CE:

- Docs: `docs/training/STAGE1_OBJECTIVE.md`, `docs/data/PACKING.md`
- Configs: `configs/stage1/set_continuation/production.yaml`, `configs/stage1/set_continuation/smoke/`
- Code: `src/trainers/stage1_set_continuation/`, `src/data_collators/stage1_set_continuation_collator.py`, `src/training_runtime/plan.py`
- Guardrail: this variant rejects dataset/eval packing and encoded-sample cache in current runtime policy.

Stage-1 compact recursive detection:

- Docs: `docs/training/STAGE1_OBJECTIVE.md`, `docs/data/PACKING.md`
- Configs: `configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2.yaml`, `configs/stage1/recursive_detection_ce_latest/smoke/`, `configs/stage1/compact_detection_sequence/smoke/`
- Code: `src/config/schema.py::LatestDetectionTrainingConfig`, `src/detection/dataset.py::DetectionTrainingDataset`, `src/detection/packing.py`, `src/sft.py::_resolve_recursive_detection_ce_cfg`, `src/sft.py::_assert_latest_detection_runtime_supported`
- Guardrail: latest recursive detection sidecars currently require packing disabled until offset/sidecar rewriting is explicitly supported.

Stage-2 two-channel training:

- Docs: `docs/training/STAGE2_RUNBOOK.md`, `docs/training/METRICS.md`
- Stable contracts: `openspec/specs/stage2-ab-training/spec.md`, `openspec/specs/teacher-forcing-unified-loss-registry/spec.md`
- Configs: `configs/stage2_two_channel/base.yaml`, `configs/stage2_two_channel/prod/`, `configs/stage2_two_channel/smoke/`
- Code: `src/training_runtime/plan.py`, `src/trainers/stage2_two_channel.py`, `src/trainers/stage2_two_channel/`, `src/trainers/stage2_ab/`, `src/trainers/teacher_forcing/`, `src/trainers/rollout_runtime/`
- Guardrail: `custom.trainer_variant: stage2_two_channel` uses `stage2_ab.pipeline`, not `rollout_matching.pipeline`.

Stage-2 rollout-aligned variant:

- Docs: `docs/training/STAGE2_RUNBOOK.md`, `docs/training/METRICS.md`
- Stable contracts: `openspec/specs/rollout-matching-sft/spec.md`
- Code: `src/trainers/stage2_rollout_aligned.py`, `src/trainers/rollout_matching_sft.py`, `src/trainers/rollout_matching/`, `src/trainers/rollout_aligned_targets.py`, `src/trainers/rollout_aligned_evaluator.py`
- Guardrail: `custom.trainer_variant: stage2_rollout_aligned` uses `rollout_matching.pipeline`, not `stage2_ab.pipeline`.

Inference, confidence, evaluation, and visualization:

- Docs: `docs/eval/README.md`, `docs/eval/CONTRACT.md`, `docs/eval/WORKFLOW.md`, `docs/ARTIFACTS.md`
- Stable contracts: `openspec/specs/inference-pipeline/spec.md`, `openspec/specs/inference-engine/spec.md`, `openspec/specs/detection-evaluator/spec.md`
- Code: `src/infer/pipeline.py::run_pipeline`, `src/infer/engine.py`, `src/infer/backends.py`, `src/infer/artifacts.py`, `src/eval/detection.py::evaluate_and_save`, `src/eval/artifacts.py`, `src/eval/confidence_postop.py`
- Guardrail: confidence post-op is for compatible scored surfaces; non-canonical bbox formats require explicit compatibility handling.

Artifacts, manifests, and provenance:

- Docs: `docs/ARTIFACTS.md`, `docs/training/METRICS.md`, `docs/eval/WORKFLOW.md`
- Code: `src/bootstrap/experiment_manifest.py`, `src/bootstrap/pipeline_manifest.py`, `src/bootstrap/run_metadata.py`, `src/bootstrap/trainer_setup.py`, `src/metrics/reporter.py`, `src/metrics/payload_contract.py`, `src/infer/artifacts.py`, `src/eval/artifacts.py`

## Guardrails

- Config-first: prefer YAML/config schema changes over new stable CLI flags.
- Offline-prepared single-dataset JSONL is the default training surface; runtime fusion remains legacy/experimental.
- Preserve Qwen3-VL chat-template compatibility and current artifact contracts.
- Training uses `do_resize=false`; do not silently introduce resize behavior that changes geometry alignment.
- Do not edit upstream HF model files such as `modeling_qwen3_vl.py`.
- Do not invent benchmark results or compare scopes without labels such as `val200`, `limit=200`, full-val, proxy, raw-text, coord-token, checkpoint id, and GPU launch shape.
