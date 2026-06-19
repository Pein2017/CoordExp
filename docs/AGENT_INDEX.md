---
doc_id: docs.agent-index
layer: docs
doc_type: agent-router
status: canonical
domain: repo
summary: Agent-first retrieval guide for CoordExp documentation and research notes.
tags: [agents, retrieval, docs]
updated: 2026-06-15
---

# Agent Index

Use this page when the consumer is an AI agent working inside the repository.

Primary machine entrypoint:

- [docs/catalog.yaml](catalog.yaml)

Human support entrypoints:

- [docs/README.md](README.md)
- [progress/README.md](../progress/README.md)
- [docs/history/README.md](history/README.md) for non-normative implementation-plan and design provenance

## Default Read Order

1. [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)
2. [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md)
3. [IMPLEMENTATION_MAP.md](IMPLEMENTATION_MAP.md)
4. the relevant domain router
5. relevant `openspec/specs/` only for stable contract semantics
   - use [`runtime-architecture-refactor-program/spec.md`](../openspec/specs/runtime-architecture-refactor-program/spec.md) for runtime structure, internal seams, and compatibility-preserving refactors
   - use [`stage2-rollout-correction/spec.md`](../openspec/specs/stage2-rollout-correction/spec.md) for active Stage-2 behavior and config contracts
6. `progress/` only when current docs do not answer the historical or empirical question

## Query Routing

- Repo structure, runtime seams, or doc precedence:
  - [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)
  - [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md)
  - [IMPLEMENTATION_MAP.md](IMPLEMENTATION_MAP.md)
  - [catalog.yaml](catalog.yaml)
  - [`runtime-architecture-refactor-program/spec.md`](../openspec/specs/runtime-architecture-refactor-program/spec.md)
- Architecture proposals and archived plan provenance:
  - [docs/architecture/README.md](architecture/README.md) for proposal/review routing
  - [docs/history/README.md](history/README.md) for non-normative historical docs
  - [docs/history/superpowers/README.md](history/superpowers/README.md) for dated plans/specs/handoffs moved out of current docs
- End-to-end system flow:
  - [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md)
- Code and test entrypoints:
  - [IMPLEMENTATION_MAP.md](IMPLEMENTATION_MAP.md)
- Dataset contracts and preprocessing:
  - [docs/data/README.md](data/README.md)
  - [docs/data/CONTRACT.md](data/CONTRACT.md)
  - [docs/data/PREPARATION.md](data/PREPARATION.md)
  - [docs/data/PACKING.md](data/PACKING.md) for Stage-1 static-packing and hard-cap questions
- Stage-1 training:
  - [docs/training/README.md](training/README.md)
  - [docs/training/STAGE1_OBJECTIVE.md](training/STAGE1_OBJECTIVE.md) for baseline Stage-1 behavior, canonical detection teacher forcing, legacy prefix-rollin ablation boundaries, and retired candidate-objective boundaries
  - [docs/data/PACKING.md](data/PACKING.md)
  - Current public Stage-1 research teacher-forcing route: `pipeline.id: stage1_research_teacher_forcing`; configs live under `configs/stage1/detection_teacher_forcing/`.
  - Public Stage-1 pipeline ids: `stage1_standard_sft` for standard assistant-label CE and `stage1_research_teacher_forcing` for compact objective research.
  - Pipeline registry and descriptor map: `src/training/pipeline_registry.py::TrainingPipelineRegistry`, `src/training/pipelines/stage1_json_ce.py`, and `src/training/pipelines/stage1_compact_trie_ce.py`.
  - Public Stage-1 objective ids: `standard_ce` and `research_teacher_forcing`; implementation terms such as `token_ce`, `trie_ce`, and `coord_soft_ce` stay internal to objective modules or term config.
  - Recursive-detection / ET-RMP is a preserved comparator and ablation family, not the default new Stage-1 SFT route. Current infer/eval comparator lineage lives under `configs/infer/recursive_detection_ce`; quarantined Stage-1 training roots and authoring snippets under `configs/archive/detection_scene_clean_break/stage1/` remain historical evidence.
- Stage-2 training:
  - [docs/training/README.md](training/README.md)
  - [docs/training/STAGE2_RUNBOOK.md](training/STAGE2_RUNBOOK.md) for current behavior, launcher workflow, and historical-context pointers
  - [docs/training/METRICS.md](training/METRICS.md)
  - [`stage2-rollout-correction/spec.md`](../openspec/specs/stage2-rollout-correction/spec.md)
  - [`rollout-matching-sft/spec.md`](../openspec/specs/rollout-matching-sft/spec.md) for the retired rollout-matching trainer contract
  - [`runtime-architecture-refactor-program/spec.md`](../openspec/specs/runtime-architecture-refactor-program/spec.md)
  - Public Stage-2 pipeline id: `stage2_rollout_correction`, resolved through `src/training/pipeline_registry.py::TrainingPipelineRegistry`.
  - Planning direction: duplicate filtering before target realization, greedy-IoU assignment over retained rollout objects, and GT correction event construction for unmatched GT.
  - Greedy IoU is the only live Stage-2 assignment strategy; do not reintroduce alternate assignment mechanisms without a new spec.
- Metrics, diagnostics, and artifacts:
  - [docs/training/METRICS.md](training/METRICS.md) for `MetricEvent`, `DiagnosticEvent`, bounded diagnostic profiles, and clean-write/tolerant-read metric behavior
  - [ARTIFACTS.md](ARTIFACTS.md) for resolved config artifacts, rank-0 artifact names, and Stage-2 policy provenance
- Inference and evaluation:
  - [docs/eval/README.md](eval/README.md)
  - [docs/eval/CONTRACT.md](eval/CONTRACT.md)
  - [docs/eval/WORKFLOW.md](eval/WORKFLOW.md)
  - [ARTIFACTS.md](ARTIFACTS.md)
  - [`inference-pipeline/spec.md`](../openspec/specs/inference-pipeline/spec.md)
  - [`inference-engine/spec.md`](../openspec/specs/inference-engine/spec.md)
  - [`detection-evaluator/spec.md`](../openspec/specs/detection-evaluator/spec.md)
  - [`runtime-architecture-refactor-program/spec.md`](../openspec/specs/runtime-architecture-refactor-program/spec.md)
- Standards and repo policy:
  - [docs/standards/README.md](standards/README.md)

## Progress Usage Rule

Use `progress/` only for:

- why a design exists
- what failed empirically
- benchmark evidence
- historical derivations

Do not answer current-behavior questions from `progress/` if `docs/` or `openspec/specs/` already cover them.

## Historical Docs Usage Rule

Use `docs/history/` only for non-normative provenance:

- dated implementation plans and design specs
- migration handoffs
- superseded training notes
- branch research planning that has not been promoted into current docs

Do not answer current-behavior questions from `docs/history/` unless a current doc explicitly points there for historical context.

## Suggested Search Seeds

```bash
rg -n "stage2_rollout_correction|stage2_coordination|stage2_rollout_runtime|rollout_runtime|rollout_aligned_targets|rollout_aligned_evaluator|stage2_vllm_server" docs openspec src scripts configs  # search
rg -n "stage1_research_teacher_forcing|DetectionScene|DetectionSupervisionView|MetricEvent|DiagnosticEvent|DetectionAssignment|CorrectionEvent" docs src tests
rg -n "runtime-architecture-refactor-program|pipeline_manifest|run_metadata|trainer_setup|resolved_config.json|effective_runtime.json" docs openspec src tests
rg -n "contract|jsonl|geometry|packing" docs/data src/datasets
rg -n "infer|engine|backends|artifacts|orchestration|confidence|metrics" docs/eval docs/training src scripts
```
