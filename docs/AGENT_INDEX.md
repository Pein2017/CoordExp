---
doc_id: docs.agent-index
layer: docs
doc_type: agent-router
status: canonical
domain: repo
summary: Agent-first retrieval guide for CoordExp documentation and research notes.
tags: [agents, retrieval, docs]
updated: 2026-05-09
---

# Agent Index

Use this page when the consumer is an AI agent working inside the repository.

Primary machine entrypoint:

- [docs/catalog.yaml](catalog.yaml)

Human support entrypoints:

- [docs/README.md](README.md)
- [progress/README.md](../progress/README.md)

## Default Read Order

1. [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)
2. [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md)
3. [IMPLEMENTATION_MAP.md](IMPLEMENTATION_MAP.md)
4. the relevant domain router
5. relevant `openspec/specs/` only for stable contract semantics
   - use [`runtime-architecture-refactor-program/spec.md`](../openspec/specs/runtime-architecture-refactor-program/spec.md) for runtime structure, internal seams, and compatibility-preserving refactors
   - use [`stage1-latest-detection-objectives/spec.md`](../openspec/specs/stage1-latest-detection-objectives/spec.md) for latest Stage-1 compact recursive detection objective contracts
   - use [`stage2-ab-training/spec.md`](../openspec/specs/stage2-ab-training/spec.md) for stable Stage-2 contract semantics and config-contract boundaries
6. `progress/` only when current docs do not answer the historical or empirical question

## Query Routing

- Repo structure, runtime seams, or doc precedence:
  - [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)
  - [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md)
  - [IMPLEMENTATION_MAP.md](IMPLEMENTATION_MAP.md)
  - [catalog.yaml](catalog.yaml)
  - [`runtime-architecture-refactor-program/spec.md`](../openspec/specs/runtime-architecture-refactor-program/spec.md)
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
  - [docs/training/STAGE1_OBJECTIVE.md](training/STAGE1_OBJECTIVE.md) for baseline Stage-1 behavior, compact recursive detection, prefix-rollin ablation boundaries, and retired candidate-objective boundaries
  - [docs/data/PACKING.md](data/PACKING.md)
  - Latest compact detection production baseline/comparator: `configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2.yaml`, `src/config/schema.py::LatestDetectionTrainingConfig`, and `src/detection/runtime.py`
  - Prefix-rollin E1 ablation route: `configs/stage1/recursive_detection_ce_latest/ablation/compact_full_prefix_rollin_balance2.yaml`; compact_full only, `<|im_end|>` only, empirical EOS prior only for smoke/ablation, production requires `calibrated_formula_ref` with a versioned artifact.
  - Prefix-rollin E2 separator diagnostic route: `configs/stage1/recursive_detection_ce_latest/ablation/compact_full_prefix_rollin_separator2.yaml`; same as E1 except `objective.boundary.separator_continue_weight=2.0` to test the diagnosed `\n` vs `<|im_end|>` free-boundary failure.
  - [`stage1-latest-detection-objectives/spec.md`](../openspec/specs/stage1-latest-detection-objectives/spec.md) for stable latest-detection objective contracts; active future variants live under `openspec/changes/<change-id>/`.
  - Latest compact detection authoring snippets, not yet consumed by canonical launch configs: `configs/_shared/latest_detection/`
  - Legacy compact bridge example only: `configs/stage1/compact_detection_sequence/smoke/compact_full_tiny.yaml`
- Stage-2 training:
  - [docs/training/README.md](training/README.md)
  - [docs/training/STAGE2_RUNBOOK.md](training/STAGE2_RUNBOOK.md)
  - [docs/training/METRICS.md](training/METRICS.md)
  - Use OpenSpec specs for stable contracts after checking the current runbook.
  - [`stage2-ab-training/spec.md`](../openspec/specs/stage2-ab-training/spec.md)
  - [`rollout-matching-sft/spec.md`](../openspec/specs/rollout-matching-sft/spec.md) for the supported `stage2_rollout_aligned` variant
  - [`runtime-architecture-refactor-program/spec.md`](../openspec/specs/runtime-architecture-refactor-program/spec.md)
  - [docs/training/STAGE2_RUNBOOK.md](training/STAGE2_RUNBOOK.md) for current behavior and historical-context pointers
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

## Suggested Search Seeds

```bash
rg -n "stage2_two_channel|stage2_ab|stage2_coordination|stage2_rollout_aligned|rollout_runtime|rollout_aligned_targets|rollout_aligned_evaluator|stage2_vllm_server|loss_duplicate_burst_unlikelihood" docs openspec src scripts configs
rg -n "runtime-architecture-refactor-program|pipeline_manifest|run_metadata|trainer_setup" docs openspec src tests
rg -n "contract|jsonl|geometry|packing" docs/data src/datasets
rg -n "infer|engine|backends|artifacts|orchestration|confidence|metrics" docs/eval docs/training src scripts
```
