---
doc_id: docs.training.index
layer: docs
doc_type: router
status: canonical
domain: training
summary: Router for Stage-1 and Stage-2 training documentation, metrics, and runbooks.
tags: [training, stage1, stage2]
updated: 2026-03-22
---

# Training Docs

Open this folder when you need current training behavior, recommended configs,
or metric interpretation.

## Read Order

1. [STAGE1_OBJECTIVE.md](STAGE1_OBJECTIVE.md) for baseline Stage-1 behavior
2. [STAGE2_RUNBOOK.md](STAGE2_RUNBOOK.md) for current Stage-2 workflows and launcher patterns
3. [METRICS.md](METRICS.md) for loss-key and logging interpretation
4. [STAGE2_DESIGN.md](STAGE2_DESIGN.md) for historical design context and deprecation rationale
5. [`stage2-ab-training/spec.md`](../../openspec/specs/stage2-ab-training/spec.md) when exact `stage2_two_channel` semantics matter
6. [`rollout-matching-sft/spec.md`](../../openspec/specs/rollout-matching-sft/spec.md) when working on the supported `stage2_rollout_aligned` variant
7. [`runtime-architecture-refactor-program/spec.md`](../../openspec/specs/runtime-architecture-refactor-program/spec.md) when the question is about runtime ownership seams or compatibility-preserving refactors

## Page Roles

- [STAGE1_OBJECTIVE.md](STAGE1_OBJECTIVE.md)
  - Stage-1 objective surfaces and coord-token training details
- [STAGE2_RUNBOOK.md](STAGE2_RUNBOOK.md)
  - YAML-first runbook, smoke workflow, and server-mode launcher entrypoints for the active `stage2_two_channel` path
- [METRICS.md](METRICS.md)
  - canonical training metric and loss interpretation
- [STAGE2_DESIGN.md](STAGE2_DESIGN.md)
  - historical design note for the removed self-context iteration surface

## Use This Router For

- "How does current Stage-2 work?"
- "Which page is runbook vs metrics vs historical design?"
- "What should I read before touching Stage-1 or Stage-2 configs?"

## Code Handles

- `src/sft.py`
- `src/bootstrap/`
- `src/trainers/stage2_two_channel.py`
- `src/trainers/stage2_two_channel/`
- `src/trainers/stage2_rollout_aligned.py`
- `src/trainers/rollout_aligned_targets.py`
- `src/trainers/rollout_aligned_evaluator.py`
- `src/trainers/rollout_runtime/`
- `src/trainers/rollout_matching/`
- `src/trainers/teacher_forcing/`
- `src/launchers/stage2_vllm_server.py`
- `configs/stage1/`
- `configs/stage2_two_channel/`
