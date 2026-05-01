---
doc_id: docs.training.index
layer: docs
doc_type: router
status: canonical
domain: training
summary: Router for Stage-1 and Stage-2 training documentation, metrics, and runbooks.
tags: [training, stage1, stage2]
updated: 2026-04-29
---

# Training Docs

Open this folder when you need current training behavior, recommended configs,
or metric interpretation.

## Read Order

1. [STAGE1_OBJECTIVE.md](STAGE1_OBJECTIVE.md) for baseline Stage-1 behavior and the set-continuation MP experiment
2. [STAGE1_ET_RMP_CE.md](STAGE1_ET_RMP_CE.md) for the factual ET-RMP-CE implementation, config, verification, and current artifact export
3. [../data/PACKING.md](../data/PACKING.md) for the current Stage-1 static-packing contract and defaults
4. [STAGE2_RUNBOOK.md](STAGE2_RUNBOOK.md) for current Stage-2 workflows and launcher patterns
5. [LVIS.md](LVIS.md) for LVIS-specific dataset, prompt, Stage-2, and evaluation semantics
6. [METRICS.md](METRICS.md) for loss-key and logging interpretation
7. [STAGE2_DESIGN.md](STAGE2_DESIGN.md) for historical design context and deprecation rationale
8. [`stage2-ab-training/spec.md`](../../openspec/specs/stage2-ab-training/spec.md) when exact `stage2_two_channel` semantics matter
9. [`rollout-matching-sft/spec.md`](../../openspec/specs/rollout-matching-sft/spec.md) when working on the supported `stage2_rollout_aligned` variant
10. [`runtime-architecture-refactor-program/spec.md`](../../openspec/specs/runtime-architecture-refactor-program/spec.md) when the question is about runtime ownership seams or compatibility-preserving refactors

## Page Roles

- [STAGE1_OBJECTIVE.md](STAGE1_OBJECTIVE.md)
  - Stage-1 objective surfaces and coord-token training details
- [STAGE1_ET_RMP_CE.md](STAGE1_ET_RMP_CE.md)
  - Factual implementation export for the experimental ET-RMP-CE objective,
    including code paths, config, smart-batch runtime, tests, and recorded
    step-100/step-200 artifacts
- [../data/PACKING.md](../data/PACKING.md)
  - Stage-1 static packing contract, hard length cap, and fail-fast behavior for overlength atomic samples
- [STAGE2_RUNBOOK.md](STAGE2_RUNBOOK.md)
  - YAML-first runbook, smoke workflow, and server-mode launcher entrypoints for the active `stage2_two_channel` path
- [LVIS.md](LVIS.md)
  - LVIS federated-label design note plus migration guide for Stage-1, Stage-2, and evaluation
- [METRICS.md](METRICS.md)
  - canonical training metric and loss interpretation
- [STAGE2_DESIGN.md](STAGE2_DESIGN.md)
  - historical design note for the removed self-context iteration surface

## Use This Router For

- "How does current Stage-2 work?"
- "Which page is runbook vs metrics vs historical design?"
- "What should I read before touching Stage-1 or Stage-2 configs?"
- "What is the current Stage-1 packing and `global_max_length` contract?"
- "How do I run the Stage-1 set-continuation production config?"
- "What exactly was implemented for ET-RMP-CE, and what artifacts exist so far?"

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
- `src/trainers/stage1_set_continuation/`
- `configs/stage1/set_continuation/`
- `src/launchers/stage2_vllm_server.py`
- `configs/_shared/datasets/`
- `configs/_shared/prompts/`
- `configs/stage1/`
- `configs/stage2_two_channel/`
