---
doc_id: docs.training.index
layer: docs
doc_type: router
status: canonical
domain: training
summary: Router for Stage-1 and Stage-2 training documentation, metrics, and runbooks.
tags: [training, stage1, stage2]
updated: 2026-03-09
---

# Training Docs

Open this folder when you need current training behavior, recommended configs, or metric interpretation.

## Read Order

1. [STAGE1_OBJECTIVE.md](STAGE1_OBJECTIVE.md) for baseline Stage-1 behavior
2. [STAGE2_DESIGN.md](STAGE2_DESIGN.md) for the current Stage-2 design frame
3. [STAGE2_RUNBOOK.md](STAGE2_RUNBOOK.md) for operational Stage-2 workflows
4. [METRICS.md](METRICS.md) for loss-key and logging interpretation

## Page Roles

- [STAGE1_OBJECTIVE.md](STAGE1_OBJECTIVE.md)
  - Stage-1 objective surfaces and coord-token training details
- [STAGE2_DESIGN.md](STAGE2_DESIGN.md)
  - stable design overview for the current Stage-2 path
- [STAGE2_RUNBOOK.md](STAGE2_RUNBOOK.md)
  - YAML-first runbook and smoke workflow
- [METRICS.md](METRICS.md)
  - canonical training metric and loss interpretation

## Use This Router For

- "How does current Stage-2 work?"
- "Which page is design vs runbook vs metrics?"
- "What should I read before touching Stage-1 or Stage-2 configs?"

## Code Handles

- `src/sft.py`
- `src/trainers/stage2_two_channel.py`
- `src/trainers/rollout_matching/`
- `src/trainers/teacher_forcing/`
- `configs/stage1/`
- `configs/stage2_two_channel/`
