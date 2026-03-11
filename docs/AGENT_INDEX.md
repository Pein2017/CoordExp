---
doc_id: docs.agent-index
layer: docs
doc_type: agent-router
status: canonical
domain: repo
summary: Agent-first retrieval guide for CoordExp documentation and research notes.
tags: [agents, retrieval, docs]
updated: 2026-03-09
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
5. relevant `openspec/specs/`
6. `progress/` only when current docs do not answer the historical or empirical question

## Query Routing

- Repo structure or doc precedence:
  - [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)
  - [catalog.yaml](catalog.yaml)
- End-to-end system flow:
  - [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md)
- Code and test entrypoints:
  - [IMPLEMENTATION_MAP.md](IMPLEMENTATION_MAP.md)
- Dataset contracts and preprocessing:
  - [docs/data/README.md](data/README.md)
  - [docs/data/CONTRACT.md](data/CONTRACT.md)
  - [docs/data/PREPARATION.md](data/PREPARATION.md)
- Stage-1 training:
  - [docs/training/README.md](training/README.md)
  - [docs/training/STAGE1_OBJECTIVE.md](training/STAGE1_OBJECTIVE.md)
- Stage-2 training:
  - [docs/training/README.md](training/README.md)
  - [docs/training/STAGE2_DESIGN.md](training/STAGE2_DESIGN.md)
  - [docs/training/STAGE2_RUNBOOK.md](training/STAGE2_RUNBOOK.md)
  - [docs/training/METRICS.md](training/METRICS.md)
- Inference and evaluation:
  - [docs/eval/README.md](eval/README.md)
  - [docs/eval/CONTRACT.md](eval/CONTRACT.md)
  - [docs/eval/WORKFLOW.md](eval/WORKFLOW.md)
  - [ARTIFACTS.md](ARTIFACTS.md)
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
rg -n "stage2|rollout|loss_dead_anchor_suppression|clean-prefix" docs progress openspec configs
rg -n "contract|jsonl|geometry|packing" docs/data src/datasets
rg -n "eval|infer|confidence|metrics" docs/eval docs/training scripts src
```
