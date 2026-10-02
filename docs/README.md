---
doc_id: docs.index
layer: docs
doc_type: router
status: canonical
domain: repo
summary: Human-first router for the stable CoordExp documentation layer.
tags: [entrypoint, docs]
updated: 2026-07-11
---

# Documentation Index

Use this page as the human starting point for current CoordExp behavior. Agents
start from the user-named evidence; when its owner is unclear, use
[AGENT_INDEX.md](AGENT_INDEX.md) or [catalog.yaml](catalog.yaml) to select one
narrow route. The canonical implementation at the current fixed point is
coordexp-infras on `main`.

## Start here

1. [Project Context & Documentation Authority](PROJECT_CONTEXT.md)
2. [Branch And Worktree Policy](BRANCH_AND_WORKTREE_POLICY.md)
3. [coordexp-infras](coordexp_infras.md)
4. [System Overview](SYSTEM_OVERVIEW.md)
5. [Implementation Map](IMPLEMENTATION_MAP.md)

## Domain routers

- [Data and datasets](data/README.md)
- [Training history and runbooks](training/README.md)
- [Inference and evaluation](eval/README.md)
- [Standards and repo policy](standards/README.md)
- [Architecture proposals and lifecycle](architecture/README.md)
- [Artifacts and provenance](ARTIFACTS.md)

## Contract and proposal boundaries

- Stable compatibility semantics live in the relevant
  [`openspec/specs/`](../openspec/specs/) contract.
- Bounded active code-change work lives only in an explicitly scoped
  `openspec/changes/<change>/` workspace, which may
  carry durable proposal/design/tasks/apply/verify/archive artifacts. Add
  delta specs only when a stable compatibility-sensitive contract changes; do
  not invent normative deltas for internal refactors.
- Architecture proposals and one-time project plans are non-normative design
  reasoning and sequencing; they do not authorize implementation.
- Current and historical architecture material is routed from
  [architecture/README.md](architecture/README.md). Completed or superseded
  proposal material is preserved and routed to [history/](history/README.md)
  when migration is safe.
- [progress/](../progress/README.md) is a deprecated historical evidence route;
  the root [research index](../research/index.md) identifies its retained source
  scope and links to the current research owner.

## Task-local read order

Start from the evidence named by the task. If its owner is unclear, follow the
task-local routes in [AGENT_INDEX.md](AGENT_INDEX.md), then read only the
selected owner and the contracts or evidence that task needs. Use
[PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) when repository-wide authority or
routing needs clarification; no fixed document chain is required.
