---
doc_id: docs.index
layer: docs
doc_type: router
status: canonical
domain: repo
summary: Human-first router for current CoordExp production and research documentation.
tags: [entrypoint, docs, research]
updated: 2026-08-24
---

# Documentation Index

Use this page as the human starting point for current CoordExp behavior. Agents
start from the user-named evidence; when its owner is unclear, use
[AGENT_INDEX.md](AGENT_INDEX.md) or [catalog.yaml](catalog.yaml) to select one
narrow route. Production implementation questions start with CoordExp-Swift on
`main`; research-probe questions start from the fixed `research-probes`
worktree and its research records. Neither route is a substitute for the other.

## Start here

1. [Project Context & Documentation Authority](PROJECT_CONTEXT.md)
2. [Branch And Worktree Policy](BRANCH_AND_WORKTREE_POLICY.md)
3. [Research knowledge index](../research/index.md) for a research question
4. [CoordExp-Swift](COORDEXP_SWIFT.md) for a production implementation question
5. [System Overview](SYSTEM_OVERVIEW.md)
6. [Implementation Map](IMPLEMENTATION_MAP.md)

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
  [research/](../research/index.md) is the active research interpretation route.

## Read-order rule

For production behavior, read `PROJECT_CONTEXT.md`, then the Swift guide,
system overview, implementation map, relevant domain router, and only then the
exact stable spec needed for compatibility-sensitive semantics. For research
work, read `PROJECT_CONTEXT.md`, `BRANCH_AND_WORKTREE_POLICY.md`, the selected
research unit or investigation, and its bound source/artifact evidence; enter
the Swift route only when the named question actually owns that interface. Do
not use an older plan, proposal, worktree, or progress note as a current source
of truth.
