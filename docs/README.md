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

## Choose an entry

| Question | Entry |
| --- | --- |
| Which surface owns a decision? | [Project Context](PROJECT_CONTEXT.md) |
| Which checkout or lifecycle applies? | [Branch And Worktree Policy](BRANCH_AND_WORKTREE_POLICY.md) |
| What do the research results mean? | [Research knowledge](../research/index.md) |
| How does the production stack fit together? | [System Overview](SYSTEM_OVERVIEW.md) |
| Where is a source/test owner? | [Implementation Map](IMPLEMENTATION_MAP.md) |
| How do I use the current production route? | [CoordExp-Swift](COORDEXP_SWIFT.md) |

## Domain routers

- [Data and datasets](data/README.md)
- [Training history and runbooks](training/README.md)
- [Inference and evaluation](eval/README.md)
- [Standards and repo policy](standards/README.md)
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
- Current system relationships are explained in [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md).
  Superseded architecture proposals are preserved under
  [history](history/architecture/README.md), outside the current reading path.
- [progress/](../progress/README.md) is a deprecated historical evidence route;
  [research/](../research/index.md) is the active research interpretation route.

## Read-order rule

Start with the user-named question, source, or artifact. When ownership is
unclear, use [AGENT_INDEX.md](AGENT_INDEX.md) or search
[catalog.yaml](catalog.yaml), then read only the selected owner and the evidence
needed for the task. Use the implementation map for source ownership, an exact
stable spec for compatibility semantics, and a research unit or investigation
for research interpretation. The pages above are alternative entry points,
not a mandatory reading sequence. An older plan, proposal, worktree, or
progress note does not establish current behavior.
