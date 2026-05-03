---
doc_id: docs.index
layer: docs
doc_type: router
status: canonical
domain: repo
summary: Human-first router for the stable CoordExp documentation layer.
tags: [entrypoint, docs]
updated: 2026-05-03
---

# Documentation Index

Use this page as the human starting point for current CoordExp behavior.

For machine-readable routing, use [docs/catalog.yaml](catalog.yaml).
For AI-agent-first routing, use [AGENT_INDEX.md](AGENT_INDEX.md).

## Start Here

1. [Project Context & Documentation Authority](PROJECT_CONTEXT.md)
2. [System Overview](SYSTEM_OVERVIEW.md)
3. [Implementation Map](IMPLEMENTATION_MAP.md)

## Domain Routers

- Data and dataset interfaces:
  - [docs/data/README.md](data/README.md)
- Training behavior and runbooks:
  - [docs/training/README.md](training/README.md)
- Inference and evaluation:
  - [docs/eval/README.md](eval/README.md)
- Standards and repo policy:
  - [docs/standards/README.md](standards/README.md)

## Cross-Cutting Docs

- [ARTIFACTS.md](ARTIFACTS.md): runtime artifacts, provenance, and logging surfaces
- [AGENT_INDEX.md](AGENT_INDEX.md): fast-path retrieval guide for coding assistants
- [catalog.yaml](catalog.yaml): curated machine-readable catalog for `docs/` and important `progress/` routes
- [`runtime-architecture-refactor-program/spec.md`](../openspec/specs/runtime-architecture-refactor-program/spec.md): stable runtime-structure and compatibility contract

## Research History

- [progress/README.md](../progress/README.md): human router for historical notes
- [progress/index.yaml](../progress/index.yaml): machine-readable progress catalog

## Read Order Rule

Read current behavior in this order:

1. `docs/PROJECT_CONTEXT.md`
2. `docs/SYSTEM_OVERVIEW.md`
3. `docs/IMPLEMENTATION_MAP.md`
4. the relevant domain router under `docs/`
5. `openspec/specs/` only for stable compatibility-sensitive contract semantics
6. `openspec/changes/<active-change>/` only when explicitly in scope
7. `progress/` for historical evidence, diagnostics, and benchmark context
