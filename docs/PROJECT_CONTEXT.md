---
doc_id: docs.project-context
layer: docs
doc_type: root-context
status: canonical
domain: repo
summary: Defines documentation precedence, document roles, and the universal read order for CoordExp.
tags: [precedence, docs, agents]
updated: 2026-03-22
---

# Project Context & Precedence

This page defines how to interpret every Markdown file in the repository.

## Precedence

When two sources disagree, use this order:

1. `openspec/specs/`
2. `docs/`
3. `openspec/changes/<active-change>/`
4. `progress/`

Interpretation:

- `openspec/specs/` is normative.
- `docs/` is the stable operator-facing explanation layer.
- `openspec/changes/` explains active deltas and implementation intent.
- `progress/` stores dated evidence, diagnostics, benchmarks, and historical reasoning.

## Layer Responsibilities

- `docs/`
  - stable interfaces, workflows, runbooks, and routing
  - concise, pointer-first, low-duplication
- `progress/`
  - historical notes, experiments, audits, and diagnostics
  - evidence-first, dated, non-normative
- `docs/catalog.yaml`
  - machine-readable inventory for every Markdown document in `docs/` and `progress/`
- `docs/AGENT_INDEX.md`
  - fast-path retrieval instructions for AI agents

## Universal Read Order

For most work:

1. [docs/README.md](README.md)
2. [docs/AGENT_INDEX.md](AGENT_INDEX.md) if the consumer is an AI agent
3. [docs/SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md)
4. [docs/IMPLEMENTATION_MAP.md](IMPLEMENTATION_MAP.md)
5. the relevant domain router under `docs/`
6. relevant `openspec/specs/`
   - use [`openspec/specs/runtime-architecture-refactor-program/spec.md`](../openspec/specs/runtime-architecture-refactor-program/spec.md) when the question is about runtime ownership, launchers, artifacts, or compatibility-preserving refactors
7. `progress/` only if you need design history or empirical evidence

## Authoring Rules

- Do not duplicate stable contracts across multiple router pages.
- Put stable workflows in `docs/`.
- Put dated evidence and investigations in `progress/`.
- Remove obsolete paths instead of preserving compatibility stubs.
- Prefer one canonical page per question:
  - data contract -> `docs/data/CONTRACT.md`
  - stage-2 runbook -> `docs/training/STAGE2_RUNBOOK.md`
  - stage-2 design history -> `docs/training/STAGE2_DESIGN.md`
  - evaluation contract -> `docs/eval/CONTRACT.md`

## Promotion Rule

Promote a note from `progress/` into `docs/` when all of the following are true:

- it is no longer tied to one dated run or diagnosis
- it defines the current recommended workflow
- people would reasonably expect it to be the first page they open

Keep a topic in `progress/` when it is primarily:

- an experiment log
- a benchmark report
- a diagnosis or audit
- long-form design history
