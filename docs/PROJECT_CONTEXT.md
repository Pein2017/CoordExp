---
doc_id: docs.project-context
layer: docs
doc_type: root-context
status: canonical
domain: repo
summary: Defines documentation ownership, contract authority, and the universal read order for CoordExp.
tags: [precedence, docs, agents]
updated: 2026-05-03
---

# Project Context & Documentation Authority

This page defines how to interpret every Markdown file in the repository.

## Authority Model

Use `docs/` as the current operator-facing truth for architecture, workflows,
routing, artifact names, and recommended development practice.

Use `openspec/specs/` only when a question needs a stable compatibility contract:
training/eval behavior, config schemas, loss semantics, artifact names, or
normative metric semantics. OpenSpec is not the default planning layer for
ordinary implementation work.

Use `openspec/changes/<active-change>/` only when an active change is explicitly
in scope.

Use `research/` for active research ideas, investigations, interpretations,
negative results, and continuation context.

Use `progress/` only as a legacy/deprecated archive for old dated evidence,
diagnostics, benchmark reports, empirical failures, design derivations, and
historical reasoning that has not yet been migrated. Do not create new
`progress/` records, and do not answer current behavior from `progress/`.

## Layer Responsibilities

- `docs/`
  - stable interfaces, workflows, runbooks, routing, architecture, and current status
  - concise, pointer-first, low-duplication
- `openspec/specs/`
  - stable compatibility contracts only
  - use for exact semantics of supported training/eval/config/artifact surfaces
- `openspec/changes/`
  - active deltas and implementation intent only when explicitly in scope
- `research/`
  - active research ideas, investigations, interpretations, negative results,
    and continuation context
- `progress/`
  - legacy/deprecated archive of old historical notes, experiments, audits,
    diagnostics, and benchmark evidence
  - read-only for reconstruction unless the user explicitly asks for migration
- `docs/catalog.yaml`
  - machine-readable curated inventory for `docs/` and legacy `progress/` routes
- `docs/AGENT_INDEX.md`
  - fast-path retrieval instructions for AI agents

## Universal Read Order

For most work:

1. [docs/README.md](README.md)
2. [docs/AGENT_INDEX.md](AGENT_INDEX.md) if the consumer is an AI agent
3. [docs/SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md)
4. [docs/IMPLEMENTATION_MAP.md](IMPLEMENTATION_MAP.md)
5. the relevant domain router under `docs/`
6. relevant `openspec/specs/` only for stable contract semantics
7. `openspec/changes/<active-change>/` only when explicitly in scope
8. `research/` for active research interpretation and continuation context
9. `progress/` only for explicit legacy evidence reconstruction before
   migration into `research/`

## Authoring Rules

- Do not duplicate stable contracts across multiple router pages.
- Put stable workflows in `docs/`.
- Put dated evidence, investigations, negative results, audits, and research
  continuation records in `research/`.
- Do not add new notes under `progress/`; migrate or synthesize old material
  into `research/` instead.
- Remove obsolete paths instead of preserving compatibility stubs.
- Prefer one canonical page per question:
  - data contract -> `docs/data/CONTRACT.md`
  - data preparation -> `docs/data/PREPARATION.md`
  - packing policy -> `docs/data/PACKING.md`
  - Stage-1 objective/status -> `docs/training/STAGE1_OBJECTIVE.md`
  - Stage-2 runbook -> `docs/training/STAGE2_RUNBOOK.md`
  - evaluation contract -> `docs/eval/CONTRACT.md`
  - evaluation workflow -> `docs/eval/WORKFLOW.md`
  - artifacts and provenance -> `docs/ARTIFACTS.md`

## Promotion Rule

Promote or migrate a legacy `progress/` note when all of the following are true:

- it is no longer tied to one dated run or diagnosis
- it defines the current recommended workflow
- people would reasonably expect it to be the first page they open

Move it to `docs/` only when it defines stable current behavior. Otherwise
synthesize it into `research/` and leave `progress/` as legacy provenance.

Leave a topic only in legacy `progress/` when it is old material that has not yet
been migrated and is primarily:

- an experiment log
- a benchmark report
- a diagnosis or audit
- long-form design history
