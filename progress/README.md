---
doc_id: progress.index
layer: progress
doc_type: router
status: legacy-router
domain: research-history
summary: Human router for historical directions, diagnostics, audits, benchmarks, explorations, and pretraining history.
tags: [progress, history, research]
updated: 2026-06-15
---

# Progress Index

This directory is the legacy/deprecated historical and evidence archive for
CoordExp.

Current behavior belongs in `docs/`. New research interpretations, negative
results, decision records, implementation notes, and continuation context belong
in `research/`. Do not add new notes here; migrate or synthesize useful legacy
material into `research/`.

The top-level `progress/` directory stays only so old links keep resolving.
Historical notes remain inside the category folders, but those folders are
read-only legacy provenance unless the user explicitly asks for migration.

## Directory Layout

- [progress/directions/README.md](directions/README.md)
  - historical research directions and Stage-2 design lineage
- [progress/diagnostics/README.md](diagnostics/README.md)
  - failure analyses, mechanism studies, threshold sweeps, and operator notes
- [progress/audits/README.md](audits/README.md)
  - structured review notes and decision audits
- [progress/benchmarks/README.md](benchmarks/README.md)
  - measured results, checkpoint comparisons, and evaluation sweeps
- [progress/explorations/README.md](explorations/README.md)
  - architecture, infrastructure, and implementation-planning explorations
- [progress/pretrain/README.md](pretrain/README.md)
  - Stage-1 foundation history and early pretraining evidence
- [progress/handoffs/](handoffs/)
  - historical handoff notes preserved for continuation provenance

## Legacy Routing

- old historical direction:
  - start with [progress/directions/README.md](directions/README.md)
- mechanism or failure diagnosis:
  - start with [progress/diagnostics/README.md](diagnostics/README.md)
- measured score comparisons:
  - start with [progress/benchmarks/README.md](benchmarks/README.md)
- repo/runtime architecture history:
  - start with [progress/explorations/README.md](explorations/README.md)
- Stage-1 background:
  - start with [progress/pretrain/README.md](pretrain/README.md)
- continuation provenance:
  - start with `progress/handoffs/` when a branch handoff is explicitly relevant

Use [progress/index.yaml](index.yaml) only when you need the machine-readable
legacy category map.

## Human Read Order

1. [docs/PROJECT_CONTEXT.md](../docs/PROJECT_CONTEXT.md)
2. [docs/SYSTEM_OVERVIEW.md](../docs/SYSTEM_OVERVIEW.md)
3. the relevant current docs router under `docs/`
4. [research/index.md](../research/index.md)
5. [progress/index.yaml](index.yaml) only for legacy evidence not yet migrated
6. the category router and old note you need for provenance

## Rules

- Do not treat `progress/` as normative.
- Do not add new `progress/` records.
- Prefer `docs/` for current workflows and stable interfaces.
- Prefer `research/` for active research interpretation and continuation
  context.
- Use `progress/` only when you need legacy:
  - historical derivation
  - experiment evidence
  - audits or diagnostics
  - benchmark context
