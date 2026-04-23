---
doc_id: progress.index
layer: progress
doc_type: router
status: canonical
domain: research-history
summary: Human router for historical directions, diagnostics, audits, benchmarks, explorations, and pretraining history.
tags: [progress, history, research]
updated: 2026-04-23
---

# Progress Index

This directory is the historical and evidence layer for CoordExp.

Current behavior belongs in `docs/`.
Historical motivation and empirical evidence belong here.

The top-level `progress/` directory should stay router-first. Historical notes
belong inside the category folders, with each folder exposing its own README as
the first entrypoint.

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

## Quick Routing

- current historical direction:
  - start with [progress/directions/README.md](directions/README.md)
- mechanism or failure diagnosis:
  - start with [progress/diagnostics/README.md](diagnostics/README.md)
- measured score comparisons:
  - start with [progress/benchmarks/README.md](benchmarks/README.md)
- repo/runtime architecture history:
  - start with [progress/explorations/README.md](explorations/README.md)
- Stage-1 background:
  - start with [progress/pretrain/README.md](pretrain/README.md)

Use [progress/index.yaml](index.yaml) when you want the machine-readable
category map.

## Human Read Order

1. [docs/PROJECT_CONTEXT.md](../docs/PROJECT_CONTEXT.md)
2. [docs/SYSTEM_OVERVIEW.md](../docs/SYSTEM_OVERVIEW.md)
3. [docs/training/STAGE2_DESIGN.md](../docs/training/STAGE2_DESIGN.md)
4. [progress/index.yaml](index.yaml)
5. the category router that matches your question
6. the specific historical note you need

## Rules

- Do not treat `progress/` as normative by itself.
- Prefer `docs/` for current workflows and stable interfaces.
- Use `progress/` when you need:
  - historical derivation
  - experiment evidence
  - audits or diagnostics
  - benchmark context
