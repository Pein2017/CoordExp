---
doc_id: progress.index
layer: progress
doc_type: router
status: canonical
domain: research-history
summary: Human router for historical directions, diagnostics, audits, benchmarks, and exploratory notes.
tags: [progress, history, research]
updated: 2026-03-11
---

# Progress Index

This directory is the historical and evidence layer for CoordExp.

Current behavior belongs in `docs/`.
Historical motivation and empirical evidence belong here.

## Directory Layout

- `progress/directions/`
  - long-form design history and superseded research directions
- `progress/diagnostics/`
  - failure analyses, threshold studies, and operator notes used to explain behavior
- `progress/audits/`
  - structured review notes and decision audits
- `progress/benchmarks/`
  - measured results, checkpoint comparisons, and evaluation sweeps
- `progress/explorations/`
  - exploratory requests and infrastructure probes
- `progress/pretrain/`
  - Stage-1 background and pretraining history

Use the folder routers when the split is unclear:

- [progress/diagnostics/README.md](diagnostics/README.md)
- [progress/benchmarks/README.md](benchmarks/README.md)

## Human Read Order

1. [docs/PROJECT_CONTEXT.md](../docs/PROJECT_CONTEXT.md)
2. [docs/SYSTEM_OVERVIEW.md](../docs/SYSTEM_OVERVIEW.md)
3. [docs/training/STAGE2_DESIGN.md](../docs/training/STAGE2_DESIGN.md)
4. [progress/index.yaml](index.yaml)
5. the specific historical note you need

## Rules

- Do not treat `progress/` as normative by itself.
- Prefer `docs/` for current workflows and stable interfaces.
- Use `progress/` when you need:
  - historical derivation
  - experiment evidence
  - audits or diagnostics
  - benchmark context
