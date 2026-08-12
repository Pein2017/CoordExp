---
doc_id: docs.history.index
layer: docs
doc_type: history-router
status: historical-router
domain: repo
summary: Router for non-normative CoordExp documentation history kept outside current behavior docs.
tags: [history, archive, docs]
updated: 2026-06-15
---

# Documentation History

This directory preserves historical plans, design specs, handoffs, and superseded notes that are useful for provenance but are not current behavior.

Use current docs first:

1. [../PROJECT_CONTEXT.md](../PROJECT_CONTEXT.md)
2. [../SYSTEM_OVERVIEW.md](../SYSTEM_OVERVIEW.md)
3. [../IMPLEMENTATION_MAP.md](../IMPLEMENTATION_MAP.md)
4. the relevant current domain router under `docs/`

Only use this archive when reconstructing history, reviewing why an implementation path existed, or importing lessons into current docs.

## Contents

- [architecture/README.md](architecture/README.md): old architecture reviews,
  simplification proposals, refactoring programs, blueprints, and decision logs
- [superpowers/README.md](superpowers/README.md): dated agent implementation plans, design specs, and handoffs
- [worktree-union/README.md](worktree-union/README.md): raw Markdown union intakes from linked worktrees before cleanup and migration
- [worktree-cleanup/README.md](worktree-cleanup/README.md): preservation receipts for recycled experimental worktrees
- [cache-retirement/](cache-retirement/): compact receipts for intentionally
  removed disposable cache payloads
- [research-intake/](research-intake/): raw research-note intakes used as provenance for synthesized `research/` reading paths
- [training/](training/): superseded training design notes moved out of the current training docs layer
