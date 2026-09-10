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

For a current question, use its named source/contract or
[AGENT_INDEX.md](../AGENT_INDEX.md) to locate the owner. There is no prerequisite
chain of architecture pages to read before consulting a dated record.

Only use this archive when reconstructing history, reviewing why an implementation path existed, or importing lessons into current docs.

## Contents

- [architecture/README.md](architecture/README.md): old architecture reviews,
  simplification proposals, refactoring programs, blueprints, and decision logs
- [superpowers/README.md](superpowers/README.md): dated agent implementation plans, design specs, and handoffs
- [worktree-union/README.md](worktree-union/README.md): raw Markdown union intakes from linked worktrees before cleanup and migration
- [worktree-cleanup/README.md](worktree-cleanup/README.md): preservation receipts for recycled experimental worktrees
- [research-intake/](research-intake/): raw research-note intakes used as provenance for synthesized `research/` reading paths

- [Historical engineering principles](engineering/2026-09-09-agent-engineering-constitution.md): superseded July guidance; current code style and agent authority remain separate owners
- [Legacy evaluation reference](evaluation/2026-09-09-legacy-eval-reference.md): old MS-Swift commands, artifact shapes and diagnostic flows, separated from the current Swift runbook
