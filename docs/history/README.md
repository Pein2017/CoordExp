---
doc_id: docs.history.index
layer: docs
doc_type: history-router
status: historical-router
domain: repo
summary: Router for non-normative CoordExp documentation history kept outside current behavior docs.
tags: [history, archive, docs]
updated: 2026-08-17
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

## Repo-root historical records moved here (2026-08-17)

Unreferenced dated files previously left at the repository root, moved here for
provenance without losing evidence:

- `2026-03-10-coco-test-benchmark-handoff.md`: COCO test-dev submission working
  brief, superseded by `docs/eval/COCO_TEST_SUBMISSION.md` routing;
- `2026-05-31-self-distillation-prompt.md`: Codex self-distillation workflow
  prompt, replaced by the packaged `workflow-self-distillation` skill;
- `2026-06-02-dual-environment-sync-notes.md`: pein-train/test-train dual
  environment SSH/container operational notes;
- `research-intake/2026-06-23-autoregressive-binding-template-study-synthesis-report.md`:
  binding-template study synthesis report whose raw evidence is ingested under
  `research-intake/2026-07-01-autoregressive-binding-template-study/`.
