---
doc_id: docs.history.worktree-union.2026-06-20
layer: docs
doc_type: history-intake
status: raw-intake
domain: repo
summary: Raw Markdown union intake from linked CoordExp worktrees before progress/docs cleanup and OKF migration.
tags: [history, worktrees, docs, progress, intake]
updated: 2026-06-20
---

# Worktree Markdown Union Intake: 2026-06-20

This bundle preserves Markdown knowledge found in linked CoordExp worktrees
before the progress/docs cleanup and OKF migration program.

It is a raw intake, not a current-behavior document set.

## Scope

- Base checkout: `/data/CoordExp`
- Collection worktree:
  `/data/CoordExp/.worktrees/progress-okf-union-collection`
- Collection branch: `codex/progress-okf-union-collection`
- Included sources:
  - tracked Markdown changes at each linked worktree branch head, compared
    against current `main`
  - dirty and untracked Markdown files in each linked worktree
- Excluded sources:
  - non-Markdown files
  - generated artifacts, checkpoints, caches, and ignored data roots

## Manifest

Use [manifest.tsv](manifest.tsv) as the source of truth for this intake.

Each row records:

- classification
- source kind: `branch-head` or `worktree-dirty`
- source branch, HEAD, worktree, and original path
- source status and SHA-256
- whether the same path or same content already existed in current main
- snapshot path when the content was new to current main

Empty manifest values are recorded as `-`.

## Classification

- `same_path_identical`: the source Markdown is already present at the same
  relative path in current main.
- `content_present_elsewhere`: the source Markdown content already exists
  somewhere else in current main, usually under `docs/history/`.
- `new_content_new_path`: the content was not present in current main, and the
  source path did not exist in current main.
- `same_path_divergent_new_content`: the source path exists in current main,
  but this worktree version contains different Markdown content.

Only `new_content_new_path` and `same_path_divergent_new_content` rows have
snapshots under [snapshots/](snapshots/).

## Intake Summary

- Manifest rows: 151
- Content already present elsewhere in current main: 29
- Same-path identical rows: 19
- New content at new paths: 42
- Same-path divergent new content: 61
- Snapshot Markdown files: 88

## Governance

Do not cite this bundle as current behavior.

Use it only to reconstruct branch provenance, recover missing progress notes,
or prepare the next cleanup and OKF migration phases. Promotion into live
`docs/`, `progress/`, or `openspec/` must happen through the follow-up cleanup
and alignment process.
