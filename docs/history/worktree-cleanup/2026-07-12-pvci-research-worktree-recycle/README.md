---
doc_id: docs.history.worktree-cleanup.2026-07-12-pvci-research-worktree-recycle
layer: docs
doc_type: history-intake
status: preserved
domain: research
summary: Preservation bundle for the temporary painted-GT and causal-proposal-bridge research worktrees.
tags: [history, worktrees, pvci, painted-gt, causal-proposal-bridge]
updated: 2026-07-12
---

# PVCI Research Worktree Recycle

This bundle preserves durable research evidence from two temporary worktrees
before their one-off implementation code and caches are discarded:

- `69ed`: `codex/continue-handoff-session` at `2fde14b37f24`
- `eb1c`: `codex/qwen3-vl-painted-gt-transcription-probe` at `f90381b15e6d`

Use [manifest.tsv](manifest.tsv) for source-path, hash, snapshot, and promotion
records. Use [local_artifacts.tsv](local_artifacts.tsv) for the seven locally
available `69ed` probe artifacts.

The curated research lineage is available at
[`research/ideas/qwen3-vl-painted-gt-transcription-probe/`](../../../../research/ideas/qwen3-vl-painted-gt-transcription-probe/).
Raw historical planning documents remain under `snapshots/`; they were not
promoted into current `docs/` or stable specs.

No source, config, test, OpenSpec change, checkpoint cache, or temporary model
implementation was imported from either branch.
