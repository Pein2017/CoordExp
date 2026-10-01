---
doc_id: docs.branch-and-worktree-policy
layer: docs
doc_type: workflow
status: canonical
domain: repo
summary: Retained branches, physical worktrees, and Codex-session routing for CoordExp.
tags: [git, branches, worktrees, codex, coordexp-infras]
updated: 2026-10-01
---

# Branch And Worktree Policy

coordexp-infras is now the canonical implementation on repository `main`.

## Canonical routing

Keep exactly these long-lived local checkouts and their matching branches:

| Branch | Physical checkout | Role |
|---|---|---|
| `main` | `/data/CoordExp` | Stable source and integration owner |
| `coordexp-infras` | `/data/CoordExp/.worktrees/coordexp-infras` | Infrastructure development and its runs |
| `research-probes` | `/data/CoordExp/.worktrees/research-probes` | Canonical research source, records and runs |
| `research-probes-web-codex` | `/data/CoordExp/.worktrees/research-probes-web-codex` | Research development and its runs |

The user retired the temporary main-runs, start-loss-benchmark and detached
WebCodex checkouts. Do not recreate a generic runtime or archival worktree.
Temporary task branches/worktrees require an explicit need and are retired after
their useful changes and necessary assets reach one of the retained owners.

For default source or implementation questions, inspect `main` at
`/data/CoordExp`. For runs, enter the actual owning retained worktree;
infrastructure examples use `coordexp-infras`. Relative artifact paths resolve
from that physical checkout. Root `/data/CoordExp/outputs/` is shared-asset
retention, not a run destination; see [storage policy](OUTPUT_STORAGE_POLICY.md).
There is no fallback directory for unknown legacy payloads.

Promote validated infrastructure commits into main explicitly. These branches
may differ: running from coordexp-infras does not imply execution of main's
exact source. Provenance records the actual producer checkout/commit and input
qualification. Moving an old run or integrating source does not authorize a
new model launch.

## Codex sessions and task worktrees

Codex session transcripts, archived sessions, attachments, and app-managed
session indexes are historical or runtime-owned state. They may contain old
task-branch or worktree descriptions from when Swift was a candidate worktree.
Do not rewrite those transcripts to make past conversations appear current.

For a new or continued task, verify the live checkout with:

```bash
git branch --show-current
git worktree list --porcelain
git rev-parse --show-toplevel
```

The stable implementation checkout should report branch `main` and path
`/data/CoordExp`; the active development checkout should report branch
`coordexp-infras` and path `/data/CoordExp/.worktrees/coordexp-infras`. A Codex
task that still displays an older task branch is stale app-owned metadata;
starting from or sending a new message in the intended live checkout should
refresh that association. Historical session content remains unchanged by
design.

## Legacy and upstream wording

References to `ms-swift`, `/data/ms-swift`, `src.sft`, legacy config roots, and
old mainline launchers are valid only when explicitly labeled as upstream
dependency, compatibility, historical evidence, or archive material. They must
not be presented as the current CoordExp entrypoint.

The current Swift entrypoints are documented in
[`coordexp_infras.md`](coordexp_infras.md) and use `src/train.py`, `src/infer.py`,
`src/inference/`, and `src/eval/detection_consumer.py`.
