---
name: project-memory
description: Maintain concise repository-local continuity when a long-running task changes durable decisions, evidence, blockers, or the next continuation point.
---

# Project Memory

Treat project memory as **continuity**, not authority. Formal evidence remains
in code, artifacts, research results, operator guidance, and stable contracts.

## Recall

When project memory is configured and the task is continuity-sensitive:

1. Read the memory configuration and current-state entrypoint once.
2. Search notes only for the active question.
3. Verify live branch, worktree, process, and artifact claims before acting.

Recall is complete when the current objective, decision, uncertainty, and next
action are clear without loading the full memory tree.

## Checkpoint

Checkpoint only after a durable transition: changed goal or user decision,
evidence-backed conclusion, rejected explanation worth preserving, blocker,
costly operation state, claim boundary, or next discriminator.

- Update the configured current-state record for changed live state.
- Add a note only when reasoning or provenance would otherwise be lost.
- Link the owning result instead of copying its evidence.
- Skip routine commands, ordinary test passes, transient progress, and facts
  cheap to rediscover.

Checkpoint is complete when a fresh agent can tell what changed, why it matters,
what remains uncertain, and where to continue.

## Curate

Keep the current-state record compact and coherent. Rewrite stale state; merge
duplicates; remove misleading or superseded material while preserving rejected
reasoning that prevents repeated work. Only the main agent or designated
consolidator rewrites it during parallel work.

For historical recovery, synthesize one canonical transcript copy and mark
uncertainty. Do not inject duplicate transcripts or raw tool logs into live
memory.

## Boundary

Memory maintenance does not authorize formal guidance or specification updates,
staging, commits, publication, or destructive cleanup. If project memory is not
configured, initialize it only when the user asks.
