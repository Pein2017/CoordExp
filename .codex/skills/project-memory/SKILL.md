---
name: project-memory
description: Recall or maintain repository-local continuation state configured by memories/config.yaml. Not for Codex-managed memory or routine task progress.
---

# Project Memory

Treat project memory as **continuity**, not authority. Formal evidence remains
in code, artifacts, research results, operator guidance, and stable contracts.

## Scope resolution

Resolve `memories/config.yaml` and its entrypoints from the current `cwd`; use
another project root only when explicitly named. If unconfigured, stop this
skill's workflow and continue the task without initializing memory unless asked.
Do not create worktree-local `.codex` copies.

This store is distinct from `/data/CoordExp/.codex/memories`. For Codex-managed
memory, follow the injected memory instructions, including explicit user
authorization and the prescribed add-on-note gateway; never edit its generated
summary or index directly. This skill does not authorize those writes.

## Recall

When project memory is configured and the task is continuity-sensitive:

1. Read the configuration once to locate the current-state record and index.
2. Search those entrypoints by the active topic, path, or artifact; read only
   matching sections and one or two linked notes. Do not default to full-index
   or transcript reads. Widen only if a decision-bearing gap remains.
3. Verify live branch, worktree, process, and artifact claims before acting.

Recall is complete when the current objective, decision, uncertainty, and next
action are clear without loading the full memory tree.

## Checkpoint

Before any write, resolve the store's write authority and gateway and check its
pending transaction or lock state. Do not take over another task's transaction
or overwrite its content. If authority or transaction state is unclear, pause
only the affected write, ask the owner, and continue independent work. Do not
invent a transaction mechanism for a store that has none.

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

Within that write authority, keep current state compact. Rewrite stale state;
merge duplicates; remove misleading or superseded material while preserving rejected
reasoning that prevents repeated work. Only the main agent or designated
consolidator rewrites it during parallel work.

For historical recovery, synthesize one canonical transcript copy and mark
uncertainty. Do not inject duplicate transcripts or raw tool logs into live
memory.

## Boundary

Memory maintenance does not authorize formal guidance or specification updates,
staging, commits, publication, or destructive cleanup.
