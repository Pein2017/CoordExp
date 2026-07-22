---
name: project-memory
description: Maintain the experimental repository-local project memory when long-running research, design, implementation, debugging, or cross-session work produces context that future agents should recall, checkpoint, revise, prune, or reconstruct. Use implicitly for non-trivial continuity-sensitive work when the repository contains memories/config.yaml, and explicitly when the user asks to create, recover, or manage project memory.
---

# Project Memory

Treat `<project-root>/memories/` as concise, agent-facing continuity written in
natural language. It is working memory, not a replacement for source code,
research evidence, documentation, specifications, or other formal project
authorities.

This skill and the repository `memories/` feature are experimental. Improve
their trigger rules and structure from observed use. Do not treat the current
workflow as a stable compatibility contract.

## Automatic Maintenance Loop

For a non-trivial continuity-sensitive task in a repository containing
`memories/config.yaml`, let the main agent use this skill without waiting for a
user reminder.

At task start, recall memory once. During work, mark memory as needing an
update when the live goal, durable user decision, supported or rejected
explanation, claim boundary, blocker, long-running continuation state, or next
action changes. Do not write immediately after every observation.

Flush the pending update at a meaningful boundary: after freezing a plan,
closing an analysis, successfully launching or pausing a costly operation,
changing direction, or before handoff, compaction, stop, or the final response.
Write at most once per logical transition unless interruption risk requires an
earlier operational checkpoint.

Choose the smallest write:

- update only `current.md` for changed live state or continuation steps;
- add a note and refresh `current.md` for durable reasoning, evidence, user
  steering, a rejected path, or a changed interpretation;
- curate or merge notes when a phase closes and existing memory has become
  stale or duplicative.

Checkpoint only when project state changed and omitting the change could make
a future agent choose incorrectly or pay substantial rediscovery cost. Skip
routine commands, ordinary test passes, per-step progress, utilization, short
estimates, and facts cheaply recovered from authoritative artifacts.

## Recall

1. Resolve the project root from Git, falling back to the current directory.
2. Read `memories/config.yaml`, `memories/README.md`, and
   `memories/current.md`.
3. Search `memories/notes/` only for material relevant to the active task.

Recall is complete when the active goal, important prior reasoning, current
uncertainties, and next continuation point are clear without loading the whole
memory tree.

Do not repeat full recall on every turn in the same logical task. Re-read only
when the task changes or another actor may have updated the project. Treat live
process, branch, artifact, and worktree statements as potentially stale;
cheaply verify them before acting. Use a `Last verified` timestamp or an
equivalent scope statement for operational claims in `current.md`.

If the repository has no `memories/config.yaml`, initialize memory only when
the user asks for it. Otherwise leave that repository unchanged.

## Checkpoint

Use semantic judgment. Checkpoint when a development would help a future agent
continue correctly: a direction changes, a durable user judgment appears, an
experiment or implementation changes expectations, a rejected path would
otherwise be repeated, or work is about to hand off, compact, or stop.

Write a natural-language note under `memories/notes/` and refresh
`memories/current.md` when the live project state changed. Include source
handles and uncertainty in prose. Prefer a useful explanation over a rigid
taxonomy. Skip routine commands, transient status, and facts that are cheap to
rediscover.

When a formal result or contract already exists, leave the complete evidence
there. Memory should record the decision-relevant conclusion, why it changes
future work, the source handle, remaining uncertainty, and where to continue.
For a long-running operation, record only the recovery-critical state and omit
process identifiers, momentary utilization, and estimates unless they are
necessary to resume safely.

A checkpoint is complete when a future agent can understand what changed, why
it matters, what remains uncertain, and where to continue.

## Curate

Agents have full create, read, update, and delete access inside `memories/`.
Merge duplicates, rewrite misleading summaries, and delete stale, rejected, or
superseded material when it no longer helps. Preserve a rejected idea when its
reasoning prevents repeated work. Git history provides rollback; the live
memory tree should favor relevance over accumulation.

Keep `current.md` compact and current. Keep detailed history in notes. When
multiple agents are active, let the main thread or an explicitly designated
consolidator rewrite `current.md`; other agents should produce bounded notes or
proposals for consolidation.

Aim for roughly one or two readable pages in `current.md`, not an append-only
log. It should answer the active objective, latest decision-relevant evidence,
live blocker or continuation state, unresolved questions, immediate next
actions, and minimum reading path. Rewrite stale sections instead of adding
contradictory updates.

Curation is complete when the live memory is coherent, searchable, and free of
obvious stale duplication.

## Recover

For a large historical transcript, stream it in bounded chronological spans.
Read user and assistant messages plus compact event summaries first; inspect
raw tool output only when a retained claim needs it. Write a few natural
chapters that follow the work's actual transitions, then reconstruct
`current.md` as the best approximate terminal state. Mark uncertainty and
historical reconstruction explicitly.

Recovery is complete when the useful history and final working state are
available without requiring future agents to reopen the full transcript.

## Authority And Git Boundary

Memory may link to formal project knowledge but does not silently promote into
it. Propose formal updates separately. Memory maintenance does not authorize
automatic staging, commits, or publication. Preserve unrelated working-tree
changes.
