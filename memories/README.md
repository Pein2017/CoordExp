# Project Memory

> **Experimental repository feature.** Its trigger rules, layout, and
> maintenance behavior are expected to evolve from real use. Memory is not a
> stable compatibility contract.

This directory is concise, agent-facing continuity for work that spans tasks,
sessions, or agents. It is written primarily in natural language so another
agent can recover the project's reasoning without reopening full transcripts.

Memory is not a formal project authority. In this repository:

- `research/` owns research interpretation and executed evidence;
- `docs/` owns current behavior and operator guidance;
- `openspec/` owns stable compatibility-sensitive contracts;
- source code and artifacts remain authoritative for their own behavior and
  evidence.

Memory may summarize and link to those sources. It must not silently replace or
rewrite them.

## Files

- `current.md` is the short starting point for the next agent. Rewrite it when
  the live goal, beliefs, decisions, blockers, or next action change.
- `notes/` preserves useful reasoning and history in whatever natural structure
  fits the material.
- `template.md` is optional guidance, not a required schema.
- `config.yaml` states the operating boundary.

Agents may create, read, update, merge, and delete anything in this directory.
Prune stale or misleading material instead of accumulating sediment. Preserve
rejected reasoning when it will prevent the same dead end from being explored
again. Git history is the recovery path for deleted or rewritten memory.

For non-trivial continuity-sensitive work, the main agent should recall memory
without waiting for a user reminder, track whether project state changed, and
checkpoint once at the next meaningful boundary. Update only `current.md` for
recovery-critical live state. Add a note when durable reasoning, evidence,
user steering, or a rejected path changed future decisions. Other agents must
not compete to rewrite `current.md`.

Operational statements in `current.md` should include `Last verified` or an
equivalent scope marker and must be rechecked before use. Avoid process
identifiers, momentary utilization, short estimates, and other state that
becomes stale faster than it helps continuation.

Do not copy raw transcripts, large tool outputs, caches, indexes, credentials,
or secrets here. Record a source handle and retrieve the original only when
needed.
