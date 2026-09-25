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

- `current.md` is a stable route to the research owners, not a live-state copy.
  Goals, boundaries, blockers and next actions belong to the selected unit state.
- `notes/` preserves useful reasoning and history in whatever natural structure
  fits the material.
- `template.md` is optional guidance, not a required schema.
- `config.yaml` states the operating boundary.

Read relevant notes when continuity matters. Writes require explicit user
authorization, consistent with `research/CONVENTIONS.md`; a normal closeout or
handoff is not an automatic memory-write trigger. Do not duplicate unit states,
acceptance ledgers or the research frontier here.

Retain non-reconstructible reasoning, user steering and rejected alternatives
with attributed sources and a historical/verification boundary. Useful scientific
methods and results belong with their research owner. Prune consumed transport
after reference checks; Git history retains old memory versions. Do not preserve
temporary process identifiers or utilization as durable continuity.

Do not copy raw transcripts, large tool outputs, caches, indexes, credentials,
or secrets here. Record a source handle and retrieve the original only when
needed.
