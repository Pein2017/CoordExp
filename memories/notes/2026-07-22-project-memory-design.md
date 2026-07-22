# Natural-Language Project Memory Design

Source: side discussion derived from `handoff/customized-memory.md` on
2026-07-22.

The initial design considered a broad event-sourced memory system with typed
candidates, confidence scores, indexes, hooks, and promotion machinery. The
user narrowed the actual need: first make a useful memory for agents, with
research continuity as the highest priority and portability to software design,
implementation, and debugging when natural.

The chosen first version uses natural-language notes and one concise current
state. It deliberately avoids hard-coded categories such as finding,
hypothesis, derivation, or open question. Agents decide semantically when a
development will matter to future work. They may fully create, read, update,
merge, and delete memory content. Rejected material is retained only while its
reasoning remains useful; Git history provides recovery after pruning.

Memory cannot override the established value of `research/`, `docs/`,
OpenSpec, source code, or executed artifacts. It is continuity and reasoning,
not a new formal authority. Updates remain confined to `memories/` unless a
separate task explicitly authorizes promotion.

The first real test was an approximate reconstruction of the long main thread
with session identifier `019f4a19-d81c-75a2-84b0-2c20379e686e`. That
reconstruction and later live research checkpoints showed that the natural
language layout is sufficient, but manual user reminders are not a reliable
trigger.

The next experimental behavior is semantic, automatic maintenance by the main
agent. For a non-trivial continuity-sensitive task, the main agent recalls
memory once, tracks whether a goal, durable decision, evidence-backed
conclusion, claim boundary, blocker, costly continuation state, or next action
changed, and writes at the next meaningful boundary. Routine progress does not
qualify. A live-state change updates only `current.md`; durable reasoning also
creates or revises a note. Only the main agent or a designated consolidator
rewrites `current.md`.

This remains an experimental repository feature rather than a stable contract.
Its trigger quality, false-positive write rate, stale-state handling, and
curation behavior should be revised from future use. No daemon, hook, or
automatic summarization service is introduced in this version.
