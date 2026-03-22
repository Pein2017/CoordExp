# Task Completion Checklist (Memory)

Role separation:
- Memory role: minimal done-definition before closing an implementation turn.
- Canonical docs: `docs/IMPLEMENTATION_MAP.md`, the relevant domain runbook under `docs/`, and `openspec/specs/` when semantics change.
- Update trigger: when validation expectations or workflow contracts change.

Pre-close sanity:
- Re-check the change against `.serena/memories/runtime_footguns.md`.
- For data, prompt, or template changes, validate one JSONL sample and inspect one rendered example.
- For behavior changes, run the narrowest targeted test or smoke path from `docs/IMPLEMENTATION_MAP.md`.

Closure:
- If user-facing defaults, artifact names, entrypoints, or recommended workflows changed, update `docs/`.
- If routing or architecture entrypoints changed, also update `docs/catalog.yaml`, `docs/AGENT_INDEX.md`, and the CoordExp retrieval memories/skills.
- If docs, memories, or skills changed, validate relative links and referenced paths before handoff.
- If stable behavior, config contracts, or metric semantics changed, update OpenSpec too.
- In the final handoff, state what was verified and what remains unverified.
