---
name: grill
description: Use when the user wants a rigorous CoordExp discussion that pressure-tests a research idea, experiment plan, model change, workflow, or architecture decision and records durable outcomes.
---

# Grill

Use this for a longer pressure-testing session where discussion should leave
repo-local records, not just chat context. For a short direct challenge loop,
use `grill-me`; for architecture-specific refactor discovery, use
`improve-codebase-architecture`.

## Operating Loop

- Start from the repo route: `docs/AGENT_INDEX.md`, `docs/catalog.yaml`, then the relevant docs, specs, configs, artifacts, or `progress/` notes.
- Ask one question at a time and include your recommended answer.
- If the repo or artifacts can answer the question, inspect them before asking.
- Separate hypothesis, mechanism, implementation plan, experiment result, interpretation, and stable contract.
- Stress-test the plan against concrete scenarios: data edge cases, geometry/order preservation, config inheritance, eval validity, artifact completeness, cost, and rollback.
- Record resolved outcomes promptly using [RECORDING.md](RECORDING.md).

## What To Challenge

Prefer questions that expose a real fork:

- what would change research meaning, reproducibility, eval validity, cost, or compatibility;
- which artifact, config, checkpoint, metric file, or test becomes source of truth;
- which evidence scope would count as `tiny`, `smoke`, `val200`, `proxy`, `partial`, or `full`;
- which default must remain backward-compatible;
- what belongs in docs, progress notes, OpenSpec, configs, tests, or executable manifests.

Avoid questions whose answer is already present in `docs/`, `openspec/specs/`,
configs, tests, artifacts, or the current conversation.

## Recording Discipline

Write only when the discussion resolves something durable. Do not create
placeholder docs, generic ADRs, root `CONTEXT.md`, or duplicate summaries.

Records should be compact and link-rich: decision, rationale, consequence,
exact paths or artifacts, evidence scope, and open follow-up.
