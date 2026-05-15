---
name: grill-me
description: Use when the user explicitly asks to be grilled, stress-test a research idea, sharpen an experiment plan, pressure-test a design, or record decisions as discussion resolves.
---

# Grill Me

Use this as a focused clarification loop that leaves durable breadcrumbs as
decisions resolve. Do not rely on the context window as the only record.

## CoordExp Ground Rules

- Ask one question at a time and include your recommended answer.
- If repo context can answer the question, inspect the repo or artifacts first instead of asking.
- Distinguish hypothesis, implementation plan, experiment result, interpretation, and stable contract.
- Keep super-power plans, OpenSpec, docs, progress notes, and repo artifacts in their existing ownership lanes.
- Capture resolved decisions promptly in the right persistent surface.

## Question Style

Prefer questions that expose a real fork:

- what would change research meaning, reproducibility, eval validity, cost, or compatibility;
- what evidence would make the user accept or reject the idea;
- what artifact, config, checkpoint, or metric file would become the source of truth;
- what needs to remain backward-compatible.

Avoid questions whose answer is already in `docs/`, configs, tests, artifacts, or the current conversation.

## Recording Loop

After each resolved branch of the discussion, decide whether it produced a durable record:

- For implementation details, command plans, verification checklists, and branch-local handoff notes, update the active super-power plan/spec.
- For measured results, diagnostics, benchmark evidence, artifact guides, and empirical findings, update the relevant `progress/` note or create a dated note when warranted.
- For stable current behavior, update `docs/` using the repo routing in `docs/AGENT_INDEX.md` and `docs/catalog.yaml`.
- For stable compatibility contracts, use `openspec/specs/` only when the contract is truly normative and compatibility-sensitive.
- For executable truth, prefer repo configs, tests, scripts, manifests, and artifact paths over prose summaries.

If the right surface is unclear, ask one short question with a recommended target. Keep records link-rich and scoped: decision, rationale, consequence, exact paths or artifacts, and open follow-up.
