---
name: grill-me
description: Use when the user explicitly asks for a short grill, stress test, pressure test, or decision-sharpening loop without durable repo recording.
---

# Grill Me

Use this as a focused clarification loop. For longer docs-aware sessions where
resolved discussion must update durable repo records, use
`grill-me-with-docs`.

## Core Loop

- Interview the user relentlessly until the plan, design, or decision reaches
  shared understanding.
- Walk the decision tree branch by branch, resolving dependencies between
  decisions in order.
- Ask exactly one question at a time, and include your recommended answer.
- If repo context can answer the question, inspect the repo or artifacts first
  instead of asking.
- Stop grilling once the remaining uncertainty no longer changes the decision,
  evidence plan, compatibility story, or next action.

## CoordExp Ground Rules

- Distinguish hypothesis, implementation plan, experiment result, interpretation, and stable contract.
- Keep super-power plans, OpenSpec, docs, progress notes, and repo artifacts in their existing ownership lanes.
- Do not rely on the context window as the only record when the decision will steer future work.

## Question Style

Prefer questions that expose a real fork:

- what would change research meaning, reproducibility, eval validity, cost, or compatibility;
- what evidence would make the user accept or reject the idea;
- what artifact, config, checkpoint, or metric file would become the source of truth;
- what needs to remain backward-compatible.

Avoid questions whose answer is already in `docs/`, configs, tests, artifacts, or the current conversation.

## Recording Boundary

Default to chat-only outcomes. If the discussion resolves durable repo state,
switch to `grill-me-with-docs` and its `RECORDING.md` guidance before writing.
If the right surface is unclear, ask one short question with a recommended
target.
