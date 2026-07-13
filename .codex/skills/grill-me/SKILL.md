---
name: grill-me
description: Use when the user explicitly asks to grill, stress-test, or pressure-test a costly or hard-to-reverse CoordExp decision, optionally recording the outcome.
---

# Grill Me

Pressure-test only choices that are expensive, hard to reverse, or likely to
change research meaning, reproducibility, evaluation validity, compatibility,
cost, or implementation direction. Default to chat; write a durable record only
when explicitly requested or approved.

## Decision Ownership

Do the repository homework and recommend an answer. Do not quiz the user on
syntax, class layout, library trivia, or reversible implementation details.

The user owns algorithm/model semantics, data and geometry, targets/loss
normalization, training trade-offs, statistical assumptions, metric
interpretation, evidence thresholds, and expensive experiment scope. The agent
owns discoverable facts, implementation alternatives, risk analysis, and the
smallest verification path.

## Mode

Choose the mode up front:

- `record=chat`: default; write no files.
- `record=local`: record a resolved decision, evidence result, or terminology
  resolution only after explicit request or approval.

If recording was not requested, recommend a target and ask before writing.

## Loop

- Start from the repo route when local evidence matters: `docs/AGENT_INDEX.md`,
  `docs/catalog.yaml`, then relevant docs, specs, configs, artifacts, tests, or
  notes.
- Inspect repo evidence before asking about discoverable facts.
- Resolve one dependency at a time. Ask only when a wrong approval would be hard
  to revise, expensive to rerun, or likely to invalidate interpretation.
- Ask exactly one question at a time and wait for the answer before continuing.
  Attach a recommended answer and the consequence for meaning, evidence, cost,
  compatibility, or architecture.
- Separate hypothesis, mechanism, implementation plan, experiment result,
  interpretation, and stable contract.
- Stress-test against concrete CoordExp risks: geometry/order preservation,
  config inheritance, eval validity, artifact completeness, cost, rollback, and
  evidence scope.
- Stop when remaining uncertainty no longer changes the decision, evidence plan,
  compatibility story, or next action. End with exactly one next state:
  `drop`, `narrow`, `probe`, `build-probe`, `implement`, `document`, or
  `needs user decision`.
- Do not implement or launch a probe until the user confirms and asks to proceed.

## Question Style

Use this compact shape:

```text
Decision: what must be chosen now
Why it matters: the semantic, evidence, cost, or compatibility consequence
Recommendation: the agent's preferred answer and why
Question: one concrete choice for the user
```

Prefer forks about acceptance evidence, source-of-truth artifacts, experiment
scope (`tiny|smoke|val200|proxy|partial|full`), compatibility, or durable
carrier. If alternatives do not change a user-owned decision, choose the sound
implementation yourself.

## Recording

For `record=local`, read [recording.md](references/recording.md) after the
decision resolves. It selects the narrowest durable carrier and defines the
record template. Recording is complete only when the decision, evidence scope,
handles, next state, and carrier are explicit. Permission to record does not
authorize implementation.
