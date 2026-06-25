---
name: grill-me-with-docs
description: Use when the user wants a rigorous CoordExp discussion that pressure-tests a research idea, experiment plan, model change, workflow, or architecture decision and records durable outcomes.
---

# Grill With Docs

Use this for a longer pressure-testing session where discussion should leave
repo-local records, not just chat context. For a short direct challenge loop,
use `grill-me`; for architecture-specific refactor discovery, use
`improve-codebase-architecture`.

## Operating Loop

- Start from the repo route: `docs/AGENT_INDEX.md`, `docs/catalog.yaml`, then the relevant docs, specs, configs, artifacts, or `progress/` notes.
- Walk the decision tree branch by branch, resolving dependencies between decisions in order.
- Batch simple, independent questions and recommendations when they can be answered together without changing upstream meaning.
- Ask exactly one question at a time only for crucial or high-impact decisions that gate later branches or change research meaning, reproducibility, eval validity, cost, compatibility, durable records, or next action.
- Include your recommended answer for every question or batched item, and make clear when "all yes" is a valid response.
- If the repo or artifacts can answer the question, inspect them before asking.
- Separate hypothesis, mechanism, implementation plan, experiment result, interpretation, and stable contract.
- Stress-test the plan against concrete scenarios: data edge cases, geometry/order preservation, config inheritance, eval validity, artifact completeness, cost, and rollback.
- Record resolved outcomes promptly using [RECORDING.md](RECORDING.md).

## Domain Awareness

- Use CoordExp docs, specs, configs, tests, manifests, artifacts, and progress notes as the discussion vocabulary.
- When the user uses a term that conflicts with existing repo language, call out the conflict immediately and ask which meaning should win.
- When language is vague or overloaded, propose a precise canonical term tied to the relevant code, config key, artifact, metric, or doc path.
- Cross-reference claims with code and artifacts. If the code says something different from the discussion, surface the contradiction as the next question.
- Do not import the official skill's root `CONTEXT.md` or generic ADR layout unless the repo already uses that surface for the topic.

## What To Challenge

Prefer questions that expose a real fork:

- what would change research meaning, reproducibility, eval validity, cost, or compatibility;
- which artifact, config, checkpoint, metric file, or test becomes source of truth;
- which evidence scope would count as `tiny`, `smoke`, `val200`, `proxy`, `partial`, or `full`;
- which default must remain backward-compatible;
- what belongs in docs, progress notes, OpenSpec, configs, tests, or executable manifests.

For simple independent items, batch them as a short checklist grouped by surface
such as naming, config defaults, metrics, artifacts, docs, or verification. Move
back to one-at-a-time only when the answer must be known before the next
question is meaningful.

Avoid questions whose answer is already present in `docs/`, `openspec/specs/`,
configs, tests, artifacts, or the current conversation.

## Recording Discipline

Write only when the discussion resolves something durable. Do not create
placeholder docs, generic ADRs, root `CONTEXT.md`, or duplicate summaries.

Records should be compact and link-rich: decision, rationale, consequence,
exact paths or artifacts, evidence scope, and open follow-up.

When a batch is approved, record the resolved outcomes in one compact pass
rather than creating one durable note per micro-decision.

Offer a durable decision record only when the decision is hard to reverse,
surprising without context, and the result of a real trade-off. Record measured
evidence, stable behavior, and executable truth on the surfaces named in
[RECORDING.md](RECORDING.md).
