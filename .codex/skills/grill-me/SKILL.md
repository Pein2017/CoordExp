---
name: grill-me
description: Use when the user explicitly asks for a grill, stress test, pressure test, decision-sharpening loop, or rigorous CoordExp discussion, with optional chat-only or local-record modes.
---

# Grill Me

Pressure-test an idea, plan, result, workflow, or architecture decision. Default
to plain chat. Record local context only when the user asks for it or approves a
durable record.

## Mode

Choose the mode up front:

- `record=chat`: default. No files are written. Return the useful pressure test
  inline.
- `record=local`: write compact durable context after a real decision, evidence
  result, or terminology resolution. Use only when explicitly requested or
  approved.

If the user says "with docs", "record this", "local context", "durable", or
names a target path, use `record=local`. If recording might be useful but was not
requested, recommend the target and ask before writing.

## Loop

- Start from the repo route when local evidence matters: `docs/AGENT_INDEX.md`,
  `docs/catalog.yaml`, then relevant docs, specs, configs, artifacts, tests, or
  notes.
- Walk the decision tree branch by branch, resolving dependencies in order.
- Inspect repo evidence before asking questions when the answer is discoverable.
- Batch independent questions with recommended answers; ask exactly one question
  only when the answer gates research meaning, reproducibility, eval validity,
  cost, compatibility, durable records, or the next action.
- Separate hypothesis, mechanism, implementation plan, experiment result,
  interpretation, and stable contract.
- Stress-test against concrete CoordExp risks: geometry/order preservation,
  config inheritance, eval validity, artifact completeness, cost, rollback, and
  evidence scope.
- Stop once the remaining uncertainty no longer changes the decision, evidence
  plan, compatibility story, or next action.

## Question Style

Prefer questions that expose a real fork:

- what evidence would accept or reject the idea;
- which artifact, config, checkpoint, metric file, or test is source of truth;
- which scope is `tiny`, `smoke`, `val200`, `proxy`, `partial`, or `full`;
- which default or contract must remain backward-compatible;
- what belongs in docs, research notes, OpenSpec, configs, tests, or manifests.

Avoid questions already answered by current docs, configs, tests, artifacts, or
the conversation.

## Recording

For `record=local`, write only resolved outcomes. Do not create placeholder
docs, generic ADRs, root `CONTEXT.md`, or duplicate summaries.

Use the narrowest durable surface:

- `research/`: new research ideas, investigations, interpretations, negative
  results, mechanisms, and continuation context.
- `docs/`: stable current behavior, recommended workflows, entrypoints,
  artifact names, metric meaning, and operator-facing architecture.
- `openspec/specs/`: normative compatibility-sensitive contracts only.
- `openspec/changes/<active-change>/`: active OpenSpec work only when explicitly
  in scope.
- configs, tests, scripts, manifests, and artifact paths: executable truth.
- `progress/`: legacy evidence only, unless the current docs explicitly route
  the topic there.

Use a compact record shape:

```md
## Decision
{What is now true or planned.}

## Rationale
{Why this trade-off won.}

## Consequence
{What must change, stay compatible, or be verified.}

## Evidence
- Scope: `tiny|smoke|val200|proxy|partial|full|none-yet`
- Handles: `{configs, artifacts, tests, metrics, docs, commits}`
```

After recording, continue the loop unless the user asked to pause, stop, or only
record.
