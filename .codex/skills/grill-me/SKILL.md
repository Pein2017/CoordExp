---
name: grill-me
description: Use when the user explicitly asks for a grill, stress test, pressure test, or rigorous CoordExp decision discussion where approvals may be expensive or hard to revise, with optional chat-only or local-record modes.
---

# Grill Me

Pressure-test an idea, plan, result, workflow, or architecture decision. Focus
only on decisions that are expensive, hard to reverse, hard to compensate after
approval, or likely to change research meaning, reproducibility, evaluation
validity, compatibility, cost, or implementation direction. Default to plain
chat. Record local context only when the user asks for it or approves a durable
record.

## Philosophy And Agency

Grilling is a coordination discipline, not an examination of the user's coding
knowledge. The agent must do the repository homework, translate implementation
facts into architectural consequences, and provide a recommended answer. The
user should not be forced to decide syntax, class layout, library trivia, or
other reversible implementation details.

The user owns decisions that change research meaning: algorithmic behavior,
model forward semantics, data construction and geometry, targets and loss
normalization, optimization/training trade-offs, statistical assumptions,
metric interpretation, evidence thresholds, and expensive experiment scope.
The agent owns discoverable facts, implementation alternatives, risk analysis,
and the smallest verification path.

Walk the decision tree together. Resolve one dependency before exposing the
next. The goal is shared understanding: the user can state what scientific or
architectural choice is being made and why, while the agent can implement it
without inventing semantics.

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
- Ask only load-bearing questions: decisions whose approval would be hard to
  revise, expensive to rerun, difficult to compensate in analysis, or likely to
  create confusing artifacts if wrong.
- Do not ask about reversible naming, local implementation details, cosmetic
  preferences, small config defaults, or choices that can be cheaply adjusted
  later unless they affect artifact identity, metric validity, stable contracts,
  or research interpretation.
- Ask exactly one question at a time and wait for the answer before continuing.
  Never batch decisions, even when they appear independent: an answer can
  change the framing, priority, or vocabulary of the next branch.
- Attach a recommended answer to every question. Explain the consequence in
  terms the user controls: research meaning, evidence, cost, compatibility, or
  architecture. Do not outsource technical due diligence to the user.
- Separate hypothesis, mechanism, implementation plan, experiment result,
  interpretation, and stable contract.
- Stress-test against concrete CoordExp risks: geometry/order preservation,
  config inheritance, eval validity, artifact completeness, cost, rollback, and
  evidence scope.
- Stop once the remaining uncertainty no longer changes the decision, evidence
  plan, compatibility story, or next action. End with exactly one next state:
  `drop`, `narrow`, `probe`, `build-probe`, `implement`, `document`, or
  `needs user decision`.
- Do not enact the plan during the grilling session. A next state such as
  `implement` or `probe` is a recommendation until the user explicitly confirms
  the shared understanding and asks to proceed.

## Question Style

Prefer questions that expose a real fork:

- what evidence would accept or reject the idea;
- what implementation must exist before that evidence can be collected;
- which artifact, config, checkpoint, metric file, or test is source of truth;
- which scope is `tiny`, `smoke`, `val200`, `proxy`, `partial`, or `full`;
- which default or contract must remain backward-compatible;
- what belongs in docs, research notes, OpenSpec, configs, tests, or manifests.

Avoid questions already answered by current docs, configs, tests, artifacts, or
the conversation. Also avoid questions where a reasonable default can be
changed cheaply after seeing evidence.

Before asking, apply this filter:

```text
Would a wrong approval be hard to revise, expensive to rerun, or likely to
invalidate interpretation?
```

If no, choose a reasonable default and continue.

When a question is needed, keep it compact but include four things:

```text
Decision: what must be chosen now
Why it matters: the semantic, evidence, cost, or compatibility consequence
Recommendation: the agent's preferred answer and why
Question: one concrete choice for the user
```

Do not ask the user to choose between code shapes before explaining which
algorithmic or architectural invariant each shape protects. If the alternatives
do not change a user-owned decision, choose the sound implementation yourself.

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
- `progress/`: deprecated legacy archive only. Read old notes only when
  reconstructing provenance; do not use it as a carrier for new records.

If the next evidence is cheap and immediate, use `probe`. If the evidence needs
new forward-pass hooks, dataset/template variants, logging, evaluator support, or
other feature work before launch, use `build-probe`. For `build-probe`, record a
minimal probe-enablement plan before implementation:

```md
## Research Question
{What uncertainty this work will resolve.}

## Evidence Target
{The metric, artifact, visualization, or comparison that will decide.}

## Build Requirements
- {Minimal code/config/data/template change needed before the probe can run.}

## Probe Run
- Scope: `tiny|smoke|val200|proxy|partial|full`
- Baseline/comparison: `{required matched target}`
- Stop condition: `{what result changes the decision}`

## Carrier
{chat|handoff|research/<slug>|docs/history/<slug>|openspec/changes/<change>}
```

Use `openspec/changes/<change>` only when probe-enablement changes a stable
compatibility-sensitive contract such as schema, artifact names, metric
semantics, training/eval behavior, or public config behavior. Use a research note
or handoff for ordinary experiment scaffolding.

Otherwise use a compact decision record:

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

## Next State
{drop|narrow|probe|build-probe|implement|document|needs user decision}

## Carrier
{chat|handoff|research|docs|openspec|none}
```

After recording, continue the decision loop unless the user asked to pause,
stop, or only record. Never treat permission to record a decision as permission
to implement it.
