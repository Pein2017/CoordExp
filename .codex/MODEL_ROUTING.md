# Dynamic Subagent Model Routing

Last reviewed: 2026-07-23.

This document is the living, human-readable routing guide for delegated
CoordExp work. New sessions should read it before substantial subagent
delegation.

It is advisory, not an executable agent profile and not a fixed role map. The
lead agent retains responsibility for selecting the model and reasoning effort
for each task. Revise this guide when repeated task evidence changes a routing
decision; do not preserve a stale rule merely for consistency.

## Primary Decision Rule

Route by two properties:

1. **Mechanical verifiability**: can a command, test, artifact receipt, exact
   path, or deterministic comparison decide whether the task succeeded?
2. **Scientific consequence**: can the answer change the experiment's meaning,
   training objective, architecture direction, or final research claim?

Use the least expensive model and lowest reasoning effort that can satisfy the
first property without under-serving the second. Escalate after evidence of a
real capability mismatch, not merely because the task is important.

## Current Model Roles

`GPT` means Generative Pre-trained Transformer. `xhigh` and `max` are the
runtime names for extra-high and maximum reasoning effort.

| Model family | Current default role | Do not assume |
| --- | --- | --- |
| GPT-5.6 Luna | Repository discovery, artifact lookup, executable preflight, and bounded implementation with mechanical acceptance | That maximum reasoning makes it a scientific or architecture judge |
| GPT-5.6 Terra | Optional middle layer for cross-file integration, implementation review, experiment-contract checking, and a second independent view | That every Luna result must pass through Terra, or that Terra is the final scientific arbiter |
| GPT-5.6 Sol | Mechanism reasoning, competing-hypothesis comparison, research design, conclusion-critical integration, and final evidence judgment | That it is automatically more reliable on exact strings, file inventory, or other mechanical details |

Observed tendencies are routing evidence, not permanent model traits.

## Reasoning-Effort Defaults

### GPT-5.6 Luna

- **Low reasoning**: only for trivial transformations with an immediate exact
  check. Repository context cost often makes this less useful than medium.
- **Medium reasoning**: default for scouting, artifact location, commands,
  inventory, aggregation, and small real-smoke preflight.
- **High reasoning**: thin implementation whose behavior is covered by
  deterministic tests or a compact runtime receipt.
- **Extra-high reasoning (`xhigh`)**: multi-file bounded implementation with a
  clear executable contract.
- **Maximum reasoning (`max`)**: difficult bounded implementation after a real
  local failure, provided no unresolved scientific judgment is delegated.

Luna at maximum reasoning remains an implementation worker until repeated
controlled evidence supports a broader role.

### GPT-5.6 Terra

- **Low reasoning**: rarely useful for CoordExp delegation.
- **Medium reasoning**: artifact synthesis, ordinary code review, configuration
  reconciliation, and independent contract checking.
- **High reasoning**: moderately complex implementation, dataset-pipeline
  review, or checking that code preserves an experiment's declared semantics.
- **Extra-high reasoning (`xhigh`)**: bounded scientific or implementation
  review when competing explanations are already explicit.
- **Maximum reasoning (`max`)**: use only in a controlled comparison or when a
  clearly specified task has exceeded high reasoning. Terra is not currently
  the default final arbiter.

Terra is valuable only when it replaces an unnecessary Sol call or supplies an
independent judgment. Do not create a mandatory Luna-to-Terra-to-Sol approval
chain.

### GPT-5.6 Sol

- **Low reasoning**: avoid for ordinary CoordExp delegation; use a less costly
  model instead.
- **Medium reasoning**: research-unit review, evidence synthesis, contract
  audit, and ordinary scientific diagnosis.
- **High reasoning**: novel hooks, cache or precision behavior, stateful model
  interventions, cross-module scientific implementation, and difficult result
  interpretation.
- **Extra-high reasoning (`xhigh`)**: mechanism modeling, architecture
  direction, causal disputes, and final conclusion-critical audit.
- **Maximum reasoning (`max`)**: reserve for unresolved contradictions that can
  change the research route, major training cost, or a paper-level claim.

Do not use Sol at maximum reasoning for work that a deterministic test can
settle.

## Dynamic Routing Procedure

1. If success is mechanically decidable, start with Luna at medium reasoning.
2. Raise Luna to high, extra-high, or maximum reasoning when implementation
   complexity is the demonstrated blocker.
3. Use Terra at medium or high reasoning when the task mainly integrates
   several contracts or needs an independent implementation review.
4. Use Sol at medium or high reasoning when the result can change scientific
   interpretation, the training objective, or a conclusion-critical runtime
   seam.
5. Raise Sol to extra-high or maximum reasoning only after the remaining
   uncertainty has been localized to a genuine scientific contradiction.
6. Escalate by failure type. Do not blindly repeat the same prompt with a more
   expensive model.

Typical escalation:

```text
missing path, command failure, or deterministic test defect
  -> continue with Luna at higher reasoning

cross-file contract conflict or ambiguous integration
  -> Terra high or Sol medium

experiment runs but the evidence has competing scientific meanings
  -> Sol high or extra-high

route-changing contradiction remains after focused evidence
  -> Sol maximum
```

## Ownership And Review

- Give one implementation owner to each semantic surface.
- Separate implementation ownership from independent scientific judgment when
  that distinction affects trust.
- Do not run three duplicate implementations merely because all three model
  families are available. A model race is justified only when it is itself a
  bounded routing benchmark.
- Give one local defect one focused follow-up in the existing context; then
  escalate according to the failure type.
- Adjudicate disagreements from tests, runtime receipts, artifacts, and claim
  boundaries rather than model prestige, majority vote, prose length, or token
  expenditure.

## Context Inheritance

- Use `fork_turns="none"` for self-contained discovery, artifact lookup,
  bounded implementation, and independent audit. Put exact paths and the
  acceptance contract in the brief.
- Use one or two recent turns only when the task depends on a recent decision
  that cannot be stated compactly.
- Use full-history inheritance only when reconstructing the discussion itself
  is the task.

More inherited context is not automatically better. It can transmit the lead
agent's preferred explanation and weaken reviewer independence.

## Example: Set-Level Detection Training

For an order-independent physical-owner set-supervision unit, a reasonable
initial split is:

1. Luna high or maximum implements grouped trajectory data, set inclusion,
   incomparable trajectories, unknown-neutral handling, and deterministic
   tests.
2. Terra high independently checks that the data and loss have not silently
   collapsed into single-positive-row cross entropy.
3. Sol extra-high reviews whether the objective genuinely rewards final unique
   physical-owner coverage, protects retained owners, and supports the claimed
   experiment conclusion.
4. The lead agent owns scope changes, Graphics Processing Unit promotion, and
   the final route decision.

This is an example, not a permanent role assignment.

## Evidence And Current Limits

The current guide is based on repeated CoordExp implementation and review work
plus the bounded Pi Stage 0 worker comparison:

- Luna, Terra, and Sol all completed mechanical artifact inventory and
  aggregation.
- In one audited task, Luna and Terra passed while both Pi Sol and native Sol
  shared the same exact capitalization error. Stronger scientific reasoning
  did not guarantee mechanical exactness.
- In another task, all three models reached the correct scientific verdict but
  failed an overly lexical verifier. Prompt and acceptance-contract quality can
  dominate model choice.
- Luna has repeatedly been effective for bounded implementation and explicitly
  bounded claims, but Luna maximum has not completed enough controlled
  science-judgment comparisons to justify promotion.
- Terra has useful middle-layer evidence, but not enough repeated evidence to
  make it mandatory or final.

Evidence handles:

- `research/investigations/qwen3-vl-dense-enumeration/2026-07-13-to-2026-07-16-weekly-research-report.md`
- `research/investigations/pi-lightweight-worker-ablation/experiments/2026-07-22-stage0-frozen-task-harness-screen/results.md`

## Continuous Calibration

Treat this document as an evolving estimate. After representative delegated
tasks, record only evidence that can change a durable routing decision:

- first-attempt test or smoke success;
- missed scientific or implementation constraints;
- defects found by independent review;
- wall time, token usage, and provider cost when comparable receipts exist;
- number and type of follow-up corrections;
- whether the model respected the claim boundary;
- whether higher reasoning reduced errors or merely increased context and
  prose.

Change a default only after repeated tasks show a stable advantage. One success
or failure is a data point, not a routing rule. Revisit this guide when a model
version changes, a new task class becomes common, or the current escalation
pattern repeatedly wastes time.
