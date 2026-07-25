# Dynamic Subagent Model Routing

Last reviewed: 2026-07-25.

This document is the living, human-readable routing guide for delegated
CoordExp work. New sessions should read it before substantial subagent
delegation.

It is advisory, not an executable agent profile and not a fixed role map. The
lead agent retains responsibility for selecting the model and reasoning effort
for each task. Revise this guide when repeated task evidence changes a routing
decision; do not preserve a stale rule merely for consistency.

## Primary Decision Rule

Route by three properties:

1. **Mechanical verifiability**: can a command, test, artifact receipt, exact
   path, or deterministic comparison decide whether the task succeeded?
2. **Scientific consequence**: can the answer change the experiment's meaning,
   training objective, architecture direction, or final research claim?
3. **Observed worker fit**: what repeated evidence exists for this worker on
   the same task class, tool environment, and required ownership mode?

Choose a worker with demonstrated or plausibly strong task fit first. Among
suitable workers, use the least expensive model and lowest reasoning effort
that preserve the scientific consequence and required ownership mode.
Escalate after evidence of a real capability mismatch, not merely because the
task is important.

## Current Model Roles

`GPT` means Generative Pre-trained Transformer. `xhigh` and `max` are the
runtime names for extra-high and maximum reasoning effort.

| Model family | Current default role | Do not assume |
| --- | --- | --- |
| GPT-5.6 Luna | Repository discovery, artifact lookup, executable preflight, and bounded implementation with mechanical acceptance | That maximum reasoning makes it a scientific or architecture judge |
| GPT-5.6 Terra | Optional middle layer for cross-file integration, implementation review, experiment-contract checking, and a second independent view | That every Luna result must pass through Terra, or that Terra is the final scientific arbiter |
| GPT-5.6 Sol | Mechanism reasoning, competing-hypothesis comparison, research design, conclusion-critical integration, and final evidence judgment | That it is automatically more reliable on exact strings, file inventory, or other mechanical details |
| Claude Code through the installed `cc:*` bridge | A peer external worker for independent audit, engineering-focused investigation, bounded implementation, and writing when task evidence supports the route | That provider diversity makes its answer correct, that it is review-only, or that every Codex task needs a Claude pass |

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

### Claude Code External Worker

Treat Claude Code as another dynamically routed worker alongside Luna, Terra,
and Sol, not as a mandatory approval gate. It may be the primary owner of a
bounded audit, investigation, implementation, or writing lane. The Codex lead
retains research interpretation, scope changes, cross-lane synthesis, and final
judgment.

Use the installed `cc:*` bridge rather than invoking the Claude command-line
interface directly:

- Prefer `cc:rescue` for independently useful investigation, implementation,
  verification, follow-through, and writing. Use `--fresh` for a new lane and
  `--resume` only for a concrete follow-up to the same Claude task.
- Pass `--write` explicitly when Claude owns edits. For a read-only rescue,
  omit `--write` and state the no-modification boundary in the task. Do not run
  concurrent writers on the same semantic surface.
- Use `cc:review` for ordinary read-only Git-diff review and
  `cc:adversarial-review` for one precise design-risk or hidden-assumption
  question. These review commands do not replace `cc:rescue` when Claude should
  investigate, edit, test, or carry the task forward.
- Use background tracked jobs for non-trivial work and retrieve them through
  `cc:status` and `cc:result`. Keep the automatic turn-end review gate disabled
  unless the user explicitly enables it for a short monitored session.

The current provisional quality prior, holding task, evidence, tools, and
execution contract comparable, is:

```text
Haiku < Claude 5 Sonnet < Claude 5 Opus < Fable
```

Within one model and task class, higher reasoning effort is also expected to
improve quality. Route model and effort as separate axes rather than treating
model identity as the only capability choice. This ordering is a calibration
prior, not a permanent guarantee: task fit and observed results can override
it. Plugin aliases such as `sonnet` and `opus` may resolve to pinned Claude 4.x
identifiers; pass the exact full model identifier when Claude 5 Sonnet or
Claude 5 Opus is required. Use Fable deliberately while accumulating enough
task-class evidence to decide where its quality advantage is dependable.

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
6. Consider Claude Code as an initial owner or independent lane when its
   engineering bias, provider independence, context, or observed task-class
   record is useful. Do not reserve it only for post-hoc review.
7. Escalate by failure type. Do not blindly repeat the same prompt with a more
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
- Apply the same ownership rule to Claude Code. A `cc:rescue --write` lane is a
  real writer/worker, not a disposable reviewer; isolate its write surface from
  concurrent Codex or subagent writers.
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
- Use a fresh, bounded evidence packet for an independent Claude audit and do
  not seed it with the lead agent's preferred conclusion. One external lane per
  decision surface is the default; add another only when a real contradiction
  remains or the comparison itself is the experiment.

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
- The `cc:*` bridge is installed and Claude Code is now an eligible peer worker,
  but task-class calibration is still sparse. Record whether Claude finds
  unique actionable defects, respects write and claim boundaries, completes
  verification, and changes the final decision before strengthening its
  defaults.

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
