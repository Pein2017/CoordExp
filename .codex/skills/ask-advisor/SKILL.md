---
name: ask-advisor
description: Consult a native Astra advisor on consequential unresolved research or design decisions that could change the next action. Use for explicit advisor requests or decision-changing uncertainty, not routine status, source lookup, or mandatory review.
---

# Ask Advisor

Use an on-demand native advisor while the lead retains interpretation,
acceptance and the next assignment. This supports a Sol lead and persistent
Sol worker without changing either model or adding a persistent Astra task.
The [agent contract](../../AGENTS.md#model-routing) owns model choice and
delegation authority; this skill does not grant research or launch permission.
For a user-mediated web GPT-Pro consultation, use [ask-pro](../ask-pro/SKILL.md).

## Decide whether advice changes the work

Consult when unresolved reasoning, competing mechanisms or a consequential
design tradeoff could change direction, acceptance or a stop/continue decision.
Do not require exhausting local attempts first. Resolve cheap discoverable facts
and executable checks locally. Skip routine progress, settled choices and
generic review without a decision. Explicit consultation requests should still
be honored within their scope; do not manufacture a larger problem.

## Give one decision and its evidence

State the objective, decision, constraints, alternatives, verified observations,
strongest counterevidence and what the answer would change. Include exact source
anchors and material identities/denominators. Separate the lead's interpretation
from observations; let the advisor challenge the framing or propose an omitted
alternative. Evidence that was not supplied or inspected remains unknown.

Select Astra and an explicit supported effort under the contract's model rules;
follow [Agent topology and delegation](../../AGENTS.md#agent-topology-and-delegation)
for fork compatibility. Verify the actual assignment; report unsupported routing
rather than silently substituting. Reuse a suitable advisor for the same unresolved
question. For a new advisor, a self-contained `fork_turns=none` brief supports fresh
framing; use selected history when constraints require it. A separate context
does not guarantee independent judgment.

Assign read-only consultation: inspect relevant sources, reason about the
decision, and recommend a discriminator. No edits, experiments, extra agents,
launches or self-acceptance. Return directly to the lead using native completion;
do not build an API wrapper, watcher or second reporting route.

## Reconcile and stop

Ask for a concise recommendation or `NEEDS_CONTEXT`/`HOLD`, decisive evidence,
strongest alternative, assumptions or counterexample that would reverse it,
and the resulting next-action change. The advisor may find the evidence cannot
decide; it should identify the smallest missing evidence, not launch new work.

The lead checks load-bearing claims against sources and records its own ruling
at the existing decision owner. Advice is neither scientific evidence nor user
authorization. Preserve ordinary consumer-facing checks and the strongest
relevant counterexample; do not repeat the worker's entire investigation.
Reconsult only for a specific material gap or new evidence that can change the
decision. Close when the bounded question is answered; no acknowledgment-only
wake, automatic second reviewer or review of the review.
