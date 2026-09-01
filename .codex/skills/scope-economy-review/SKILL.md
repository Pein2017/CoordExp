---
name: scope-economy-review
description: Use when a proposed or in-progress task would place a new generic adapter, schema, runtime, orchestration layer, mock-only acceptance, auxiliary proxy/null/control gate, or repeated review ahead of the shortest existing real path, or when the user flags overdesign, overengineering, or overauditing.
---

# Scope Economy Review

## Overview

Run one adversarial **decision-path** review, not a general code review. New
machinery stays only when it closes a named acceptance-changing risk that the
existing real path cannot close more cheaply.

## Invocation contract

When a description trigger is present and this frozen phase and target have not
already received this review, launch exactly one read-only reviewer:

- `model: gpt-5.6-sol`
- `reasoning_effort: high`
- `fork_turns: "none"`
- no writes, subagents, or full-history reconstruction

Give it only the frozen outcome, acceptance and stop rule; proposed next actions;
shortest existing real path; claimed consumer of the new work; and explicit
user-owned architecture, claim, cost, or safety decisions. Without a trigger,
do not launch it.

## Decision rule

Try to falsify the necessity and ordering of the proposed work:

| Verdict | Required evidence |
| --- | --- |
| `KEEP` | A counterexample changes acceptance, evidence identity, safety, or requested behavior; this is the cheapest closure and is ordered correctly. |
| `CUT` | No such risk is demonstrated, an existing/native path closes it, or the work only makes a speculative interface self-consistent. |
| `REORDER` | The shortest real vertical path or frozen primary contrast can test necessity first. |
| `USER_DECISION` | Proceeding would silently change user-owned semantics, architecture, material cost, claim scope, or stop rule. State the smallest decision needed. |

An auxiliary proxy, null, mechanism diagnostic, or mock receipt is not primary
evidence merely because it is rigorous. A nonblocking check stays off the
critical path. Explicit safety, identity, data-integrity, public compatibility,
and user-mandated architecture remain binding.

## Output contract

Return only these fields, one concise statement each:

```text
VERDICT: KEEP | CUT | REORDER | USER_DECISION
DECISION-OWNING OUTCOME:
SHORTEST REAL PATH:
CONCLUSION-CHANGING RISK:
CHEAPER DISCRIMINATOR:
ACTION:
```

A blocking verdict requires a concrete counterexample. Do not propose a full
alternative architecture, optional hardening, or another reviewer. Stop after
one pass; the lead owns reconciliation and acceptance.

## Pressure checks

| Rationalization | Decision test |
| --- | --- |
| “It must scale later.” | First show failure at the next decision-bearing scale. |
| “The mathematics is cleaner.” | Cleanliness does not make a diagnostic decision-owning. |
| “Cost is allowed.” | Permission to spend does not prove necessity or ordering. |
| “Mocks are green.” | Self-consistency does not close a real-path risk. |
| “Subagents are cheap.” | Count briefing, integration, and acceptance cost. |

## Example

A real `N=2` runner exists, while a generic adapter and fake-runtime suite are
proposed before real `N=4`. Return `REORDER`: extend the existing runner to
`N=4`; abstract only if that run exposes unavoidable duplication, receipt
ambiguity, or backend divergence.
