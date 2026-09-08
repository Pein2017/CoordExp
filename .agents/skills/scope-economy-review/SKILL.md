---
name: scope-economy-review
description: Use when proposed or active work puts a generic adapter, schema, runtime, orchestration layer, mock-only acceptance, auxiliary proxy/null/control gate, or repeated review ahead of the shortest real path, or when the user flags overdesign, overengineering, or overauditing.
---

# Scope Economy Review

## Overview

Apply one adversarial **decision-path** check, not a general code review. New
machinery stays only when it closes a named acceptance-changing risk more
cheaply than the existing real path.

## Invocation contract

Automatic invocation loads a lead-local check; it does not authorize delegation.
A user correction about overdesign triggers local reconsideration, never a
spawn. Do not launch a reviewer to decide whether to launch one.

One read-only reviewer is allowed only when all four conditions hold:

1. A named unresolved risk can change acceptance, evidence identity, safety, or behavior.
2. Local inspection or a deterministic check cannot close it more cheaply.
3. Expected decision value or lead-context/wall-time savings exceed briefing,
   integration, and acceptance cost.
4. This frozen phase and target have not already received the review.

Choose an available model and supported effort sufficient for the review,
following the user-wide routing rules. Keep:

- `fork_turns: "none"`
- no writes, subagents, or full-history reconstruction

Give it only the frozen outcome, acceptance and stop rule; next actions; shortest
real path; claimed consumer; and user-owned decisions. If any condition fails,
keep the check local.

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

For a delegated review, return only these fields, one concise statement each:

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
one pass; the lead owns reconciliation and acceptance. Keep a local verdict
internal unless it changes the next action or requires `USER_DECISION`.

Future scale, cleaner mathematics, allowed cost, green mocks, or cheap subagents
do not establish necessity. Apply the cheaper-discriminator and net-cost tests.

## Example

With a real `N=2` runner, return `REORDER` for an adapter proposed before real
`N=4`; abstract only if `N=4` exposes a concrete need.
