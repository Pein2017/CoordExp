---
name: agent-routing
description: Compose and adapt bounded Codex and Claude Code teams by deciding whether to delegate, role, acceptance, verifier, surface, model-effort route, escalation, joining, and stopping. Use for substantial delegation, team or model selection, recovery, or receipt-backed routing calibration; skip simple work that should remain with the lead.
---

# Agent Routing

> Experimental. The adaptive state is a constrained tie-breaker, never a model
> leaderboard or an authority source.

## Keep Learning Inside Hard Policy

1. Keep the lead responsible for decomposition, global context, research
   meaning, cross-lane decisions, and final acceptance. Keep simple work local.
2. Choose in this order: **role -> acceptance -> verifier -> surface -> model
   and effort**. Select the verifier before selecting a cheap worker; it must
   cover the consequential failure, not merely syntax.
3. Fix the eligible routes, correction budget, escalation boundary, join state,
   and user-owned decisions before consulting learned state. A learned score may
   choose only among that eligible set; it cannot lower a role, weaken a
   verifier, authorize cost, change a claim, or alter a stop rule.
4. Read [references/model-priors.md](references/model-priors.md) for current
   cold-start eligibility. Verify the live picker and resolved model identity
   before every non-default route. Unknown alias or build drift means abstain or
   reset that fixed slot; never inherit its old evidence.

Use these fixed responsibility cells:

- `read_only_scout`: locate source, extract receipts, inventory behavior, or
  reproduce a failure without write, diagnosis, integration, or conclusion
  ownership;
- `bounded_builder`: frozen intent, enumerable present/absent/failure/
  compatibility paths, and deterministic acceptance;
- `semantic_builder`: mathematical, algorithmic, research-semantic, missing-
  invariant, or silent-correctness implementation;
- `lifecycle_builder`: framework lifecycle, compatibility, source archaeology,
  serialization, concurrency, durability, or broad cross-module engineering;
- `semantic_reviewer` and `lifecycle_reviewer`: fixed-target independent review
  or diagnosis, never ordinary building;
- `major_decision`: decision-grade advice only; the lead and user retain
  authority.

Judge semantic dataflow rather than file count. An unresolved invariant, silent
false acceptance, compatibility or concurrency risk, or one value crossing
multiple lifecycle stages is complex even when the diff is small.

## Compose And Hand Off

Use the smallest useful team:

- scout -> lead for bounded evidence;
- bounded builder -> deterministic checker;
- one complex builder -> independent fixed-target reviewer -> lead when risk
  survives executable checks;
- one major-decision adviser -> counterargument -> lead/user only when
  independence can change the decision.

Do not summon every model or duplicate writers on one semantic surface. Admit
parallel writers only for disjoint paths behind a frozen shared interface. If
lanes share state, lifecycle, or contract ownership, serialize them or name one
integrator. The lead runs the cross-lane verifier after joining.

For every Codex or Claude spawn:

- set `fork_turns: "none"`;
- give a self-contained brief with goal, exact cwd and owned paths, evidence,
  permissions/write intent, non-negotiable constraints, output contract,
  verifier, completion condition, and stop boundary;
- state whether the lane needs a fresh process or exact continuation;
- keep interpretation and scope changes with the lead.

## Correct, Join, And Stop

Evaluate outside worker prose. Allow at most one focused correction for a
diagnosed local gap in an already suitable bounded lane. Keep command, path, or
deterministic-test failures there; reroute newly discovered semantic complexity
to its owning builder or reviewer. Do not turn a failed builder into an `xhigh`
builder, and do not mistake higher effort for correctness.

Front-load the declared success, absence, failure, continuation, publication,
and compatibility paths before an expensive review. Freeze consequential
targets by tree/commit, runtime identity, and evidence set. Any edit invalidates
that certification attempt. A reviewer returns reproducible `PASS` or `HOLD`
and never becomes the repair owner.

Join required lanes before synthesis. Detachment requires explicit user intent.
Advance useful lead work while agents run, wait at meaningful boundaries, and
use one narrow follow-up only when new evidence changes the lane. Never blind
poll, blind retry, or run overlapping reassurance reviews.

## Route Distributed And One-Shot Execution

Treat rank-varying forward counts, wrapped versus unwrapped model calls,
autograd-hook order, activation recomputation, optimizer mutation, and
collective choreography as one semantic execution surface. Route its
implementation to `semantic_builder` (normally Sol/high), even when each local
edit looks mechanical. Require an independent frozen-target lifecycle review
(normally Opus/xhigh) when framework or distributed ordering can hang, double
reduce, silently skip parameters, or mutate before a failing consensus. A unit
test for the leaf helper is insufficient: acceptance must exercise the
production wrapper and at least one real multi-process path, then preserve the
exact GPU/NCCL or production-shaped evidence gap if it remains.

Provider choice, each quota preference, temporary availability, and a user's
requested provider preference constrain the live **eligible/available set** for
this episode. They are not capability priors and must not update controller
quality summaries. If a requested route is unavailable, remove it from the
available set and select another already-eligible route; do not reinterpret the
outage as model-quality evidence or inherit evidence across providers.

Builders and reviewers never own irreversible execution. For a costly launch,
publication, external mutation, or at-most-once job, reserve one `lead-only executor`
after the implementation is committed, required reviewers have
settled on that fixed target, and preflight receipts are current. The executor
must publish intent before activation, bind the observed process/run identity,
and never blind-retry an uncertain or partially activated attempt. A consumed
claim stays immutable. Recovery requires explicit user authority and an
append-only parent-linked successor that binds the failed claim and receipts,
uses a fresh nonce/attempt identity, and carries a fixed retry ceiling. Review
or implementation permission alone never grants that recovery authority.

Do not train the adaptive controller from unmatched smoke, provider-outage,
moving-target review, or one-shot launch evidence. Record such outcomes as
reporting-only evidence unless a predeclared matched routing contract and its
deterministic verifier both survived unchanged.

## Use The Fixed-Capacity Controller

The deterministic controller is
`scripts/routing_state.py`. Its runtime state defaults to
`$CODEX_HOME/state/agent-routing.json`, outside tracked skill prose. The
topology is fixed at 7 responsibility cells x 3 route slots x 24 learned
numeric summaries: **504 learned scalars**. Route identities may be deliberately
replaced, but slots and statistic keys cannot grow. Each slot keeps only fixed
receipt/evidence rings, plus a fixed 32-entry pending-episode table; full
evidence remains in existing sessions and ledgers. Each role also keeps exactly
one active comparison identity, contract, cohort, signature, and monotone run
epoch; these are bounded identity metadata, not learned weights. A fixed
262,144-bit replay filter prevents a completed root/episode from training twice
even after its receipt leaves the bounded rings. A second fixed filter records
only accepted episodes and binds them to the original role and internal route
generation for late false-accept quarantine. A third records which accepted
episode has already had its false-accept correction processed. Eventual false
positives fail closed rather than duplicating or misattributing evidence. At the reviewed
10,000-terminal ceiling, new reservations abstain before saturation; rotate or
migrate only through an explicit ledger-backed review, never silently.

Initialize once and validate whenever state is material:

```bash
conda run -n ms python "$CODEX_HOME/skills/agent-routing/scripts/routing_state.py" init
conda run -n ms python "$CODEX_HOME/skills/agent-routing/scripts/routing_state.py" validate
```

After the lead fixes policy eligibility and verifies live availability, ask the
controller for a recommendation by passing both sets explicitly:

```bash
conda run -n ms python "$CODEX_HOME/skills/agent-routing/scripts/routing_state.py" select \
  --role bounded_builder \
  --eligible codex:gpt-5.6-terra:high \
  --available codex:gpt-5.6-terra:high
```

Cold, stale, sparse, quarantined, or overlapping evidence falls back to the
reviewed static route. `select` abstains when no lead-approved live route
remains. Copy its `route_generation`, `comparable_eligible_routes`, and
controller-derived `route_cohort_hash` into any comparable plan. Treat its
output as a recommendation, not a dispatch authorization. `major_decision`
returns the static policy route when it is eligible and live, otherwise it
abstains; learned summaries may inform advice but never auto-promote a decision
adviser. Fable/xhigh and Fable/max remain explicit independent-adviser choices;
they are not automatic substitutes when Sol/max is unavailable.

Before spawn, fill
[references/routing-plan-template.json](references/routing-plan-template.json)
with root/episode identity, eligible routes, exact route generation, resolved
model identity, task-shape, brief, verifier and target hashes, correction
budget, terminal budget/due time, and evidence grade. A comparable plan must
also bind the route-cohort hash returned by `select` and leave
`comparison_run_epoch` at zero. Under the exclusive reservation lock, the
controller reuses the role's active run only for the same comparison id,
contract, and route cohort; every transition—including `A -> B -> A`—increments
the fixed role-level run epoch and returns it for the receipt to echo. Replace
every template sentinel; all-zero
hashes and placeholder model identities are rejected. Reserve it atomically; if the
fixed pending table is full or an older episode is overdue, adaptation abstains.
The first reservation binds the verified resolved model identity to that fixed
slot. A different resolved identity under the same route epoch is rejected and
must use an explicit slot reset. If a post-replacement directory `fsync` fails,
the command reports an uncertain visible commit with its digest; validate that
digest before any retry.

```bash
conda run -n ms python "$CODEX_HOME/skills/agent-routing/scripts/routing_state.py" reserve \
  --plan /path/to/lead-routing-plan.json
```

After every terminal lane—including rejection, explicit abandonment, invalid
target, or surface failure—write one structured lead receipt from
[references/routing-receipt-template.json](references/routing-receipt-template.json)
and resolve the reserved episode. Follow-ups and one correction remain the same
episode; worker completion cannot silently remove a pending record.

```bash
conda run -n ms python "$CODEX_HOME/skills/agent-routing/scripts/routing_state.py" update \
  --receipt /path/to/lead-routing-receipt.json
```

Receipt rules:

- worker completion or green focused tests are never acceptance;
- a reviewer whose reproducible `HOLD` is accepted by the lead can be
  `accepted_first_pass` even though the target is rejected;
- non-review delivery acceptance requires `target_outcome=accepted` and no
  review verdict; rejection or a later false acceptance requires
  `target_outcome=rejected` and no review verdict;
- accepted reviewer delivery binds the reviewed target to exactly one of
  `accepted/pass`, `rejected/hold`, or `not_reached/invalidated`; rejected
  reviewer delivery did not reach a trustworthy target verdict;
- indeterminate or non-OK surface outcomes carry only `not_reached` or
  `not_applicable` target status and no review verdict;
- ordinary receipts update route monitoring and can quarantine a route, but
  never promote a challenger;
- only `comparable` receipts with a frozen comparison id, route-cohort hash, and
  controller-assigned shared run epoch enter adaptive selection; hold task
  shape, brief, target class, tools, permissions, output contract, verifier, and
  policy epoch fixed;
- each route slot retains one active comparison projection. A new active run
  resets only that slot's bounded comparable summaries. Grouping requires the
  same id, contract, route cohort, and shared run epoch, so neither a reset peer
  nor an `A -> B -> A` comparison cycle can pair with stale evidence. Different
  contracts never pool; active same-id contract drift is rejected;
- auth/client, runtime, target-invalid, and receipt-invalid outcomes update
  reporting-only surface summaries, leave capability summaries unchanged, and
  cannot promote a challenger;
- future-dated receipts fail closed. Legitimate late arrivals still close their
  reserved episode and enter the current sufficient-statistic clock with the
  appropriate exponential weight; an older comparison run cannot replace a
  newer projection;
- a later verified false acceptance is a planless negative receipt that must
  supersede a previously accepted episode. When the accepted receipt remains in
  the fixed ring, the controller rejects a correction that predates it. After
  it ages out, the lead must verify chronology from the named external evidence;
  the accepted-route filter proves only the original route identity. The
  correction immediately quarantines that route cell and requires explicit
  identity reset or review. Each accepted episode changes numeric summaries at
  most once; repeated corrections still quarantine but add neither numeric nor
  bounded-ring evidence. An aged-out correction also makes no unverifiable
  numeric quality update;
- a model/client/tool identity change resets one existing slot with the
  controller's confirmed `replace` command; never append a new model row.

The controller applies a 30-day half-life to quality summaries and a 7-day
half-life to surface health. It promotes no route from static prose
pseudo-counts. Adaptive choice requires comparable support from multiple recent
roots for at least two peers sharing one comparison id, route cohort, run epoch,
and matched contract, then uses a tiered lexicographic objective: conservative
acceptance quality first, a fixed 0.05 quality-equivalence margin before time,
expected time-to-final-acceptance plus lead burden next, and spend last. A fast
unsafe route can never compensate for a policy or acceptance failure.

Do not run autonomous exploration on consequential work. If uncertainty changes
a real routine decision, run a bounded matched or safely randomized trial only
among already eligible reversible routes with an adequate deterministic
verifier. Freeze one routing variable, stop on false acceptance, and keep
ordinary production receipts observational.

## Report

State the task cell, eligible and rejected routes, verifier, selected route and
reason codes, correction/escalation/join boundaries, final lead disposition,
state update or abstention, and what remains unproven.
