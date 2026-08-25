---
name: native-depth2-contracting-pilot
description: Use when a user-approved long or multi-package task should run as a native max-depth-2 pilot and L0 must preserve top-level context while delegating bounded packages.
---

# Native Depth-2 Contracting Pilot

## Purpose

Use native subagents as a contracting hierarchy, not merely as extra parallelism:

```text
L0 user-facing lead
  -> multiple L1 package leads
       -> multiple L2 execution workers
```

The hierarchy succeeds only when L1 absorbs implementation detail, correction,
and verification. L0 retains user intent, cross-package design, and final
acceptance instead of rereading L2 transcripts.

## Activation boundary

Use this pilot only when the user opts in or an existing task contract names it.
Keep effective depth 1 when the task is small, single-lane, semantically
unsettled, or cheaper for L0 to verify directly. Do not delegate implementation
that would silently freeze an unresolved user-owned decision.

Depth 2 is earned, not the default. An L1 may add L2 only when it can name at
least two independent outputs whose parallel work is expected to save more
time/context than their contracts and integration cost. Otherwise the L1 works
directly. “Full speed” and spare slots do not satisfy this predicate.

Before spawning, read [references/contracts.md](references/contracts.md).

## Layer contract

### L0: user-facing lead

- Own user dialogue, goals, architecture/research meaning, package boundaries,
  material permissions, cross-package dependencies, and final acceptance.
- Freeze one self-contained contract per L1. Send evidence handles and deltas,
  not the main-thread transcript.
- Spawn multiple L1s only for independent read surfaces or disjoint write
  surfaces. Reserve enough live capacity for their L2 workers.
- Receive package acceptance packets and decision escalations. Do not routinely
  receive raw L2 outputs, run their correction loop, or become an implementor.

### L1: package lead or contractor

- Own exactly one non-trivial package: decomposition, L2 routing, integration,
  one bundled correction round, verification, and a compact package receipt.
- May use any model in the live spawn allowlist. Prefer `gpt-5.6-terra` at the
  lowest sufficient effort for a general package lead: medium for bounded
  work, high for multi-file integration. Use Sol only after observed evidence
  that Terra cannot close a cross-package semantic, lifecycle, or integration
  problem; do not upgrade merely because a task is long or important.
- Normally use `fork_turns: "none"`. A read-only design adviser may fork at most
  the last five turns only when the conversational nuance cannot be expressed
  in a smaller contract.
- May spawn multiple L2s, but only on independent read work or disjoint write
  surfaces. L1 remains the sole integration and acceptance boundary.
- Escalate only contract-changing facts: user-owned semantics, architecture or
  scope changes, material cost, permissions, incompatible evidence, or a
  required write-surface expansion.
- Treat already-authorized generated output, content-addressed pin updates,
  fixture propagation, formatting differences, and expected verifier drift as
  local closure, not escalation. A failed exact patch is not a blocker: inspect
  the current shape, apply one bundled correction, and replay the verifier.
- Return `candidate`, `NEEDS_CONTEXT`, `HOLD`, `BLOCKED`, or `SUPERSEDED`. L1
  never marks user-owned meaning or final L0 acceptance.

### L2: execution worker

- **Hard model allowlist:** use only `gpt-5.6-luna` or `gpt-5.6-terra`, with an
  explicit supported reasoning effort and `fork_turns: "none"`.
- Never substitute Sol or another model. If the work requires Sol-level
  semantic or lifecycle judgment, create a separate L1 reviewer/adviser.
- Own one narrow output with exact paths, invariants, verifier, permissions,
  and stop rule. Implement, test, inspect, or mechanically review; do not
  redefine goals, architecture, claims, or acceptance.
- Must not spawn another agent. Report only to its parent L1 unless the harness
  forces a delivery path; L1 still validates and disposes the result.

Every spawn explicitly sets `fork_turns`, `model`, and `reasoning_effort` after
checking the live allowlist. Every brief states cwd, owned paths, frozen goal,
non-goals, authoritative constants, permissions, acceptance commands, output
contract, known failure modes, tier, budget, and stop rule.

## Operating pattern

1. L0 identifies packages and keeps unresolved semantic work at L0 or in a
   read-only L1 advisory lane.
2. L0 defaults each package to one Terra L1 and adds L2 only after the
   independent-output predicate above passes.
3. L0 allocates concurrency without filling every slot. In one shared physical
   worktree or Git index, writer L1s and commits run serially even when file
   lists differ; concurrent writers require genuinely isolated worktrees.
4. L1 partitions only qualifying lanes into L2 tickets. Never run concurrent
   writers on one semantic surface.
5. L1 independently checks L2 receipts, integrates the package, and bundles at
   most one correction round. L2 self-report is evidence, not acceptance.
6. L1 returns one acceptance packet. L0 inspects decision-bearing receipts and
   accepts, rejects, or escalates without replaying the whole implementation.
7. If the same local closure condition is returned twice, do not ask the same
   L1 for a third patch. Reframe the contract or dispatch one fresh L1 at the
   next justified tier; keep the takeover narrow and preserve accepted work.
8. Wait on L1 from L0 and on L2 from L1. Use long event waits; do not status-poll.

## Pilot evaluation

At each package boundary, report the topology, routes, accepted outcome,
correction rounds, L0 implementation interventions, semantic escalations,
wall time, and available token/cost evidence. Distinguish measurement from
inference.

Evidence favors retaining the pilot when L0 context/intervention decreases
without weaker acceptance. Recommend revising or rejecting it when L1 becomes
a transcript relay, L0 repeatedly redoes package work, role or write ownership
drifts, L2 model constraints are violated, or added latency/cost has no
acceptance or context benefit. The user decides whether to retain, iterate, or
reject the pilot.

Do not infer savings from model labels or agent count. Report measured wall
time and available usage evidence; label token/cost claims as unknown when the
runtime does not expose them. Prefer the simpler depth when acceptance quality
is equal.

## Red flags

- L2 uses Sol because the review is difficult.
- L2 findings bypass L1 and become L0's correction queue.
- L1 forwards raw child transcripts instead of a package delta.
- L0 edits the package while L1 is its implementation owner.
- One persistent L1 accumulates unrelated packages.
- All slots are occupied by L1s, leaving no capacity for L2 fan-out.
- L2 is added because slots exist rather than because independent outputs exist.
- Shared-index writer L1s commit concurrently.
- An authorized generated/pin closure is returned to L0 as a blocker.
- Sol is selected for an L1 before Terra shows an actual capability gap.
- Green tests or a worker final answer are treated as package acceptance.
