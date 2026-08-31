---
name: native-depth2-contracting-pilot
description: Use when a user-approved long or multi-package task should run as a native max-depth-2 pilot and L0 must preserve top-level context while delegating bounded packages.
---

# Native Depth-2 Contracting Pilot

## Scope

Use this opt-in pilot only for a long task with at least one non-trivial package.
An L1 package lead compresses L2 execution into one verified acceptance packet.

The [user-wide agent contract](../../AGENTS.md) owns general topology, spawning,
waiting, review, and model routing. Research must also satisfy
[research-flow](../research-flow/SKILL.md). Before spawning, read the
[contract and receipt shapes](references/contracts.md).

## Activation

- L0 freezes the package goal, identity, verifier, permissions, and stop rule.
- L1 may add L2 only after naming at least two independent outputs and why their
  parallel completion should beat the contracting and integration cost.
- If fewer than two L2 outputs actually materialize, record `effective_depth: 1`
  and let L1 close the package directly. Do not report a depth-2 success.
- Keep unresolved user-owned semantics at L0; topology must not decide them by
  accumulating worker opinions.

## Package boundary

### L0

- Owns user dialogue, package boundaries, cross-package decisions, and acceptance.
- Receives one L1 packet. Raw L2 transcripts stay behind evidence handles unless
  a finding changes acceptance.

### L1

- Owns exactly one package: qualifying L2 tickets, integration, one bundled
  correction, verifier replay, and the acceptance packet.
- Returns `candidate`, `NEEDS_CONTEXT`, `HOLD`, `BLOCKED`, or `SUPERSEDED`.
- Does not hand L2 findings to L0 as an implementation correction queue.

### L2

- Owns one executable output, reports only to L1, and never spawns. It does not
  redefine goals, acceptance, architecture, or claims.
- Use Luna or Terra. If the ticket appears to require Sol-level semantic or
  lifecycle judgment, promote it to a separate L1 adviser instead of using Sol
  at L2.

## L1 routing

- Use Terra/high as the general L1 default; Terra/medium is sufficient for a
  bounded package with a strong deterministic verifier. Other routes remain
  task-shaped under the user-wide contract.
- Sol requires a receipt linking the prior Terra attempt, unchanged frozen
  target and verifier, acceptance-changing counterexample, and failed replay
  after Terra's one bundled correction.
- When that witness exists, start a fresh Sol/high L1 on the same contract.
  Task length or importance alone is not an escalation witness.

## Receipt and evaluation

Every final pilot receipt records effective depth, the independent-output
predicate, routes, verifier outcome, L0 reads/edits, L1 corrections, critical
path, usage coverage, and role/write-surface drift.

Retain only when receipts show lower L0 context or intervention without weaker
acceptance. Without matched depth 1, make no savings or model-superiority claim.
