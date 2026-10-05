---
name: native-agent-team-guidance
description: Coordinate authorized native workers and real handoffs, including package continuation, consumer acceptance and durable external jobs.
---

# Native Agent Team Guide

User authority, model preferences and final acceptance belong to the
[user contract](../../AGENTS.md#agent-topology-and-delegation). This skill owns
worker mechanics. Persistent transport additionally uses
[lead-worker](../lead-worker/SKILL.md); no skill creates delegation or launch grants.

Choose topology before tools. A requested independent lead/worker pair uses
ordinary sessions through `lead-worker`; do not replace it with subagents.
For authorized child agents, use the current harness's declared tools:
Codex native agent tools, or Pi Web's `Agent`, `get_subagent_result`, and
`steer_subagent` when enabled. Pi Web control tools must be called directly,
not inside codemode. Pi Web's `Agent(resume=...)` continues a child owned by
the same parent; it does not message arbitrary independent sessions. Do not
assume a terminal Pi installation exposes Pi Web's extensions.

## Assign and choose the team

Reconcile existing workers and owners. Reuse reliable context; choose topology
from actual dependencies and integration cost, not fixed layers. Deeper nesting
needs a concrete benefit and authorization. Resolve supported model/effort and
fork settings from the current tool schema; full-history forks may disallow
setting overrides. Verify actual settings against global model routing.

Bind the package to outcome/non-goals, physical checkout and owned paths,
authority, source/input identities, dependencies, actual consumer/schema,
acceptance evidence, return target, authorized next actions, retry/relaunch limits
and stop boundary. Add constants only when they affect execution. An exploration
brief names entrypoints and the uncertainty, not a speculative implementation.

For the experimental shared-memory MCP, name recall/candidate duties and the
consolidator. Follow [its skill](../shared-memory/SKILL.md): a child needs its own
real caller context. Otherwise it returns source-linked candidate content for
lead capture, never submitting under an inherited parent identity.

## Coordinate and continue

Use the current harness's native assignment/message/observation contract. Message
delivery is not proof of a started turn. Peers may coordinate facts and dependencies
inside their packages, not grant authority or change shared meaning. Agree ownership
before overlapping writes and reconcile external jobs after interruption.

Workers own routine implementation, technical checks, repair/recheck and continuation
through already-authorized results within the package's resource, compatibility,
retry and stop bounds. A failed check alone needs no lead signoff or arbitrary
repair-count ceiling. Preserve frozen identities and meaningful checks; escalate
changes to research meaning, scope, material cost or stop rules. Continue unaffected
work and report decision-bearing failures with evidence, impact and a next step.

Checkpoint at a decision, first production-shaped result, acceptance or material
rework risk, not every phase. For changed process/artifact ownership, use
[full-pipeline-smoke](../full-pipeline-smoke/SKILL.md) for the earliest real
caller-to-terminal-consumer slice; CPU lifecycle evidence does not prove native
model/distributed behavior. Logs should retain raw evidence while observations
use bounded relevant intervals, exit status and terminal artifacts.

## Correct the brief

After a demonstrated semantic misunderstanding, clarify the invariant and a
counterexample or take over tightly coupled work. More effort alone is not a fix.

Return `candidate`, `NEEDS_CONTEXT`, `HOLD`, `BLOCKED` or `SUPERSEDED`, with changed
paths, source/input/runtime/artifact identities, actual exit statuses, structured
checks, raw-evidence pointers, consumer coverage, unresolved questions and job state.
The lead reuses unchanged package checks and verifies the final consumer/decision
boundary before `lead-accepted`; user-owned decisions need `user-accepted` evidence.
A worker's technical acceptance is not final scientific acceptance.

## Durable job handoff

A handoff is transport, not a new owner or inherited authority. Use it for a real
transfer, machine boundary or unreliable context, not simply a new phase. Inline
is default unless the task requests a durable destination. Link the current
objective, ruling, decisive evidence, superseded routes/reopening conditions,
next action/stop boundary and minimum reading path. Include only recovery-critical
checkout/dirty scope, config/checkpoint/artifacts, commands and pending checks.

For a live external job, carry exact run/command identity, process/session,
completed units, logs/results, terminal success and failure signals, job owner
and direct return route. Verify survival after agent return, persistence and
return delivery before promising continuation. If the route is unavailable,
retain explicit ownership and disclose it. Reconcile before replacement; never
rerun completed units or assume agent interruption stopped an external process.

## Conditional resources

- Already authorized research execution: [frozen probe packet](../research-flow/references/probe-execution-packet.md).
- Briefs crossing configuration/caller/consumer seams: [Luna examples](references/luna-delegation.md).
- Requested acceptance-cost comparisons: [routing evidence](references/evidence-workflow.md),
  backed by `scripts/routing_evidence.py` and its tests. These receipts are not
  mandatory for ordinary delegation and cannot establish a global model ranking.
