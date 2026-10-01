---
name: native-agent-team-guidance
description: Coordinate authorized temporary native agent teams through bounded assignments, peer coordination, reports, and durable job handoffs.
---

# Native Agent Team Guide

Use only when native delegation is authorized; this skill grants no authority to
create agents or launch research. The lead owns decomposition, user questions,
shared decisions and acceptance. A `research-worker-main` may implement or
delegate only within its package; it cannot schedule the research program.
Execution agents own bounded packages. Create a root or persistent thread only
when the user explicitly asks; use [lead-worker](../lead-worker/SKILL.md) for
that persistent-worker path.

## Assign and choose the team

Reconcile existing workers first. Give each semantic surface one owner and one
writer; parallelize independent outcomes that can be checked through their real
consumer. Reuse a worker while its assignment and context remain reliable;
reconcile changed ownership, inputs, permissions and acceptance before resuming.
Deeper nesting needs a concrete benefit and authorization.

Choose topology and `fork_turns` from dependencies, isolation and integration
cost. Follow global [model routing](../../AGENTS.md#model-routing) for role
authority and defaults; resolve supported model IDs and efforts from the live
tool schema, and set overrides only when the chosen fork supports them.

Brief the outcome and non-goals, cwd and owned paths, permissions, authoritative
source and input identities, dependencies, real consumer/schema and acceptance
evidence, output format, and stop rule. For exploration, name known entry paths
or symbols and the uncertainty to resolve. Add budgets or other constants only
when they affect execution.

## Coordinate and return

Agents may coordinate directly on facts and dependencies inside their ownership.
They cannot grant permissions or change shared contracts. Agree ownership before
overlapping writes; return scope, meaning or architecture changes to the lead.
For native agents, `send_message` queues information but does not start a turn;
use `followup_task` to assign or resume an idle/interrupted worker, not for a
status check. After interruption, reconcile the roster and any external process:
interrupting an agent does not stop its job.

Follow [global checkpoint and waiting rules](../../AGENTS.md#checkpoints-and-waiting).
Report decision-bearing failures promptly with evidence, impact and a next step;
there is no default repair-count ceiling. Continue unaffected work.

## Correct the brief

After a demonstrated semantic misunderstanding, clarify the governing invariant
and a counterexample, or take over the coupled work. Increasing effort alone
does not correct the brief.

Return `candidate`, `NEEDS_CONTEXT`, `HOLD`, `BLOCKED` or `SUPERSEDED`, with
changed paths, source identities, consumer checks, unresolved questions and job
state. The lead verifies the stable candidate and alone marks it
`lead-accepted`; user-owned decisions require separate `user-accepted` evidence.
Worker self-acceptance and transport completion are not acceptance.

## Durable job handoff

Keep one owner and one live invocation. Hand off the exact command/run identity,
process/session, stable logs and results, saved completed units, terminal success
and failure signals, completed checks, responsible job owner and direct return
route. Before root ends its turn, verify the producer survives agent return,
completed units are saved, both terminal outcomes are signaled, and the return
route works. The job owner
inspects terminal artifacts and reports directly to root; a launch or agent
return is not completion. If no return route works, disclose the gap, retain
explicit ownership, and do not promise an automatic notification. Reconcile
ownership before replacement and never rerun completed units.

For an already authorized frozen probe, bind the assignment to the
[Frozen Probe Execution Packet](../research-flow/references/probe-execution-packet.md).
When a Luna brief crosses configuration, caller or artifact-consumer boundaries,
use the worked [Luna delegation examples](references/luna-delegation.md).
